package classifier

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/bekonat/go-dsa/lru"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/singleflight"

	"github.com/bekonat/spamguard/gateway/internal/model"
)

const MaxBatch = 100

type ValidationError struct {
	Field   string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Field, e.Message)
}

type Predictor interface {
	Predict(ctx context.Context, text string) (model.Prediction, string, error)
}

type Verdict struct {
	Label           string  `json:"label"`
	SpamProbability float64 `json:"spam_probability"`
	ModelVersion    string  `json:"model_version"`
	Cached          bool    `json:"cached"`
}

type entry struct {
	verdict Verdict
	expires time.Time
}

type Options struct {
	CacheSize   int
	CacheTTL    time.Duration
	MaxTextLen  int
	CallTimeout time.Duration
	OnCache     func(hit bool)
	OnVerdict   func(label string)
}

type Service struct {
	p     Predictor
	opts  Options
	cache *lru.Cache[string, entry]
	group singleflight.Group
	now   func() time.Time
}

func New(p Predictor, opts Options) *Service {
	if opts.OnCache == nil {
		opts.OnCache = func(bool) {}
	}
	if opts.OnVerdict == nil {
		opts.OnVerdict = func(string) {}
	}
	return &Service{
		p:     p,
		opts:  opts,
		cache: lru.New[string, entry](opts.CacheSize),
		now:   time.Now,
	}
}

func (s *Service) Classify(ctx context.Context, text string) (Verdict, error) {
	text, err := s.normalize("text", text)
	if err != nil {
		return Verdict{}, err
	}
	return s.classify(ctx, text)
}

func (s *Service) ClassifyBatch(ctx context.Context, texts []string) ([]Verdict, error) {
	if len(texts) == 0 || len(texts) > MaxBatch {
		return nil, &ValidationError{Field: "texts", Message: fmt.Sprintf("must contain 1 to %d items", MaxBatch)}
	}
	norm := make([]string, len(texts))
	for i, t := range texts {
		t, err := s.normalize(fmt.Sprintf("texts[%d]", i), t)
		if err != nil {
			return nil, err
		}
		norm[i] = t
	}

	out := make([]Verdict, len(norm))
	g, ctx := errgroup.WithContext(ctx)
	for i, t := range norm {
		g.Go(func() error {
			v, err := s.classify(ctx, t)
			out[i] = v
			return err
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Service) classify(ctx context.Context, text string) (Verdict, error) {
	key := cacheKey(text)
	if e, ok := s.cache.Get(key); ok && s.now().Before(e.expires) {
		s.opts.OnCache(true)
		v := e.verdict
		v.Cached = true
		return v, nil
	}
	s.opts.OnCache(false)

	// Identical texts in flight share one model call. The call is detached
	// from the first caller's context so that its cancellation does not fail
	// the other waiters.
	ch := s.group.DoChan(key, func() (any, error) {
		callCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), s.opts.CallTimeout)
		defer cancel()

		pred, version, err := s.p.Predict(callCtx, text)
		if err != nil {
			return nil, err
		}
		v := Verdict{Label: pred.Label, SpamProbability: pred.SpamProbability, ModelVersion: version}
		s.cache.Put(key, entry{verdict: v, expires: s.now().Add(s.opts.CacheTTL)})
		s.opts.OnVerdict(v.Label)
		return v, nil
	})

	select {
	case res := <-ch:
		if res.Err != nil {
			return Verdict{}, res.Err
		}
		return res.Val.(Verdict), nil
	case <-ctx.Done():
		return Verdict{}, ctx.Err()
	}
}

func (s *Service) normalize(field, text string) (string, error) {
	text = strings.TrimSpace(text)
	switch {
	case text == "":
		return "", &ValidationError{Field: field, Message: "must not be empty"}
	case !utf8.ValidString(text):
		return "", &ValidationError{Field: field, Message: "must be valid UTF-8"}
	case utf8.RuneCountInString(text) > s.opts.MaxTextLen:
		return "", &ValidationError{Field: field, Message: fmt.Sprintf("must be at most %d characters", s.opts.MaxTextLen)}
	}
	return text, nil
}

func cacheKey(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}
