package classifier

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bekonat/spamguard/gateway/internal/model"
)

type fakePredictor struct {
	calls atomic.Int32
	delay time.Duration
	err   error
}

func (f *fakePredictor) Predict(ctx context.Context, text string) (model.Prediction, string, error) {
	f.calls.Add(1)
	select {
	case <-time.After(f.delay):
	case <-ctx.Done():
		return model.Prediction{}, "", ctx.Err()
	}
	if f.err != nil {
		return model.Prediction{}, "", f.err
	}
	label := "ham"
	if strings.Contains(text, "приз") {
		label = "spam"
	}
	return model.Prediction{Label: label, SpamProbability: 0.9}, "v1", nil
}

func newService(p Predictor) *Service {
	return New(p, Options{CacheSize: 16, CacheTTL: time.Minute, MaxTextLen: 20, CallTimeout: time.Second})
}

func TestValidation(t *testing.T) {
	s := newService(&fakePredictor{})
	tests := []struct {
		name  string
		text  string
		field string
	}{
		{"empty", "", "text"},
		{"spaces only", "   \n\t", "text"},
		{"too long", strings.Repeat("я", 21), "text"},
		{"invalid utf8", "\xff\xfe", "text"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.Classify(context.Background(), tt.text)
			var ve *ValidationError
			if !errors.As(err, &ve) || ve.Field != tt.field {
				t.Fatalf("expected validation error on %q, got %v", tt.field, err)
			}
		})
	}
}

func TestLengthIsCountedInRunes(t *testing.T) {
	s := newService(&fakePredictor{})
	if _, err := s.Classify(context.Background(), strings.Repeat("я", 20)); err != nil {
		t.Fatalf("20 cyrillic letters must fit MaxTextLen 20: %v", err)
	}
}

func TestCacheHit(t *testing.T) {
	f := &fakePredictor{}
	s := newService(f)
	ctx := context.Background()

	first, err := s.Classify(ctx, "ваш приз")
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.Classify(ctx, "  ваш приз  ")
	if err != nil {
		t.Fatal(err)
	}

	if first.Cached || !second.Cached {
		t.Fatalf("expected miss then hit, got %v then %v", first.Cached, second.Cached)
	}
	if second.Label != "spam" || second.ModelVersion != "v1" {
		t.Fatalf("unexpected cached verdict %+v", second)
	}
	if f.calls.Load() != 1 {
		t.Fatalf("expected 1 model call, got %d", f.calls.Load())
	}
}

func TestCacheExpires(t *testing.T) {
	f := &fakePredictor{}
	s := newService(f)
	now := time.Now()
	s.now = func() time.Time { return now }

	_, _ = s.Classify(context.Background(), "привет")
	now = now.Add(2 * time.Minute)
	v, _ := s.Classify(context.Background(), "привет")

	if v.Cached || f.calls.Load() != 2 {
		t.Fatalf("expired entry must be refreshed, cached=%v calls=%d", v.Cached, f.calls.Load())
	}
}

func TestErrorsAreNotCached(t *testing.T) {
	f := &fakePredictor{err: model.ErrUnavailable}
	s := newService(f)

	for range 2 {
		if _, err := s.Classify(context.Background(), "привет"); !errors.Is(err, model.ErrUnavailable) {
			t.Fatalf("expected ErrUnavailable, got %v", err)
		}
	}
	if f.calls.Load() != 2 {
		t.Fatalf("expected 2 model calls, got %d", f.calls.Load())
	}
}

func TestIdenticalConcurrentRequestsShareOneCall(t *testing.T) {
	f := &fakePredictor{delay: 30 * time.Millisecond}
	s := newService(f)

	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := s.Classify(context.Background(), "одинаковый"); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()

	if f.calls.Load() != 1 {
		t.Fatalf("expected 1 model call, got %d", f.calls.Load())
	}
}

func TestFirstCallerCancellationDoesNotFailOthers(t *testing.T) {
	f := &fakePredictor{delay: 50 * time.Millisecond}
	s := newService(f)

	ctx, cancel := context.WithCancel(context.Background())
	firstErr := make(chan error, 1)
	go func() {
		_, err := s.Classify(ctx, "общий")
		firstErr <- err
	}()
	time.Sleep(10 * time.Millisecond)

	secondErr := make(chan error, 1)
	go func() {
		_, err := s.Classify(context.Background(), "общий")
		secondErr <- err
	}()
	time.Sleep(10 * time.Millisecond)
	cancel()

	if err := <-firstErr; !errors.Is(err, context.Canceled) {
		t.Fatalf("first caller: expected context.Canceled, got %v", err)
	}
	if err := <-secondErr; err != nil {
		t.Fatalf("second caller must succeed, got %v", err)
	}
}

func TestClassifyBatch(t *testing.T) {
	s := newService(&fakePredictor{})

	texts := []string{"привет", "ваш приз", " привет "}
	vs, err := s.ClassifyBatch(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vs) != 3 || vs[0].Label != "ham" || vs[1].Label != "spam" || vs[2].Label != "ham" {
		t.Fatalf("unexpected verdicts %+v", vs)
	}
	if texts[2] != " привет " {
		t.Fatal("input slice must not be modified")
	}
}

func TestClassifyBatchValidation(t *testing.T) {
	s := newService(&fakePredictor{})
	tests := []struct {
		name  string
		texts []string
		field string
	}{
		{"empty", nil, "texts"},
		{"too many", make([]string, MaxBatch+1), "texts"},
		{"bad item", []string{"ok", ""}, "texts[1]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := s.ClassifyBatch(context.Background(), tt.texts)
			var ve *ValidationError
			if !errors.As(err, &ve) || ve.Field != tt.field {
				t.Fatalf("expected validation error on %q, got %v", tt.field, err)
			}
		})
	}
}
