package batcher

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/bekonat/spamguard/gateway/internal/model"
)

var ErrClosed = errors.New("batcher is closed")

type Predictor interface {
	Predict(ctx context.Context, texts []string) (model.Result, error)
}

type Options struct {
	MaxSize     int
	MaxWait     time.Duration
	MaxInFlight int
	Timeout     time.Duration
	OnBatch     func(size int, elapsed time.Duration, err error)
}

type outcome struct {
	pred    model.Prediction
	version string
	err     error
}

type job struct {
	text string
	resp chan outcome
}

// Batcher groups concurrent single-text requests into one model call.
// A single loop goroutine owns the pending batch; flushed batches run in
// their own goroutines, limited by MaxInFlight.
type Batcher struct {
	p    Predictor
	opts Options
	in   chan *job
	sem  chan struct{}
	done chan struct{}
	once sync.Once
	wg   sync.WaitGroup
}

func New(p Predictor, opts Options) *Batcher {
	if opts.OnBatch == nil {
		opts.OnBatch = func(int, time.Duration, error) {}
	}
	b := &Batcher{
		p:    p,
		opts: opts,
		in:   make(chan *job),
		sem:  make(chan struct{}, opts.MaxInFlight),
		done: make(chan struct{}),
	}
	b.wg.Add(1)
	go b.loop()
	return b
}

func (b *Batcher) Predict(ctx context.Context, text string) (model.Prediction, string, error) {
	j := &job{text: text, resp: make(chan outcome, 1)}
	select {
	case b.in <- j:
	case <-ctx.Done():
		return model.Prediction{}, "", ctx.Err()
	case <-b.done:
		return model.Prediction{}, "", ErrClosed
	}

	select {
	case o := <-j.resp:
		return o.pred, o.version, o.err
	case <-ctx.Done():
		return model.Prediction{}, "", ctx.Err()
	}
}

// Close stops accepting requests and waits for pending batches to finish.
func (b *Batcher) Close() {
	b.once.Do(func() { close(b.done) })
	b.wg.Wait()
}

func (b *Batcher) loop() {
	defer b.wg.Done()

	var (
		batch []*job
		timer *time.Timer
		tick  <-chan time.Time
	)
	flush := func() {
		if timer != nil {
			timer.Stop()
		}
		tick = nil
		if len(batch) == 0 {
			return
		}
		jobs := batch
		batch = nil
		b.sem <- struct{}{}
		b.wg.Add(1)
		go b.run(jobs)
	}

	for {
		select {
		case j := <-b.in:
			batch = append(batch, j)
			if len(batch) == 1 {
				timer = time.NewTimer(b.opts.MaxWait)
				tick = timer.C
			}
			if len(batch) >= b.opts.MaxSize {
				flush()
			}
		case <-tick:
			flush()
		case <-b.done:
			flush()
			return
		}
	}
}

func (b *Batcher) run(jobs []*job) {
	defer func() {
		<-b.sem
		b.wg.Done()
	}()

	texts := make([]string, len(jobs))
	for i, j := range jobs {
		texts[i] = j.text
	}

	ctx, cancel := context.WithTimeout(context.Background(), b.opts.Timeout)
	defer cancel()

	start := time.Now()
	res, err := b.p.Predict(ctx, texts)
	b.opts.OnBatch(len(jobs), time.Since(start), err)

	for i, j := range jobs {
		if err != nil {
			j.resp <- outcome{err: err}
			continue
		}
		j.resp <- outcome{pred: res.Predictions[i], version: res.Version}
	}
}
