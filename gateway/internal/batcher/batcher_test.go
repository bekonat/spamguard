package batcher

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/bekonat/spamguard/gateway/internal/model"
)

type fakePredictor struct {
	mu       sync.Mutex
	batches  [][]string
	delay    time.Duration
	err      error
	inFlight atomic.Int32
	maxSeen  atomic.Int32
}

func (f *fakePredictor) Predict(ctx context.Context, texts []string) (model.Result, error) {
	n := f.inFlight.Add(1)
	defer f.inFlight.Add(-1)
	for {
		m := f.maxSeen.Load()
		if n <= m || f.maxSeen.CompareAndSwap(m, n) {
			break
		}
	}

	f.mu.Lock()
	f.batches = append(f.batches, append([]string(nil), texts...))
	f.mu.Unlock()

	select {
	case <-time.After(f.delay):
	case <-ctx.Done():
		return model.Result{}, ctx.Err()
	}
	if f.err != nil {
		return model.Result{}, f.err
	}

	res := model.Result{Version: "v1"}
	for _, t := range texts {
		res.Predictions = append(res.Predictions, model.Prediction{Label: "label-" + t, SpamProbability: 0.5})
	}
	return res, nil
}

func (f *fakePredictor) batchCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.batches)
}

func defaultOpts() Options {
	return Options{MaxSize: 8, MaxWait: 20 * time.Millisecond, MaxInFlight: 2, Timeout: time.Second}
}

func TestConcurrentRequestsAreBatched(t *testing.T) {
	f := &fakePredictor{}
	b := New(f, defaultOpts())
	defer b.Close()

	const n = 8
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text := fmt.Sprint(i)
			p, version, err := b.Predict(context.Background(), text)
			if err != nil {
				errs <- err
				return
			}
			if p.Label != "label-"+text || version != "v1" {
				errs <- fmt.Errorf("text %q got %+v %q", text, p, version)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}

	if got := f.batchCount(); got != 1 {
		t.Fatalf("expected 1 batch, got %d", got)
	}
}

func TestBatchIsFlushedByTimer(t *testing.T) {
	f := &fakePredictor{}
	opts := defaultOpts()
	opts.MaxSize = 100
	b := New(f, opts)
	defer b.Close()

	start := time.Now()
	if _, _, err := b.Predict(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < opts.MaxWait {
		t.Fatalf("flushed after %v, before MaxWait %v", elapsed, opts.MaxWait)
	}
}

func TestBatchSizeIsLimited(t *testing.T) {
	f := &fakePredictor{}
	opts := defaultOpts()
	opts.MaxSize = 3
	opts.MaxInFlight = 10
	b := New(f, opts)
	defer b.Close()

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _ = b.Predict(context.Background(), fmt.Sprint(i))
		}()
	}
	wg.Wait()

	f.mu.Lock()
	defer f.mu.Unlock()
	total := 0
	for _, batch := range f.batches {
		if len(batch) > opts.MaxSize {
			t.Fatalf("batch of %d exceeds MaxSize %d", len(batch), opts.MaxSize)
		}
		total += len(batch)
	}
	if total != 10 {
		t.Fatalf("expected 10 texts in total, got %d", total)
	}
}

func TestInFlightIsBounded(t *testing.T) {
	f := &fakePredictor{delay: 30 * time.Millisecond}
	opts := defaultOpts()
	opts.MaxSize = 1
	opts.MaxInFlight = 2
	b := New(f, opts)
	defer b.Close()

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _, _ = b.Predict(context.Background(), fmt.Sprint(i))
		}()
	}
	wg.Wait()

	if got := f.maxSeen.Load(); got > int32(opts.MaxInFlight) {
		t.Fatalf("saw %d concurrent model calls, limit is %d", got, opts.MaxInFlight)
	}
}

func TestErrorIsDeliveredToEveryCaller(t *testing.T) {
	boom := errors.New("boom")
	f := &fakePredictor{err: boom}
	b := New(f, defaultOpts())
	defer b.Close()

	var wg sync.WaitGroup
	var failed atomic.Int32
	for i := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := b.Predict(context.Background(), fmt.Sprint(i)); errors.Is(err, boom) {
				failed.Add(1)
			}
		}()
	}
	wg.Wait()

	if failed.Load() != 4 {
		t.Fatalf("expected 4 failures, got %d", failed.Load())
	}
}

func TestCallerCancellation(t *testing.T) {
	f := &fakePredictor{delay: 200 * time.Millisecond}
	b := New(f, defaultOpts())
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	_, _, err := b.Predict(ctx, "slow")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got %v", err)
	}
}

func TestClose(t *testing.T) {
	f := &fakePredictor{delay: 20 * time.Millisecond}
	b := New(f, defaultOpts())

	done := make(chan error, 1)
	go func() {
		_, _, err := b.Predict(context.Background(), "pending")
		done <- err
	}()
	time.Sleep(5 * time.Millisecond)
	b.Close()

	if err := <-done; err != nil {
		t.Fatalf("pending request must complete on Close, got %v", err)
	}
	if _, _, err := b.Predict(context.Background(), "late"); !errors.Is(err, ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}
