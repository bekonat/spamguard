package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bekonat/spamguard/gateway/internal/batcher"
	"github.com/bekonat/spamguard/gateway/internal/classifier"
	"github.com/bekonat/spamguard/gateway/internal/config"
	"github.com/bekonat/spamguard/gateway/internal/httpapi"
	"github.com/bekonat/spamguard/gateway/internal/metrics"
	"github.com/bekonat/spamguard/gateway/internal/model"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log); err != nil {
		log.Error("fatal", slog.String("error", err.Error()))
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	m := metrics.New()
	client := model.NewHTTPClient(cfg.ModelURL, cfg.ModelTimeout, cfg.ModelRetries)

	// a batch may spend up to (retries+1) timeouts plus backoff talking to the model
	batchTimeout := time.Duration(cfg.ModelRetries+1)*cfg.ModelTimeout + time.Second
	b := batcher.New(client, batcher.Options{
		MaxSize:     cfg.BatchSize,
		MaxWait:     cfg.BatchWait,
		MaxInFlight: cfg.MaxInFlight,
		Timeout:     batchTimeout,
		OnBatch:     m.ObserveBatch,
	})
	defer b.Close()

	svc := classifier.New(b, classifier.Options{
		CacheSize:   cfg.CacheSize,
		CacheTTL:    cfg.CacheTTL,
		MaxTextLen:  cfg.MaxTextLen,
		CallTimeout: batchTimeout + time.Second,
		OnCache:     m.ObserveCache,
		OnVerdict:   m.ObserveVerdict,
	})
	handler := httpapi.NewHandler(svc, client, m, log)

	srv := &http.Server{
		Addr:              cfg.HTTPAddr,
		Handler:           handler.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      batchTimeout + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("http server started", slog.String("addr", cfg.HTTPAddr), slog.String("model_url", cfg.ModelURL))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}
