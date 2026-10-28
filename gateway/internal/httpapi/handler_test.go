package httpapi

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bekonat/spamguard/gateway/internal/classifier"
	"github.com/bekonat/spamguard/gateway/internal/metrics"
	"github.com/bekonat/spamguard/gateway/internal/model"
)

type fakePredictor struct {
	err error
}

func (f *fakePredictor) Predict(context.Context, string) (model.Prediction, string, error) {
	if f.err != nil {
		return model.Prediction{}, "", f.err
	}
	return model.Prediction{Label: "spam", SpamProbability: 0.97}, "v1", nil
}

type fakeHealth struct {
	err error
}

func (f fakeHealth) Ping(context.Context) error { return f.err }

func newServer(t *testing.T, predErr, healthErr error) *httptest.Server {
	t.Helper()
	m := metrics.New()
	svc := classifier.New(&fakePredictor{err: predErr}, classifier.Options{
		CacheSize: 16, CacheTTL: time.Minute, MaxTextLen: 100, CallTimeout: time.Second,
		OnCache: m.ObserveCache, OnVerdict: m.ObserveVerdict,
	})
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	h := NewHandler(svc, fakeHealth{err: healthErr}, m, log)
	srv := httptest.NewServer(h.Routes())
	t.Cleanup(srv.Close)
	return srv
}

func do(t *testing.T, method, url, body string) (int, string) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

func TestClassify(t *testing.T) {
	srv := newServer(t, nil, nil)
	tests := []struct {
		name   string
		body   string
		status int
		want   string
	}{
		{"ok", `{"text":"вы выиграли приз"}`, http.StatusOK, `"label":"spam"`},
		{"empty text", `{"text":""}`, http.StatusUnprocessableEntity, `"field":"text"`},
		{"broken json", `{"text":`, http.StatusBadRequest, "invalid JSON"},
		{"unknown field", `{"message":"hi"}`, http.StatusBadRequest, "invalid JSON"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			status, body := do(t, http.MethodPost, srv.URL+"/api/v1/classify", tt.body)
			if status != tt.status || !strings.Contains(body, tt.want) {
				t.Fatalf("got %d %s, want %d containing %s", status, body, tt.status, tt.want)
			}
		})
	}
}

func TestClassifyBatch(t *testing.T) {
	srv := newServer(t, nil, nil)

	status, body := do(t, http.MethodPost, srv.URL+"/api/v1/classify/batch", `{"texts":["a","b"]}`)
	if status != http.StatusOK || strings.Count(body, `"label":"spam"`) != 2 {
		t.Fatalf("got %d %s", status, body)
	}

	status, body = do(t, http.MethodPost, srv.URL+"/api/v1/classify/batch", `{"texts":["a",""]}`)
	if status != http.StatusUnprocessableEntity || !strings.Contains(body, `"field":"texts[1]"`) {
		t.Fatalf("got %d %s", status, body)
	}
}

func TestModelUnavailable(t *testing.T) {
	srv := newServer(t, model.ErrUnavailable, nil)
	status, _ := do(t, http.MethodPost, srv.URL+"/api/v1/classify", `{"text":"привет"}`)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("expected 503, got %d", status)
	}
}

func TestProbes(t *testing.T) {
	healthy := newServer(t, nil, nil)
	if status, _ := do(t, http.MethodGet, healthy.URL+"/healthz", ""); status != http.StatusOK {
		t.Fatalf("healthz: %d", status)
	}
	if status, _ := do(t, http.MethodGet, healthy.URL+"/readyz", ""); status != http.StatusOK {
		t.Fatalf("readyz: %d", status)
	}

	broken := newServer(t, nil, errors.New("down"))
	if status, _ := do(t, http.MethodGet, broken.URL+"/healthz", ""); status != http.StatusOK {
		t.Fatalf("healthz must not depend on the model: %d", status)
	}
	if status, _ := do(t, http.MethodGet, broken.URL+"/readyz", ""); status != http.StatusServiceUnavailable {
		t.Fatalf("readyz: %d", status)
	}
}

func TestMetricsUseRoutePatterns(t *testing.T) {
	srv := newServer(t, nil, nil)
	do(t, http.MethodPost, srv.URL+"/api/v1/classify", `{"text":"привет"}`)
	do(t, http.MethodPost, srv.URL+"/api/v1/classify", `{"text":"привет"}`)
	do(t, http.MethodGet, srv.URL+"/no/such/path", "")

	_, body := do(t, http.MethodGet, srv.URL+"/metrics", "")
	for _, want := range []string{
		`gateway_http_requests_total{code="200",method="POST",route="POST /api/v1/classify"} 2`,
		`route="unmatched"`,
		`gateway_cache_lookups_total{result="hit"} 1`,
		`gateway_verdicts_total{label="spam"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output is missing %s", want)
		}
	}
}

func TestRequestID(t *testing.T) {
	srv := newServer(t, nil, nil)
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/healthz", nil)
	req.Header.Set("X-Request-ID", "abc123")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := resp.Header.Get("X-Request-ID"); got != "abc123" {
		t.Fatalf("expected request id to be echoed, got %q", got)
	}
}

func TestOpenAPI(t *testing.T) {
	srv := newServer(t, nil, nil)
	status, body := do(t, http.MethodGet, srv.URL+"/openapi.yaml", "")
	if status != http.StatusOK || !strings.HasPrefix(body, "openapi: 3") {
		t.Fatalf("got %d %.40s", status, body)
	}
}
