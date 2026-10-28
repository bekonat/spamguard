package model

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func predictServer(t *testing.T, fail int32, status int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n <= fail {
			w.WriteHeader(status)
			return
		}
		var req struct {
			Texts []string `json:"texts"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode: %v", err)
		}
		res := Result{Version: "v1"}
		for range req.Texts {
			res.Predictions = append(res.Predictions, Prediction{Label: "ham", SpamProbability: 0.1})
		}
		_ = json.NewEncoder(w).Encode(res)
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func newClient(url string) *HTTPClient {
	c := NewHTTPClient(url, time.Second, 2)
	c.backoff = time.Millisecond
	return c
}

func TestPredict(t *testing.T) {
	srv, _ := predictServer(t, 0, 0)
	res, err := newClient(srv.URL).Predict(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Version != "v1" || len(res.Predictions) != 2 {
		t.Fatalf("unexpected result %+v", res)
	}
}

func TestRetriesServerErrors(t *testing.T) {
	srv, calls := predictServer(t, 2, http.StatusBadGateway)
	if _, err := newClient(srv.URL).Predict(context.Background(), []string{"a"}); err != nil {
		t.Fatalf("expected success on third attempt, got %v", err)
	}
	if calls.Load() != 3 {
		t.Fatalf("expected 3 attempts, got %d", calls.Load())
	}
}

func TestGivesUpAfterRetries(t *testing.T) {
	srv, calls := predictServer(t, 10, http.StatusInternalServerError)
	_, err := newClient(srv.URL).Predict(context.Background(), []string{"a"})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected ErrUnavailable, got %v", err)
	}
	if calls.Load() != 3 {
		t.Fatalf("expected 3 attempts, got %d", calls.Load())
	}
}

func TestDoesNotRetryClientErrors(t *testing.T) {
	srv, calls := predictServer(t, 10, http.StatusUnprocessableEntity)
	if _, err := newClient(srv.URL).Predict(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected an error")
	}
	if calls.Load() != 1 {
		t.Fatalf("4xx must not be retried, got %d attempts", calls.Load())
	}
}

func TestPredictionCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(Result{Version: "v1"})
	}))
	defer srv.Close()

	if _, err := newClient(srv.URL).Predict(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected an error when the model returns fewer predictions")
	}
}

func TestUnreachableModel(t *testing.T) {
	srv, _ := predictServer(t, 0, 0)
	url := srv.URL
	srv.Close()

	if _, err := newClient(url).Predict(context.Background(), []string{"a"}); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected ErrUnavailable, got %v", err)
	}
}

func TestPing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	if err := newClient(srv.URL).Ping(context.Background()); err != nil {
		t.Fatal(err)
	}
}
