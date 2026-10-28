package httpapi

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/bekonat/spamguard/gateway/api"
	"github.com/bekonat/spamguard/gateway/internal/batcher"
	"github.com/bekonat/spamguard/gateway/internal/classifier"
	"github.com/bekonat/spamguard/gateway/internal/model"
)

type Classifier interface {
	Classify(ctx context.Context, text string) (classifier.Verdict, error)
	ClassifyBatch(ctx context.Context, texts []string) ([]classifier.Verdict, error)
}

type HealthChecker interface {
	Ping(ctx context.Context) error
}

type Metrics interface {
	Middleware(next http.Handler) http.Handler
	Handler() http.Handler
}

type Handler struct {
	svc     Classifier
	health  HealthChecker
	metrics Metrics
	log     *slog.Logger
}

func NewHandler(svc Classifier, health HealthChecker, metrics Metrics, log *slog.Logger) *Handler {
	return &Handler{svc: svc, health: health, metrics: metrics, log: log}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", h.healthz)
	mux.HandleFunc("GET /readyz", h.readyz)
	mux.Handle("GET /metrics", h.metrics.Handler())
	mux.HandleFunc("GET /openapi.yaml", h.openapi)
	mux.HandleFunc("POST /api/v1/classify", h.classify)
	mux.HandleFunc("POST /api/v1/classify/batch", h.classifyBatch)

	return recoverer(h.log)(requestID(logging(h.log)(h.metrics.Middleware(mux))))
}

type classifyRequest struct {
	Text string `json:"text"`
}

type batchRequest struct {
	Texts []string `json:"texts"`
}

type batchResponse struct {
	Results []classifier.Verdict `json:"results"`
}

func (h *Handler) classify(w http.ResponseWriter, r *http.Request) {
	var req classifyRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	v, err := h.svc.Classify(r.Context(), req.Text)
	if err != nil {
		h.handleError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, v)
}

func (h *Handler) classifyBatch(w http.ResponseWriter, r *http.Request) {
	var req batchRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	vs, err := h.svc.ClassifyBatch(r.Context(), req.Texts)
	if err != nil {
		h.handleError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, batchResponse{Results: vs})
}

func (h *Handler) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (h *Handler) readyz(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), time.Second)
	defer cancel()
	if err := h.health.Ping(ctx); err != nil {
		h.log.WarnContext(ctx, "model is not ready", slog.String("error", err.Error()))
		writeError(w, http.StatusServiceUnavailable, "model service is not ready")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *Handler) openapi(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	_, _ = w.Write(api.OpenAPI)
}

func (h *Handler) handleError(w http.ResponseWriter, r *http.Request, err error) {
	var ve *classifier.ValidationError
	switch {
	case errors.As(err, &ve):
		writeJSON(w, http.StatusUnprocessableEntity, errorResponse{Error: ve.Message, Field: ve.Field})
	case errors.Is(err, context.Canceled):
		// the client went away, nothing useful to send
	case errors.Is(err, model.ErrUnavailable), errors.Is(err, context.DeadlineExceeded), errors.Is(err, batcher.ErrClosed):
		h.log.WarnContext(r.Context(), "model call failed", slog.String("error", err.Error()))
		writeError(w, http.StatusServiceUnavailable, "model service is unavailable, try again later")
	default:
		h.log.ErrorContext(r.Context(), "internal error", slog.String("error", err.Error()),
			slog.String("path", r.URL.Path))
		writeError(w, http.StatusInternalServerError, "internal server error")
	}
}
