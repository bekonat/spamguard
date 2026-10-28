package metrics

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Metrics struct {
	registry      *prometheus.Registry
	requests      *prometheus.CounterVec
	duration      *prometheus.HistogramVec
	cache         *prometheus.CounterVec
	verdicts      *prometheus.CounterVec
	batchSize     prometheus.Histogram
	batchDuration prometheus.Histogram
	batchErrors   prometheus.Counter
}

func New() *Metrics {
	m := &Metrics{
		registry: prometheus.NewRegistry(),
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_http_requests_total",
			Help: "HTTP requests by route, method and status code.",
		}, []string{"route", "method", "code"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "gateway_http_request_duration_seconds",
			Help:    "HTTP request latency by route.",
			Buckets: prometheus.DefBuckets,
		}, []string{"route"}),
		cache: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_cache_lookups_total",
			Help: "Prediction cache lookups by result.",
		}, []string{"result"}),
		verdicts: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "gateway_verdicts_total",
			Help: "Fresh model verdicts by label.",
		}, []string{"label"}),
		batchSize: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "gateway_model_batch_size",
			Help:    "Texts per model call.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64},
		}),
		batchDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "gateway_model_call_duration_seconds",
			Help:    "Latency of model calls, including retries.",
			Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5},
		}),
		batchErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "gateway_model_call_errors_total",
			Help: "Model calls that failed after all retries.",
		}),
	}
	m.registry.MustRegister(
		m.requests, m.duration, m.cache, m.verdicts, m.batchSize, m.batchDuration, m.batchErrors,
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
	)
	return m
}

func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

func (m *Metrics) ObserveCache(hit bool) {
	if hit {
		m.cache.WithLabelValues("hit").Inc()
		return
	}
	m.cache.WithLabelValues("miss").Inc()
}

func (m *Metrics) ObserveVerdict(label string) {
	m.verdicts.WithLabelValues(label).Inc()
}

func (m *Metrics) ObserveBatch(size int, elapsed time.Duration, err error) {
	m.batchSize.Observe(float64(size))
	m.batchDuration.Observe(elapsed.Seconds())
	if err != nil {
		m.batchErrors.Inc()
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Middleware must wrap the ServeMux directly: the mux sets r.Pattern on the
// request it receives, and the pattern keeps the route label cardinality low.
func (m *Metrics) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)

		route := r.Pattern
		if route == "" {
			route = "unmatched"
		}
		m.requests.WithLabelValues(route, r.Method, strconv.Itoa(rec.status)).Inc()
		m.duration.WithLabelValues(route).Observe(time.Since(start).Seconds())
	})
}
