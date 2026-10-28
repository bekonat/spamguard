# spamguard

A spam-detection service built from two parts. A Go API gateway sits in front of a Python model service and handles the serving work: request batching, caching, retries, metrics and health probes. The model service handles only inference.

The model comes from my university NLP project. It is a TF-IDF model over character n-grams (3–5) with logistic regression, trained on a corpus of Russian-language SMS (bank phishing, loans, casinos, deliveries and ordinary messages). I picked character n-grams because they hold up against typos and against Latin look-alike letters inside Cyrillic words, a common obfuscation trick.

```
                    ┌──────────────────────── gateway (Go) ────────────────────────┐
 POST /classify ──▶ │ validate ─▶ LRU cache ─▶ singleflight ─▶ micro-batcher ──────┼──▶ POST /predict ──▶ model (FastAPI)
                    │              (TTL)      (dedupe same    (≤32 texts or 5 ms,  │     {"texts":[...]}    TF-IDF + LogReg
                    │                          text in flight) ≤4 batches in flight)│
                    │ /metrics  /healthz  /readyz  /openapi.yaml                    │      /metrics /healthz
                    └───────────────────────────────────────────────────────────────┘
```

## Why a gateway

Inference cost is paid per model call, not per text. Vectorizing and scoring 32 texts at once costs about the same as one, and the gap is much bigger on a GPU. The gateway groups concurrent single-text requests into one batch:

- The **micro-batcher** (`internal/batcher`) has one goroutine that owns the pending batch. It flushes the batch when it reaches `BATCH_SIZE` or when `BATCH_WAIT` has passed since the first request. A semaphore caps concurrent model calls at `MAX_IN_FLIGHT`. When every slot is busy, the loop stops accepting work, so the backpressure reaches the callers and goroutines do not pile up.
- **singleflight** sends a text that is already in flight to the model only once. The shared call is detached from the first caller's context, so one client disconnecting does not fail the others waiting on the same result.
- The **LRU cache with TTL** reuses `lru.Cache` from my [go-dsa](https://github.com/bekonat/go-dsa) library. Entries are keyed by the SHA-256 of the trimmed text, and errors are never cached.
- The **model client** retries network errors and 5xx responses with exponential backoff, does not retry 4xx, and checks that the model returned exactly one prediction per text.

## Load test

Run locally on a 4-core laptop: the model runs with 2 uvicorn workers, and the client sends 3000 requests from 64 concurrent connections, each with a unique text.

| Mode | Throughput | p50 | p95 | p99 | Model calls |
|------|-----------:|----:|----:|----:|------------:|
| Batching off (`BATCH_SIZE=1`) | 930 req/s | 70 ms | 102 ms | 115 ms | 3000 |
| Batching on (`BATCH_SIZE=32`) | **1283 req/s** | **50 ms** | **53 ms** | **54 ms** | **97** |
| Same text repeated (cache hits) | 14137 req/s | 2.8 ms | 8.2 ms | 33 ms | 1 |

Batching gave 38% more throughput, roughly halved p95, and cut model calls by 31×. It also made latency far more predictable, since p99 is almost equal to p50. To reproduce: `make load`.

## Model quality

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|---------:|----------:|-------:|---:|
| Held-out test split (20%) | 0.981 | 0.982 | 0.973 | 0.977 |
| Hand-written stress set (typos, obfuscation) | 0.925 | 1.000 | 0.850 | 0.919 |

The Docker build trains the model and records the metrics in `artifacts/metrics.json`. Each image therefore carries exactly one model version, which is returned in every response as `model_version`.

## Running

```bash
docker compose up --build        # gateway :8080, model (internal), Prometheus :9090
./scripts/smoke.sh               # end-to-end check against the running stack
```

```bash
curl -X POST localhost:8080/api/v1/classify \
  -d '{"text":"Ваш Kaspi Gold ограничен. Пройдите верификацию по ссылке kaspi-verify.top"}'

curl -X POST localhost:8080/api/v1/classify/batch \
  -d '{"texts":["Срочный займ без проверок!","Купи хлеб по дороге домой"]}'
# {"results":[{"label":"spam","spam_probability":0.9305,"model_version":"20261028-170643","cached":false},
#             {"label":"ham","spam_probability":0.0539,"model_version":"20261028-170643","cached":false}]}
```

Without Docker:

```bash
cd model && pip install -r requirements-dev.txt && python -m app.train && uvicorn app.main:app --port 8000
cd gateway && go run ./cmd/gateway
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/classify` | Classify one text |
| POST | `/api/v1/classify/batch` | Classify up to 100 texts, returned in input order |
| GET | `/healthz` | Liveness, does not depend on the model |
| GET | `/readyz` | Readiness, pings the model service |
| GET | `/metrics` | Prometheus metrics |
| GET | `/openapi.yaml` | OpenAPI 3 specification |

Errors are JSON: `422 {"error":"must not be empty","field":"texts[1]"}`, `400` for malformed JSON, and `503` when the model is unavailable after retries.

## Observability

The gateway exports:
- `gateway_http_requests_total{route,method,code}` and `gateway_http_request_duration_seconds{route}`. The route label is the ServeMux pattern, so cardinality stays bounded.
- `gateway_model_batch_size` and `gateway_model_call_duration_seconds`, which show how well batching works under the current load.
- `gateway_model_call_errors_total`, `gateway_cache_lookups_total{result}` and `gateway_verdicts_total{label}`. The verdict counter makes a drift in the spam rate visible after a model release.

The model service exports `model_inference_seconds`, `model_batch_size` and `model_predictions_total{label}`. Logs are structured JSON (`log/slog`) and carry a request ID that is accepted from or returned in `X-Request-ID`.

## Configuration (gateway)

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_ADDR` | `:8080` | Listen address |
| `MODEL_URL` | `http://localhost:8000` | Model service base URL |
| `MODEL_TIMEOUT` | `2s` | Timeout of a single model HTTP call |
| `MODEL_RETRIES` | `2` | Retries on network errors and 5xx |
| `BATCH_SIZE` | `32` | Maximum texts per model call |
| `BATCH_WAIT` | `5ms` | Maximum time a text waits for its batch to fill |
| `MAX_IN_FLIGHT` | `4` | Concurrent model calls |
| `CACHE_SIZE` | `10000` | LRU cache entries |
| `CACHE_TTL` | `10m` | Cache entry lifetime |
| `MAX_TEXT_LEN` | `2000` | Maximum text length in characters |

## Project structure

```
gateway/
  cmd/gateway/          entrypoint, wiring, graceful shutdown
  cmd/loadtest/         load generator used for the table above
  internal/batcher/     micro-batching with bounded in-flight calls
  internal/classifier/  validation, cache, singleflight, batch fan-out
  internal/model/       HTTP client for the model with retries
  internal/httpapi/     handlers, middleware, error mapping
  internal/metrics/     Prometheus metrics and HTTP middleware
  api/openapi.yaml      API specification, served by the gateway
model/
  app/train.py          trains the model, writes model.joblib and metrics.json
  app/main.py           FastAPI inference service
  app/text.py           text normalization (URLs, phones, homoglyphs, stemming)
  app/corpus.py         synthetic SMS corpus generator
deploy/
  k8s/                  Deployments, Services, ConfigMap, probes, resource limits
  prometheus.yml        scrape config for docker compose
```

## Testing

```bash
make test        # go test -race ./... and pytest
```

The gateway tests run with `-race`. They cover batch flushing by size and by timer, the in-flight limit, delivery of an error to every caller in a batch, cancellation, draining on shutdown, cache TTL, singleflight deduplication when the first caller cancels, retry policy, and the HTTP error mapping. CI runs both test suites and then brings up the stack with docker compose for a smoke test.
