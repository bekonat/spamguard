package model

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var ErrUnavailable = errors.New("model service unavailable")

type Prediction struct {
	Label           string  `json:"label"`
	SpamProbability float64 `json:"spam_probability"`
}

type Result struct {
	Version     string       `json:"model_version"`
	Predictions []Prediction `json:"predictions"`
}

type HTTPClient struct {
	baseURL string
	http    *http.Client
	retries int
	backoff time.Duration
}

func NewHTTPClient(baseURL string, timeout time.Duration, retries int) *HTTPClient {
	return &HTTPClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: timeout},
		retries: retries,
		backoff: 50 * time.Millisecond,
	}
}

func (c *HTTPClient) Predict(ctx context.Context, texts []string) (Result, error) {
	body, err := json.Marshal(map[string][]string{"texts": texts})
	if err != nil {
		return Result{}, err
	}

	var lastErr error
	for attempt := 0; attempt <= c.retries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return Result{}, ctx.Err()
			case <-time.After(c.backoff << (attempt - 1)):
			}
		}

		res, retry, err := c.predictOnce(ctx, body)
		if err == nil {
			if len(res.Predictions) != len(texts) {
				return Result{}, fmt.Errorf("model returned %d predictions for %d texts", len(res.Predictions), len(texts))
			}
			return res, nil
		}
		lastErr = err
		if !retry || ctx.Err() != nil {
			break
		}
	}
	return Result{}, fmt.Errorf("%w: %v", ErrUnavailable, lastErr)
}

func (c *HTTPClient) predictOnce(ctx context.Context, body []byte) (Result, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/predict", bytes.NewReader(body))
	if err != nil {
		return Result{}, false, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return Result{}, true, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return Result{}, resp.StatusCode >= 500, fmt.Errorf("model responded %d: %s", resp.StatusCode, msg)
	}

	var res Result
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return Result{}, false, fmt.Errorf("decode model response: %w", err)
	}
	return res, false, nil
}

func (c *HTTPClient) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/healthz", nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("model healthz responded %d", resp.StatusCode)
	}
	return nil
}
