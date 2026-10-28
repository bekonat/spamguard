package config

import (
	"testing"
	"time"
)

func TestDefaults(t *testing.T) {
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HTTPAddr != ":8080" || cfg.BatchSize != 32 || cfg.BatchWait != 5*time.Millisecond {
		t.Fatalf("unexpected defaults %+v", cfg)
	}
}

func TestOverrides(t *testing.T) {
	t.Setenv("MODEL_URL", "http://model:8000")
	t.Setenv("BATCH_SIZE", "64")
	t.Setenv("CACHE_TTL", "1m")

	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ModelURL != "http://model:8000" || cfg.BatchSize != 64 || cfg.CacheTTL != time.Minute {
		t.Fatalf("overrides were not applied: %+v", cfg)
	}
}

func TestInvalid(t *testing.T) {
	tests := map[string]string{
		"BATCH_SIZE":    "zero",
		"MODEL_TIMEOUT": "soon",
		"MAX_IN_FLIGHT": "0",
		"MODEL_RETRIES": "-1",
	}
	for key, val := range tests {
		t.Run(key, func(t *testing.T) {
			t.Setenv(key, val)
			if _, err := Load(); err == nil {
				t.Fatalf("expected an error for %s=%s", key, val)
			}
		})
	}
}
