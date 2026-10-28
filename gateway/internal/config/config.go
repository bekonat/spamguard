package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	HTTPAddr        string
	ModelURL        string
	ModelTimeout    time.Duration
	ModelRetries    int
	BatchSize       int
	BatchWait       time.Duration
	MaxInFlight     int
	CacheSize       int
	CacheTTL        time.Duration
	MaxTextLen      int
	ShutdownTimeout time.Duration
}

func Load() (Config, error) {
	cfg := Config{
		HTTPAddr:        getEnv("HTTP_ADDR", ":8080"),
		ModelURL:        getEnv("MODEL_URL", "http://localhost:8000"),
		ShutdownTimeout: 10 * time.Second,
	}

	var err error
	if cfg.ModelTimeout, err = durationEnv("MODEL_TIMEOUT", 2*time.Second); err != nil {
		return Config{}, err
	}
	if cfg.ModelRetries, err = intEnv("MODEL_RETRIES", 2); err != nil {
		return Config{}, err
	}
	if cfg.BatchSize, err = intEnv("BATCH_SIZE", 32); err != nil {
		return Config{}, err
	}
	if cfg.BatchWait, err = durationEnv("BATCH_WAIT", 5*time.Millisecond); err != nil {
		return Config{}, err
	}
	if cfg.MaxInFlight, err = intEnv("MAX_IN_FLIGHT", 4); err != nil {
		return Config{}, err
	}
	if cfg.CacheSize, err = intEnv("CACHE_SIZE", 10000); err != nil {
		return Config{}, err
	}
	if cfg.CacheTTL, err = durationEnv("CACHE_TTL", 10*time.Minute); err != nil {
		return Config{}, err
	}
	if cfg.MaxTextLen, err = intEnv("MAX_TEXT_LEN", 2000); err != nil {
		return Config{}, err
	}

	if cfg.BatchSize < 1 || cfg.MaxInFlight < 1 || cfg.CacheSize < 1 || cfg.MaxTextLen < 1 {
		return Config{}, fmt.Errorf("BATCH_SIZE, MAX_IN_FLIGHT, CACHE_SIZE and MAX_TEXT_LEN must be positive")
	}
	if cfg.ModelRetries < 0 {
		return Config{}, fmt.Errorf("MODEL_RETRIES must not be negative")
	}
	return cfg, nil
}

func getEnv(key, def string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return def
}

func intEnv(key string, def int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return n, nil
}

func durationEnv(key string, def time.Duration) (time.Duration, error) {
	v := os.Getenv(key)
	if v == "" {
		return def, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return d, nil
}
