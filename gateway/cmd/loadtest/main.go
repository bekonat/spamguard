package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	url := flag.String("url", "http://localhost:8080", "gateway base URL")
	n := flag.Int("n", 2000, "total requests")
	c := flag.Int("c", 50, "concurrent clients")
	unique := flag.Bool("unique", true, "make every text unique so the cache does not help")
	flag.Parse()

	client := &http.Client{Timeout: 10 * time.Second}
	jobs := make(chan int)
	latencies := make([]time.Duration, *n)
	var failed atomic.Int64

	var wg sync.WaitGroup
	start := time.Now()
	for range *c {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				text := "Срочный займ без проверок, деньги на карту за 5 минут"
				if *unique {
					text = fmt.Sprintf("%s #%d", text, i)
				}
				body, _ := json.Marshal(map[string]string{"text": text})

				t := time.Now()
				resp, err := client.Post(*url+"/api/v1/classify", "application/json", bytes.NewReader(body))
				latencies[i] = time.Since(t)
				if err != nil {
					failed.Add(1)
					continue
				}
				resp.Body.Close()
				if resp.StatusCode != http.StatusOK {
					failed.Add(1)
				}
			}
		}()
	}
	for i := range *n {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	elapsed := time.Since(start)

	slices.Sort(latencies)
	pct := func(p float64) time.Duration { return latencies[int(float64(len(latencies)-1)*p)] }

	fmt.Printf("requests: %d, concurrency: %d, errors: %d\n", *n, *c, failed.Load())
	fmt.Printf("elapsed:  %v, throughput: %.0f req/s\n", elapsed.Round(time.Millisecond), float64(*n)/elapsed.Seconds())
	fmt.Printf("latency:  p50 %v, p95 %v, p99 %v\n",
		pct(0.50).Round(time.Microsecond*100), pct(0.95).Round(time.Microsecond*100), pct(0.99).Round(time.Microsecond*100))
	if failed.Load() > 0 {
		os.Exit(1)
	}
}
