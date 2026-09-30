package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"
)

const upstreamAddress = "127.0.0.1:8080"

func main() {
	logger := log.New(os.Stderr, "[penclaw-health-proxy] ", log.LstdFlags|log.LUTC)
	serverArgs := append([]string{"--host", "127.0.0.1", "--port", "8080"}, os.Args[1:]...)
	child := exec.Command("/app/llama-server", serverArgs...)
	child.Stdout, child.Stderr = os.Stdout, os.Stderr
	if err := child.Start(); err != nil {
		logger.Fatalf("start llama-server: %v", err)
	}

	target, _ := url.Parse("http://" + upstreamAddress)
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.FlushInterval = -1
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		logger.Printf("upstream unavailable: %v", err)
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "model server unavailable"})
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ping", pingHandler("http://"+upstreamAddress+"/health"))
	mux.Handle("/", proxy)

	server := &http.Server{Addr: ":80", Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	serverDone := make(chan error, 1)
	go func() { serverDone <- server.ListenAndServe() }()
	childDone := make(chan error, 1)
	go func() { childDone <- child.Wait() }()
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-signals:
		logger.Printf("received %s", sig)
	case err := <-childDone:
		logger.Printf("llama-server exited: %v", err)
	case err := <-serverDone:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Printf("proxy exited: %v", err)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = server.Shutdown(ctx)
	if child.Process != nil {
		_ = child.Process.Signal(syscall.SIGTERM)
	}
}

func pingHandler(healthURL string) http.HandlerFunc {
	client := &http.Client{Timeout: 2 * time.Second}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
			return
		}
		response, err := client.Get(healthURL)
		if err != nil {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodHead {
			return
		}
		body, _ := io.ReadAll(io.LimitReader(response.Body, 64*1024))
		if len(body) == 0 {
			body = []byte(`{"status":"ok"}`)
		}
		_, _ = w.Write(body)
	}
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
