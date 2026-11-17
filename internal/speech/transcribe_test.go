package speech

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

func TestEndpointJoinsV1(t *testing.T) {
	if got := Endpoint("http://192.168.68.85:1234"); got != "http://192.168.68.85:1234/v1/audio/transcriptions" {
		t.Fatal(got)
	}
	if got := Endpoint("http://host:1234/v1/"); got != "http://host:1234/v1/audio/transcriptions" {
		t.Fatal(got)
	}
}

func TestTranscribePostsWhisperRequest(t *testing.T) {
	var gotModel, gotFormat, gotGran, gotLang, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("path %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		form, err := r.MultipartReader()
		if err != nil {
			t.Fatal(err)
		}
		for {
			part, err := form.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			b, _ := io.ReadAll(part)
			switch part.FormName() {
			case "model":
				gotModel = string(b)
			case "response_format":
				gotFormat = string(b)
			case "timestamp_granularities[]":
				gotGran = string(b)
			case "language":
				gotLang = string(b)
			case "file":
				if len(b) < 44 {
					t.Errorf("file %d bytes", len(b))
				}
			}
		}
		io.WriteString(w, "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n")
	}))
	defer srv.Close()
	c := Client{BaseURL: srv.URL, APIKey: "secret", Model: "whisper-1", HTTP: srv.Client()}
	f, err := os.CreateTemp(t.TempDir(), "a.wav")
	if err != nil {
		t.Fatal(err)
	}
	pcm := make([]byte, 320) // 10 ms
	if err := writePCM(f, pcm, 16000); err != nil {
		t.Fatal(err)
	}
	f.Close()
	out, err := c.TranscribeFile(context.Background(), f.Name(), "en")
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "whisper-1" || gotFormat != "vtt" || gotGran != "" || gotLang != "en" || gotAuth != "Bearer secret" {
		t.Fatalf("model %q format %q granularity %q lang %q auth %q", gotModel, gotFormat, gotGran, gotLang, gotAuth)
	}
	if !strings.Contains(out, "Hello") {
		t.Fatal(out)
	}
}

func TestTranscribeFallsBackWhenVTTRejected(t *testing.T) {
	// A parakeet.cpp-server rejects vtt; ether must retry with verbose_json
	// + timestamp_granularities[]=word and still return timed cues.
	var reqs []string
	var gotGran string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		form, err := r.MultipartReader()
		if err != nil {
			t.Fatal(err)
		}
		var format string
		for {
			part, err := form.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			b, _ := io.ReadAll(part)
			switch part.FormName() {
			case "response_format":
				format = string(b)
			case "timestamp_granularities[]":
				gotGran = string(b)
			}
		}
		reqs = append(reqs, format)
		if format == "vtt" {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"unsupported response_format 'vtt'"}}`)
			return
		}
		io.WriteString(w, `{"text":"Hi there. Bye now.","segments":[{"id":0,"start":0,"end":4,"text":"Hi there. Bye now."}],`+
			`"words":[{"word":"Hi","start":0.2,"end":0.5},{"word":"there.","start":0.5,"end":0.9},`+
			`{"word":"Bye","start":1.2,"end":1.4},{"word":"now.","start":1.4,"end":1.8}]}`)
	}))
	defer srv.Close()
	c := Client{BaseURL: srv.URL + "/v1", Model: "parakeet", HTTP: srv.Client()}
	out, err := c.Transcribe(context.Background(), strings.NewReader("x"), "a.wav", "en")
	if err != nil {
		t.Fatal(err)
	}
	if len(reqs) != 2 || reqs[0] != "vtt" && reqs[1] != "verbose_json" {
		t.Fatalf("requests = %v, want vtt then verbose_json", reqs)
	}
	if gotGran != "word" {
		t.Fatalf("granularity %q", gotGran)
	}
	if !strings.Contains(out, "Hi there.") || !strings.Contains(out, "Bye now.") {
		t.Fatalf("out = %q", out)
	}
}

func TestTranscribeDoesNotRetryOnOtherErrors(t *testing.T) {
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"could not decode audio"}}`)
	}))
	defer srv.Close()
	c := Client{BaseURL: srv.URL + "/v1", Model: "parakeet", HTTP: srv.Client()}
	_, err := c.Transcribe(context.Background(), strings.NewReader("x"), "a.wav", "en")
	if err == nil || !strings.Contains(err.Error(), "could not decode audio") {
		t.Fatalf("err = %v", err)
	}
	if n != 1 {
		t.Fatalf("requests = %d, want 1", n)
	}
}

func TestTranscribeReportsAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"model not loaded"}}`)
	}))
	defer srv.Close()
	c := Client{BaseURL: srv.URL + "/v1", Model: "whisper-1", HTTP: srv.Client()}
	_, err := c.Transcribe(context.Background(), strings.NewReader("x"), "a.wav", "")
	if err == nil || !strings.Contains(err.Error(), "model not loaded") {
		t.Fatal(err)
	}
}

func TestSplitKeepsShortWavWhole(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "a.wav")
	if err != nil {
		t.Fatal(err)
	}
	if err := writePCM(f, make([]byte, 64), 16000); err != nil {
		t.Fatal(err)
	}
	f.Close()
	chunks, cleanup, err := splitWAV(f.Name(), 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if len(chunks) != 1 || chunks[0].path != f.Name() || chunks[0].offset != 0 {
		t.Fatalf("%+v", chunks)
	}
}

func TestCueOffsetStitchesChunks(t *testing.T) {
	start, end, ok := cueTimes("00:00:01.000 --> 00:00:02.000")
	if !ok || start != time.Second || end != 2*time.Second {
		t.Fatal(start, end, ok)
	}
	text := renderVTT([]cue{{start: start + time.Minute, end: end + time.Minute, text: "Later"}})
	if !strings.Contains(text, "00:01:01.000") {
		t.Fatal(text)
	}
}
