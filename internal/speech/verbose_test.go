package speech

import (
	"strings"
	"testing"
	"time"
)

func TestParseVerboseJSONGroupsWordsBySentence(t *testing.T) {
	// Two sentences, 5 words each: parakeet.cpp shape (top-level words).
	body := `{"text":"Hello there. How are you?","segments":[{"id":0,"start":0,"end":4,"text":"Hello there. How are you?"}],` +
		`"words":[{"word":"Hello","start":0.2,"end":0.5},{"word":"there.","start":0.5,"end":0.9},` +
		`{"word":"How","start":1.2,"end":1.4},{"word":"are","start":1.4,"end":1.7},{"word":"you?","start":1.7,"end":2.0}]}`
	cues, err := parseVerboseJSON([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(cues) != 2 {
		t.Fatalf("cues = %d, want 2: %+v", len(cues), cues)
	}
	if cues[0].text != "Hello there." || cues[0].start != 200*time.Millisecond || cues[0].end != 900*time.Millisecond {
		t.Fatalf("cue0 = %+v", cues[0])
	}
	if cues[1].text != "How are you?" || cues[1].start != 1200*time.Millisecond || cues[1].end != 2*time.Second {
		t.Fatalf("cue1 = %+v", cues[1])
	}
}

func TestParseVerboseJSONWordsInsideSegments(t *testing.T) {
	// whisper.cpp shape: per-segment words, no top-level words array.
	body := `{"text":"One two three","segments":[{"id":0,"start":0,"end":2,"text":"One two three",` +
		`"words":[{"word":"One","start":0.1,"end":0.4},{"word":"two","start":0.4,"end":0.7},{"word":"three","start":0.7,"end":1.0}]}]}`
	cues, err := parseVerboseJSON([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(cues) != 1 || cues[0].text != "One two three" {
		t.Fatalf("cues = %+v", cues)
	}
	if cues[0].start != 100*time.Millisecond || cues[0].end != time.Second {
		t.Fatalf("cue = %+v", cues[0])
	}
}

func TestParseVerboseJSONFallsBackToSegments(t *testing.T) {
	body := `{"text":"Hello","segments":[{"id":0,"start":0,"end":3,"text":"Hello"}]}`
	cues, err := parseVerboseJSON([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(cues) != 1 || cues[0].text != "Hello" || cues[0].start != 0 || cues[0].end != 3*time.Second {
		t.Fatalf("cues = %+v", cues)
	}
}

func TestParseVerboseJSONRejectsBadBodies(t *testing.T) {
	for _, body := range []string{`not json`, `{}`, `{"segments":[]}`, `{"words":[]}`} {
		if _, err := parseVerboseJSON([]byte(body)); err == nil {
			t.Fatalf("body %q: expected error", body)
		}
	}
}

func TestGroupWordsCapsCueDuration(t *testing.T) {
	// 120 words at 0.5s apart is 60s of audio: groupWords must split.
	var words []spokenWord
	for i := 0; i < 120; i++ {
		s := 0.2 + float64(i)*0.5
		words = append(words, spokenWord{Word: "word", Start: s, End: s + 0.4})
	}
	cues := groupWords(words)
	if len(cues) < 8 {
		t.Fatalf("cues = %d, want many splits", len(cues))
	}
	for i, c := range cues {
		if c.end-c.start > maxCueDuration+50*time.Millisecond {
			t.Fatalf("cue %d too long: %s > %s", i, c.end-c.start, maxCueDuration)
		}
		if c.start < 0 || c.end <= c.start {
			t.Fatalf("cue %d bad times: %+v", i, c)
		}
		if strings.TrimSpace(c.text) == "" {
			t.Fatalf("cue %d empty text", i)
		}
	}
}

func TestGroupWordsKeepsUntimedWordInCue(t *testing.T) {
	words := []spokenWord{
		{Word: "one", Start: 0.1, End: 0.3},
		{Word: "two", Start: 0.3, End: 0.5},
		{Word: "three", Start: 0, End: 0}, // no timing: joins, does not extend
	}
	cues := groupWords(words)
	if len(cues) != 1 {
		t.Fatalf("cues = %+v", cues)
	}
	if cues[0].text != "one two three" {
		t.Fatalf("text = %q", cues[0].text)
	}
	if cues[0].end != 500*time.Millisecond {
		t.Fatalf("end = %v", cues[0].end)
	}
}

func TestSentenceEnd(t *testing.T) {
	cases := []struct {
		word string
		want bool
	}{
		{"there.", true}, {"you?", true}, {"wow!", true}, {"no…", true},
		{"hello", false}, {"Mr.", true}, {"end.”", true}, {"okay", false}, {".", true},
	}
	for _, c := range cases {
		if got := sentenceEnd(c.word); got != c.want {
			t.Fatalf("sentenceEnd(%q) = %v, want %v", c.word, got, c.want)
		}
	}
}
