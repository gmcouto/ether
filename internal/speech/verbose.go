package speech

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// parseVerboseJSON turns an OpenAI-compatible verbose_json transcription
// body into timed cues. Transcription servers place per-word timestamps in
// different spots: parakeet.cpp-server puts one synthetic segment plus a
// top-level "words" array, while whisper.cpp-server puts a "words" array on
// each segment. Both shapes are handled, and a server that returns no words
// at all falls back to segment-level cues.
func parseVerboseJSON(b []byte) ([]cue, error) {
	var v verboseResponse
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, fmt.Errorf("speech-to-text returned unreadable json: %w", err)
	}
	if len(v.Segments) > maxVerboseSegments {
		v.Segments = v.Segments[:maxVerboseSegments]
	}
	words := v.Words
	if len(words) == 0 {
		for i := range v.Segments {
			words = append(words, v.Segments[i].Words...)
		}
	}
	if len(words) == 0 {
		// No word timestamps: use whole segments as cues.
		var out []cue
		for _, s := range v.Segments {
			text := strings.TrimSpace(s.Text)
			start, end := secs(s.Start), secs(s.End)
			if text == "" || end <= start {
				continue
			}
			out = append(out, cue{start: start, end: end, text: text})
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("speech-to-text returned no timed cues")
		}
		return out, nil
	}
	if len(words) > maxVerboseWords {
		words = words[:maxVerboseWords]
	}
	return groupWords(words), nil
}

// verboseResponse is the subset of an OpenAI verbose_json transcription
// that ether uses to build timed cues.
type verboseResponse struct {
	Segments []verboseSegment `json:"segments"`
	Words    []spokenWord     `json:"words"`
}

type verboseSegment struct {
	Start float64      `json:"start"`
	End   float64      `json:"end"`
	Text  string       `json:"text"`
	Words []spokenWord `json:"words"`
}

type spokenWord struct {
	Word  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

// Body guards: 8 MiB responses, 4k segments, 200k words.
const (
	maxVerboseSegments = 4_000
	maxVerboseWords    = 200_000
)

const (
	// maxCueDuration and maxCueWords hard-cap a cue; sentence punctuation
	// may end a cue sooner once it has been on screen for minCueDuration.
	maxCueDuration = 7 * time.Second
	maxCueWords    = 24
	minCueDuration = 700 * time.Millisecond
)

func secs(f float64) time.Duration { return time.Duration(f * float64(time.Second)) }

// groupWords packs per-word timestamps into readable cue-sized lines,
// breaking at sentence punctuation when the cue is long enough to stand
// alone and hard-capping duration and word count. Words without a usable
// end time join the current cue without extending it.
func groupWords(words []spokenWord) []cue {
	var out []cue
	var cueStart, lastEnd time.Duration
	var buf []string
	prevPunct := false
	open := false

	flush := func() {
		text := strings.TrimSpace(strings.Join(buf, " "))
		end := lastEnd
		if end <= cueStart {
			end = cueStart + time.Second
		}
		if text != "" && end > cueStart {
			out = append(out, cue{start: cueStart, end: end, text: text})
		}
		buf = buf[:0]
		open = false
	}

	for _, w := range words {
		word := strings.TrimSpace(w.Word)
		if word == "" {
			continue
		}
		s, e := secs(w.Start), secs(w.End)
		if e < s {
			e = s
		}
		// Boundary is decided before the next word lands: once its start
		// would push the cue past maxCueDuration (or the cue is full, or a
		// finished sentence has been on screen long enough), flush first.
		if open && len(buf) > 0 &&
			(s-cueStart >= maxCueDuration || len(buf) >= maxCueWords ||
				(prevPunct && lastEnd-cueStart >= minCueDuration)) {
			flush()
		}
		if !open {
			cueStart = s
			if cueStart < 0 {
				cueStart = 0
			}
			open = true
		}
		buf = append(buf, word)
		if e > lastEnd {
			lastEnd = e
		}
		prevPunct = sentenceEnd(word)
	}
	if open {
		flush()
	}
	return out
}

// sentenceEnd reports whether a word ends a sentence, so a cue boundary is
// natural. Trailing closing quotes next to the punctuation still count.
func sentenceEnd(word string) bool {
	s := []rune(strings.TrimRight(word, "\"'”’)]}"))
	if len(s) == 0 {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?', '…', '。':
		return true
	}
	return false
}
