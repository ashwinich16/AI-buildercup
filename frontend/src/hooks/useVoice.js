import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// When Scout speaks through speakers, the microphone can hear Scout's own voice.
// If what was heard is mostly words Scout is saying, it is an echo, not the shopper.
// Works in any language: it compares words, not phrases. Matching is fuzzy, because
// speech recognition often mishears Scout's voice ("shirt" -> "short").
export const words = (text) => (text || "").toLowerCase().match(/[\p{L}\p{N}']+/gu) || [];

function editDistance(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i += 1) {
    let diagonal = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const above = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] === b[j - 1] ? 0 : 1));
      diagonal = above;
    }
  }
  return row[b.length];
}

function soundsAlike(heardWord, spokenWord) {
  if (heardWord === spokenWord) return true;
  const shorter = Math.min(heardWord.length, spokenWord.length);
  if (shorter < 3) return false;
  if (heardWord.startsWith(spokenWord) || spokenWord.startsWith(heardWord)) return true;
  const longer = Math.max(heardWord.length, spokenWord.length);
  return 1 - editDistance(heardWord, spokenWord) / longer >= 0.7;
}

function looksLikeEcho(heard, spokenText, threshold = 0.6) {
  if (!spokenText) return false;
  const heardWords = words(heard);
  if (!heardWords.length) return true;
  const spoken = [...new Set(words(spokenText))];
  const echoed = heardWords.filter((word) => spoken.some((said) => soundsAlike(word, said)));
  return echoed.length / heardWords.length >= threshold;
}

// Speech that STARTS this soon after Scout stops may still be Scout's last words reaching the
// recogniser late; later than that, it is the shopper answering.
const ECHO_TAIL_MS = 1200;

// Listening: the browser's speech recognition, kept open for the whole conversation
// (no start-up delay between turns). It restarts itself if the browser closes it.
// `language` is the shopper's accent/language, e.g. "en-IN", "hi-IN".
export function useSpeechInput({ language, onInterim, onFinal, onReady }) {
  const Recognition =
    typeof window !== "undefined" && (window.SpeechRecognition || window.webkitSpeechRecognition);
  const [active, setActive] = useState(false); // the shopper wants Scout to listen
  const [hearing, setHearing] = useState(false); // the microphone is actually capturing
  const [interim, setInterim] = useState("");
  const [blocked, setBlocked] = useState(false); // microphone permission refused

  const activeRef = useRef(false);
  const recognitionRef = useRef(null);
  const utteranceStartRef = useRef(null); // when the phrase being recognised started (first partial result)
  const launchRef = useRef(null);
  const languageRef = useRef(language);
  languageRef.current = language;
  const handlers = useRef({});
  handlers.current = { onInterim, onFinal, onReady };

  launchRef.current = () => {
    if (!Recognition || recognitionRef.current || !activeRef.current) return;
    const recognition = new Recognition();
    recognition.lang = languageRef.current || "en-IN";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onaudiostart = () => {
      setHearing(true);
      handlers.current.onReady?.();
    };
    recognition.onresult = (event) => {
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0].transcript.trim();
        if (result.isFinal) {
          const startedAt = utteranceStartRef.current ?? Date.now();
          utteranceStartRef.current = null;
          if (text) handlers.current.onFinal?.(text, startedAt);
        } else {
          interimText += `${text} `;
        }
      }
      setInterim(interimText.trim());
      if (interimText.trim()) {
        utteranceStartRef.current ??= Date.now();
        handlers.current.onInterim?.(interimText.trim(), utteranceStartRef.current);
      }
    };
    recognition.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        activeRef.current = false;
        setActive(false);
        setBlocked(true);
      }
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      utteranceStartRef.current = null;
      setHearing(false);
      setInterim("");
      if (activeRef.current) setTimeout(() => launchRef.current?.(), 150); // keep the conversation open
    };

    recognitionRef.current = recognition;
    try {
      recognition.start();
    } catch {
      recognitionRef.current = null;
    }
  };

  const start = useCallback(() => {
    activeRef.current = true;
    setActive(true);
    setBlocked(false);
    launchRef.current?.();
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    setActive(false);
    recognitionRef.current?.stop();
  }, []);

  // A new language takes effect at once: end the current recognition; it restarts with the new one.
  useEffect(() => {
    recognitionRef.current?.stop();
  }, [language]);

  useEffect(() => stop, [stop]);
  return { supported: Boolean(Recognition), active, hearing, interim, blocked, start, stop };
}

// Speaking: Google Cloud Text-to-Speech audio from the backend.
//
// Replies arrive as a stream of text while Gemini writes them. Scout speaks them sentence by sentence:
// as soon as the first short sentence is complete it is synthesised and played (about a second), and
// later sentences are grouped a few at a time and synthesised while earlier ones are playing.

const FIRST_CHUNK_MIN_WORDS = 4;   // "Oh!" alone is too short to be worth its own request
const LATER_CHUNK_CHARS = 150;     // later chunks: a couple of sentences each

// Markdown the model sometimes writes (**bold**, bullets) must not be read out.
const forSpeech = (text) =>
  (text || "").replace(/[*_#`>]+/g, "").replace(/^\s*[-•]\s+/gm, "").replace(/\s+/g, " ").trim();

// Where the complete sentences in `text` end: after . ! ? followed by a space ("$12.99" stays whole).
function completeSentencesEnd(text) {
  let end = 0;
  for (const match of text.matchAll(/[.!?…]+["')\]]*\s+/g)) end = match.index + match[0].length;
  return end;
}

export function useSpeaker() {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef(null);     // the chunk playing right now
  const activeRef = useRef(false);   // a reply is being spoken (including the moment between chunks)
  const requestIdRef = useRef(0);
  const spokenTextRef = useRef("");
  const startedAtRef = useRef(0);
  const endedAtRef = useRef(0);

  const stop = useCallback(() => {
    requestIdRef.current += 1; // cancels chunks still being fetched or queued, and the "finished" callback
    audioRef.current?.pause();
    audioRef.current = null;
    if (activeRef.current) endedAtRef.current = Date.now();
    activeRef.current = false;
    setSpeaking(false);
  }, []);

  const playChunk = (blob, requestId) =>
    new Promise((resolve) => {
      if (requestId !== requestIdRef.current) return resolve();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      const done = () => {
        URL.revokeObjectURL(url);
        if (audioRef.current === audio) audioRef.current = null;
        resolve();
      };
      audio.onended = done;
      audio.onpause = done; // stop() pauses it
      audio.onerror = done;
      audio.play().catch(done);
    });

  // Start speaking a reply that is still being written. Returns { push(text), end() }.
  // onFinished runs once everything pushed has been spoken (not if Scout is interrupted).
  const startStream = useCallback(
    (rate = 1, onFinished) => {
      stop();
      const requestId = requestIdRef.current;
      const current = () => requestId === requestIdRef.current;
      spokenTextRef.current = "";
      let pending = "";        // text received but not yet sent for synthesis
      let chunksSent = 0;
      let ended = false;
      const audio = [];        // promises of synthesised chunks, in speaking order
      let wake = null;         // resolves when a new chunk is queued or the reply ends

      const queue = (text) => {
        const clean = forSpeech(text);
        if (!clean) return;
        audio.push(api.speech(clean, rate).catch(() => null));
        chunksSent += 1;
        wake?.();
      };

      const flush = () => {
        const end = ended ? pending.length : completeSentencesEnd(pending);
        const ready = pending.slice(0, end);
        const enough = ended
          || (chunksSent === 0 ? words(ready).length >= FIRST_CHUNK_MIN_WORDS : ready.length >= LATER_CHUNK_CHARS);
        if (ready.trim() && enough) {
          queue(ready);
          pending = pending.slice(end);
        }
      };

      (async () => {
        for (let next = 0; ; next += 1) {
          while (next >= audio.length && !ended) await new Promise((resolve) => (wake = resolve));
          if (!current()) return;                       // interrupted
          if (next >= audio.length) break;              // everything has been spoken
          const blob = await audio[next];
          if (!current()) return;
          if (!blob) continue;                          // one failed chunk: carry on with the rest
          if (!activeRef.current) {
            activeRef.current = true;
            startedAtRef.current = Date.now();
            setSpeaking(true);
          }
          await playChunk(blob, requestId);
        }
        if (activeRef.current) endedAtRef.current = Date.now();
        activeRef.current = false;
        setSpeaking(false);
        onFinished?.(); // the reply is on screen even if no audio played; carry on with the conversation
      })();

      return {
        push(text) {
          if (!current() || ended) return;
          spokenTextRef.current += text;
          pending += text;
          flush();
        },
        end() {
          if (!current() || ended) return;
          ended = true;
          flush();
          wake?.();
        },
      };
    },
    [stop]
  );

  const speak = useCallback(
    (text, rate = 1, onFinished) => {
      const stream = startStream(rate, onFinished);
      stream.push(text);
      stream.end();
    },
    [startStream]
  );

  // Do these words sound like what Scout last said?
  const isEcho = useCallback((heard) => looksLikeEcho(heard, spokenTextRef.current), []);

  // Was Scout audible when a phrase started? "playing", "tail" (just stopped) or null (quiet).
  const audibleAt = useCallback((time) => {
    const playing = time >= startedAtRef.current && (activeRef.current || time <= endedAtRef.current);
    if (startedAtRef.current && playing) return "playing";
    if (time - endedAtRef.current < ECHO_TAIL_MS) return "tail";
    return null;
  }, []);

  useEffect(() => stop, [stop]);
  return { speaking, speak, startStream, stop, isEcho, audibleAt };
}
