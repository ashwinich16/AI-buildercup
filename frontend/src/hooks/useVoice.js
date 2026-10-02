import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";

// When Scout speaks through speakers, the microphone can hear Scout's own voice.
// If what was heard is mostly words Scout is saying, it is an echo, not the shopper.
// Works in any language: it compares words, not phrases.
const words = (text) => (text || "").toLowerCase().match(/[\p{L}\p{N}']+/gu) || [];

function looksLikeEcho(heard, spokenText, threshold = 0.7) {
  if (!spokenText) return false;
  const heardWords = words(heard);
  if (!heardWords.length) return true;
  const spoken = new Set(words(spokenText));
  return heardWords.filter((word) => spoken.has(word)).length / heardWords.length >= threshold;
}

const ECHO_TAIL_MS = 2000; // the microphone can still deliver Scout's words shortly after it stops

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
          if (text) handlers.current.onFinal?.(text);
        } else {
          interimText += `${text} `;
        }
      }
      setInterim(interimText.trim());
      if (interimText.trim()) handlers.current.onInterim?.(interimText.trim());
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
export function useSpeaker() {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef(null);
  const requestIdRef = useRef(0);
  const spokenTextRef = useRef("");
  const endedAtRef = useRef(0);

  const stop = useCallback(() => {
    requestIdRef.current += 1; // cancels audio still being fetched, and its "finished" callback
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      endedAtRef.current = Date.now();
    }
    setSpeaking(false);
  }, []);

  const speak = useCallback(
    async (text, rate = 1, onFinished) => {
      stop();
      const requestId = requestIdRef.current;
      spokenTextRef.current = text;
      const finish = () => {
        endedAtRef.current = Date.now();
        setSpeaking(false);
        if (requestId === requestIdRef.current) onFinished?.();
      };
      try {
        const blob = await api.speech(text, rate);
        if (requestId !== requestIdRef.current) return; // interrupted while loading
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          URL.revokeObjectURL(url);
          audioRef.current = null;
          finish();
        };
        setSpeaking(true);
        await audio.play();
      } catch {
        finish(); // the reply is still on screen; carry on with the conversation
      }
    },
    [stop]
  );

  // Is what the microphone heard just Scout's own voice?
  const isEcho = useCallback((heard) => {
    const recentlySpoke = audioRef.current || Date.now() - endedAtRef.current < ECHO_TAIL_MS;
    return recentlySpoke && looksLikeEcho(heard, spokenTextRef.current);
  }, []);

  useEffect(() => stop, [stop]);
  return { speaking, speak, stop, isEcho };
}
