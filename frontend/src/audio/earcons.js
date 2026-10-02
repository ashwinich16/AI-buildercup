// Short, gentle sounds that tell the shopper what is happening without words.
// Generated with the Web Audio API, so there are no sound files to load.
// Browsers only allow page sounds once the audio engine was started during a click,
// so `unlock()` is called from click handlers (picking a shopper, Start talking).

let context = null;

function audioContext() {
  context = context || new (window.AudioContext || window.webkitAudioContext)();
  if (context.state === "suspended") context.resume();
  return context;
}

function play(frequencies, { duration = 0.16, gap = 0.04, volume = 0.3 } = {}) {
  try {
    const context = audioContext();
    let start = context.currentTime;
    for (const frequency of frequencies) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, start);
      gain.gain.exponentialRampToValueAtTime(volume, start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(start);
      oscillator.stop(start + duration + 0.02);
      start += duration + gap;
    }
  } catch {
    /* sound is a helpful extra, never essential */
  }
}

export const earcons = {
  unlock: () => {
    try {
      audioContext();
    } catch {
      /* no Web Audio: the status text still tells the shopper what is happening */
    }
  },
  yourTurn: () => play([660, 880]), // rising: "I'm listening, go ahead"
  thinking: () => play([520], { duration: 0.1, volume: 0.18 }), // soft tick: "got it, working on it"
  paused: () => play([700, 500]), // falling: "I'll be quiet"
  goodbye: () => play([880, 660, 440]), // descending: "conversation ended"
};
