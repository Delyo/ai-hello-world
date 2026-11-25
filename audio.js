/**
 * audio.js — Subtle Web Audio Harmonic Feedback
 *
 * Maps token surprisal (-log2 p) to soft celesta/marimba-like harmonic plucks.
 * Low surprisal (predictable tokens) land on grounded root/fifth tones;
 * high surprisal (unexpected leaps) shimmer on upper Lydian intervals.
 * Strictly opt-in (muted by default) and requires zero external audio files.
 */
(function (global) {
  'use strict';

  // D Lydian pentatonic / extended frequencies (warm, contemplative, never harsh)
  // D4, E4, F#4, A4, B4, C#5, D5, E5, G#5, A5
  const SCALE = [293.66, 329.63, 369.99, 440.0, 493.88, 554.37, 587.33, 659.25, 830.61, 880.0];

  let ctx = null;
  let enabled = false;

  function ensureContext() {
    if (!ctx) {
      const AudioCtx = global.AudioContext || global.webkitAudioContext;
      if (!AudioCtx) return null;
      ctx = new AudioCtx();
    }
    if (ctx.state === 'suspended') {
      ctx.resume().catch(() => {});
    }
    return ctx;
  }

  function setEnabled(next) {
    enabled = Boolean(next);
    if (enabled) {
      ensureContext();
    }
    return enabled;
  }

  function isEnabled() {
    return enabled;
  }

  function playNote(freq, duration, gainPeak, type) {
    const audio = ensureContext();
    if (!audio || !enabled) return;

    const now = audio.currentTime;
    const osc = audio.createOscillator();
    const overtone = audio.createOscillator();
    const gain = audio.createGain();
    const overtoneGain = audio.createGain();

    osc.type = type || 'sine';
    osc.frequency.setValueAtTime(freq, now);

    overtone.type = 'triangle';
    overtone.frequency.setValueAtTime(freq * 2, now);

    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(gainPeak, now + 0.012);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    overtoneGain.gain.setValueAtTime(0.0001, now);
    overtoneGain.gain.exponentialRampToValueAtTime(gainPeak * 0.14, now + 0.008);
    overtoneGain.gain.exponentialRampToValueAtTime(0.0001, now + duration * 0.55);

    osc.connect(gain);
    overtone.connect(overtoneGain);
    overtoneGain.connect(gain);
    gain.connect(audio.destination);

    osc.start(now);
    overtone.start(now);
    osc.stop(now + duration + 0.03);
    overtone.stop(now + duration + 0.03);
  }

  /**
   * Play a subtle harmonic note shaped by the token's surprisal (bits).
   */
  function playToken(surprisalBits, isManual) {
    if (!enabled) return;
    // Map surprisal [0 .. 4.5 bits] across SCALE indices
    const clamped = Math.max(0, Math.min(4.5, Number(surprisalBits) || 0));
    const idx = Math.min(SCALE.length - 1, Math.round((clamped / 4.5) * (SCALE.length - 1)));
    const freq = SCALE[idx];
    playNote(freq, isManual ? 0.42 : 0.34, isManual ? 0.065 : 0.05, 'sine');
  }

  /**
   * Soft resolution dyad when <EOS> completes a greeting.
   */
  function playResolve() {
    if (!enabled) return;
    playNote(293.66, 0.72, 0.042, 'sine'); // D4
    setTimeout(() => {
      if (enabled) playNote(440.0, 0.75, 0.034, 'sine'); // A4
    }, 55);
  }

  global.HelloWorldAudio = {
    setEnabled,
    isEnabled,
    playToken,
    playResolve,
  };
})(window);
