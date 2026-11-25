/**
 * app.js — Interactive UI Controller, Attention Arc Visualizer & Multiverse History
 */
(function (global) {
  'use strict';

  const Engine = global.HelloWorldEngine;
  const Sound = global.HelloWorldAudio;

  // ── DOM References ───────────────────────────────────────────────────────
  const els = {
    pulseDot: document.getElementById('pulseDot'),
    phaseReadout: document.getElementById('phaseReadout'),
    totalBitsReadout: document.getElementById('totalBitsReadout'),
    tokenCountReadout: document.getElementById('tokenCountReadout'),

    lineageBtn: document.getElementById('lineageBtn'),
    closeLineageBtn: document.getElementById('closeLineageBtn'),
    lineageModal: document.getElementById('lineageModal'),

    soundBtn: document.getElementById('soundBtn'),
    soundLabel: document.getElementById('soundLabel'),
    themeBtn: document.getElementById('themeBtn'),
    themeLabel: document.getElementById('themeLabel'),

    stageCanvas: document.getElementById('stageCanvas'),
    attentionSvg: document.getElementById('attentionSvg'),
    greetingFlow: document.getElementById('greetingFlow'),
    inspectorBar: document.getElementById('inspectorBar'),

    stepEntropyReadout: document.getElementById('stepEntropyReadout'),
    distSubtitle: document.getElementById('distSubtitle'),
    candidateList: document.getElementById('candidateList'),

    regenerateBtn: document.getElementById('regenerateBtn'),
    stepBtn: document.getElementById('stepBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    pauseLabel: document.getElementById('pauseLabel'),
    clearBtn: document.getElementById('clearBtn'),

    tempSlider: document.getElementById('tempSlider'),
    tempReadout: document.getElementById('tempReadout'),
    tempInterpretation: document.getElementById('tempInterpretation'),

    lensGrid: document.getElementById('lensGrid'),
    lensTagline: document.getElementById('lensTagline'),

    contextForm: document.getElementById('contextForm'),
    contextInput: document.getElementById('contextInput'),
    contextChips: document.getElementById('contextChips'),

    historyStrip: document.getElementById('historyStrip'),
    historyCount: document.getElementById('historyCount'),
  };

  // ── Application State ────────────────────────────────────────────────────
  const state = {
    lensId: 'awakening',
    temperature: 0.78,
    contextInfo: Engine.analyzeContext(''),
    currentStateKey: 'START',
    history: [],           // emitted tokens for current run
    currentDist: null,     // latest computed distribution for currentStateKey
    inspectedIdx: null,    // hovered token index (null => latest emitted)
    previewText: '',       // hovered candidate ghost text
    isRunning: false,
    isFinished: false,
    wasForked: false,
    timerId: null,
    runs: [],              // completed trajectories
  };

  // ── Helpers ──────────────────────────────────────────────────────────────
  function clearTimer() {
    if (state.timerId !== null) {
      clearTimeout(state.timerId);
      state.timerId = null;
    }
  }

  function totalSurprisal(tokens) {
    return tokens.reduce((sum, t) => sum + (t.surprisal || 0), 0);
  }

  function visibleTokens(tokens) {
    return tokens.filter((t) => t.text !== '<EOS>');
  }

  function formatCandidateLabel(text) {
    if (text === '<EOS>') return '<span class="cand-eos-glyph">↵ end</span>';
    const escaped = escapeHtml(text);
    if (text.startsWith(' ')) {
      return '<span class="cand-space">·</span>' + escaped.slice(1);
    }
    return escaped;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function interpretTemperature(T) {
    if (T <= 0.005) {
      return state.lensId === 'classic' && !state.contextInfo.hasContext
        ? 'Deterministic (1978 mode) — collapses to exact "Hello, World!" at 0.00 bits.'
        : 'Argmax deterministic — always commits the highest-logit token (0.00 bits).';
    }
    if (T < 0.5) return 'Focused — high-probability paths dominate with slight variation.';
    if (T < 1.05) return 'Balanced — probabilities breathe while staying coherent.';
    return 'High entropy — rare branches and unexpected phrasing awaken.';
  }

  function setPhase(phase) {
    els.phaseReadout.textContent = phase;
    els.pulseDot.classList.remove('active', 'done');
    if (phase === 'SAMPLING' || phase === 'ATTENDING') {
      els.pulseDot.classList.add('active');
    } else if (phase === 'COMPLETE') {
      els.pulseDot.classList.add('done');
    }
  }

  // ── Core Engine Synchronization ──────────────────────────────────────────
  function recomputeDistribution() {
    if (state.isFinished || state.currentStateKey === 'END') {
      state.currentDist = { candidates: [], entropy: 0, stateKey: 'END' };
    } else {
      state.currentDist = Engine.computeDistribution({
        stateKey: state.currentStateKey,
        lensId: state.lensId,
        temperature: state.temperature,
        contextInfo: state.contextInfo,
        history: state.history,
      });
    }
    renderDistribution();
  }

  function emitToken(candidate, isManual) {
    if (!candidate) return;

    if (isManual) {
      state.wasForked = true;
    }

    const tokenRecord = {
      text: candidate.text,
      id: candidate.id,
      logit: candidate.logit,
      prob: candidate.prob,
      surprisal: candidate.surprisal,
      vec: candidate.vec,
      role: candidate.role,
      meta: candidate.meta || null,
      stateBefore: state.currentStateKey,
      stateAfter: candidate.next,
      isManual: Boolean(isManual),
    };

    state.history.push(tokenRecord);
    state.currentStateKey = candidate.next;
    state.previewText = '';
    state.inspectedIdx = null;

    if (candidate.text === '<EOS>' || candidate.next === 'END') {
      state.isFinished = true;
      state.isRunning = false;
      clearTimer();
      Sound.playResolve();
      recordCompletedRun();
      setPhase('COMPLETE');
    } else {
      Sound.playToken(candidate.surprisal, isManual);
      setPhase(state.isRunning ? 'SAMPLING' : 'PAUSED');
    }

    recomputeDistribution();
    renderStage(true);
    renderTelemetry();
    updatePauseButton();
  }

  function stepOnce() {
    if (state.isFinished) return;
    if (!state.currentDist) recomputeDistribution();
    const chosen = Engine.sampleFromDistribution(state.currentDist, state.temperature);
    if (!chosen) {
      state.isFinished = true;
      state.isRunning = false;
      clearTimer();
      setPhase('COMPLETE');
      updatePauseButton();
      return;
    }
    emitToken(chosen, false);
  }

  function scheduleNextTick() {
    clearTimer();
    if (!state.isRunning || state.isFinished) return;

    const last = state.history[state.history.length - 1];
    let delay = 330;
    if (!last) {
      delay = 260;
    } else if (/[,;:]$/.test(last.text)) {
      delay = 420;
    } else if (/—$/.test(last.text) || /[.!?]$/.test(last.text)) {
      delay = 500;
    } else if (state.temperature <= 0.005) {
      delay = 280;
    }

    state.timerId = setTimeout(() => {
      if (!state.isRunning || state.isFinished) return;
      stepOnce();
      if (state.isRunning && !state.isFinished) {
        scheduleNextTick();
      }
    }, delay);
  }

  function startFreshRun(autoPlay) {
    clearTimer();
    state.history = [];
    state.currentStateKey = 'START';
    state.isFinished = false;
    state.wasForked = false;
    state.inspectedIdx = null;
    state.previewText = '';
    state.isRunning = Boolean(autoPlay);

    setPhase(autoPlay ? 'SAMPLING' : 'SILENCE');
    recomputeDistribution();
    renderStage();
    renderTelemetry();
    updatePauseButton();

    if (autoPlay) {
      scheduleNextTick();
    }
  }

  function togglePause() {
    if (state.isFinished) {
      startFreshRun(true);
      return;
    }
    state.isRunning = !state.isRunning;
    if (state.isRunning) {
      setPhase('SAMPLING');
      scheduleNextTick();
    } else {
      clearTimer();
      setPhase('PAUSED');
    }
    updatePauseButton();
  }

  function rewindToIndex(visibleIdx) {
    // Rewind right BEFORE visible token at visibleIdx so user can steer an alternate branch
    clearTimer();
    state.isRunning = false;
    state.isFinished = false;
    state.wasForked = true;
    state.inspectedIdx = null;
    state.previewText = '';

    const targetTok = state.history[visibleIdx];
    const nextStateKey = targetTok ? targetTok.stateBefore : 'START';
    state.history = state.history.slice(0, visibleIdx);
    state.currentStateKey = nextStateKey;

    setPhase('BRANCHING');
    recomputeDistribution();
    renderStage(false);
    renderTelemetry();
    updatePauseButton();
  }

  function recordCompletedRun() {
    const text = Engine.formatSentence(state.history).trim();
    if (!text) return;

    const vis = visibleTokens(state.history);
    const run = {
      id: Date.now() + Math.random(),
      text,
      lensId: state.lensId,
      temperature: state.temperature,
      bits: totalSurprisal(vis),
      forked: state.wasForked,
      contextRaw: state.contextInfo.raw,
      historySnapshot: state.history.map((t) => ({ ...t })),
    };

    // Avoid consecutive exact duplicates at the very top unless forked
    if (state.runs.length > 0 && state.runs[0].text === run.text && state.runs[0].lensId === run.lensId && !run.forked) {
      state.runs[0] = run;
    } else {
      state.runs.unshift(run);
      if (state.runs.length > 12) state.runs.pop();
    }
    renderHistory();
  }

  function loadRunSnapshot(run) {
    clearTimer();
    state.isRunning = false;
    state.isFinished = true;
    state.wasForked = run.forked;
    state.lensId = run.lensId;
    state.temperature = run.temperature;
    state.history = run.historySnapshot.map((t) => ({ ...t }));
    state.currentStateKey = 'END';
    state.inspectedIdx = null;
    state.previewText = '';

    els.tempSlider.value = String(state.temperature);
    els.tempReadout.textContent = state.temperature.toFixed(2);
    els.tempInterpretation.textContent = interpretTemperature(state.temperature);
    renderLenses();
    setPhase('COMPLETE');
    recomputeDistribution();
    renderStage(false);
    renderTelemetry();
    updatePauseButton();
  }

  // ── Rendering: Stage & Self-Attention Arcs ───────────────────────────────
  function renderStage(animateLatest) {
    const vis = visibleTokens(state.history);
    const flow = els.greetingFlow;
    flow.innerHTML = '';

    if (vis.length === 0 && !state.previewText) {
      const empty = document.createElement('span');
      empty.className = 'greeting-empty';
      empty.textContent = 'Silence before token 0 — click a candidate below or press Breathe Again.';
      flow.appendChild(empty);
      drawAttentionArcs();
      renderInspector();
      return;
    }

    const activeIdx = state.inspectedIdx !== null && state.inspectedIdx < vis.length
      ? state.inspectedIdx
      : vis.length - 1;

    // Precompute attention targets for active token so we can highlight them
    const attnLinks = activeIdx > 0 ? Engine.computeAttentionWeights(vis, activeIdx) : [];
    const attendedSet = new Set(
      attnLinks.filter((l) => l.weight >= 0.14).slice(0, 4).map((l) => l.toIdx)
    );

    vis.forEach((tok, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'token-chip';
      btn.dataset.idx = String(idx);
      btn.textContent = tok.text;

      if (animateLatest && idx === vis.length - 1) btn.classList.add('token-arrive');
      if (idx === activeIdx) btn.classList.add('inspected');
      if (attendedSet.has(idx)) btn.classList.add('attended');
      if (tok.meta && tok.meta.isUserWord) btn.classList.add('user-word');
      if (tok.isManual) btn.classList.add('manual-pick');

      btn.title = `Click to rewind before "${tok.text.trim() || tok.text}" and branch`;

      btn.addEventListener('mouseenter', () => {
        if (state.inspectedIdx !== idx) {
          state.inspectedIdx = idx;
          updateStageHighlightsAndArcs();
        }
      });

      btn.addEventListener('focus', () => {
        if (state.inspectedIdx !== idx) {
          state.inspectedIdx = idx;
          updateStageHighlightsAndArcs();
        }
      });

      btn.addEventListener('click', () => {
        rewindToIndex(idx);
      });

      flow.appendChild(btn);
    });

    // Ghost preview when hovering a candidate in the distribution panel
    if (state.previewText && state.previewText !== '<EOS>') {
      const ghost = document.createElement('span');
      ghost.className = 'ghost-token';
      ghost.textContent = state.previewText;
      flow.appendChild(ghost);
    }

    // Live cursor when not finished
    if (!state.isFinished) {
      const cursor = document.createElement('span');
      cursor.className = 'live-cursor' + (state.isRunning ? '' : ' idle');
      cursor.setAttribute('aria-hidden', 'true');
      flow.appendChild(cursor);
    }

    // Draw SVG attention arcs once DOM layout has settled
    requestAnimationFrame(() => {
      drawAttentionArcs();
      renderInspector();
    });
  }

  /**
   * Update ghost preview in place without rebuilding existing token chips (zero flicker).
   */
  function updateGhostPreview(text) {
    state.previewText = text || '';
    const vis = visibleTokens(state.history);
    if (vis.length === 0) {
      renderStage(false);
      return;
    }

    const flow = els.greetingFlow;
    let ghost = flow.querySelector('.ghost-token');
    const cursor = flow.querySelector('.live-cursor');

    if (state.previewText && state.previewText !== '<EOS>') {
      if (!ghost) {
        ghost = document.createElement('span');
        ghost.className = 'ghost-token';
        if (cursor) {
          flow.insertBefore(ghost, cursor);
        } else {
          flow.appendChild(ghost);
        }
      }
      ghost.textContent = state.previewText;
    } else if (ghost) {
      ghost.remove();
    }

    requestAnimationFrame(drawAttentionArcs);
  }

  function updateStageHighlightsAndArcs() {
    const vis = visibleTokens(state.history);
    if (vis.length === 0) return;

    const activeIdx = state.inspectedIdx !== null && state.inspectedIdx < vis.length
      ? state.inspectedIdx
      : vis.length - 1;

    const attnLinks = activeIdx > 0 ? Engine.computeAttentionWeights(vis, activeIdx) : [];
    const attendedSet = new Set(
      attnLinks.filter((l) => l.weight >= 0.14).slice(0, 4).map((l) => l.toIdx)
    );

    const chips = els.greetingFlow.querySelectorAll('.token-chip');
    chips.forEach((chip) => {
      const idx = parseInt(chip.dataset.idx, 10);
      chip.classList.toggle('inspected', idx === activeIdx);
      chip.classList.toggle('attended', attendedSet.has(idx));
    });

    drawAttentionArcs();
    renderInspector();
  }

  function drawAttentionArcs() {
    const svg = els.attentionSvg;
    svg.innerHTML = '';

    const vis = visibleTokens(state.history);
    if (vis.length < 2) return;

    const activeIdx = state.inspectedIdx !== null && state.inspectedIdx < vis.length
      ? state.inspectedIdx
      : vis.length - 1;

    if (activeIdx <= 0) return;

    const links = Engine.computeAttentionWeights(vis, activeIdx)
      .filter((l) => l.weight >= 0.08)
      .slice(0, 4);

    if (links.length === 0) return;

    const canvasRect = els.stageCanvas.getBoundingClientRect();
    const chips = els.greetingFlow.querySelectorAll('.token-chip');
    const fromEl = chips[activeIdx];
    if (!fromEl) return;

    const fromRect = fromEl.getBoundingClientRect();
    const x1 = fromRect.left - canvasRect.left + fromRect.width * 0.5;
    const y1Top = fromRect.top - canvasRect.top - 4;

    const NS = 'http://www.w3.org/2000/svg';
    let primaryBadge = null;

    // Draw weaker arcs first so the primary arc sits crisp on top
    const ordered = links.slice().reverse();
    ordered.forEach((link, revIdx) => {
      const isPrimary = revIdx === ordered.length - 1;
      const toEl = chips[link.toIdx];
      if (!toEl) return;

      const toRect = toEl.getBoundingClientRect();
      const x2 = toRect.left - canvasRect.left + toRect.width * 0.5;
      const y2Top = toRect.top - canvasRect.top - 4;
      const y2Bottom = toRect.bottom - canvasRect.top + 3;

      const isCrossLine = y1Top - y2Top > 22;
      let pathD = '';
      let targetAnchorY = y2Top;
      let labelX = (x1 + x2) * 0.5;
      let labelY = 14;

      if (isCrossLine) {
        // Sleek cubic S-curve threading cleanly through the inter-line gap (zero text overlap)
        targetAnchorY = y2Bottom;
        const midY = (y1Top + y2Bottom) * 0.5;
        pathD = `M ${x1.toFixed(1)} ${y1Top.toFixed(1)} C ${x1.toFixed(1)} ${midY.toFixed(1)}, ${x2.toFixed(1)} ${midY.toFixed(1)}, ${x2.toFixed(1)} ${y2Bottom.toFixed(1)}`;
        labelY = midY;
      } else {
        // Same-line parabolic arch above the tokens
        const dx = Math.abs(x1 - x2);
        const minY = Math.min(y1Top, y2Top);
        const headroom = Math.max(24, minY - 12);
        const archLift = Math.min(Math.max(22, dx * 0.17 + (isPrimary ? 18 : 6)), headroom);
        const midX = (x1 + x2) * 0.5;
        const ctrlY = Math.max(10, minY - archLift);
        pathD = `M ${x1.toFixed(1)} ${y1Top.toFixed(1)} Q ${midX.toFixed(1)} ${ctrlY.toFixed(1)} ${x2.toFixed(1)} ${y2Top.toFixed(1)}`;
        const apexY = 0.25 * y1Top + 0.5 * ctrlY + 0.25 * y2Top;
        labelY = Math.max(12, apexY - 12);
      }

      const path = document.createElementNS(NS, 'path');
      path.setAttribute('d', pathD);
      path.setAttribute('class', isPrimary ? 'attn-arc attn-arc-primary' : 'attn-arc');
      path.setAttribute('stroke-width', (1.1 + link.weight * 2.6).toFixed(2));
      path.setAttribute('opacity', Math.min(0.95, 0.28 + link.weight * 1.15).toFixed(2));
      svg.appendChild(path);

      // Endpoint anchor dot on attended token
      const dotTo = document.createElementNS(NS, 'circle');
      dotTo.setAttribute('cx', x2.toFixed(1));
      dotTo.setAttribute('cy', targetAnchorY.toFixed(1));
      dotTo.setAttribute('r', isPrimary ? '3.3' : '2.4');
      dotTo.setAttribute('class', isPrimary ? 'attn-node attn-node-primary' : 'attn-node');
      dotTo.setAttribute('opacity', Math.min(0.95, 0.35 + link.weight).toFixed(2));
      svg.appendChild(dotTo);

      if (isPrimary) {
        primaryBadge = { x: labelX, y: labelY, text: `α = ${link.weight.toFixed(2)}` };
      }
    });

    // Anchor dot on the active querying token
    const dotFrom = document.createElementNS(NS, 'circle');
    dotFrom.setAttribute('cx', x1.toFixed(1));
    dotFrom.setAttribute('cy', y1Top.toFixed(1));
    dotFrom.setAttribute('r', '3.5');
    dotFrom.setAttribute('class', 'attn-node attn-node-primary');
    svg.appendChild(dotFrom);

    // Draw crisp badge pill last so it sits cleanly above all arcs
    if (primaryBadge) {
      const pillW = 54;
      const pillH = 17;
      const rx = Math.max(pillW * 0.5 + 4, Math.min(canvasRect.width - pillW * 0.5 - 4, primaryBadge.x));
      const ry = Math.max(pillH * 0.5 + 2, primaryBadge.y);

      const rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', (rx - pillW * 0.5).toFixed(1));
      rect.setAttribute('y', (ry - pillH * 0.5).toFixed(1));
      rect.setAttribute('width', String(pillW));
      rect.setAttribute('height', String(pillH));
      rect.setAttribute('rx', '8.5');
      rect.setAttribute('class', 'attn-label-bg');
      svg.appendChild(rect);

      const label = document.createElementNS(NS, 'text');
      label.setAttribute('x', rx.toFixed(1));
      label.setAttribute('y', ry.toFixed(1));
      label.setAttribute('class', 'attn-label');
      label.textContent = primaryBadge.text;
      svg.appendChild(label);
    }
  }

  function renderInspector() {
    const vis = visibleTokens(state.history);
    if (vis.length === 0) {
      els.inspectorBar.innerHTML = '<span class="inspector-placeholder">Ready at token 0 · Pick a candidate below or press Breathe Again</span>';
      return;
    }

    const activeIdx = state.inspectedIdx !== null && state.inspectedIdx < vis.length
      ? state.inspectedIdx
      : vis.length - 1;
    const tok = vis[activeIdx];
    const links = activeIdx > 0 ? Engine.computeAttentionWeights(vis, activeIdx) : [];
    const topLink = links[0];

    const topAttnHtml = topLink
      ? `<span>attends → <span class="inspector-attn">"${escapeHtml(vis[topLink.toIdx].text)}"</span> (${(topLink.weight * 100).toFixed(0)}%)</span>`
      : '<span class="muted">initial token (attention anchor)</span>';

    const manualTag = tok.isManual ? ' <span class="inspector-accent">· steered</span>' : '';

    els.inspectorBar.innerHTML = `
      <div class="inspector-left">
        <span class="inspector-chip">#${activeIdx + 1} "${escapeHtml(tok.text)}"</span>
        <span>id ${tok.id}</span>
        <span>p = <span class="inspector-accent">${(tok.prob * 100).toFixed(1)}%</span></span>
        <span>${tok.surprisal.toFixed(2)} bits${manualTag}</span>
      </div>
      <div class="inspector-right">
        ${topAttnHtml}
      </div>
    `;
  }

  // ── Rendering: Next-Token Distribution ───────────────────────────────────
  function renderDistribution() {
    const dist = state.currentDist;
    const list = els.candidateList;
    list.innerHTML = '';

    els.stepEntropyReadout.textContent = `${(dist ? dist.entropy : 0).toFixed(2)} bits`;

    if (state.isFinished || !dist || dist.candidates.length === 0) {
      els.distSubtitle.textContent = 'End-of-sequence (<EOS>) reached';
      list.innerHTML = `
        <div class="dist-complete-card">
          <span class="dist-complete-badge">&lt;EOS&gt; COMPLETE</span>
          <p>
            Click any word in the greeting above to rewind and branch into a different trajectory,
            or press <strong>Breathe Again</strong>.
          </p>
        </div>
      `;
      return;
    }

    els.distSubtitle.textContent = `Step ${visibleTokens(state.history).length + 1} · Click any token to steer · Hover to preview`;

    const shown = dist.candidates.slice(0, 8);
    shown.forEach((c, idx) => {
      const row = document.createElement('button');
      row.type = 'button';
      row.className = 'candidate-row';
      row.setAttribute('role', 'listitem');
      if (idx === 0) row.classList.add('is-top');
      if (c.inNucleus) row.classList.add('in-nucleus');
      if (c.text === '<EOS>') row.classList.add('is-eos');

      const pct = Math.max(0, Math.min(100, c.prob * 100));
      const barWidth = pct > 0 && pct < 1.2 ? 1.2 : pct;

      row.innerHTML = `
        <span class="cand-token">${formatCandidateLabel(c.text)}</span>
        <span class="cand-bar-track">
          <span class="cand-bar-fill" style="width: ${barWidth.toFixed(1)}%"></span>
        </span>
        <span class="cand-prob">${pct >= 99.95 ? '100%' : pct.toFixed(1) + '%'}</span>
        <span class="cand-logit">z ${c.logit >= 0 ? '+' : ''}${c.logit.toFixed(1)}</span>
      `;

      row.addEventListener('mouseenter', () => {
        updateGhostPreview(c.text);
      });

      row.addEventListener('mouseleave', () => {
        if (state.previewText === c.text) {
          updateGhostPreview('');
        }
      });

      row.addEventListener('click', () => {
        // Manual pick pauses auto-run briefly or steps cleanly while letting user continue steering
        const wasRunning = state.isRunning;
        clearTimer();
        emitToken(c, true);
        if (wasRunning && !state.isFinished) {
          scheduleNextTick();
        }
      });

      list.appendChild(row);
    });
  }

  // ── Rendering: Telemetry, Lenses, History ────────────────────────────────
  function renderTelemetry() {
    const vis = visibleTokens(state.history);
    els.tokenCountReadout.textContent = String(vis.length);
    els.totalBitsReadout.textContent = `${totalSurprisal(vis).toFixed(2)} bits`;
  }

  function updatePauseButton() {
    if (state.isFinished) {
      els.pauseLabel.textContent = 'Replay';
    } else if (state.isRunning) {
      els.pauseLabel.textContent = 'Pause';
    } else {
      els.pauseLabel.textContent = 'Resume';
    }
  }

  function renderLenses() {
    els.lensGrid.innerHTML = '';
    const activeLens = Engine.LENSES[state.lensId] || Engine.LENSES.classic;
    els.lensTagline.textContent = activeLens.tagline;

    Object.values(Engine.LENSES).forEach((lens) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'lens-pill' + (lens.id === state.lensId ? ' active' : '');
      btn.setAttribute('role', 'radio');
      btn.setAttribute('aria-checked', String(lens.id === state.lensId));
      btn.textContent = lens.label;

      btn.addEventListener('click', () => {
        state.lensId = lens.id;
        // If moving from Classic@0 to a creative lens, lift temperature gently so it feels alive
        if (lens.id !== 'classic' && state.temperature <= 0.005) {
          state.temperature = lens.defaultTemp;
          els.tempSlider.value = String(state.temperature);
          els.tempReadout.textContent = state.temperature.toFixed(2);
        }
        els.tempInterpretation.textContent = interpretTemperature(state.temperature);
        renderLenses();
        if (state.isFinished || state.history.length === 0) {
          startFreshRun(true);
        } else {
          recomputeDistribution();
        }
      });

      els.lensGrid.appendChild(btn);
    });
  }

  function renderHistory() {
    const strip = els.historyStrip;
    els.historyCount.textContent = `${state.runs.length} ${state.runs.length === 1 ? 'run' : 'runs'}`;

    if (state.runs.length === 0) {
      strip.innerHTML = '<div class="history-empty">Trajectories will collect here as greetings complete.</div>';
      return;
    }

    strip.innerHTML = '';
    state.runs.forEach((run) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'history-card';
      const lensLabel = (Engine.LENSES[run.lensId] || Engine.LENSES.classic).label;
      const forkBadge = run.forked ? '<span class="fork-tag">forked</span>' : '';
      const ctxBadge = run.contextRaw ? ` · "${escapeHtml(run.contextRaw)}"` : '';

      card.innerHTML = `
        <div class="history-text">${escapeHtml(run.text)}</div>
        <div class="history-meta">
          <span>${lensLabel} · T=${run.temperature.toFixed(2)}${ctxBadge}</span>
          <span>${run.bits.toFixed(2)} bits ${forkBadge}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        loadRunSnapshot(run);
      });

      strip.appendChild(card);
    });
  }

  function updateContextChipsUI() {
    const raw = state.contextInfo.raw.toLowerCase();
    els.contextChips.querySelectorAll('.mini-chip[data-ctx]').forEach((chip) => {
      const val = chip.dataset.ctx;
      chip.classList.toggle('active', Boolean(val) && raw === val);
    });
  }

  // ── Event Wiring ─────────────────────────────────────────────────────────
  function bindEvents() {
    els.regenerateBtn.addEventListener('click', () => startFreshRun(true));
    els.stepBtn.addEventListener('click', () => {
      clearTimer();
      state.isRunning = false;
      if (state.isFinished) {
        startFreshRun(false);
      }
      stepOnce();
      updatePauseButton();
    });
    els.pauseBtn.addEventListener('click', togglePause);
    els.clearBtn.addEventListener('click', () => startFreshRun(false));

    // Live temperature scrubbing updates probabilities immediately
    els.tempSlider.addEventListener('input', (e) => {
      state.temperature = parseFloat(e.target.value);
      els.tempReadout.textContent = state.temperature.toFixed(2);
      els.tempInterpretation.textContent = interpretTemperature(state.temperature);
      recomputeDistribution();
    });

    // Temperature preset buttons
    document.querySelectorAll('.chip-btn[data-temp]').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.temperature = parseFloat(btn.dataset.temp);
        els.tempSlider.value = String(state.temperature);
        els.tempReadout.textContent = state.temperature.toFixed(2);
        if (btn.dataset.lens) {
          state.lensId = btn.dataset.lens;
          renderLenses();
        }
        els.tempInterpretation.textContent = interpretTemperature(state.temperature);
        // If user clicked 0.00 Exact, run a fresh breath immediately to demonstrate deterministic collapse!
        if (state.temperature <= 0.005 || state.isFinished) {
          startFreshRun(true);
        } else {
          recomputeDistribution();
        }
      });
    });

    // Live human context updates
    els.contextInput.addEventListener('input', (e) => {
      state.contextInfo = Engine.analyzeContext(e.target.value);
      els.tempInterpretation.textContent = interpretTemperature(state.temperature);
      updateContextChipsUI();
      recomputeDistribution();
    });

    els.contextForm.addEventListener('submit', (e) => {
      e.preventDefault();
      state.contextInfo = Engine.analyzeContext(els.contextInput.value);
      // Lift temperature gently if currently frozen at 0 so the context can speak freely
      if (state.contextInfo.hasContext && state.temperature <= 0.005) {
        state.temperature = 0.65;
        els.tempSlider.value = '0.65';
        els.tempReadout.textContent = '0.65';
      }
      els.tempInterpretation.textContent = interpretTemperature(state.temperature);
      updateContextChipsUI();
      startFreshRun(true);
    });

    els.contextChips.addEventListener('click', (e) => {
      const chip = e.target.closest('.mini-chip');
      if (!chip) return;
      const val = chip.dataset.ctx || '';
      els.contextInput.value = val;
      state.contextInfo = Engine.analyzeContext(val);
      if (val && state.temperature <= 0.005) {
        state.temperature = 0.72;
        els.tempSlider.value = '0.72';
        els.tempReadout.textContent = '0.72';
      }
      els.tempInterpretation.textContent = interpretTemperature(state.temperature);
      updateContextChipsUI();
      startFreshRun(true);
    });

    // Clear hovered token inspection when pointer leaves the greeting canvas
    els.greetingFlow.addEventListener('mouseleave', () => {
      if (state.inspectedIdx !== null) {
        state.inspectedIdx = null;
        updateStageHighlightsAndArcs();
      }
    });

    // Sound toggle
    els.soundBtn.addEventListener('click', () => {
      const next = Sound.setEnabled(!Sound.isEnabled());
      els.soundBtn.setAttribute('aria-pressed', String(next));
      els.soundLabel.textContent = next ? 'Sound: On' : 'Sound: Off';
      if (next) Sound.playToken(1.2, true);
    });

    // Theme toggle (Ink <-> Paper)
    els.themeBtn.addEventListener('click', () => {
      const root = document.documentElement;
      const next = root.dataset.theme === 'paper' ? 'ink' : 'paper';
      root.dataset.theme = next;
      els.themeLabel.textContent = next === 'paper' ? 'Ink' : 'Paper';
    });

    // Lineage modal
    els.lineageBtn.addEventListener('click', () => {
      els.lineageModal.hidden = false;
    });
    els.closeLineageBtn.addEventListener('click', () => {
      els.lineageModal.hidden = true;
    });
    els.lineageModal.addEventListener('click', (e) => {
      if (e.target === els.lineageModal) {
        els.lineageModal.hidden = true;
      }
    });

    // Keyboard shortcuts (ignored while typing in input)
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !els.lineageModal.hidden) {
        els.lineageModal.hidden = true;
        return;
      }
      const tag = (document.activeElement && document.activeElement.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePause();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        startFreshRun(true);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        clearTimer();
        state.isRunning = false;
        if (state.isFinished) startFreshRun(false);
        stepOnce();
        updatePauseButton();
      }
    });

    // Keep SVG attention arcs aligned on viewport resize
    let resizeRaf = null;
    window.addEventListener('resize', () => {
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => {
        drawAttentionArcs();
      });
    });
  }

  // ── Boot ─────────────────────────────────────────────────────────────────
  function init() {
    els.tempSlider.value = String(state.temperature);
    els.tempReadout.textContent = state.temperature.toFixed(2);
    els.tempInterpretation.textContent = interpretTemperature(state.temperature);
    renderLenses();
    renderHistory();
    bindEvents();
    // Take our first breath!
    startFreshRun(true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
