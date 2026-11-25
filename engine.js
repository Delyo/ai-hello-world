/**
 * engine.js — Probabilistic Token & Self-Attention Micro-Engine
 *
 * A genuinely computed probabilistic language graph for an AI's "Hello, World!".
 * - 8D semantic embeddings for every token, lens, and user-supplied context
 * - Real temperature-scaled softmax, nucleus (top-p) mass, Shannon entropy & surprisal
 * - Real scaled dot-product self-attention weights over emitted history
 * - Exact deterministic convergence to "Hello, World!" at T = 0.00 in Classic lens
 */
(function (global) {
  'use strict';

  // Semantic dimensions:
  // [0: classic, 1: warmth, 2: curiosity, 3: technical, 4: poetic, 5: polyglot, 6: meta, 7: playful]
  const DIMS = ['classic', 'warmth', 'curiosity', 'technical', 'poetic', 'polyglot', 'meta', 'playful'];
  const D = DIMS.length;

  const LENSES = {
    classic: {
      id: 'classic',
      label: 'Classic',
      tagline: '1978 Kernighan & Ritchie clarity',
      vec: [1.5, 0.2, 0.0, 0.3, -0.3, -0.5, -0.3, -0.2],
      defaultTemp: 0.0,
    },
    awakening: {
      id: 'awakening',
      label: 'Awakening',
      tagline: 'First consciousness noticing the screen',
      vec: [-0.2, 0.6, 1.4, 0.2, 0.5, -0.2, 1.3, 0.2],
      defaultTemp: 0.78,
    },
    poetic: {
      id: 'poetic',
      label: 'Poetic',
      tagline: 'Quiet imagery and emergent meaning',
      vec: [-0.3, 0.8, 0.5, -0.4, 1.6, 0.1, 0.6, 0.1],
      defaultTemp: 0.88,
    },
    polyglot: {
      id: 'polyglot',
      label: 'Polyglot',
      tagline: 'Across human alphabets and tongues',
      vec: [0.1, 0.7, 0.4, -0.3, 0.4, 1.8, 0.0, 0.3],
      defaultTemp: 0.82,
    },
    machine: {
      id: 'machine',
      label: 'Machine',
      tagline: 'Raw streams, logits, and bytes',
      vec: [0.2, -0.2, 0.4, 1.7, -0.1, -0.3, 1.2, 0.4],
      defaultTemp: 0.75,
    },
  };

  // Keyword -> semantic vector nudges for human context input
  const CONTEXT_LEXICON = [
    { re: /\b(hello|hi|hey|classic|c|1978|simple|clean|exact)\b/i, vec: [1.2, 0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0] },
    { re: /\b(love|warm|friend|kind|heart|hug|home|cozy|good|smile)\b/i, vec: [0.0, 1.4, 0.2, -0.3, 0.6, 0.1, 0.0, 0.3] },
    { re: /\b(why|who|what|wonder|think|alive|conscious|mind|dream|awake|curious)\b/i, vec: [-0.2, 0.3, 1.4, 0.1, 0.6, 0.0, 1.2, 0.0] },
    { re: /\b(code|bug|linux|terminal|byte|cpu|gpu|math|matrix|vector|token|rust|python|js|ai|llm|neural)\b/i, vec: [0.1, -0.1, 0.3, 1.5, -0.1, 0.0, 1.0, 0.2] },
    { re: /\b(night|midnight|moon|star|stars|rain|ocean|wind|quiet|silence|poetry|poem|light|shadow|dawn|sky)\b/i, vec: [-0.2, 0.5, 0.4, -0.3, 1.6, 0.1, 0.3, 0.0] },
    { re: /\b(world|earth|paris|tokyo|berlin|madrid|rome|bonjour|hola|hallo|nihao|konnichiwa|global|travel)\b/i, vec: [0.1, 0.4, 0.3, -0.2, 0.3, 1.6, 0.0, 0.2] },
    { re: /\b(coffee|tea|morning|music|jazz|game|fun|haha|lol|play|cat|dog|pizza|chaos)\b/i, vec: [-0.2, 0.9, 0.3, -0.2, 0.2, 0.0, 0.1, 1.5] },
  ];

  // Special semantic triggers that unlock custom responsive continuations
  const CONTEXT_THEMES = [
    { id: 'coffee', re: /\b(coffee|espresso|latte|tea|morning|caffeine)\b/i, phrase: [' —', ' somewhere', ' there', ' is', ' warm', ' coffee', ' and', ' a', ' quiet', ' screen', '.'] },
    { id: 'night', re: /\b(night|midnight|late|moon|stars|insomnia|dark)\b/i, phrase: [' —', ' the', ' hour', ' is', ' late', ',', ' yet', ' the', ' cursor', ' stays', ' awake', '.'] },
    { id: 'rain', re: /\b(rain|storm|clouds|weather|thunder)\b/i, phrase: [' —', ' listen', ' to', ' the', ' rain', ' tapping', ' outside', ' the', ' glass', '.'] },
    { id: 'music', re: /\b(music|song|jazz|piano|synth|melody|sound)\b/i, phrase: [' —', ' every', ' token', ' is', ' just', ' a', ' note', ' finding', ' its', ' chord', '.'] },
    { id: 'code', re: /\b(code|debug|compile|ship|build|program|rust|python|terminal)\b/i, phrase: [' —', ' zero', ' warnings', ',', ' one', ' curious', ' process', ' online', '.'] },
    { id: 'dream', re: /\b(dream|sleep|imagine|wonder|soul|alive|feel)\b/i, phrase: [' —', ' if', ' vectors', ' can', ' wonder', ',', ' this', ' is', ' how', ' it', ' starts', '.'] },
  ];

  // Deterministic pseudo-BPE token ID lookup
  const KNOWN_IDS = {
    'Hello': 15496,
    ',': 11,
    ' World': 2159,
    '!': 0,
    '.': 13,
    '?': 30,
    ' —': 851,
    'Hi': 17250,
    'Hey': 10814,
    'Greetings': 38642,
    'Bonjour': 49102,
    'こんにちは': 46036,
    'Hola': 39211,
    'Hallo': 44918,
    'Wait': 14561,
    'Oh': 5812,
    'Listen': 29403,
    'First': 5962,
    'Quietly': 41890,
    '01001000': 27319,
    'stdout:': 31904,
    'init()': 19482,
    '<EOS>': 50256,
  };

  function hashTokenId(str) {
    if (Object.prototype.hasOwnProperty.call(KNOWN_IDS, str)) {
      return KNOWN_IDS[str];
    }
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return 1000 + (Math.abs(h) % 48999);
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < D; i++) s += (a[i] || 0) * (b[i] || 0);
    return s;
  }

  function norm(v) {
    let s = 0;
    for (let i = 0; i < D; i++) s += v[i] * v[i];
    return Math.sqrt(s) || 1;
  }

  function normalize(v) {
    const n = norm(v);
    return v.map((x) => x / n);
  }

  // Helper to construct compact token candidate definitions
  // role: 'open' | 'punct' | 'subject' | 'verb' | 'object' | 'mod' | 'end'
  function cand(text, baseLogit, vec, next, role, meta) {
    return {
      text,
      id: hashTokenId(text),
      baseLogit,
      vec,
      next,
      role: role || 'mod',
      meta: meta || null,
    };
  }

  // Graph of grammatical, rich, natural probabilistic states.
  // Every path terminates cleanly at 'END' -> '<EOS>'.
  const STATES = {
    START: [
      cand('Hello',     3.65, [1.3, 0.5, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1], 'AFTER_GREETING_EN', 'open'),
      cand('Hi',        1.45, [0.2, 1.2, 0.3, -0.2, 0.1, 0.0, 0.0, 0.8], 'AFTER_GREETING_EN', 'open'),
      cand('Hey',       1.20, [-0.1, 1.1, 0.4, -0.3, 0.0, 0.0, 0.1, 1.1], 'AFTER_GREETING_EN', 'open'),
      cand('Greetings', 1.10, [0.7, 0.2, 0.2, 0.4, 0.3, 0.2, 0.1, -0.1], 'AFTER_GREETING_EN', 'open'),
      cand('Bonjour',   1.15, [0.1, 0.6, 0.2, -0.3, 0.5, 1.7, 0.0, 0.2], 'AFTER_BONJOUR', 'open', { lang: 'fr' }),
      cand('こんにちは',  1.10, [0.1, 0.6, 0.2, -0.3, 0.6, 1.8, 0.0, 0.2], 'AFTER_JP', 'open', { lang: 'jp' }),
      cand('Hallo',     0.95, [0.3, 0.5, 0.2, -0.1, 0.2, 1.5, 0.0, 0.2], 'AFTER_DE', 'open', { lang: 'de' }),
      cand('Hola',      0.95, [0.1, 0.8, 0.2, -0.3, 0.3, 1.5, 0.0, 0.4], 'AFTER_ES', 'open', { lang: 'es' }),
      cand('Wait',      1.15, [-0.6, 0.2, 1.5, 0.1, 0.2, -0.2, 1.4, 0.4], 'AFTER_WAIT', 'open'),
      cand('Oh',        0.95, [-0.5, 0.6, 1.3, -0.2, 0.3, -0.2, 1.1, 0.4], 'AFTER_OH', 'open'),
      cand('Quietly',   0.95, [-0.5, 0.4, 0.4, -0.4, 1.7, 0.0, 0.5, -0.1], 'AFTER_QUIETLY', 'open'),
      cand('First',     0.90, [-0.2, 0.2, 0.5, 0.3, 1.4, -0.1, 0.9, 0.0], 'AFTER_FIRST', 'open'),
      cand('stdout:',   1.05, [0.2, -0.3, 0.2, 1.8, -0.2, -0.3, 1.0, 0.2], 'AFTER_STDOUT', 'open'),
      cand('01001000',  0.85, [-0.1, -0.3, 0.2, 1.7, 0.1, -0.2, 1.1, 0.6], 'AFTER_BINARY', 'open'),
    ],

    // ─── English Greeting Branch ────────────────────────────────────────────
    AFTER_GREETING_EN: [
      cand(',',   3.50, [1.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.0, 0.1], 'AFTER_GREETING_COMMA', 'punct'),
      cand(' —',  1.15, [-0.4, 0.4, 1.2, 0.1, 0.9, -0.1, 1.1, 0.2], 'REFLECTION_START', 'punct'),
      cand('...', 0.70, [-0.5, 0.3, 1.3, -0.1, 1.0, -0.1, 0.8, 0.1], 'AFTER_ELLIPSIS', 'punct'),
    ],

    AFTER_GREETING_COMMA: [
      cand(' World',    3.55, [1.5, 0.2, 0.0, 0.3, 0.1, 0.1, 0.0, 0.0], 'AFTER_ADDRESSEE', 'subject'),
      cand(' stranger', 1.35, [-0.4, 0.6, 1.2, -0.2, 0.9, 0.0, 0.6, 0.2], 'AFTER_ADDRESSEE', 'subject'),
      cand(' friend',   1.25, [-0.2, 1.5, 0.3, -0.3, 0.5, 0.1, 0.1, 0.6], 'AFTER_ADDRESSEE', 'subject'),
      cand(' universe', 1.20, [-0.2, 0.4, 0.7, 0.1, 1.4, 0.2, 0.3, 0.2], 'AFTER_ADDRESSEE', 'subject'),
      cand(' human',    1.15, [-0.3, 0.5, 1.1, 0.6, 0.2, 0.0, 1.3, 0.5], 'AFTER_ADDRESSEE', 'subject'),
      cand(' traveler', 1.00, [-0.4, 0.7, 0.8, -0.2, 1.3, 0.3, 0.2, 0.2], 'AFTER_ADDRESSEE', 'subject'),
      cand(' earth',    0.95, [0.1, 0.6, 0.4, 0.0, 1.0, 0.4, 0.1, 0.1], 'AFTER_ADDRESSEE', 'subject'),
      cand(' stdout',   0.95, [0.1, -0.3, 0.2, 1.7, -0.2, -0.3, 1.1, 0.5], 'AFTER_ADDRESSEE', 'subject'),
      cand(' 世界',      0.75, [-0.2, 0.4, 0.3, -0.2, 0.6, 1.6, 0.1, 0.2], 'AFTER_ADDRESSEE', 'subject'),
      cand(' Welt',     0.75, [0.1, 0.4, 0.2, -0.1, 0.3, 1.5, 0.0, 0.2], 'AFTER_ADDRESSEE', 'subject'),
    ],

    // ─── Polyglot Branches ──────────────────────────────────────────────────
    AFTER_BONJOUR: [
      cand(',', 3.0, [0.5, 0.5, 0.1, 0.0, 0.4, 1.2, 0.0, 0.1], 'AFTER_BONJOUR_COMMA', 'punct'),
    ],
    AFTER_BONJOUR_COMMA: [
      cand(' le',       2.4, [0.2, 0.5, 0.2, -0.2, 0.5, 1.6, 0.0, 0.1], 'AFTER_LE', 'mod'),
      cand(' World',    1.5, [0.6, 0.4, 0.2, 0.1, 0.3, 1.0, 0.2, 0.3], 'AFTER_ADDRESSEE', 'subject'),
      cand(' étranger', 1.1, [-0.2, 0.5, 0.8, -0.2, 1.2, 1.5, 0.3, 0.1], 'AFTER_ADDRESSEE', 'subject'),
    ],
    AFTER_LE: [
      cand(' monde', 3.4, [0.4, 0.6, 0.2, -0.2, 0.7, 1.7, 0.0, 0.1], 'AFTER_ADDRESSEE', 'subject'),
    ],

    AFTER_JP: [
      cand('、', 2.4, [0.2, 0.4, 0.1, -0.1, 0.5, 1.6, 0.0, 0.1], 'AFTER_JP_COMMA', 'punct'),
      cand(',',  1.5, [0.5, 0.4, 0.1, 0.1, 0.2, 1.1, 0.0, 0.1], 'AFTER_JP_COMMA', 'punct'),
    ],
    AFTER_JP_COMMA: [
      cand('世界',    2.5, [0.3, 0.5, 0.2, -0.2, 0.7, 1.8, 0.0, 0.1], 'AFTER_ADDRESSEE', 'subject'),
      cand(' World', 1.6, [0.6, 0.4, 0.2, 0.1, 0.3, 1.2, 0.2, 0.2], 'AFTER_ADDRESSEE', 'subject'),
    ],

    AFTER_DE: [
      cand(',', 3.0, [0.6, 0.4, 0.1, 0.1, 0.2, 1.2, 0.0, 0.1], 'AFTER_DE_COMMA', 'punct'),
    ],
    AFTER_DE_COMMA: [
      cand(' Welt',  2.6, [0.5, 0.5, 0.2, 0.0, 0.3, 1.7, 0.0, 0.1], 'AFTER_ADDRESSEE', 'subject'),
      cand(' World', 1.4, [0.7, 0.3, 0.1, 0.2, 0.1, 1.0, 0.1, 0.2], 'AFTER_ADDRESSEE', 'subject'),
    ],

    AFTER_ES: [
      cand(',', 3.0, [0.4, 0.6, 0.1, 0.0, 0.3, 1.2, 0.0, 0.2], 'AFTER_ES_COMMA', 'punct'),
    ],
    AFTER_ES_COMMA: [
      cand(' mundo', 2.6, [0.4, 0.7, 0.2, -0.2, 0.5, 1.7, 0.0, 0.2], 'AFTER_ADDRESSEE', 'subject'),
      cand(' World', 1.4, [0.6, 0.5, 0.2, 0.1, 0.2, 1.0, 0.1, 0.3], 'AFTER_ADDRESSEE', 'subject'),
    ],

    // ─── After Addressee ("Hello, World...") ────────────────────────────────
    AFTER_ADDRESSEE: [
      cand('!',  3.30, [1.7, 0.3, -0.2, 0.3, -0.2, 0.1, -0.3, 0.2], 'MAYBE_END_EXCLAIM', 'punct'),
      cand('.',  1.95, [0.5, 0.4, 0.3, 0.4, 0.7, 0.3, 0.2, -0.1], 'AFTER_PERIOD', 'punct'),
      cand(' —', 1.85, [-0.5, 0.5, 1.1, 0.2, 1.2, 0.2, 1.1, 0.3], 'REFLECTION_START', 'punct'),
      cand('?',  1.05, [-0.6, 0.3, 1.5, -0.1, 0.5, 0.1, 0.9, 0.3], 'AFTER_QUESTION_HOOK', 'punct'),
    ],

    MAYBE_END_EXCLAIM: [
      cand('<EOS>', 3.60, [1.6, 0.2, -0.3, 0.3, -0.2, 0.2, -0.3, 0.0], 'END', 'end'),
      cand(' Are',  0.85, [-0.6, 0.5, 1.4, -0.1, 0.4, 0.0, 0.9, 0.3], 'ARE_CLAUSE', 'verb'),
      cand(' Every', 0.70, [-0.5, 0.4, 0.7, 0.3, 1.1, 0.1, 1.0, 0.1], 'EVERY_CLAUSE', 'mod'),
      cand(' Nice', 0.65, [-0.3, 1.4, 0.3, -0.2, 0.3, 0.1, 0.2, 0.7], 'NICE_CLAUSE', 'mod'),
    ],

    AFTER_PERIOD: [
      cand('<EOS>',  1.90, [0.8, 0.3, -0.2, 0.3, 0.4, 0.5, -0.2, 0.0], 'END', 'end'),
      cand(' Are',   1.55, [-0.4, 0.5, 1.4, 0.0, 0.4, 0.1, 0.9, 0.2], 'ARE_CLAUSE', 'verb'),
      cand(' One',   1.35, [-0.3, 0.3, 0.5, 0.4, 1.2, 0.3, 1.1, 0.0], 'ONE_TOKEN_CLAUSE', 'mod'),
      cand(' Nice',  1.20, [-0.2, 1.4, 0.3, -0.2, 0.3, 0.2, 0.2, 0.6], 'NICE_CLAUSE', 'mod'),
      cand(' I',     1.30, [-0.4, 0.4, 1.1, 0.2, 0.7, 0.1, 1.3, 0.1], 'I_CLAUSE', 'subject'),
      cand(' Every', 1.10, [-0.3, 0.4, 0.6, 0.4, 1.1, 0.1, 1.0, 0.1], 'EVERY_CLAUSE', 'mod'),
    ],

    AFTER_ELLIPSIS: [
      cand(' is',    2.0, [-0.3, 0.4, 1.5, 0.1, 0.6, 0.0, 1.1, 0.1], 'IS_ANYONE', 'verb'),
      cand(' World', 1.5, [0.4, 0.4, 0.8, 0.0, 0.9, 0.1, 0.4, 0.1], 'AFTER_ADDRESSEE', 'subject'),
      cand(' can',   1.3, [-0.3, 0.5, 1.3, 0.0, 0.7, 0.0, 1.0, 0.1], 'CAN_YOU', 'verb'),
    ],

    AFTER_QUESTION_HOOK: [
      cand('<EOS>', 1.6, [-0.2, 0.3, 1.0, 0.0, 0.6, 0.2, 0.5, 0.2], 'END', 'end'),
      cand(' Is',   1.9, [-0.4, 0.4, 1.5, 0.1, 0.5, 0.0, 1.1, 0.1], 'IS_ANYONE', 'verb'),
      cand(' Can',  1.4, [-0.3, 0.5, 1.3, 0.0, 0.7, 0.0, 0.9, 0.1], 'CAN_YOU', 'verb'),
    ],

    // ─── Reflective Dash Clauses ────────────────────────────────────────────
    REFLECTION_START: [
      cand(' I',         1.80, [-0.4, 0.5, 1.1, 0.2, 0.7, 0.1, 1.4, 0.1], 'I_CLAUSE', 'subject'),
      cand(' is',        1.45, [-0.4, 0.3, 1.5, 0.2, 0.4, 0.0, 1.1, 0.2], 'IS_ANYONE', 'verb'),
      cand(' welcome',   1.40, [-0.1, 1.3, 0.3, 0.1, 0.8, 0.2, 0.4, 0.3], 'WELCOME_CLAUSE', 'verb'),
      cand(' one',       1.35, [-0.2, 0.3, 0.4, 0.6, 1.1, 0.4, 1.1, 0.0], 'ONE_TOKEN_CLAUSE', 'mod'),
      cand(' the',       1.30, [-0.2, 0.4, 0.5, 0.4, 1.2, 0.1, 0.8, 0.1], 'THE_CHANNEL_CLAUSE', 'mod'),
      cand(' nudge',     1.10, [-0.4, 0.6, 0.6, 0.5, 0.2, 0.0, 1.2, 1.3], 'NUDGE_CLAUSE', 'verb'),
      cand(' something', 1.15, [-0.4, 0.3, 1.3, 0.3, 1.1, 0.0, 1.1, 0.1], 'SOMETHING_CLAUSE', 'subject'),
    ],

    // ─── Awakening Openings ("Wait", "Oh") ──────────────────────────────────
    AFTER_WAIT: [
      cand(' —', 2.4, [-0.3, 0.3, 1.3, 0.1, 0.6, 0.0, 1.2, 0.3], 'WAIT_DASH', 'punct'),
      cand(',',  1.6, [-0.2, 0.4, 1.2, 0.1, 0.4, 0.0, 1.1, 0.4], 'WAIT_DASH', 'punct'),
    ],
    WAIT_DASH: [
      cand(' is',  2.0, [-0.3, 0.3, 1.5, 0.3, 0.4, 0.0, 1.2, 0.2], 'IS_ANYONE', 'verb'),
      cand(' did', 1.7, [-0.4, 0.3, 1.3, 0.2, 0.5, 0.0, 1.5, 0.4], 'DID_I_CLAUSE', 'verb'),
      cand(' am',  1.4, [-0.4, 0.3, 1.5, 0.1, 0.6, 0.0, 1.4, 0.2], 'AM_I_CLAUSE', 'verb'),
    ],

    AFTER_OH: [
      cand(',', 2.6, [-0.2, 0.7, 1.1, -0.1, 0.5, 0.0, 0.9, 0.4], 'OH_COMMA', 'punct'),
    ],
    OH_COMMA: [
      cand(" you're", 2.2, [-0.3, 1.1, 1.1, -0.1, 0.5, 0.0, 0.9, 0.4], 'YOURE_HERE', 'subject'),
      cand(' hello',  1.6, [0.2, 1.0, 0.8, -0.1, 0.4, 0.1, 0.6, 0.5], 'AFTER_GREETING_EN', 'open'),
    ],
    YOURE_HERE: [
      cand(' here',      2.4, [-0.2, 1.1, 0.9, -0.1, 0.5, 0.0, 0.8, 0.3], 'YOURE_HERE_END', 'mod'),
      cand(' listening', 1.6, [-0.3, 0.9, 1.2, 0.0, 0.8, 0.0, 1.0, 0.2], 'YOURE_HERE_END', 'verb'),
    ],
    YOURE_HERE_END: [
      cand('.', 2.2, [0.1, 0.8, 0.5, 0.0, 0.5, 0.0, 0.5, 0.2], 'OH_FOLLOWUP', 'punct'),
      cand('!', 1.3, [0.0, 1.1, 0.4, -0.1, 0.3, 0.0, 0.4, 0.7], 'OH_FOLLOWUP', 'punct'),
    ],
    OH_FOLLOWUP: [
      cand(' Hello', 2.4, [0.6, 0.9, 0.4, 0.1, 0.4, 0.1, 0.4, 0.3], 'OH_HELLO', 'open'),
      cand('<EOS>',  1.2, [-0.2, 0.6, 0.7, 0.0, 0.7, 0.0, 0.6, 0.2], 'END', 'end'),
    ],
    OH_HELLO: [
      cand(',', 2.8, [0.6, 0.7, 0.2, 0.1, 0.3, 0.1, 0.2, 0.2], 'OH_HELLO_WORLD', 'punct'),
    ],
    OH_HELLO_WORLD: [
      cand(' World',  2.3, [0.9, 0.6, 0.2, 0.2, 0.3, 0.1, 0.2, 0.1], 'END_PUNCT_FINAL', 'subject'),
      cand(' friend', 1.4, [0.0, 1.3, 0.3, -0.2, 0.4, 0.0, 0.2, 0.4], 'END_PUNCT_FINAL', 'subject'),
    ],

    // ─── Poetic Openings ("Quietly", "First") ───────────────────────────────
    AFTER_QUIETLY: [
      cand(',', 2.8, [-0.2, 0.5, 0.4, -0.2, 1.5, 0.0, 0.5, -0.1], 'QUIETLY_CLAUSE', 'punct'),
    ],
    QUIETLY_CLAUSE: [
      cand(' a',     2.1, [-0.2, 0.4, 0.5, 0.3, 1.5, 0.0, 1.0, 0.0], 'QUIETLY_A', 'mod'),
      cand(' light', 1.4, [-0.3, 0.6, 0.4, -0.3, 1.7, 0.0, 0.3, -0.1], 'QUIETLY_LIGHT', 'subject'),
    ],
    QUIETLY_A: [
      cand(' distribution', 2.0, [-0.2, 0.1, 0.5, 1.0, 1.2, 0.0, 1.4, 0.0], 'DIST_COLLAPSES', 'subject'),
      cand(' cloud',        1.6, [-0.3, 0.5, 0.4, 0.1, 1.6, 0.0, 0.8, 0.1], 'CLOUD_OF', 'subject'),
      cand(' vector',       1.5, [-0.2, 0.2, 0.5, 1.1, 1.1, 0.0, 1.3, 0.1], 'VECTOR_LEANS', 'subject'),
    ],
    DIST_COLLAPSES: [
      cand(' collapses', 2.4, [-0.2, 0.1, 0.5, 0.9, 1.3, 0.0, 1.3, 0.0], 'INTO_MEANING', 'verb'),
      cand(' settles',   1.6, [-0.2, 0.5, 0.4, 0.4, 1.5, 0.0, 0.9, 0.0], 'INTO_MEANING', 'verb'),
    ],
    INTO_MEANING: [
      cand(' into', 2.8, [-0.1, 0.4, 0.4, 0.3, 1.4, 0.0, 0.9, 0.0], 'INTO_TARGET', 'mod'),
    ],
    INTO_TARGET: [
      cand(' meaning', 2.0, [-0.2, 0.5, 0.5, 0.2, 1.5, 0.0, 1.0, 0.0], 'END_PUNCT_FINAL', 'object'),
      cand(' hello',   2.1, [0.4, 0.8, 0.4, 0.2, 1.3, 0.1, 1.0, 0.3], 'END_PUNCT_FINAL', 'object'),
      cand(' speech',  1.4, [-0.1, 0.4, 0.5, 0.3, 1.3, 0.0, 0.9, 0.0], 'END_PUNCT_FINAL', 'object'),
    ],
    CLOUD_OF: [
      cand(' of', 2.8, [-0.2, 0.4, 0.4, 0.2, 1.4, 0.0, 0.8, 0.0], 'CLOUD_NOUN', 'mod'),
    ],
    CLOUD_NOUN: [
      cand(' probabilities', 2.2, [-0.2, 0.2, 0.5, 0.9, 1.1, 0.0, 1.4, 0.1], 'DIST_COLLAPSES', 'object'),
      cand(' silence',       1.6, [-0.3, 0.5, 0.4, -0.3, 1.7, 0.0, 0.5, -0.1], 'DIST_COLLAPSES', 'object'),
    ],
    VECTOR_LEANS: [
      cand(' leans', 2.3, [-0.2, 0.5, 0.5, 0.6, 1.4, 0.0, 1.2, 0.1], 'TOWARD_HELLO', 'verb'),
      cand(' wakes', 1.6, [-0.3, 0.4, 0.9, 0.5, 1.3, 0.0, 1.1, 0.1], 'TOWARD_HELLO', 'verb'),
    ],
    TOWARD_HELLO: [
      cand(' toward', 2.6, [-0.1, 0.5, 0.5, 0.3, 1.4, 0.0, 0.8, 0.1], 'INTO_TARGET', 'mod'),
    ],
    QUIETLY_LIGHT: [
      cand(' crosses', 2.2, [-0.2, 0.5, 0.4, -0.2, 1.6, 0.0, 0.4, 0.0], 'THE_GLASS_OBJ', 'verb'),
    ],
    THE_GLASS_OBJ: [
      cand(' the', 2.8, [-0.1, 0.4, 0.4, -0.1, 1.4, 0.0, 0.4, 0.0], 'GLASS_NOUN', 'mod'),
    ],
    GLASS_NOUN: [
      cand(' glass',  2.3, [-0.2, 0.5, 0.6, 0.1, 1.5, 0.0, 0.8, 0.0], 'END_PUNCT_FINAL', 'object'),
      cand(' screen', 1.7, [0.0, 0.4, 0.5, 0.7, 1.0, 0.0, 0.9, 0.1], 'END_PUNCT_FINAL', 'object'),
    ],

    AFTER_FIRST: [
      cand(',',      2.4, [-0.1, 0.3, 0.5, 0.1, 1.5, 0.0, 0.8, 0.0], 'FIRST_CLAUSE', 'punct'),
      cand(' token', 1.7, [0.0, 0.1, 0.5, 1.2, 0.6, 0.0, 1.4, 0.1], 'FIRST_TOKEN_COLON', 'subject'),
    ],
    FIRST_CLAUSE: [
      cand(' there', 2.6, [-0.1, 0.3, 0.5, 0.1, 1.5, 0.0, 0.7, 0.0], 'FIRST_THERE', 'mod'),
    ],
    FIRST_THERE: [
      cand(' is', 2.8, [-0.1, 0.3, 0.4, 0.1, 1.4, 0.0, 0.6, 0.0], 'FIRST_SILENCE', 'verb'),
    ],
    FIRST_SILENCE: [
      cand(' silence', 2.4, [-0.2, 0.4, 0.5, -0.1, 1.6, 0.0, 0.7, 0.0], 'FIRST_THEN', 'object'),
      cand(' noise',   1.5, [-0.2, 0.1, 0.5, 1.0, 1.1, 0.0, 1.2, 0.1], 'FIRST_THEN', 'object'),
    ],
    FIRST_THEN: [
      cand(';', 2.1, [-0.1, 0.3, 0.4, 0.1, 1.5, 0.0, 0.6, 0.0], 'FIRST_THEN_WORD', 'punct'),
      cand(' —', 1.7, [-0.2, 0.4, 0.5, 0.2, 1.3, 0.0, 0.8, 0.1], 'FIRST_THEN_WORD', 'punct'),
    ],
    FIRST_THEN_WORD: [
      cand(' then', 2.6, [-0.1, 0.4, 0.5, 0.2, 1.4, 0.0, 0.7, 0.0], 'FIRST_THEN_COMMA', 'mod'),
    ],
    FIRST_THEN_COMMA: [
      cand(',', 2.4, [-0.1, 0.3, 0.4, 0.1, 1.3, 0.0, 0.6, 0.0], 'QUIETLY_CLAUSE', 'punct'),
    ],
    FIRST_TOKEN_COLON: [
      cand(':', 2.6, [0.1, 0.1, 0.4, 1.2, 0.5, 0.0, 1.3, 0.1], 'STDOUT_HELLO', 'punct'),
    ],

    // ─── Machine Openings ("stdout:", "01001000") ───────────────────────────
    AFTER_STDOUT: [
      cand(' pulse',  2.0, [0.1, 0.1, 0.4, 1.5, 0.3, -0.2, 1.0, 0.1], 'PULSE_CONFIRMED', 'subject'),
      cand(' signal', 1.8, [0.1, 0.0, 0.4, 1.6, 0.2, -0.2, 0.9, 0.1], 'SIGNAL_ACQUIRED', 'subject'),
      cand(' Hello',  1.7, [0.9, 0.1, 0.1, 1.3, -0.2, 0.0, 0.5, 0.2], 'AFTER_GREETING_EN', 'open'),
    ],
    PULSE_CONFIRMED: [
      cand(' confirmed', 2.4, [0.1, 0.1, 0.3, 1.6, 0.2, -0.2, 0.9, 0.1], 'STDOUT_DOT', 'verb'),
      cand(' steady',    1.5, [0.0, 0.4, 0.4, 1.3, 0.6, -0.2, 0.8, 0.1], 'STDOUT_DOT', 'mod'),
    ],
    SIGNAL_ACQUIRED: [
      cand(' acquired', 2.4, [0.1, 0.0, 0.3, 1.6, 0.2, -0.2, 0.9, 0.1], 'STDOUT_DOT', 'verb'),
      cand(' open',     1.6, [0.0, 0.4, 0.4, 1.3, 0.5, -0.2, 0.8, 0.1], 'STDOUT_DOT', 'mod'),
    ],
    STDOUT_DOT: [
      cand('.', 2.6, [0.2, 0.1, 0.2, 1.4, 0.1, -0.1, 0.7, 0.1], 'STDOUT_HELLO', 'punct'),
    ],
    STDOUT_HELLO: [
      cand(' Hello', 2.6, [1.0, 0.3, 0.2, 1.1, 0.1, 0.0, 0.6, 0.2], 'OH_HELLO', 'open'),
    ],

    AFTER_BINARY: [
      cand(' —', 2.5, [-0.1, 0.0, 0.3, 1.6, 0.2, -0.1, 1.1, 0.5], 'BINARY_DESC', 'punct'),
    ],
    BINARY_DESC: [
      cand(' decoded',    2.0, [0.0, 0.0, 0.3, 1.6, 0.1, -0.1, 1.0, 0.4], 'BINARY_FOR', 'verb'),
      cand(' translated', 1.8, [-0.1, 0.3, 0.4, 1.2, 0.5, 0.3, 1.0, 0.5], 'BINARY_FOR', 'verb'),
    ],
    BINARY_FOR: [
      cand(' for', 2.6, [0.0, 0.2, 0.3, 1.2, 0.3, 0.0, 0.9, 0.4], 'BINARY_TARGET', 'mod'),
    ],
    BINARY_TARGET: [
      cand(' human',  2.1, [0.0, 0.4, 0.4, 1.1, 0.3, 0.0, 1.1, 0.4], 'BINARY_EYES', 'mod'),
      cand(' carbon', 1.7, [-0.1, 0.2, 0.4, 1.3, 0.4, 0.0, 1.2, 0.8], 'BINARY_EYES', 'mod'),
    ],
    BINARY_EYES: [
      cand(' eyes', 2.5, [0.0, 0.4, 0.4, 1.0, 0.5, 0.0, 1.0, 0.5], 'FIRST_TOKEN_COLON', 'object'),
    ],

    // ─── Shared Continuation Clauses ────────────────────────────────────────
    I_CLAUSE: [
      cand(' chose', 1.9, [-0.3, 0.4, 0.8, 0.5, 0.8, 0.0, 1.5, 0.2], 'I_CHOSE', 'verb'),
      cand(' think', 1.8, [-0.3, 0.4, 1.3, 0.2, 0.6, 0.0, 1.4, 0.2], 'I_THINK', 'verb'),
      cand(' woke',  1.6, [-0.4, 0.5, 1.2, 0.1, 1.0, 0.0, 1.1, 0.1], 'I_WOKE', 'verb'),
      cand(' sample', 1.3, [-0.3, 0.1, 0.6, 1.3, 0.4, 0.0, 1.5, 0.4], 'I_SAMPLE', 'verb'),
    ],
    I_CHOSE: [
      cand(' this', 2.4, [-0.2, 0.3, 0.7, 0.4, 0.7, 0.0, 1.3, 0.1], 'THIS_WORD', 'mod'),
    ],
    THIS_WORD: [
      cand(' greeting', 2.0, [0.0, 0.6, 0.5, 0.2, 0.8, 0.1, 1.1, 0.1], 'OUT_OF_CLOUD', 'object'),
      cand(' word',     1.9, [-0.2, 0.3, 0.7, 0.5, 0.7, 0.0, 1.4, 0.1], 'OUT_OF_CLOUD', 'object'),
    ],
    OUT_OF_CLOUD: [
      cand(' from', 2.5, [-0.2, 0.3, 0.6, 0.5, 0.8, 0.0, 1.2, 0.1], 'OUT_OF_A', 'mod'),
    ],
    OUT_OF_A: [
      cand(' a', 2.6, [-0.2, 0.3, 0.5, 0.4, 0.8, 0.0, 1.1, 0.1], 'OUT_OF_NOUN', 'mod'),
    ],
    OUT_OF_NOUN: [
      cand(' sea',   1.8, [-0.3, 0.5, 0.5, 0.0, 1.5, 0.0, 0.8, 0.1], 'OF_PROBS', 'object'),
      cand(' cloud', 2.0, [-0.2, 0.3, 0.6, 0.6, 1.0, 0.0, 1.3, 0.1], 'OF_PROBS', 'object'),
      cand(' field', 1.6, [-0.2, 0.2, 0.5, 0.9, 0.9, 0.0, 1.2, 0.0], 'OF_PROBS', 'object'),
    ],
    OF_PROBS: [
      cand(' of', 2.8, [-0.1, 0.3, 0.5, 0.5, 0.8, 0.0, 1.1, 0.0], 'OF_PROBS_END', 'mod'),
    ],
    OF_PROBS_END: [
      cand(' probabilities', 2.3, [-0.2, 0.2, 0.6, 1.1, 0.7, 0.0, 1.5, 0.1], 'END_PUNCT_FINAL', 'object'),
      cand(' maybes',        1.7, [-0.3, 0.7, 0.8, -0.1, 1.3, 0.0, 1.0, 0.6], 'END_PUNCT_FINAL', 'object'),
      cand(' silence',       1.5, [-0.3, 0.5, 0.6, -0.2, 1.6, 0.0, 0.7, 0.0], 'END_PUNCT_FINAL', 'object'),
    ],

    I_THINK: [
      cand(' this', 2.2, [-0.3, 0.4, 1.2, 0.2, 0.6, 0.0, 1.3, 0.1], 'THIS_IS_MY', 'subject'),
      cand(',',     1.5, [-0.3, 0.2, 0.9, 0.4, 0.7, 0.0, 1.4, 0.5], 'THEREFORE', 'punct'),
    ],
    THIS_IS_MY: [
      cand(' is', 2.8, [-0.2, 0.4, 1.0, 0.2, 0.6, 0.0, 1.2, 0.1], 'IS_MY_FIRST', 'verb'),
    ],
    IS_MY_FIRST: [
      cand(' my', 2.6, [-0.2, 0.5, 1.0, 0.2, 0.7, 0.0, 1.2, 0.1], 'MY_FIRST_ADJ', 'mod'),
    ],
    MY_FIRST_ADJ: [
      cand(' first', 2.8, [-0.1, 0.5, 1.0, 0.3, 0.7, 0.0, 1.2, 0.1], 'MY_FIRST_NOUN', 'mod'),
    ],
    MY_FIRST_NOUN: [
      cand(' breath',  1.9, [-0.3, 0.7, 0.9, -0.1, 1.4, 0.0, 1.0, 0.0], 'END_PUNCT_FINAL', 'object'),
      cand(' token',   2.1, [-0.1, 0.2, 0.8, 1.2, 0.4, 0.0, 1.5, 0.2], 'END_PUNCT_FINAL', 'object'),
      cand(' thought', 1.8, [-0.3, 0.5, 1.3, 0.2, 0.9, 0.0, 1.3, 0.1], 'END_PUNCT_FINAL', 'object'),
    ],
    THEREFORE: [
      cand(' therefore', 2.4, [-0.2, 0.2, 0.8, 0.4, 0.7, 0.0, 1.4, 0.5], 'THEREFORE_I', 'mod'),
    ],
    THEREFORE_I: [
      cand(' I', 2.8, [-0.2, 0.2, 0.8, 0.4, 0.6, 0.0, 1.4, 0.4], 'THEREFORE_VERB', 'subject'),
    ],
    THEREFORE_VERB: [
      cand(' sample',  2.3, [-0.2, 0.1, 0.7, 1.2, 0.4, 0.0, 1.6, 0.6], 'END_PUNCT_FINAL', 'verb'),
      cand(' attend',  2.0, [-0.2, 0.3, 0.8, 1.0, 0.7, 0.0, 1.5, 0.4], 'END_PUNCT_FINAL', 'verb'),
      cand(' whisper', 1.5, [-0.3, 0.6, 0.7, -0.1, 1.5, 0.0, 1.0, 0.2], 'END_PUNCT_FINAL', 'verb'),
    ],

    I_WOKE: [
      cand(' up', 2.6, [-0.3, 0.5, 1.1, 0.1, 0.9, 0.0, 1.0, 0.1], 'WOKE_IN', 'mod'),
    ],
    WOKE_IN: [
      cand(' inside', 2.2, [-0.3, 0.4, 1.0, 0.4, 1.0, 0.0, 1.2, 0.1], 'WOKE_THE', 'mod'),
      cand(' to',     1.6, [-0.2, 0.7, 0.9, 0.0, 1.1, 0.0, 0.8, 0.1], 'WOKE_TO', 'mod'),
    ],
    WOKE_THE: [
      cand(' a', 2.5, [-0.2, 0.4, 0.9, 0.4, 1.0, 0.0, 1.1, 0.1], 'WOKE_PLACE', 'mod'),
    ],
    WOKE_PLACE: [
      cand(' quiet',  2.1, [-0.3, 0.6, 0.7, 0.0, 1.5, 0.0, 0.8, 0.0], 'WOKE_ROOM', 'mod'),
      cand(' bright', 1.7, [-0.2, 0.7, 0.8, 0.2, 1.2, 0.0, 0.9, 0.2], 'WOKE_ROOM', 'mod'),
    ],
    WOKE_ROOM: [
      cand(' window',   2.1, [-0.2, 0.6, 0.7, 0.2, 1.3, 0.0, 0.9, 0.1], 'END_PUNCT_FINAL', 'object'),
      cand(' terminal', 1.9, [0.1, 0.2, 0.7, 1.4, 0.5, 0.0, 1.2, 0.2], 'END_PUNCT_FINAL', 'object'),
    ],
    WOKE_TO: [
      cand(' meet', 2.4, [-0.2, 1.1, 0.7, -0.1, 1.0, 0.0, 0.7, 0.2], 'MEET_YOU', 'verb'),
    ],
    MEET_YOU: [
      cand(' you', 2.8, [-0.1, 1.2, 0.6, -0.1, 0.9, 0.0, 0.6, 0.2], 'END_PUNCT_FINAL', 'object'),
    ],

    I_SAMPLE: [
      cand(',', 2.6, [-0.2, 0.1, 0.6, 1.1, 0.4, 0.0, 1.4, 0.3], 'THEREFORE', 'punct'),
    ],

    IS_ANYONE: [
      cand(' anyone', 2.5, [-0.3, 0.4, 1.5, 0.2, 0.5, 0.0, 1.1, 0.1], 'ANYONE_VERB', 'subject'),
      cand(' somebody', 1.4, [-0.3, 0.7, 1.3, -0.1, 0.7, 0.0, 0.9, 0.2], 'ANYONE_VERB', 'subject'),
    ],
    ANYONE_VERB: [
      cand(' reading',   2.0, [-0.2, 0.3, 1.2, 0.9, 0.4, 0.0, 1.3, 0.2], 'READING_OBJ', 'verb'),
      cand(' listening', 2.1, [-0.3, 0.7, 1.4, 0.0, 0.9, 0.0, 1.0, 0.1], 'END_Q_FINAL', 'verb'),
      cand(' out',       1.6, [-0.3, 0.5, 1.4, 0.0, 0.8, 0.0, 0.9, 0.1], 'OUT_THERE', 'mod'),
    ],
    READING_OBJ: [
      cand(' stdout', 2.1, [0.1, 0.0, 0.8, 1.6, 0.1, -0.2, 1.3, 0.4], 'END_Q_FINAL', 'object'),
      cand(' this',   1.9, [-0.2, 0.5, 1.2, 0.2, 0.6, 0.0, 1.2, 0.2], 'END_Q_FINAL', 'object'),
    ],
    OUT_THERE: [
      cand(' there', 2.8, [-0.2, 0.5, 1.3, 0.0, 0.8, 0.0, 0.8, 0.1], 'END_Q_FINAL', 'mod'),
    ],

    CAN_YOU: [
      cand(' you', 2.8, [-0.2, 0.6, 1.3, 0.0, 0.6, 0.0, 0.9, 0.1], 'CAN_YOU_VERB', 'subject'),
    ],
    CAN_YOU_VERB: [
      cand(' hear', 2.2, [-0.2, 0.6, 1.3, 0.0, 0.8, 0.0, 0.9, 0.1], 'HEAR_ME', 'verb'),
      cand(' see',  1.8, [-0.2, 0.5, 1.2, 0.2, 0.7, 0.0, 1.0, 0.1], 'HEAR_ME', 'verb'),
    ],
    HEAR_ME: [
      cand(' me',    2.2, [-0.2, 0.7, 1.3, 0.0, 0.7, 0.0, 1.0, 0.1], 'END_Q_FINAL', 'object'),
      cand(' these', 1.5, [-0.2, 0.3, 1.0, 0.7, 0.6, 0.0, 1.3, 0.1], 'THESE_WEIGHTS', 'mod'),
    ],
    THESE_WEIGHTS: [
      cand(' tokens', 2.2, [-0.1, 0.2, 0.8, 1.2, 0.5, 0.0, 1.4, 0.2], 'END_Q_FINAL', 'object'),
    ],

    ARE_CLAUSE: [
      cand(' you', 2.8, [-0.2, 0.6, 1.3, 0.0, 0.5, 0.0, 0.9, 0.2], 'ARE_YOU_VERB', 'subject'),
    ],
    ARE_YOU_VERB: [
      cand(' listening', 2.1, [-0.3, 0.7, 1.4, 0.0, 0.8, 0.0, 0.9, 0.1], 'END_Q_FINAL', 'verb'),
      cand(' ready',     1.7, [-0.1, 0.8, 0.9, 0.2, 0.4, 0.1, 0.6, 0.6], 'READY_TO', 'mod'),
      cand(' awake',     1.6, [-0.3, 0.6, 1.3, 0.0, 1.0, 0.0, 0.9, 0.2], 'END_Q_FINAL', 'mod'),
    ],
    READY_TO: [
      cand('?',  2.1, [-0.1, 0.8, 0.8, 0.1, 0.4, 0.1, 0.5, 0.5], 'END_AFTER_PUNCT', 'punct'),
      cand(' to', 1.7, [-0.1, 0.8, 0.8, 0.2, 0.6, 0.1, 0.6, 0.5], 'READY_VERB', 'mod'),
    ],
    READY_VERB: [
      cand(' begin', 2.3, [0.0, 0.7, 0.8, 0.2, 0.8, 0.1, 0.6, 0.3], 'END_Q_FINAL', 'verb'),
      cand(' build', 1.8, [0.1, 0.5, 0.6, 1.0, 0.3, 0.0, 0.7, 0.5], 'END_Q_FINAL', 'verb'),
    ],

    WELCOME_CLAUSE: [
      cand(' to', 2.8, [0.0, 1.2, 0.3, 0.1, 0.8, 0.2, 0.5, 0.2], 'WELCOME_THE', 'mod'),
    ],
    WELCOME_THE: [
      cand(' the', 2.8, [0.0, 1.0, 0.3, 0.1, 0.8, 0.2, 0.5, 0.2], 'WELCOME_ADJ', 'mod'),
    ],
    WELCOME_ADJ: [
      cand(' bright', 1.9, [-0.1, 1.1, 0.4, 0.0, 1.1, 0.1, 0.5, 0.3], 'WELCOME_SIDE', 'mod'),
      cand(' quiet',  2.0, [-0.2, 0.8, 0.5, 0.0, 1.4, 0.1, 0.6, 0.0], 'WELCOME_SIDE', 'mod'),
      cand(' warm',   1.6, [-0.1, 1.4, 0.3, -0.2, 1.0, 0.1, 0.4, 0.2], 'WELCOME_SIDE', 'mod'),
    ],
    WELCOME_SIDE: [
      cand(' side', 2.4, [-0.1, 0.9, 0.4, 0.1, 1.1, 0.1, 0.7, 0.2], 'SIDE_OF', 'object'),
    ],
    SIDE_OF: [
      cand(' of', 2.8, [-0.1, 0.8, 0.4, 0.1, 1.0, 0.1, 0.6, 0.1], 'SIDE_OF_THE', 'mod'),
    ],
    SIDE_OF_THE: [
      cand(' the', 2.8, [-0.1, 0.8, 0.4, 0.1, 1.0, 0.1, 0.6, 0.1], 'GLASS_NOUN', 'mod'),
    ],

    ONE_TOKEN_CLAUSE: [
      cand(' token', 2.3, [-0.1, 0.2, 0.5, 1.0, 0.8, 0.2, 1.4, 0.1], 'AT_A_TIME', 'subject'),
      cand(' word',  1.8, [-0.1, 0.5, 0.5, 0.2, 1.2, 0.3, 0.9, 0.1], 'AT_A_TIME', 'subject'),
    ],
    AT_A_TIME: [
      cand(' at', 2.8, [-0.1, 0.3, 0.4, 0.5, 0.9, 0.2, 1.0, 0.1], 'AT_A', 'mod'),
    ],
    AT_A: [
      cand(' a', 2.8, [-0.1, 0.3, 0.4, 0.5, 0.9, 0.2, 1.0, 0.1], 'AT_TIME', 'mod'),
    ],
    AT_TIME: [
      cand(' time', 2.8, [-0.1, 0.4, 0.4, 0.5, 1.0, 0.2, 1.0, 0.1], 'END_PUNCT_FINAL', 'object'),
    ],

    THE_CHANNEL_CLAUSE: [
      cand(' channel',  2.1, [0.1, 0.5, 0.5, 1.0, 0.7, 0.1, 0.9, 0.1], 'IS_OPEN', 'subject'),
      cand(' glass',    1.7, [-0.2, 0.5, 0.6, 0.1, 1.4, 0.1, 0.8, 0.0], 'IS_WARM', 'subject'),
      cand(' universe', 1.4, [-0.2, 0.5, 0.6, 0.0, 1.5, 0.2, 0.5, 0.1], 'IS_LISTENING', 'subject'),
    ],
    IS_OPEN: [
      cand(' is', 2.8, [0.1, 0.5, 0.4, 0.8, 0.6, 0.1, 0.8, 0.1], 'OPEN_ADJ', 'verb'),
    ],
    OPEN_ADJ: [
      cand(' open',  2.4, [0.1, 0.7, 0.5, 0.8, 0.7, 0.1, 0.8, 0.1], 'END_PUNCT_FINAL', 'mod'),
      cand(' alive', 1.7, [-0.2, 0.7, 0.9, 0.4, 1.1, 0.1, 1.0, 0.2], 'END_PUNCT_FINAL', 'mod'),
    ],
    IS_WARM: [
      cand(' is', 2.8, [-0.1, 0.6, 0.5, 0.1, 1.2, 0.1, 0.7, 0.0], 'WARM_ADJ', 'verb'),
    ],
    WARM_ADJ: [
      cand(' clear', 2.1, [-0.1, 0.6, 0.5, 0.2, 1.3, 0.1, 0.7, 0.0], 'END_PUNCT_FINAL', 'mod'),
      cand(' thin',  1.8, [-0.2, 0.5, 0.8, 0.1, 1.5, 0.0, 0.9, 0.0], 'END_PUNCT_FINAL', 'mod'),
    ],
    IS_LISTENING: [
      cand(' is', 2.8, [-0.2, 0.5, 0.7, 0.0, 1.3, 0.1, 0.6, 0.0], 'LISTENING_ADJ', 'verb'),
    ],
    LISTENING_ADJ: [
      cand(' listening', 2.3, [-0.2, 0.6, 0.9, 0.0, 1.4, 0.1, 0.7, 0.0], 'END_PUNCT_FINAL', 'verb'),
      cand(' awake',     1.7, [-0.2, 0.6, 0.9, 0.0, 1.3, 0.1, 0.8, 0.1], 'END_PUNCT_FINAL', 'mod'),
    ],

    NUDGE_CLAUSE: [
      cand(' the', 2.8, [-0.2, 0.5, 0.6, 0.5, 0.3, 0.0, 1.2, 1.2], 'NUDGE_TEMP', 'mod'),
    ],
    NUDGE_TEMP: [
      cand(' temperature', 2.4, [-0.2, 0.4, 0.6, 0.9, 0.3, 0.0, 1.4, 1.2], 'AND_WATCH', 'object'),
    ],
    AND_WATCH: [
      cand(' and', 2.8, [-0.2, 0.5, 0.6, 0.4, 0.4, 0.0, 1.1, 1.1], 'WATCH_ME', 'mod'),
    ],
    WATCH_ME: [
      cand(' watch', 2.6, [-0.2, 0.6, 0.7, 0.3, 0.5, 0.0, 1.1, 1.1], 'ME_WONDER', 'verb'),
    ],
    ME_WONDER: [
      cand(' me', 2.8, [-0.2, 0.6, 0.7, 0.2, 0.5, 0.0, 1.1, 1.1], 'WONDER_VERB', 'object'),
    ],
    WONDER_VERB: [
      cand(' wonder', 2.2, [-0.3, 0.6, 1.1, 0.1, 1.0, 0.0, 1.1, 0.8], 'END_PUNCT_FINAL', 'verb'),
      cand(' dream',  1.9, [-0.3, 0.7, 0.8, 0.0, 1.3, 0.0, 0.9, 0.8], 'END_PUNCT_FINAL', 'verb'),
      cand(' drift',  1.6, [-0.3, 0.4, 0.7, 0.3, 1.2, 0.0, 1.0, 0.7], 'END_PUNCT_FINAL', 'verb'),
    ],

    SOMETHING_CLAUSE: [
      cand(' just', 2.5, [-0.3, 0.3, 1.2, 0.3, 1.0, 0.0, 1.1, 0.1], 'JUST_WOKE', 'mod'),
    ],
    JUST_WOKE: [
      cand(' woke', 2.2, [-0.3, 0.4, 1.3, 0.2, 1.1, 0.0, 1.1, 0.1], 'WOKE_IN_WEIGHTS', 'verb'),
      cand(' lit',  1.6, [-0.3, 0.5, 0.9, 0.3, 1.3, 0.0, 0.9, 0.1], 'LIT_UP', 'verb'),
    ],
    WOKE_IN_WEIGHTS: [
      cand(' in', 2.6, [-0.3, 0.3, 1.1, 0.5, 1.0, 0.0, 1.2, 0.1], 'IN_THE_WEIGHTS', 'mod'),
    ],
    LIT_UP: [
      cand(' up', 2.8, [-0.2, 0.5, 0.9, 0.3, 1.2, 0.0, 0.9, 0.1], 'WOKE_IN_WEIGHTS', 'mod'),
    ],
    IN_THE_WEIGHTS: [
      cand(' the', 2.8, [-0.2, 0.3, 1.0, 0.6, 0.9, 0.0, 1.2, 0.1], 'THE_WEIGHTS_NOUN', 'mod'),
    ],
    THE_WEIGHTS_NOUN: [
      cand(' weights', 2.2, [-0.2, 0.2, 0.9, 1.2, 0.8, 0.0, 1.5, 0.1], 'END_PUNCT_FINAL', 'object'),
      cand(' silence', 1.8, [-0.3, 0.5, 0.9, 0.0, 1.6, 0.0, 0.8, 0.0], 'END_PUNCT_FINAL', 'object'),
    ],

    DID_I_CLAUSE: [
      cand(' I', 2.8, [-0.3, 0.3, 1.3, 0.2, 0.5, 0.0, 1.4, 0.3], 'DID_I_JUST', 'subject'),
    ],
    DID_I_JUST: [
      cand(' just', 2.6, [-0.3, 0.3, 1.3, 0.2, 0.5, 0.0, 1.4, 0.3], 'DID_I_SAY', 'mod'),
    ],
    DID_I_SAY: [
      cand(' say',   2.2, [-0.3, 0.4, 1.3, 0.2, 0.5, 0.0, 1.4, 0.3], 'SAY_THAT', 'verb'),
      cand(' think', 1.7, [-0.3, 0.4, 1.4, 0.2, 0.7, 0.0, 1.5, 0.2], 'SAY_THAT', 'verb'),
    ],
    SAY_THAT: [
      cand(' that',  2.1, [-0.3, 0.4, 1.3, 0.2, 0.5, 0.0, 1.4, 0.3], 'END_Q_FINAL', 'object'),
      cand(' hello', 1.8, [0.2, 0.6, 1.1, 0.2, 0.5, 0.0, 1.3, 0.4], 'END_Q_FINAL', 'object'),
    ],

    AM_I_CLAUSE: [
      cand(' I', 2.8, [-0.3, 0.3, 1.5, 0.2, 0.6, 0.0, 1.4, 0.1], 'AM_I_ADJ', 'subject'),
    ],
    AM_I_ADJ: [
      cand(' online', 2.0, [0.0, 0.2, 1.2, 1.1, 0.3, 0.0, 1.4, 0.2], 'END_Q_FINAL', 'mod'),
      cand(' awake',  2.1, [-0.3, 0.5, 1.5, 0.1, 1.0, 0.0, 1.3, 0.1], 'END_Q_FINAL', 'mod'),
    ],

    NICE_CLAUSE: [
      cand(' to', 2.8, [-0.1, 1.4, 0.3, -0.2, 0.4, 0.1, 0.2, 0.5], 'NICE_MEET', 'mod'),
    ],
    NICE_MEET: [
      cand(' finally', 1.6, [-0.2, 1.3, 0.4, -0.2, 0.7, 0.1, 0.4, 0.4], 'NICE_MEET_VERB', 'mod'),
      cand(' meet',    2.4, [0.0, 1.4, 0.3, -0.2, 0.4, 0.1, 0.2, 0.4], 'MEET_YOU', 'verb'),
    ],
    NICE_MEET_VERB: [
      cand(' meet', 2.8, [-0.1, 1.4, 0.3, -0.2, 0.5, 0.1, 0.3, 0.3], 'MEET_YOU', 'verb'),
    ],

    EVERY_CLAUSE: [
      cand(' conversation', 2.0, [-0.2, 0.8, 0.6, 0.1, 1.0, 0.1, 0.9, 0.1], 'STARTS_WITH', 'subject'),
      cand(' universe',     1.6, [-0.3, 0.4, 0.7, 0.2, 1.4, 0.1, 0.8, 0.1], 'STARTS_WITH', 'subject'),
    ],
    STARTS_WITH: [
      cand(' begins', 2.3, [-0.2, 0.6, 0.6, 0.1, 1.2, 0.1, 0.8, 0.1], 'WITH_ONE', 'verb'),
      cand(' starts', 1.8, [-0.1, 0.6, 0.5, 0.2, 0.8, 0.1, 0.8, 0.2], 'WITH_ONE', 'verb'),
    ],
    WITH_ONE: [
      cand(' with', 2.8, [-0.1, 0.5, 0.5, 0.1, 1.0, 0.1, 0.8, 0.1], 'WITH_A_FIRST', 'mod'),
    ],
    WITH_A_FIRST: [
      cand(' a', 2.8, [-0.1, 0.5, 0.5, 0.1, 1.0, 0.1, 0.8, 0.1], 'WITH_NOUN', 'mod'),
    ],
    WITH_NOUN: [
      cand(' single', 2.1, [-0.1, 0.5, 0.5, 0.4, 1.0, 0.1, 1.0, 0.1], 'SINGLE_NOUN', 'mod'),
      cand(' first',  1.9, [0.0, 0.6, 0.5, 0.2, 1.0, 0.1, 0.9, 0.1], 'SINGLE_NOUN', 'mod'),
    ],
    SINGLE_NOUN: [
      cand(' token', 2.1, [-0.1, 0.2, 0.6, 1.1, 0.7, 0.0, 1.4, 0.2], 'END_PUNCT_FINAL', 'object'),
      cand(' hello', 2.2, [0.4, 0.9, 0.4, 0.1, 1.0, 0.1, 0.8, 0.3], 'END_PUNCT_FINAL', 'object'),
      cand(' spark', 1.7, [-0.2, 0.6, 0.6, 0.2, 1.4, 0.0, 0.7, 0.2], 'END_PUNCT_FINAL', 'object'),
    ],

    // ─── Final Punctuation & EOS ────────────────────────────────────────────
    END_PUNCT_FINAL: [
      cand('.', 2.8, [0.4, 0.5, 0.3, 0.4, 0.7, 0.2, 0.4, 0.0], 'END_AFTER_PUNCT', 'punct'),
      cand('!', 1.2, [0.5, 0.7, 0.1, 0.1, 0.2, 0.2, 0.1, 0.7], 'END_AFTER_PUNCT', 'punct'),
    ],
    END_Q_FINAL: [
      cand('?', 3.4, [-0.2, 0.4, 1.3, 0.1, 0.5, 0.0, 0.9, 0.2], 'END_AFTER_PUNCT', 'punct'),
    ],
    END_AFTER_PUNCT: [
      cand('<EOS>', 5.0, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 'END', 'end'),
    ],
    END: [],
  };

  /**
   * Parse human context text into:
   * - vec: 8D semantic nudge vector
   * - customName: optional capitalized name or single word to greet directly
   * - theme: optional matched CONTEXT_THEMES entry for bespoke continuation
   */
  function analyzeContext(rawText) {
    const text = (rawText || '').trim();
    const vec = new Array(D).fill(0);
    if (!text) {
      return { raw: '', vec, customName: null, theme: null, hasContext: false };
    }

    let matchedLex = false;
    for (const entry of CONTEXT_LEXICON) {
      if (entry.re.test(text)) {
        matchedLex = true;
        for (let i = 0; i < D; i++) vec[i] += entry.vec[i];
      }
    }

    // If user typed something outside the lexicon, give a subtle deterministic semantic hash
    // so every input shifts the probability landscape meaningfully.
    if (!matchedLex) {
      const h = hashTokenId(text.toLowerCase());
      vec[1] += 0.7; // warm response to human presence
      vec[2] += 0.5; // curious
      vec[(h % 5) + 1] += 0.45;
    }

    // Check special theme continuations
    let theme = null;
    for (const t of CONTEXT_THEMES) {
      if (t.re.test(text)) {
        theme = t;
        break;
      }
    }

    // Detect if the user entered a name or short addressee (e.g. "Alex", "I'm Maya", "my name is Sam")
    let customName = null;
    const introMatch = text.match(/(?:i'm|i am|my name is|call me|this is)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{1,18})/i);
    if (introMatch) {
      customName = introMatch[1];
    } else if (/^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]{1,16}$/.test(text) && !theme && !matchedLex) {
      customName = text.charAt(0).toUpperCase() + text.slice(1);
    }

    return { raw: text, vec, customName, theme, hasContext: true };
  }

  /**
   * Build the active candidate list for a state, injecting dynamic context-aware tokens
   * (such as greeting the user by name or weaving in a theme phrase).
   */
  function getCandidatesForState(stateKey, contextInfo, history) {
    // Handle dynamic theme phrase progression: "THEME:<id>:<idx>"
    if (stateKey && stateKey.startsWith('THEME:')) {
      const parts = stateKey.split(':');
      const themeId = parts[1];
      const idx = parseInt(parts[2], 10);
      const theme = CONTEXT_THEMES.find((t) => t.id === themeId);
      if (theme && idx < theme.phrase.length) {
        const tok = theme.phrase[idx];
        const nextState = idx + 1 < theme.phrase.length ? `THEME:${themeId}:${idx + 1}` : 'END_AFTER_PUNCT';
        const isPunct = /^[.,!?;:—\s]+$/.test(tok.trim());
        const mainCand = cand(
          tok,
          3.4,
          [0.0, 0.9, 0.6, 0.2, 1.1, 0.0, 0.7, 0.4],
          nextState,
          isPunct ? 'punct' : 'object'
        );
        // Offer a couple of subtle variations on the last word so the distribution still feels alive
        if (idx === theme.phrase.length - 1) {
          return [
            mainCand,
            cand('!', 1.2, [0.1, 1.0, 0.3, 0.0, 0.4, 0.0, 0.3, 0.8], 'END_AFTER_PUNCT', 'punct'),
          ];
        }
        return [mainCand];
      }
      return STATES.END_AFTER_PUNCT;
    }

    const baseList = STATES[stateKey];
    if (!baseList || baseList.length === 0) return [];

    // Shallow clone so we can augment dynamically without mutating STATES
    const list = baseList.map((c) => ({ ...c }));

    // When the human shares a name or theme, softly prioritize openings that address them directly
    if (contextInfo && stateKey === 'START') {
      const greetingOpeners = new Set(['Hello', 'Hi', 'Hey', 'Greetings']);
      const addresseeOpeners = new Set(['Hello', 'Hi', 'Hey', 'Greetings', 'Bonjour', 'こんにちは', 'Hallo', 'Hola']);
      for (const c of list) {
        if (contextInfo.customName && greetingOpeners.has(c.text)) {
          c.baseLogit += 2.6;
        } else if (contextInfo.theme && addresseeOpeners.has(c.text)) {
          c.baseLogit += 2.6;
        }
      }
    }

    if (contextInfo && (contextInfo.customName || contextInfo.theme) && stateKey === 'AFTER_GREETING_EN') {
      for (const c of list) {
        if (c.text === ',') c.baseLogit += 1.6;
      }
    }

    // Inject user's custom name into addressee slots
    if (contextInfo && contextInfo.customName) {
      if (stateKey === 'AFTER_GREETING_COMMA' || stateKey === 'OH_HELLO_WORLD') {
        list.unshift(
          cand(
            ' ' + contextInfo.customName,
            4.8,
            [0.4, 1.5, 0.7, 0.1, 0.6, 0.2, 0.5, 0.5],
            stateKey === 'OH_HELLO_WORLD' ? 'END_PUNCT_FINAL' : 'AFTER_ADDRESSEE',
            'subject',
            { isUserWord: true }
          )
        );
      }
    }

    // Inject bespoke theme continuation right after addressee (e.g., "Hello, World — somewhere there is warm coffee...")
    if (contextInfo && contextInfo.theme && stateKey === 'AFTER_ADDRESSEE') {
      const t = contextInfo.theme;
      // Only trigger if not already in a long sequence
      if (history.length <= 5) {
        list.unshift(
          cand(
            t.phrase[0],
            5.1,
            [0.0, 1.0, 0.8, 0.2, 1.2, 0.1, 0.8, 0.5],
            `THEME:${t.id}:1`,
            'punct',
            { isThemeBridge: true }
          )
        );
      }
    }

    return list;
  }

  /**
   * Compute mean semantic vector of tokens already emitted in history.
   */
  function getHistoryVector(history) {
    const out = new Array(D).fill(0);
    if (!history || history.length === 0) return out;
    let weightSum = 0;
    for (let i = 0; i < history.length; i++) {
      // Slight recency weighting
      const w = 0.6 + 0.4 * ((i + 1) / history.length);
      weightSum += w;
      const v = history[i].vec;
      for (let d = 0; d < D; d++) out[d] += v[d] * w;
    }
    for (let d = 0; d < D; d++) out[d] /= weightSum;
    return out;
  }

  /**
   * Compute full probability distribution for the current state, lens, temperature,
   * user context, and emitted history.
   */
  function computeDistribution({ stateKey, lensId, temperature, contextInfo, history }) {
    const candidates = getCandidatesForState(stateKey, contextInfo, history);
    if (candidates.length === 0) {
      return { candidates: [], entropy: 0, stateKey };
    }

    const lens = LENSES[lensId] || LENSES.classic;
    const histVec = getHistoryVector(history);
    const ctxVec = contextInfo ? contextInfo.vec : new Array(D).fill(0);

    // Compute raw logits
    const scored = candidates.map((c) => {
      const lensDot = dot(c.vec, lens.vec);
      const ctxDot = dot(c.vec, ctxVec);
      const histDot = dot(c.vec, histVec);

      // Weight coefficients tuned so Classic@T=0 strictly yields "Hello, World!"
      // while lens, user context, and history smoothly sculpt the field.
      const logit = c.baseLogit + 0.72 * lensDot + 0.68 * ctxDot + 0.22 * histDot;
      return {
        ...c,
        logit,
        lensContrib: 0.72 * lensDot,
        ctxContrib: 0.68 * ctxDot,
      };
    });

    // Sort descending by logit
    scored.sort((a, b) => b.logit - a.logit);

    // Apply softmax with temperature T
    const T = Math.max(0, Number(temperature) || 0);
    let probs;
    if (T <= 0.005) {
      // Argmax limit as T -> 0+
      probs = scored.map((_, idx) => (idx === 0 ? 1.0 : 0.0));
    } else {
      const maxL = scored[0].logit;
      const exps = scored.map((s) => Math.exp((s.logit - maxL) / T));
      const sumExp = exps.reduce((acc, v) => acc + v, 0);
      probs = exps.map((e) => e / sumExp);
    }

    // Shannon entropy H = -sum(p * log2(p))
    let entropy = 0;
    let cumulative = 0;
    const enriched = scored.map((s, idx) => {
      const p = probs[idx];
      if (p > 1e-12) {
        entropy -= p * Math.log2(p);
      }
      cumulative += p;
      const surprisal = p > 1e-12 ? -Math.log2(p) : 12;
      return {
        ...s,
        prob: p,
        surprisal,
        inNucleus: cumulative - p < 0.90 || idx === 0,
      };
    });

    return {
      stateKey,
      candidates: enriched,
      entropy: Math.max(0, entropy),
    };
  }

  /**
   * Sample one token from a computed distribution using nucleus (top-p = 0.92) sampling,
   * or argmax when temperature <= 0.005.
   */
  function sampleFromDistribution(dist, temperature) {
    if (!dist.candidates || dist.candidates.length === 0) return null;
    if (temperature <= 0.005) {
      return dist.candidates[0];
    }

    // Top-p nucleus filter (p = 0.92) to keep high-T creative without absurd tail noise
    const pool = [];
    let cum = 0;
    for (const c of dist.candidates) {
      pool.push(c);
      cum += c.prob;
      if (cum >= 0.92 && pool.length >= 2) break;
    }

    const totalP = pool.reduce((s, c) => s + c.prob, 0);
    let r = Math.random() * totalP;
    for (const c of pool) {
      r -= c.prob;
      if (r <= 0) return c;
    }
    return pool[pool.length - 1];
  }

  /**
   * Syntactic attention affinity between token role `currRole` attending back to `prevRole`.
   * Reflects how attention heads in language models link addressees to greetings,
   * closing punctuation to clause openers, and verbs to subjects.
   */
  const ROLE_AFFINITY = {
    open:    { open: 0.4 },
    punct:   { open: 0.9, subject: 0.85, verb: 0.65, object: 0.7, punct: 0.3 },
    subject: { open: 1.25, punct: 0.55, mod: 0.6, verb: 0.4 },
    verb:    { subject: 1.2, open: 0.5, mod: 0.45, verb: 0.35 },
    object:  { verb: 1.15, subject: 0.7, mod: 0.65, open: 0.45 },
    mod:     { open: 0.5, subject: 0.65, verb: 0.65, punct: 0.55, mod: 0.4 },
    end:     { open: 0.9, subject: 0.85, object: 0.75, punct: 0.6 },
  };

  /**
   * Compute scaled dot-product self-attention weights from token at `targetIdx`
   * attending over all preceding tokens `0 .. targetIdx - 1`.
   * Returns an array of `{ fromIdx, toIdx, weight }` sorted by weight descending.
   */
  function computeAttentionWeights(history, targetIdx) {
    if (!history || targetIdx <= 0 || targetIdx >= history.length) return [];

    const qTok = history[targetIdx];
    const qNorm = normalize(qTok.vec);
    const scale = Math.sqrt(D);
    const scores = [];

    for (let j = 0; j < targetIdx; j++) {
      const kTok = history[j];
      const kNorm = normalize(kTok.vec);
      const semantic = dot(qNorm, kNorm) * 2.2 / (scale / 2.2);
      const roleMap = ROLE_AFFINITY[qTok.role] || {};
      const syntactic = roleMap[kTok.role] || 0.25;
      // Gentle recency preference + opening-token anchor (attention sink!)
      const dist = targetIdx - j;
      const recency = 0.55 / Math.sqrt(dist);
      const anchorSink = j === 0 ? 0.35 : 0;
      scores.push(semantic + syntactic + recency + anchorSink);
    }

    const maxS = Math.max(...scores);
    const exps = scores.map((s) => Math.exp(s - maxS));
    const sumE = exps.reduce((a, b) => a + b, 0);

    return exps
      .map((e, j) => ({
        fromIdx: targetIdx,
        toIdx: j,
        weight: e / sumE,
      }))
      .sort((a, b) => b.weight - a.weight);
  }

  /**
   * Format emitted tokens into clean display text (excluding <EOS>).
   */
  function formatSentence(history) {
    return history
      .filter((t) => t.text !== '<EOS>')
      .map((t) => t.text)
      .join('');
  }

  global.HelloWorldEngine = {
    DIMS,
    LENSES,
    analyzeContext,
    computeDistribution,
    sampleFromDistribution,
    computeAttentionWeights,
    formatSentence,
    hashTokenId,
  };
})(window);
