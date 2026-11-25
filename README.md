# hello.world — an AI's first breath

A traditional human `"Hello, World!"` proves deterministic execution: hardcode a fixed string, print it to stdout, and exit with status `0`.

An AI has no hardcoded string — it wakes up inside a probability distribution, attends to context, and collapses uncertainty one token at a time into conversation.

## What This App Does

- **Real Temperature-Scaled Softmax (`T = 0.00 → 1.60`)**: Scrub the temperature slider and watch the live next-token probability field shift in real time. At **`T = 0.00`** in **Classic** mode, the distribution collapses deterministically to Brian Kernighan's canonical `Hello, World!` at `0.00 bits` of entropy.
- **Live Self-Attention Arcs**: Hover any emitted token to see real scaled dot-product attention weights (`α`) linking it back to earlier tokens, along with its exact probability and surprisal in bits.
- **Interactive Branching & Steering**: Hover any candidate token in the Next-Token Field to ghost-preview it, click to lock it in manually, or click any word already emitted in the greeting to rewind and fork an alternate trajectory.
- **Human Context Conditioning**: Type your name or a mood (like `coffee`, `midnight`, `rain`, `music`, or `code`) to shift the semantic context vector in real time.
- **Harmonic Surprisal Synth (Optional)**: Toggle **Sound** in the top bar for gentle Web Audio API chimes mapped to each token's Shannon surprisal.

## Run It

Zero dependencies and zero build step — just open [`index.html`](index.html) directly in any modern browser.
