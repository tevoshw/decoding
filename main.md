# LLM Decoding

How an LLM turns a prompt into text: from the first input token to the last generated one.

---

## 1. Decoding

**Decoding** is the entire process of taking an input and producing an output with an LLM. It happens in two stages: **prefill** and the **decoding loop**.

```
Input prompt ──► Prefill ──► Decoding Loop ──► Output text
```

## 2. Prefill

The first stage. The model receives the user's input and processes **all prompt tokens in parallel**.

- Builds the internal state (the **KV cache**) that later steps reuse
- Produces the logits for the **first** generated token
- Compute-bound: this is what determines *time to first token*

## 3. Decoding Loop

The second stage. The model generates **one token at a time** (autoregressive), feeding each new token back as input.

```
repeat:
    logits  = model(tokens)
    next    = choose(logits)        # deterministic or sampling
    tokens += [next]
until next == EOS or len(tokens) == max_tokens
```

**Stopping conditions:**

- The model emits the **EOS token** (end of sequence)
- The **maximum token limit** is reached

## 4. Deterministic vs. Sampling

How the next token is chosen from the logits.

| | Deterministic | Sampling |
|---|---|---|
| **How** | Always picks the highest-probability token (greedy / argmax) | Draws a token at random, weighted by probability |
| **Same input → same output?** | Yes | No |
| **Use when** | There is **one correct answer** | **More than one answer** is valid |
| **Typical tasks** | Classification, tools & agents, evaluation & benchmarks, code | Assistants, creative writing |

---

## 5. Sampling Strategies

Sampling is controlled by a few knobs that reshape the probability distribution before the draw.

### Temperature

Rescales logits before softmax: `softmax(logits / T)`.

| Value | Effect | Result |
|---|---|---|
| `T < 1` | **Widens** the gap between logits (sharper distribution) | More focused, predictable |
| `T = 1` | Keeps the original gap | Model's native distribution |
| `T > 1` | **Narrows** the gap between logits (flatter distribution) | More diverse, riskier |

> As `T → 0`, sampling approaches deterministic (greedy) decoding.

### Top-K

Keeps only the **K highest logits**, discards the rest, then applies softmax.

- Simple and fast
- Fixed cutoff regardless of how confident the model is

### Top-P (Nucleus Sampling)

Keeps the **smallest set of top tokens whose cumulative probability reaches P**, then renormalizes.

- Adaptive: few candidates when the model is confident, many when it is uncertain
- Usually preferred over Top-K for that reason

### Multinomial

The actual **draw**. Think of a roulette wheel where each token's slice is proportional to its softmax probability.

- Applied after Temperature, Top-K, and Top-P have filtered and reshaped the distribution

### Repetition Penalty

Lowers the score of tokens that **already appeared** in the sequence, discouraging loops.

- Too low: repetition persists
- Too high: the model avoids necessary words and loses coherence

### Typical Pipeline

```
logits → repetition penalty → temperature → top-k → top-p → softmax → multinomial draw
```

---

## 6. Common Problems

### Repetition

The model gets stuck repeating words, phrases, or whole sentences with no new information.

- **Causes:** very low temperature, greedy decoding, weak context
- **Fixes:** repetition penalty, slightly higher temperature, Top-P

### Hallucination

The model produces content that sounds **fluent and plausible but is false or fabricated**, and states it with confidence.

- **Causes:** the model optimizes for likely text, not verified truth; gaps in training data
- **Fixes:** grounding with retrieval (RAG), tool use, lower temperature, asking for sources, and verifying critical outputs

---

## Quick Cheat Sheet

| Goal | Suggested setup |
|---|---|
| Code, extraction, classification | Deterministic or `T ≈ 0` |
| General assistant | `T ≈ 0.7`, Top-P ≈ 0.9 |
| Creative writing | `T ≈ 0.9–1.2`, Top-P ≈ 0.95 |
| Looping output | Add repetition penalty |