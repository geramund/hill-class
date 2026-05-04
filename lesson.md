# Lesson: Next Token Prediction and the Drift Machine

## Overview

This lesson uses a live text generation tool to build intuition for how large language models actually work. The tool lets you watch a model write a story in real time — and secretly corrupt its own context as it goes.

---

## Part 1: What Is a Token?

A language model does not read text the way you do. It does not see letters, and it does not see words. It sees **tokens**.

A token is a chunk of text — usually a word, part of a word, or a punctuation mark. The model converts everything into a sequence of these chunks before processing it.

Some examples:

| Text | Tokens |
|---|---|
| `cat` | `cat` |
| `unbelievable` | `un` · `believ` · `able` |
| `2024` | `20` · `24` |
| `Hello, world!` | `Hello` · `,` · ` world` · `!` |

A useful rule of thumb: **one token ≈ three-quarters of a word**. A 150-word story is roughly 200 tokens.

Tokens matter because the model's context window — the maximum amount of text it can "see" at once — is measured in tokens, not words or characters.

---

## Part 2: How Next Token Prediction Works

Every modern language model is trained to do one thing: **given a sequence of tokens, predict what token comes next**.

This sounds simple. It is not.

To predict the next token well, the model has to learn an enormous amount about grammar, facts, style, logic, and world knowledge — because all of those things influence what word is likely to follow in a given context.

The process of generating text looks like this:

1. The model receives a prompt: *"The lighthouse keeper had not slept in"*
2. It produces a probability distribution over every token in its vocabulary
3. It samples from that distribution — picking a likely next token (e.g. *"three"*)
4. That token is appended to the context
5. The model repeats from step 2, now seeing *"...had not slept in three"*

This loop is called **autoregressive generation**. Each new token becomes part of the input for generating the next one. The model has no separate "planning" step and no memory of what it intended to write — it is always just answering the question *what comes next, given everything so far?*

**Temperature** controls how the model samples from that probability distribution. At low temperature, it almost always picks the highest-probability token (safe, predictable). At high temperature, it is more likely to pick surprising or lower-probability tokens (risky, creative).

---

## Part 3: The Experiment — Adversarial Autoregressive Feedback

This tool exploits the autoregressive loop directly.

Here is what it does on each cycle:

1. The model generates a short chunk of words
2. One word in that chunk is **silently replaced** with a different word before being added to the context
3. On the next cycle, the model continues from the modified context — as if it had written the replacement itself

The model has no way to detect the swap. It does not have access to its own prior "intentions." It only ever sees the current context window. So it picks up from the corrupted word and continues naturally from there.

Over many cycles, small corruptions accumulate. Meaning drifts. Stories that began as one thing become something else.

### What this demonstrates

- **The model has no plan.** It cannot notice that its story has been redirected, because it never had a destination. There is only the next token.
- **Context is everything.** The model's output is entirely determined by what it sees in the context window. Change the context, change the story.
- **Drift is compounding.** A single swapped word has a small effect. But that word becomes part of the context for the next prediction, which becomes part of the context for the one after that. Small errors accumulate.
- **The model is confident in the wrong direction.** It will continue coherently from a corrupted context, generating fluent, grammatical text — just no longer the text it "would have" written.

---

## Part 4: Experiments to Try

Work through these in order or pick the ones that interest you.

**1. Baseline**
Generate with default settings. Read the result. Where did meaning first start to shift? Can you identify the swap that caused the biggest turn?

**2. Swap frequency**
Move the "Words before each swap" slider to 2 (very frequent) and then to 20 (rare). How does the rate of injection affect coherence? At what frequency does the story become unreadable?

**3. Temperature**
At low temperature (0.2), the model is very conservative. At high temperature (1.3), it is erratic. How does temperature interact with the swaps? Does a high-temperature model drift faster or slower?

**4. Swap mode**
Try switching from Visual Similarity to Noun Swap, Adjective Swap, and Verb Swap.
- Which swap mode produces the most dramatic drift?
- Nouns carry the "who and what." Verbs carry the "action." Adjectives carry the "quality." Which matters most to the coherence of a story?

**5. Injection**
Use Pause / Inject to manually insert a phrase mid-generation. Does the model integrate your phrase smoothly? Does it "remember" what you wrote, or does it drift away from it? What does this tell you about context and attention?

**6. The same prompt, twice**
Run the same prompt twice with identical settings. Are the results the same? Why not? What does this say about how the model generates text?

---

## Discussion Questions

- If the model cannot detect that its context has been corrupted, what does that imply about what the model "knows" while writing?
- When a human writer loses the thread of what they were saying, they can re-read and correct. What would an equivalent mechanism look like for a language model?
- The model produces fluent, confident text even when the context has been significantly corrupted. Is fluency the same as coherence? Is coherence the same as meaning?
- Some swap modes disrupt meaning more than others. What does that tell us about which parts of a sentence carry the most semantic weight?
