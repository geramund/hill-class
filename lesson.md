# Next Token Prediction and Context Drift

## Overview

This lesson uses a live text generation tool to build intuition for how large language models produce text. The tool generates a story in real time while periodically substituting words into the context the model reads from — allowing you to observe how those substitutions propagate forward.

---

## Part 1: What Is a Token?

A language model does not process text as letters or words. It processes **tokens**.

A token is a subword unit produced by a **tokenizer** — a component that converts raw text into a sequence of integers before it reaches the model. The tokenizer uses a fixed vocabulary (typically 32,000 to 128,000 entries) built from the training data, where common words become single tokens and rare or long words are split into pieces.

Some examples:

| Text | Tokens |
|---|---|
| `cat` | `cat` |
| `unbelievable` | `un` · `believ` · `able` |
| `2024` | `20` · `24` |
| `Hello, world!` | `Hello` · `,` · ` world` · `!` |

A useful rule of thumb: **one token ≈ three-quarters of a word**. A 150-word passage is roughly 200 tokens.

### From text to numbers to vectors

When you submit a prompt, the tokenizer converts each token to an integer — its index in the vocabulary. The word *lighthouse* might map to the integer 17483. The phrase *had not slept* might produce the sequence [750, 451, 21839].

These integers are not what the model computes with directly. Each integer is used as a lookup index into an **embedding matrix**: a large table where each row is a dense vector of floating-point numbers, typically several thousand values long. The integer 17483 retrieves one specific row — a list of, say, 4096 numbers representing the model's learned encoding of that token.

A token, at the level of computation, is a position in a high-dimensional numerical space. The model processes a sequence of these vectors, updates them through many layers of attention and transformation, and ultimately produces a probability distribution over the vocabulary. Sampling from that distribution yields the next token ID, which is decoded back to text.

The context window — the maximum amount of text the model can attend to — is a limit on how many of these vectors the model processes at once, measured in tokens.

---

## Part 2: How Next Token Prediction Works

Every modern language model is trained to do one thing: **given a sequence of tokens, assign a probability to every possible next token**.

To do this well, the model has to encode information about grammar, syntax, facts, style, and discourse structure, because all of these affect what token is likely to follow a given sequence. This predictive objective, applied to enormous amounts of text, is how the model acquires its apparent knowledge of language.

The process of generating text looks like this:

1. The model receives a prompt: *"From swerve of shore to bend of bay"*
2. It produces a probability distribution over its entire vocabulary
3. It samples from that distribution, selecting the next token
4. That token is appended to the context
5. The model repeats from step 2, now attending to the extended sequence

This loop is called **autoregressive generation**. Each generated token becomes part of the input for all subsequent predictions. The model has no planning mechanism and no stored record of what it has generated — on each step, it sees only the current context window and produces the next token from that alone.

**Temperature** scales the probability distribution before sampling. At low temperature, the distribution is sharper and the model almost always selects the highest-probability token. At high temperature, the distribution is flatter and lower-probability tokens are sampled more often.

---

## Part 3: The Experiment

This tool intervenes in the autoregressive loop.

On each generation cycle:

1. The model generates a short chunk of text
2. One word in that chunk is **substituted** with a different word before being written into the context
3. On the next cycle, the model reads the modified context and continues from it

The model has no access to its prior outputs or any record of what was changed. It reads the context as given and produces the next token from it. Over many cycles, each substitution shifts the context slightly, and those shifts accumulate — the text drifts from what it would have been without intervention.

### What this demonstrates

- **Generation is stateless between tokens.** The model cannot detect that its context has been modified, because it holds no state from previous steps. There is only the current context window.
- **Context determines output.** The model's next token is a function of its input sequence. Alter the sequence, and the output changes accordingly.
- **Small changes compound.** A single substituted word influences the next prediction, which influences the one after it. The effect of each intervention propagates forward through subsequent generations.
- **Fluency is independent of fidelity to a prior direction.** The model will produce grammatical, locally coherent text from a modified context — it has no basis for recognising that the direction has shifted.

---

## Part 4: Experiments

**1. Baseline**
Generate with default settings. Read the output. At what point did the text begin to move away from the opening? Can you identify which substitution had the largest effect?

**2. Swap frequency**
Set "Words before each swap" to 2 (very frequent) and then to 20 (infrequent). How does intervention rate affect the coherence of the output? Is there a frequency at which the text becomes locally incoherent?

**3. Temperature**
Run the same prompt at temperature 0.2 and at temperature 1.3. How does the model's baseline variability interact with the substitutions? Does a higher-temperature model show more or less drift per substitution?

**4. Swap mode**
Compare Visual Similarity, Noun Swap, Adjective Swap, and Verb Swap.
- Which produces the most significant shift in the trajectory of the text?
- Nouns identify entities. Verbs specify relations and actions. Adjectives modify. Which class of substitution disrupts coherence most?

**5. Manual injection**
Use Pause / Inject to insert your own text mid-generation. Does the model continue smoothly from your phrase? After several more cycles, does it maintain the direction you introduced, or does it drift away from it?

**6. Repeated runs**
Generate from the same prompt twice with identical settings. Are the results the same? What determines the variation? What does this imply about the model's relationship to a given input?

---

## Discussion Questions

- The model cannot detect that its context has been modified. What does this imply about the kind of understanding, if any, the model has of what it is producing?
- Human writers can re-read, revise, and correct. What would the equivalent look like for an autoregressive model? What architectural feature would be required?
- The model produces fluent, grammatical text even from a substantially altered context. Is fluency evidence of coherence? Is coherence evidence of meaning?
- Some substitution modes disrupt the text more than others. What does this suggest about the distribution of semantic weight across grammatical categories?
- The model's output is a function of its context window alone. What are the implications of this for how we interpret what a language model "knows" or "intends"?
