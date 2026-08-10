# Karpathy — [1hr Talk] Intro to Large Language Models

Video: https://www.youtube.com/watch?v=zjkBMFhNj_g

## What an LLM literally is
- An LLM is **just two files**: a parameters file (the weights — e.g. Llama-2-70B is 70 billion parameters, ~140 GB at 2 bytes each) and a small run file (~500 lines of C) that executes the network. Fully self-contained — you can run it on a laptop with no internet.
- Open-weights models (Llama) let you hold both files; closed models (ChatGPT) only expose an API.

## Training = compressing the internet
- Pretraining: take a huge chunk of internet text (~10 TB), run a GPU cluster for weeks at a cost of millions of dollars, and "compress" it into the parameters. It's **lossy compression** — the model keeps the gestalt, not exact copies.
- The task is deceptively simple: **predict the next word**. Getting good at next-word prediction forces the network to learn a lot about the world (facts, grammar, reasoning patterns), because knowledge helps prediction.
- **Prediction is deeply related to compression** — this is why such a simple objective produces so much capability.

## How it runs, and what we don't know
- Generation: sample the next word, feed it back in, repeat. The output is the model **"dreaming" internet text** — often correct, sometimes fabricated (hallucination). The form is right even when the facts aren't.
- The transformer architecture is fully known — we understand the math at every step. What we **don't** understand well is what the billions of parameters actually do or where knowledge lives. Interpretability is a young field. Treat LLMs as **empirical artifacts, not engineered ones**: we measure their behavior more than we design it.
- Knowledge in the network is strange/one-dimensional — e.g. a model can answer "Who is Tom Cruise's mother?" but fail the reverse question.

## From base model to assistant (finetuning)
- Stage 1 (pretraining, ~yearly): produces a **base model** — an internet-document dreamer, not helpful by itself.
- Stage 2 (finetuning, can be weekly): swap the dataset for ~100k high-quality human-written Q&A conversations (label instructions matter more than volume: **quality over quantity**). Result: an **assistant model** that answers questions in a helpful format.
- Stage 3 (optional, RLHF): humans pick the best of several model answers — **comparisons are easier for labelers than writing answers from scratch**. Increasingly, labeling is a human–machine collaboration.
- Leaderboards (e.g. Chatbot Arena, ELO ratings): closed models lead, open models chase.

## Scaling laws
- Performance on next-word prediction is a **smooth, predictable function of just two numbers: N (parameters) and D (training data)**. Bigger model + more data = reliably better, with no ceiling in sight yet.
- This is why the industry is in a compute gold rush: you can forecast returns on buying more GPUs.

## LLMs becoming more than text predictors
- **Tool use**: modern assistants call a browser, calculator, code interpreter, and image generators — like a person, they reach for tools rather than doing everything in their head.
- **Multimodality**: seeing images, generating images, speech in/out.
- **System 1 vs System 2** (future direction): today's LLMs only do fast, instinctive System 1 next-word generation. The goal is System 2 — trading time for accuracy, "thinking" before answering.
- **Self-improvement** (future direction): AlphaGo surpassed humans via self-play, not imitation. Open question: what's the equivalent for LLMs, where there's no easy reward signal in open language?
- **Customization**: GPT store, custom instructions, RAG — models specializing per task.

## LLM as an operating system
- Karpathy's framing: think of the LLM as the **kernel of a new OS** — context window ≈ RAM, tools/browsing ≈ peripherals, other models ≈ processes. Closed models (GPT, Claude) parallel Windows/macOS; open models (Llama) parallel Linux.

## Security (new attack surface)
- **Jailbreaks**: roleplay or encodings (e.g. base64) that route around refusal training.
- **Prompt injection**: hidden instructions in web pages/images/docs hijack the model when it reads them.
- **Data poisoning / backdoors**: trigger phrases planted in training data that flip model behavior.
- Cat-and-mouse dynamic, just like traditional security.

## Three lines to remember
1. An LLM = two files; the parameters are a lossy compression of the internet, trained by next-word prediction.
2. Base model → (finetune on Q&A) → assistant → (RLHF) → better assistant; scaling N and D predictably buys capability.
3. Trajectory: tool use, multimodality, System 2 thinking, self-improvement — an emerging "LLM OS" with its own new security problems.

---

## Questions I asked while watching (2026-08-07)

### Q1 — ~12 min mark: Karpathy says we don't know what the parameters are actually doing. Is that still true?

**Mostly still true, but meaningfully less true than in late 2023.** The field is called **mechanistic interpretability**, and it has made real progress:

- Still true: for any given output, nobody can point at the billions of parameters and give the exact mechanism. No complete "circuit diagram" of a frontier model exists.
- **Features, not neurons:** individual neurons are mostly meaningless because each participates in many unrelated concepts (*superposition*). **Sparse autoencoders** (dictionary learning, 2023–24) decompose activations into millions of interpretable "features" — Anthropic's "Scaling Monosemanticity" found features for the Golden Gate Bridge, code bugs, sycophancy, deception — and showed you can causally manipulate them ("Golden Gate Claude").
- **Circuits:** Anthropic's 2025 circuit-tracing work followed multi-step reasoning inside Claude (e.g. "capital of the state containing Dallas" activates Texas → state-capital lookup; the model picks a rhyme word before writing the line of a poem).
- Honest picture: we've gone from total black box to **early neuroscience** — specific concepts and short circuits are traceable and even editable, but coverage is tiny and the tools are lossy approximations. Karpathy's one-sentence framing holds; "not at all" has become "partially, in well-studied cases."
- Follow-ups if curious: Anthropic's "Golden Gate Claude" post, then "On the Biology of a Large Language Model."

### Q2 — Does the reversal curse still exist ("Tom Cruise's mother" works, the reverse doesn't)?

**The phenomenon is still real; the specific demo no longer works.** From Berglund et al. 2023: models trained on "A is B" don't automatically learn "B is A," because an autoregressive model stores knowledge in the direction the training text flows — a one-way lookup, not a symmetric fact database. Scaling doesn't fix it.

Why the famous example now works anyway:
1. **The example contaminated itself** — the paper and talk went viral, so post-2023 training data states the fact both ways.
2. **Reasoning sidesteps it** — the curse is about direct retrieval from weights; a model that thinks step-by-step can pull the forward fact into context and invert it there (in-context reversal was never broken).
3. **Search/retrieval tools** bypass it entirely.

Takeaway: weight-stored "knowledge" is **directional pattern-completion, not a symmetric database** — a fresh obscure one-directional fact would likely still reproduce the failure with reasoning and search off.

### Q3 — Do open-weight releases include the base (pretrained) model, or just the fine-tuned one?

**Most release both** (e.g. `Llama-3.1-70B` + `Llama-3.1-70B-Instruct`; also Qwen, Mistral, Gemma, DeepSeek). The base model is the valuable raw material others fine-tune for their own purposes.

- **Instruct-only exceptions:** OpenAI's gpt-oss (2025), Phi, Command — withheld for safety (a base model has no refusal behavior at all) and commercial reasons (post-training is the secret sauce).
- **"Open source" caveat:** almost all of this is really **open *weights*** — final parameters only, no training data/code. Truly reproducible releases are rare: AI2's **OLMo** and EleutherAI's **Pythia** (data, code, and intermediate checkpoints included).
- Cheap exercise for later in the roadmap: download a base Qwen/Llama and watch it complete documents instead of answering questions — makes the pretraining vs. finetuning distinction concrete.

### Q4 — What's the latest released base model? Do labs still do this?

**Yes — and the most notable recent release is specifically a base model:** **Inkling** (July 15, 2026, Thinking Machines Lab / Mira Murati). 975B-parameter MoE (41B active), 45T training tokens, Apache 2.0, on Hugging Face. Explicitly positioned not as an assistant but as raw material for others to fine-tune (flagship for their Tinker platform) — the "document completer" as the product.

- Pattern continues elsewhere: Qwen `-Base` checkpoints, Kimi K2-Base, Kimi K3 (July 2026, 2.8T params, billed as largest open release ever). Instruct-only holdouts remain the exception.
- Nuance: modern "base" checkpoints aren't always perfectly raw — some labs mix instruction-formatted data into late pretraining, so today's base models can be slightly more conversational than the pure document completers in the talk.
- Sources: [Artificial Analysis on Inkling](https://artificialanalysis.ai/articles/thinking-machines-has-released-inkling-the-new-leading-us-open-weights-model) · [AIwire](https://www.hpcwire.com/aiwire/2026/07/16/thinking-machines-launches-open-weight-inkling-foundation-model-for-fine-tuning/) · [Simon Willison](https://simonwillison.net/2026/Jul/16/inkling/) · [OpenRouter: Open Weight Models that Matter, June 2026](https://openrouter.ai/blog/insights/the-open-weight-models-that-matter-june-2026/)
