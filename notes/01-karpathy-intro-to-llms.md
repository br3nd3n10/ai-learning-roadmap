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
