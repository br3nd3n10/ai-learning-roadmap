# The Full-Stack AI Learning Roadmap

**Who this is for:** You already build and ship apps that call AI APIs (vibe coding). This roadmap takes you down the entire stack — what models *are*, how they're *trained*, how they're *made fast*, how they're *served from data centers* — back up to building AI products with real rigor, and then across the wider AI landscape: vision, diffusion, audio, reinforcement learning, robotics, causal reasoning, and what's actually going on inside the models.

**Structure:**
- **Part 1 (Phases 0–7)** — the LLM stack, top to bottom. This is the core; do it roughly in order.
- **Part 2 (Tracks A–G)** — breadth: vision & VLMs, image/video generation, speech & audio, general RL, robotics & embodied AI, causal reasoning, alignment & interpretability. Modular — pick by interest once you have the core.

**Honest time budget:** At ~1–1.5 hr/day, 5–6 days/week, Part 1 is realistically **11–14 months**, not 34 calendar weeks. A "week" below is a *unit of work* (~6–9 hours) — some will take you two calendar weeks, and that's the plan working, not failing. Part 2 at the scheduled survey depth adds roughly **6–8 months** for all seven tracks; completing every course in them at mastery depth would be more like 10–14+ months (each track states both numbers). There is no deadline; there is only the next block.

**How items are tagged:**
- 🟢 **Core** — do carefully and completely. The spine.
- 🔵 **Survey** — skim/watch for the mental model; don't grind through every chapter.
- ⚪ **Reference** — bookmark; return when a gap actually bites.

**The golden rule:** Watch/read → then *build*. Every phase has a milestone project. Understanding you didn't type doesn't stick.

**Milestone rubric (lightweight, every project):** (1) it runs, reproducibly — someone else could clone and run it; (2) at least one sanity assertion in the code (e.g. "loss decreased," "output shape is right" — the habit matters more than the coverage); (3) a short write-up of what you built and learned. The two capstones (Phase 4 and Week 33) get the heavier rubric noted there.

All links live in [RESOURCES.md](RESOURCES.md) with format, time, and cost details.

---

## The Map

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 6 · AI PRODUCTS      apps, agents, RAG, evals    │  ← you are here (vibe coding)
├─────────────────────────────────────────────────────────┤
│  LAYER 5 · MODEL APIs       prompting, context, tools   │
├─────────────────────────────────────────────────────────┤
│  LAYER 4 · POST-TRAINING    SFT, RLHF, RLVR, reasoning  │
├─────────────────────────────────────────────────────────┤
│  LAYER 3 · PRETRAINING      transformers, MoE, DATA     │
├─────────────────────────────────────────────────────────┤
│  LAYER 2 · FRAMEWORKS       PyTorch, autograd, backprop │
├─────────────────────────────────────────────────────────┤
│  LAYER 1 · KERNELS          CUDA, Triton, flash attn    │
├─────────────────────────────────────────────────────────┤
│  LAYER 0 · HARDWARE         GPUs, NVLink, data centers  │
└─────────────────────────────────────────────────────────┘
  cross-cutting: PRODUCTION OPS — evals-in-CI · monitoring
                 · versioning · rollback  (Phase 4.5)
     Part 2 branches: vision · diffusion · audio · RL · robotics
                      · causal · alignment & interpretability
```

---

# PART 1 — THE LLM STACK (Phases 0–7)

## Phase 0 — The Grand Tour (Week 1)

*Goal: a correct high-level mental model of the entire stack before going deep on anything.*

| Day | Do this | Tag |
|-----|---------|-----|
| 1 | Karpathy — **[1hr Talk] Intro to Large Language Models** | 🟢 |
| 2 | 3Blue1Brown — **Neural Networks series**, ch. 1–4 (what a net is, gradient descent, backprop) | 🟢 |
| 3 | 3Blue1Brown — **"But what is a GPT?" + attention chapters** (5–7) | 🟢 |
| 4 | Karpathy — **Deep Dive into LLMs like ChatGPT** (first half) | 🟢 |
| 5 | Karpathy — **Deep Dive into LLMs** (second half) | 🟢 |
| 6 | **Bycroft's 3D LLM Visualization** + **Transformer Explainer** — poke at a real GPT-2 in your browser. Then a 20-minute build: call a model API from a script and print the raw token-by-token output with logprobs — look at what the machine actually emits. | 🟢 |

**Milestone:** You can explain to a friend what an LLM is, what "training" means, and what happens between prompt and response.

---

## Phase 1 — Foundations: Math + ML Basics (Weeks 2–4)

*Goal: just enough math to never be scared of it, plus classical ML vocabulary. This phase is deliberately intuition-first — depth arrives later, just-in-time, via the reference texts.*

**Week 2 — Math intuition**
- Days 1–3: 3Blue1Brown — **Essence of Linear Algebra** 🟢 (vectors, matrices as transformations — this is what tensors and embeddings are)
- Days 4–5: 3Blue1Brown — **Essence of Calculus** 🟢 (derivatives + chain rule — this IS backprop)
- Day 6: StatQuest — probability basics, plus one short read on **entropy & cross-entropy** 🟢 (cross-entropy is literally the loss function you'll stare at from Phase 2 onward)
- ⚪ Reference for any gap, forever: **Mathematics for Machine Learning** (free book)

**Week 3 — Classical ML, hands-on first**
- Days 1–2: Kaggle Learn — **Intro to Machine Learning** 🟢 (you train models on day one)
- Days 3–6: Structured intro ML course 🟢 — pick ONE path:
  - **Note:** Coursera killed free auditing in Aug 2025 — Ng's ML Specialization now only previews the first module free. Your options: (a) pay ~$49 for one focused month, (b) apply for Coursera financial aid (~15 days), (c) free: **DeepLearning.AI's official YouTube playlist** (partial, Course 1 material) + StatQuest to fill gaps, or (d) free but math-heavier: Ng's **Stanford CS229** lectures.
  - Whichever path: cover linear/logistic regression, gradient descent, train/test splits, overfitting, regularization.

**Week 4 — Neural nets + PyTorch hello world**
- Days 1–2: Finish your intro-ML path's neural net material 🟢; StatQuest for anything that doesn't click
- Days 3–4: **UvA "Introduction to PyTorch" notebook** 🟢 — tensors, autograd, a full training loop
- Day 5: PyTorch official **"Learn the Basics"** 🟢
- Day 6: **Milestone project:** train a small classifier in PyTorch end-to-end from a blank file. Rubric: reproducible + short write-up.

> **Skip-ahead rule:** If this phase feels slow, compress it. If math feels shaky later, come back — that's what ⚪ references are for.

---

## Phase 2 — Deep Learning From Scratch (Weeks 5–8)

*Goal: no black boxes. You build backprop, then language models, then a working GPT — by hand. The heart of the roadmap.*

This phase = **Karpathy's "Neural Networks: Zero to Hero"** 🟢, done properly (code along, don't just watch).

**Week 5 — Backprop from scratch**
- Days 1–3: **micrograd** video 🟢 — build an autograd engine in ~100 lines of Python
- Days 4–6: Rebuild micrograd from memory 🟢. CS231n notes on optimization + backprop 🔵 to cement it.

**Week 6 — Language models begin**
- Days 1–3: **makemore part 1** 🟢 (bigrams → neural net language model)
- Days 4–6: **makemore parts 2–3** 🟢 (MLPs, activations, batch norm)

**Week 7 — Fluency**
- Days 1–3: **makemore parts 4–5** 🟢 (manual backprop; WaveNet-style architecture)
- Days 4–6: Daniel Bourke's **learnpytorch.io** 🔵 (workflow + classification modules — translate scratch-knowledge into framework fluency)

**Week 8 — Build GPT**
- Days 1–3: **"Let's build GPT: from scratch, in code, spelled out"** 🟢
- Days 4–5: **"Let's build the GPT Tokenizer"** 🟢 (why LLMs can't spell)
- Day 6: **The Illustrated Transformer** 🟢 — it will now feel *obvious*. That feeling is the milestone.

**Milestone project:** Your own trained mini-GPT generating text from a dataset you chose. Rubric: reproducible repo + write-up. ⚠️ *If you train on personal data (your messages, emails): small models memorize their training data — keep those weights private, don't push them to a public repo.*

> **Parallel/alternative tracks:** **fast.ai Practical Deep Learning** 🔵 for breadth on weekends; **Understanding Deep Learning** (Prince) ⚪ as your reference text from here on.

---

## Phase 3 — How Real LLMs Are Made: Architecture, Data, Evals, Post-Training (Weeks 9–15)

*Goal: from toy GPT to the actual modern pipeline — and the four disciplines that separate rigor from vibes: architecture literacy, data work, evaluation design, and RL. Note the order: eval design comes BEFORE fine-tuning, on purpose.*

**Week 9 — The real pretraining recipe**
- Days 1–4: Karpathy — **build-nanogpt** 🟢 (reproduce GPT-2 124M: mixed precision, LR schedules, real data loading)
- Days 5–6: EleutherAI — **Transformer Math 101** 🔵 (the arithmetic of scale: params, FLOPs, memory)

**Week 10 — From GPT-2 to 2026: modern architecture + scaling laws**

> *Why this week exists: Phase 2 and Week 9 hand you a GPT-2 — a 2019 design. Every model you actually use has moved on, and the differences are not cosmetic. Skip this week and Phase 6's inference economics won't land, because you won't know what a KV cache is.*

- Days 1–2: Raschka — **The Big LLM Architecture Comparison** 🟢 (DeepSeek-V3 → Kimi K2 in one article: RoPE instead of learned positions, RMSNorm instead of LayerNorm, SwiGLU instead of GELU-MLP, GQA/MLA instead of plain multi-head attention, sliding-window and hybrid attention). Go to the **RoPE** and **GQA** papers ⚪ only if a specific detail nags.
- Days 3–4: **Mixture-of-Experts** 🟢 — HF's **MoE Explained** + Grootendorst's **Visual Guide to MoE**. This is how most frontier models are now built: total parameters ≫ active parameters. That one fact drives training cost, serving cost, and why "how big is the model?" stopped being a single number.
- Day 5: **The KV cache** 🟢 — what it holds, why it grows with context length × batch size, and why it (not the weights) usually decides how many users fit on one GPU. Compute it by hand for a model you use. This is the object Weeks 32–33 are really about.
- Day 6: **Scaling laws** 🟢 — Epoch AI's **scaling laws literature review**, then the shape of the argument: Kaplan (2020) → **Chinchilla** (2022, compute-optimal ≈ 20 tokens/param) → the modern correction that *inference* cost, not training cost, sets the real optimum, which is why small models are trained far past Chinchilla (Harm de Vries — *Go smol or go home*). This is the bridge from Layer 3 down to Layer 0: it's why data centers are the size they are.

**Milestone (small, do it the same day):** write ~200 words answering *"how does the model I use daily differ from the GPT-2 I built, and why does each difference exist?"* If you can't, re-read before moving on. Architecture literacy is what makes every later phase readable.

**Week 11 — Data: the underrated layer**
- Days 1–2: **FineWeb technical report** (Hugging Face) 🟢 — how a real pretraining dataset is actually built: filtering, dedup, quality classifiers, and how each choice was validated with ablations
- Day 3: Hugging Face **LLM Course** data-processing chapters 🔵 (the `datasets` library in practice)
- Days 4–5: Hands-on 🟢: take a messy raw text corpus and build a cleaning pipeline — dedup, filter, split train/val/test properly. (If you scrape it yourself, check the site's terms and robots.txt first — data provenance starts at acquisition.) Understand **leakage** and **benchmark contamination** (Evaluation Guidebook's contamination section ⚪) — why test data hiding in training data quietly invalidates results.
- Day 6: Read on dataset licensing & provenance basics 🔵 — what you can legally train on, why PII in training data matters. Then **synthetic data** 🔵: increasingly the majority of post-training data, and the place where model collapse and quiet contamination both hide.

**Week 12 — Evaluation design (before you fine-tune anything)**
- Days 1–2: Hugging Face — **LLM Evaluation Guidebook** 🟢 (benchmarks, LLM-as-judge, contamination)
- Day 3: Hamel Husain — **Your AI Product Needs Evals** 🟢 (yes, from Phase 4 — the mindset belongs here)
- Days 4–6: Build the eval for next week's project 🟢, *before touching training*: a frozen held-out test set, a baseline measurement of the un-tuned model, and 3–5 failure categories you want to improve. Freeze it — no edits while the experiment runs. (New failure modes you discover later become a versioned eval-v2 for the *next* experiment; frozen-per-experiment, versioned-across-experiments is the real-world policy.)

**Week 13 — Fine-tuning in practice**
- Days 1–2: Hugging Face **LLM Course** fine-tuning chapters 🔵
- Days 3–6: **Unsloth fine-tuning guide** + free Colab 🟢: LoRA/QLoRA a current open model on your own dataset (cleaned in Week 11, eval'd against Week 12's frozen set)

**Milestone project (the full honest loop):** data pipeline → frozen eval → baseline → fine-tune → re-measure → write-up including what did NOT improve. This is the loop that most people fake; you're building the muscle to do it for real.

**Week 14 — Post-training I: SFT → preference alignment**
- Days 1–4: **smol-course** 🔵 — a ~20-hour, six-unit course (instruction tuning, evaluation, preference alignment, VLMs, RL, synthetic data; GPU recommended). Do the **instruction-tuning + preference-alignment** units now; treat the rest as ⚪ to return to.
- Days 5–6: Nathan Lambert — **RLHF Book** 🟢 (intro + reward models + DPO chapters). Know why DPO exists and where it stops being enough.

**Week 15 — Post-training II: reasoning + RL with verifiable rewards**

> *Why this got its own week: post-training in 2026 is where the differentiation happens, and RLVR is the fastest-moving layer in the entire stack. It's also the direct answer to "how did reasoning models get good," which one skimmed paper cannot give you.*

- Days 1–2: Raschka — **Understanding Reasoning LLMs** 🟢 + the **DeepSeek-R1 paper** 🔵. Precision matters here: **R1-Zero** demonstrated reasoning emerging from pure RL with verifiable rewards; the production **R1** used cold-start data and multi-stage training. Then **RLHF Book ch. 14 (Reasoning & RLVR)** 🟢 for the mechanics.
- Day 3: **GRPO** 🟢 — how it differs from PPO (no learned value model; advantage estimated from a group of sampled completions), and why that made RL post-training cheap enough to spread. TRL's **GRPO Trainer** docs are the concrete reference.
- Days 4–5: **Hands-on** 🟢 — run a small GRPO job on a *verifiable* task (math with a checkable answer, or code with a passing test) using **Unsloth's RL guide** free Colab. The point is not a good model; the point is watching a reward curve respond to a reward function you wrote.
- Day 6: **Environments and reward hacking** 🟢 — skim **open-r1** and **verifiers** to see how RL environments are actually structured, then think adversarially: for the reward you wrote on Day 4, what is the laziest policy that scores well without solving the task? Write it down. Reward hacking is the field's central practical problem and you should meet it with your hands on a real reward.
- ⚪ *If this hooks you:* Track D (Part 2) is the proper RL foundation underneath all of this, and it retroactively makes this week much deeper.

> **Go-deeper options:** Stanford **CS336** ⚪ (the rigorous university version of this whole phase); Raschka's **Build a Large Language Model (From Scratch)** ⚪ (~$40, the book version).

---

## Phase 4 — AI Engineering, Done Rigorously (Weeks 16–22)

*Goal: transform your vibe-coding into engineering: prompting as a discipline, agent-assisted development, RAG, agents, evals, and security.*

**Spine for the whole phase:** Chip Huyen — **AI Engineering** 🟢 (book ~$45, or her free blog + the free `aie-book` repo). A chapter or two per week alongside the work below.

**Week 16 — The right mental model**
- Days 1–2: **"What We Learned from a Year of Building with LLMs"** I–III 🟢
- Day 3: Anthropic — **Building Effective Agents** 🟢
- Day 4: Anthropic — **Effective Context Engineering for AI Agents** 🟢
- Day 5: Chip Huyen — **Agents** blog post 🟢
- Day 6: Eugene Yan — **Patterns for Building LLM-based Systems** 🔵

**Week 17 — Working with coding agents (the layer you actually live in)**

> *Why this week exists: you arrived here as a vibe coder, and this roadmap otherwise teaches you to* build *agents while never teaching you to* wield *them. Agent-assisted development is where you'll spend more hours than any other item in this document, and it is a learnable engineering discipline rather than a knack. Compress it to 2–3 days if you're already deliberate about it.*

- Days 1–2: Anthropic — **Claude Code Best Practices** 🟢 and Simon Willison — **Using LLMs for code** 🟢. Read them as engineering doctrine, not tool tips: context curation, tight feedback loops, planning before editing, and treating agent output as a junior engineer's PR rather than an oracle.
- Day 3: Anthropic — **Writing Tools for Agents** 🟢 — this connects directly to Week 20's MCP work. Tool design *is* prompt design.
- Day 4: **Context engineering for code** 🟢 — what your agent can and can't see, why large-repo performance degrades, and how instruction files, retrieval, and subagents change that. Re-read Week 16's context-engineering piece with a codebase in mind.
- Days 5–6: **Hands-on, on your own repo** 🟢: write a project instruction file, define one repeatable agent task, and run it three times. Record where it succeeded, where it silently went wrong, and what context would have prevented that. *How you'll know it worked:* you can state one concrete change to your setup that measurably reduced a failure mode — and evaluating agents on your own work is the same discipline as Week 21's evals, applied to yourself.

**Week 18 — Prompting as engineering**
- Days 1–4: **Anthropic's Interactive Prompt Engineering Tutorial** 🟢 (graded exercises)
- Days 5–6: Claude prompt-engineering docs 🔵 + OpenAI's current prompting guide 🔵 (compare provider doctrines)

**Week 19 — RAG properly**
- Days 1–4: DeepLearning.AI — **RAG course** 🟢 (embeddings, hybrid search, chunking, reranking)
- Days 5–6: DLAI — **Building and Evaluating Advanced RAG** 🔵
- *Framing note, and it matters:* much early RAG doctrine was shaped by 4k–8k context windows. With long-context models and agentic retrieval, aggressive chunking is often the wrong reflex. Learn the components — they're durable — but ask of each technique: *is this solving a problem I still have?*

**Week 20 — Agents + MCP**
- Days 1–2: Anthropic Academy — **Intro to Model Context Protocol** 🟢 (build an MCP server)
- Days 3–4: **LangGraph intro course** 🔵 — or build the same patterns raw
- Days 5–6: OpenAI — **A Practical Guide to Building Agents** 🔵

**Week 21 — Evals + observability (the differentiator)**
- Days 1–3: Husain & Shankar — **LLM Evals FAQ** 🟢 (free distillation of the course frontier-lab teams take)
- Days 4–5: DLAI + Arize — **Evaluating AI Agents** 🟢 (tracing + evals in code)
- Day 6: LangSmith evaluation concepts ⚪ (vocabulary transfers to any platform)

**Week 22 — Security (non-optional if you build agents)**
- Days 1–2: **OWASP Top 10 for LLM Applications** 🟢 — prompt injection, insecure output handling, excessive agency
- Days 3–4: Simon Willison's **prompt injection series** 🟢 — esp. the "lethal trifecta" (private data + untrusted content + external communication); why injection is unsolved and how to design around it
- Day 5: Tool permissioning & least privilege for agents 🟢 — audit one of your own agents: what can it *actually* do if its input is hostile? Include the coding agent from Week 17: it reads untrusted code and has a shell.
- Day 6: NIST Generative AI Profile 🔵 (skim — the governance vocabulary). Production reliability patterns are next: **Phase 4.5**.

**CAPSTONE (heavier rubric):** Take an AI app you've already built. Add: (1) an eval suite with a frozen test set and real failure cases, (2) tracing/observability, (3) an MCP server exposing one capability, (4) a security pass — prompt injection testing + least-privilege tool permissions, (5) cost + latency measurement. Publish a sanitized write-up (respecting any privacy, client, or security constraints in the app). This artifact is worth more than any certificate in this roadmap.

---

## Phase 4.5 — Production AI Systems (Weeks 23–25)

*Goal: the AI-specific operational layer that turns your apps from demos into products. This is core, not elective, for one reason: you ship apps to real users. (A pure researcher could skip it; you can't.)*

**Gate — answer honestly before starting:** can you containerize an API, run tests in CI, manage secrets, deploy a service, and authenticate requests? **All yes** → compress this phase to ~1 week and jump to the Week 25 project. **Mostly no** → take a short general software-production bridge first (a solid Docker + CI/CD tutorial for your stack; this roadmap deliberately doesn't teach backend engineering), then return.

**Week 23 — Reliability patterns for AI apps**
- Days 1–4: Huyen's **AI Engineering** serving/optimization chapters 🟢, re-read as an operator: streaming, retries/timeouts/idempotency, caching, routing and fallbacks, budget enforcement.
- Days 5–6: Provider-specific **prompt caching** 🟢 — precision matters here: on Anthropic, cache-*hit input* tokens cost ~90% less than normal input tokens, but cache *writes* cost more and output tokens are unaffected. Caching slashes the repeated-context slice of the bill, not the whole bill. Learn your provider's exact pricing model and measure your own app's cacheable fraction. (Week 10 told you *why* this works: you're renting someone else's KV cache.)

**Week 24 — Versioning, eval gates + monitoring**
- Days 1–3: Prompt/model/dataset/experiment versioning 🟢 — then wire your Phase 4 eval suite into CI as a gate: no prompt or model change ships without the evals passing.
- Days 4–6: Observability 🔵 — structured logs, traces, and metrics for LLM calls (OpenTelemetry as the durable vocabulary; LangSmith/Braintrust-style tools as implementations). Online quality monitoring and drift: when yesterday's eval stops representing today's traffic.

**Week 25 — The project (on YOUR existing deploy stack)**
- Take your Phase 4 capstone app, deployed however you already deploy (Vercel, Railway, a VPS — no new infrastructure stack required), and add: eval-gated deploys, per-request cost + latency logging, a cache layer, a simulated provider outage (revoke the API key — what does the user actually see?), and a documented, *tested* rollback path.
- *How you'll know it worked:* you can answer "what does a request cost, what's p95 latency, and what happens when the provider goes down?" with measurements and a demonstration, not guesses.

---

## Phase 5 — Down to the Metal: GPUs & Kernels (Weeks 26–29)

*Goal: understand why AI is fast or slow — from "GPUs go brrr" intuition to writing your own kernels.*

**Week 26 — GPU intuition (highest-leverage week in the roadmap)**
- Day 1: Horace He — **Making Deep Learning Go Brrrr** 🟢 (compute-bound vs memory-bound vs overhead)
- Days 2–3: Stephen Jones (NVIDIA) — **How GPU Computing Works** + **How CUDA Programming Works** 🟢
- Days 4–6: Modal — **GPU Glossary** 🔵 (read linearly once; ⚪ forever after)

**Week 27 — First CUDA kernels**
- Day 1: NVIDIA — **An Even Easier Introduction to CUDA** 🟢
- Days 2–6: **GPU MODE lectures** 1–5 🟢 (follows the PMPP textbook ⚪ ~$70 if you want the full spine)
- *Prereq honesty: CUDA is C++. If you've never touched C/C++, spend 2–3 days on basics first — enough to read pointer syntax. You don't need mastery.*

**Week 28 — Triton: kernels in Python**
- Days 1–4: **Official Triton tutorials** 🟢: vector add → fused softmax → matmul, benchmarked against PyTorch
- Days 5–6: **Triton Puzzles** 🔵 (deliberate practice)

**Week 29 — The famous kernels**
- Days 1–3: Simon Boehm — **How to Optimize a CUDA Matmul Kernel** 🟢 (naive → ~80% of cuBLAS)
- Days 4–5: Gordić — **ELI5: FlashAttention** 🟢 — and note what it really is: not a better approximation, but the same math made memory-aware. That distinction is the whole lesson of this phase.
- Day 6: **torch.compile tutorial** 🔵 (connects kernels back to everyday PyTorch)

**Milestone project:** A fused softmax (or matmul) in Triton that beats naive PyTorch, with a benchmark chart. Join the **GPU MODE Discord**.

> **You're on a Mac:** experiment locally with **MLX** (Apple's array framework for Apple silicon) for the concepts, but do the CUDA/Triton work on rented NVIDIA GPUs — Google Colab (free tier), Modal, or Lightning AI credits. CUDA is the industry language; MLX is your convenient local lab.

---

## Phase 6 — Scale: Training Clusters, Inference, Data Centers (Weeks 30–34)

**Week 30 — Distributed training I: the theory**
- Day 1: Lilian Weng — **How to Train Really Large Models on Many GPUs** 🔵 (pre-read)
- Days 2–5: Hugging Face — **The Ultra-Scale Playbook** 🟢 — but *survey-first*: read Parts 1–2 carefully (data/tensor/pipeline parallelism, ZeRO), skim the rest, return as ⚪ when you need it. It's a 10–20 hr book; don't grind it linearly.
- Day 6: Sumanth Hegde — **Distributed Training & Efficient Finetuning** 🔵 (what FSDP/DeepSpeed actually do)

**Week 31 — Distributed training II: what actually happens**

> *Why this week exists: Week 30 teaches you the parallelism strategies. It does not teach you that large runs fail constantly, that a node dies mid-epoch, that your throughput is 40% of theoretical and you have to find out why. That gap is the difference between reading about training and training.*

- Days 1–3: Stas Bekman — **ML Engineering** 🟢, selectively: the performance/throughput debugging, fault-tolerance, and training-instability chapters. This is the field manual — written from real large-run scar tissue, not from theory.
- Days 4–5: Hugging Face — **The Smol Training Playbook** 🟢: one team's honest end-to-end account of training a real model, *including the ablations that failed and the decisions they'd reverse*. Read it right after the theory, while you still believe training is tidy.
- Day 6: **Low precision and collectives** 🔵 — why FP8 (and increasingly FP4) training is now standard and what it costs in stability (PyTorch's float8 + FSDP2 post), plus the collective operations everything above is built from: all-reduce, all-gather, reduce-scatter (NCCL). Once you can name the collectives, "the interconnect is the bottleneck" in Week 34 becomes a concrete claim instead of a slogan.

**Week 32 — Inference: the economics of serving**
- Days 1–2: kipply — **Transformer Inference Arithmetic** 🟢 (predict latency + cost from first principles)
- Day 3: **Prefill vs decode** 🟢 — the single most useful distinction in serving: prefill is compute-bound and parallel, decode is memory-bandwidth-bound and sequential. Nearly every 2026 serving optimization — continuous batching, chunked prefill, prefill/decode *disaggregation* onto separate GPU pools, KV-cache-aware routing — is a consequence of those two workloads wanting opposite hardware.
- Day 4: Gordić — **Inside vLLM** 🟢 (paged attention, continuous batching, speculative decoding) + skim **SGLang** 🔵 for a second engine's take (RadixAttention / prefix-cache reuse). Two engines is what turns "how vLLM works" into "how serving works."
- Days 5–6: Grootendorst — **A Visual Guide to Quantization** 🟢 + BentoML **LLM Inference Handbook** ⚪

**Week 33 — Serve a model yourself (capstone #2, heavier rubric)**
- Days 1–4: Deploy an open model with **vLLM** on a rented GPU. Measure tokens/sec; try a quantized version; compute cost-per-million-tokens vs an API. Report throughput and latency **separately** at more than one concurrency level — a single number here is almost always a misleading one.
- Days 5–6: Read the original **vLLM PagedAttention** post 🔵 — it maps onto what you just ran, and onto the KV cache you computed by hand back in Week 10.
- Rubric: reproducible deploy script, benchmark numbers, cost analysis, and a write-up including failure modes you hit.

**Week 34 — The physical layer: AI data centers** 🔵 *(elective-but-fascinating tier — skim freely)*
- Days 1–2: Vikram Sekar — **Beginner's Guide to AI Datacenter Interconnects** (NVLink in the node; InfiniBand/RoCE across nodes)
- Days 3–4: SemiAnalysis — **Datacenter Anatomy Parts 1–2** (power + cooling; why 130kW liquid-cooled racks changed everything) + **100,000 H100 Clusters**
- Days 5–6: **The other architecture** 🟢 — Google's **How To Scale Your Model** (the JAX/TPU book). Everything above this line in the roadmap is NVIDIA-shaped; a meaningful share of the world's training runs on TPUs, and this is simply the best free treatment of scaling arithmetic that exists — roofline reasoning, collectives, and how to answer "how much memory do I need to serve this" from first principles. Read Parts 1–3 even if you never touch a TPU. Knowing that one alternative exists is what turns the SemiAnalysis strategy pieces from received wisdom into something you can argue with.

**Milestone:** You can trace a single token's journey: transistor → tensor core → kernel → attention layer → KV cache → serving engine → API → your app.

---

## Phase 7 — Staying Current (ongoing, from whenever you like)

**Subscribe (pick 3–4, don't drown):** **Latent Space** (+ AI News daily digest) · **Interconnects** (Nathan Lambert) · **Ahead of AI** (Raschka) · **Simon Willison's blog** — and copy Willison's habit: write up what you learn, publicly.

**Deep-dive menus:** train real models → **CS336** + **nanochat** (~$100 compute) · interpretability → **ARENA Ch. 1** + transformer-circuits.pub (or Track G) · kernels as a craft → PMPP cover-to-cover + GPU MODE competitions · research literacy → **CS224N** + one paper/week.

---

# PART 2 — BREADTH TRACKS (pick by interest)

*Seven major fields beyond the LLM stack — chosen because they're the ones you want to learn. (They're not "the rest of AI"; the Further Specializations menu at the end points to what else is out there.) Each track assumes Part 1 through Phase 2 (deep learning + PyTorch); some want more, noted per track. A→E build on each other loosely; F and G are standalone — but follow your curiosity.*

*Each track lists two numbers: the **survey** estimate (the scheduled 🟢 path below) and the **mastery** estimate (actually completing the full courses). The week-by-week plans below are the survey path; going deeper is always allowed and never required.*

*Full links: [RESOURCES.md](RESOURCES.md) sections 18–24.*

> **If you only do two tracks:** the two that most change how you think are **D** (RL — it retroactively explains Phase 3's post-training) and **G** (alignment & interpretability — it's the only place you look *inside* a model rather than at its outputs). Everything else is genuinely a matter of what you find beautiful.

## Track A — Computer Vision & Vision-Language Models
*Survey ~4 weeks · Mastery ~7+ weeks. Needs: Phase 2.*

- **Weeks A1–2 — Vision fundamentals:** Hugging Face **Community Computer Vision Course** 🟢 as the code-first spine (it's a 25–40 hr course — do the fundamentals + ViT + Unit 4 multimodal units; rest ⚪). CS231n 🔵 selectively for lecture-depth on CNNs/ViTs, or fast.ai's vision lessons 🔵 for the pragmatic path.
- **Week A3 — How machines connect images and language:** **CLIP, Intuitively and Exhaustively Explained** 🟢 → HF **Vision Language Models Explained** + 2025 update 🟢 → Raschka — **Understanding Multimodal LLMs** 🟢. This teaches the vision-encoder + projector + LLM recipe used by *open* VLMs (LLaVA, PaliGemma, Qwen-VL, Idefics). Frontier models (GPT-4V/5-class, Claude) are behaviorally similar multimodal systems, but their internals are undisclosed — don't assume they're built the same way.
- **Week A4 — Detection, segmentation + project:** Roboflow's **YOLO guide** 🔵 + **SAM 2 explainer** 🔵. **Project:** build a CLIP-powered semantic image search over your own photo library, or fine-tune a small open VLM. *How you'll know it worked:* measure retrieval recall@k on ~30 held-out queries you write before building, and collect a slice of qualitative failures (what kinds of queries miss?).

## Track B — Image & Video Generation / Diffusion
*Survey ~5 weeks · Mastery ~8+ weeks. Needs: Phase 2; the MIT course additionally assumes solid linear algebra + probability and is math-forward (its listed prereqs include real analysis) — the Alammar → HF-course path needs none of that.*

- **Week B1 — Intuition → math:** Jay Alammar — **The Illustrated Stable Diffusion** 🟢 → Lilian Weng — **What are Diffusion Models?** 🔵 (skim derivations first pass) → **The Annotated Diffusion Model** 🔵 (line-by-line PyTorch DDPM).
- **Weeks B2–3 — Hands-on:** Hugging Face **Diffusion Models Course** — HF's own estimate is ~6–8 hr *per unit* across 4 units, so the survey cut is explicit: **Units 1–2** 🟢 (DDPMs from scratch + fine-tuning/guidance) fill these two weeks; **Units 3–4** (Stable Diffusion internals + going further) are ⚪ — do them if the track hooks you.
- **Week B4 — The modern frontier:** MIT **6.S184: Flow Matching and Diffusion Models** 🔵 at survey depth (lectures 1–3 + skim the notes; the full course with labs is a 🟢 mastery option and the best single treatment of how current image/video models actually work) → Weng — **Diffusion Models for Video Generation** 🔵 + OpenAI's **Sora technical report** 🔵.
- **Week B5 — Project:** train a small diffusion model on a dataset of your choosing. *How you'll know it worked:* fix a prompt/condition set before training, show the denoising trajectory, and compare outputs across checkpoints — plus an honest note on failure modes you observe: low diversity, memorization of training images, conditioning failures, visual artifacts.
- ⚪ *Mastery options:* full MIT 6.S184 with labs; fast.ai Part 2 — rebuild Stable Diffusion from scratch (30+ hrs).

## Track C — Speech, Audio & Multimodal
*Survey ~3 weeks · Mastery ~4–5 weeks. Needs: Phase 2; Track A helps for the multimodal framing.*

- **Weeks C1–2 — The audio spine:** Hugging Face **Audio Course** 🟢 (8 units at 15–20 hr total — two work-weeks honestly): spectrograms → ASR → TTS with transformers. Read OpenAI's **Whisper** post 🔵 alongside the ASR unit.
- **Week C3 — Modern audio + project:** **LLM-based Audio Models** 🟢 (neural audio codecs → audio tokens → language models predicting sound) → Chip Huyen — **Multimodality and LMMs** 🟢. **Project:** fine-tune Whisper on your own voice, or build a local voice assistant (Whisper in → LLM → TTS out). *How you'll know it worked:* measure word-error rate (WER) on a held-out recording set before vs after fine-tuning; for the assistant, measure end-to-end latency per turn.

## Track D — General Reinforcement Learning
*Survey ~6 weeks · Mastery ~10+ weeks. Needs: Phase 2. Bonus: retroactively deepens everything from Phase 3 on RLHF, reasoning, and RLVR — this is the theory under Week 15's hands-on work, and doing it after Week 15 (rather than before) means every algorithm arrives with a reason to exist.*

- **Weeks D1–2 — Foundations:** David Silver's **RL course** 🟢 (lectures 1–5: MDPs, value functions, TD learning — ~1.5 hr each plus digestion time; two work-weeks) with **Sutton & Barto** ⚪ as companion text.
- **Week D3 — Foundations II:** Silver lectures 6–10 🔵 + Sutton & Barto policy-gradient chapters 🟢.
- **Weeks D4–5 — Deep RL, hands-on:** OpenAI **Spinning Up** 🟢 (concepts essay + implement VPG yourself — this is the slow, valuable part) → **Gymnasium** 🟢. HF **Deep RL Course** 🔵 as the fun parallel track.
- **Week D6 — PPO for real + project:** Weng — **Policy Gradient Algorithms** 🔵 → **The 37 Implementation Details of PPO** 🟢 (a direct window into RLHF training stacks). **Project:** train PPO on LunarLander from a CleanRL-style single file. *How you'll know it worked:* run ≥3 random seeds (5 if compute permits), plot learning curves with variance — single-seed RL results are noise, and learning that lesson viscerally is half the point of the project.
- ⚪ *Go deeper:* Berkeley **CS285** (Levine) — offline RL, model-based RL; the graduate capstone.

## Track E — Robotics & Embodied AI
*Survey ~4 weeks · Mastery ~8+ weeks (more with hardware). Needs: Track D (RL/imitation literacy) + Track A helps. This is where everything converges: VLMs + RL + control.*

> ⚠️ **Course availability (checked Aug 2026):** the HF Robotics Course currently has only its intro + classical-robotics units released; the RL, imitation-learning, and foundation-model units are "Coming Soon." The plan below uses what exists today — recheck the course quarterly and fold in new units as they land.

- **Week E1 — The big idea (VLAs):** DeepMind — **RT-2** 🟢 (web-scale VLM → robot actions) → **OpenVLA** 🔵 → Physical Intelligence — **π0 / openpi** 🟢 (VLM backbone + flow matching for continuous actions — Track B's flow matching, reappearing in robotics). Sergey Levine's Substack 🔵 for state-of-the-field commentary.
- **Week E2 — Foundations:** HF **Robotics Course**, released units 🟢 (intro + classical robotics: kinematics, control, and their limitations — the "why learning-based robotics won" context).
- **Weeks E3–4 — Hands-on via LeRobot directly:** since the course's hands-on units aren't out yet, go straight to the **LeRobot docs and tutorials** 🟢. First, a half-day imitation-learning primer 🟢 (Track D taught you RL, not IL): behavior cloning = supervised learning on demonstration data; ACT adds action chunking + a transformer — the LeRobot docs' policy pages cover this. Then: run a simulation environment, use a public LeRobot dataset, and train an ACT-style policy in sim (**MuJoCo** 🔵 for the physics side). *Platform note:* some sim environments (e.g. LIBERO) are Linux-only, and training wants a real GPU — budget a cheap Linux cloud box or Colab, roughly $10–30 for this project; Apple-silicon Macs handle the lighter environments but don't fight the toolchain. **Project:** a trained policy completing a sim manipulation task. *How you'll know it worked:* task success rate over ≥50 evaluation episodes, compared against a scripted or random baseline.
- **Optional but delightful — real hardware:** the **SO-101 arm** (~$100–200 3D-printed, ~$300+ pre-assembled): teleoperate, record demonstrations, train a policy on your desk (NVIDIA's free sim-to-real SO-101 tutorial bridges Isaac ↔ LeRobot). Success metric stays the same: task success rate — plus a workspace the arm can't damage anything in.
- ⚪ *Rigor track:* Tedrake's MIT **Robotic Manipulation** — the classical foundations VLA papers assume.

## Track F — Causal Reasoning & Experimentation
*Survey ~4 weeks · Mastery ~7+ weeks. Needs: all of Phase 1 (not just the math week) — causal inference leans on regression, sampling, and uncertainty. If your statistics are thin, spend 2–3 bridge days first: StatQuest's regression videos + ISLP's inference chapter. Standalone otherwise — and it quietly upgrades your Phase 3–4 skills by making you harder to fool, including by yourself.*

- **Week F1 — The mental model:** Pearl — **The Book of Why** 🟢 (~$15; the causal ladder, why prediction ≠ causation) — a reading week, enjoy it.
- **Weeks F2–3 — The toolkit:** **Causal Inference for the Brave and True** (free, Python-native; a 30–50 hr book, so the survey cut is explicit): 🟢 = Part I's core arc — bias & why association ≠ causation, randomized experiments, the stats refresher, graphical causal models & confounding, and one quasi-experimental design of your choice (IV, diff-in-diff, or regression discontinuity). The remaining Part I designs and all of Part II (ML-based methods, CATE) are 🔵/⚪ — return when a real question needs them. The **Mixtape** ⚪ is the econometrics-flavored alternative.
- **Week F4 — Experimentation + project:** Kohavi et al. — **Trustworthy Online Controlled Experiments** 🔵 (ch. 1 free) → **DoWhy/EconML tutorial** 🟢. **Project:** take a real or simulated product dataset, express a causal graph in DoWhy, estimate an effect, and run DoWhy's refutation tests. *How you'll know it worked — and a caveat that IS the lesson:* refuters stress-test your estimate under specific perturbations (placebo treatments, random confounders, subset stability); passing them increases confidence but cannot prove your causal assumptions. State which assumptions remain untestable in your write-up. That sentence is the deliverable.

## Track G — Alignment & Interpretability
*Survey ~4 weeks · Mastery ~10+ weeks. Needs: Phase 2 (Phase 3 strongly recommended — you need to know what post-training is before asking whether it worked). Standalone otherwise.*

> *Why this is a track and not a footnote: Phase 4's Week 22 secures your* application *against hostile input. Nothing else in this roadmap asks whether the* model *is doing what you think — or lets you look inside one and check. This is also the layer that will most change what you believe about the rest of the stack, because it's where you discover how much of "the model reasons" is a description of behavior rather than mechanism.*

- **Week G1 — The problem, stated precisely:** **AI Safety Fundamentals — Alignment Course** 🟢 (free, well-curated, ~4–5 weeks of reading if taken in full — do the first units now). The goal is vocabulary and honest problem statements: specification vs. robustness, reward hacking (you met it in Week 15 with your hands), scalable oversight, evaluations for dangerous capabilities. Read it as an engineer, skeptically — the field contains both rigorous work and unfalsifiable argument, and telling them apart is part of the skill.
- **Week G2 — Looking inside:** Anthropic — **Mapping the Mind of a Large Language Model** 🟢 (features, superposition, dictionary learning) → **transformer-circuits.pub** 🔵 for the primary sources. Then Neel Nanda's **Getting Started in Mech Interp** 🟢 for the concrete on-ramp and an honest account of what the field can and can't currently do.
- **Weeks G3–4 — Hands-on:** **ARENA Chapter 1: Transformer Interpretability** 🟢 — the real curriculum, and the reason Phase 2 was non-negotiable: you'll be indexing into activations of a transformer you already understand. Do the induction-heads and activation-patching material at minimum. **Project:** take a small open model and localize one specific behavior — find the heads or features responsible, then intervene to break or strengthen it. *How you'll know it worked:* a causal claim backed by an ablation, not a correlation backed by a pretty plot. State explicitly what your intervention does *not* prove.
- ⚪ *Go deeper:* the rest of ARENA (RL and evals chapters); Anthropic's **introspection** and model-organisms research for where the frontier actually is.

## Further Specializations (pointers, not scheduled)

Fields this roadmap deliberately doesn't cover — each is its own world. If one calls to you, ask me and we'll build a track:
**Recommender systems** (what powers feeds/suggestions — Eugene Yan writes well on this) · **Search & learning-to-rank** · **Time-series forecasting** · **Graph neural networks** · **Bayesian/probabilistic ML** · **Classical planning & symbolic AI** · **Edge AI / TinyML** (models on microcontrollers) · **Privacy-preserving & federated learning** · **Alternative architectures** (state-space models/Mamba, linear and hybrid attention — Week 10 tells you why the transformer's quadratic attention and growing KV cache are the pressure these are trying to relieve).

---

## How to Use This Plan

1. **Track progress in this file** — check things off, add notes, reorder. It's yours.
2. **Core > Survey > Reference is the pressure valve.** Behind schedule? Surveys become skims, references get skipped. The 🟢 spine is non-negotiable; everything else flexes.
3. **Projects > completion.** Skipping a week of videos to finish a milestone is always the right trade.
4. **When stuck, switch resources, not topics.** Every topic has 2–3 alternative explainers in [RESOURCES.md](RESOURCES.md).
5. **Dependencies, honestly:** Phase 0 → 1 → 2 → 3 is the hard chain (Phase 2 needs Phase 1's math and PyTorch; Week 10's architecture week needs Week 9's GPT-2). Phases 4 → 4.5 go together; Phase 5 and Part 2 branch off Phase 2–3. Phase 6 wants Phase 5's intuitions *and* Week 10's KV cache. Within Part 2: D before E; A before B is nice; F needs all of Phase 1; G wants Phase 3; C floats free. **The real branch point is the end of Phase 3** — from there, every track is profitably open to you, so if a track is calling louder than the next phase, take it.
6. **Budget, honestly:** ~$150 in books if you buy all three (AI Engineering, PMPP, Raschka's LLM book) + possibly ~$49 for one focused Coursera month in Week 3 + realistically **$60–180 in GPU rental and API credits** for the projects (incl. ~$10–30 for the Track E sim work and ~$5–20 for Week 15's GRPO run) + optional ~$100 nanochat training run + optional ~$100–300 SO-101 arm. Everything else is free.
7. **Use AI to learn AI.** Ask Claude to quiz you, re-explain stuck concepts, or review your micrograd. Just write the code yourself first.
8. **Retention, or: the month-9 problem.** This plan is ~a year long. Without a review loop you will not remember Week 5 by Week 30, and the later phases quietly depend on the earlier ones. Two cheap habits, ~8 hours total across the whole roadmap: (a) keep a `NOTES.md` per phase — three bullets after each week, what clicked and what didn't; (b) at every phase boundary, spend one hour re-deriving something from *two* phases back from a blank file (rebuild micrograd's backward pass, re-compute a KV cache size, re-explain ZeRO). If you can't, that's not failure — that's the review loop doing precisely its job.
9. **The Pre-Phase Audit Protocol.** Before starting any phase or track, ask Claude: *"Audit Phase N first."* The audit: (1) verify every link in that section still resolves and hasn't been superseded, (2) re-check availability claims (course units, pricing, prereqs), (3) **check whether the *practice* changed, not just the link** — the real decay mode here is not a 404 but a resource that loads perfectly while teaching a superseded default (chunking-heavy RAG written for 8k context windows; PPO-shaped RLHF written before GRPO; dense-model assumptions written before MoE went mainstream). Ask of each 🟢 item: *is this still how it's done, or just still online?* (4) research what's genuinely new since the last revision — filtered by the end-to-end-systems principle (a few transparent systems studied deeply beat dozens of model-release links), (5) propose a diff, then update the files + HTML. Audit depth by volatility:
   - 🟩 **Stable** (skip or 5-min link check): Phase 0–2, math, Track D foundations, Track F
   - 🟨 **Medium** (light audit): Phase 3 Weeks 9–13, Phase 5 (concepts stable, tools move), Track D deep-RL
   - 🟥 **Fast-moving** (full audit): Phase 3 Weeks 10 + 14–15 (architecture and RLVR move fastest of anything in Part 1), Phase 4, 4.5, 6, Tracks A, B, C, E, G — and anything marked "recheck quarterly"

   *Last full revision: Aug 2026 (structural review — added architecture/scaling-laws, RLVR, coding-agents, training-practice weeks and Track G). This protocol is why the roadmap doesn't chase today's frontier — content gets refreshed exactly when it becomes relevant to you.*
