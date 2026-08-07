# The Full-Stack AI Resource Library

Companion to [ROADMAP.md](ROADMAP.md). Links verified August 2026. **Free unless marked otherwise.** ⭐ = if you only do one thing in that section, do this one.

---

## Part 1 — The LLM Stack

### 1. Math Foundations

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Essence of Linear Algebra | 3Blue1Brown | Videos, ~4 hr | https://www.3blue1brown.com/topics/linear-algebra |
| Essence of Calculus | 3Blue1Brown | Videos, ~3 hr | https://www.3blue1brown.com/topics/calculus |
| Visual Information Theory (entropy, cross-entropy) | Chris Olah | Blog, ~1 hr | https://colah.github.io/posts/2015-09-Visual-Information/ |
| Mathematics for Machine Learning | Deisenroth, Faisal & Ong | Free book PDF, reference | https://mml-book.github.io/ |
| Statistics 110: Probability | Blitzstein (Harvard) | Video course, ~40 hr (optional depth) | https://projects.iq.harvard.edu/stat110 |
| StatQuest video index | Josh Starmer | Short videos, à la carte | https://statquest.org/video_index.html |

### 2. Intro Machine Learning

> **Coursera note (accurate as of Aug 2026):** Coursera ended free auditing in Aug 2025 — courses now offer only a free first-module preview. Full free access requires financial aid (~15-day approval). Plan accordingly.

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Machine Learning Specialization | Andrew Ng | Course, ~60–90 hr — first module free; then ~$49/mo or financial aid | https://www.coursera.org/specializations/machine-learning-introduction |
| ML Specialization — official YouTube playlist (partial, Course 1) | DeepLearning.AI | Videos, free | https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI |
| Stanford CS229 (Ng, Autumn 2018) — free, math-heavier alternative | Stanford Online | Full lecture course, free | https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU |
| Kaggle Learn: Intro to ML | Kaggle | Interactive notebooks, ~3 hr | https://www.kaggle.com/learn/intro-to-machine-learning |
| An Introduction to Statistical Learning (ISLP) | James, Witten, Hastie, Tibshirani | Free book PDF | https://www.statlearning.com/ |

### 3. Deep Learning

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Neural Networks series (incl. GPT/attention chapters) | 3Blue1Brown | Videos, ~4 hr | https://www.3blue1brown.com/topics/neural-networks |
| **Neural Networks: Zero to Hero** ⭐ | Andrej Karpathy | Code-along videos, 30–50 hr | https://karpathy.ai/zero-to-hero.html |
| micrograd (build backprop from scratch) | Karpathy | Repo + video | https://github.com/karpathy/micrograd |
| Practical Deep Learning for Coders | fast.ai (Jeremy Howard) | Course + free book, ~70 hr | https://course.fast.ai/ |
| Understanding Deep Learning | Simon Prince | Free book PDF (modern) | https://udlbook.github.io/udlbook/ |
| Dive into Deep Learning | Zhang, Lipton, Li, Smola | Free interactive book | https://d2l.ai/ |
| Deep Learning (the classic) | Goodfellow, Bengio, Courville | Free book, theory reference | https://www.deeplearningbook.org/ |
| CS231n notes (backprop, optimization) | Stanford | Written notes, ~20 hr | https://cs231n.github.io/ |
| Gradient descent / backprop lessons | 3Blue1Brown | 3 videos, ~40 min | https://www.3blue1brown.com/lessons/backpropagation/ |

### 4. PyTorch

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Learn PyTorch for Deep Learning (Zero to Mastery) | Daniel Bourke | Book + videos, ~40–60 hr (materials free) | https://www.learnpytorch.io/ |
| Official PyTorch tutorials | PyTorch | Docs, 2–3 hr for basics | https://pytorch.org/tutorials/ |
| UvA Intro to PyTorch notebook | Univ. of Amsterdam | Notebook, ~3 hr | https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html |
| UvA Deep Learning notebooks (advanced) | Univ. of Amsterdam | Notebooks, à la carte | https://uvadlc.github.io/ |
| MLX — Apple-silicon local experimentation | Apple | Framework, reference | https://github.com/ml-explore/mlx |

### 5. Transformers & LLMs

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| The Illustrated Transformer | Jay Alammar | Blog, ~1 hr | https://jalammar.github.io/illustrated-transformer/ |
| LLM Visualization (3D, interactive) | Brendan Bycroft | Browser tool, ~1 hr | https://bbycroft.net/llm/ |
| Transformer Explainer (live GPT-2 in browser) | Georgia Tech Polo Club | Browser tool, ~30 min | https://poloclub.github.io/transformer-explainer/ |
| [1hr Talk] Intro to Large Language Models | Karpathy | Video, 1 hr | https://www.youtube.com/watch?v=zjkBMFhNj_g |
| **Deep Dive into LLMs like ChatGPT** ⭐ | Karpathy | Video, 3.5 hr | https://www.youtube.com/watch?v=7xTGNNLPyMI |
| Let's build GPT: from scratch | Karpathy | Code-along, ~2 hr + practice | https://www.youtube.com/watch?v=kCc8FmEb1nY |
| Let's build the GPT Tokenizer | Karpathy | Code-along, ~2 hr | https://www.youtube.com/watch?v=zduSFxRajkE |
| build-nanogpt (reproduce GPT-2) | Karpathy | Repo + 4 hr video | https://github.com/karpathy/build-nanogpt |
| nanochat (full pipeline: pretrain→SFT→serve) | Karpathy | Repo, ~$100 compute to train | https://github.com/karpathy/nanochat |
| Build a Large Language Model (From Scratch) | Sebastian Raschka | Book ~$40 + free repo | https://github.com/rasbt/LLMs-from-scratch |
| Hugging Face LLM Course | Hugging Face | Interactive course, 20–40 hr (do chapters selectively) | https://huggingface.co/learn/llm-course/en/chapter1/1 |
| Stanford CS336: Language Modeling from Scratch | Percy Liang et al. | Full course + assignments | https://cs336.stanford.edu/spring2025 |
| Stanford CS224N: NLP with Deep Learning | Chris Manning et al. | Full course | https://web.stanford.edu/class/cs224n/ |
| Sampling explained (temperature, top-p/k) | MachineLearningPlus | Blog + code, ~1 hr | https://machinelearningplus.com/gen-ai/llm-temperature-top-p-top-k-explained/ |

### 6. Modern LLM Architecture & Scaling Laws

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **The Big LLM Architecture Comparison** ⭐ (DeepSeek-V3 → Kimi K2: RoPE, RMSNorm, SwiGLU, GQA/MLA, sliding-window & hybrid attention — the whole GPT-2→2026 delta in one article) | Sebastian Raschka | Article, 2–3 hr | https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison |
| Mixture of Experts Explained | Hugging Face | Blog, ~1 hr | https://huggingface.co/blog/moe |
| A Visual Guide to Mixture of Experts | Maarten Grootendorst | Visual blog, ~1 hr | https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts |
| RoFormer (the RoPE paper) | Su et al. | Paper, reference | https://arxiv.org/abs/2104.09864 |
| GQA: Training Generalized Multi-Query Transformer Models | Ainslie et al. (Google) | Paper, reference | https://arxiv.org/abs/2305.13245 |
| Scaling laws literature review (the map of the whole literature) | Epoch AI | Article, 1–2 hr | https://epoch.ai/blog/scaling-laws-literature-review |
| Scaling Laws for Neural Language Models (Kaplan, 2020) | OpenAI | Paper, reference | https://arxiv.org/abs/2001.08361 |
| Training Compute-Optimal LLMs (Chinchilla, 2022) | Hoffmann et al. (DeepMind) | Paper, reference | https://arxiv.org/abs/2203.15556 |
| Go smol or go home (why *inference* cost, not training cost, sets the real optimum) | Harm de Vries | Blog, ~45 min | https://www.harmdevries.com/post/model-size-vs-compute-overhead/ |
| The Transformer Taxonomy (a compact index of architectural variants) | kipply (Carol Chen) | Reference list | https://kipp.ly/transformer-taxonomy/ |

*The KV cache has no single canonical explainer — it's covered inside the Raschka comparison (via GQA/MLA), kipply's inference arithmetic (§16), and Inside vLLM (§16). Compute one by hand: `2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_param`.*

### 7. Data: The Underrated Layer

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **FineWeb: decanting the web for the finest text data** ⭐ | Hugging Face | Technical report, 2–3 hr | https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1 |
| HF LLM Course — data processing chapters | Hugging Face | Course chapters, ~4 hr | https://huggingface.co/learn/llm-course/en/chapter5/1 |
| Contamination section, LLM Evaluation Guidebook | Hugging Face | Guidebook section, ~1 hr | https://github.com/huggingface/evaluation-guidebook |

### 8. Fine-Tuning & Post-Training

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| smol-course — now 6 units: SFT, evaluation, preference alignment, VLMs, RL, synthetic data | Hugging Face | Course, ~3–4 hr/unit (~20 hr total); GPU recommended | https://huggingface.co/learn/smol-course |
| Fine-tuning LLMs Guide (LoRA/QLoRA) | Unsloth | Docs + free Colabs, 2–5 hr | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide |
| **RLHF Book** ⭐ | Nathan Lambert (Ai2) | Free open textbook, 20–40 hr | https://rlhfbook.com/ |

*Reasoning models and RL with verifiable rewards — the other half of modern post-training — are in §9.*

### 9. Reasoning & RL with Verifiable Rewards

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **RLHF Book — ch. 14, Reasoning & RLVR** ⭐ | Nathan Lambert (Ai2) | Book chapter, 1–2 hr | https://rlhfbook.com/c/14-reasoning.html |
| Understanding Reasoning LLMs | Sebastian Raschka | Article, 1–2 hr | https://magazine.sebastianraschka.com/p/understanding-reasoning-llms |
| DeepSeek-R1 paper (R1-Zero = pure-RL reasoning; R1 = cold-start + multi-stage) | DeepSeek-AI | Paper, 2–3 hr | https://arxiv.org/pdf/2501.12948 |
| **Unsloth RL / GRPO guide + free Colabs** ⭐ (the hands-on path — a GRPO run you can actually afford) | Unsloth | Docs + notebooks, 3–5 hr | https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide |
| GRPO Trainer docs (the concrete implementation reference) | Hugging Face TRL | Docs, 1–2 hr | https://huggingface.co/docs/trl/grpo_trainer |
| open-r1 (open reproduction of the R1 pipeline) | Hugging Face | Repo, browse | https://github.com/huggingface/open-r1 |
| verifiers (how RL environments are actually structured) | Will Brown | Repo, browse | https://github.com/willccbb/verifiers |
| Reasoning From Scratch | Sebastian Raschka | Book + repo (in progress) | https://github.com/rasbt/reasoning-from-scratch |

*This section will churn faster than any other in this file. Treat the mechanics (GRPO, verifiable rewards, reward hacking) as durable and the specific tooling as disposable.*

### 10. Evaluation

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| LLM Evaluation Guidebook | Hugging Face (Fourrier) | Guidebook, 5–10 hr | https://github.com/huggingface/evaluation-guidebook |
| CS224N Benchmarking lecture | Yann Dubois | Video, 1.5 hr | https://www.youtube.com/watch?v=TO0CqzqiArM |

*Interpretability moved to §24 (Track G), where it has room to be a curriculum rather than three bookmarks.*

### 11. AI Engineering (Apps, Prompting, RAG, Agents)

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **AI Engineering** (book) ⭐ | Chip Huyen | Book ~$45; free repo w/ summaries | https://github.com/chiphuyen/aie-book |
| Chip Huyen's blog | Chip Huyen | Blog | https://huyenchip.com/blog/ |
| What We Learned from a Year of Building with LLMs (I–III) | Yan, Husain, et al. | Articles, 3–4 hr | https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/ |
| Interactive Prompt Engineering Tutorial | Anthropic | Graded notebooks, 6–10 hr | https://github.com/anthropics/prompt-eng-interactive-tutorial |
| Prompt engineering docs | Anthropic | Docs, 2–3 hr | https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview |
| Current OpenAI prompting guide | OpenAI Cookbook | Article, 1–2 hr | https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide |
| RAG course | DeepLearning.AI | Course, 10–15 hr | https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/ |
| Building & Evaluating Advanced RAG | DeepLearning.AI + TruEra | Short course, ~2 hr | https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/ |
| Patterns for Building LLM-based Systems | Eugene Yan | Essay, 2–3 hr | https://eugeneyan.com/writing/llm-patterns/ |
| **Building Effective Agents** ⭐ | Anthropic | Article, 45 min | https://www.anthropic.com/engineering/building-effective-agents |
| Effective Context Engineering for AI Agents | Anthropic | Article, 45 min | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents |
| Agents | Chip Huyen | Blog, 1.5 hr | https://huyenchip.com/2025/01/07/agents.html |
| A Practical Guide to Building Agents | OpenAI | PDF, 1–2 hr | https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/ |
| Intro to Model Context Protocol | Anthropic Academy | Course, 3–6 hr | https://anthropic.skilljar.com/introduction-to-model-context-protocol |
| MCP short course | DeepLearning.AI + Anthropic | Short course, ~2 hr | https://www.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic |
| Intro to LangGraph | LangChain Academy | Course, 6–8 hr | https://academy.langchain.com/courses/intro-to-langgraph |
| Your AI Product Needs Evals | Hamel Husain | Essay, 1 hr | https://hamel.dev/blog/posts/evals/index.html |
| **LLM Evals FAQ** ⭐ | Husain & Shankar | FAQ, 3–5 hr | https://hamel.dev/blog/posts/evals-faq/ |
| LLM-as-a-Judge guide | Hamel Husain | Article | https://hamel.dev/blog/posts/llm-judge/ |
| Evaluating AI Agents | DeepLearning.AI + Arize | Short course, ~2.5 hr | https://www.deeplearning.ai/courses/evaluating-ai-agents/ |
| LangSmith evaluation concepts | LangChain | Docs, 1–2 hr | https://docs.langchain.com/langsmith/evaluation-concepts |
| Full Stack LLM Bootcamp | The Full Stack | Lectures, ~10 hr | https://fullstackdeeplearning.com/llm-bootcamp/ |
| Prompt caching docs (cache-hit input tokens ~0.1× price; writes cost more; output unaffected) | Anthropic | Docs, 1 hr | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| Designing Machine Learning Systems | Chip Huyen | Book ~$45; free summaries | https://github.com/chiphuyen/dmls-book |

### 12. Working with Coding Agents
*The discipline of using AI to build software — distinct from building AI software. It's the layer you spend the most hours in and the one most people never study deliberately.*

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **Claude Code Best Practices** ⭐ (read as engineering doctrine, not tool tips) | Anthropic | Guide, 1–2 hr | https://www.anthropic.com/engineering/claude-code-best-practices |
| Using LLMs for code (the honest practitioner account) | Simon Willison | Article, ~1 hr | https://simonwillison.net/2025/Mar/11/using-llms-for-code/ |
| Writing Tools for Agents (tool design *is* prompt design — pairs with the MCP work) | Anthropic | Article, ~1 hr | https://www.anthropic.com/engineering/writing-tools-for-agents |
| Effective Context Engineering for AI Agents (re-read with a codebase in mind) | Anthropic | Article, 45 min | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents |

### 13. Security & Safety for AI Apps

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **OWASP Top 10 for LLM Applications** ⭐ | OWASP GenAI project | Guide, 2–3 hr | https://genai.owasp.org/llm-top-10/ |
| Prompt injection series (incl. the "lethal trifecta") | Simon Willison | Blog series, 3–4 hr | https://simonwillison.net/series/prompt-injection/ |
| NIST AI Risk Management Framework + GenAI Profile | NIST | Framework docs, skim ~1 hr | https://www.nist.gov/itl/ai-risk-management-framework |

### 14. GPUs & CUDA

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **Making Deep Learning Go Brrrr** ⭐ | Horace He | Blog, 1–2 hr | https://horace.io/brrr_intro.html |
| How GPU Computing Works (GTC talk) | Stephen Jones, NVIDIA | Video, ~40 min | https://www.youtube.com/watch?v=3l10o0DYJXg |
| How CUDA Programming Works (GTC talk) | Stephen Jones, NVIDIA | Video, ~40 min | https://www.youtube.com/watch?v=QQceTDjA4f4 |
| GPU Glossary | Modal Labs | Reference hypertext, 2–4 hr | https://modal.com/gpu-glossary |
| An Even Easier Introduction to CUDA | Mark Harris, NVIDIA | Tutorial, 1–2 hr | https://developer.nvidia.com/blog/even-easier-introduction-cuda/ |
| Programming Massively Parallel Processors (PMPP, 4th ed.) | Hwu, Kirk, El Hajj | Textbook, ~$70, 40–80 hr | https://www.oreilly.com/library/view/programming-massively-parallel/9780323984638/ |
| **GPU MODE lectures + Discord** ⭐ | Saroufim, Köpf + community | Lecture series + community | https://www.gpumode.com/lectures |
| GPU MODE lecture notes (written) | Christian Mills | Notes | https://christianjmills.com/series/notes/cuda-mode-notes.html |
| GPU MODE resource-stream | GPU MODE community | Curated index | https://github.com/gpu-mode/resource-stream |

### 15. Kernels (Triton, matmul, flash attention)

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Official Triton tutorials | Triton project | Code tutorials, 6–10 hr | https://triton-lang.org/main/getting-started/tutorials/index.html |
| How to Optimize a CUDA Matmul Kernel | Simon Boehm | Blog + repo, 3–6 hr | https://siboehm.com/articles/22/CUDA-MMM |
| Advanced Matmul on Modern NVIDIA GPUs | Aman Salykov | Blog + code, 3–5 hr | https://salykova.github.io/sgemm-gpu |
| ELI5: FlashAttention | Aleksa Gordić | Blog, 1–1.5 hr | https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad |
| triton-resources (incl. Triton Puzzles) | rkinas | Curated list | https://github.com/rkinas/triton-resources |
| torch.compile tutorial | PyTorch | Tutorial, 1–2 hr | https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html |

### 16. Distributed Training, Inference & Data Centers

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **The Ultra-Scale Playbook** ⭐ | Hugging Face | Interactive book, 10–20 hr (survey first) | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| How to Train Really Large Models on Many GPUs | Lilian Weng | Blog, ~1 hr | https://lilianweng.github.io/posts/2021-09-25-train-large/ |
| Distributed Training & Efficient Finetuning | Sumanth Hegde | Blog, 1–2 hr | https://sumanthrh.com/post/distributed-and-efficient-finetuning/ |
| Transformer Math 101 | EleutherAI | Blog, ~1 hr | https://blog.eleuther.ai/transformer-math/ |
| **Inside vLLM** ⭐ | Aleksa Gordić | Blog, 2–4 hr | https://www.aleksagordic.com/blog/vllm |
| vLLM + PagedAttention (original post) | vLLM team | Blog, ~30 min | https://blog.vllm.ai/2023/06/20/vllm.html |
| Transformer Inference Arithmetic | kipply (Carol Chen) | Blog, 1–2 hr | https://kipp.ly/transformer-inference-arithmetic/ |
| A Visual Guide to Quantization | Maarten Grootendorst | Blog, ~1 hr | https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization |
| LLM Inference Handbook | BentoML | Handbook, 4–8 hr | https://bentoml.com/llm/ |
| Hitchhiker's Guide to Speculative Decoding | PyTorch | Blog | https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ |
| Beginner's Guide to AI Datacenter Interconnects | Vikram Sekar | Blog, ~45 min | https://www.viksnewsletter.com/p/a-beginners-guide-to-ai-interconnects |
| Datacenter Anatomy Part 1: Electrical | SemiAnalysis | Article, partial paywall | https://newsletter.semianalysis.com/p/datacenter-anatomy-part-1-electrical |
| Datacenter Anatomy Part 2: Cooling | SemiAnalysis | Article, partial paywall | https://newsletter.semianalysis.com/p/datacenter-anatomy-part-2-cooling-systems |
| 100,000 H100 Clusters | SemiAnalysis | Article, partial paywall | https://newsletter.semianalysis.com/p/100000-h100-clusters-power-network |
| **ML Engineering (the large-run field manual: throughput debugging, fault tolerance, instabilities)** ⭐ | Stas Bekman | Open book, à la carte | https://github.com/stas00/ml-engineering |
| **The Smol Training Playbook** ⭐ (one team's honest end-to-end account — including the ablations that failed) | Hugging Face | Interactive book, 5–10 hr | https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook |
| Training using float8 with FSDP2 (why FP8 training is standard and what it costs in stability) | PyTorch | Blog, ~45 min | https://pytorch.org/blog/training-using-float8-fsdp2/ |
| NCCL (the collectives everything above is built from: all-reduce, all-gather, reduce-scatter) | NVIDIA | Library + docs, reference | https://github.com/NVIDIA/nccl |
| SGLang (the second serving engine — RadixAttention / prefix-cache reuse) | LMSYS / SGLang team | Docs, 2–3 hr | https://docs.sglang.ai/ |
| Mastering LLM Techniques: Inference Optimization (prefill vs decode, in NVIDIA's words) | NVIDIA | Article, 1–2 hr | https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ |
| **How To Scale Your Model** ⭐ (the JAX/TPU book — the non-NVIDIA view, and the best free treatment of scaling arithmetic anywhere) | Austin et al. (Google DeepMind) | Interactive book, 10–15 hr | https://jax-ml.github.io/scaling-book/ |

### 17. Staying Current

| Resource | Author | Cadence | Link |
|---|---|---|---|
| Latent Space | swyx & Alessio Fanelli | Weekly | https://www.latent.space/ |
| AI News (daily digest) | smol.ai | Daily | https://news.smol.ai/ |
| Interconnects | Nathan Lambert | 1–3×/week | https://www.interconnects.ai/ |
| Ahead of AI | Sebastian Raschka | ~Monthly | https://magazine.sebastianraschka.com/ |
| Simon Willison's blog | Simon Willison | Daily | https://simonwillison.net/ |
| Eugene Yan's writing | Eugene Yan | Occasional | https://eugeneyan.com/writing/ |

---

## Part 2 — Breadth Tracks

### 18. Track A — Computer Vision & Vision-Language Models

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| CS231n: Deep Learning for Computer Vision | Stanford | Full course (selective use), 60–80 hr | https://cs231n.stanford.edu/ |
| CS231n Spring 2025 lecture playlist | Stanford | Videos | https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16 |
| **Community Computer Vision Course** ⭐ | Hugging Face | Code-first course, 25–40 hr (Unit 4 = best free VLM unit) | https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome |
| CLIP, Intuitively and Exhaustively Explained | Daniel Warfield | Blog, ~30 min | https://towardsdatascience.com/clip-intuitively-and-exhaustively-explained-1d02c07dbf40/ |
| Vision Language Models Explained (+ 2025 update) | Hugging Face | 2 blog posts, ~1 hr | https://huggingface.co/blog/vlms and https://huggingface.co/blog/vlms-2025 |
| Understanding Multimodal LLMs | Sebastian Raschka | Article, ~1 hr | https://magazine.sebastianraschka.com/p/understanding-multimodal-llms |
| Guide to YOLO Models | Roboflow | Blog, ~45 min | https://blog.roboflow.com/guide-to-yolo-models/ |
| What is SAM 2 (Segment Anything)? | Roboflow | Blog, ~45 min | https://blog.roboflow.com/what-is-segment-anything-2/ |

### 19. Track B — Image/Video Generation & Diffusion

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| The Illustrated Stable Diffusion | Jay Alammar | Visual blog, ~45 min | https://jalammar.github.io/illustrated-stable-diffusion/ |
| What are Diffusion Models? | Lilian Weng | Math-heavy blog, 2–3 hr | https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ |
| The Annotated Diffusion Model | Rogge & Rasul (HF) | Line-by-line PyTorch, ~2 hr | https://huggingface.co/blog/annotated-diffusion |
| **Hugging Face Diffusion Models Course** ⭐ | HF + Jonathan Whitaker | Course + Colabs, ~6–8 hr per unit × 4 units (~24–32 hr) | https://huggingface.co/learn/diffusion-course/unit0/1 |
| **MIT 6.S184: Flow Matching & Diffusion Models** ⭐ (math-forward: listed prereqs incl. linear algebra, real analysis, probability) | Holderrieth & Erives (MIT) | Lectures + labs, 15–20 hr | https://diffusion.csail.mit.edu/ |
| Diffusion Models for Video Generation | Lilian Weng | Blog, 1–1.5 hr | https://lilianweng.github.io/posts/2024-04-12-diffusion-video/ |
| Sora: video generation models as world simulators | OpenAI | Technical report, ~30 min | https://openai.com/index/video-generation-models-as-world-simulators/ |
| fast.ai Part 2: Foundations → Stable Diffusion | Jeremy Howard | 30+ hr course (mastery option) | https://course.fast.ai/Lessons/part2.html |

### 20. Track C — Speech, Audio & Multimodal

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **Hugging Face Audio Course** ⭐ | Hugging Face | 8 units + Colabs, 15–20 hr | https://huggingface.co/learn/audio-course/chapter0/introduction |
| Introducing Whisper | OpenAI | Blog + paper, ~30 min | https://openai.com/index/whisper/ |
| LLM-based Audio Models (codecs → audio tokens → TTS) | Yatharth S. (HF community) | Blog, ~45 min | https://huggingface.co/blog/YatharthS/llm-tts-models |
| Multimodality and Large Multimodal Models | Chip Huyen | Blog, ~1.5 hr | https://huyenchip.com/2023/10/10/multimodal.html |

### 21. Track D — General Reinforcement Learning

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Reinforcement Learning: An Introduction (2nd ed.) | Sutton & Barto | Free book PDF, companion text | http://incompleteideas.net/book/the-book-2nd.html |
| **David Silver's RL course (DeepMind/UCL)** ⭐ | David Silver | 10 lectures, ~15 hr | https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ |
| Spinning Up in Deep RL | OpenAI (Achiam) | Docs + reference code, 2–4 wks | https://spinningup.openai.com/en/latest/ |
| Deep RL Course | Hugging Face | Hands-on units, 30–40 hr | https://huggingface.co/learn/deep-rl-course/en/unit0/introduction |
| Policy Gradient Algorithms | Lilian Weng | Blog (the derivation reference), 2 hr | https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ |
| **The 37 Implementation Details of PPO** ⭐ | Huang, Dossa, Raffin et al. | Blog + CleanRL code, 3–4 hr | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ |
| CleanRL (single-file RL implementations) | Costa Huang et al. | Code + docs | https://docs.cleanrl.dev/ |
| Gymnasium (environments) | Farama Foundation | Library + tutorials | https://gymnasium.farama.org/ |
| CS285: Deep RL (grad capstone) | Sergey Levine (Berkeley) | Full course, 60+ hr | https://rail.eecs.berkeley.edu/deeprlcourse/ |

### 22. Track E — Robotics & Embodied AI

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| Hugging Face Robotics Course (LeRobot) — ⚠️ as of Aug 2026 only intro + classical units released; RL/imitation/foundation-model units "Coming Soon" (recheck quarterly) | Hugging Face | Course (partial), released units ~8–10 hr | https://huggingface.co/learn/robotics-course |
| **LeRobot library + docs** ⭐ (the executable hands-on path today: sim envs, datasets, train ACT policies) | Hugging Face | Code + docs + tutorials | https://huggingface.co/docs/lerobot |
| SO-101 robot arm (build docs) | Robot Studio / HF | Hardware, ~$100–300 | https://huggingface.co/docs/lerobot/so101 |
| RT-2 (web-scale VLM → robot actions) | Google DeepMind | Blog + paper, ~1 hr | https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/ |
| OpenVLA | Stanford et al. | Project page + paper, ~2 hr | https://openvla.github.io/ |
| π0 / openpi (open VLA foundation model) | Physical Intelligence | Open weights + code | https://github.com/Physical-Intelligence/openpi |
| π0 in LeRobot | Hugging Face | Docs | https://huggingface.co/docs/lerobot/pi0 |
| MuJoCo (physics sim) | Google DeepMind | Simulator | https://mujoco.org/ |
| Sim-to-real SO-101 tutorial (Isaac ↔ LeRobot) | NVIDIA | Free tutorial | https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/ |
| Robotic Manipulation (rigor track) | Russ Tedrake (MIT) | Free book + lectures, 50+ hr | https://manipulation.csail.mit.edu/ |
| Learning and Control (state-of-the-field essays) | Sergey Levine | Substack | https://sergeylevine.substack.com/ |
| Physical Intelligence blog | Physical Intelligence | Blog | https://www.physicalintelligence.company/blog |

### 23. Track F — Causal Reasoning & Experimentation

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| The Book of Why | Pearl & Mackenzie | Book ~$15, ~15 hr | https://www.amazon.com/Book-Why-Science-Cause-Effect/dp/046509760X |
| **Causal Inference for the Brave and True** ⭐ | Matheus Facure | Free Python-native book, 30–50 hr | https://matheusfacure.github.io/python-causality-handbook/ |
| Causal Inference: The Mixtape | Scott Cunningham | Free online book (econometrics flavor) | https://mixtape.scunning.com/ |
| Trustworthy Online Controlled Experiments | Kohavi, Tang, Xu | Book ~$35 (ch. 1 free at companion site) | https://experimentguide.com/ |
| DoWhy + EconML (PyWhy) | Microsoft Research | Library + tutorial notebooks, 5–10 hr | https://www.pywhy.org/dowhy/ |

### 24. Track G — Alignment & Interpretability

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **AI Safety Fundamentals — Alignment Course** ⭐ | BlueDot Impact | Free course, ~4–5 wks of reading | https://aisafetyfundamentals.com/alignment/ |
| Mapping the Mind of a Large Language Model | Anthropic | Article, ~45 min | https://www.anthropic.com/research/mapping-mind-language-model |
| Getting Started in Mechanistic Interpretability (the honest on-ramp) | Neel Nanda | Guide, 1–2 hr | https://www.neelnanda.io/mechanistic-interpretability/getting-started |
| **ARENA Ch. 1: Transformer Interpretability** ⭐ (the actual hands-on curriculum) | McDougall / Nanda | Hands-on, 20–40 hr | https://learn.arena.education/chapter1_transformer_interp/ |
| Transformer Circuits (primary sources) | Anthropic | Research site | https://transformer-circuits.pub/ |
| Introspection in LLMs (where the frontier is) | Anthropic | Research post, ~45 min | https://www.anthropic.com/research/introspection |
| Alignment Forum (the field's working discourse — read critically) | Community | Site | https://www.alignmentforum.org/ |

### 25. Phase 4.5 — Production AI Systems

*Mostly regrouped from section 11 (that's deliberate — the phase applies material you've already met), plus durable general-purpose tools. This section will churn fastest; expect to swap tools, not concepts.*

| Resource | Author | Format / Time | Link |
|---|---|---|---|
| **AI Engineering — serving, optimization & ops chapters** ⭐ | Chip Huyen | Book chapters (re-read as operator) | https://github.com/chiphuyen/aie-book |
| Prompt caching docs (see §11 note on true pricing) | Anthropic | Docs, 1 hr | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| LLM Evals FAQ — the CI/production-integration answers | Husain & Shankar | FAQ sections | https://hamel.dev/blog/posts/evals-faq/ |
| LangSmith evaluation & monitoring concepts | LangChain | Docs (vocabulary transfers to Braintrust/Phoenix) | https://docs.langchain.com/langsmith/evaluation-concepts |
| OpenTelemetry (structured logs/traces/metrics — the durable observability vocabulary) | CNCF | Docs, survey the concepts pages | https://opentelemetry.io/docs/ |
| k6 (load testing your endpoints) | Grafana Labs | Tool + docs, ~2 hr to first test | https://k6.io/ |

---

**Budget, honestly:** ~$150 in books if you buy all three core ones (AI Engineering, PMPP, Raschka's LLM book) + possibly ~$49 for one Coursera month in Week 3 + ~$15–50 for Part 2 books (Book of Why; Kohavi optional) + realistically **$60–180 in GPU rental & API credits** for projects (incl. ~$10–30 for Track E sim work and ~$5–20 for Week 15's GRPO run) + optional ~$100 nanochat training run + optional ~$100–300 SO-101 robot arm. Everything else is free.
