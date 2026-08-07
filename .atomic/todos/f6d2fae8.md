{
  "id": "f6d2fae8",
  "title": "Roadmap Aug-2026 review: apply full diff on branch roadmap-2026-08-review",
  "tags": [
    "roadmap",
    "docs"
  ],
  "status": "in_progress",
  "created_at": "2026-08-07T14:54:10.775Z"
}

## New Part 1 structure (30 -> 34 units)
- P0: W1 | P1: W2-4 | P2: W5-8
- **P3: W9-15** (was 9-13): 9 pretrain recipe; **10 NEW modern arch + scaling laws**; 11 data; 12 evals; 13 fine-tuning; 14 post-training SFT/DPO; **15 NEW reasoning + RLVR/GRPO**
- **P4: W16-22** (was 14-19): 16 mental model; **17 NEW coding agents**; 18 prompting; 19 RAG; 20 agents+MCP; 21 evals+obs; 22 security
- **P4.5: W23-25** (was 20-22)
- **P5: W26-29** (was 23-26)
- **P6: W30-34** (was 27-30): 30 distributed theory; **31 NEW practitioner reality (ml-engineering, smol playbook, FP8, NCCL)**; 32 inference (+SGLang, prefill/decode); 33 serve capstone; 34 datacenters (+JAX scaling book, TPU)
- Track D stays 6 weeks (user decision)
- New Track G: Alignment & Interpretability (flag as beyond-letter-of-diff)

## Tasks
1. ROADMAP.md: new weeks, renumber, fix ALL cross-refs, time budget, rule 8 obsolescence check, retention ritual, dependencies+budget
2. RESOURCES.md: new sections (modern arch/scaling laws, RLVR, agentic coding, training practice, alignment) + SGLang/JAX/TPU entries
3. README.md: counts, date, task count
4. roadmap.html: mirror all changes + FIX positional task IDs (line 805) -> stable data-id slugs
5. Verify: link check all new URLs, week-number consistency, HTML/MD parity, adversarial review
