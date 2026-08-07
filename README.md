# Full-Stack AI Learning Roadmap

A complete curriculum for going from vibe-coding AI apps to understanding the entire stack — what models are, how they're built and trained, how they're made fast, how they're served from data centers, what's actually going on inside them, and the wider AI landscape beyond LLMs.

**It is not a single line.** Phases 0→3 are a genuine chain — each needs the one before. After that the plan forks into two descents you can take in either order:

```
0 → 1 → 2 → 3 ─┬─→ ↑ Phase 4 → 4.5    products, evals, production
               └─→ ↓ Phase 5 → 6      kernels, clusters, data centers
```

Phase 4 doesn't need Phase 3, and Phase 5 doesn't need Phase 4. Take the descent that pulls you.

## Contents

| File | What it is |
|---|---|
| **[ROADMAP.md](ROADMAP.md)** | The plan. A **spine** (Phases 0→3, in order), then **two independent descents** — ↑ products (Phase 4 → 4.5) and ↓ metal (Phase 5 → 6) — then **7 breadth tracks**. Numbered steps, not weeks: it's a map, not a calendar. Core/Survey/Reference tags and a milestone project per phase. |
| **[RESOURCES.md](RESOURCES.md)** | The library: ~212 curated, link-verified resources across 25 sections. |
| **[roadmap.html](roadmap.html)** | Interactive tracker: open locally in any browser — checkable tasks with progress saved in localStorage, every resource one click away. |

## How this was built

Researched and assembled with Claude (Aug 2026): parallel research agents gathered and link-verified resources per stack layer, then the plan went through four rounds of adversarial review — schedule realism, data/eval discipline, production coverage, and frontier currency — with every factual claim independently verified. A fifth structural review (Aug 2026) audited it for coverage gaps rather than dead links — adding the modern-architecture/scaling-laws, RLVR, coding-agents and training-practice steps plus Track G — then reshaped it from a 34-week line into a spine plus two independent descents, and refreshed the fast-moving sections to mid-2026 material.

## Keeping it current

The roadmap uses a **pre-phase audit protocol** (rule 9 in ROADMAP.md): before starting any phase, its section gets re-audited — links verified, availability re-checked, genuinely-new material added. Critically, the audit asks whether the *practice* changed, not just whether the link resolves: the real decay mode is a resource that loads perfectly while teaching a superseded default. Fast-moving sections (architecture, RLVR, AI engineering, inference, vision, robotics, alignment) get full audits; stable foundations (math, backprop, Karpathy) don't need them. Last full revision: **August 2026**.

## The one-sentence version

Watch/read → then build. Core > Survey > Reference. Projects > completion. Climb the spine, then pick your descent. Start with [Karpathy's 1-hour intro](https://www.youtube.com/watch?v=zjkBMFhNj_g).
