# SG-MCAE: Spec-Grounded Monte-Carlo Agent Evolution

A framework that automatically evolves AI Agent code using product flowcharts as formal behavioral specifications.

## Overview

SG-MCAE takes a product flowchart (PDF/PNG) and PRD document, then runs a closed-loop cycle of **Evaluate → Locate → Patch → Verify → Backpropagate** on the target Agent codebase until it achieves full behavioral alignment with the spec.

## Pipeline

```
whiteboard.pdf ──► FlowParser ──► FlowGraph (Formal Spec IR)
     + PRD                            │
                                      ▼ enumerate_paths() → 500+ test paths
                                      │
                         ┌────────────┴────────────┐
                         │  MCTS Evolution Loop     │
                         │  Select ─► Expand ─► Simulate ─► Backpropagate │
                         └─────────────────────────┘
                                       │
                                      Agent code (git-versioned)
```

## Key Features

- **Flowchart as Formal Spec** — Parses product flowcharts into a directed graph IR with guards and counter thresholds. Enumerates 500+ complete test paths including loop-3-times meltdown paths that unit tests can't cover.
- **MCTS on Version Tree** — UCT + progressive widening on git commit DAG. Every tree node maps to a git commit for precise rollback. Degradation penalty (child < parent → score × 0.5) prevents oscillating patches.
- **Dual-Channel Reward** — Semantic scoring (LLM-as-Judge: node coverage, guard correctness, terminal match) + Deterministic hard checks (counter thresholds, terminal whitelist). 0.2 × worst-path regularization prevents over-optimizing trunk paths.
- **Fully Auditable** — Native git backtracking, evolution chain reproducible, human-intervenable cherry-pick.

## Quick Start

```bash
pip install requests pymupdf

export LLM_API_KEY=sk-...
export LLM_MODEL=deepseek-chat

# Parse flowchart into formal spec
python main.py parse --flow ../flowchart.pdf --prd ../prd.md

# Baseline evaluation
python main.py assess --repo ..

# MCTS evolution
python main.py evolve --repo .. --iterations 20
```

## Architecture

| File | Purpose |
|------|---------|
| `flow_parser.py` | PDF/image → FlowGraph IR |
| `flow_graph.py` | Graph IR, cyclic path enumeration, spec text generation |
| `spec_validator.py` | Spec consistency checking |
| `spec_overrides.py` | Manual spec overrides |
| `simulator.py` | LLM-as-Driver: path-guided multi-turn simulation |
| `evaluator.py` | LLM-as-Judge + deterministic counter checks |
| `patcher.py` | Candidate patch generation, apply, git rollback |
| `mcts_evolver.py` | UCT search loop, progressive widening, degradation penalty |
| `search_model.py` | Search space modeling |
| `fsm_reward.py` | FSM-based reward computation |
| `llm_client.py` | Unified LLM API client |
| `main.py` | CLI: `parse` / `assess` / `evolve` |
