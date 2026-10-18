# Codemia Notebooks

Practice notebooks for the Codemia learning platform. Each notebook is the executable companion to a lesson, designed to be opened directly in Google Colab via the URL pattern:

```
https://colab.research.google.com/github/Codemia-io/codemia-notebooks/blob/main/<path>.ipynb
```

## Layout

- `foundations/` — math + neural net basics (autograd, MLP, training dynamics, regularization, CNNs, embeddings)
- `pretraining/` — pretraining objectives, tiny GPT from scratch
- `attention/` — attention mechanisms, transformer blocks, positional encoding, efficient attention, beyond transformers
- `finetuning/` — full fine-tune, LoRA, instruction tuning, RLHF, DPO
- `inference/` — quantization, paged attention, speculative decoding
- `training/` — distributed training (FSDP, ZeRO)
- `retrieval/` — vector search (HNSW from scratch), advanced RAG
- `interpretability/` — induction heads with TransformerLens, activation patching
- `research/` — paper reproduction template
- `networking/` (no GPU, numpy and matplotlib only): bandwidth-delay product and window sizing, congestion-window trajectories, latency percentiles, DNS TTL and cache hit rate, connection capacity and port exhaustion, the tail-latency cost of a retransmission

## Notebook conventions

Every notebook follows the same structure:

1. **Title cell** — markdown with lesson name + link back to the codemia lesson page
2. **Setup cell** — `pip install` and imports
3. **Concept recap** — 1-2 markdown paragraphs framing what we are about to build
4. **Implementation** — code cells with dense comments, building up the technique step by step
5. **Experiment** — run the implementation on a small example, print/visualize results
6. **Variations** — try one or two variations to build intuition (different hyperparameters, different inputs, ablations)
7. **What to try next** — suggestions for the reader to extend the notebook

## Code style

- Use `numpy` and `torch` (sometimes `jax`). Avoid heavy dependencies unless the lesson is about them.
- Pin small models (gpt2, distilbert, llama-3-1B) for anything that touches a real LLM.
- Comments should explain WHY, not WHAT. Each non-obvious line gets a comment.
- Print intermediate values aggressively. Notebooks teach by showing.
- Keep cells short (under 30 lines). Long cells become un-debuggable.

## Karpathy attribution

Many notebooks adapt code from Andrej Karpathy's open-source educational repos:

- `karpathy/micrograd` — autograd from scratch (MIT)
- `karpathy/makemore` — character-level language models (MIT)
- `karpathy/nanoGPT` — minimal GPT implementation (MIT)
- `karpathy/build-nanogpt` — full GPT-2 reproduction (MIT)

When a notebook adapts code from these repos, the title cell includes a link to the source and a credit line.

## Requirements

Most notebooks need only `numpy`, `torch`, and `matplotlib`. Notebooks that touch real LLMs additionally need `transformers`, `accelerate`, sometimes `peft` or `trl`. The free Colab tier (T4 GPU, 12GB VRAM) is sufficient for everything except the largest fine-tuning experiments — those notebooks include a note saying so.
