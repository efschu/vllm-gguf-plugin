# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## This fork: Qwen3.5/3.6 support (branch `qwen35-support`)

Additions on top of upstream:

- **Qwen3.5/3.6 (dense + MoE) GGUF support** — full weight adapter incl.
  inversion of the llama.cpp conversion transforms (Gemma-style norms,
  `ssm_a`, GDN v-head retiling). All GGUF quant types; K-quants with
  misaligned packed columns fall back to dequant transparently.
- **MTP speculative decoding from the GGUF** — the file's own MTP layer is
  mapped as draft model (`--speculative-config '{"method":"mtp",...}'`).
- **Vision**: drop the matching `mmproj-*.gguf` next to the model file and
  multimodality is enabled automatically (image inputs via OpenAI API).
- **~8× faster prefill** for all GGUF models: MMQ kernels are only used up to
  `VLLM_GGUF_MMQ_MAX_TOKENS` (default 16) tokens; larger batches dequantize
  and use cuBLAS.
- Loader fixes: tuple shard ids (hybrid/GDN models), a ~6 GiB/rank load-time
  memory leak, embedding dequant fallback, newer-vLLM compatibility.

Usage: place `config.json` + tokenizer files (from the original HF repo) next
to the `.gguf`, then `vllm serve /path/to/model.gguf`. Requires the
[shvllm fork](https://github.com/efschu/shvllm/tree/qwen35-gguf-rankgpu) (or a
vLLM with its Qwen3.5 core fixes) for this model family; other architectures
work with stock vLLM. A prebuilt Docker image with both is available:
`ghcr.io/efschu/shvllm-qwen35-gguf:cu129`.

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. Install the plugin in development mode:

   ```bash
   uv pip install -e . --torch-backend=auto
   ```

Or install directly:

```bash
uv pip install . --torch-backend=auto
```

## Development

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```
