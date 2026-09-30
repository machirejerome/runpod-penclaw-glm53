# Penclaw GLM-5.3 on RunPod

This project contains the reproducible preprocessing and serving path for a
private `Q4_K_M` quantization of `audnai/penclaw-GLM-5.3-abliterated`.

The production design deliberately keeps model weights out of the container:

- a one-time Pod converts the gated BF16 checkpoint to a sharded GGUF and
  quantizes it with a published GLM-5.3 importance matrix;
- the resulting shards are uploaded to a private, single-quant Hugging Face
  repository;
- RunPod Cached Models stages that repository on local worker storage;
- a small llama.cpp image serves an OpenAI-compatible API through a `/ping`
  health proxy.

The target Serverless worker uses four H200 GPUs, starts with 131,072 tokens of
context, Q8 KV cache, one concurrent slot, FlashAttention, and scale-to-zero.

No model weights or credentials are committed to this repository.
