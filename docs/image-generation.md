# Image generation

oMLX serves text-to-image diffusion models next to LLMs. They share the
engine pool and the process memory enforcer with every other model type:
loading goes through admission control, idle models unload on TTL, models can
be pinned, and an image model is an ordinary LRU eviction candidate when it
is not generating.

Supported families:

| Family | Checkpoints | Generate | Edit | Defaults |
|---|---|---|---|---|
| Krea 2 | `krea/Krea-2-Turbo`, `krea/Krea-2-Raw`, `mflux-community/krea-2-turbo-mflux-*` | yes | no | 8 steps, 16 px grid |
| FLUX.3 Image Klein | `black-forest-labs/flux-3-image-klein` (and `-base`) | yes | up to 10 references, no mask | 16 steps, 16 px grid |
| Qwen-Image 2.1 | `Qwen/Qwen-Image-2.1-Turbo`, `Qwen/Qwen-Image-2.1` | yes | up to 10 references, mask, strength | 8 steps, 32 px grid |

All three run on mflux's native MLX implementations. Each model is loaded
once and serves both generation and editing.

Install the optional dependency:

```bash
pip install "omlx[image]"
```

## Getting a model

Any of these layouts is detected automatically, both in the model
directories and in the Hugging Face cache:

- a diffusers repo with a `model_index.json` (`_class_name: Krea2Pipeline`);
- what mflux downloads itself (`turbo.safetensors` + `text_encoder/` + `vae/` + `tokenizer/`);
- an mflux-saved or mflux-community quantized directory (root shards + `text_encoder/` + `vae/`).

For other converted layouts, add `omlx_image.json` with `{"adapter": "krea2"}`.
Incomplete downloads (missing components or shards) are skipped with a warning.

```bash
# The Krea 2 repos are gated: accept the license on Hugging Face first.
hf auth login
hf download krea/Krea-2-Turbo --exclude "transformer/*" "images/*" \
  --local-dir ~/.omlx/models/Krea-2-Turbo
```

Excluding `transformer/` skips the duplicate diffusers copy of the weights
(26 GB). The loader uses `turbo.safetensors`.

## API

`POST /v1/images/generations` (OpenAI-compatible):

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "Krea-2-Turbo", "prompt": "a fox in the snow", "size": "1024x1024"}' \
  | jq -r '.data[0].b64_json' | base64 -d > fox.png
```

| Field | Default | Notes |
|---|---|---|
| `model`, `prompt` | required | |
| `n` | 1 | 1-8; seeds are `seed`, `seed+1`, ... |
| `size` | model default (1024x1024) | `WIDTHxHEIGHT`; snapped to a multiple of 16, 256-2048 per side |
| `response_format` | `b64_json` | `url` returns a `data:` URL |
| `output_format` | `png` | `png`, `jpeg`, `webp` |
| `seed` | random | returned per image |
| `steps` / `num_inference_steps` | 8 | |
| `guidance` / `guidance_scale` | 1.0 | mflux semantics: 1.0 disables CFG (Turbo is distilled) |
| `negative_prompt` | none | only used when guidance ≠ 1.0 |
| `scheduler` | `er_sde` | Krea 2: `er_sde` or `euler` |

Unknown OpenAI fields (`quality`, `style`, `user`) are ignored. The response
includes `size`, per-image `seed`, and `usage` (`steps`, `seconds`).

Errors: 404 for an unknown model, 400 when the model isn't an image model,
422 for invalid parameters, 507 when memory is insufficient, 409 while the
model is loading or busy.

`GET /v1/images/models` lists image models with their defaults and
`capabilities` (edit, maximum references, mask, strength, and the optional
lossy accelerations).

### Editing

`POST /v1/images/edits` follows the OpenAI shape and accepts two body formats:

- **multipart/form-data:** `image` or `image[]` files, an optional `mask` file
  (white = repaint), plus the same fields as generations.
- **JSON:** `images` (base64 strings or `data:` URLs) and an optional `mask`.

Other fields:

| Field | Notes |
|---|---|
| `strength` | 0–1 (Qwen-Image only). Below 1, the edit starts from the source image instead of pure noise. |
| `size` | If omitted, the output follows the reference image, aligned to the model's grid. |

```bash
curl http://localhost:8000/v1/images/edits -H "Authorization: Bearer $KEY" \
  -F model=Qwen--Qwen-Image-2.1-Turbo -F "prompt=make the teapot blue" \
  -F "image[]=@teapot.png" | jq -r '.data[0].b64_json' | base64 -d > blue.png
```

Input limits:

- 32 MB per image.
- 64 MP decoded; larger images are rejected before any pixels are decoded.
- At most 16 files per request; each model enforces its own lower reference limit.
- EXIF orientation is applied.

Qwen-Image outputs RGBA. PNG and WebP keep a non-opaque alpha channel; JPEG
is flattened.

## Per-model settings

You can set these in the admin dashboard under Model settings, or in
`model_settings.json`:

| Setting | Default | Notes |
|---|---|---|
| `image_quantize` | 8 | Load-time weight quantization of the transformer: `0` = bf16, or `3/4/5/6/8`. The text encoder stays bf16. Changing it reloads the model. Pre-quantized checkpoints keep their stored bits. |
| `image_default_steps` | adapter default | |
| `image_default_guidance` | adapter default | |
| `image_default_size` | adapter default | `WIDTHxHEIGHT` |
| `image_a8` | on | INT8-activation GEMM (near-lossless, M5 only). Changing it reloads the model. |
| `image_step_cache` | off | Opt-in lossy step skipping (Qwen-Image only). A request can also set `accel: {"step_cache": true}`. |

## Speed optimizations

These numbers were measured with `scripts/bench_image.py` on an M5 Max
(1024², 8-bit weights, default steps, model kept loaded).

| Model | mflux as shipped | oMLX | Speedup | Near-lossless optimizations applied |
|---|---|---|---|---|
| Krea 2 Turbo, generate | 3.33 s/step (28.0 s) | 1.70 s/step (15.0 s) | **1.96×** | bf16 DiT activations + A8 |
| FLUX.3 Klein, generate | ~0.96 s/step (~16 s) | 0.67 s/step (11.6 s) | **~1.42×** | A8 + per-block compile |
| FLUX.3 Klein, edit (1 reference) | ~2.08 s/step (~34 s) | 1.63 s/step (27.0 s) | ~1.27× | A8 + per-block compile + reference cache |
| Qwen-Image 2.1 Turbo, generate | 1.50 s/step (13.2 s) | 1.15 s/step (10.4 s) | **1.31×** | A8 |
| Qwen-Image 2.1 Turbo, edit (1 reference) | 1.68 s/step (16.4 s) | 1.35 s/step (13.3 s) | 1.25× | A8 + reference caches |

Klein's "as shipped" numbers are estimated from runs that overlapped with
other GPU work, so treat them as approximate.

- **A8 (INT8-activation GEMM).** DiT projections run on thousands of tokens,
  which is the same shape as LLM prefill. oMLX routes them through its M5
  tensor-unit INT8 kernel (`omlx.custom_kernels.quant_matmul`, the same idea
  as vpipe's `i8_gemm`).
  - Each projection matches the stock kernel at cosine ≥ 0.9999.
  - Every distinct (bits, K, N) shape is timed once at load and only routed
    when it is faster.
  - The module is `omlx/image_gen/accel/a8.py`. On machines without NAX it is
    disabled automatically.
- **bf16 activations (Krea 2).** mflux fed float32 latents into the DiT,
  which ran every block in float32. Like the reference diffusers pipeline,
  the DiT now runs in bf16 while the sampler state stays float32.
- **Per-block compile (FLUX.3).** mflux can't compile FLUX.3's step because
  RoPE is computed on the host. Each block is compiled instead, which fuses
  the modulation, normalization and RoPE elementwise chains
  (`omlx/image_gen/accel/compile.py`).
- **Cross-request caches.** These cover text embeddings, VAE-encoded
  references (FLUX.3 and Qwen), and Qwen's reference-conditioned prompt
  encodings.
  - All are bounded LRUs, released on unload and reported in `get_stats()`.
  - Repeated edits of the same image skip encoding (about 1 s on Qwen).
- **Already in mflux:** Qwen-Image's text/reference prefix K/V reuse across
  steps, and Krea's whole-step compile.

**Evaluated but not adopted:**

- **Caching FLUX.3's text and reference stacks across steps.** It isn't
  exact for the distilled checkpoint, because its flow-map target time
  conditions every mode.
- **SageAttention and sparse attention.** They need new Metal kernels and
  vpipe reports only 1.02–1.24× for them.

Lossy step skipping is available for Qwen-Image as an opt-in (see settings).

## Memory behaviour

Each adapter describes **memory plans** for a request size: execution modes
ordered fastest first, each with a calibrated transient peak. Before every
generation the engine picks the first plan whose peak fits the budget under
the enforcer's soft limit, and returns 507 if none fits. While generating, an
owner-scoped MLX memory limit holds the process at that watermark.

Krea 2 numbers, measured on an M5 Max with an 8-bit transformer:

| | Resident | 1024² peak (extra) | 2048² peak (extra) |
|---|---|---|---|
| Weights, bf16 checkpoint quantized to 8-bit | 22.2 GB | | |
| Untiled VAE decode | | 15.6 GB | ~24 GB |
| 512 px tiled decode | | 7.9 GB | 7.9 GB |
| 256 px tiled decode | | 3.7 GB | 3.7 GB |

Time per 1024² image at 8 steps is about 28 s.

Admission control reserves the weights plus the leanest plan, so loading an
image model doesn't evict other models just to make room for the faster
plans. Generations run one at a time per model. Text-encoder outputs are
cached in a small LRU (8 prompts), and the Metal buffer pool is cleared after
every request and on unload.

## Adding a model family

1. Implement `omlx.image_gen.base.ImageModelAdapter` in a new module under
   `omlx/image_gen/`. You need to provide:
   - detection (`matches`)
   - `component_files`
   - `memory_plans`
   - `load`
   - `generate`
   - optionally `capabilities`, `edit_canvas`, `is_prequantized`,
     `quantizable_components`, `release` and `runtime_stats`

   For mflux-backed families, call `_mflux.apply_load_accel()` in `load` to
   get A8 and eager weight loading.
2. Append an instance to `ADAPTERS` in `omlx/image_gen/registry.py`.

Nothing else changes: discovery, the engine, the pool, the API and the
settings are all family-agnostic.
