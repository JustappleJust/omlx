# Image generation

oMLX can serve image models locally on Apple Silicon through
[mflux](https://github.com/mflux-community/mflux). The API is OpenAI-compatible,
so existing image clients work against the same server and the same API key as
chat, embeddings, and audio.

Image generation is an optional extra: nothing in the server imports MLX image
code until the first image request, and the rest of oMLX runs unchanged if
mflux is not installed.

```bash
pip install "omlx[image]"     # or: make image
```

mflux pins `mlx>=0.32,<0.33` (matching oMLX's own pin), so the extra installs
cleanly next to the server. It also depends on PyTorch (used by its export
utilities, not by MLX inference), which adds roughly 2 GB to the environment;
the model weights themselves are the larger cost below.

## Supported models

| Model | Family | Steps (default) | Guidance | Native size | Editing |
|---|---|---|---|---|---|
| `flux-3-image-klein` | FLUX 3 Image Klein (distilled, 5.6B + Qwen3-VL-4B) | 16 | none | 1024², multiple of 16 | up to 10 reference images |
| `flux-3-image-klein-base` | FLUX 3 Image Klein Base (undistilled) | 50 | 3.5 | 1024², multiple of 16 | up to 10 reference images |
| `krea-2-turbo` | Krea 2 Turbo (12.9B MMDiT + Qwen3-VL-4B) | 8 | 1.0 | 1024²–2048² | strength-based img2img, one image |
| `qwen-image-2.1-turbo` | Qwen-Image-2.1-Turbo (7B DiT + Qwen3-VL-8B) | 8 | 1.0 | 1024² and up | up to 10 reference images, masks, alpha |
| `qwen-image-2.1` | Qwen-Image-2.1 (undistilled) | 40 | 1.0 | 1024² and up | up to 10 reference images, masks, alpha |

Defaults follow the vendor model cards and can be overridden per request with
`steps`, `guidance`, and `size` (or `width`/`height`). Restoring the historical
values is a single field away in any client.

Models are detected from the model directory, so they appear in `/v1/models`
and the admin dashboard like every other checkpoint. Two layouts work:

* a folder written by `mflux-save` (per-component subdirectories,
  `mflux_model.json`, `<n>.safetensors` shards, e.g.
  `Krea-2-Turbo-mflux-q8/`), or
* an upstream diffusers snapshot with a `model_index.json`, including one
  downloaded into the Hugging Face cache.

Unrelated diffusion checkpoints (Stable Diffusion, FLUX.1, …) are ignored
rather than listed as servable, because mflux does not run them.

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/images/generations` | Text-to-image (JSON body) |
| `POST /v1/images/edits` | Reference editing / img2img (multipart) |
| `GET /v1/images/generated/{id}` | Fetch an image returned with `response_format="url"` |

### Generate

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Krea-2-Turbo-mflux-q8",
    "prompt": "a red fox in fresh snow, soft morning light, 35mm photograph",
    "size": "1024x1024",
    "seed": 42
  }'
```

The response is OpenAI's shape: `data[0].b64_json` (default) or
`data[0].url` when `response_format` is `"url"`. `created`, `model` and the
effective `seed` are included; pass the seed back to reproduce an image.

### Edit

```bash
curl http://localhost:8000/v1/images/edits \
  -H "Authorization: Bearer $OMLX_API_KEY" \
  -F model=Qwen-Image-2.1-Turbo-mflux-q8 \
  -F prompt="replace the sky with a sunset" \
  -F image=@room.png \
  -F image=@sky-reference.png
```

`image` may be repeated (or sent as `image[]`) for multi-reference editing.
`mask` is accepted by Qwen-Image-2.1 only; Krea 2 ignores instruction editing
and runs strength-based img2img on a single reference image (`strength`,
default 0.65).

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-key")

image = client.images.generate(
    model="Krea-2-Turbo-mflux-q8",
    prompt="a red fox in fresh snow",
    size="1024x1024",
    extra_body={"steps": 8, "guidance": 1.0, "seed": 42},
)
```

## Request fields

Standard OpenAI fields: `model`, `prompt`, `n` (1–4), `size`
(`"auto"`/`"<width>x<height>"`), `response_format`
(`b64_json` | `url`), `output_format` (`png` | `jpeg` | `webp`),
`background` (`transparent` is accepted only by families that emit alpha),
and `quality`/`style`/`moderation`/`user` (accepted and ignored).

With `response_format="url"` the image is kept in memory for one hour and
served from `GET /v1/images/generated/{id}`. That URL is part of the
authenticated API, so fetch it with the same `Authorization` header; ask for
`b64_json` when you need bytes that outlive the server process.

oMLX extensions: `width`/`height` (instead of `size`), `steps`, `guidance`,
`seed`, `negative_prompt`, and `strength` (edits).

Validation is family-aware and returns `400` with an explanation:

* size must be a multiple of 16 and within 256–4096;
* FLUX 3 Klein uses no classifier-free guidance, so `guidance` is rejected
  (omit it, or pass `0`) unless you run the Base checkpoint;
* Qwen-Image-2.1-Turbo samples at guidance 1.0, so other values are rejected;
* Krea 2 rejects multi-reference edits and masks;
* `n > 1` is rejected for reference editing, because each reference image
  produces exactly one output.

## Memory and performance

One image pipeline is resident per model. mflux's text-to-image and edit
pipelines are separate classes that each hold a full copy of the weights, so
mixing generation and editing requests on the same model reloads the
pipeline (logged at INFO). Sizing guidance:

| Model | Approximate weights |
|---|---|
| FLUX 3 Image Klein (BF16) | 11.2 GB |
| FLUX 3 Image Klein (FP8) | 5.6 GB |
| Krea 2 Turbo | ~26 GB at BF16, ~13 GB at 8-bit |
| Qwen-Image-2.1-Turbo | ~14 GB at 8-bit plus the Qwen3-VL-8B text encoder |

Prefer an 8-bit `mflux-save` checkpoint on 32–64 GB machines; the first request
after loading also compiles Metal kernels, so it is slower than steady state.
Image work runs on the shared MLX executor, so it serializes with other MLX
traffic instead of competing for the GPU.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `503` "optional mflux runtime is not installed" | `pip install "omlx[image]"`, then restart the server |
| `404` "no image models are installed" | the model directory holds no supported checkpoint |
| `400` "matches several loaded checkpoints" | two checkpoints of one family are installed; pass the directory name |
| Output looks muddled at low step counts | distilled checkpoints expect the vendor step count (16 / 8 / 8) |
| Very slow first request | Metal kernel compilation after load; later requests are much faster |

mflux requires macOS 15+ on Apple Silicon. It pins its own MLX version, so
install the `[image]` extra into a virtual environment shared with the server
rather than into the macOS app bundle.
