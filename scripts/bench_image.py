#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark oMLX image-generation adapters (speed, memory, fidelity).

Loads each model once (as the server does), warms up, then times text-to-image
and optional editing runs. Per-step timestamps from the progress callback
split the time into denoise (steps) and the rest (prompt/reference encoding +
VAE decode). Images are saved so runs with different settings can be compared
(PSNR against a reference directory).

Examples:
  python scripts/bench_image.py --model ~/.cache/.../flux-3-image-klein/snapshots/<rev> \\
      --out /tmp/bench/klein-base --edit
  python scripts/bench_image.py --model <path> --accel a8=0 --out /tmp/bench/klein-noa8 \\
      --compare /tmp/bench/klein-base
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROMPT = "a red fox sitting in fresh snow at golden hour, detailed fur, photorealistic"
EDIT_PROMPT = "turn the scene into a watercolor painting, keep the composition"


def _psnr(a, b) -> float:
    import numpy as np

    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def _parse_accel(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = value.lower() not in ("0", "false", "off", "no")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantize", type=int, default=8, help="0 = bf16")
    ap.add_argument("--accel", action="append", default=[], help="engine accel option, e.g. a8=0")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edit", action="store_true", help="also benchmark editing (needs an edit-capable model)")
    ap.add_argument("--edit-image", default=None, help="reference image (default: the generated image)")
    ap.add_argument("--step-cache", action="store_true", help="request the opt-in step cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="directory of a previous run to PSNR against")
    args = ap.parse_args()

    import mlx.core as mx
    from PIL import Image

    from omlx.image_gen import find_adapter

    path = Path(args.model).expanduser()
    adapter = find_adapter(path)
    if adapter is None:
        print(f"no image adapter matches {path}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.lower().split("x"))
    accel = {"a8": True, **_parse_accel(args.accel)}
    quantize = args.quantize or None
    if quantize and adapter.is_prequantized(path):
        quantize = None

    t0 = time.monotonic()
    model = adapter.load(path, quantize, accel)
    mx.synchronize()
    load_s = time.monotonic() - t0
    resident = mx.get_active_memory()
    print(f"[{adapter.name}] load {load_s:.1f}s, resident {resident / 1e9:.2f} GB, accel={accel}", flush=True)

    from omlx.image_gen.base import ImageGenParams

    def run(name: str, params: ImageGenParams):
        stamps: list[float] = []

        def on_step(i, total):
            mx.synchronize()
            stamps.append(time.monotonic())

        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        start = time.monotonic()
        image = adapter.generate(model, params, on_step, None)
        mx.synchronize()
        total = time.monotonic() - start
        steps = [b - a for a, b in zip(stamps, stamps[1:])]
        denoise = (stamps[-1] - stamps[0]) + (statistics.median(steps) if steps else 0) if stamps else 0.0
        rec = {
            "name": name,
            "total_s": round(total, 3),
            "step_s": round(statistics.median(steps), 4) if steps else None,
            "denoise_s": round(denoise, 3),
            "other_s": round(total - denoise, 3),
            "peak_extra_gb": round((mx.get_peak_memory() - resident) / 1e9, 2),
        }
        return image, rec

    d = adapter.defaults
    steps = args.steps or d.steps
    base = dict(prompt=PROMPT, width=w, height=h, steps=steps, guidance=d.guidance, seed=args.seed)
    if args.step_cache:
        base["accel"] = {"step_cache": True}
    records = []

    jobs = [("t2i", ImageGenParams(**base))]
    if args.edit and adapter.capabilities.edit:
        ref_path = Path(args.edit_image) if args.edit_image else None
        jobs.append(("edit", None))
    else:
        ref_path = None

    for name, params in jobs:
        if name == "edit":
            ref = Image.open(ref_path) if ref_path else Image.open(out / "t2i.png")
            ew, eh = adapter.edit_canvas([ref])
            params = ImageGenParams(**{**base, "prompt": EDIT_PROMPT, "width": ew, "height": eh}, references=[ref])
        # warm-up (compiles kernels / fills caches), then timed repeats
        image, rec = run(name + ":warmup", params)
        records.append(rec)
        print(json.dumps(rec), flush=True)
        for i in range(args.repeat):
            image, rec = run(name, params)
            records.append(rec)
            print(json.dumps(rec), flush=True)
        image.save(out / f"{name}.png")
        if args.compare and (Path(args.compare) / f"{name}.png").exists():
            psnr = _psnr(image, Image.open(Path(args.compare) / f"{name}.png"))
            print(f"  {name}: PSNR vs {args.compare}: {psnr:.2f} dB", flush=True)
            records.append({"name": name + ":psnr", "psnr_db": round(psnr, 2)})

    summary = {"adapter": adapter.name, "load_s": round(load_s, 2), "resident_gb": round(resident / 1e9, 2),
               "accel": accel, "quantize": quantize, "size": args.size, "steps": steps,
               "stats": adapter.runtime_stats(model), "runs": records}
    (out / "bench.json").write_text(json.dumps(summary, indent=2, default=str))
    timed = {}
    for r in records:
        if ":" not in r["name"]:
            timed.setdefault(r["name"], []).append(r)
    for name, rs in timed.items():
        print(
            f"== {adapter.name} {name}: total {statistics.median(x['total_s'] for x in rs):.2f}s "
            f"step {statistics.median(x['step_s'] for x in rs if x['step_s']):.3f}s "
            f"other {statistics.median(x['other_s'] for x in rs):.2f}s "
            f"peak+{max(x['peak_extra_gb'] for x in rs):.1f}GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
