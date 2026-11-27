#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark oMLX image-generation adapters (speed, memory, concurrency, fidelity).

Loads each model once (as the server does), warms up, then times text-to-image
and optional editing runs. Per-step timestamps from the progress callback split
the time into denoise (steps) and the rest (prompt/reference encoding + VAE
decode). Images are saved so runs with different settings can be compared
(PSNR against a reference directory or a previous report).

Examples:
  python scripts/bench_image.py --model ~/.cache/.../flux-3-image-klein/snapshots/<rev> \\
      --out /tmp/bench/klein-base --edit
  python scripts/bench_image.py --model <path> --accel a8=0 --out /tmp/bench/klein-noa8 \\
      --compare /tmp/bench/klein-base
  python scripts/bench_image.py --model <path> --out /tmp/bench/krea \\
      --sizes 512x512,1024x1024 --concurrent 2
  python scripts/bench_image.py --model <path> --out /tmp/bench/after \\
      --baseline /tmp/bench/before/bench.json --fail-ratio 1.10
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _parse_sizes(text: str | None, fallback: str) -> list[tuple[int, int]]:
    raw = text or fallback
    return [tuple(int(v) for v in item.lower().split("x")) for item in raw.split(",")]


def _parse_ints(text: str | None) -> list[int | None]:
    if not text:
        return [None]
    return [int(v) for v in text.split(",")]


def _parse_accel(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = value.lower() not in ("0", "false", "off", "no")
    return out


def _hardware() -> dict:
    from omlx.utils.hardware import get_chip_name, get_total_memory_bytes

    return {
        "chip": get_chip_name(),
        "memory_gb": round(get_total_memory_bytes() / 1e9, 1),
    }


def _summaries(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    cold: dict[tuple, float] = {}
    for record in runs:
        if "total_s" not in record:
            continue
        key = (record["size"], record["steps"])
        if record["name"].endswith(":warmup"):
            cold[(record["name"][: -len(":warmup")], *key)] = record["total_s"]
            continue
        groups.setdefault((record["name"], *key), []).append(record)
    summaries = []
    for (name, size, steps), records in sorted(groups.items()):
        totals = [record["total_s"] for record in records]
        p50 = _percentile(totals, 0.5)
        summaries.append(
            {
                "name": name,
                "size": size,
                "steps": steps,
                "total_p50": round(p50, 3),
                "total_p90": round(_percentile(totals, 0.9), 3),
                "step_p50": round(
                    statistics.median(
                        record["step_s"] for record in records if record["step_s"]
                    ),
                    4,
                ),
                "other_p50": round(
                    statistics.median(record["other_s"] for record in records), 3
                ),
                "peak_extra_gb": round(
                    max(record["peak_extra_gb"] for record in records), 2
                ),
                "steps_per_s": round(steps / p50, 3) if p50 else None,
                "images_per_min": round(60.0 / p50, 2) if p50 else None,
                "cold_s": cold.get((name, size, steps)),
            }
        )
    return summaries


def _load_model(adapter, path: Path, quantize, accel):
    import mlx.core as mx

    started = time.monotonic()
    model = adapter.load(path, quantize, accel)
    mx.synchronize()
    load_s = time.monotonic() - started
    resident = mx.get_active_memory()
    print(
        f"[{adapter.name}] load {load_s:.1f}s, resident {resident / 1e9:.2f} GB, accel={accel}",
        flush=True,
    )
    return model, load_s, resident


def _run_once(adapter, model, params, resident: int, name: str, size: str) -> tuple:
    import mlx.core as mx

    stamps: list[float] = []

    def on_step(i, total):
        mx.synchronize()
        stamps.append(time.monotonic())

    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.monotonic()
    image = adapter.generate(model, params, on_step, None)
    mx.synchronize()
    total = time.monotonic() - started
    steps = [b - a for a, b in zip(stamps, stamps[1:])]
    denoise = (
        (stamps[-1] - stamps[0]) + (statistics.median(steps) if steps else 0)
        if stamps
        else 0.0
    )
    record = {
        "name": name,
        "size": size,
        "steps": params.steps,
        "total_s": round(total, 3),
        "step_s": round(statistics.median(steps), 4) if steps else None,
        "denoise_s": round(denoise, 3),
        "other_s": round(total - denoise, 3),
        "peak_extra_gb": round((mx.get_peak_memory() - resident) / 1e9, 2),
    }
    return image, record


def _decode(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data))


def _concurrent(args, adapter, path: Path, quantize, accel, configs, serial_images, serial_totals):
    """Run K identical requests through the engine's dispatcher."""
    from omlx.engine.image_gen import ImageGenEngine

    engine = ImageGenEngine(str(path), quantize=quantize, adapter=adapter, accel=accel)
    rows = []

    async def main():
        await engine.start()
        try:
            for size, steps in configs:
                width, height = (int(v) for v in size.split("x"))
                params = engine.build_params(
                    PROMPT, width=width, height=height, steps=steps, seed=args.seed
                )
                await engine.generate(params)
                started = time.monotonic()
                results = await asyncio.gather(
                    *(engine.generate(params) for _ in range(args.concurrent))
                )
                total = time.monotonic() - started
                reference = serial_images.get((size, steps))
                psnr = (
                    min(_psnr(reference, _decode(result[0]["data"])) for result in results)
                    if reference is not None
                    else None
                )
                serial_total = serial_totals.get((size, steps))
                rows.append(
                    {
                        "size": size,
                        "steps": steps,
                        "concurrent": args.concurrent,
                        "total_s": round(total, 3),
                        "per_image_s": round(total / args.concurrent, 3),
                        "serial_total_s": serial_total,
                        "speedup": (
                            round(serial_total * args.concurrent / total, 2)
                            if serial_total
                            else None
                        ),
                        "psnr_min": round(psnr, 2) if psnr is not None else None,
                        "batching": engine.get_stats()["batching"],
                    }
                )
        finally:
            await engine.stop()

    asyncio.run(main())
    return rows


def _compare_baseline(report: dict, baseline_path: Path, fail_ratio, fail_p50) -> int:
    baseline = json.loads(baseline_path.read_text())
    before = {
        (item["name"], item["size"], item["steps"]): item
        for item in baseline.get("summaries", [])
    }
    failures = 0
    print("\n== Baseline comparison")
    print(f"{'run':<28} {'size':<10} {'steps':>5} {'base':>9} {'now':>9} {'ratio':>7}")
    for item in report["summaries"]:
        key = (item["name"], item["size"], item["steps"])
        if key not in before:
            continue
        base = before[key]["total_p50"]
        now = item["total_p50"]
        ratio = now / base if base else 0.0
        mark = ""
        if fail_ratio is not None and ratio > fail_ratio:
            mark += "  FAIL"
            failures += 1
        if fail_p50 is not None and now > fail_p50:
            mark += "  FAIL"
            failures += 1
        print(
            f"{item['name']:<28} {item['size']:<10} {item['steps']:>5} "
            f"{base:>9.3f} {now:>9.3f} {ratio:>7.2f}{mark}"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantize", type=int, default=8, help="0 = bf16")
    parser.add_argument("--accel", action="append", default=[], help="engine accel option, e.g. a8=0")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--sizes", default=None, help="comma-separated widthxheight grid")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--steps-list", default=None, help="comma-separated step counts")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edit", action="store_true", help="also benchmark editing (needs an edit-capable model)")
    parser.add_argument("--edit-image", default=None, help="reference image (default: the generated image)")
    parser.add_argument("--step-cache", action="store_true", help="request the opt-in step cache")
    parser.add_argument("--concurrent", type=int, default=0, help="run K identical requests through the engine")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default=None, help="report path (default: <out>/bench.json)")
    parser.add_argument("--compare", default=None, help="directory of a previous run to PSNR against")
    parser.add_argument("--baseline", default=None, help="previous report to compare totals against")
    parser.add_argument("--fail-ratio", type=float, default=None, help="fail when now/base exceeds this ratio")
    parser.add_argument("--fail-p50", type=float, default=None, help="fail when total p50 exceeds this many seconds")
    args = parser.parse_args()

    import mlx.core as mx
    from PIL import Image

    from omlx.image_gen import find_adapter
    from omlx.image_gen.base import ImageGenParams

    path = Path(args.model).expanduser()
    adapter = find_adapter(path)
    if adapter is None:
        print(f"no image adapter matches {path}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = _parse_sizes(args.sizes, args.size)
    if args.steps_list:
        requested_steps = _parse_ints(args.steps_list)
    elif args.steps:
        requested_steps = [args.steps]
    else:
        requested_steps = [None]
    accel = {"a8": True, **_parse_accel(args.accel)}
    load_accel = dict(accel)
    driver_accel = {**accel, "step_cache": args.step_cache}
    quantize = args.quantize or None
    if quantize and adapter.is_prequantized(path):
        quantize = None

    model, load_s, resident = _load_model(adapter, path, quantize, accel)
    runtime_stats = adapter.runtime_stats(model)
    records: list[dict] = []
    serial_images: dict[tuple[str, int], object] = {}
    configs: list[tuple[str, int]] = []

    for width, height in sizes:
        size = f"{width}x{height}"
        for steps in requested_steps:
            steps = steps or adapter.defaults.steps
            configs.append((size, steps))
            base = dict(
                prompt=PROMPT,
                width=width,
                height=height,
                steps=steps,
                guidance=adapter.defaults.guidance,
                seed=args.seed,
            )
            if args.step_cache:
                base["accel"] = {"step_cache": True}
            params = ImageGenParams(**base)
            image, record = _run_once(adapter, model, params, resident, "t2i:warmup", size)
            records.append(record)
            print(json.dumps(record), flush=True)
            serial_images[(size, steps)] = image
            for _ in range(args.repeat):
                image, record = _run_once(adapter, model, params, resident, "t2i", size)
                records.append(record)
                print(json.dumps(record), flush=True)
            image.save(out / f"t2i_{width}x{height}_{steps}.png")
            if args.compare:
                reference = Path(args.compare) / f"t2i_{width}x{height}_{steps}.png"
                if reference.exists():
                    value = _psnr(image, Image.open(reference))
                    print(f"  t2i {size} {steps}: PSNR vs {args.compare}: {value:.2f} dB", flush=True)
                    records.append(
                        {"name": "t2i:psnr", "size": size, "steps": steps, "psnr_db": round(value, 2)}
                    )

            if args.edit and adapter.capabilities.edit:
                ref_path = Path(args.edit_image) if args.edit_image else None
                reference_image = (
                    Image.open(ref_path)
                    if ref_path
                    else Image.open(out / f"t2i_{width}x{height}_{steps}.png")
                )
                edit_width, edit_height = adapter.edit_canvas([reference_image])
                edit = ImageGenParams(
                    **{
                        **base,
                        "prompt": EDIT_PROMPT,
                        "width": edit_width,
                        "height": edit_height,
                    },
                    references=[reference_image],
                )
                edit_size = f"{edit_width}x{edit_height}"
                _, record = _run_once(adapter, model, edit, resident, "edit:warmup", edit_size)
                records.append(record)
                print(json.dumps(record), flush=True)
                for _ in range(args.repeat):
                    edit_image, record = _run_once(
                        adapter, model, edit, resident, "edit", edit_size
                    )
                    records.append(record)
                    print(json.dumps(record), flush=True)
                edit_image.save(out / f"edit_{width}x{height}_{steps}.png")
                if args.compare:
                    compare_path = Path(args.compare) / f"edit_{width}x{height}_{steps}.png"
                    if compare_path.exists():
                        value = _psnr(edit_image, Image.open(compare_path))
                        print(
                            f"  edit {size} {steps}: PSNR vs {args.compare}: {value:.2f} dB",
                            flush=True,
                        )
                        records.append(
                            {"name": "edit:psnr", "size": size, "steps": steps, "psnr_db": round(value, 2)}
                        )

    summaries = _summaries(records)
    serial_totals = {
        (item["size"], item["steps"]): item["total_p50"]
        for item in summaries
        if item["name"] == "t2i"
    }
    concurrent = []
    if args.concurrent:
        adapter.release(model)
        del model
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        concurrent = _concurrent(
            args, adapter, path, quantize, driver_accel, configs, serial_images, serial_totals
        )

    report = {
        "created": int(time.time()),
        "hardware": _hardware(),
        "adapter": adapter.name,
        "model": str(path),
        "quantize": quantize,
        "accel": load_accel,
        "prompt": PROMPT,
        "load_s": round(load_s, 2),
        "resident_gb": round(resident / 1e9, 2),
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "steps": [item["steps"] for item in summaries if item["name"] == "t2i"],
        "stats": runtime_stats,
        "summaries": summaries,
        "concurrent": concurrent,
        "runs": records,
    }
    report_path = Path(args.json) if args.json else out / "bench.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print("\n== Summary")
    for item in summaries:
        print(
            f"{item['name']:<12} {item['size']:<10} {item['steps']:>3} steps  "
            f"p50 {item['total_p50']:.2f}s  p90 {item['total_p90']:.2f}s  "
            f"step {item['step_p50']:.3f}s  other {item['other_p50']:.2f}s  "
            f"peak+{item['peak_extra_gb']:.1f}GB  {item['steps_per_s']:.2f} steps/s"
        )
    for item in concurrent:
        print(
            f"concurrent={item['concurrent']} {item['size']} {item['steps']} steps: "
            f"{item['total_s']:.2f}s total, {item['per_image_s']:.2f}s/image, "
            f"speedup {item['speedup']}, PSNR {item['psnr_min']} dB, "
            f"batching={item['batching']}"
        )
    print(f"report: {report_path}")

    if args.baseline:
        failures = _compare_baseline(report, Path(args.baseline), args.fail_ratio, args.fail_p50)
        if failures:
            print(f"{failures} benchmark check(s) failed")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
