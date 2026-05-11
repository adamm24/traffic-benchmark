#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.backends.base import DummyBackend
from evaluation.backends.hf_backend import HuggingFaceBackend
from evaluation.loader import load_dataset
from evaluation.metrics import write_report
from evaluation.runner import (
    run_model,
    select_models,
    write_failure_summary,
)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    backend_name = args.backend
    limit = args.limit_per_task

    if args.smoke:
        output_root = args.output_root or Path("results/pilot_smoke")
        limit = args.limit_per_task or 10
    else:
        output_root = output_root or Path("results/pilot")

    dataset = load_dataset(args.dataset_dir)
    specs = select_models(args.model)

    if args.report_only:
        report_path = write_report(output_root, _model_order(specs))
        print(f"Report: {report_path}")
        return 0

    for spec in specs:
        print(f"Running {spec.key} with {backend_name} backend...")
        try:
            backend = build_backend(
                backend_name,
                spec.model_id,
                device=args.device,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
                trust_remote_code=args.trust_remote_code,
            )
            summary = run_model(
                spec,
                backend,
                dataset,
                output_root,
                backend_name=backend_name,
                limit_per_task=limit,
                resume=not args.no_resume,
                qualitative_limit=args.qualitative_limit,
            )
            print_summary(summary)
        except Exception as exc:
            reason = str(exc)
            write_failure_summary(spec, output_root, backend_name, reason)
            print(f"{spec.key}: failed: {reason}")

    report_path = write_report(output_root, _model_order(specs))
    print(f"Report: {report_path}")
    return 0


def _model_order(specs) -> list[dict[str, str]]:
    return [{"key": spec.key, "model_id": spec.model_id, "profile": spec.profile} for spec in specs]


def build_backend(
    backend_name: str,
    model_id: str,
    *,
    device: str,
    dtype: str | None,
    max_new_tokens: int,
    trust_remote_code: bool,
):
    if backend_name == "dummy":
        return DummyBackend()
    if backend_name == "hf":
        return HuggingFaceBackend(
            model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f"Unknown backend: {backend_name}")


def print_summary(summary: dict) -> None:
    accuracy = summary["overall_accuracy"] * 100
    errors = summary["format_errors"]["percentage"]
    print(
        f"{summary['model_key']}: "
        f"{summary['overall_correct']}/{summary['num_examples']} correct "
        f"({accuracy:.2f}%), parse failures {errors:.2f}%"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pilot evaluation on dataset/core_simulation.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/core_simulation"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model", default="all")
    parser.add_argument("--backend", choices=("hf", "dummy"), default="hf")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--qualitative-limit", type=int, default=12)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
