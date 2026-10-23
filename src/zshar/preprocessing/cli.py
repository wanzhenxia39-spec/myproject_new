"""Run dataset preprocessing and feature extraction into an explicit directory."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

DATASET_MODULES = {
    "pamap2": "preprocess_pamap2",
    "mhealth": "preprocess_mhealth",
    "dsads": "preprocess_DSADS",
    "uschad": "preprocess_uschad",
    "wisdm": "preprocess_wisdm",
}
RAW_SAMPLE_RATES = {"pamap2": 100, "mhealth": 50, "dsads": 25, "uschad": 100, "wisdm": 20}


def main(argv=None, *, default_dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_MODULES, default=default_dataset,
                        required=default_dataset is None)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--target-sr", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--version", help="Output suffix; defaults to <target-sr>_<seq-len>")
    args = parser.parse_args(argv)
    if not args.data_path.is_dir():
        parser.error(f"Input directory does not exist: {args.data_path}")
    if not 0 < args.target_sr <= RAW_SAMPLE_RATES[args.dataset]:
        parser.error("target-sr must be positive and no greater than the raw sampling rate")
    if args.seq_len <= 12:
        parser.error("seq-len must exceed 12 for the gravity low-pass filter")
    version = args.version or f"{args.target_sr}_{args.seq_len}"
    if not version or Path(version).name != version or version in {".", ".."}:
        parser.error("version must be a filename suffix, not a path")
    module = importlib.import_module(f"zshar.preprocessing.{DATASET_MODULES[args.dataset]}")
    preprocess = getattr(module, f"preprocess_{args.dataset}")
    preprocess(data_path=str(args.data_path), save_path=str(args.save_path),
               version=version, target_sr=args.target_sr, seq_len=args.seq_len)


if __name__ == "__main__":
    main()
