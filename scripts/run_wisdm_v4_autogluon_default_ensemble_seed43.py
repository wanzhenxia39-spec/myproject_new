#!/usr/bin/env python3
"""WISDM seed-43 full A2 ZSL with AutoGluon default ensembles on GPU 1.

Requires all 20 validated seed-43 matrices in configs/attrIbutes_v4/wisdm/seed43.
Exports activity metrics and mean attribute test accuracy across all 20 splits.
"""
from attributes_v4_autogluon_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="wisdm", seed=43))
