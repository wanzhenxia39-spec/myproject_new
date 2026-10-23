#!/usr/bin/env python3
"""Train/test DSADS v4 independent AutoGluon default ensembles on 20 ZSL splits.

Default: physical GPU 0, seed 42, six unseen activities x 30 fixed test windows.
Run separately after all full A2 matrices have been generated and validated.
Exports activity metrics, equal-split mean attribute accuracy, and per-attribute
accuracy across the splits in which each attribute is present. The shell entry
run_dsads_v4_zsl.sh supplies the default training options without waiting for GPU.
"""
from attributes_v4_autogluon_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="dsads"))
