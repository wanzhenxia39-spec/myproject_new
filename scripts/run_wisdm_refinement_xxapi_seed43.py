#!/usr/bin/env python3
"""Generate WISDM seed-43 split-specific plans and full A2 matrices via xxapi.

Uses gpt-6.1-sol, temperature 0.7, non-streaming, and NEW_API_KEY.
Default output: configs/attrIbutes_v4/wisdm/seed43/split_<index>/.
"""
from refinement_xxapi_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="wisdm", seed=43))
