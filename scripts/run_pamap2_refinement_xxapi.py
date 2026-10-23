#!/usr/bin/env python3
"""Generate PAMAP2 seed-42 semantic plans and complete A2 vectors through xxapi.

Default output: /home/wanzhen/myproject_new/configs/attrIbutes_v4/pamap2/
"""
from refinement_xxapi_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="pamap2"))
