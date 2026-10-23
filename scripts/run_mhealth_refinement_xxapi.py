#!/usr/bin/env python3
"""Generate MHEALTH seed-42 semantic plans and complete A2 vectors through xxapi.

Default output: /home/wanzhen/myproject_new/configs/attrIbutes_v4/mhealth/
"""
from refinement_xxapi_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="mhealth"))
