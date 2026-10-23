#!/usr/bin/env python3
"""Generate WISDM seed-42 plans and full A2 matrices with xxapi gpt-6.1-sol.

Output: /home/wanzhen/myproject_new/configs/attrIbutes_v4/wisdm/
"""
from refinement_xxapi_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="wisdm"))
