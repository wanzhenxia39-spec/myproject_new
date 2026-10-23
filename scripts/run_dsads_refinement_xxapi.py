#!/usr/bin/env python3
"""Generate DSADS seed-42 semantic plans and complete A2 matrices through xxapi.

Default: gpt-6.1-sol, temperature 0.7, all 20 unseen6 splits.
Output: /home/wanzhen/myproject_new/configs/attrIbutes_v4/dsads/
This entry point does not train classifiers.
"""
from refinement_xxapi_common import main


if __name__ == "__main__":
    raise SystemExit(main(dataset="dsads"))
