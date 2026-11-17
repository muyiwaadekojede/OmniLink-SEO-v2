"""Automated Quality Gate & 6-Specialist Verification Committee CLI entry point."""

import json
import sys
from pathlib import Path

from lb.verify import audit_workbook

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".linkbert_muyiwa/link_architecture_master.xlsx")
    res = audit_workbook(path)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["status"] == "PASS" else 1)
