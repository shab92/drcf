#!/usr/bin/env python3
"""Build the DRCF dashboard from a recorded run.

    cd build && ../run.sh ./drcf_demo          # writes build/trace.js
    python3 scripts/make_dashboard.py build/trace.js

Writes docs/index.html (self-contained, opens in any browser) and copies the trace to docs/trace.js.
"""
import pathlib
import shutil
import sys

root = pathlib.Path(__file__).resolve().parent.parent
trace = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else root / "build" / "trace.js")
template = (root / "docs" / "template.html").read_text(encoding="utf-8")
if "/*__TRACE__*/" not in template:
    sys.exit("docs/template.html has no /*__TRACE__*/ placeholder")
data = trace.read_text(encoding="utf-8")
(root / "docs" / "index.html").write_text(template.replace("/*__TRACE__*/", data), encoding="utf-8")
if trace.resolve() != (root / "docs" / "trace.js").resolve():
    shutil.copyfile(trace, root / "docs" / "trace.js")
print("wrote", root / "docs" / "index.html")
