#!/usr/bin/env bash
# Manuscript audit: every table cell and the in-text ranges of the NAACL manuscript against experiments/*.json.
# Usage: bash run_all.sh   (MANUSCRIPT_DIR and RESULTS_DIR select the manuscript and the JSONs; defaults in common.py)
cd "$(dirname "$0")"
for s in t01_05 t02b t06_10 t11_15 t16_22 t22_length t_new text_checks; do python3 "$s.py"; done 2>&1 | tee audit.log
python3 - <<'PY'
import re
s=open('audit.log').read()
cells=sum(int(m) for m in re.findall(r'^### .*?: (\d+) values checked',s,re.M))
mm=sum(int(m) for m in re.findall(r'(\d+) mismatches',s))
print(f'TOTAL table values checked: {cells}; total mismatches (tables + text): {mm}')
PY
