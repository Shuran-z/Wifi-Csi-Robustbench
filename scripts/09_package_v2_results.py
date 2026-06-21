#!/usr/bin/env python3
import argparse, os, zipfile
from pathlib import Path
from csi_er.utils import PROJECT_ROOT
ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); out=PROJECT_ROOT/a.output_dir
zip_path=PROJECT_ROOT/f'wifi_csi_robustbench_v2_{a.output_dir.replace("outputs_v2_","")}.zip'
include_dirs=['src','scripts','tests','configs','docs','artifacts','data/splits','references','.github','examples',a.output_dir]
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for d in include_dirs:
        base=PROJECT_ROOT/d
        if not base.exists(): continue
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc'):
                z.write(p, p.relative_to(PROJECT_ROOT))
    for f in ['README.md','LICENSE','CITATION.cff','CODE_OF_CONDUCT.md','CONTRIBUTING.md','SECURITY.md','THIRD_PARTY_NOTICES.md','CHANGELOG.md','DATA_CARD.md','BENCHMARK_CARD.md','MODEL_CARD.md','REPRODUCIBILITY.md','requirements.txt','requirements-lock.txt','environment.yml','pyproject.toml','Dockerfile','Makefile','.gitignore','.gitattributes','.pre-commit-config.yaml','run_all_v2.sh','run_all.sh','run_smoke.sh']:
        p=PROJECT_ROOT/f
        if p.exists(): z.write(p,p.relative_to(PROJECT_ROOT))
print(zip_path)
