#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
from csi_er.utils import PROJECT_ROOT, ensure_dirs, append_log
ensure_dirs(); dst=PROJECT_ROOT/'third_party/WiFi-CSI-Sensing-Benchmark'
if not dst.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark.git',str(dst)], check=False)
append_log('Prepared SenseFi reference repository under third_party/WiFi-CSI-Sensing-Benchmark')
print(dst)
