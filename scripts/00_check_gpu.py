#!/usr/bin/env python3
import argparse, json, platform, subprocess
from pathlib import Path
from csi_er.utils import PROJECT_ROOT, ensure_dirs, write_json
p=argparse.ArgumentParser(); p.add_argument('--output-dir'); a=p.parse_args(); ensure_dirs()
out=PROJECT_ROOT/(a.output_dir or 'outputs'); (out/'logs').mkdir(parents=True,exist_ok=True)
info={'python':platform.python_version()}
try:
 import torch
 info.update(torch_version=torch.__version__, cuda_available=bool(torch.cuda.is_available()), cuda_version=torch.version.cuda, gpu_count=torch.cuda.device_count(), current_device=torch.cuda.current_device() if torch.cuda.is_available() else None, gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
 if torch.cuda.is_available():
  props=torch.cuda.get_device_properties(0); info['memory_total_mb']=props.total_memory//1024//1024; info['memory_allocated_mb']=torch.cuda.memory_allocated(0)//1024//1024
except Exception as e: info['torch_error']=repr(e)
try: info['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True,timeout=10)
except Exception as e: info['nvidia_smi_error']=repr(e)
write_json(out/'logs/gpu_check.json',info); Path(out/'logs/gpu_check.txt').write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(info,indent=2,ensure_ascii=False))
