#!/usr/bin/env bash
set -eu
cd ~/Desktop/WFR-Memory-Test
PY=.venv/bin/python
"$PY" -c "import torch; c=torch.cuda.is_available(); print('CUDA', c); print(torch.cuda.get_device_name(0) if c else 'cpu')"
export PYTHONUNBUFFERED=1
"$PY" -u experiments/09-lm-parity/run_parity_pair.py --quick --fair-parity
