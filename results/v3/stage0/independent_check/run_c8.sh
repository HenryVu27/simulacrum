#!/bin/bash
cd "$(dirname "$0")"
for args in "ridge 1024" "mlp 1024" "ridge 4096" "mlp 8192 0,1" "ridge 8192 0,1"; do
  uv run --project /Users/henryvu/personal/umwelt --no-sync python c8_paired.py $args 2>&1 | grep -v Warn
done
