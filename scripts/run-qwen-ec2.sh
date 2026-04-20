#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run-qwen-torchtitan.sh" "$@"
