#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd -P )
cd "$SCRIPT_DIR" || exit
echo "Current working directory: $(pwd)"

# # source .venv/bin/activate
pkill -f "uvicorn" || true
uvicorn src.app:app --reload --port 8000