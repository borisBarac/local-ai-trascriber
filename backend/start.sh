#!/bin/bash

# source .venv/bin/activate
pkill -f "uvicorn" || true
uvicorn src.app:app --reload --port 8000