#!/bin/bash

# source .venv/bin/activate
pkill -f "uvicorn" || true
uvicorn app:app --reload --port 8000
