#!/usr/bin/env bash
# Test runner script for the transcribe module.
# Run this script to execute all tests for the transcribe_audio_stream function.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running tests for transcribe_audio_stream..."

if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Please install pytest:"
    echo "   pip install pytest pytest-asyncio"
    exit 1
fi

if pytest src/test_*.py -v --tb=short; then
    echo "\n✅ All tests passed!"
    exit 0
else
    echo "\n❌ Some tests failed."
    exit 1
fi
