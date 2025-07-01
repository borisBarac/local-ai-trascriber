#!/usr/bin/env python3
"""
Test runner script for the transcribe module.
Run this script to execute all tests for the transcribe_audio_stream function.
"""

import sys
import subprocess
import os


def run_tests():
    """Run the test suite."""
    print("Running tests for transcribe_audio_stream...")

    # Change to the backend directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        # Run pytest with verbose output
        subprocess.run(
            [sys.executable, "-m", "pytest", "test_transcribe.py", "-v", "--tb=short"],
            check=True,
            capture_output=False,
        )

        print("\n✅ All tests passed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest:")
        print("   pip install pytest pytest-asyncio")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
