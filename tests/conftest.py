"""Pytest configuration and fixtures for movement tests."""

import pytest
from pathlib import Path
import subprocess
import sys


@pytest.fixture(scope="session")
def ensure_vlm_test_data():
    """Ensure VLM test data exists by running the pipeline if needed."""
    output_dir = Path("/root/movement/data/pose_processed")

    # Check if we have any files with VLM descriptions
    if output_dir.exists():
        npz_files = list(output_dir.rglob("*.npz"))
        # Quick check if any have descriptions (we'd need to load to be sure)
        if len(npz_files) >= 2:
            return  # Assume we have test data

    # Run the pipeline to generate test data
    print("\nGenerating VLM test data (this may take a few minutes)...")
    cmd = [
        sys.executable,
        "data_prep/process_videos.py",
        "--enable_vlm",
        "--limit", "2",
        "--debug"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"Failed to generate test data:\n{result.stderr}")

    print("VLM test data generated successfully")


# Automatically use this fixture for VLM tests
def pytest_collection_modifyitems(config, items):
    """Automatically add ensure_vlm_test_data fixture to VLM tests."""
    for item in items:
        if "vlm" in item.nodeid.lower() and "test_vlm_output" in item.nodeid:
            item.fixturenames.append("ensure_vlm_test_data")