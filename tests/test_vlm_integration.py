#!/usr/bin/env python3
"""
Test script for VLM action description integration.
"""

import numpy as np
import json
from pathlib import Path
import pytest

@pytest.mark.skip(reason="VLM requires large model download - enable when needed")
def test_vlm_output():
    """Test that VLM integration produces expected outputs.

    Prerequisites: Run the following command first to generate test data:
        python data_prep/process_videos.py --enable_vlm --limit 2 --debug

    Note: This test is skipped by default as it requires downloading a large VLM model.
    To run: pytest tests/test_vlm_integration.py::test_vlm_output -v --no-skip
    """

    # Find a processed file with descriptions
    output_dir = Path("/root/movement/data/pose_processed")

    # Look for NPZ files
    npz_files = list(output_dir.rglob("*.npz"))

    assert npz_files, (
        "No processed files found. Please run first:\n"
        "  python data_prep/process_videos.py --enable_vlm --limit 2 --debug"
    )

    print(f"Found {len(npz_files)} processed files")

    # Check a few files for action descriptions
    files_with_descriptions = 0
    sample_descriptions = []

    for npz_file in npz_files[:10]:  # Check first 10
        try:
            data = np.load(npz_file, allow_pickle=True)

            if 'action_descriptions' in data:
                desc_json = str(data['action_descriptions'][0])
                descriptions = json.loads(desc_json)

                if descriptions:
                    files_with_descriptions += 1
                    sample_descriptions.append({
                        'file': npz_file.name,
                        'descriptions': descriptions[:2]  # First 2 descriptions
                    })

        except Exception as e:
            print(f"Error reading {npz_file}: {e}")

    # Report results
    print(f"\nFiles with action descriptions: {files_with_descriptions}/{min(10, len(npz_files))}")

    if sample_descriptions:
        print("\nSample action descriptions:")
        for sample in sample_descriptions[:3]:
            print(f"\n  File: {sample['file']}")
            for desc in sample['descriptions']:
                print(f"    - Frames {desc['frames']}: {desc['description'][:100]}...")
                print(f"      Confidence: {desc['confidence']:.2f}")

    # FAIL if no VLM descriptions found - that's what we're testing!
    assert files_with_descriptions > 0, (
        f"No files with action descriptions found out of {min(10, len(npz_files))} checked.\n"
        "Did you run with --enable_vlm flag? Run:\n"
        "  python data_prep/process_videos.py --enable_vlm --limit 2 --debug"
    )


def test_npz_file_structure():
    """Test that NPZ files have expected structure and keys."""
    output_dir = Path("/root/movement/data/pose_processed")
    npz_files = list(output_dir.rglob("*.npz"))

    if not npz_files:
        pytest.skip("No processed files found")

    # Test first available file
    npz_file = npz_files[0]
    data = np.load(npz_file, allow_pickle=True)

    # Check if it's Kinetics or NTU format based on path
    is_ntu = 'nturgb+d_rgb' in str(npz_file)

    # Check required keys (both formats have slightly different keys)
    base_keys = ['meta', 'density_ok', 'dynamic_ok', 'quality']
    for key in base_keys:
        assert key in data, f"Missing required key: {key}"

    # Check for pose data (different key names)
    assert 'pose_3d' in data or 'pose3d' in data, "Missing pose data (neither pose_3d nor pose3d found)"

    # Check metadata structure
    meta = data['meta']
    assert len(meta) >= 4, "Metadata should have at least 4 elements"
    assert meta[0] > 0, "FPS should be positive"
    assert meta[1] > 0, "Frame count should be positive"

    # Check quality metrics
    assert 'density_ok' in data
    assert 'dynamic_ok' in data
    assert 0 <= data['quality'][0] <= 1, "Quality score should be between 0 and 1"


def test_action_description_format():
    """Test that action descriptions have correct format when present."""
    output_dir = Path("/root/movement/data/pose_processed")
    npz_files = list(output_dir.rglob("*.npz"))

    if not npz_files:
        pytest.skip("No processed files found")

    # Find a file with action descriptions
    found_descriptions = False
    for npz_file in npz_files[:10]:
        data = np.load(npz_file, allow_pickle=True)

        if 'action_descriptions' in data:
            desc_json = str(data['action_descriptions'][0])
            descriptions = json.loads(desc_json)

            if descriptions:
                found_descriptions = True
                # Check format of descriptions
                for desc in descriptions:
                    assert 'frames' in desc, "Description missing 'frames' key"
                    assert 'description' in desc, "Description missing 'description' key"
                    assert 'confidence' in desc, "Description missing 'confidence' key"

                    # Validate types
                    assert isinstance(desc['frames'], list), "Frames should be a list"
                    assert len(desc['frames']) == 2, "Frames should have start and end"
                    assert isinstance(desc['description'], str), "Description should be a string"
                    assert 0 <= desc['confidence'] <= 1, "Confidence should be between 0 and 1"
                break

    if not found_descriptions:
        pytest.skip("No files with action descriptions found. Run with --enable_vlm to generate them.")


