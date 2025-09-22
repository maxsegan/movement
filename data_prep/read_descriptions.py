#!/usr/bin/env python3
"""
Utility to read and display action descriptions from NPZ files.
"""

import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional


def read_action_descriptions(npz_path: str, verbose: bool = False):
    """Read and display action descriptions from an NPZ file."""

    # Load the NPZ file
    data = np.load(npz_path, allow_pickle=True)

    print(f"\n=== File: {Path(npz_path).name} ===")

    # Check basic metadata
    if 'meta' in data:
        meta = data['meta']
        fps, frames, width, height = meta
        print(f"Video: {width:.0f}x{height:.0f}, {frames:.0f} frames @ {fps:.1f} FPS")

    # Check quality metrics
    if verbose:
        if 'density_ok' in data:
            print(f"Density OK: {data['density_ok'][0]}")
        if 'dynamic_ok' in data:
            print(f"Dynamic OK: {data['dynamic_ok'][0]}")
        if 'quality' in data:
            print(f"Quality Score: {data['quality'][0]:.3f}")

    # Check for action descriptions
    if 'action_descriptions' not in data:
        print("No action descriptions found in this file.")
        return

    # Parse action descriptions
    desc_json = str(data['action_descriptions'][0])

    if desc_json == '[]':
        print("Action descriptions field exists but is empty.")
        return

    try:
        descriptions = json.loads(desc_json)

        if not descriptions:
            print("No action descriptions generated.")
            return

        print(f"\nFound {len(descriptions)} action description(s):")
        print("-" * 60)

        for i, desc in enumerate(descriptions, 1):
            print(f"\nSegment {i}:")

            if 'frames' in desc:
                frames = desc['frames']
                if isinstance(frames, list) and len(frames) > 0:
                    print(f"  Frames: {frames[0]} to {frames[-1]}")
                else:
                    print(f"  Frames: {frames}")

            if 'description' in desc:
                # Clean up the description text
                text = desc['description']
                # Remove extra quotes if present
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                print(f"  Action: {text}")

            if 'confidence' in desc:
                print(f"  Confidence: {desc['confidence']:.2f}")

    except json.JSONDecodeError as e:
        print(f"Error parsing action descriptions: {e}")
        print(f"Raw data: {desc_json[:200]}...")


def main():
    """Main function to read descriptions from command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python read_descriptions.py <npz_file> [--verbose]")
        print("\nExample:")
        print("  python read_descriptions.py /root/movement/data/pose_processed/nturgb+d_rgb/S001C001P001R001A001_rgb.npz")
        sys.exit(1)

    npz_path = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    if not Path(npz_path).exists():
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)

    read_action_descriptions(npz_path, verbose)


if __name__ == "__main__":
    main()