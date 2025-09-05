#!/usr/bin/env python3
"""
Standalone CLI for Log2 Ratio Analysis

A dedicated command-line interface for running log2 ratio analysis on BIRDMAn results.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import from the package
sys.path.insert(0, str(Path(__file__).parent))

from log2_ratio_analysis import main

if __name__ == "__main__":
    main()