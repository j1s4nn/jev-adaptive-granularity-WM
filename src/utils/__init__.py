"""
Utility Functions
=================

Common utilities for logging, tree generation, figure styling, Kaggle I/O.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def save_meta_json(output_dir: Path, meta: Dict[str, Any], run_id: str = None):
    """
    Save meta.json with run metadata.

    Args:
        output_dir: Output directory
        meta: Metadata dict
        run_id: Optional run ID (generated if None)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    meta['run_id'] = run_id
    meta['timestamp'] = datetime.now().isoformat()

    # Add git info
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        meta['git_commit'] = git_commit
    except:
        meta['git_commit'] = 'unknown'

    output_file = output_dir / f"{run_id}_meta.json"
    with open(output_file, 'w') as f:
        json.dump(meta, f, indent=2)

    return output_file


def load_jsonl(filepath: Path):
    """Load JSONL file as list of dicts."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def save_jsonl(filepath: Path, data: list):
    """Save list of dicts as JSONL."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_figure_style():
    """
    Get consistent matplotlib style for all figures.

    Colorblind-safe palette (Okabe-Ito colors).
    """
    style = {
        'colors': [
            '#E69F00',  # Orange
            '#56B4E9',  # Sky blue
            '#009E73',  # Green
            '#F0E442',  # Yellow
            '#0072B2',  # Dark blue
            '#D55E00',  # Vermillion
            '#CC79A7',  # Pink
        ],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
    }

    return style


class StructuredLogger:
    """
    Structured logging with automatic JSON export.
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.entries = []

    def log(self, level: str, message: str, **kwargs):
        """
        Log an entry.

        Args:
            level: 'INFO', 'WARNING', 'ERROR'
            message: Log message
            **kwargs: Additional fields
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }

        self.entries.append(entry)

        # Print to console
        print(f"[{level}] {message}")

    def info(self, message: str, **kwargs):
        self.log('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log('WARNING', message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log('ERROR', message, **kwargs)

    def save(self):
        """Save log to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
