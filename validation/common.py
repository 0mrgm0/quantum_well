"""Shared helpers for scientific validation scripts."""

from datetime import datetime
from pathlib import Path


def create_validation_directory():
    """Create a timestamped validation directory in the working directory."""

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path.cwd() / f"validation_{timestamp}"
    output_dir.mkdir(parents=False, exist_ok=False)

    return output_dir


def write_log(output_dir, lines):
    """Write accumulated validation output to validation.log."""

    log_path = output_dir / "validation.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return log_path


def add_header(lines, title):
    """Add a section header to the validation log."""

    lines.append("")
    lines.append("=" * 70)
    lines.append(title)
    lines.append("=" * 70)
