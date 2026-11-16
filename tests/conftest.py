"""Shared pytest configuration and diagnostics for quantum-well tests."""

from datetime import datetime
from pathlib import Path
import builtins
import sys

import pytest


# ============================================================
# Test-session output directory and log
# ============================================================


def pytest_configure(config):
    """Create a timestamped output directory for this pytest session."""

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    output_dir = Path.cwd() / f"tests_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "pytest.log"

    config._qw_test_output_dir = output_dir
    config._qw_test_log_path = log_path
    config._qw_test_counts = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Quantum-well pytest log\n")
        log_file.write("==========================================\n")
        log_file.write(
            f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )
        log_file.write(f"Working directory: {Path.cwd()}\n")
        log_file.write(f"Output directory:  {output_dir}\n")
        log_file.write("==========================================\n\n")


def _write_to_log(config, text):
    """Append text to the current pytest-session log."""

    log_path = config._qw_test_log_path

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def diagnostic_plot_dir(request):
    """Return the timestamped directory for diagnostic plots."""

    return request.config._qw_test_output_dir


@pytest.fixture(autouse=True)
def redirect_test_prints_to_log(request, monkeypatch):
    """
    Redirect print output produced during tests to pytest.log.

    This keeps the terminal output limited to pytest's own normal
    reporting while preserving detailed numerical diagnostics in
    the timestamped session log.
    """

    original_print = builtins.print

    def log_print(*args, **kwargs):
        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        file = kwargs.pop("file", None)
        flush = kwargs.pop("flush", False)

        # Preserve explicitly redirected output to files other than
        # stdout/stderr.
        if file not in (None, sys.stdout, sys.stderr):
            return original_print(
                *args,
                sep=sep,
                end=end,
                file=file,
                flush=flush,
                **kwargs,
            )

        if kwargs:
            return original_print(
                *args,
                sep=sep,
                end=end,
                file=file,
                flush=flush,
                **kwargs,
            )

        text = sep.join(str(arg) for arg in args) + end
        _write_to_log(request.config, text)

    monkeypatch.setattr(
        builtins,
        "print",
        log_print,
    )


# ============================================================
# Pytest reporting hooks
# ============================================================


def pytest_collection_finish(session):
    """Record the number of collected tests in the session log."""

    _write_to_log(
        session.config,
        f"Collected tests: {len(session.items)}\n\n",
    )


def pytest_runtest_logreport(report):
    """Record each test result in the session log."""

    config = report.config if hasattr(report, "config") else None

    # pytest TestReport objects do not consistently expose config.
    # The config is attached by pytest_runtest_makereport below.
    if config is None:
        config = getattr(report, "_qw_config", None)

    if config is None:
        return

    counts = config._qw_test_counts

    if report.when == "call":
        if report.passed:
            counts["passed"] += 1
            status = "PASSED"

        elif report.failed:
            counts["failed"] += 1
            status = "FAILED"

        else:
            counts["skipped"] += 1
            status = "SKIPPED"

        _write_to_log(
            config,
            (
                f"\n[{status}] {report.nodeid}\n"
                f"Duration: {report.duration:.6f} s\n"
            ),
        )

        if report.failed:
            _write_to_log(
                config,
                f"\n{report.longreprtext}\n",
            )

    elif report.when == "setup" and report.failed:
        counts["failed"] += 1

        _write_to_log(
            config,
            (
                f"\n[FAILED DURING SETUP] {report.nodeid}\n"
                f"Duration: {report.duration:.6f} s\n"
                f"\n{report.longreprtext}\n"
            ),
        )

    elif report.when == "setup" and report.skipped:
        counts["skipped"] += 1

        _write_to_log(
            config,
            (
                f"\n[SKIPPED] {report.nodeid}\n"
                f"Duration: {report.duration:.6f} s\n"
            ),
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach the current pytest config to each TestReport."""

    outcome = yield
    report = outcome.get_result()
    report._qw_config = item.config


def pytest_sessionfinish(session, exitstatus):
    """Write the final pytest summary to the session log."""

    config = session.config
    counts = config._qw_test_counts

    _write_to_log(
        config,
        (
            "\n==========================================\n"
            "Pytest summary\n"
            "==========================================\n"
            f"Passed:  {counts['passed']}\n"
            f"Failed:  {counts['failed']}\n"
            f"Skipped: {counts['skipped']}\n"
            f"Exit status: {exitstatus}\n"
            f"Ended: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            "==========================================\n"
        ),
    )
