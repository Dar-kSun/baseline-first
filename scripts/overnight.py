"""Run the long data and benchmark steps unattended, with retries and a log.

    python scripts/overnight.py [STEP ...] [--after-pid PID]

Steps, in order:
  1. stream    scripts/11_pseudobulk_replogle_nadig.py  (resumes from its checkpoint)
  2. benchmark scripts/13_unseen_gene_unseen_context.py (needs step 1)
  3. h1        scripts/10_pseudobulk_support_set.py competition_train, the 2025
               H1 CRISPRi data, with control moments on the step-1 gene list
  4. tests     pytest, as a final check that the code is intact

Optional steps, run only when named: shrinkage (scripts/14),
submission-ridge and submission-mean (scripts/20; build files, never submit).

A failed step is retried (steps 1 and 3 resume or restart cleanly); a step
whose prerequisite failed is skipped. Output goes to logs/overnight.log and a
summary to logs/overnight-status.json. While it runs, Windows is asked not
to sleep; the request ends with the process.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOG = LOGS / "overnight.log"
STATUS = LOGS / "overnight-status.json"
PY = sys.executable
SUBMIT = "scripts/20_make_vcc_submission.py"

STEPS = [
    ("stream", [PY, "-u", "scripts/11_pseudobulk_replogle_nadig.py"], 6, None),
    ("benchmark", [PY, "-u", "scripts/13_unseen_gene_unseen_context.py"], 2, "stream"),
    (
        "h1",
        [
            PY,
            "-u",
            "scripts/10_pseudobulk_support_set.py",
            "competition_train",
            "--cell-type",
            "h1",
            "--moments-of",
            "non-targeting",
            "--moments-genes",
            "data/processed/replogle_genes.txt",
        ],
        3,
        None,
    ),
    ("tests", [PY, "-m", "pytest", "-q"], 1, None),
    ("shrinkage", [PY, "-u", "scripts/14_transfer_with_shrinkage.py"], 2, None),
    ("submission-ridge", [PY, "-u", SUBMIT, "--method", "ridge"], 2, None),
    ("submission-mean", [PY, "-u", SUBMIT, "--method", "mean"], 2, None),
    ("local-h1", [PY, "-u", "scripts/15_local_vcc_score_h1.py"], 2, None),
]
DEFAULT = ["stream", "benchmark", "h1", "tests"]


def keep_awake() -> None:
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(status: dict) -> None:
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def wait_for(pid: int) -> None:
    """Block until process `pid` has exited (Windows and POSIX)."""
    while True:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True
            ).stdout
            if str(pid) not in out:
                return
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", default=DEFAULT, help="steps to run, in order")
    parser.add_argument("--after-pid", type=int, help="wait for this process to exit first")
    args = parser.parse_args()
    unknown = set(args.steps) - {name for name, *_ in STEPS}
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")
    LOGS.mkdir(exist_ok=True)
    keep_awake()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    status = {"started": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "pid": os.getpid(), "steps": {}}
    if args.after_pid:
        log(f"waiting for pid {args.after_pid} to exit")
        wait_for(args.after_pid)
    log(f"run started (pid {os.getpid()}): {' '.join(args.steps)}")
    for name, cmd, attempts, needs in [step for step in STEPS if step[0] in args.steps]:
        if needs and status["steps"].get(needs, {}).get("result") != "ok":
            status["steps"][name] = {"result": f"skipped: {needs} did not succeed"}
            log(f"{name}: skipped because {needs} did not succeed")
            write_status(status)
            continue
        for attempt in range(1, attempts + 1):
            log(f"{name}: attempt {attempt}/{attempts}: {' '.join(cmd[1:])}")
            status["steps"][name] = {"result": "running", "attempt": attempt}
            write_status(status)
            began = time.time()
            with open(LOG, "a", encoding="utf-8") as out:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=out).returncode
            minutes = round((time.time() - began) / 60, 1)
            if code == 0:
                status["steps"][name] = {"result": "ok", "attempt": attempt, "minutes": minutes}
                log(f"{name}: ok after {minutes} min")
                break
            status["steps"][name] = {"result": f"failed (exit {code})", "attempt": attempt}
            log(f"{name}: exit {code} after {minutes} min")
            if attempt < attempts:
                time.sleep(120)
        write_status(status)
    status["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    write_status(status)
    log("overnight run finished")


if __name__ == "__main__":
    main()
