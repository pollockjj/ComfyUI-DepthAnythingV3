import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("depthanythingv3.install")


def main():
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        log.info("No requirements.txt present; skipping install step")
        return

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
    )


main()
