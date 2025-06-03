"""
Pipeline utilities: run the external separation & analysis script.
"""
import os
import sys
import subprocess
from typing import Optional, Iterator


def run_subprocess_pipeline(input_path: str,
                           start_sec: float,
                           end_sec: Optional[float],
                           base_dir: str) -> Iterator[str]:
    """
    Execute the testing_split script in unbuffered mode for real-time logs.

    Yields:
        Each line of stdout from the subprocess.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'testing', 'testing_split.py')
    cmd = [sys.executable, '-u', script_path, input_path,
           '--start-sec', str(start_sec),
           '--output-dir', base_dir]
    if end_sec is not None:
        cmd.extend(['--end-sec', str(end_sec)])
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)
    for line in proc.stdout or []:
        yield line
    proc.wait()