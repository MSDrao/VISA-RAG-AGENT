"""
Legacy entrypoint kept for convenience.
Delegates to scripts/eval_retrieval.py.
"""

import os
import runpy
import sys

SCRIPT = os.path.join(os.path.dirname(__file__), "eval_retrieval.py")
sys.argv[0] = SCRIPT
runpy.run_path(SCRIPT, run_name="__main__")
