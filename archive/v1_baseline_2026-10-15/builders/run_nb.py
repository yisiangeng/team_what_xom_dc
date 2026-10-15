"""Execute a notebook in place from the repo root: python run_nb.py <notebook.ipynb>"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
    print(f"{path.name}: executed in {time.time() - t0:.0f} s")
