"""Execute the notebook top-to-bottom in a fresh kernel and save outputs in place."""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "mnist_einops_einsum_cnn.ipynb"

# Register a throwaway kernelspec bound to this interpreter so the script works
# on any machine (no manual `ipykernel install` needed).
_KERNEL_DATA_DIR = Path(tempfile.mkdtemp(prefix="ai231-me1-kernels-"))
_SPEC_DIR = _KERNEL_DATA_DIR / "kernels" / "mnisteinops"
_SPEC_DIR.mkdir(parents=True)
(_SPEC_DIR / "kernel.json").write_text(
    json.dumps(
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "ai231-me1 (venv)",
            "language": "python",
            "metadata": {},
            "interrupt_mode": "message",
        }
    )
)
os.environ["JUPYTER_DATA_DIR"] = str(_KERNEL_DATA_DIR)

import nbformat
from nbclient import NotebookClient

nb = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="mnisteinops",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
print("executed and saved", NOTEBOOK)
