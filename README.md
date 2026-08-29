# Three-Layer MNIST CNN with `einops` and `einsum`

A single executed Jupyter notebook (`mnist_einops_einsum_cnn.ipynb`) that builds a
three-convolutional-layer CNN from raw PyTorch tensors, trains it on the full official
MNIST training split (60,000 images) for exactly 5 epochs, and evaluates it once on
all 10,000 official test images. Measured result of the committed run: **99.04%**
test accuracy (9,904 / 10,000).

Every learned contraction is an explicit `torch.einsum` equation and every tensor
layout change or reduction is a named-axis `einops` operation. Parameters are plain
leaf tensors in a dictionary, optimized directly with `torch.optim.Adam`. All figures
are Matplotlib PNG images, which render directly in GitHub's notebook preview. The
notebook was authored with QWEN 3.8 via opencode.

## Libraries

| library      | version      | role                          |
| ------------ | ------------ | ----------------------------- |
| Python       | 3.12         | runtime                       |
| PyTorch      | 2.13.0+cu130 | tensors, autograd, CUDA       |
| torchvision  | 0.28.0+cu130 | MNIST data loading            |
| einops       | 0.8.2        | named-axis layouts/reductions |
| matplotlib   | 3.11.1       | figures                       |
| nbformat     | 5.11.1       | notebook format               |
| nbclient     | 0.11.0       | notebook execution            |
| ipykernel    | 7.3.0        | Jupyter kernel                |

## Getting started

```bash
uv venv --python 3.12
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Then open `mnist_einops_einsum_cnn.ipynb` in Jupyter - it ships fully executed, with
all outputs and figures already in place.

On a machine without an NVIDIA GPU, replace the first two pins in
`requirements.txt` with plain `torch` and `torchvision` and drop the
`--extra-index-url`; the notebook then trains on CPU.

## Re-running from a clean kernel

Optional. The committed notebook is a complete record; these regenerate it:

```bash
uv run scripts/run_notebook.py      # re-execute all cells (downloads MNIST into data/ once)
uv run scripts/verify_notebook.py   # validate notebook format and contents
uv run scripts/build_notebook.py    # regenerate the source cells from scratch
```

The run is deterministic (one seed drives every random stream), so a clean re-run on
the same hardware reproduces the committed numbers. `data/` and `.venv/` are
gitignored; only the notebook, the scripts, and this file are tracked.
