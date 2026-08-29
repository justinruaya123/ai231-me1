"""External verification of the executed notebook."""

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "mnist_einops_einsum_cnn.ipynb"

failures = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(label)


nb = nbformat.read(NOTEBOOK, as_version=4)
try:
    nbformat.validate(nb)
    check("nbformat validation", True)
except Exception as error:
    check("nbformat validation", False, str(error))
    sys.exit(1)

code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
md_cells = [cell for cell in nb.cells if cell.cell_type == "markdown"]
code = "\n".join(cell.source for cell in code_cells)
all_source = "\n".join(cell.source for cell in nb.cells)

# numpy is allowed in the figures section (plotting only), never in model code
model_code_parts = []
in_figures = False
for cell in nb.cells:
    if cell.cell_type == "markdown" and "## 7. Figures" in cell.source:
        in_figures = True
    elif cell.cell_type == "code" and not in_figures:
        model_code_parts.append(cell.source)
model_code = "\n".join(model_code_parts)

output_parts = []
for cell in code_cells:
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            output_parts.append(output.get("text", ""))
        elif output.get("output_type") in {"execute_result", "display_data"}:
            output_parts.append(str(output.get("data", {})))
output_text = "".join(output_parts)

# 1) prohibited operations in code cells
forbidden_patterns = {
    "nn model layer": r"torch\.nn\b|\bnn\.functional\b|\bnn\.(Conv2d|Linear|Sequential|ReLU|Sigmoid|Tanh|MaxPool2d|AvgPool2d|AdaptiveAvgPool2d|Dropout|CrossEntropyLoss)\b",
    "functional helper": r"\bF\.(conv2d|conv1d|conv_transpose2d|linear|unfold|relu|relu6|sigmoid|tanh|softmax|log_softmax|cross_entropy|max_pool|avg_pool|adaptive_avg_pool|adaptive_max_pool|dropout|interpolate)\b",
    "cross-entropy loss layer": r"CrossEntropyLoss|F\.cross_entropy",
    "pooling": r"\bpool\w*\(",
    "unfold/as_strided": r"\bunfold\b|\bas_strided\b",
    "softmax token": r"\bsoftmax\b",
    "relu helper": r"torch\.relu|F\.relu|nn\.ReLU",
    "opaque layout op": r"\.(view|reshape|flatten|permute|transpose|squeeze|unsqueeze)\(",
    "numpy in model cells": r"\bnumpy\b|\bnp\.",
}
for label, pattern in forbidden_patterns.items():
    scope = model_code if label == "numpy in model cells" else code
    matches = list(re.finditer(pattern, scope))
    details = []
    for match in matches:
        start = scope.rfind("\n", 0, match.start()) + 1
        end = scope.find("\n", match.end())
        details.append(scope[start:end if end != -1 else len(scope)].strip())
    check(f"no prohibited: {label}", not matches, "; ".join(details))

# 2) required primitives present
required_patterns = {
    "conv einsum equation": r'torch\.einsum\("bhwcij,ocij->bohw"',
    "loss-selection einsum equation": r'torch\.einsum\("bc,bc->b"',
    "einops.rearrange": r"\brearrange\(",
    "einops.reduce": r"\breduce\(",
    "einops.repeat": r"\brepeat\(",
    "Adam optimizer": r"torch\.optim\.Adam\(",
    "no_grad evaluation": r"torch\.no_grad\(\)",
    "exactly five epochs": r"for epoch in range\(1, 6\):",
    "named-axis logits conversion": r'rearrange\(h3, "b classes 1 1 -> b classes"\)',
    "matplotlib import": r"import matplotlib\.pyplot as plt",
}
for label, pattern in required_patterns.items():
    check(f"required present: {label}", re.search(pattern, code) is not None)

# 3) writeup rules
assert_cells = [i for i, cell in enumerate(code_cells) if re.search(r"^\s*assert\s", cell.source, re.M)]
check("assert statements only in final validation cell",
      assert_cells == [len(code_cells) - 1] and len(assert_cells) == 1, str(assert_cells))

constraint_hits = [
    i for i, cell in enumerate(md_cells)
    if i > 0 and re.search(r"prohibit|forbidden|not used anywhere|not counted", cell.source, re.I)
]
check("constraint language only in first markdown cell", not constraint_hits, str(constraint_hits))

reference_hits = re.findall(r"roatienza|deep[- ]learning[- ]experiments", all_source, re.I)
check("no external references mentioned", not reference_hits, str(reference_hits))

# 4) executed outputs
counts = [cell.get("execution_count") for cell in code_cells]
check("every code cell executed", all(count is not None for count in counts),
      f"missing: {[i for i, c in enumerate(counts) if c is None]}")
errors = [o for cell in code_cells for o in cell.get("outputs", []) if o.get("output_type") == "error"]
check("no execution errors", not errors, "; ".join(o.get("ename", "") for o in errors))
check("pre-training checks all matched", "all matched" in output_text and "MISMATCH" not in output_text)
check("final validation printed", "final validation passed" in output_text)

epochs_seen = set(re.findall(r"epoch (\d)/5  loss=", output_text))
check("exactly 5 epoch summaries", epochs_seen == {"1", "2", "3", "4", "5"}, str(sorted(epochs_seen)))

accuracy_match = re.search(r"final test accuracy:\s+([\d.]+)\s+\(([\d.]+)%\)", output_text)
check("final accuracy over 10000 examples", accuracy_match is not None and "test examples      : 10000" in output_text)
if accuracy_match:
    print(f"       measured test accuracy: {accuracy_match.group(2)}%")

def png_size(output):
    from base64 import b64decode
    from io import BytesIO

    from PIL import Image
    image = Image.open(BytesIO(b64decode(output["data"]["image/png"])))
    return image.width, image.height


figures = [o for cell in code_cells for o in cell.get("outputs", [])
           if o.get("output_type") == "display_data"
           and "image/png" in o.get("data", {})]
check("matplotlib figures saved in notebook", len(figures) >= 2, f"found {len(figures)}")

aspect_ratios = [width / height for width, height in map(png_size, figures)] if figures else []
check("history figure rendered as a wide 1x2 panel", any(ratio > 2.0 for ratio in aspect_ratios))
check("grid figure rendered as a square 4x4 panel", any(0.9 <= ratio <= 1.1 for ratio in aspect_ratios))

grid_cell = next((cell for cell in code_cells if "plt.subplots(4, 4" in cell.source), None)
check("4x4 grid cell present", grid_cell is not None)
if grid_cell is not None:
    grid_pngs = [o for o in grid_cell.get("outputs", [])
                 if o.get("output_type") == "display_data" and "image/png" in o.get("data", {})]
    check("grid cell produced a PNG", len(grid_pngs) >= 1, f"found {len(grid_pngs)}")
check("grid title carries final accuracy",
      grid_cell is not None and "final test accuracy" in grid_cell.source and "test_accuracy" in grid_cell.source)

print()
if failures:
    print("VERIFICATION FAILED:")
    for failure in failures:
        print(" -", failure)
    sys.exit(1)
print("ALL CHECKS PASSED")
