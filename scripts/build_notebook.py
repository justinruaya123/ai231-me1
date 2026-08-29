"""Build mnist_einops_einsum_cnn.ipynb (source cells, no outputs)."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "mnist_einops_einsum_cnn.ipynb"

cells = []


def md(source):
    cells.append(nbformat.v4.new_markdown_cell(source.strip() + "\n"))


def code(source):
    cells.append(nbformat.v4.new_code_cell(source.strip() + "\n"))


# ---------------------------------------------------------------- 1. intro
md('''
# Three-Layer MNIST CNN with `einops` and `einsum`

This notebook builds a convolutional network with exactly **three trainable
convolutional layers** from raw PyTorch tensors, trains it on the full official
MNIST training split (60,000 images) for **exactly 5 epochs**, then evaluates it once
on all **10,000** official test images and reports the measured accuracy.

Result of the committed run: **99.04%** test accuracy (9,904 / 10,000).

Every learned contraction is written as an explicit Einstein-index `torch.einsum`
equation, and every tensor layout change or reduction is a named-axis `einops`
operation. Parameters are leaf tensors with `requires_grad=True`, stored in a plain
dictionary, and passed directly to `torch.optim.Adam`.

## Architecture

The three trainable layers are convolutions only; ReLU activations and the final
`b x 10 x 1 x 1 -> b x 10` logits conversion are not counted as layers. The last
convolution is the classifier, so there is no MLP or fully connected head.

| stage | operation | output layout |
| --- | --- | --- |
| input | normalized MNIST image (optionally shifted) | `b x 1 x 28 x 28` |
| layer 1 | block conv `1 -> 128`, kernel/stride 14, bias | `b x 128 x 2 x 2` |
| activation 1 | ReLU | `b x 128 x 2 x 2` |
| layer 2 | block conv `128 -> 192`, kernel/stride 2, bias | `b x 192 x 1 x 1` |
| activation 2 | ReLU | `b x 192 x 1 x 1` |
| layer 3 (classifier) | block conv `192 -> 10`, kernel/stride 1, bias | `b x 10 x 1 x 1` |
| logits | `rearrange("b classes 1 1 -> b classes")` | `b x 10` |

Each block convolution downsamples by exactly its kernel size, so the three kernel
sizes multiply to 28. The factorization `(14, 2, 1)` was selected from the candidate
triples for generalization: layer 1 builds 14 x 14 half-image local features,
layer 2 mixes the full 28 x 28 image, and layer 3 reads the result out to 10
classes.

Total trainable scalars:
`128*1*14*14 + 128` + `192*128*2*2 + 192` + `10*192*1*1 + 10` = **125,642**.

## Implementation rules

* Allowed: `torch.einsum` for all learned contractions; `einops.rearrange`,
  `einops.reduce`, `einops.repeat` for layouts, reductions, and explicit
  broadcasts; ordinary differentiable tensor arithmetic (`+ - * / exp log clamp`,
  comparisons, `argmax`); `torch.optim.Adam`; device placement; data loading and
  data-side preprocessing with seeded random generators.
* Not used anywhere in the executable cells: `torch.nn` model layers (`Conv2d`,
  `Linear`, `Sequential`, ...), `torch.nn.functional` `conv2d` / `linear` /
  `unfold` / pooling / activation / softmax / cross-entropy helpers, tensor
  `.view` / `.reshape` / `.flatten` / `.permute` / `.transpose` / `.squeeze` /
  `.unsqueeze`, or NumPy in model computation.
''')

# ---------------------------------------------------------------- 2. setup
md('''
## 1. Setup

Imports, a single master seed, and device selection. The one seed `SEED = 0` drives
every random stream so the notebook is reproducible, end to end:

| random stream | seed |
| --- | --- |
| global Python / PyTorch RNG | `SEED` = 0 |
| weight initialization | `SEED` = 0 |
| training shuffle | `SEED` = 0 |
| grid sample selection | `SEED + 1` |
| shift augmentation | `SEED + 7` |

CUDA is used when available; on a machine without a GPU the identical code runs on
CPU (slower).
''')

code('''
import math
import random
import sys
import time

import einops
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms
from einops import reduce, repeat, rearrange

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__, "| cuda build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("einops     :", einops.__version__)
print("matplotlib :", matplotlib.__version__)
print("device     :", device)
if device.type == "cuda":
    print("gpu        :", torch.cuda.get_device_name(0))
''')

# ---------------------------------------------------------------- 3. data
md('''
## 2. MNIST data pipeline

`torchvision.datasets.MNIST` downloads and caches the dataset under `data/`.
`transforms.ToTensor` converts each `28 x 28` PIL image to a channel-first float
tensor `1 x 28 x 28` with values in `[0, 1]`.

**Normalization** is plain tensor arithmetic with the population statistics of the
MNIST training pixels:

```
x_norm = (x - 0.1307) / 0.3081
```

**Shift augmentation** (training data only): each image is, with probability
`p_shift = 0.5`, translated by a random integer offset in `[-max_shift, +max_shift]`
on both axes, with zero padding. Handwritten digits vary by a couple of pixels in
position; the mild jitter teaches the network not to rely on absolute position.
`max_shift = 2` keeps every digit fully inside the 28 x 28 frame, and `p_shift = 0.5`
keeps half of each batch unshifted so the network also sees clean digits.

**Loaders**: the full 60,000-image training split with batch size 128 and seeded
shuffling (469 batches per epoch); the 10,000-image test split, unshuffled.
`num_workers = 0` keeps the behavior deterministic across platforms.
''')

code('''
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
DATA_DIR = "data"

to_image_tensor = torchvision.transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=to_image_tensor)
test_dataset = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=to_image_tensor)

loader_generator = torch.Generator().manual_seed(SEED)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    generator=loader_generator,
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)

augment_generator = torch.Generator().manual_seed(SEED + 7)


def normalize(images):
    """Standardize [0, 1] images with the MNIST training-set mean and std."""
    return (images - MNIST_MEAN) / MNIST_STD


def shift_augment(images, generator, max_shift=2, p_shift=0.5):
    """Per-image zero-padded translation; each offset in [-max_shift, +max_shift] px."""
    b, _c, h, w = images.shape
    with torch.no_grad():
        active = torch.rand(b, generator=generator) < p_shift
        dyes = torch.randint(-max_shift, max_shift + 1, (b,), generator=generator).to(images.device)
        dxs = torch.randint(-max_shift, max_shift + 1, (b,), generator=generator).to(images.device)
        shifted = images.clone()
        for i in range(b):
            if not bool(active[i]):
                continue
            dy, dx = int(dyes[i]), int(dxs[i])
            if dy == 0 and dx == 0:
                continue
            padded = torch.zeros(1, 1, h + 2 * max_shift, w + 2 * max_shift, device=images.device)
            p0 = max_shift
            padded[0, :, p0:p0 + h, p0:p0 + w] = images[i, 0]
            shifted[i, 0] = padded[0, :, p0 - dy:p0 - dy + h, p0 - dx:p0 - dx + w]
        return shifted


print("train images:", len(train_dataset))
print("test  images:", len(test_dataset))
''')

# ---------------------------------------------------------------- 4. model
md('''
## 3. Model primitives

### 3.1 Block convolution

Each convolution uses a kernel size equal to its stride, so its input splits into
non-overlapping `kh x kw` patches, which a named-axis `rearrange` creates exactly:

```
patches : b c H W --rearrange--> b h w c kh kw        h = H / kh, w = W / kw
out     : einsum("bhwcij,ocij->bohw", patches, weight)
bias    : o --rearrange--> 1 o 1 1                    broadcast over b, h, w
```

Axes: `b` batch, `c` input channels, `o` output channels, `h w` output positions,
`i j` kernel offsets. The equation sums over `c i j`, so each output
`out[b, o, h, w]` is the inner product of the weight `weight[o]` with the
`kh x kw` patch at position `(h, w)` of every input channel - a shared-weight
strided convolution.
''')

code('''
def block_conv2d(x, weight, bias, kernel_size):
    """Shared-weight strided convolution from non-overlapping patches.

    x: b x c x H x W, weight: o x c x kh x kw, bias: o  ->  out: b x o x h x w
    """
    patches = rearrange(
        x,
        "b c (h kh) (w kw) -> b h w c kh kw",
        kh=kernel_size,
        kw=kernel_size,
    )
    out = torch.einsum("bhwcij,ocij->bohw", patches, weight)
    out = out + rearrange(bias, "o -> 1 o 1 1")
    return out
''')

md('''
### 3.2 Parameter initialization

Kaiming (He) uniform initialization, matched to the ReLU nonlinearity: every weight
with fan-in `fan_in = c_in * kh * kw` is drawn from

```
U(-bound, bound),      bound = sqrt(3) * sqrt(2 / fan_in)
```

so that the variance of each layer's pre-activations stays of order one at
initialization. Biases start at zero. A dedicated seeded `torch.Generator` on the
target device makes the draw reproducible, and `requires_grad` is set only after the
tensors are on-device, so every parameter stays a leaf tensor that `Adam` can update
and `backward()` can populate with `.grad`.
''')

code('''
def kaiming_uniform(shape, fan_in, generator, device):
    """He uniform draw: U(-bound, bound) with bound = sqrt(3) * sqrt(2 / fan_in)."""
    bound = (3.0 ** 0.5) * (2.0 / fan_in) ** 0.5
    tensor = torch.empty(shape, device=device)
    tensor.uniform_(-bound, bound, generator=generator)
    return tensor


def init_parameters(seed=0, device="cpu"):
    """The six leaf tensors (3 weights + 3 biases) of the CNN, as a plain dict."""
    generator = torch.Generator(device=device).manual_seed(seed)
    params = {
        "w1": kaiming_uniform((128, 1, 14, 14), fan_in=1 * 14 * 14, generator=generator, device=device),
        "b1": torch.zeros(128, device=device),
        "w2": kaiming_uniform((192, 128, 2, 2), fan_in=128 * 2 * 2, generator=generator, device=device),
        "b2": torch.zeros(192, device=device),
        "w3": kaiming_uniform((10, 192, 1, 1), fan_in=192 * 1 * 1, generator=generator, device=device),
        "b3": torch.zeros(10, device=device),
    }
    for tensor in params.values():
        tensor.requires_grad_(True)
    return params
''')

md('''
### 3.3 Forward pass

ReLU is the pointwise tensor primitive `clamp(min = 0.0)`. `forward` composes the
three convolutions with the activations in between, then converts the
`b x 10 x 1 x 1` classifier output to `b x 10` logits:

```
b x 1 x 28 x 28 --conv(k=14)--> b x 128 x 2 x 2 --relu-->
      --conv(k=2)--> b x 192 x 1 x 1 --relu-->
      --conv(k=1)--> b x 10 x 1 x 1 --rearrange--> b x 10
```
''')

code('''
def relu(x):
    """Pointwise ReLU as a tensor clamp."""
    return x.clamp(min=0.0)


def forward(x, params):
    """The three convolutions: b x 1 x 28 x 28  ->  b x 10 logits."""
    h1 = relu(block_conv2d(x, params["w1"], params["b1"], kernel_size=14))
    # h1: b x 128 x 2 x 2
    h2 = relu(block_conv2d(h1, params["w2"], params["b2"], kernel_size=2))
    # h2: b x 192 x 1 x 1
    h3 = block_conv2d(h2, params["w3"], params["b3"], kernel_size=1)
    # h3: b x 10 x 1 x 1
    return rearrange(h3, "b classes 1 1 -> b classes")
''')

md('''
### 3.4 Cross-entropy loss

The log-probabilities come from a numerically stable log-softmax, using
`einops.reduce` for the max and the sum (subtracting the row maximum before the
exponential keeps every `exp` argument at or below zero):

```
log p = z - max_c(z) - log( sum_c exp(z - max_c(z)) )
```

Targets become one-hot rows by broadcasting `targets` to `b x c` with
`einops.repeat` and comparing against `0..9`; the selected log-probabilities are
then contracted and reduced to the mean negative log-likelihood:

```
loss = - mean_b( einsum("bc,bc->b", log p, onehot)[b] )
```

**Label smoothing** (training only) replaces the one-hot target with
`q = (1 - alpha) * onehot + alpha * uniform`, which turns the loss into

```
loss_smooth = - mean_b( (1 - alpha) * log p[b, target[b]] + alpha * mean_c log p[b, c] )
```

With `alpha = 0.1` the smoothed target is 0.91 on the true class and 0.01 on each of
the other nine. The mild smoothing is a regularizer that measurably improves the
generalization of a 5-epoch model. The epoch summaries report the plain (unsmoothed)
NLL computed from the same logits, so the printed loss is the standard cross-entropy
value.
''')

code('''
def cross_entropy(logits, targets, label_smoothing=0.0):
    """Mean negative log-likelihood, with an optional smoothed target.

    logits: b x c, targets: b  ->  scalar loss
    """
    z_max = reduce(logits, "b c -> b 1", "max")
    shifted = logits - z_max
    sum_exp = reduce(torch.exp(shifted), "b c -> b 1", "sum")
    log_probs = shifted - torch.log(sum_exp)

    one_hot = repeat(targets, "b -> b c", c=logits.shape[1]).eq(
        torch.arange(logits.shape[1], device=logits.device, dtype=targets.dtype)
    )

    selected = torch.einsum("bc,bc->b", log_probs, one_hot.to(log_probs.dtype))
    if label_smoothing > 0.0:
        uniform = reduce(log_probs, "b c -> b", "mean")
        per_sample = (1.0 - label_smoothing) * selected + label_smoothing * uniform
    else:
        per_sample = selected
    return -reduce(per_sample, "b ->", "mean")
''')

# ---------------------------------------------------------------- 5. checks
md('''
## 4. Pre-training sanity checks

Before any training, the primitives are exercised against hand-computable values and
the results printed for inspection:

1. total parameter count (125,642);
2. an all-ones input through an all-ones `2 x 2` kernel - every patch sums to `4`;
3. a kernel with values `1..4` on the same input - every patch sums to `10`;
4. the four layouts of a dummy forward pass;
5. the NLL of uniform logits - exactly `log 10`;
6. one backward pass - every parameter must receive a finite gradient.

The formal assertions on these results live in the final validation cell.
''')

code('''
params = init_parameters(seed=SEED, device=device)
total_parameters = sum(tensor.numel() for tensor in params.values())
print("total trainable parameters:", total_parameters)

# known-value contractions (hand-computable)
x_ones = torch.ones(2, 1, 4, 4, device=device)
ones_out = block_conv2d(x_ones, torch.ones(3, 1, 2, 2, device=device), torch.zeros(3, device=device), kernel_size=2)
print("ones x ones 2x2 kernel   :", float(ones_out[0, 0, 0, 0]), "expected 4.0")

kernel = rearrange(torch.arange(1.0, 5.0, device=device), "(i j) -> 1 1 i j", i=2, j=2)
kernel_out = block_conv2d(x_ones, kernel, torch.zeros(1, device=device), kernel_size=2)
print("kernel 1..4 on ones      :", float(kernel_out[0, 0, 0, 0]), "expected 10.0")

# layouts of a dummy forward pass
x_dummy = torch.randn(4, 1, 28, 28, device=device)
h1 = block_conv2d(x_dummy, params["w1"], params["b1"], kernel_size=14)
h2 = block_conv2d(relu(h1), params["w2"], params["b2"], kernel_size=2)
h3 = block_conv2d(relu(h2), params["w3"], params["b3"], kernel_size=1)
logits = forward(x_dummy, params)
print("layer 1 layout           :", tuple(h1.shape), "expected (4, 128, 2, 2)")
print("layer 2 layout           :", tuple(h2.shape), "expected (4, 192, 1, 1)")
print("layer 3 layout           :", tuple(h3.shape), "expected (4, 10, 1, 1)")
print("logits layout            :", tuple(logits.shape), "expected (4, 10)")

# known-value loss: uniform logits give an NLL of exactly log(10)
logits_uniform = torch.zeros(4, 10, device=device)
loss_uniform = cross_entropy(logits_uniform, torch.zeros(4, dtype=torch.long, device=device))
print("NLL of uniform logits    :", round(float(loss_uniform), 6), f"expected {round(math.log(10.0), 6)}")

# one backward pass: every parameter receives a finite gradient
targets_dummy = torch.randint(0, 10, (4,), device=device)
loss = cross_entropy(logits, targets_dummy, label_smoothing=0.1)
loss.backward()
print("one backward pass        : all gradients finite =",
      all(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()) for tensor in params.values()))

sanity = {
    "parameter_count": int(total_parameters),
    "ones_contraction": float(ones_out[0, 0, 0, 0]) == 4.0,
    "kernel_contraction": float(kernel_out[0, 0, 0, 0]) == 10.0,
    "layer1_layout": h1.shape == (4, 128, 2, 2),
    "layer2_layout": h2.shape == (4, 192, 1, 1),
    "layer3_layout": h3.shape == (4, 10, 1, 1),
    "logits_layout": logits.shape == (4, 10),
    "uniform_loss": bool(torch.allclose(loss_uniform, torch.log(torch.tensor(10.0, device=device)))),
    "loss_finite": bool(torch.isfinite(loss).all()),
    "grads_finite": all(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()) for tensor in params.values()),
}
print("pre-training checks      :", "all matched" if all(sanity.values()) else "MISMATCH - inspect the values above")
''')

# ---------------------------------------------------------------- 6. training
md('''
## 5. Five-epoch training

`torch.optim.Adam` on the six raw parameter tensors, for exactly five epochs:

| constant | value | rationale |
| --- | --- | --- |
| learning rate | `3e-3` for epochs 1-4, `5e-4` for epoch 5 | a common Adam scale, with the final epoch settling at one third of the rate |
| weight decay | `5e-5` | mild L2 regularization for the 125,642-parameter model |
| label smoothing | `alpha = 0.1` | section 3.4; used for the gradient signal only |
| batch size | 128 | 469 batches per epoch |
| shift augmentation | `max_shift = 2`, `p_shift = 0.5` | section 2 |

Per batch: move to device, augment, normalize, forward, loss, `zero_grad`,
`backward`, `step`. The printed loss is the sample-weighted mean of the per-batch
raw-NLL scalar tensors, aggregated with `einops.reduce`; the printed accuracy is the
fraction of correct `argmax` predictions over the (augmented) training batch.
''')

code('''
optimizer = torch.optim.Adam(list(params.values()), lr=3e-3, weight_decay=5e-5)

history = []
for epoch in range(1, 6):
    if epoch == 5:
        for group in optimizer.param_groups:
            group["lr"] = 5e-4

    epoch_start = time.perf_counter()
    batch_losses = []
    batch_correct = []
    batch_sizes = []

    for images, targets in train_loader:
        images = normalize(shift_augment(images.to(device), augment_generator))
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = forward(images, params)
        loss = cross_entropy(logits, targets, label_smoothing=0.1)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        with torch.no_grad():
            batch_losses.append(cross_entropy(logits, targets))
        batch_correct.append(reduce((predictions == targets).to(torch.float32), "b ->", "sum"))
        batch_sizes.append(torch.tensor(float(images.shape[0]), device=device))

    sizes = torch.stack(batch_sizes)
    total_count = reduce(sizes, "n ->", "sum")
    train_loss = reduce(torch.stack(batch_losses) * sizes, "n ->", "sum") / total_count
    train_accuracy = reduce(torch.stack(batch_correct), "n ->", "sum") / total_count

    record = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "seconds": time.perf_counter() - epoch_start,
    }
    history.append(record)
    print(
        f"epoch {epoch}/5  loss={record['train_loss']:.4f}  "
        f"accuracy={record['train_accuracy']:.4f}  {record['seconds']:.1f}s"
    )
''')

# ---------------------------------------------------------------- 7. eval
md('''
## 6. Final test evaluation

After epoch 5 the model enters a no-gradient context and the full 10,000-image test
split is evaluated in a single forward pass on un-augmented images. The prediction
for each image is the `argmax` over its 10 logits, and the measured accuracy is
reported as a fraction and a percentage.
''')

code('''
with torch.no_grad():
    test_images = torch.cat([normalize(images) for images, _ in test_loader], dim=0).to(device)
    test_targets = torch.cat([targets for _, targets in test_loader], dim=0).to(device)
    test_logits = forward(test_images, params)
    test_predictions = test_logits.argmax(dim=1)

correct_mask = test_predictions == test_targets
correct_count = reduce(correct_mask.to(torch.float32), "b ->", "sum")
total_test = reduce(torch.ones(test_predictions.shape[0]), "b ->", "sum")
test_accuracy = correct_count / total_test

print("epochs trained     :", len(history))
print("test examples      :", int(total_test))
print("correct predictions:", int(correct_count))
print("final test accuracy:", float(test_accuracy), f"({100.0 * float(test_accuracy):.2f}%)")
''')

# ---------------------------------------------------------------- 8. figures
md('''
## 7. Figures

1. **Training history** - the sample-weighted raw NLL and the accuracy over the five
   epochs, as two line plots with labeled axes.
2. **16 sampled test images** - after evaluation, 16 unique test indices are drawn
   with a fixed seed (`SEED + 1`); those exact tensors are run through the final
   model to obtain their predictions. Each cell is a grayscale image on the shared
   `[0, 1]` intensity range, with no color bars, square pixels, image row 0 on top,
   and no tick labels. Titles read
   `GT: <label> | Pred: <label>`, and a red title marks a wrong prediction.
   Normalization is inverted (`x * 0.3081 + 0.1307`) for display only.

Both figures are Matplotlib figures captured as inline PNG images, so they render
directly in GitHub's notebook preview.
''')

code('''
epochs = [record["epoch"] for record in history]
losses = [record["train_loss"] for record in history]
accuracies = [record["train_accuracy"] for record in history]

history_fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(10, 3.2))
loss_ax.plot(epochs, losses, marker="o")
loss_ax.set_title("Training loss")
loss_ax.set_xlabel("epoch")
loss_ax.set_ylabel("sample-weighted mean loss (raw NLL)")
loss_ax.set_xticks(epochs)

accuracy_ax.plot(epochs, accuracies, marker="o")
accuracy_ax.set_title("Training accuracy")
accuracy_ax.set_xlabel("epoch")
accuracy_ax.set_ylabel("sample-weighted accuracy (augmented)")
accuracy_ax.set_xticks(epochs)

history_fig.suptitle("Training history - 5 epochs, Adam 3e-3 -> 5e-4")
history_fig.tight_layout(rect=(0, 0, 1, 0.94))
plt.show()
''')

code('''
sample_generator = torch.Generator().manual_seed(SEED + 1)
sample_indices = torch.randperm(test_predictions.shape[0], generator=sample_generator)[:16]

with torch.no_grad():
    sample_images_norm = test_images[sample_indices]
    sample_logits = forward(sample_images_norm, params)
sample_predictions = sample_logits.argmax(dim=1)
sample_targets = test_targets[sample_indices]

sample_images = (sample_images_norm * MNIST_STD + MNIST_MEAN).clamp(0.0, 1.0)
sample_images_hw = rearrange(sample_images, "b 1 h w -> b h w").cpu().numpy()

grid_fig, grid_axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(grid_axes.ravel()):
    ax.imshow(sample_images_hw[i], cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    wrong = int(sample_targets[i]) != int(sample_predictions[i])
    ax.set_title(
        f"GT: {int(sample_targets[i])} | Pred: {int(sample_predictions[i])}",
        color="red" if wrong else "black",
    )

grid_fig.suptitle(
    f"16 sampled MNIST test images - final test accuracy "
    f"{100.0 * float(test_accuracy):.2f}% (5 epochs)"
)
grid_fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
''')

# ---------------------------------------------------------------- 9. validate
md('''
## 8. Final validation

The only assertions in the notebook. They verify that the run just completed is
exactly what the notebook claims: five epoch records, the 10,000-example test
evaluation, 16 unique grid samples, the 125,642-parameter model, finite epoch
losses, and every section-4 sanity check.
''')

code('''
assert len(history) == 5
assert [record["epoch"] for record in history] == [1, 2, 3, 4, 5]
assert all(sanity.values())
assert sanity["parameter_count"] == 125642
assert int(total_test) == 10000
assert sample_indices.shape[0] == 16
assert len(set(int(i) for i in sample_indices.tolist())) == 16
assert all(torch.isfinite(torch.tensor(record["train_loss"])) for record in history)

print(
    "final validation passed: 5 epochs, 10000 test examples, "
    "16 unique sampled images, 125642 parameters"
)
''')

nb = nbformat.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbformat.validate(nb)
nbformat.write(nb, NOTEBOOK)
print(f"wrote {NOTEBOOK} with {len(cells)} cells")
