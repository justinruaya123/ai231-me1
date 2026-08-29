# Plan: Three-Layer MNIST CNN with `einops` and `einsum`

## Goal

Create one reproducible, executed Jupyter notebook, `mnist_einops_einsum_cnn.ipynb`, that:

- builds a three-trainable-layer CNN from raw PyTorch tensors;
- implements every learned contraction with `torch.einsum` and every tensor layout change or reduction with `einops`;
- does not use PyTorch CNN, MLP, pooling, flattening, or loss layers;
- trains on the official MNIST training split for exactly 5 epochs;
- evaluates once on all 10,000 examples in the official test split and reports the measured accuracy; and
- displays 16 sampled test images in a Plotly 4 x 4 grid, with ground-truth and predicted labels on every image.

## Constraint interpretation

PyTorch remains the tensor/autograd backend and may provide device placement, random initialization, scalar elementwise functions, data loading, and an optimizer. It will not provide any model or loss layer.

Required implementation primitives:

- `einops.rearrange` for patch extraction, axis composition/decomposition, and removing the final unit spatial axes;
- `einops.reduce` for batch means, sums, maxima, and accuracy aggregation;
- `einops.repeat` only where explicit broadcasting is useful;
- `torch.einsum` for every trainable convolutional contraction and label/log-probability contraction; and
- ordinary differentiable tensor scalar operations only where the two libraries do not represent the operation, such as addition, subtraction, multiplication, `exp`, `log`, and ReLU/clamping.

The notebook must not call any of the following:

- `torch.nn.Conv2d`, `torch.nn.Linear`, `torch.nn.Sequential`, or another ready-made model layer;
- `torch.nn.functional.conv2d`, `linear`, `unfold`, pooling, activation, softmax, or cross-entropy helpers;
- tensor `.view`, `.reshape`, `.flatten`, `.permute`, `.transpose`, `.squeeze`, or `.unsqueeze`; or
- NumPy for model computation.

Parameters will be leaf tensors with `requires_grad=True`, stored in a small dictionary and passed directly to `torch.optim.Adam`. This keeps the model visibly tensor-based and avoids using `nn.Module` as an implicit layer abstraction.

## Model design

“Three-layer CNN” will mean three trainable convolutional layers. ReLU operations and the final logits layout conversion are not counted as layers. The last convolution acts as the classifier, so there is no MLP or fully connected head.

To satisfy the strict `einops`/`einsum` implementation constraint, each convolution uses a kernel size equal to its stride. Non-overlapping patches can then be created exactly with:

```python
patches = rearrange(
    x,
    "b c (h kh) (w kw) -> b h w c kh kw",
    kh=kernel_size,
    kw=kernel_size,
)
out = torch.einsum("bhwcij,ocij->bohw", patches, weight)
out = out + rearrange(bias, "o -> 1 o 1 1")
```

This is a valid strided 2-D convolution with shared weights and no padding, while avoiding `Conv2d`, `unfold`, `as_strided`, or overlapping-window helpers that are outside `einops`.

| Stage | Parameters / operation | Output shape |
| --- | --- | --- |
| Input | normalized MNIST tensor | `b x 1 x 28 x 28` |
| Layer 1 | block convolution, `1 -> 16`, kernel/stride `2`, bias | `b x 16 x 14 x 14` |
| Activation 1 | tensor ReLU | `b x 16 x 14 x 14` |
| Layer 2 | block convolution, `16 -> 32`, kernel/stride `2`, bias | `b x 32 x 7 x 7` |
| Activation 2 | tensor ReLU | `b x 32 x 7 x 7` |
| Layer 3 | classifier convolution, `32 -> 10`, kernel/stride `7`, bias | `b x 10 x 1 x 1` |
| Logits | `rearrange("b classes 1 1 -> b classes")` | `b x 10` |

The model has 17,850 trainable scalars:

- Layer 1: `16 * 1 * 2 * 2 + 16 = 80`;
- Layer 2: `32 * 16 * 2 * 2 + 32 = 2,080`; and
- Layer 3: `10 * 32 * 7 * 7 + 10 = 15,690`.

Weights will use a reproducible Kaiming-style random initialization computed from tensor dimensions. Biases start at zero.

## Notebook structure

### 1. Introduction and reproducibility

- State the goal, the exact definition of the three layers, and the allowed/prohibited operations.
- Link the two basis notebooks.
- Import only the necessary packages: PyTorch/torchvision, `einops`, Plotly, and standard-library utilities.
- Fix Python and PyTorch seeds and create a seeded data-loader generator.
- Select CUDA when available and otherwise use CPU; print the device and package versions.
- Keep `num_workers=0` so the notebook behaves consistently on Windows and in hosted notebook environments.

### 2. MNIST data pipeline

- Download/cache MNIST under `data/` through `torchvision.datasets.MNIST`.
- Convert each image to a float tensor in channel-first layout and normalize it with the standard MNIST mean and standard deviation using tensor arithmetic.
- Use the complete official 60,000-image training split and 10,000-image test split; do not train on or tune against test examples.
- Use a training batch size of 128 with seeded shuffling and a non-shuffled test loader.
- Preserve an unnormalized copy, or invert normalization, only for the final image visualization.

### 3. Tensor-only model primitives

Implement small functions in separate, readable cells:

1. `init_parameters(...)`: returns the six leaf tensors for three weights and three biases.
2. `block_conv2d(x, weight, bias, kernel_size)`: patchifies with `rearrange`, contracts with `torch.einsum`, and broadcasts bias with `rearrange`.
3. `relu(x)`: applies only a pointwise tensor primitive.
4. `forward(x, params)`: composes exactly the three convolutional layers and two activations, then converts `b x 10 x 1 x 1` to `b x 10` with `rearrange`.
5. `cross_entropy(logits, targets)`: implements stable log-softmax from tensor exponentials/logarithms plus `einops.reduce`, forms tensor one-hot targets, contracts the selected log probabilities with `torch.einsum`, and reduces to a scalar mean without `nn.CrossEntropyLoss` or `F.cross_entropy`.

Each function will include its expected named axes and input/output shapes in the surrounding Markdown.

### 4. Pre-training correctness checks

Run fast checks before downloading/training is allowed to obscure a tensor bug:

- check a known all-ones input/kernel case, where every 2 x 2 contraction has a known value;
- assert every intermediate shape in a dummy forward pass;
- assert the exact parameter count is 17,850;
- run one backward pass and assert that every parameter receives a finite gradient; and
- verify loss finiteness and that logits have shape `batch x 10`.

The checks will not use a PyTorch CNN/MLP implementation as a hidden reference path.

### 5. Five-epoch training

- Use Adam with a documented initial learning rate of `1e-3` and no scheduler unless a correctness issue is found before the final run.
- Train for the literal `range(1, 6)` and no more.
- For each batch: move tensors to the selected device, compute logits and the manual loss, zero gradients, call `backward()`, and update the raw parameter tensors.
- Aggregate sample-weighted training loss and training accuracy with `einops.reduce`-based tensor reductions.
- Store one history record per epoch and print a compact epoch summary including loss, accuracy, and elapsed time.
- Do not select checkpoints or hyperparameters using the test split.

### 6. Final test evaluation and report

- Switch to a no-gradient evaluation context after epoch 5.
- Evaluate all 10,000 official test examples exactly once.
- Compute `argmax` predictions and accumulate correct/total counts.
- Display the final test accuracy as both a fraction and percentage, alongside the evaluated sample count and the fixed epoch count.
- Report the measured result as-is. Do not hard-code or pre-claim an accuracy value; if the result is unexpectedly low, diagnose the implementation while retaining the five-epoch requirement and then rerun the notebook from a clean kernel.

### 7. Plotly outputs

All figures in the notebook will be Plotly figures; Matplotlib and Seaborn will not be imported.

- Plot a compact epoch-history figure for training loss and training accuracy, with labeled axes and hover details.
- Select 16 unique test indices with a fixed random seed after evaluation.
- Run those exact tensors through the final model and construct a 4 x 4 `plotly.subplots.make_subplots` grid.
- Render each digit as a grayscale `go.Heatmap` with a shared `[0, 1]` intensity range, hidden color bars, equal aspect ratio, reversed image y-axes, and no tick labels.
- Title every cell `GT: <label> | Pred: <label>` and visually distinguish incorrect predictions without hiding them or resampling.
- Give the figure a clear overall title containing the final test accuracy.

### 8. Final notebook verification

- Restart the kernel and run all cells top-to-bottom so the committed notebook contains the real five-epoch outputs and final figures.
- Confirm there are no execution errors, stale counters, or results from a different run.
- Search notebook code cells for every prohibited layer/helper and opaque layout operation listed above.
- Confirm the history has exactly five epochs, the final evaluation count is 10,000, and the sampled grid has exactly 16 unique examples arranged 4 x 4.
- Confirm every displayed figure is Plotly-based and usable inline without an external GUI.
- Run a notebook-format validation and a whitespace/diff check before handoff.

## Planned deliverables

1. `mnist_einops_einsum_cnn.ipynb` — the fully executed narrative, implementation, training report, and Plotly figures.
2. A minimal dependency declaration if the environment does not already provide a reproducible notebook stack; it will include only PyTorch, torchvision, einops, Plotly, Jupyter, and notebook validation support.

Large downloaded MNIST files and transient notebook checkpoints will remain outside version control.

## Acceptance checklist

- [ ] The notebook runs from a clean kernel and uses tensors throughout.
- [ ] Exactly three trainable convolutional layers are implemented without a PyTorch CNN or MLP layer.
- [ ] Learned contractions use `torch.einsum`; layouts/reductions use `einops` with named axes.
- [ ] No prohibited model, loss, pooling, unfold, or opaque reshaping helper appears in executable cells.
- [ ] Training completes exactly 5 epochs on MNIST's official training split.
- [ ] Accuracy is measured over and reported for all 10,000 official test images.
- [ ] Sixteen unique sampled test images appear in a Plotly 4 x 4 grid with both GT and prediction.
- [ ] All figures are Plotly figures and all requested outputs are visible inside the saved notebook.
