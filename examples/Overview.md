Overview

If your signal shows self-similarity across scales and structure that persists into higher time frames, the best approach is usually not a plain LSTM-first workflow.
You will likely get better results from a multi-scale model, with TCN / dilated 1D CNN as the first model to try, and possibly a wavelet or multiresolution front end before the network.

What your description implies
A process with “fractal” propagation across higher time frames usually suggests:

scale invariance or approximate scale invariance
long-range dependence
repeating motifs across resolutions
possible power-law behavior
strong benefit from analyzing both local and global temporal structure

That means the model should be able to:

see short-term details
preserve long-horizon dependencies
compare patterns across multiple resolutions
avoid forgetting older structure too quickly

Best neural-network direction for this case

1. Best first choice: TCN / Dilated 1D CNN

This is likely your strongest starting point.

Why it fits:

dilated convolutions naturally capture patterns across multiple temporal scales
easier to train than recurrent models
often more memory-efficient than Transformers
good fit for your RTX 3060 12GB
can be made compact enough for 1060 3GB inference

This matters because fractal-like propagation often means patterns recur at wider and wider receptive fields.
A dilated CNN expands receptive field efficiently without exploding parameter count.

2. Very strong option: Wavelet + Neural Network

If your signal is truly multiscale/fractal, this can be even more aligned with the physics of the data.

Typical setup:

decompose signal into multiple scales using:
wavelets
multiresolution analysis
scattering transform
feed coefficients into:
TCN
small MLP
CNN
hybrid model

Why this is attractive:

fractal structure often becomes more separable in the time-frequency / multiscale domain
reduces burden on the network to discover scale structure from raw input alone
3. Useful if very long context matters: Transformer with patching/downsampling

Only use this if:

long-range interactions are truly critical
dataset is large enough
you can control memory carefully

Transformers can model cross-scale behavior, but they are often heavier than needed.
For your hardware, they are usually a second-stage experiment, not the first model.

4. Less ideal as first model: LSTM / GRU

They can work, especially for moderate-length sequences, but for fractal, multiscale behavior they are often not the most natural choice.

Main limitations:

weaker inductive bias for scale hierarchy
harder to preserve very long dependencies
slower training
less efficient than TCNs in many real signal tasks
Best architecture ranking for your problem

For your description, I would rank them like this:

Wavelet + TCN
Raw-signal TCN / dilated 1D CNN
Multibranch CNN with different kernel sizes
Temporal Transformer with downsampling/patching
GRU/LSTM
Best framework for this specific problem

Still:

PyTorch for development and training
ONNX Runtime for inference/deployment

Why this matters here:

fractal and multiscale architectures are easier to customize in PyTorch
TCNs, wavelet hybrids, and custom multibranch nets are straightforward to build
ONNX export is good for compact inference on the 1060 3GB
What I would build first
Model 1: Baseline multiscale TCN

Input:

raw time-series windows

Architecture:

several 1D convolution blocks
increasing dilation rates
residual connections
global pooling
output head for your task

Why:

simplest strong baseline
very good fit for scale-propagating patterns
Model 2: Wavelet-TCN hybrid

Input:

raw signal transformed into multiscale wavelet coefficients

Architecture:

one branch per scale, or concatenated scale features
small TCN or CNN head
fusion layer
prediction head

Why:

explicitly respects the multiscale/fractal hypothesis
Model 3: Dual-resolution model

Input:

same sequence represented at:
original resolution
coarser aggregation
even coarser aggregation

Architecture:

separate CNN/TCN branch for each resolution
feature fusion at the end

Why:

practical approximation of self-similar propagation across time frames
Important modeling idea

If the process is fractal across time frames, do not train only on one sampling resolution and expect the model to generalize perfectly across scales.

Better:

train with multi-resolution inputs
augment by resampling/downsampling
expose the model to equivalent structures across scales
optionally predict both:
local target
coarse-scale target

That teaches the model scale-consistent representations.

Best deployment choice for your GPUs
Training on RTX 3060 12GB

Good for:

TCNs
multibranch CNNs
wavelet hybrids
modest Transformers
Inference on GTX 1060 3GB

Best to deploy:

compact TCN
quantized or reduced-width CNN
ONNX Runtime inference
batch size 1 if real-time

Avoid very wide or deep Transformers for that card.

Best practical recommendation

For your case, I would do this:

Primary stack

PyTorch
TCN or Wavelet-TCN
ONNX Runtime for execution

First experiment

build a dilated residual TCN
compare against:
simple 1D CNN
GRU baseline
then add:
wavelet decomposition
multi-timeframe branches
One caution

Sometimes people call a process “fractal” when it is really just:

persistent
nonstationary
regime-switching
heavy-tailed
long-memory but not truly scale-invariant

So before locking the architecture, it is worth checking:

Hurst exponent
power spectral slope
DFA / detrended fluctuation analysis
whether scaling law is stable across windows

That helps confirm whether the multiscale assumption is truly present.

Bottom line

For a time series whose features propagate in a fractal-like way across higher time frames, the best neural approach is usually:

PyTorch + multiscale TCN (preferably wavelet-assisted) + ONNX Runtime for inference

A plain LSTM is usually not the best first choice here.

If you want, next I can outline the best end-to-end modeling strategy for this exact problem:
windowing, labels, preprocessing, feature extraction, model architecture, and validation design.