# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Trains the neural OIT network of `ARM_002_NeuralOIT` (DFAOIT, Tsopouridis et al., CGF 2024) and writes
`dfaoit-<hidden1>x<hidden2>.bin` next to it; the defaults train the deployed `dfaoit-16x16.bin`. The `README.md` here says
how it was trained and how to train and check another; the training data comes from `dfaoit_scenes.py`.

The features are computed exactly as the sample computes them at run time, including the 8-bit quantization of the k
nearest fragments (the sample keeps them as RGBA8), so that training inputs and inference inputs agree:

    inputs  = [a_avg, C_avg.rgb, C_acc.rgb / (1 + C_acc.rgb), g.rgb, a_0 .. a_{k-1}]
              a_avg, C_avg  average opacity and colour of the fragments behind the k nearest, divided by n - k
              C_acc         sum of a_i * C_i over every fragment, the only feature that is not bounded by one, so it
                            is squashed into [0, 1) the way the paper suggests in its section 5.3
              g             the k nearest composited exactly, front to back
              a_0 .. a_k-1  their opacity, whose product is what the colour behind them is attenuated by
    target  = the tail, the exact composite of the fragments behind the k nearest ones

The network predicts the tail rather than the whole pixel. The composite splits exactly into

    colour = g + prod(1 - a_i) * tail

and both `g` and the attenuation are already known exactly at inference, so regressing the whole colour would spend the
network on reproducing a term it is handed as an input.

The error of the tail is weighted by a power of that attenuation. The square of it makes the loss exactly the mean
squared error of the image, and also makes a pixel behind dense surfaces almost invisible to the optimizer, which is
where the network turns out to be relatively worst; dropping the weight entirely abandons the opacities that carry most
of the error instead. `--loss-power 0.5` measured best on Bistro on both counts at once.

Only pixels with more than NUM_NEAREST fragments are generated: the sample composites the other ones exactly and never
calls the network. NUM_NEAREST is the DFAOIT_k of the paper's Table 1; the sample reads it back out of the network as
`NUM_INPUTS - 10` and sizes its buffer of exactly kept fragments accordingly, so changing it here is enough.

The network ships quantized to 8 bits, the integer graph that hardware running data graphs natively evaluates. It is
trained in float first, then fine-tuned for a few epochs with the quantization in the loop (weights per output channel
to int8, the input and every activation to 8 bits, the logits to int8 and the sigmoid to a 256-entry table), so that it
learns to live on the grid the integer graph will evaluate it on. The graph is TOSA's own integer scheme: every layer
is an int8 CONV2D accumulating into int32, followed by a RESCALE that multiplies by a fixed-point scale per output
channel and saturates to int8. All the features are in [0, 1], so the input is stored as `round(x * 255) - 128`; a
hidden layer is stored the same way, which makes its ReLU free, since anything negative saturates at -128 = 0; the
last layer's logits are int8 with a zero point of 0 and a TABLE looks the sigmoid up. The script evaluates the integer
graph itself, bit for bit as the emulation layer computes it, and reports its validation error next to the float
network's, so the price of the quantization is known before the sample runs. On Bistro it is a few percent of the
mean squared error, all of it the resolution of the int8 weights: the 8-bit inputs, activations and sigmoid table cost
nothing measurable.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

CONTENT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(CONTENT))  # for the shared tosa_int8.py

import dfaoit_scenes
from tosa_int8 import quantizeMultiplier, rescale, roundSte

try:
    import torch
except ImportError:
    torch = None

ACTIVATION_RELU = 1
ACTIVATION_SIGMOID = 2

NUM_NEAREST = 4
NUM_INPUTS = 10 + NUM_NEAREST
NUM_HIDDEN1 = 16
NUM_HIDDEN2 = 16
NUM_OUTPUTS = 3

MAGIC = 0x51504C4D

UNORM_ZERO_POINT = -128
LOGIT_LIMIT = 6.24


def quantize(x):
    """The 8-bit quantization of `packUnorm4x8()`, which the sample stores the nearest fragments with."""
    return np.round(np.clip(x, 0.0, 1.0) * 255.0) / 255.0


def computeFeatures(color, alpha, count):
    """The features and the tail target of depth-ordered fragment lists. `color` is (..., fragments, 3) and `alpha` is
    (..., fragments), both sorted front to back with everything past `count` zeroed. This is the one place the feature
    convention lives, and the sample's `loadPixel()` has to agree with it exactly."""
    sumAlpha = alpha.sum(axis=-1)
    sumColor = color.sum(axis=-2)
    accumulated = (alpha[..., None] * color).sum(axis=-2)

    transmittance = np.ones_like(alpha)
    np.cumprod(1.0 - alpha[..., :-1], axis=-1, out=transmittance[..., 1:])
    target = ((alpha * transmittance)[..., None] * color).sum(axis=-2)

    nearColor = quantize(color[..., :NUM_NEAREST, :])
    nearAlpha = quantize(alpha[..., :NUM_NEAREST])

    nearest = np.zeros(color.shape[:-2] + (3,), np.float32)
    attenuation = np.ones(color.shape[:-2], np.float32)
    for i in range(NUM_NEAREST):
        nearest += (attenuation * nearAlpha[..., i])[..., None] * nearColor[..., i, :]
        attenuation = attenuation * (1.0 - nearAlpha[..., i])

    behind = np.maximum(count - NUM_NEAREST, 1).astype(np.float32)
    averageAlpha = np.maximum(sumAlpha - nearAlpha.sum(axis=-1), 0.0) / behind
    averageColor = np.maximum(sumColor - nearColor.sum(axis=-2), 0.0) / behind[..., None]
    squashed = accumulated / (1.0 + accumulated)

    inputs = np.concatenate([averageAlpha[..., None], averageColor, squashed, nearest, nearAlpha], axis=-1)
    tail = np.clip((target - nearest) / np.maximum(attenuation, 1e-6)[..., None], 0.0, 1.0)

    return inputs.astype(np.float32), tail.astype(np.float32), attenuation.astype(np.float32)


def generatePixels(numSamples, rng, shallowFraction, shallowMax, tileSize):
    """Fires tiles of orthographic rays through little scenes of spheres, boxes, quads and tori. The tile is only how
    the rays are sampled; what comes back is a flat list of pixels, filtered down to the ones the sample actually asks
    the network about."""
    inputs = np.empty((numSamples, NUM_INPUTS), np.float32)
    targets = np.empty((numSamples, NUM_OUTPUTS), np.float32)
    weights = np.empty(numSamples, np.float32)
    filled = 0
    started = time.time()

    while filled < numSamples:
        primitives = dfaoit_scenes.sampleDepthComplexity(rng, shallowFraction, shallowMax)
        tiles = dfaoit_scenes.sampleTiles(max(24, 6000 // primitives), primitives, tileSize, rng)
        x, tail, attenuation = computeFeatures(tiles.color, tiles.alpha, tiles.count)

        keep = (tiles.count > NUM_NEAREST).ravel()
        x = x.reshape(-1, NUM_INPUTS)[keep]
        tail = tail.reshape(-1, NUM_OUTPUTS)[keep]
        attenuation = attenuation.ravel()[keep]

        count = min(numSamples - filled, len(x))
        inputs[filled:filled + count] = x[:count]
        targets[filled:filled + count] = tail[:count]
        weights[filled:filled + count] = attenuation[:count]
        filled += count
        if filled % 1_000_000 < count:
            print(f'  generated {filled / 1e6:.1f}M of {numSamples / 1e6:.1f}M pixels', flush=True)

    print(f'  {numSamples / 1e6:.1f}M pixels in {time.time() - started:.0f} s')
    return inputs, targets, weights


# one layer of the integer graph: an int8 CONV2D into an int32 sum, then a RESCALE by `multiplier * 2^-shift` per output
# channel to int8 around `outputZeroPoint`, then the sigmoid table if there is one
QuantizedLayer = collections.namedtuple('QuantizedLayer', 'weights biases multiplier shift outputZeroPoint activation table')


def weightScale(weight):
    """Symmetric int8 per output channel: the largest magnitude of a row maps to 127."""
    return weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-4) / 127.0


def linearLayers(network):
    return [module for module in network if isinstance(module, torch.nn.Linear)]


def calibrate(network, inputs):
    """The ranges the integer graph is built with: the largest activation of every hidden layer, which becomes its 255,
    and the largest logit the sigmoid table has to cover, capped where the sigmoid is already 0 or 1 in 8 bits."""
    layers = linearLayers(network)
    limits = []
    with torch.no_grad():
        x = inputs
        for i, layer in enumerate(layers):
            y = layer(x)
            if i + 1 < len(layers):
                x = torch.relu(y)
                limits.append(x.amax().item())
            else:
                limits.append(min(LOGIT_LIMIT, y.abs().amax().item()))
    return limits


def forwardQuantized(network, x, limits):
    """The network evaluated the way the integer graph evaluates it: every quantity rounded onto the grid the graph
    stores it on, with the gradient passed straight through the rounding."""
    layers = linearLayers(network)
    step = 1.0 / 255.0
    x = roundSte(x.clamp(0.0, 1.0) * 255.0) * step
    for i, layer in enumerate(layers):
        scale = weightScale(layer.weight.detach())
        weight = roundSte(layer.weight / scale) * scale
        biasScale = step * scale[:, 0]
        bias = roundSte(layer.bias / biasScale) * biasScale
        y = torch.nn.functional.linear(x, weight, bias)
        if i + 1 < len(layers):
            step = limits[i] / 255.0
            x = roundSte(y / step).clamp(0.0, 255.0) * step
        else:
            step = limits[i] / 127.0
            logits = roundSte(y / step).clamp(-128.0, 127.0) * step
            x = roundSte(torch.sigmoid(logits) * 255.0) / 255.0
    return x


def quantizeNetwork(network, limits):
    """The integer graph of a network: int8 weights, int32 biases and the RESCALE of every layer, with the sigmoid of
    the last one as a 256-entry table."""
    quantized = []
    inputScale = 1.0 / 255.0
    layers = linearLayers(network)

    for i, layer in enumerate(layers):
        weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        bias = layer.bias.detach().cpu().numpy().astype(np.float64)
        scale = weightScale(torch.from_numpy(weight)).numpy()[:, 0]
        weights = np.clip(np.round(weight / scale[:, None]), -127, 127).astype(np.int8)
        biases = np.round(bias / (inputScale * scale.astype(np.float64)))
        assert np.abs(biases).max() < 2 ** 31, 'a bias does not fit int32'

        if i + 1 < len(layers):
            outputScale = limits[i] / 255.0
            outputZeroPoint = UNORM_ZERO_POINT
            activation = ACTIVATION_RELU
            table = None
        else:
            outputScale = limits[i] / 127.0
            outputZeroPoint = 0
            activation = ACTIVATION_SIGMOID
            logits = np.arange(-128, 128, dtype=np.float64) * outputScale
            sigmoid = 1.0 / (1.0 + np.exp(-logits))
            table = np.clip(np.round(sigmoid * 255.0) + UNORM_ZERO_POINT, -128, 127).astype(np.int8)

        multiplier, shift = quantizeMultiplier(inputScale * scale / outputScale)
        quantized.append(QuantizedLayer(weights, biases.astype(np.int32), multiplier, shift, outputZeroPoint, activation, table))
        inputScale = outputScale

    return quantized


def runInteger(quantized, inputs):
    """Evaluates the integer graph exactly, in the integers the data graph uses, on float inputs in [0, 1]."""
    q = np.clip(np.round(inputs.astype(np.float64) * 255.0) + UNORM_ZERO_POINT, -128, 127).astype(np.int8)
    zeroPoint = UNORM_ZERO_POINT
    for layer in quantized:
        centred = q.astype(np.int64) - zeroPoint
        acc = centred @ layer.weights.astype(np.int64).T + layer.biases.astype(np.int64)
        assert np.abs(acc).max() < 2 ** 31, 'an accumulator does not fit int32'
        q = rescale(acc, layer.multiplier, layer.shift, layer.outputZeroPoint).astype(np.int8)
        zeroPoint = layer.outputZeroPoint
        if layer.table is not None:
            q = layer.table[q.astype(np.int32) + 128]
    return (q.astype(np.float32) - UNORM_ZERO_POINT) / 255.0


def writeWeights(quantized, path):
    """The little container `ARM_002_NeuralOIT` reads:
        uint32 magic 'MLPQ', uint32 numInputs, uint32 numLayers, int32 inputZeroPoint
        per layer: uint32 numOutputs, uint32 activation (1 ReLU, 2 sigmoid), int32 outputZeroPoint
        then, per layer, numOutputs * numInputs int8 weights, numOutputs int32 biases, numOutputs int32 multipliers,
        numOutputs int8 shifts, and 256 int8 table entries for a sigmoid"""
    header = [MAGIC, NUM_INPUTS, len(quantized), UNORM_ZERO_POINT & 0xFFFFFFFF]
    for layer in quantized:
        header += [layer.weights.shape[0], layer.activation, layer.outputZeroPoint & 0xFFFFFFFF]

    with open(path, 'wb') as f:
        f.write(np.array(header, np.uint32).tobytes())
        for layer in quantized:
            f.write(np.ascontiguousarray(layer.weights, np.int8).tobytes())
            f.write(np.ascontiguousarray(layer.biases, np.int32).tobytes())
            f.write(np.ascontiguousarray(layer.multiplier, np.int32).tobytes())
            f.write(np.ascontiguousarray(layer.shift, np.int8).tobytes())
            if layer.table is not None:
                f.write(np.ascontiguousarray(layer.table, np.int8).tobytes())


def fit(network, forward, epochs, lr, data, batch, loss, label):
    """Trains for `epochs` with Adam and a cosine schedule, keeping the weights of the best validation epoch."""
    trainInputs, trainTargets, trainWeights, validationInputs, validationTargets, validationWeights = data
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    numBatches = len(trainInputs) // batch
    best = float('inf')
    bestState = None
    started = time.time()

    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(trainInputs), device=trainInputs.device)
        total = 0.0
        for i in range(numBatches):
            indices = order[i * batch:(i + 1) * batch]
            optimizer.zero_grad(set_to_none=True)
            error = loss(forward(trainInputs[indices]), trainTargets[indices], trainWeights[indices])
            error.backward()
            optimizer.step()
            total += error.item()
        schedule.step()

        network.eval()
        with torch.no_grad():
            validation = loss(forward(validationInputs), validationTargets, validationWeights).item()
        if validation < best:
            best = validation
            bestState = {k: v.detach().clone() for k, v in network.state_dict().items()}

        elapsed = time.time() - started
        print(f'  {label} epoch {epoch + 1:4d}/{epochs}  train {total / numBatches:.6f}  validation {validation:.6f}'
              f'  [{elapsed:.0f} s]', flush=True)

    if bestState is not None:
        network.load_state_dict(bestState)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', type=int, default=20_000_000)
    parser.add_argument('--validation', type=int, default=500_000)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--quantized-epochs', type=int, default=30,
                        help='epochs of fine-tuning with the quantization in the loop, after the float training; 0 '
                             'quantizes the float network as it is')
    parser.add_argument('--quantized-lr', type=float,
                        help='learning rate of that fine-tuning, a tenth of --lr by default')
    parser.add_argument('--cache',
                        help='a file to keep the sampled pixels in: sampling the scenes is most of a run, so a second '
                             'run with the same sampling arguments can read them back instead (it shuffles them '
                             'differently, so it does not reproduce the first run bit for bit)')
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--shallow-fraction', type=float, default=0.90)
    parser.add_argument('--shallow-max', type=int, default=16)
    parser.add_argument('--tile', type=int, default=8)
    parser.add_argument('--loss-power', type=float, default=0.5,
                        help='the exponent of the attenuation the error is weighted by. 2 makes the loss exactly the '
                             'mean squared error of the image, and also makes a pixel behind dense surfaces nearly '
                             'invisible to the optimizer; 0 treats every pixel alike and abandons the opacities that '
                             'carry most of the error. 0.5 measured best on Bistro on both counts')
    parser.add_argument('--hidden1', type=int, default=NUM_HIDDEN1,
                        help='width of the first hidden layer; it sets the widest intermediate tensor and dominates inference cost')
    parser.add_argument('--hidden2', type=int, default=NUM_HIDDEN2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--content', default=CONTENT)
    args = parser.parse_args()

    if torch is None:
        print('ERROR: PyTorch is required: pip install torch')
        return 1

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    wanted = args.samples + args.validation
    if args.cache and os.path.exists(args.cache):
        cached = np.load(args.cache)
        inputs, targets, weights = cached['inputs'], cached['targets'], cached['weights']
        if len(inputs) != wanted or inputs.shape[1] != NUM_INPUTS:
            print(f'ERROR: {args.cache} holds {len(inputs)} pixels of {inputs.shape[1]} features, this run wants {wanted} of {NUM_INPUTS}')
            return 1
        print(f'Read {wanted / 1e6:.1f}M pixels from {args.cache}')
    else:
        print(f'Generating {wanted / 1e6:.1f}M pixels from synthetic scenes')
        inputs, targets, weights = generatePixels(wanted, rng, args.shallow_fraction, args.shallow_max, args.tile)
        if args.cache:
            np.savez(args.cache, inputs=inputs, targets=targets, weights=weights)
            print(f'Wrote {args.cache}')

    permutation = rng.permutation(len(inputs))
    inputs = torch.from_numpy(inputs[permutation])
    targets = torch.from_numpy(targets[permutation])
    attenuation = torch.from_numpy(weights[permutation])[:, None]

    weights = attenuation.clamp(min=1e-4) ** (args.loss_power / 2.0)

    trainInputs, validationInputs = inputs[args.validation:], inputs[:args.validation]
    trainTargets, validationTargets = targets[args.validation:], targets[:args.validation]
    trainWeights = weights[args.validation:]
    validationWeights = attenuation[:args.validation]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    network = torch.nn.Sequential(
        torch.nn.Linear(NUM_INPUTS, args.hidden1),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden1, args.hidden2),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden2, NUM_OUTPUTS),
        torch.nn.Sigmoid(),
    ).to(device)

    trainInputs, trainTargets = trainInputs.to(device), trainTargets.to(device)
    validationInputs, validationTargets = validationInputs.to(device), validationTargets.to(device)
    trainWeights, validationWeights = trainWeights.to(device), validationWeights.to(device)
    data = (trainInputs, trainTargets, trainWeights, validationInputs, validationTargets, validationWeights)

    def loss(prediction, target, weight):
        """The tail is attenuated by the opacity of the exactly kept fragments before it reaches the pixel, so weighting
        the error by that attenuation makes this the mean squared error of the final colour rather than of the
        normalized tail. A weight of zero also excludes a pixel outright, which is how the tile borders and the pixels
        the sample never asks about are kept out of the average."""
        counted = (weight > 0.0).sum().clamp(min=1)
        return torch.square((prediction - target) * weight).sum() / (counted * NUM_OUTPUTS)

    best = fit(network, network, args.epochs, args.lr, data, args.batch, loss, 'float')
    print(f'Validation MSE of the float network: {best:.6f}')

    def integerValidation(quantized):
        prediction = torch.from_numpy(runInteger(quantized, validationInputs.cpu().numpy())).to(device)
        return loss(prediction, validationTargets, validationWeights).item()

    limits = calibrate(network, trainInputs[:1_000_000])
    print('Quantization ranges: ' + ', '.join(f'{v:.3f}' for v in limits))
    print(f'Validation MSE of the float network quantized as it is: {integerValidation(quantizeNetwork(network, limits)):.6f}')

    if args.quantized_epochs:
        fit(network, lambda x: forwardQuantized(network, x, limits), args.quantized_epochs,
            args.quantized_lr if args.quantized_lr else args.lr * 0.1, data, args.batch, loss, 'int8')

    quantized = quantizeNetwork(network, limits)
    integer = runInteger(quantized, validationInputs.cpu().numpy())
    with torch.no_grad():
        simulated = forwardQuantized(network, validationInputs, limits).cpu().numpy()
    mismatch = np.abs(integer - simulated) * 255.0
    print(f'Integer graph against the training-time simulation: {np.mean(mismatch > 0.5):.2%} of the outputs differ, '
          f'at most by {mismatch.max():.0f}/255')
    print(f'Validation MSE of the quantized network: {integerValidation(quantized):.6f}')

    os.makedirs(args.content, exist_ok=True)
    stem = f'dfaoit-{args.hidden1}x{args.hidden2}'
    weightsPath = os.path.join(args.content, f'{stem}.bin')
    writeWeights(quantized, weightsPath)
    print(f'Wrote {weightsPath}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
