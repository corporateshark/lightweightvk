#!/usr/bin/env python3
# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains the Neural Light Grid generator of `ARM_003_NeuralGI` and writes `nlg.bin` (point pattern, PCA, float layers,
output scale, int8 TOSA graph) to `third-party/content/src/nlg/`.

    python tools/train_nlg.py generate --bakes DIR [--out DIR]
    python tools/train_nlg.py train --train DIR --val SET.npz... [--epochs 40] [--output GEN.npz]
    python tools/train_nlg.py quantize GEN.npz --calibrate SET.npz... --val SET.npz... [--output INT8.npz]
    python tools/train_nlg.py export GEN.npz [--integer INT8.npz]

`generate`: GDS support bakes (`--gi-nlg-bake`) -> weighting functions per scene.
`train`: PCA-512 -> 6x512 LeakyReLU MLP with skips -> 113 coefficients, loss through the 3-8-8-1 network.
`quantize`: per-channel int8 graph, evaluated bit for bit; `--qat-epochs` fine-tunes, `--save-generator` keeps the weights.
`export`: refuses a graph not quantized from the given float weights. The sample builds the data graph out of the integer
layers itself (`writeIntegerGraph()` has the layout). Needs PyTorch.
"""
import argparse
import glob
import hashlib
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from tosa_int8 import quantizeMultiplier, rescale, roundSte

BAKE_MAGIC = 0x31474C4E
GENERATOR_MAGIC = 0x47474C4E
HIDDEN = 8
COEFFICIENTS = 3 * HIDDEN + HIDDEN + HIDDEN * HIDDEN + HIDDEN + HIDDEN + 1
SLOPE = 0.01
INPUT_ZERO_POINT = -128
STREAM_FRACTION = 256
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, 'third-party', 'content', 'src', 'nlg')


def readBake(path):
    with open(path, 'rb') as f:
        h = np.frombuffer(f.read(64), np.uint32)
        if h[0] != BAKE_MAGIC or h[1] != 1:
            raise ValueError(f'{path}: not an NLG support bake')
        n, s, k, b = int(h[2]), int(h[3]), int(h[4]), int(h[5])
        shHalves = int(h[10]) if h[10] else 28
        info = np.frombuffer(f.read(32), np.float32)
        points = np.frombuffer(f.read(16 * s), np.float32).reshape(s, 4)[:, :3].copy()
        neighbours = np.frombuffer(f.read(4 * s * k), np.uint32).reshape(s, k).astype(np.int64)
        bakePoints = np.frombuffer(f.read(4 * b), np.uint32).astype(np.int64)
        cells = np.frombuffer(f.read(4 * n), np.uint32).copy()
        offset = f.tell()
    recordBytes = 8 * s + 2 * shHalves * b
    records = np.memmap(path, np.uint8, 'r', offset=offset, shape=(n, recordBytes))
    return {
        'name': os.path.splitext(os.path.basename(path))[0], 'count': n, 's': s, 'k': k, 'b': b,
        'shHalves': shHalves, 'points': points, 'neighbours': neighbours, 'bakePoints': bakePoints,
        'cells': cells, 'lattice': h[6:9].copy(), 'origin': info[0:3].copy(), 'spacing': float(info[3]),
        'records': records,
    }


def readProbes(bake, first, count):
    raw = np.asarray(bake['records'][first:first + count])
    s, b, hs = bake['s'], bake['b'], bake['shHalves']
    words = raw[:, :8 * s].reshape(count, s, 8).view(np.uint32).reshape(count, s, 2)
    sh = raw[:, 8 * s:].reshape(count, b * hs * 2).view(np.float16).reshape(count, b, hs).astype(np.float32)
    return words[:, :, 0].astype(np.int64), words[:, :, 1].astype(np.int64), sh


def trilinear(points):
    return torch.clamp(1.0 - points.abs(), min=0.0).prod(-1)


class FunctionGenerator:
    """The weighting functions of a bake, batch by batch, as the sample computes them on the GPU."""

    def __init__(self, bake, device, sigma, passes, clusters):
        self.device = device
        self.points = torch.from_numpy(bake['points']).to(device)
        self.neighbours = torch.from_numpy(bake['neighbours']).to(device)
        self.bakePoints = torch.from_numpy(bake['bakePoints']).to(device)
        self.k = bake['k']
        self.passes = passes
        self.clusters = clusters
        d = self.points[self.neighbours] - self.points[:, None]
        self.distance = (d * d).sum(-1).sqrt()
        self.gaussian = torch.exp(-(self.distance ** 2) / (2.0 * sigma * sigma))
        self.tri = trilinear(self.points)
        self.shifts = torch.arange(self.k, device=device)

    def diffuse(self, w, received, valid, vis, passes):
        for _ in range(passes):
            wj = w[:, self.neighbours]
            m = vis & received[:, self.neighbours]
            g = self.gaussian[None] * m
            den = g.sum(-1)
            num = (g * wj).sum(-1)
            fresh = ~received & valid & (den > 0.0)
            w = torch.where(fresh, num / den.clamp(min=1e-12), w)
            received = received | fresh
        return w, received

    def propagateColour(self, colour, known, vis, passes=48):
        for _ in range(passes):
            m = (vis & known[:, self.neighbours]).float()
            den = m.sum(-1)
            fresh = ~known & (den > 0.0)
            if not fresh.any():
                break
            mean = (m[..., None] * colour[:, self.neighbours]).sum(-2) / den.clamp(min=1e-12)[..., None]
            colour = torch.where(fresh[..., None], mean, colour)
            known = known | fresh
        return colour, known

    def components(self, member, vis, passes=96):
        n, s = member.shape
        label = torch.where(member, torch.arange(s, device=self.device)[None].expand(n, s), s)
        for _ in range(passes):
            lj = torch.where(vis & member[:, self.neighbours], label[:, self.neighbours], s)
            new = torch.minimum(label, lj.min(-1).values)
            new = torch.where(member, new, s)
            if torch.equal(new, label):
                break
            label = new
        return label

    def insideStart(self, valid, vis, sh):
        n, s = valid.shape
        colour = torch.zeros(n, s, 3, device=self.device)
        known = torch.zeros(n, s, dtype=torch.bool, device=self.device)
        bakeValid = valid[:, self.bakePoints] & (sh[:, :, 27] < 0.25)
        colour[:, self.bakePoints] = torch.where(bakeValid[..., None], sh[:, :, 0:3], 0.0)
        known[:, self.bakePoints] = bakeValid
        colour, known = self.propagateColour(colour, known, vis)
        lit = known & valid
        feature = torch.log2(colour.clamp(min=0.0) + 1e-3)
        start = torch.zeros(n, s, device=self.device)
        for p in range(n):
            pts = lit[p].nonzero().squeeze(-1)
            if len(pts) < self.clusters:
                continue
            f = feature[p, pts]
            centres = f[torch.randperm(len(pts), device=self.device)[: self.clusters]]
            for _ in range(10):
                owner = torch.cdist(f, centres).argmin(-1)
                for c in range(self.clusters):
                    sel = owner == c
                    if sel.any():
                        centres[c] = f[sel].mean(0)
            owner = torch.cdist(f, centres).argmin(-1)
            largest = torch.bincount(owner, minlength=self.clusters).argmax()
            member = torch.zeros(s, dtype=torch.bool, device=self.device)
            member[pts[owner == largest]] = True
            label = self.components(member[None], vis[p:p + 1])[0]
            best = torch.bincount(label[member], minlength=s + 1)
            best[s] = 0
            component = member & (label == best.argmax())
            start[p] = torch.where(component, self.tri, 0.0)
        return start

    def generate(self, bits, flags, sh):
        bits = torch.from_numpy(bits).to(self.device)
        flags = torch.from_numpy(flags).to(self.device)
        sh = torch.from_numpy(sh).to(self.device)
        vis = ((bits[:, :, None] >> self.shifts) & 1).bool()
        valid = (flags & 2) == 0
        seen = ((flags & 1) != 0) & valid
        inside = ((flags & 4) != 0).float().mean(1) > 0.5
        w = torch.where(seen, self.tri[None], 0.0)
        received = seen.clone()
        if inside.any():
            idx = inside.nonzero().squeeze(-1)
            start = self.insideStart(valid[idx], vis[idx], sh[idx])
            w[idx] = start
            received[idx] = start > 0.0
        w, received = self.diffuse(w, received, valid, vis, self.passes)
        return w, valid, seen, inside


def generate(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = sorted(glob.glob(os.path.join(args.bakes, '*.nlg'))) if os.path.isdir(args.bakes) else [args.bakes]
    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        bake = readBake(path)
        target = os.path.join(args.out, bake['name'] + '.npz')
        if args.skip_existing and os.path.isfile(target):
            continue
        n = min(bake['count'], args.limit) if args.limit else bake['count']
        gen = FunctionGenerator(bake, device, args.sigma, args.passes, args.clusters)
        phi = np.zeros((n, bake['s']), np.float16)
        validBits = np.zeros((n, bake['s'] // 8), np.uint8)
        inside = np.zeros(n, bool)
        t0 = time.time()
        for first in range(0, n, args.batch):
            count = min(args.batch, n - first)
            bits, flags, sh = readProbes(bake, first, count)
            w, valid, _, ins = gen.generate(bits, flags, sh)
            phi[first:first + count] = w.cpu().numpy().astype(np.float16)
            validBits[first:first + count] = np.packbits(valid.cpu().numpy(), axis=1)
            inside[first:first + count] = ins.cpu().numpy()
        np.savez(target, cells=bake['cells'][:n], phi=phi, valid=validBits, inside=inside, lattice=bake['lattice'],
                 origin=bake['origin'], spacing=bake['spacing'], points=bake['points'],
                 neighbours=bake['neighbours'].astype(np.uint16), bakePoints=bake['bakePoints'].astype(np.uint16))
        print(f'{bake["name"]}: {n} functions, {inside.mean() * 100:.1f} % inside geometry, {time.time() - t0:.0f} s', flush=True)


def unpack(theta):
    n = theta.shape[0]
    at = 0

    def take(count, shape):
        nonlocal at
        out = theta[:, at: at + count].reshape(n, *shape)
        at += count
        return out

    w1 = take(3 * HIDDEN, (HIDDEN, 3))
    b1 = take(HIDDEN, (HIDDEN,))
    w2 = take(HIDDEN * HIDDEN, (HIDDEN, HIDDEN))
    b2 = take(HIDDEN, (HIDDEN,))
    w3 = take(HIDDEN, (1, HIDDEN))
    b3 = take(1, (1,))
    return w1, b1, w2, b2, w3, b3


def evaluate(theta, x):
    """The weight every network of theta (N, 113) gives the points x (P, 3): (N, P)."""
    w1, b1, w2, b2, w3, b3 = unpack(theta)
    h = torch.nn.functional.leaky_relu(torch.einsum('pi,nhi->nph', x, w1) + b1[:, None], SLOPE)
    h = torch.nn.functional.leaky_relu(torch.einsum('nph,ngh->npg', h, w2) + b2[:, None], SLOPE)
    return (torch.einsum('nph,noh->npo', h, w3) + b3[:, None])[..., 0]


class Generator(nn.Module):
    def __init__(self, inputs, width, layers):
        super().__init__()
        self.first = nn.Linear(inputs, width)
        self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(layers - 1)])
        self.last = nn.Linear(width, COEFFICIENTS)

    def forward(self, z):
        h = nn.functional.leaky_relu(self.first(z), SLOPE)
        for block in self.blocks:
            h = h + nn.functional.leaky_relu(block(h), SLOPE)
        return self.last(h)

    def linears(self):
        return [self.first] + list(self.blocks) + [self.last]


class Functions:
    """All functions of several sets as one f16 array, with their validity; at most `limit` random functions per set."""

    def __init__(self, paths, limit=0, seed=1):
        rng = np.random.default_rng(seed)
        self.names = []
        phi = []
        valid = []
        self.points = None
        for p in paths:
            data = np.load(p)
            keep = np.arange(len(data['phi']))
            if limit and len(keep) > limit:
                keep = np.sort(rng.choice(len(keep), limit, replace=False))
            phi.append(data['phi'][keep])
            valid.append(np.unpackbits(data['valid'][keep], axis=1).astype(bool))
            self.names.append(os.path.splitext(os.path.basename(p))[0])
            if self.points is None:
                self.points = data['points'].astype(np.float32)
        self.phi = np.concatenate(phi)
        self.valid = np.concatenate(valid)

    def __len__(self):
        return len(self.phi)

    def batch(self, idx, device):
        phi = torch.from_numpy(self.phi[idx].astype(np.float32)).to(device)
        valid = torch.from_numpy(self.valid[idx]).to(device).float()
        return phi, valid


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, '*.npz')))
        else:
            out += sorted(glob.glob(p)) or [p]
    return out


def pca(functions, samples, components, device, seed):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(functions), min(samples, len(functions)), replace=False))
    s = functions.points.shape[0]
    mean = torch.zeros(s, device=device, dtype=torch.float64)
    for first in range(0, len(idx), 4096):
        phi, _ = functions.batch(idx[first:first + 4096], device)
        mean += phi.double().sum(0)
    mean /= len(idx)
    cov = torch.zeros(s, s, device=device, dtype=torch.float32)
    for first in range(0, len(idx), 4096):
        phi, _ = functions.batch(idx[first:first + 4096], device)
        x = phi - mean.float()[None]
        cov += x.T @ x
    cov /= len(idx)
    values, vectors = torch.linalg.eigh(cov)
    order = torch.argsort(values, descending=True)[:components]
    explained = values[order].sum() / values.clamp(min=0).sum()
    print(f'PCA of {len(idx)} functions: {components} components explain {explained.item() * 100:.2f} %', flush=True)
    return mean.float(), vectors[:, order].contiguous()


def project(phi, mean, basis):
    return (phi - mean[None]) @ basis


def networkError(theta, functions, idx, device):
    """RMSE over the valid points of the networks `theta` of these functions."""
    x = torch.from_numpy(functions.points).to(device)
    phi, valid = functions.batch(idx, device)
    out = evaluate(theta, x)
    err = (((out - phi) ** 2) * valid).sum(1) / valid.sum(1).clamp(min=1.0)
    return err.sqrt().cpu().numpy()


def generatedError(forward, functions, device, batch=512):
    rmse = np.zeros(len(functions), np.float32)
    with torch.no_grad():
        for first in range(0, len(functions), batch):
            idx = np.arange(first, min(first + batch, len(functions)))
            phi, _ = functions.batch(idx, device)
            rmse[idx] = networkError(forward(phi), functions, idx, device)
    return rmse


def loadGenerator(path, device):
    g = np.load(path)
    pcaComponents, width, layers = (int(v) for v in g['config'])
    generator = Generator(pcaComponents, width, layers).to(device)
    generator.load_state_dict({k[len('generator.'):]: torch.from_numpy(g[k]) for k in g.files if k.startswith('generator.')})
    generator.eval()
    mean = torch.from_numpy(g['mean']).to(device)
    basis = torch.from_numpy(g['basis']).to(device)
    points = g['points'] if 'points' in g.files else None
    return generator, mean, basis, points, (pcaComponents, width, layers)


def saveGenerator(path, generator, mean, basis, points, config):
    state = {'generator.' + k: v.detach().cpu().numpy() for k, v in generator.state_dict().items()}
    if points is not None:
        state['points'] = points
    np.savez(path, mean=mean.cpu().numpy(), basis=basis.cpu().numpy(), config=np.array(config, np.int32), **state)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainSet = Functions(expand(args.train), args.train_limit, args.seed)
    valSet = Functions(expand(args.val), args.val_limit, args.seed) if args.val else None
    print(f'{len(trainSet)} training functions from {len(trainSet.names)} sets; '
          f'{len(valSet) if valSet is not None else 0} validation functions', flush=True)
    generator = Generator(args.pca, args.width, args.layers).to(device)
    mean, basis = pca(trainSet, args.pca_samples, args.pca, device, args.seed)
    x = torch.from_numpy(trainSet.points).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    stepsPerEpoch = (len(trainSet) + args.batch - 1) // args.batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * stepsPerEpoch)
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        t0 = time.time()
        order = rng.permutation(len(trainSet))
        total = 0.0
        for step in range(stepsPerEpoch):
            idx = np.sort(order[step * args.batch:(step + 1) * args.batch])
            phi, valid = trainSet.batch(idx, device)
            subset = torch.from_numpy(rng.choice(len(x), args.points, replace=False)).to(device)
            out = evaluate(generator(project(phi, mean, basis)), x[subset])
            v = valid[:, subset]
            loss = ((((out - phi[:, subset]) ** 2) * v).sum(1) / v.sum(1).clamp(min=1.0)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        line = f'epoch {epoch + 1}: train mse {total / stepsPerEpoch:.3e}'
        if valSet is not None:
            generator.eval()
            rmse = generatedError(lambda phi: generator(project(phi, mean, basis)), valSet, device)
            generator.train()
            line += f', val rmse mean {rmse.mean():.4f} median {np.median(rmse):.4f} 90% {np.quantile(rmse, 0.9):.4f}'
        print(line + f', {time.time() - t0:.0f} s', flush=True)
    saveGenerator(args.output, generator, mean, basis, trainSet.points, (args.pca, args.width, args.layers))
    print('saved', args.output)


def leakyTable():
    q = np.arange(-128, 128, dtype=np.float64)
    return np.clip(np.round(np.where(q > 0.0, q, SLOPE * q))).astype(np.int8)


def tableLeaky(a, step):
    """LeakyReLU as the int8 TABLE applies it: on the grid of the activation, so the leak below 50 steps rounds away."""
    q = a / step[None]
    return roundSte(torch.where(q > 0.0, q, SLOPE * q)) * step[None]


class QuantizedGenerator:
    """The generator rounded like the integer graph, differentiable through straight-through rounding."""

    def __init__(self, generator, mean, basis, limits=None):
        self.generator = generator
        self.mean = mean
        self.basis = basis
        self.limits = limits

    def weights(self):
        """Float weights in graph order: the PCA projection (with the mean in its bias), then the linear layers."""
        pcaWeight = self.basis.T
        pcaBias = -(self.mean @ self.basis)
        return [(pcaWeight, pcaBias)] + [(layer.weight, layer.bias) for layer in self.generator.linears()]

    def forward(self, phi, collect=None):
        limits = self.limits
        device = phi.device
        q = roundSte(phi * 255.0).clamp(0.0, 255.0) / 255.0
        layers = self.weights()
        n = len(layers)

        def quantizedLinear(x, xScale, index, limit):
            weight, bias = layers[index]
            folded = weight * xScale[None]
            wScale = folded.detach().abs().amax(dim=1).clamp(min=1e-8) / 127.0
            w = roundSte(folded / wScale[:, None]).clamp(-127.0, 127.0) * wScale[:, None]
            b = roundSte(bias / wScale) * wScale
            y = (x / xScale[None]) @ w.T + b
            if collect is not None:
                collect.append(y.detach().abs().amax(dim=0))
            if limit is None:
                return y
            step = limit / 127.0
            return roundSte(y / step[None]).clamp(-127.0, 127.0) * step[None], step

        inputScale = torch.full((phi.shape[1],), 1.0 / 255.0, device=device)
        if limits is None:
            z = quantizedLinear(q, inputScale, 0, None)
            h = nn.functional.leaky_relu(quantizedLinear(z, torch.ones_like(z[0]), 1, None), SLOPE)
            if collect is not None:
                collect.append(h.detach().abs().amax(dim=0))
            for k in range(2, n - 1):
                h = h + nn.functional.leaky_relu(quantizedLinear(h, torch.ones_like(h[0]), k, None), SLOPE)
                if collect is not None:
                    collect.append(h.detach().abs().amax(dim=0))
            return quantizedLinear(h, torch.ones_like(h[0]), n - 1, None)
        z, zStep = quantizedLinear(q, inputScale, 0, limits['z'])
        a, aStep = quantizedLinear(z, zStep, 1, limits['a'][0])
        h = tableLeaky(a, aStep)
        streamStep = aStep / STREAM_FRACTION
        for k in range(2, n - 1):
            hStep = limits['h'][k - 2] / 127.0
            x = roundSte(h / hStep[None]).clamp(-127.0, 127.0) * hStep[None]
            a, aStep = quantizedLinear(x, hStep, k, limits['a'][k - 1])
            y = tableLeaky(a, aStep)
            h = h + roundSte(y / streamStep[None]) * streamStep[None]
        hStep = limits['h'][n - 3] / 127.0
        x = roundSte(h / hStep[None]).clamp(-127.0, 127.0) * hStep[None]
        return quantizedLinear(x, hStep, n - 1, None)


def calibrate(quantized, functions, device, samples):
    """Per-channel limits of every activation the integer graph keeps in 8 bits, from `samples` random functions."""
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(functions), min(samples, len(functions)), replace=False))
    collected = None
    limits = quantized.limits
    quantized.limits = None
    with torch.no_grad():
        for first in range(0, len(idx), 512):
            phi, _ = functions.batch(idx[first:first + 512], device)
            batch = []
            quantized.forward(phi, collect=batch)
            collected = batch if collected is None else [torch.maximum(a, b) for a, b in zip(collected, batch)]
    quantized.limits = limits
    layers = len(quantized.generator.linears())
    z = collected[0]
    a = [collected[1]] + [collected[3 + 2 * k] for k in range(layers - 2)]
    h = [collected[2]] + [collected[4 + 2 * k] for k in range(layers - 2)]
    return {'z': z.clamp(min=1e-6), 'a': [v.clamp(min=1e-6) for v in a], 'h': [v.clamp(min=1e-6) for v in h]}


class IntegerGraph:
    """Int8 layers with RESCALE multipliers, the LeakyReLU table, the output scale, the source fingerprint."""

    def __init__(self, layers, table, outputScale, source=''):
        self.layers = layers
        self.table = table
        self.outputScale = outputScale
        self.source = source

    def save(self, path):
        arrays = {'table': self.table, 'outputScale': self.outputScale, 'count': np.array([len(self.layers)]),
                  'source': np.array(self.source)}
        for i, layer in enumerate(self.layers):
            for key, value in layer.items():
                arrays[f'{key}{i}'] = value
        np.savez(path, **arrays)

    @staticmethod
    def load(path):
        data = np.load(path)
        layers = []
        for i in range(int(data['count'][0])):
            layer = {}
            for key in data.files:
                if key.endswith(str(i)) and key[:-len(str(i))] in ('weights', 'biases', 'toActivation', 'toActivationShift',
                                                                   'fromStream', 'fromStreamShift', 'toStream', 'toStreamShift'):
                    layer[key[:-len(str(i))]] = data[key]
            layers.append(layer)
        return IntegerGraph(layers, data['table'], data['outputScale'], str(data['source']) if 'source' in data.files else '')


def fingerprint(arrays, mean, basis, layers):
    """The SHA-256 of a float generator: its linear layers in order, the PCA mean and basis, all as float32."""
    h = hashlib.sha256()
    for name in ['first'] + [f'blocks.{i}' for i in range(layers - 1)] + ['last']:
        for part in ('weight', 'bias'):
            h.update(np.ascontiguousarray(arrays[f'generator.{name}.{part}'], np.float32).tobytes())
    h.update(np.ascontiguousarray(mean, np.float32).tobytes())
    h.update(np.ascontiguousarray(basis, np.float32).tobytes())
    return h.hexdigest()


def buildIntegerGraph(quantized):
    """The integer graph of a calibrated `QuantizedGenerator`."""
    limits = quantized.limits
    layers = quantized.weights()
    n = len(layers)
    out = []

    def fold(index, xScale):
        weight, bias = layers[index]
        folded = (weight.detach().double() * xScale.double()[None]).cpu()
        wScale = folded.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weights = torch.round(folded / wScale[:, None]).clamp(-127, 127).to(torch.int8).numpy()
        biases = np.round(bias.detach().double().cpu().numpy() / wScale.numpy())
        assert np.abs(biases).max() < 2 ** 31, 'a bias does not fit int32'
        return weights, biases.astype(np.int32), wScale.numpy()

    inputScale = torch.full((layers[0][0].shape[1],), 1.0 / 255.0, device=quantized.mean.device)
    weights, biases, wScale = fold(0, inputScale)
    zStep = (limits['z'] / 127.0).double().cpu().numpy()
    toActivation, toActivationShift = quantizeMultiplier(wScale / zStep)
    out.append({'weights': weights, 'biases': biases, 'toActivation': toActivation, 'toActivationShift': toActivationShift})
    weights, biases, wScale = fold(1, limits['z'] / 127.0)
    aStep = (limits['a'][0] / 127.0).double().cpu().numpy()
    toActivation, toActivationShift = quantizeMultiplier(wScale / aStep)
    streamStep = aStep / STREAM_FRACTION
    toStream, toStreamShift = quantizeMultiplier(aStep / streamStep)
    out.append({'weights': weights, 'biases': biases, 'toActivation': toActivation, 'toActivationShift': toActivationShift,
                'toStream': toStream, 'toStreamShift': toStreamShift})
    for k in range(2, n - 1):
        hStep = (limits['h'][k - 2] / 127.0).double().cpu().numpy()
        fromStream, fromStreamShift = quantizeMultiplier(streamStep / hStep)
        weights, biases, wScale = fold(k, limits['h'][k - 2] / 127.0)
        aStep = (limits['a'][k - 1] / 127.0).double().cpu().numpy()
        toActivation, toActivationShift = quantizeMultiplier(wScale / aStep)
        toStream, toStreamShift = quantizeMultiplier(aStep / streamStep)
        out.append({'weights': weights, 'biases': biases, 'fromStream': fromStream, 'fromStreamShift': fromStreamShift,
                    'toActivation': toActivation, 'toActivationShift': toActivationShift, 'toStream': toStream,
                    'toStreamShift': toStreamShift})
    hStep = (limits['h'][n - 3] / 127.0).double().cpu().numpy()
    fromStream, fromStreamShift = quantizeMultiplier(streamStep / hStep)
    weights, biases, wScale = fold(n - 1, limits['h'][n - 3] / 127.0)
    out.append({'weights': weights, 'biases': biases, 'fromStream': fromStream, 'fromStreamShift': fromStreamShift})
    return IntegerGraph(out, leakyTable(), wScale.astype(np.float32))


def runInteger(graph, phi):
    """The integer graph evaluated exactly on functions (N, points)."""
    q = np.clip(np.round(phi.astype(np.float64) * 255.0) + INPUT_ZERO_POINT).astype(np.int64)
    layers = graph.layers
    int32 = (-(2 ** 31), 2 ** 31 - 1)

    def conv(x, layer):
        acc = (x - 0).astype(np.float64) @ layer['weights'].astype(np.float64).T + layer['biases'].astype(np.float64)
        assert np.abs(acc).max() < 2 ** 31, 'an accumulator does not fit int32'
        return acc.astype(np.int64)

    acc = (q - INPUT_ZERO_POINT).astype(np.float64) @ layers[0]['weights'].astype(np.float64).T + layers[0]['biases'].astype(np.float64)
    assert np.abs(acc).max() < 2 ** 31, 'the PCA accumulator does not fit int32'
    z = rescale(acc.astype(np.int64), layers[0]['toActivation'], layers[0]['toActivationShift'])
    a = rescale(conv(z, layers[1]), layers[1]['toActivation'], layers[1]['toActivationShift'])
    y = graph.table[(a + 128).astype(np.int32)].astype(np.int64)
    stream = rescale(y, layers[1]['toStream'], layers[1]['toStreamShift'], 0, *int32)
    for layer in layers[2:-1]:
        x = rescale(stream, layer['fromStream'], layer['fromStreamShift'])
        a = rescale(conv(x, layer), layer['toActivation'], layer['toActivationShift'])
        y = graph.table[(a + 128).astype(np.int32)].astype(np.int64)
        stream = np.clip(stream + rescale(y, layer['toStream'], layer['toStreamShift'], 0, *int32), *int32)
    last = layers[-1]
    x = rescale(stream, last['fromStream'], last['fromStreamShift'])
    return (conv(x, last).astype(np.float64) * graph.outputScale.astype(np.float64)).astype(np.float32)


def integerError(graph, functions, device, batch=1024):
    rmse = np.zeros(len(functions), np.float32)
    for first in range(0, len(functions), batch):
        idx = np.arange(first, min(first + batch, len(functions)))
        theta = torch.from_numpy(runInteger(graph, functions.phi[idx].astype(np.float32))).to(device)
        with torch.no_grad():
            rmse[idx] = networkError(theta, functions, idx, device)
    return rmse


def report(label, rmse):
    print(f'{label}: rmse mean {rmse.mean():.4f} median {np.median(rmse):.4f} 90% {np.quantile(rmse, 0.9):.4f}', flush=True)


def quantize(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator, mean, basis, points, config = loadGenerator(args.generator, device)
    calibration = Functions(expand(args.calibrate), args.calibrate_limit, args.seed)
    valSet = Functions(expand(args.val), args.val_limit, args.seed) if args.val else None
    print(f'{len(calibration)} calibration functions; {len(valSet) if valSet is not None else 0} validation functions', flush=True)
    quantized = QuantizedGenerator(generator, mean, basis)
    if valSet is not None:
        report('float generator', generatedError(lambda phi: generator(project(phi, mean, basis)), valSet, device))
        report('float generator with int8 weights', generatedError(quantized.forward, valSet, device))
    quantized.limits = calibrate(quantized, calibration, device, args.calibrate_samples)
    if valSet is not None:
        report('quantized generator', generatedError(quantized.forward, valSet, device))
    if args.qat_epochs:
        x = torch.from_numpy(calibration.points).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
        stepsPerEpoch = (len(calibration) + args.batch - 1) // args.batch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.qat_epochs * stepsPerEpoch)
        rng = np.random.default_rng(args.seed)
        generator.train()
        for epoch in range(args.qat_epochs):
            t0 = time.time()
            order = rng.permutation(len(calibration))
            total = 0.0
            for step in range(stepsPerEpoch):
                idx = np.sort(order[step * args.batch:(step + 1) * args.batch])
                phi, valid = calibration.batch(idx, device)
                subset = torch.from_numpy(rng.choice(len(x), args.points, replace=False)).to(device)
                out = evaluate(quantized.forward(phi), x[subset])
                v = valid[:, subset]
                loss = ((((out - phi[:, subset]) ** 2) * v).sum(1) / v.sum(1).clamp(min=1.0)).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total += loss.item()
            generator.eval()
            line = f'quantization-aware epoch {epoch + 1}: train mse {total / stepsPerEpoch:.3e}'
            if valSet is not None:
                rmse = generatedError(quantized.forward, valSet, device)
                line += f', val rmse mean {rmse.mean():.4f} median {np.median(rmse):.4f}'
            generator.train()
            print(line + f', {time.time() - t0:.0f} s', flush=True)
        generator.eval()
        if args.save_generator:
            saveGenerator(args.save_generator, generator, mean, basis, points, config)
            print('saved', args.save_generator)
    graph = buildIntegerGraph(quantized)
    arrays = {'generator.' + k: v.detach().cpu().numpy() for k, v in generator.state_dict().items()}
    graph.source = fingerprint(arrays, mean.cpu().numpy(), basis.cpu().numpy(), config[2])
    if args.qat_epochs and not args.save_generator:
        print('the fine-tuned generator is not saved (--save-generator): export cannot pair this graph with float weights')
    if valSet is not None:
        report('integer graph', integerError(graph, valSet, device))
    graph.save(args.output)
    print('saved', args.output)


def writeIntegerGraph(out, graph):
    """Appends the int8 graph to `nlg.bin`, whose header counts its layers: per layer the int8 weights [outputs, inputs]
    and the int32 biases, then the RESCALEs the layer has, in the order fromStream, toActivation, toStream, each as int32
    multipliers and int8 shifts; then the 256 int8 entries of the LeakyReLU table. The first layer (the PCA) has only
    toActivation, the second toActivation and toStream, the last only fromStream, the others all three."""
    for layer in graph.layers:
        out.write(np.ascontiguousarray(layer['weights'], np.int8).tobytes())
        out.write(np.ascontiguousarray(layer['biases'], np.int32).tobytes())
        for key in ('fromStream', 'toActivation', 'toStream'):
            if key in layer:
                out.write(np.ascontiguousarray(layer[key], np.int32).tobytes())
                out.write(np.ascontiguousarray(layer[key + 'Shift'], np.int8).tobytes())
    out.write(np.ascontiguousarray(graph.table, np.int8).tobytes())


def export(args):
    g = np.load(args.generator)
    points = (g['points'] if 'points' in g.files else np.load(args.points)['points']).astype(np.float32)
    mean = g['mean'].astype(np.float32)
    basis = g['basis'].astype(np.float32)
    pcaComponents, width, layers = (int(v) for v in g['config'])
    names = ['first'] + [f'blocks.{i}' for i in range(layers - 1)] + ['last']
    assert points.shape == (mean.shape[0], 3) and basis.shape == (points.shape[0], pcaComponents)
    graph = IntegerGraph.load(args.integer) if args.integer else None
    if graph and graph.source != fingerprint(g, mean, basis, layers):
        sys.exit(f'{args.integer} was not quantized from the weights of {args.generator}: the float layers and the '
                 f'data graph of nlg.bin would give different GI')
    os.makedirs(args.out, exist_ok=True)
    binPath = os.path.join(args.out, 'nlg.bin')
    with open(binPath, 'wb') as out:
        graphLayers = len(graph.layers) if graph else 0
        out.write(struct.pack('<8I', GENERATOR_MAGIC, 3, points.shape[0], pcaComponents, width, layers, COEFFICIENTS, graphLayers))
        out.write(points.tobytes())
        out.write(mean.tobytes())
        out.write(basis.astype(np.float16).tobytes())
        for name in names:
            w = g[f'generator.{name}.weight'].astype(np.float32)
            b = g[f'generator.{name}.bias'].astype(np.float32)
            out.write(np.ascontiguousarray(w.T).tobytes())
            out.write(b.tobytes())
        scale = graph.outputScale.astype(np.float32) if graph else np.zeros(COEFFICIENTS, np.float32)
        out.write(scale.tobytes())
        if graph:
            writeIntegerGraph(out, graph)
    print(f'wrote {binPath}: {points.shape[0]} points, PCA {pcaComponents}, {layers} layers {width} wide, '
          f'{COEFFICIENTS} coefficients, {graphLayers} integer layers, {os.path.getsize(binPath)} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='weighting functions from support bakes')
    g.add_argument('--bakes', required=True, help='a .nlg file or a directory of them')
    g.add_argument('--out', default='build/nlg_functions')
    g.add_argument('--sigma', type=float, default=0.15, help='Gaussian falloff of the diffusion, in spacings')
    g.add_argument('--passes', type=int, default=2)
    g.add_argument('--clusters', type=int, default=4, help='lighting clusters of a probe inside geometry')
    g.add_argument('--batch', type=int, default=256)
    g.add_argument('--limit', type=int, default=0, help='at most this many probes per bake')
    g.add_argument('--skip-existing', action='store_true')
    g.add_argument('--seed', type=int, default=1)
    t = sub.add_parser('train', help='the generator from function sets')
    t.add_argument('--train', nargs='+', required=True, help='function sets or directories of them')
    t.add_argument('--val', nargs='*', default=[], help='function sets of scenes not in --train')
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--batch', type=int, default=256)
    t.add_argument('--points', type=int, default=2048, help='support points evaluated per step')
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--width', type=int, default=512)
    t.add_argument('--layers', type=int, default=6)
    t.add_argument('--pca', type=int, default=512)
    t.add_argument('--pca-samples', type=int, default=65536)
    t.add_argument('--train-limit', type=int, default=50000, help='at most this many functions per training set')
    t.add_argument('--val-limit', type=int, default=8192)
    t.add_argument('--output', default='build/nlg-generator.npz')
    t.add_argument('--seed', type=int, default=1)
    q = sub.add_parser('quantize', help='the integer graph of a generator')
    q.add_argument('generator')
    q.add_argument('--calibrate', nargs='+', required=True, help='function sets the activation ranges come from')
    q.add_argument('--val', nargs='*', default=[])
    q.add_argument('--calibrate-limit', type=int, default=4000, help='at most this many functions per calibration set')
    q.add_argument('--calibrate-samples', type=int, default=65536)
    q.add_argument('--val-limit', type=int, default=8192)
    q.add_argument('--qat-epochs', type=int, default=0, help='fine-tune with the rounding in the loop')
    q.add_argument('--lr', type=float, default=1e-4)
    q.add_argument('--batch', type=int, default=256)
    q.add_argument('--points', type=int, default=2048)
    q.add_argument('--save-generator', default='', help='the fine-tuned float generator, when --qat-epochs')
    q.add_argument('--output', default='build/nlg-generator-int8.npz')
    q.add_argument('--seed', type=int, default=1)
    e = sub.add_parser('export', help='the generator as the files the sample reads')
    e.add_argument('generator')
    e.add_argument('--integer', default='', help='the integer graph of `quantize`, for the data graph')
    e.add_argument('--points', default='', help='a function set with the point pattern, for a generator saved without one')
    e.add_argument('--out', default=CONTENT)
    args = parser.parse_args()
    {'generate': generate, 'train': train, 'quantize': quantize, 'export': export}[args.command](args)


if __name__ == '__main__':
    main()
