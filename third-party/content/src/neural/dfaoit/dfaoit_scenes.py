# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Synthetic scenes of transparent primitives, sampled as tiles of depth-ordered fragment lists for `train_dfaoit.py`.

A scene is a handful of spheres, boxes, quads and tori with random placement, colour and opacity; a tile of
orthographic rays is fired through it and every entry and exit of every primitive becomes a fragment, so silhouettes,
nesting and back-face pairs all come out on their own rather than being modelled by hand.

Rays are orthographic along +z: the network never sees camera parameters, and every intersection stays closed form.
The ray parameter t is the depth, since the direction is a unit vector and the origin sits at z = 0.

    tiles = sampleTiles(numTiles, numPrimitives, tileSize, rng)

`tiles.color` is (numTiles, tileSize * tileSize, maxFragments, 3) and `tiles.alpha` the matching opacity, both sorted
by depth with the misses pushed to the end and zeroed; `tiles.count` is how many of those entries are real.
"""
import collections

import numpy as np

SPHERE, BOX, QUAD, TORUS = 0, 1, 2, 3
NUM_TYPES = 4

TORUS_STEPS = 48
TORUS_REFINE = 6
MAX_TORUS_HITS = 4


Tiles = collections.namedtuple('Tiles', 'color alpha count')


def randomRotations(n, rng):
    """Uniformly distributed rotation matrices, from random quaternions."""
    q = rng.normal(0.0, 1.0, (n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(n, 3, 3)


def intersectSphere(origin, centre, radius):
    """Orthographic rays along +z against a sphere; returns the entry and exit depth and a hit mask."""
    offset = origin[..., :2] - centre[..., None, :2]
    squared = radius[..., None] ** 2 - (offset ** 2).sum(axis=-1)
    hit = squared > 0.0
    half = np.sqrt(np.maximum(squared, 0.0))
    return centre[..., None, 2] - half, centre[..., None, 2] + half, hit


def intersectBox(origin, centre, extent, rotation):
    """The slab method in the box's own frame."""
    local = np.einsum('bji,bpj->bpi', rotation, origin - centre[:, None, :])
    direction = rotation[:, 2, :]

    safe = np.where(np.abs(direction) < 1e-6, 1e-6, direction)[:, None, :]
    lo = (-extent[:, None, :] - local) / safe
    hi = (extent[:, None, :] - local) / safe

    near = np.maximum(np.minimum(lo, hi), -1e9)
    far = np.minimum(np.maximum(lo, hi), 1e9)

    parallel = np.abs(direction)[:, None, :] < 1e-6
    inside = np.abs(local) <= extent[:, None, :]
    near = np.where(parallel, np.where(inside, -1e9, 1e9), near)
    far = np.where(parallel, np.where(inside, 1e9, -1e9), far)

    entry = near.max(axis=-1)
    exit = far.min(axis=-1)
    return entry, exit, entry < exit


def intersectQuad(origin, centre, rotation, size):
    """A one-sided rectangle: a single fragment, the way a pane of glass modelled as one quad behaves."""
    normal = rotation[:, :, 2]
    denominator = np.where(np.abs(normal[:, None, 2]) < 1e-6, 1e-6, normal[:, None, 2])

    t = ((centre[:, None, :] - origin) * normal[:, None, :]).sum(axis=-1) / denominator
    point = origin - centre[:, None, :]
    point = point + t[..., None] * np.array([0.0, 0.0, 1.0], np.float32)

    u = (point * rotation[:, :, 0][:, None, :]).sum(axis=-1)
    v = (point * rotation[:, :, 1][:, None, :]).sum(axis=-1)
    hit = (np.abs(u) < size[:, None, 0]) & (np.abs(v) < size[:, None, 1])
    return t, hit


def torusDistance(point, major, minor):
    """The signed distance of a torus whose axis is the object-space z."""
    radial = np.sqrt((point[..., :2] ** 2).sum(axis=-1)) - major
    return np.sqrt(radial ** 2 + point[..., 2] ** 2) - minor


def intersectTorus(origin, centre, rotation, major, minor):
    """A torus is a quartic, so it is marched instead: up to four depths per ray, which is the nesting no other
    primitive here produces."""
    local = np.einsum('bji,bpj->bpi', rotation, origin - centre[:, None, :])
    direction = rotation[:, 2, :][:, None, :]

    bound = (major + minor)[:, None]
    along = -(local * direction).sum(axis=-1)
    squared = bound ** 2 - ((local ** 2).sum(axis=-1) - along ** 2)
    reaches = squared > 0.0
    half = np.sqrt(np.maximum(squared, 0.0))
    start, stop = along - half, along + half

    steps = np.linspace(0.0, 1.0, TORUS_STEPS + 1)
    t = start[..., None] + (stop - start)[..., None] * steps
    distance = torusDistance(local[..., None, :] + t[..., None] * direction[..., None, :],
                             major[:, None, None], minor[:, None, None])

    crossing = (distance[..., :-1] * distance[..., 1:]) < 0.0
    crossing &= reaches[..., None]

    order = np.argsort(~crossing, axis=-1, kind='stable')[..., :MAX_TORUS_HITS]
    found = np.take_along_axis(crossing, order, axis=-1)

    low = np.take_along_axis(t, order, axis=-1)
    high = np.take_along_axis(t, order + 1, axis=-1)
    lowDistance = np.take_along_axis(distance, order, axis=-1)

    for _ in range(TORUS_REFINE):
        middle = 0.5 * (low + high)
        value = torusDistance(local[..., None, :] + middle[..., None] * direction[..., None, :],
                              major[:, None, None], minor[:, None, None])
        same = (value * lowDistance) > 0.0
        low = np.where(same, middle, low)
        lowDistance = np.where(same, value, lowDistance)
        high = np.where(same, high, middle)

    return 0.5 * (low + high), found


def sampleColors(numScenes, numPrimitives, rng):
    """Real scenes are full of greys, plaster and stone, so a third of the scenes are desaturated."""
    color = rng.random((numScenes, numPrimitives, 3)).astype(np.float32)
    luminance = color.mean(axis=2, keepdims=True)
    desaturated = (rng.random(numScenes) < 0.35)[:, None, None]
    color = np.where(desaturated, luminance + (color - luminance) * 0.25, color)
    return np.clip(color * rng.uniform(0.15, 1.0, size=(numScenes, 1, 1)).astype(np.float32), 0.0, 1.0)


def sampleOpacities(numScenes, numPrimitives, rng):
    """A scene draws a band of opacity and fills it. The band may sit anywhere, so the opacities come out spread over
    the whole range rather than piled up where a fixed band would put them: the network turns out to end up as close to
    the best its features allow as the training density at that opacity, so an even spread matters more than the shape
    of any one scene."""
    first = rng.uniform(0.01, 0.99, size=(numScenes, 1)).astype(np.float32)
    second = rng.uniform(0.01, 0.99, size=(numScenes, 1)).astype(np.float32)
    low, high = np.minimum(first, second), np.maximum(first, second)

    alpha = low + (high - low) * rng.random((numScenes, numPrimitives)).astype(np.float32)

    constant = low + (high - low) * rng.random((numScenes, 1)).astype(np.float32)
    alpha = np.where((rng.random(numScenes) < 0.25)[:, None], constant, alpha)
    return np.clip(alpha, 0.004, 0.995)


def sampleTiles(numTiles, numPrimitives, tileSize, rng):
    """Fires a `tileSize` square of orthographic rays through each of `numTiles` independent scenes."""
    b, p, n = numTiles, numPrimitives, tileSize * tileSize

    kind = rng.integers(0, NUM_TYPES, size=(b, p))
    centre = rng.uniform(-0.8, 0.8, size=(b, p, 3)).astype(np.float32)
    rotation = randomRotations(b * p, rng).reshape(b, p, 3, 3).astype(np.float32)
    scale = np.exp(rng.uniform(np.log(0.25), np.log(1.4), size=(b, p))).astype(np.float32)

    color = sampleColors(b, p, rng)
    alpha = sampleOpacities(b, p, rng)

    span = 2.0 / 128.0 * tileSize
    corner = rng.uniform(-0.9, 0.9 - span, size=(b, 2)).astype(np.float32)
    axis = np.arange(tileSize, dtype=np.float32) * (span / tileSize)
    gridY, gridX = np.meshgrid(axis, axis, indexing='ij')
    origin = np.zeros((b, n, 3), np.float32)
    origin[:, :, 0] = corner[:, None, 0] + gridX.ravel()[None, :]
    origin[:, :, 1] = corner[:, None, 1] + gridY.ravel()[None, :]

    maxFragments = MAX_TORUS_HITS * p
    depth = np.full((b, n, maxFragments), np.inf, np.float32)
    fragColor = np.zeros((b, n, maxFragments, 3), np.float32)
    fragAlpha = np.zeros((b, n, maxFragments), np.float32)
    written = np.zeros((b, n), np.int64)
    columns = np.arange(n)[None, :]

    def emit(scene, value, valid, index):
        """Appends one fragment per ray of the scenes in `scene` wherever `valid` says there is one. A slot that the
        mask rejects is left at infinity and simply reused by the next primitive."""
        rows = scene[:, None]
        slot = written[scene]
        depth[rows, columns, slot] = np.where(valid, value, np.inf)
        fragColor[rows, columns, slot] = np.broadcast_to(color[scene, index][:, None, :], (len(scene), n, 3))
        fragAlpha[rows, columns, slot] = np.broadcast_to(alpha[scene, index][:, None], (len(scene), n))
        written[scene] = slot + valid

    for i in range(p):
        for t in (SPHERE, BOX, QUAD, TORUS):
            which = np.nonzero(kind[:, i] == t)[0]
            if not len(which):
                continue
            o = origin[which]
            c = centre[which, i]
            r = rotation[which, i]
            s = scale[which, i]

            if t == SPHERE:
                entry, exit, hit = intersectSphere(o, c, s)
                emit(which, entry, hit, i)
                emit(which, exit, hit, i)
            elif t == BOX:
                extent = s[:, None] * rng.uniform(0.4, 1.0, size=(len(which), 3)).astype(np.float32)
                entry, exit, hit = intersectBox(o, c, extent, r)
                emit(which, entry, hit, i)
                emit(which, exit, hit, i)
            elif t == QUAD:
                size = s[:, None] * rng.uniform(0.5, 1.6, size=(len(which), 2)).astype(np.float32)
                value, hit = intersectQuad(o, c, r, size)
                emit(which, value, hit, i)
            else:
                minor = s * rng.uniform(0.15, 0.45, size=len(which)).astype(np.float32)
                values, found = intersectTorus(o, c, r, s, minor)
                for k in range(MAX_TORUS_HITS):
                    emit(which, values[..., k], found[..., k], i)

    order = np.argsort(depth, axis=-1)
    depth = np.take_along_axis(depth, order, axis=-1)
    fragAlpha = np.take_along_axis(fragAlpha, order, axis=-1)
    fragColor = np.take_along_axis(fragColor, order[..., None], axis=2)

    count = np.isfinite(depth).sum(axis=-1)

    real = np.arange(maxFragments)[None, None, :] < count[..., None]
    return Tiles(fragColor * real[..., None], fragAlpha * real, count)


def sampleDepthComplexity(rng, shallowFraction, shallowMax, maxPrimitives=160):
    """Primitives per scene, log-uniform and mostly below `shallowMax`, so that the training set has far more pixels of
    low depth complexity than of high, as real scenes do. Roughly two fragments per primitive land on a ray."""
    upper = shallowMax if rng.random() < shallowFraction else maxPrimitives
    count = int(round(np.exp(rng.uniform(np.log(2.0), np.log(max(3.0, upper))))))
    return max(2, min(maxPrimitives, count))
