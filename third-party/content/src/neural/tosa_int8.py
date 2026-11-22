# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The integer arithmetic of the int8 TOSA graphs `train_dfaoit.py` writes."""
import numpy as np


def roundSte(x):
    """Rounds in the forward pass and passes the gradient straight through, so the rounding can sit inside training."""
    return x + (x.round() - x).detach()


def quantizeMultiplier(scale):
    """A positive scale as TOSA's `scale32` fixed point: `multiplier * 2^-shift` with the multiplier in [2^30, 2^31)."""
    mantissa, exponent = np.frexp(scale.astype(np.float64))
    multiplier = np.round(mantissa * (1 << 31)).astype(np.int64)
    shift = 31 - exponent.astype(np.int64)
    overflow = multiplier == (1 << 31)
    multiplier[overflow] //= 2
    shift[overflow] -= 1
    assert shift.min() >= 2 and shift.max() <= 62, 'a RESCALE shift is out of TOSA range'
    return multiplier.astype(np.int32), shift.astype(np.int8)


def rescale(value, multiplier, shift, zeroPoint=0, low=-128, high=127):
    """TOSA RESCALE with `scale32` and single rounding, as `applyScale()` of the emulation layer computes it."""
    value = value.astype(np.int64)
    rounding = np.int64(1) << (shift.astype(np.int64) - 1)
    result = (value * multiplier.astype(np.int64) + rounding) >> shift.astype(np.int64)
    return np.clip(result + zeroPoint, low, high)
