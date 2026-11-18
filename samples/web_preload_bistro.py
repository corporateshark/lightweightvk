# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# usage: python web_preload_bistro.py <content dir> <output .rsp>
# The web build of Tiny_MeshLarge preloads only what it reads: the mesh cache, the skybox, and the ambient/diffuse/alpha
# textures of the Bistro exterior with their BC7 .ktx caches. Writes them as Emscripten `--preload-file` options.
import os
import struct
import sys

content, out = sys.argv[1], sys.argv[2]
N = 128  # MAX_MATERIAL_NAME in samples/Bistro.h
STRIDE = N + 12 + 12 + 5 * N  # CachedMaterial

with open(os.path.join(content, 'cache.data'), 'rb') as f:
    data = f.read()
textures = set()
for i in range(struct.unpack_from('<I', data, 4)[0]):
    base = 16 + i * STRIDE + N + 24
    for k in range(3):  # ambient, diffuse, alpha
        name = data[base + k * N:base + (k + 1) * N].split(b'\0')[0].decode()
        if name:
            textures.add(name.replace('\\', '/'))

# exterior.mtl creates `Exterior/`: the texture names are relative to it (`../BuildingTextures/x.png`)
files = ['cache.data', 'immenstadter_horn_2k_ReferenceMap.ktx', 'immenstadter_horn_2k_IrradianceMap.ktx', 'src/bistro/Exterior/exterior.mtl']
for name in sorted(textures):
    files.append(os.path.normpath('src/bistro/Exterior/' + name).replace('\\', '/'))
    ktx = 'src/bistro/Exterior/' + name  # convertFileName() in samples/Bistro.h
    for c in ':./\\':
        ktx = ktx.replace(c, '_')
    files.append(ktx + '.ktx')

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    for name in files:
        path = os.path.join(content, name).replace('\\', '/')
        if os.path.exists(path):
            # quoted: some texture names have spaces
            f.write('"--preload-file=%s@/third-party/content/%s"\n' % (path, name))
