# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shape-specializes the Arm NSS network for another render resolution.

Arm publishes the network shape-agnostic (`nss_v1_0_1_high_int8.vgf`): the 35-operator graph and the weights are there,
but no tensor carries a shape, so it cannot be run as it stands. This script works out every shape for the requested
resolution and writes a model the sample can load:

    python tools/make_nss_model.py --render 1920x1080

writes `third-party/content/src/nss/2_nss-1920x1080-v1_0_1.vgf`, which `DEMO_003_NeuralSuperSampling --4k` loads. The
shape-agnostic model is downloaded once into the same folder; `deploy_content.py` only fetches the model pre-shaped for
960x540 -> 1920x1080, which the sample uses by default.

The shapes come from TOSA inference over the graph: CONV2D and RESIZE move H and W, CONCAT moves the channels, RESCALE
and TABLE keep the shape. Every operator result then gets a shaped tensor type, the graph interface and the model
resource table are shaped to match, and nothing else is touched - not the weights, not the operator attributes, not the
2x upscale ratio of NSS itself. A resolution the network cannot divide is refused, naming the operator that fails.

The graph input is the render resolution rounded up to a multiple of 8, the alignment the sample uses: 540 -> 544 for
the published model, 1080 -> 1080 for the 4K one.

Needs the Vulkan SDK (`spirv-dis`, `spirv-as`, `spirv-val`), a C++ compiler and the Arm VGF library from `deploy_deps.py`:
the script builds `vgf_respecialize` itself, in the LightweightVK build directory when there is one and in a small build
directory of its own otherwise.
"""
import os
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, 'third-party', 'content', 'src', 'nss')
MODEL = os.path.join(CONTENT, 'nss_v1_0_1_high_int8.vgf')
MODEL_URL = 'https://huggingface.co/Arm/neural-super-sampling/resolve/main/nss_v1_0_1_high_int8.vgf'
ALIGNMENT = 8
SPIRV_MAGIC = 0x07230203


def option(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def fail(message):
    print(f'ERROR: {message}')
    sys.exit(1)


def findTool(name):
    sdk = os.environ.get('VULKAN_SDK')
    if sdk:
        for folder in ('Bin', 'bin'):
            candidate = os.path.join(sdk, folder, name + ('.exe' if os.name == 'nt' else ''))
            if os.path.exists(candidate):
                return candidate
    found = shutil.which(name)
    if not found:
        fail(f'`{name}` not found: install the Vulkan SDK and set VULKAN_SDK, or put it on PATH')
    return found


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        fail(f'{os.path.basename(str(command[0]))} failed:\n{result.stderr.strip() or result.stdout.strip()}')
    return result.stdout


def buildTool(buildDir):
    """Uses `vgf_respecialize` from a configured LightweightVK build, or configures a build directory of its own."""
    name = 'vgf_respecialize' + ('.exe' if os.name == 'nt' else '')
    standalone = os.path.join(buildDir, 'nss-model-tool')

    def found():
        for folder in (buildDir, os.path.join(buildDir, 'tools'), os.path.join(buildDir, 'tools', 'Release'),
                       standalone, os.path.join(standalone, 'Release')):
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                return candidate
        return None

    tool = found()
    if tool:
        return tool

    target = buildDir if os.path.exists(os.path.join(buildDir, 'CMakeCache.txt')) else standalone
    if target == standalone and not os.path.exists(os.path.join(standalone, 'CMakeCache.txt')):
        print(f'Configuring `vgf_respecialize` in {standalone}...')
        run(['cmake', '-S', os.path.join(ROOT, 'tools'), '-B', standalone, '-DCMAKE_BUILD_TYPE=Release'])
    print(f'Building `vgf_respecialize` in {target}...')
    run(['cmake', '--build', target, '--config', 'Release', '--target', 'vgf_respecialize'])

    tool = found()
    if not tool:
        fail('`vgf_respecialize` was built but cannot be found')
    return tool


def extractModule(path):
    """Copies the SPIR-V graph module out of a VGF: the module is walked instruction by instruction, so its end is exact."""
    data = open(path, 'rb').read()
    words = memoryview(data).cast('I')
    start = next((i for i in range(len(words)) if words[i] == SPIRV_MAGIC), None)
    if start is None:
        fail(f'no SPIR-V module in {path}')
    i = start + 5
    while i < len(words):
        length = words[i] >> 16
        if length == 0 or i + length > len(words):
            break
        i += length
    return bytes(data[start * 4:i * 4])


class Module:
    """The disassembled graph module: its constants, its tensor types and its operators."""

    def __init__(self, lines):
        self.lines = lines
        self.scalars, self.vectors, self.types, self.valueTypes = {}, {}, {}, {}
        for line in lines:
            m = re.match(r'\s*(%\S+) = OpConstant \S+ (-?\d+)\s*$', line)
            if m:
                self.scalars[m.group(1)] = int(m.group(2))
            m = re.match(r'\s*(%\S+) = OpConstantComposite \S+ (.*)', line)
            if m:
                self.vectors[m.group(1)] = [self.scalars.get(c) for c in m.group(2).split()]
            m = re.match(r'\s*(%\S+) = OpConstantCompositeReplicateEXT \S+ (%\S+)', line)
            if m:
                self.vectors[m.group(1)] = [self.scalars.get(m.group(2))] * 8
            m = re.match(r'\s*(%\S+) = OpConstantNull', line)
            if m:
                self.vectors[m.group(1)] = [0] * 8
            m = re.match(r'\s*(%\S+) = OpTypeTensorARM (%\S+) (%\S+)(?: (%\S+))?\s*$', line)
            if m:
                self.types[m.group(1)] = (m.group(2), self.scalars.get(m.group(3)), m.group(4))
            m = re.match(r'\s*(%\S+) = Op\w+ (%\S+)', line)
            if m:
                self.valueTypes[m.group(1)] = m.group(2)

    def signed(self, name, count):
        values = self.vectors.get(name) or []
        return [v - (1 << 32) if v is not None and v >= (1 << 31) else v for v in values][:count]

    def shapeOfType(self, typeId):
        entry = self.types.get(typeId)
        return self.vectors.get(entry[2]) if entry and entry[2] else None

    def shapeOfValue(self, valueId):
        return self.shapeOfType(self.valueTypes.get(valueId))

    def ops(self):
        """(result id, result type, opcode, operands, line) for every graph operator, in program order."""
        out = []
        for index, line in enumerate(self.lines):
            m = re.match(r'\s*(%\S+) = OpExtInst (%\S+) %\S+ (\w+) (.*)', line)
            if m:
                out.append((m.group(1), m.group(2), m.group(3), m.group(4).split(), index))
        return out

    def graphInput(self):
        for index, line in enumerate(self.lines):
            m = re.match(r'\s*(%\S+) = OpGraphInputARM', line)
            if m:
                return m.group(1), index
        fail('the module has no graph input')

    def graphOutputs(self):
        out = []
        for line in self.lines:
            m = re.match(r'\s*OpGraphSetOutputARM (%\S+) ', line)
            if m:
                out.append(m.group(1))
        return out

    def find(self, pattern):
        for index, line in enumerate(self.lines):
            if re.search(pattern, line):
                return index
        return -1


def conv2dOut(inHW, kernelHW, pad, stride, dilation):
    out = []
    for i in range(2):
        padded = inHW[i] + pad[2 * i] + pad[2 * i + 1] - dilation[i] * (kernelHW[i] - 1) - 1
        if padded < 0 or padded % stride[i]:
            return None
        out.append(padded // stride[i] + 1)
    return out


def resizeOut(inHW, scale, offset, border):
    out = []
    for i in range(2):
        num = (inHW[i] - 1) * scale[2 * i] - offset[i] + border[i]
        if num < 0 or num % scale[2 * i + 1]:
            return None
        out.append(num // scale[2 * i + 1] + 1)
    return out


def infer(module, inputShape):
    """result id -> shape, for the graph input and every operator result."""
    shapes = {module.graphInput()[0]: list(inputShape)}

    for resultId, _, opcode, operands, _ in module.ops():
        if opcode == 'CONV2D':
            pad, stride, dilation = module.signed(operands[0], 4), module.signed(operands[1], 2), module.signed(operands[2], 2)
            src, weights = shapes[operands[5]], module.shapeOfValue(operands[6])
            hw = conv2dOut(src[1:3], weights[1:3], pad, stride, dilation)
            if hw is None:
                fail(f'CONV2D on {src} with pad {pad}, stride {stride} does not produce whole dimensions: '
                     f'pick a resolution the whole network divides')
            shapes[resultId] = [src[0], hw[0], hw[1], weights[0]]
        elif opcode == 'RESIZE':
            scale, offset, border = module.signed(operands[2], 4), module.signed(operands[3], 2), module.signed(operands[4], 2)
            src = shapes[operands[1]]
            hw = resizeOut(src[1:3], scale, offset, border)
            if hw is None:
                fail(f'RESIZE on {src} with scale {scale} does not produce whole dimensions')
            shapes[resultId] = [src[0], hw[0], hw[1], src[3]]
        elif opcode == 'CONCAT':
            axis = module.scalars.get(operands[0])
            parts = [shapes[o] for o in operands[1:]]
            shape = list(parts[0])
            shape[axis] = sum(p[axis] for p in parts)
            shapes[resultId] = shape
        elif opcode in ('RESCALE', 'TABLE'):
            shapes[resultId] = list(shapes[next(o for o in operands if o in shapes)])
        else:
            fail(f'unhandled operator {opcode}: this script only knows the NSS graph')

    return shapes


def specialize(module, inputShape):
    """Gives every operator result and the graph interface a shaped tensor type. Returns (lines, output shapes)."""
    shapes = infer(module, inputShape)
    inputId, inputLine = module.graphInput()
    outputIds = module.graphOutputs()

    uintByValue = {}
    for name, value in module.scalars.items():
        if re.match(r'%uint_\d+$', name) and value not in uintByValue:
            uintByValue[value] = name

    arrayType = None
    for line in module.lines:
        m = re.match(r'\s*(%\S+) = OpTypeArray %uint (%\S+)\s*$', line)
        if m and module.scalars.get(m.group(2)) == 4:
            arrayType = m.group(1)
    if not arrayType:
        fail('the module declares no uint[4] array type')

    block, newScalars, tensorTypes = [], {}, {}

    def scalarId(value):
        if value in uintByValue:
            return uintByValue[value]
        if value not in newScalars:
            newScalars[value] = f'%nss_uint_{value}'
            block.append(f'{newScalars[value]} = OpConstant %uint {value}')
        return newScalars[value]

    def tensorType(element, shape):
        key = (element, tuple(shape))
        if key not in tensorTypes:
            index = len(tensorTypes)
            shapeId = f'%nss_shape_{index}'
            block.append(f'{shapeId} = OpConstantComposite {arrayType} {" ".join(scalarId(d) for d in shape)}')
            tensorTypes[key] = f'%nss_tensor_{index}'
            block.append(f'{tensorTypes[key]} = OpTypeTensorARM {element} {scalarId(len(shape))} {shapeId}')
        return tensorTypes[key]

    newTypeOf = {inputId: tensorType(module.types[module.valueTypes[inputId]][0], shapes[inputId])}
    for resultId, resultType, _, _, _ in module.ops():
        newTypeOf[resultId] = tensorType(module.types[resultType][0], shapes[resultId])

    lines = list(module.lines)
    for resultId, resultType, _, _, index in module.ops():
        lines[index] = lines[index].replace(f'= OpExtInst {resultType} ', f'= OpExtInst {newTypeOf[resultId]} ', 1)
    lines[inputLine] = re.sub(r'= OpGraphInputARM \S+', f'= OpGraphInputARM {newTypeOf[inputId]}', lines[inputLine])

    moved = [index for index, line in enumerate(lines) if re.search(r'= OpVariable \S+ UniformConstant|= OpTypeGraphARM ', line)]
    interfaceTypes = [newTypeOf[inputId]] + [newTypeOf[o] for o in outputIds]
    for slot, typeId in enumerate(interfaceTypes):
        block.append(f'%nss_ptr_{slot} = OpTypePointer UniformConstant {typeId}')
    variables = [lines[i] for i in moved if 'OpVariable' in lines[i]]
    if len(variables) != len(interfaceTypes):
        fail(f'the module declares {len(variables)} interface variables, the graph has {len(interfaceTypes)}')
    for slot, line in enumerate(variables):
        block.append(re.sub(r'= OpVariable \S+ UniformConstant', f'= OpVariable %nss_ptr_{slot} UniformConstant', line.strip()))
    graphTypeLine = next(lines[i] for i in moved if 'OpTypeGraphARM' in lines[i])
    graphTypeId = re.match(r'\s*(%\S+) = ', graphTypeLine).group(1)
    block.append(f'{graphTypeId} = OpTypeGraphARM 1 {" ".join(interfaceTypes)}')

    insertAt = module.find(r'OpGraphEntryPointARM')
    if insertAt < 0:
        fail('the module has no graph entry point')
    out = []
    for index, line in enumerate(lines):
        if index == insertAt:
            out.extend('       ' + b for b in block)
        if index not in moved:
            out.append(line)
    return out, [shapes[o] for o in outputIds]


def dumpResources(tool, path):
    """(index, category, shape) for every resource, and the resource each graph input/output is bound to."""
    resources, inputs, outputs = [], [], []
    for line in run([tool, '--dump', path]).splitlines():
        m = re.match(r'resource (\d+) (\w+) format -?\d+ shape \[(.*)\]', line)
        if m:
            resources.append((int(m.group(1)), m.group(2), [int(v) for v in m.group(3).split(', ')]))
        m = re.match(r'(input|output) \d+: binding \d+ -> resource (\d+)', line)
        if m:
            (inputs if m.group(1) == 'input' else outputs).append(int(m.group(2)))
    if not resources:
        fail(f'cannot read the resource table of {path}')
    return resources, inputs, outputs


def main():
    render = option('--render', '1920x1080').lower()
    if not re.fullmatch(r'\d+x\d+', render):
        fail('--render takes <width>x<height>, for example 1920x1080')
    width, height = (int(v) for v in render.split('x'))

    model = option('--source', MODEL)
    buildDir = option('--build-dir', os.path.join(ROOT, 'build'))
    output = option('--output', os.path.join(CONTENT, f'2_nss-{width}x{height}-v1_0_1.vgf'))
    graphWidth = (width + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    graphHeight = (height + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    if not os.path.exists(model) and model == MODEL:
        print(f'Downloading {MODEL_URL}...')
        os.makedirs(CONTENT, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)
    if not os.path.exists(model):
        fail(f'`{model}` is missing')
    if os.path.abspath(output) == os.path.abspath(model):
        fail('the output would overwrite the source model: pick another --render or --output')

    spirvAs, spirvDis, spirvVal = findTool('spirv-as'), findTool('spirv-dis'), findTool('spirv-val')
    tool = buildTool(buildDir)

    work = os.path.join(buildDir, 'nss-model')
    os.makedirs(work, exist_ok=True)
    extracted = os.path.join(work, 'nss_source.spv')
    open(extracted, 'wb').write(extractModule(model))
    module = Module(run([spirvDis, '--no-color', extracted]).splitlines())

    resources, inputSlots, outputSlots = dumpResources(tool, model)
    inputShape = [1, graphHeight, graphWidth, resources[inputSlots[0]][2][3]]
    lines, outputShapes = specialize(module, inputShape)
    print(f'inferred input {inputShape} -> {" and ".join(str(s) for s in outputShapes)}')

    shapes = {inputSlots[0]: inputShape}
    for slot, shape in zip(outputSlots, outputShapes):
        shapes[slot] = shape

    assembly = os.path.join(work, f'nss_{graphHeight}x{graphWidth}.spvasm')
    spirv = os.path.splitext(assembly)[0] + '.spv'
    open(assembly, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    run([spirvAs, '--target-env', 'spv1.6', '-o', spirv, assembly])
    run([spirvVal, '--target-env', 'vulkan1.3', spirv])
    print(f'{os.path.getsize(spirv)} bytes of SPIR-V, validated')

    args = [f'{index}:{",".join(str(d) for d in shape)}' for index, shape in sorted(shapes.items())]
    print(run([tool, model, spirv, output, *args]).strip())
    print(output)


if __name__ == '__main__':
    main()
