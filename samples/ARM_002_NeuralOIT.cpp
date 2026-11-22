/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 Order-independent transparency on a fully transparent Bistro:

   A-buffer             the reference: the K nearest fragments of a pixel, kept sorted by a lock-free insertion and blended front
                        to back; the fragments pushed out of the list are tail-blended behind them
   Weighted blended     "Weighted Blended Order-Independent Transparency" (McGuire and Bavoil, JCGT 2013): one geometry pass
                        accumulates every fragment weighted by a function of its depth and its alpha, and the pixel is the
                        weighted average revealed against the background. `--weight-distance` is the distance at which the
                        depth term of the weight reaches 0.03; it has to be tuned per scene, and 250 is the value that
                        minimizes the error on Bistro at this scale
   DFAOIT               "Deep and Fast Approximate Order Independent Transparency" (Tsopouridis et al., CGF 2024): per-pixel
                        features from one geometry pass, the colour of the pixel predicted from them by a small neural network
                        dispatched as a VK_ARM_data_graph pipeline over tensors. The k nearest fragments of a pixel are kept
                        exactly and composited analytically, and the network only predicts what is behind them; k is the
                        paper's DFAOIT_k, and the sample reads it out of the network as `inputs - 10` rather than being told.
                        A pixel with no more than k fragments therefore never reads the network, and is left out of its
                        input tensor: the pack pass compacts the rest into its rows (as wide as the window, with room for
                        3/4 of its pixels)

 The network is `third-party/content/src/neural/dfaoit/dfaoit-16x16.bin` (14 -> 16 -> 16 -> 3), the integer weights that
 `train_dfaoit.py` next to it writes, and the sample builds its TOSA graph for its tensor shape at start-up. The network is quantized to
 8 bits, the integer graph an NPU runs natively: int8 tensors in and out, the features quantized to `round(x * 255) - 128` on the way in
 and the tail read back as `(v + 128) / 255`.

 `--technique wboit|abuffer|dfaoit` (default `wboit`, the cheapest), `--alpha A`, `--layers N`, `--weight-distance D`, `--compare` (mean
 squared error against the A-buffer), `--diff` (show the difference), `--timings` (also prints the GPU time of every stage to the log); it
 renders at the size of the monitor, or `--width W --height H`. On Android the same options are read from `args.txt` next to the OBB, and
 the numbers go to logcat.
*/

#include <algorithm>
#include <cstring>
#include <string>

#include "Bistro.h"

#define MODEL_PATH "src/bistro/Exterior/exterior.obj"
#define CACHE_FILE_NAME "cache.data"

enum {
#if defined(ANDROID)
  kAndroidTensorWidth = 1170,
  kAndroidTensorHeight = 408,
#endif // ANDROID
  kMaxOitLayers = 32,
};

constexpr size_t getKeysSize(uint32_t numPixels, uint32_t numSlots) {
  return sizeof(uint64_t) * numPixels * numSlots;
}

// GLSL: the push constants of the mesh vertex and fragment shaders, `MeshPushConstants`
#define MESH_PUSH_CONSTANTS_GLSL                                         \
  "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n" \
  "layout(std430, buffer_reference) readonly buffer PerFrame {\n"        \
  "  mat4 proj;\n"                                                       \
  "  mat4 view;\n"                                                       \
  "  mat4 model;\n"                                                      \
  "};\n"                                                                 \
  "layout(std430, buffer_reference) readonly buffer Materials {\n"       \
  "  Material mtl[];\n"                                                  \
  "};\n"                                                                 \
  "layout(std430, buffer_reference) coherent buffer Keys {\n"            \
  "  uint64_t keys[];\n"                                                 \
  "};\n"                                                                 \
  "layout(push_constant) uniform constants {\n"                          \
  "  PerFrame perFrame;\n"                                               \
  "  Materials materials;\n"                                             \
  "  Keys aBuffer;\n"                                                    \
  "  Keys nearest;\n"                                                    \
  "  uint sampler0;\n"                                                   \
  "  uint layers;\n"                                                     \
  "  uint viewWidth;\n"                                                  \
  "  float alpha;\n"                                                     \
  "  float weightDistance;\n"                                            \
  "} pc;\n"

const char* kCodeMeshVS = BISTRO_GLSL MESH_PUSH_CONSTANTS_GLSL R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

layout (location=0) out vec3 normalWS;
layout (location=1) out vec2 uvOut;
layout (location=2) flat out uint materialId;
layout (location=3) out float viewDistance;

void main() {
  vec4 posVS = pc.perFrame.view * pc.perFrame.model * vec4(pos, 1.0);
  gl_Position = pc.perFrame.proj * posVS;
  normalWS = unpackOctahedral16(normal);
  uvOut = uv;
  materialId = mtlIndex;
  viewDistance = length(posVS.xyz);
}
)";

const char* kCodeMeshFS = BISTRO_GLSL MESH_PUSH_CONSTANTS_GLSL R"(
layout (location=0) in vec3 normalWS;
layout (location=1) in vec2 uv;
layout (location=2) flat in uint materialId;

vec4 shade() {
  Material mtl = pc.materials.mtl[materialId];
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, pc.sampler0, uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, pc.sampler0, uv);
  float mask = mtl.texAlpha > 0 ? textureBindless2D(mtl.texAlpha, pc.sampler0, uv).r : 1.0;

  vec3 n = normalize(normalWS);
  float NdotL1 = clamp(dot(n, normalize(vec3(-1, 1, +1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1 + NdotL2);

  return vec4(clamp(Ka.rgb + Kd.rgb * (0.4 + 0.6 * NdotL), vec3(0.0), vec3(1.0)), pc.alpha * mask * Kd.a);
}

const uint64_t kEmptyKey = 0xFFFFFFFFFFFFFFFFul;

uint64_t insertKey(Keys list, uint numSlots, vec4 color) {
  if (gl_HelperInvocation)
    return kEmptyKey;

  uint first = (uint(gl_FragCoord.y) * pc.viewWidth + uint(gl_FragCoord.x)) * numSlots;
  uint64_t key = (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(packUnorm4x8(color));

  if (numSlots <= 8u && key >= list.keys[first + numSlots - 1u]) {
    return key;
  }

  for (uint slot = 0u; slot != numSlots && key != kEmptyKey; slot++) {
    uint64_t previous = atomicMin(list.keys[first + slot], key);
    key = previous > key ? previous : key;
  }

  return key;
}
)";

const char* kCodeABufferFS = R"(
layout (location=0) out vec4 out_Tail;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  uint64_t evicted = insertKey(pc.aBuffer, pc.layers & 0x7FFFFFFFu, color);

  out_Tail = vec4(0.0);

  if (evicted != kEmptyKey && (pc.layers & 0x80000000u) == 0u) {
    vec4 tail = unpackUnorm4x8(uint(evicted));
    out_Tail = vec4(tail.rgb * tail.a, tail.a);
  }
}
)";

const char* kCodeWeightedFS = R"(
layout (location=3) in float viewDistance;

layout (location=0) out vec4 out_Accumulated;
layout (location=1) out vec4 out_Revealage;
layout (location=2) out float out_Count;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  vec3 premultiplied = color.rgb * color.a;

  float distanceWeight = clamp(0.03 / (1e-5 + pow(viewDistance / pc.weightDistance, 4.0)), 1e-2, 3e3);
  float alphaWeight = min(1.0, max(max(premultiplied.r, premultiplied.g), max(premultiplied.b, color.a)) * 40.0 + 0.01);

  out_Accumulated = vec4(premultiplied, color.a) * (alphaWeight * alphaWeight * distanceWeight);
  out_Revealage = vec4(color.a);
  out_Count = 1.0;
}
)";

const char* kCodeWeightedResolveCS = R"(
layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 2, rgba16f) uniform writeonly image2D kImagesRGBA16F[];

layout(push_constant) uniform constants {
  vec4 background;
  uint texAccumulated;
  uint texRevealage;
  uint imageOut;
  uint viewWidth;
  uint viewHeight;
} pc;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  if (coord.x >= int(pc.viewWidth) || coord.y >= int(pc.viewHeight))
    return;

  vec4 accumulated = texelFetch(kTextures2D[pc.texAccumulated], coord, 0);
  float revealage = texelFetch(kTextures2D[pc.texRevealage], coord, 0).r;

  vec3 color = accumulated.rgb / max(accumulated.a, 1e-5);

  imageStore(kImagesRGBA16F[pc.imageOut], coord, vec4(mix(color, pc.background.rgb, revealage), 1.0));
}
)";

const char* kCodeFeaturesFS = R"(
layout (location=0) out vec4 out_Sum;
layout (location=1) out vec4 out_Accumulated;
layout (location=2) out float out_Count;

void main() {
  vec4 color = shade();

  if (color.a < 1.0 / 255.0)
    discard;

  out_Sum = color;
  out_Accumulated = vec4(color.rgb * color.a, color.a);
  out_Count = 1.0;

  insertKey(pc.nearest, kDfaoitNumNearest, color);
}
)";

const char* kCodeFullscreenVS = R"(
void main() {
  gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeCompositeFS = R"(
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer Keys {
  uint64_t keys[];
};

layout(push_constant) uniform constants {
  Keys aBuffer;
  uint layers;
  uint viewWidth;
  uint tailTexture;
} pc;

const uint64_t kEmptyKey = 0xFFFFFFFFFFFFFFFFul;

void blendOver(inout vec4 dst, vec4 src) {
  dst.rgb += (1.0 - dst.a) * src.rgb;
  dst.a += (1.0 - dst.a) * src.a;
}

void main() {
  vec4 tail = texelFetch(kTextures2D[pc.tailTexture], ivec2(gl_FragCoord.xy), 0);
  vec4 color = vec4(0.0);

  if (!gl_HelperInvocation) {
    uint first = (uint(gl_FragCoord.y) * pc.viewWidth + uint(gl_FragCoord.x)) * pc.layers;

    for (uint i = 0u; i != pc.layers; i++) {
      uint64_t key = pc.aBuffer.keys[first + i];
      if (key == kEmptyKey)
        break;
      vec4 fragment = unpackUnorm4x8(uint(key));
      blendOver(color, vec4(fragment.rgb * fragment.a, fragment.a));
    }
  }

  blendOver(color, tail);

  out_FragColor = color;
}
)";

const char* kCodeDfaoitCommon = R"(
layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];

layout(std430, buffer_reference) readonly buffer Nearest {
  uint words[];
};

layout(std430, buffer_reference) buffer Uints {
  uint value[];
};

layout(std430, buffer_reference) buffer TensorBytes {
  int8_t value[];
};

layout(push_constant) uniform constants {
  vec4 background;
  Nearest nearest;
  Uints slots;
  Uints counter;
  TensorBytes tensorInput;
  TensorBytes tensorOutput;
  uint texSum;
  uint texAccumulated;
  uint texCount;
  uint imageOut;
  uint viewWidth;
  uint viewHeight;
} pc;

const uint kNoSlot = 0xFFFFFFFFu;

struct Pixel {
  uint count;
  vec3 nearest;
  float attenuation;
  float transmittance;
  float inputs[kDfaoitNumInputs];
};

Pixel loadPixel(ivec2 coord) {
  vec4 sum = texelFetch(kTextures2D[pc.texSum], coord, 0);
  vec4 accumulated = texelFetch(kTextures2D[pc.texAccumulated], coord, 0);
  uint index = uint(coord.y) * pc.viewWidth + uint(coord.x);

  Pixel p;
  p.count = uint(texelFetch(kTextures2D[pc.texCount], coord, 0).r + 0.5);
  p.transmittance = accumulated.a;

  vec4 nearSum = vec4(0.0);
  vec3 exact = vec3(0.0);
  float alphas[kDfaoitNumNearest];

  p.attenuation = 1.0;

  for (uint i = 0u; i != kDfaoitNumNearest; i++) {
    vec4 f = p.count > i ? unpackUnorm4x8(pc.nearest.words[2u * kDfaoitNumNearest * index + 2u * i]) : vec4(0.0);
    exact += p.attenuation * f.a * f.rgb;
    p.attenuation *= 1.0 - f.a;
    nearSum += f;
    alphas[i] = f.a;
  }

  p.nearest = exact;

  vec4 average = p.count > kDfaoitNumNearest ? max(sum - nearSum, vec4(0.0)) / float(p.count - kDfaoitNumNearest) : vec4(0.0);

  vec3 squashed = accumulated.rgb / (1.0 + accumulated.rgb);

  p.inputs[0] = average.a;
  p.inputs[1] = average.r;
  p.inputs[2] = average.g;
  p.inputs[3] = average.b;
  p.inputs[4] = squashed.r;
  p.inputs[5] = squashed.g;
  p.inputs[6] = squashed.b;
  p.inputs[7] = p.nearest.r;
  p.inputs[8] = p.nearest.g;
  p.inputs[9] = p.nearest.b;

  for (uint i = 0u; i != kDfaoitNumNearest; i++) {
    p.inputs[10u + i] = alphas[i];
  }

  return p;
}
)";

const char* kCodeDfaoitExtensions = R"(
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
)";

const char* kCodeDfaoitPackCS = R"(
shared uint sGroupCount;
shared uint sGroupBase;

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
  bool inside = coord.x < int(pc.viewWidth) && coord.y < int(pc.viewHeight);

  Pixel p;
  bool wanted = false;
  if (inside) {
    p = loadPixel(coord);
    wanted = p.count > kDfaoitNumNearest;
  }

  if (gl_LocalInvocationIndex == 0u)
    sGroupCount = 0u;
  barrier();
  uint local = wanted ? atomicAdd(sGroupCount, 1u) : 0u;
  barrier();
  if (gl_LocalInvocationIndex == 0u)
    sGroupBase = atomicAdd(pc.counter.value[0], sGroupCount);
  barrier();

  if (!inside)
    return;

  uint slot = kNoSlot;

  if (wanted) {
    const uint candidate = sGroupBase + local;
    if (candidate < kDfaoitTensorPixels) {
      slot = candidate;
      for (uint i = 0u; i != kDfaoitNumInputs; i++) {
        pc.tensorInput.value[slot * kDfaoitNumInputs + i] = int8_t(clamp(int(round(p.inputs[i] * 255.0)) - 128, -128, 127));
      }
    }
  }

  pc.slots.value[uint(coord.y) * pc.viewWidth + uint(coord.x)] = slot;
}
)";

const char* kCodeDfaoitResolveCS = R"(
layout (set = 0, binding = 2, rgba16f) uniform writeonly image2D kImagesRGBA16F[];

void main() {
  ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

  if (coord.x >= int(pc.viewWidth) || coord.y >= int(pc.viewHeight))
    return;

  Pixel p = loadPixel(coord);

  vec3 color = p.nearest;

  if (p.count > kDfaoitNumNearest) {
    const uint slot = pc.slots.value[uint(coord.y) * pc.viewWidth + uint(coord.x)];
    if (slot != kNoSlot) {
      const uint base = slot * kDfaoitNumOutputs;
      const vec3 outputs = vec3(pc.tensorOutput.value[base], pc.tensorOutput.value[base + 1u], pc.tensorOutput.value[base + 2u]);
      color += p.attenuation * (outputs + 128.0) / 255.0;
    } else {
      const float coverage = 1.0 - pow(1.0 - p.inputs[0], float(p.count - kDfaoitNumNearest));
      color += p.attenuation * coverage * vec3(p.inputs[1], p.inputs[2], p.inputs[3]);
    }
  }

  imageStore(kImagesRGBA16F[pc.imageOut], coord, vec4(color + p.transmittance * pc.background.rgb, 1.0));
}
)";

const char* kCodeErrorCS = R"(
layout (local_size_x = 1, local_size_y = 64) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];

layout(std430, buffer_reference) writeonly buffer RowStats {
  vec4 row[];
};

layout(push_constant) uniform constants {
  RowStats rows;
  uint texResult;
  uint texReference;
  uint texCount;
  uint layers;
  uint viewWidth;
  uint viewHeight;
} pc;

void main() {
  uint y = gl_GlobalInvocationID.y;

  if (y >= pc.viewHeight)
    return;

  vec4 stats = vec4(0.0);

  for (uint x = 0u; x != pc.viewWidth; x++) {
    vec3 d = texelFetch(kTextures2D[pc.texResult], ivec2(x, y), 0).rgb - texelFetch(kTextures2D[pc.texReference], ivec2(x, y), 0).rgb;
    float count = texelFetch(kTextures2D[pc.texCount], ivec2(x, y), 0).r;
    stats.x += dot(d, d);
    stats.y += count;
    stats.z = max(stats.z, count);
    stats.w += count > float(pc.layers) ? 1.0 : 0.0;
  }

  pc.rows.row[y] = stats;
}
)";

// GLSL: the push constants of the present vertex and fragment shaders, `PresentPushConstants`
#define PRESENT_PUSH_CONSTANTS_GLSL             \
  "layout(push_constant) uniform constants {\n" \
  "  vec2 clipRotation;\n"                      \
  "  uint texResult;\n"                         \
  "  uint texReference;\n"                      \
  "  uint sampler0;\n"                          \
  "  uint showDifference;\n"                    \
  "  float gain;\n"                             \
  "} pc;\n"

const char* kCodePresentVS = FULLSCREEN_TRIANGLE_GLSL PRESENT_PUSH_CONSTANTS_GLSL R"(
layout (location=0) out vec2 uv;

void main() {
  gl_Position = fullscreenTriangle(pc.clipRotation, uv);
}
)";

const char* kCodePresentFS = PRESENT_PUSH_CONSTANTS_GLSL R"(
layout (location=0) in vec2 uv;

layout (location=0) out vec4 out_FragColor;

void main() {
  vec3 color = textureBindless2D(pc.texResult, pc.sampler0, uv).rgb;

  if (pc.showDifference != 0u) {
    color = pc.gain * abs(color - textureBindless2D(pc.texReference, pc.sampler0, uv).rgb);
  }

  out_FragColor = vec4(color, 1.0);
}
)";

enum Technique {
  Technique_ABuffer = 0,
  Technique_WeightedBlended,
  Technique_DFAOIT,
};

enum GPUTimestamp {
  GPUTimestamp_Begin = 0,
  GPUTimestamp_ABuffer,
  GPUTimestamp_Features,
  GPUTimestamp_Inference,
  GPUTimestamp_Resolve,
  GPUTimestamp_Present,
  GPUTimestamp_NUM,
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  mat4 model;
};

struct MeshPushConstants {
  uint64_t perFrame;
  uint64_t materials;
  uint64_t aBuffer;
  uint64_t nearest;
  uint32_t sampler0;
  uint32_t layers;
  uint32_t viewWidth;
  float alpha;
  float weightDistance;
};

struct CompositePushConstants {
  uint64_t aBuffer;
  uint32_t layers;
  uint32_t viewWidth;
  uint32_t tailTexture;
};

struct WeightedResolvePushConstants {
  vec4 background;
  uint32_t texAccumulated;
  uint32_t texRevealage;
  uint32_t imageOut;
  uint32_t viewWidth;
  uint32_t viewHeight;
};

struct DfaoitPushConstants {
  vec4 background;
  uint64_t nearest;
  uint64_t slots;
  uint64_t counter;
  uint64_t tensorInput;
  uint64_t tensorOutput;
  uint32_t texSum;
  uint32_t texAccumulated;
  uint32_t texCount;
  uint32_t imageOut;
  uint32_t viewWidth;
  uint32_t viewHeight;
};
static_assert(sizeof(DfaoitPushConstants) == 80);

struct ErrorPushConstants {
  uint64_t rows;
  uint32_t texResult;
  uint32_t texReference;
  uint32_t texCount;
  uint32_t layers;
  uint32_t viewWidth;
  uint32_t viewHeight;
};

struct RowStats {
  float squaredError = 0.0f;
  float numFragments = 0.0f;
  float maxFragments = 0.0f;
  float numPixelsAboveLayers = 0.0f;
};

struct PresentPushConstants {
  vec2 clipRotation;
  uint32_t texResult;
  uint32_t texReference;
  uint32_t sampler0;
  uint32_t showDifference;
  float gain;
};

// the integer network `train_dfaoit.py` writes (its docstring has the layout)
struct MlpLayer {
  uint32_t numInputs = 0;
  uint32_t numOutputs = 0;
  uint32_t activation = 0; // 1 ReLU, 2 sigmoid
  int32_t outputZeroPoint = 0;
  const int8_t* weights = nullptr; // OHWI, H = W = 1
  const int32_t* biases = nullptr;
  const int32_t* multipliers = nullptr;
  const int8_t* shifts = nullptr;
  const int8_t* table = nullptr; // 256 entries, a sigmoid only
};

struct Mlp {
  uint32_t numInputs = 0;
  int32_t inputZeroPoint = 0;
  std::vector<MlpLayer> layers;
};

enum {
  kMlpMagic = 0x51504C4D, // 'MLPQ'
  kMlpTableSize = 256,
  kActivationReLU = 1,
  kActivationSigmoid = 2,
};

bool parseMlp(const std::vector<uint8_t>& blob, Mlp& out) {
  if (blob.size() < 4 * sizeof(uint32_t)) {
    return false;
  }
  const uint32_t* header = (const uint32_t*)blob.data();
  const uint32_t numLayers = header[2];
  const size_t headerWords = 4 + 3 * size_t(numLayers);
  if (header[0] != kMlpMagic || !header[1] || !numLayers || blob.size() < headerWords * sizeof(uint32_t)) {
    return false;
  }
  out = {.numInputs = header[1], .inputZeroPoint = (int32_t)header[3]};
  const uint8_t* data = blob.data() + headerWords * sizeof(uint32_t);
  const uint8_t* end = blob.data() + blob.size();
  auto take = [&data, end](size_t bytes) -> const uint8_t* {
    if (bytes > size_t(end - data)) {
      return nullptr;
    }
    const uint8_t* start = data;
    data += bytes;
    return start;
  };
  uint32_t numInputs = out.numInputs;
  for (uint32_t i = 0; i != numLayers; i++) {
    const uint32_t* layerHeader = header + 4 + 3 * i;
    const uint32_t numOutputs = layerHeader[0];
    const uint32_t activation = layerHeader[1];
    const bool isSigmoid = activation == kActivationSigmoid;
    const MlpLayer layer = {
        .numInputs = numInputs,
        .numOutputs = numOutputs,
        .activation = activation,
        .outputZeroPoint = (int32_t)layerHeader[2],
        .weights = (const int8_t*)take(size_t(numOutputs) * numInputs),
        .biases = (const int32_t*)take(numOutputs * sizeof(int32_t)),
        .multipliers = (const int32_t*)take(numOutputs * sizeof(int32_t)),
        .shifts = (const int8_t*)take(numOutputs),
        .table = isSigmoid ? (const int8_t*)take(kMlpTableSize) : nullptr,
    };
    const bool isValid = numOutputs && (activation == kActivationReLU || isSigmoid) && layer.weights && layer.biases && layer.multipliers &&
                         layer.shifts && (!isSigmoid || layer.table);
    if (!isValid) {
      return false;
    }
    out.layers.push_back(layer);
    numInputs = numOutputs;
  }
  return true;
}

// the TOSA integer graph of an `Mlp` over [1, height, width, channels]: per layer an int8 1x1 CONV2D into int32, a per-channel
// RESCALE back to int8 and, for a sigmoid, a TABLE; a ReLU needs no operator because its output zero point of -128 saturates
// the negatives
std::vector<uint32_t> buildMlpGraph(const Mlp& mlp, uint32_t width, uint32_t height) {
  TosaGraphBuilder graph(width, height, mlp.numInputs);
  uint32_t value = graph.input();
  int32_t zeroPoint = mlp.inputZeroPoint;
  for (const MlpLayer& layer : mlp.layers) {
    value = graph.rescale(graph.conv2d(value, layer.numOutputs, zeroPoint), false, layer.multipliers, layer.shifts, layer.outputZeroPoint);
    zeroPoint = layer.outputZeroPoint;
    if (layer.activation == kActivationSigmoid) {
      value = graph.table(value, layer.table);
    }
  }
  return graph.build(value);
}

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      // the size of the monitor unless `--width` and `--height` say otherwise
      .width = 0,
      .height = 0,
      .resizable = false,
      .initialCameraPos = vec3(-100, 40, -47),
      .initialCameraTarget = vec3(0, 35, 0),
      .contextConfig =
          {
              .enableValidationGpuAV = false,
              .swapchainUseSurfacePreTransform = true,
              .layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"},
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  int technique = Technique_WeightedBlended;
  int numLayers = 16;
  float weightDistance = 250.0f;
  float alpha = 0.3f;
  bool compare = false;
  bool showDifference = false;
  bool measureTime = false;

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];
    const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
    if (!strcmp(arg, "--technique") && next) {
      technique = !strcmp(next, "dfaoit") ? Technique_DFAOIT : !strcmp(next, "abuffer") ? Technique_ABuffer : Technique_WeightedBlended;
      i++;
    } else if (!strcmp(arg, "--weight-distance") && next) {
      weightDistance = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--alpha") && next) {
      alpha = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--layers") && next) {
      numLayers = atoi(next);
      i++;
    } else if (!strcmp(arg, "--compare")) {
      compare = true;
    } else if (!strcmp(arg, "--diff")) {
      compare = true;
      showDifference = true;
    } else if (!strcmp(arg, "--timings")) {
      measureTime = true;
    }
  }

  lvk::IContext* ctx = app.ctx_.get();

  {
    const std::string cacheFileName = app.folderContentRoot_ + CACHE_FILE_NAME;

    if (!loadFromCache(app, cacheFileName.c_str())) {
      if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), MODEL_PATH))) {
        LVK_ASSERT_MSG(false, "Cannot load 3D model. Run `deploy_content.py` before running this app.");
        VULKAN_APP_EXIT();
      }
    }
  }

  loadMaterialTextures(app, "src/bistro/Exterior/");

  lvk::Holder<lvk::BufferHandle> sbMaterials = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GPUMaterial) * materials_.size(),
      .data = materials_.data(),
      .debugName = "Buffer: materials",
  });
  lvk::Holder<lvk::BufferHandle> vb0 = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Vertex,
      .storage = lvk::StorageType_Device,
      .size = sizeof(VertexData) * vertexData_.size(),
      .data = vertexData_.data(),
      .debugName = "Buffer: vertex",
  });
  lvk::Holder<lvk::BufferHandle> ib0 = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Index,
      .storage = lvk::StorageType_Device,
      .size = sizeof(uint32_t) * indexData_.size(),
      .data = indexData_.data(),
      .debugName = "Buffer: index",
  });
  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });

  const uint32_t numIndices = (uint32_t)indexData_.size();

  vertexData_ = {};
  indexData_ = {};

  const std::string modelFileName = app.folderContentRoot_ + "src/neural/dfaoit/dfaoit-16x16.bin";

  Mlp mlp;
  lvk::Holder<lvk::BufferHandle> tensorInputBuffer;
  lvk::Holder<lvk::BufferHandle> tensorOutputBuffer;
  lvk::Holder<lvk::TensorHandle> tensorInput;
  lvk::Holder<lvk::TensorHandle> tensorOutput;
  lvk::Holder<lvk::DataGraphPipelineHandle> pipelineGraph;

  const std::vector<uint8_t> blobMlp = app.loadFile(modelFileName.c_str());

  // the graph needs a fixed shape: the tensors are as wide as the window and have room for 3/4 of its pixels
#if defined(ANDROID)
  const uint32_t tensorWidth = kAndroidTensorWidth;
  const uint32_t tensorHeight = kAndroidTensorHeight;
#else
  const uint32_t tensorWidth = (uint32_t)app.width_;
  const uint32_t tensorHeight = (uint32_t)app.height_ * 3 / 4;
#endif // ANDROID

  auto createGraph = [&]() -> bool {
    if (blobMlp.empty() || !parseMlp(blobMlp, mlp)) {
      LLOGW("Cannot load `%s`. Run `train_dfaoit.py` next to it.\n", modelFileName.c_str());
      return false;
    }
    // pack writes the k nearest fragments after 10 features (k >= 2) and resolve reads 3 outputs
    const bool isExpectedNetwork = mlp.layers.size() == 3 && mlp.numInputs >= 12 && mlp.layers.back().numOutputs == 3;
    if (!isExpectedNetwork) {
      LLOGW("`%s` is not the network this sample expects: 3 layers from at least 12 inputs to 3 outputs\n", modelFileName.c_str());
      return false;
    }
    if (!ctx->supportsTensorsARM() || !ctx->supportsDataGraphARM()) {
      LLOGW("VK_ARM_tensors and VK_ARM_data_graph are not available (configure with LVK_WITH_ML_EMULATION_LAYER=ON to emulate them)\n");
      return false;
    }
    auto tensorDesc = [&](uint32_t channels) -> lvk::TensorDesc {
      return {.format = lvk::Format_R_I8, .rank = 4, .dimensions = {1, tensorHeight, tensorWidth, channels}};
    };
    const lvk::TensorDesc descInput = tensorDesc(mlp.numInputs);
    const lvk::TensorDesc descOutput = tensorDesc(mlp.layers.back().numOutputs);
    lvk::Result result;
    auto isFailed = [&result]() {
      if (!result.isOk()) {
        LLOGW("Cannot create the data graph: %s\n", result.message);
      }
      return !result.isOk();
    };
    tensorInput = createTensorWithBuffer(ctx, descInput, tensorInputBuffer, "DFAOIT: input tensor", &result);
    if (isFailed()) {
      return false;
    }
    tensorOutput = createTensorWithBuffer(ctx, descOutput, tensorOutputBuffer, "DFAOIT: output tensor", &result);
    if (isFailed()) {
      return false;
    }
    const std::vector<uint32_t> spirv = buildMlpGraph(mlp, tensorWidth, tensorHeight);
    const lvk::Holder<lvk::ShaderModuleHandle> smGraph =
        ctx->createShaderModule({spirv.data(), spirv.size() * sizeof(uint32_t), lvk::Stage_DataGraph, "Shader Module: DFAOIT graph"});
    std::vector<lvk::DataGraphConstant> constants;
    for (uint32_t i = 0; i != mlp.layers.size(); i++) {
      const MlpLayer& layer = mlp.layers[i];
      constants.push_back({.id = 2 * i,
                           .format = lvk::Format_R_I8,
                           .rank = 4,
                           .dimensions = {layer.numOutputs, 1, 1, layer.numInputs},
                           .data = layer.weights});
      constants.push_back(
          {.id = 2 * i + 1, .format = lvk::Format_R_I32, .rank = 1, .dimensions = {layer.numOutputs}, .data = layer.biases});
    }
    // the pipeline is described by the tensors it gets bound to, which may be optimally tiled copies of the buffer-aliasing ones
    const lvk::TensorDesc boundInput = ctx->getTensorDesc(tensorInput);
    const lvk::TensorDesc boundOutput = ctx->getTensorDesc(tensorOutput);
    pipelineGraph = ctx->createDataGraphPipeline(
        {
            .smGraph = smGraph,
            .inputs = {&boundInput, 1},
            .outputs = {&boundOutput, 1},
            .constants = {constants.data(), constants.size()},
            .debugName = "DFAOIT: data graph",
        },
        &result);
    return !isFailed();
  };
  const bool graphReady = createGraph();
  technique = technique == Technique_DFAOIT && !graphReady ? Technique_WeightedBlended : technique;

  // the network reads k = inputs - 10 nearest fragments; without it nothing reads them
  const uint32_t numInputs = graphReady ? mlp.numInputs : 12u;
  const uint32_t numNearest = numInputs - 10u;
  const std::string nearest = "const uint kDfaoitNumNearest = " + std::to_string(numNearest) + ";\n";

  const std::string codeABufferFS = std::string(kCodeMeshFS) + kCodeABufferFS;
  const std::string codeWeightedFS = std::string(kCodeMeshFS) + kCodeWeightedFS;
  const std::string codeFeaturesFS = nearest + kCodeMeshFS + kCodeFeaturesFS;
  const std::string codeDfaoitCommon = kCodeDfaoitExtensions + nearest + "const uint kDfaoitNumInputs = " + std::to_string(numInputs) +
                                       ";\nconst uint kDfaoitNumOutputs = 3;\nconst uint kDfaoitTensorPixels = " +
                                       std::to_string(tensorWidth * tensorHeight) + ";\n" + kCodeDfaoitCommon;
  const std::string codeResolveCS = codeDfaoitCommon + kCodeDfaoitResolveCS;
  const std::string codePackCS = codeDfaoitCommon + kCodeDfaoitPackCS;

  lvk::Holder<lvk::ShaderModuleHandle> smMeshVert = ctx->createShaderModule({kCodeMeshVS, lvk::Stage_Vert, "Shader Module: mesh (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smABufferFrag =
      ctx->createShaderModule({codeABufferFS.c_str(), lvk::Stage_Frag, "Shader Module: A-buffer (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smWeightedFrag =
      ctx->createShaderModule({codeWeightedFS.c_str(), lvk::Stage_Frag, "Shader Module: weighted blended (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smWeightedResolveComp =
      ctx->createShaderModule({kCodeWeightedResolveCS, lvk::Stage_Comp, "Shader Module: weighted blended resolve (comp)"});
  lvk::Holder<lvk::ShaderModuleHandle> smFeaturesFrag =
      ctx->createShaderModule({codeFeaturesFS.c_str(), lvk::Stage_Frag, "Shader Module: DFAOIT features (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert =
      ctx->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smCompositeFrag =
      ctx->createShaderModule({kCodeCompositeFS, lvk::Stage_Frag, "Shader Module: A-buffer composite (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentFrag =
      ctx->createShaderModule({kCodePresentFS, lvk::Stage_Frag, "Shader Module: present (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentVert =
      ctx->createShaderModule({kCodePresentVS, lvk::Stage_Vert, "Shader Module: present (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smErrorComp =
      ctx->createShaderModule({kCodeErrorCS, lvk::Stage_Comp, "Shader Module: error (comp)"});
  lvk::Holder<lvk::ShaderModuleHandle> smResolveComp;
  lvk::Holder<lvk::ShaderModuleHandle> smPackComp;
  lvk::Holder<lvk::ComputePipelineHandle> pipelineResolve;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePack;

  if (graphReady) {
    smResolveComp = ctx->createShaderModule({codeResolveCS.c_str(), lvk::Stage_Comp, "Shader Module: DFAOIT resolve (comp)"});
    smPackComp = ctx->createShaderModule({codePackCS.c_str(), lvk::Stage_Comp, "Shader Module: DFAOIT pack (comp)"});
    pipelineResolve = ctx->createComputePipeline({.smComp = smResolveComp});
    pipelinePack = ctx->createComputePipeline({.smComp = smPackComp});
  }

  lvk::Holder<lvk::SamplerHandle> sampler = ctx->createSampler({
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
  });
  lvk::Holder<lvk::SamplerHandle> samplerClamp = ctx->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .debugName = "Sampler: clamp",
  });

  const lvk::Format formatColor = lvk::Format_RGBA_F16;
  const lvk::Format formatCount = lvk::Format_R_F16;

  const lvk::VertexInput vertexInput = {
      .attributes = {{.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
                     {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
                     {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
                     {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };
  const lvk::ColorAttachment blendPremultiplied = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_One,
      .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
  };
  auto additive = [](lvk::Format format) -> lvk::ColorAttachment {
    return {
        .format = format,
        .blendEnabled = true,
        .srcRGBBlendFactor = lvk::BlendFactor_One,
        .srcAlphaBlendFactor = lvk::BlendFactor_One,
        .dstRGBBlendFactor = lvk::BlendFactor_One,
        .dstAlphaBlendFactor = lvk::BlendFactor_One,
    };
  };
  const lvk::ColorAttachment blendAdditive = additive(formatColor);
  const lvk::ColorAttachment blendAdditiveTransmittance = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_One,
      .srcAlphaBlendFactor = lvk::BlendFactor_Zero,
      .dstRGBBlendFactor = lvk::BlendFactor_One,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcAlpha,
  };
  const lvk::ColorAttachment blendRevealage = {
      .format = formatColor,
      .blendEnabled = true,
      .srcRGBBlendFactor = lvk::BlendFactor_Zero,
      .srcAlphaBlendFactor = lvk::BlendFactor_Zero,
      .dstRGBBlendFactor = lvk::BlendFactor_OneMinusSrcColor,
      .dstAlphaBlendFactor = lvk::BlendFactor_OneMinusSrcColor,
  };
  const lvk::ColorAttachment blendCount = additive(formatCount);

  lvk::Holder<lvk::RenderPipelineHandle> pipelineABuffer = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smABufferFrag,
      .color = {blendPremultiplied},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: A-buffer",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineWeighted = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smWeightedFrag,
      .color = {blendAdditive, blendRevealage, blendCount},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: weighted blended",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineFeatures = ctx->createRenderPipeline({
      .vertexInput = vertexInput,
      .smVert = smMeshVert,
      .smFrag = smFeaturesFrag,
      .color = {blendAdditive, blendAdditiveTransmittance, blendCount},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: DFAOIT features",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelineComposite = ctx->createRenderPipeline({
      .smVert = smFullscreenVert,
      .smFrag = smCompositeFrag,
      .color = {blendPremultiplied},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: A-buffer composite",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelinePresent = ctx->createRenderPipeline({
      .smVert = smPresentVert,
      .smFrag = smPresentFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: present",
  });
  lvk::Holder<lvk::ComputePipelineHandle> pipelineError = ctx->createComputePipeline({.smComp = smErrorComp});
  lvk::Holder<lvk::ComputePipelineHandle> pipelineWeightedResolve = ctx->createComputePipeline({.smComp = smWeightedResolveComp});

  GPUStageTimer gpuTimer;
  gpuTimer.create(ctx, GPUTimestamp_NUM);

  lvk::Holder<lvk::BufferHandle> aBuffer;
  lvk::Holder<lvk::TextureHandle> tailTexture;
  lvk::Holder<lvk::TextureHandle> texExact;
  lvk::Holder<lvk::TextureHandle> texApproximate;
  lvk::Holder<lvk::TextureHandle> texSum;
  lvk::Holder<lvk::TextureHandle> texAccumulated;
  lvk::Holder<lvk::TextureHandle> texCount;
  lvk::Holder<lvk::BufferHandle> nearestBuffer;
  lvk::Holder<lvk::BufferHandle> slotsBuffer;
  lvk::Holder<lvk::BufferHandle> counterBuffer;
  lvk::Holder<lvk::BufferHandle> rowSums;

  uint32_t builtWidth = 0;
  uint32_t builtHeight = 0;

  bool tailBlend = true;
  float differenceGain = 8.0f;
  double meanSquaredError = 0.0;
  double meanFragments = 0.0;
  double maxFragments = 0.0;
  double percentAboveLayers = 0.0;
  uint64_t frameIndex = 0;

  const vec4 background = vec4(0.38f, 0.51f, 0.71f, 1.0f);

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const lvk::Dimensions dimSwapchain = ctx->getDimensions(views[0].colorTexture);
    const lvk::SurfaceTransform surfaceTransform = ctx->getSwapchainSurfaceTransform();
    const lvk::Dimensions dim = lvk::isQuarterTurn(surfaceTransform) ? lvk::Dimensions{dimSwapchain.height, dimSwapchain.width, 1}
                                                                     : dimSwapchain;
    const uint32_t viewSize = dim.width * dim.height;
    numLayers = std::clamp(numLayers, 1, (int)kMaxOitLayers);
    const uint32_t layers = (uint32_t)numLayers;

    if (builtWidth != dim.width || builtHeight != dim.height) {
      builtWidth = dim.width;
      builtHeight = dim.height;

      auto createTarget = [&](lvk::Format format, uint8_t usage, const char* debugName) -> lvk::Holder<lvk::TextureHandle> {
        return ctx->createTexture({.format = format, .dimensions = dim, .usage = usage, .debugName = debugName});
      };
      const uint8_t usageAttachment = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled;

      // room for the most layers the slider offers, so that moving it never reallocates the A-buffer
      aBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = getKeysSize(viewSize, kMaxOitLayers),
          .debugName = "Buffer: A-buffer",
      });
      tailTexture = createTarget(formatColor, usageAttachment, "Texture: A-buffer tail");
      texExact = createTarget(formatColor, usageAttachment, "Texture: A-buffer result");
      texApproximate = createTarget(formatColor, lvk::TextureUsageBits_Storage | lvk::TextureUsageBits_Sampled, "Texture: DFAOIT result");
      texSum = createTarget(formatColor, usageAttachment, "Texture: DFAOIT sum");
      texAccumulated = createTarget(formatColor, usageAttachment, "Texture: DFAOIT accumulated");
      texCount = createTarget(formatCount, usageAttachment, "Texture: DFAOIT count");
      nearestBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = getKeysSize(viewSize, numNearest),
          .debugName = "Buffer: DFAOIT nearest fragments",
      });
      slotsBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = sizeof(uint32_t) * viewSize,
          .debugName = "Buffer: DFAOIT tensor row of every pixel",
      });
      counterBuffer = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = sizeof(uint32_t),
          .debugName = "Buffer: DFAOIT compaction counter",
      });
      rowSums = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_HostVisible,
          .size = sizeof(RowStats) * dim.height,
          .debugName = "Buffer: error row statistics",
      });
    }

    const bool needApproximate = technique != Technique_ABuffer;
    const bool needExact = technique == Technique_ABuffer || compare;

    const PerFrame perFrame = {
        .proj = glm::perspective(float(45.0f * (M_PI / 180.0f)), views[0].aspectRatio, 0.5f, 500.0f),
        .view = app.camera_.getViewMatrix(),
        .model = glm::scale(mat4(1.0f), vec3(0.05f)),
    };

    const lvk::Dimensions groups16 = {(dim.width + 15) / 16, (dim.height + 15) / 16, 1};

    const MeshPushConstants pcMesh = {
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .materials = ctx->gpuAddress(sbMaterials),
        .aBuffer = ctx->gpuAddress(aBuffer),
        .nearest = ctx->gpuAddress(nearestBuffer),
        .sampler0 = sampler.index(),
        .layers = layers | (tailBlend ? 0u : 0x80000000u),
        .viewWidth = dim.width,
        .alpha = alpha,
        .weightDistance = weightDistance,
    };

    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();

    buf.cmdUpdateBuffer(bufPerFrame, perFrame);
    processLoadedMaterialTextures(buf, sbMaterials);

    gpuTimer.beginFrame(buf, frameIndex);
    auto timestamp = [&](GPUTimestamp t) { gpuTimer.timestamp(buf, t); };

    auto drawScene = [&](lvk::RenderPipelineHandle pipeline) {
      buf.cmdBindRenderPipeline(pipeline);
      buf.cmdBindVertexBuffer(0, vb0);
      buf.cmdBindIndexBuffer(ib0, lvk::IndexFormat_UI32);
      buf.cmdPushConstants(pcMesh);
      buf.cmdDrawIndexed(numIndices);
    };

    if (needExact) {
      buf.cmdPushDebugGroupLabel("A-buffer", 0xff0000ff);
      buf.cmdFillBuffer(aBuffer, 0, getKeysSize(viewSize, layers), 0xFFFFFFFF);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}}}},
          lvk::Framebuffer{.color = {{.texture = tailTexture}}},
          {.buffers = {aBuffer}});
      drawScene(pipelineABuffer);
      buf.cmdEndRendering();
      buf.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear,
                                                       .storeOp = lvk::StoreOp_Store,
                                                       .clearColor = {background.r, background.g, background.b, background.a}}}},
                            lvk::Framebuffer{.color = {{.texture = texExact}}},
                            {.sampledImages = {tailTexture}, .buffers = {aBuffer}, .buffersSrcStages = lvk::PipelineStageBits_Fragment});
      buf.cmdBindRenderPipeline(pipelineComposite);
      buf.cmdPushConstants(CompositePushConstants{
          .aBuffer = ctx->gpuAddress(aBuffer),
          .layers = layers,
          .viewWidth = dim.width,
          .tailTexture = tailTexture.index(),
      });
      buf.cmdDraw(3);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_ABuffer);

    if (technique == Technique_WeightedBlended) {
      buf.cmdPushDebugGroupLabel("Weighted blended: accumulate", 0xffff00ff);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {1, 1, 1, 1}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}}}},
          lvk::Framebuffer{.color = {{.texture = texSum}, {.texture = texAccumulated}, {.texture = texCount}}});
      drawScene(pipelineWeighted);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Features);
      timestamp(GPUTimestamp_Inference);

      buf.cmdPushDebugGroupLabel("Weighted blended: resolve", 0xffff00ff);
      buf.cmdBindComputePipeline(pipelineWeightedResolve);
      buf.cmdPushConstants(WeightedResolvePushConstants{
          .background = background,
          .texAccumulated = texSum.index(),
          .texRevealage = texAccumulated.index(),
          .imageOut = texApproximate.index(),
          .viewWidth = dim.width,
          .viewHeight = dim.height,
      });
      buf.cmdDispatch(groups16, {.sampledImages = {texSum, texAccumulated}, .storageImages = {texApproximate}});
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Resolve);
    } else if (needApproximate) {
      const DfaoitPushConstants pcDfaoit = {
          .background = background,
          .nearest = ctx->gpuAddress(nearestBuffer),
          .slots = ctx->gpuAddress(slotsBuffer),
          .counter = ctx->gpuAddress(counterBuffer),
          .tensorInput = ctx->gpuAddress(tensorInputBuffer),
          .tensorOutput = ctx->gpuAddress(tensorOutputBuffer),
          .texSum = texSum.index(),
          .texAccumulated = texAccumulated.index(),
          .texCount = texCount.index(),
          .imageOut = texApproximate.index(),
          .viewWidth = dim.width,
          .viewHeight = dim.height,
      };

      buf.cmdPushDebugGroupLabel("DFAOIT: features", 0xff00ff00);
      buf.cmdFillBuffer(nearestBuffer, 0, getKeysSize(viewSize, numNearest), 0xFFFFFFFF);
      buf.cmdBeginRendering(
          lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 1}},
                                    {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0, 0, 0, 0}}}},
          lvk::Framebuffer{.color = {{.texture = texSum}, {.texture = texAccumulated}, {.texture = texCount}}},
          {.buffers = {nearestBuffer}});
      drawScene(pipelineFeatures);
      buf.cmdEndRendering();
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Features);

      buf.cmdPushDebugGroupLabel("DFAOIT: data graph", 0xff00ff00);
      buf.cmdFillBuffer(counterBuffer, 0, sizeof(uint32_t), 0);
      buf.cmdBindComputePipeline(pipelinePack);
      buf.cmdPushConstants(pcDfaoit);
      buf.cmdDispatch(groups16,
                      {
                          .sampledImages = {texSum, texAccumulated, texCount},
                          .buffers = {nearestBuffer, slotsBuffer, counterBuffer},
                          .buffersSrcStages = lvk::PipelineStageBits_Fragment | lvk::PipelineStageBits_Transfer,
                          .tensors = {tensorInput},
                      });
      const lvk::TensorHandle graphInputs[] = {tensorInput};
      const lvk::TensorHandle graphOutputs[] = {tensorOutput};
      buf.cmdDispatchDataGraph(pipelineGraph, graphInputs, graphOutputs);
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Inference);

      buf.cmdPushDebugGroupLabel("DFAOIT: resolve", 0xff00ff00);
      buf.cmdBindComputePipeline(pipelineResolve);
      buf.cmdPushConstants(pcDfaoit);
      buf.cmdDispatch(groups16,
                      {
                          .sampledImages = {texSum, texAccumulated, texCount},
                          .storageImages = {texApproximate},
                          .buffers = {nearestBuffer, slotsBuffer},
                          .tensors = {tensorOutput},
                      });
      buf.cmdPopDebugGroupLabel();

      timestamp(GPUTimestamp_Resolve);
    } else {
      timestamp(GPUTimestamp_Features);
      timestamp(GPUTimestamp_Inference);
      timestamp(GPUTimestamp_Resolve);
    }

    const bool measureError = compare && needApproximate;

    if (measureError) {
      buf.cmdBindComputePipeline(pipelineError);
      buf.cmdPushConstants(ErrorPushConstants{
          .rows = ctx->gpuAddress(rowSums),
          .texResult = texApproximate.index(),
          .texReference = texExact.index(),
          .texCount = texCount.index(),
          .layers = layers,
          .viewWidth = dim.width,
          .viewHeight = dim.height,
      });
      buf.cmdDispatch({1, (dim.height + 63) / 64, 1}, {.sampledImages = {texApproximate, texExact, texCount}, .buffers = {rowSums}});
    }

    const lvk::TextureHandle texResult = needApproximate ? texApproximate : texExact;
    const lvk::Framebuffer framebuffer = {.color = {{.texture = views[0].colorTexture}}};

    buf.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_DontCare, .storeOp = lvk::StoreOp_Store}}},
                          framebuffer,
                          {.sampledImages = {texResult, measureError ? lvk::TextureHandle(texExact) : texResult}});
    buf.cmdBindRenderPipeline(pipelinePresent);
    buf.cmdPushConstants(PresentPushConstants{
        .clipRotation = getClipRotation(ctx),
        .texResult = texResult.index(),
        .texReference = texExact.index(),
        .sampler0 = samplerClamp.index(),
        .showDifference = measureError && showDifference ? 1u : 0u,
        .gain = differenceGain,
    });
    buf.cmdDraw(3);

    app.imgui_->beginFrame(framebuffer);
    drawMaterialTexturesLoadingProgress();
    ImGui::Begin("Order-independent transparency", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::SeparatorText("Technique");
    ImGui::RadioButton("A-buffer", &technique, Technique_ABuffer);
    ImGui::RadioButton("Weighted blended", &technique, Technique_WeightedBlended);
    ImGui::BeginDisabled(!graphReady);
    ImGui::RadioButton(graphReady ? "DFAOIT" : "DFAOIT (the data graph is not available)", &technique, Technique_DFAOIT);
    ImGui::EndDisabled();
    ImGui::SliderFloat("Alpha", &alpha, 0.0f, 1.0f);
    if (technique == Technique_WeightedBlended) {
      ImGui::SeparatorText("Weighted blended");
      ImGui::SliderFloat("Weight distance", &weightDistance, 1.0f, 500.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::SeparatorText("A-buffer");
    ImGui::SliderInt("Layers", &numLayers, 1, kMaxOitLayers);
    ImGui::Checkbox("Tail blending", &tailBlend);
    ImGui::Text("A-buffer: %.1f MB", double(getKeysSize(viewSize, kMaxOitLayers)) / (1024.0 * 1024.0));
    ImGui::SeparatorText("Quality");
    ImGui::Checkbox("Compare with the A-buffer", &compare);
    if (measureError) {
      ImGui::Checkbox("Show the difference", &showDifference);
      ImGui::SliderFloat("Gain", &differenceGain, 1.0f, 64.0f);
      ImGui::Text("MSE: %.6f", meanSquaredError);
      ImGui::Text("Fragments per pixel: %.1f mean, %.0f max", meanFragments, maxFragments);
      ImGui::Text("Above %u layers: %.1f%% of pixels", layers, percentAboveLayers);
    }
    ImGui::End();
    const GPUStageTimer::Row rows[] = {
        {"Frame", GPUTimestamp_Begin, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)},
        {"A-buffer", GPUTimestamp_ABuffer, ImVec4(0.4f, 0.7f, 1.0f, 1.0f)},
        {technique == Technique_WeightedBlended ? "Geometry" : "Features", GPUTimestamp_Features, ImVec4(1.0f, 0.85f, 0.3f, 1.0f)},
        {"Inference", GPUTimestamp_Inference, ImVec4(1.0f, 0.5f, 0.2f, 1.0f)},
        {"Resolve", GPUTimestamp_Resolve, ImVec4(0.4f, 1.0f, 0.5f, 1.0f)},
        {"Present", GPUTimestamp_Present, ImVec4(0.75f, 0.6f, 1.0f, 1.0f)},
    };
    gpuTimer.drawWindow(rows);
    app.drawFPS();
    app.imgui_->endFrame(buf);

    buf.cmdEndRendering();

    timestamp(GPUTimestamp_Present);

    const lvk::SubmitHandle submitHandle = ctx->submit(buf, views[0].colorTexture);

    if (measureError) {
      ctx->wait(submitHandle);
      ctx->invalidateMappedMemory(rowSums, 0, sizeof(RowStats) * dim.height);
      const RowStats* rows = (const RowStats*)ctx->getMappedPtr(rowSums);
      double squaredError = 0.0;
      double numFragments = 0.0;
      double numAbove = 0.0;
      maxFragments = 0.0;
      for (uint32_t y = 0; y != dim.height; y++) {
        squaredError += rows[y].squaredError;
        numFragments += rows[y].numFragments;
        numAbove += rows[y].numPixelsAboveLayers;
        maxFragments = std::max(maxFragments, (double)rows[y].maxFragments);
      }
      meanSquaredError = squaredError / (3.0 * viewSize);
      meanFragments = numFragments / viewSize;
      percentAboveLayers = 100.0 * numAbove / viewSize;
      if (frameIndex % 100 == 0) {
        LLOGL("MSE %.8f | fragments per pixel: mean %.1f, max %.0f | above %u layers: %.1f%%\n",
              meanSquaredError,
              meanFragments,
              maxFragments,
              layers,
              percentAboveLayers);
      }
    }

    gpuTimer.endFrame(ctx);
    if (measureTime && gpuTimer.isMeasuring && frameIndex % 100 == 0) {
      const double* stageTime = gpuTimer.stageTime;
      LLOGL("GPU time, ms: frame %.2f | A-buffer %.2f, %s %.2f, inference %.2f, resolve %.2f, present %.2f | %.1f FPS\n",
            gpuTimer.frameTime,
            stageTime[GPUTimestamp_ABuffer],
            technique == Technique_WeightedBlended ? "geometry" : "features",
            stageTime[GPUTimestamp_Features],
            stageTime[GPUTimestamp_Inference],
            stageTime[GPUTimestamp_Resolve],
            stageTime[GPUTimestamp_Present],
            app.fpsCounter_.getFPS());
    }

    frameIndex++;
  });

  ctx->wait({});
  cancelLoadingMaterialTextures();

  VULKAN_APP_EXIT();
}
