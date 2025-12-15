/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 Indirect diffuse light by a Neural Light Grid (Iwanicki et al., "Neural Light Grid", Advances in Real-Time Rendering
 in Games, SIGGRAPH 2024).

 Lit with the specular color, gloss and normal maps the MTL names (GGX sun, the indirect light around the reflection
 for the specular part). Every probe gets a weighting function over 8192 support points; the generator (`nlg.bin`: its int8 layers as a
 `VK_ARM_data_graph` pipeline built at start-up, its float layers as f16 `VK_KHR_cooperative_matrix` tiles accumulated in f32, or its
 float layers in a compute shader; the UI switches between them, each grayed out without what it runs on, and bakes the probes again
 with the new one) turns it into a 3-8-8-1 network. Probe lighting is integrated over 256 support points whenever the sun moves. Probes form
 `--probe-levels` lattices, each half the spacing of the last, the finer ones only near geometry; a pixel blends the 8 probes of its cell on
 the finest level that weights it. The whole bake runs in the sample.

 `--scene exterior|interior|sponza`, `--probe-spacing S`, `--probe-levels N`, `--sun-elevation E`, `--sun-azimuth A`,
 `--no-gi` (G toggles), `--textures` (T toggles), `--no-auto-exposure`,
 `--timings`, `--rebake`, `--no-ui`, `--generator compute|data-graph|cm` (by default the data graph, else the
 compute shader); it renders at the size of the monitor, or `--width W --height H`.

 The baked probes are saved to `nlg_<scene>.bake` in the content folder and loaded on the next run with the same lattice,
 generator and sun; only the lighting is computed then. Textures are off at first and stream in the first time they are enabled:
 until then the scene is baked untextured (MTL colors, no alpha testing, `nlg_<scene>_untextured.bake`), and once they are
 in it is baked again with them, the untextured probes lighting the scene until the textured ones replace them; the
 lighting averages into the untextured lighting rather than starting over.
*/

#include <algorithm>
#include <cstring>
#include <string>
#include <thread>

#include <lmath/Random.h>

#include "Bistro.h"

enum {
  kMaxProbesPerAxis = 96,
  kMaxProbeLevels = 4,
  kMaxLatticePoints = 1 << 26,
  kSupportPoints = 8192,
  kSupportNeighbours = 32,
  kBakePoints = 256,
  kBakeRays = 16,
  kBakeBatch = 256,
  kNlgBudget = 1024, // probes lit in the first frame; then as many as fit into kNlgTimeBoxUs of GPU time
  kNlgBudgetMin = 64,
  kNlgBudgetMax = 65535, // the guaranteed maxComputeWorkGroupCount[0]
  kNlgTimeBoxUs = 4000,
  kNlgTimingSlots = 4, // the GPU time of the lighting is read back this many frames later, so it never stalls
  // probe flags, written by the bake (the number of lighting updates of a probe is in `.w` of its SH coefficient kShUpdates)
  kProbeHasLighting = 1,
  kProbeInside = 2,
  kProbeNotBaked = 4, // until the bake overwrites the flags
  kShUpdates = 1,
  kNlgConvergePasses = 4,
  kNlgWarmupPasses = 2,
  kPcaComponents = 512,
  kGeneratorWidth = 512,
  kGeneratorLayers = 6,
  kGeneratorProbes = 4,
  kHidden = 8,
  kCoefficients = 3 * kHidden + kHidden + kHidden * kHidden + kHidden + kHidden + 1,
  kNetworkVec4s = (kCoefficients + 7) / 8,
  kFunctionThreads = 256,
  kGeneratorThreads = kGeneratorWidth / 2,
  kGeneratorWeights =
      kGeneratorLayers * (kGeneratorWidth * kGeneratorWidth + kGeneratorWidth) + kGeneratorWidth * kCoefficients + kCoefficients,
  kDiffusionPasses = 2,
  kLightingClusters = 4,
  kGeneratorMagic = 0x47474C4E,
  kGeneratorVersion = 3, // version 2 has no integer graph
  kGraphLayers = kGeneratorLayers + 2, // the PCA, the float layers, the output layer
  kGraphTableSize = 256,
  kGeneratorInputZeroPoint = -128,
  kCooperativeOutputs = 128, // kCoefficients padded to whole tiles
  kCooperativeTilesPerGroup = 2,
  kCooperativeBlock = 2, // a subgroup multiplies 2x2 tiles, loading every A and B tile once for two of them
  // the sums over k are split into slices summed by a second pass, which also finishes the layer
  kCooperativePcaSplits = 8,
  kCooperativeSplits = 4,
  // the layers of the cooperative matrix generator
  kCooperativePca = 0,
  kCooperativeFirst,
  kCooperativeSkip,
  kCooperativeOutput,
};

enum GeneratorPath {
  GeneratorPath_Compute = 0,
  GeneratorPath_DataGraph,
  GeneratorPath_Cooperative,
};
static_assert(kHidden == 8 && kNetworkVec4s == 15);
constexpr float kDiffusionSigma = 0.15f;
constexpr float kLookupOffset = 0.2f;
constexpr float kLevelCoverage = 0.5f;
constexpr float kAmbientSky = 0.5f;
constexpr float kCameraNear = 0.1f;
constexpr float kCameraFar = 500.0f;
constexpr float kCameraFov = 60.0f;
constexpr float kSceneScale = 0.05f;
constexpr float kExposureTau = 0.3f;
constexpr float kExposureTauFast = 0.05f;
constexpr float kProbeBackfaceLimit = 0.25f;
constexpr float kFreeSpaceExtent = 1.001f;

const vec3 kSunColor = vec3(1.0f, 0.95f, 0.85f) * 10.0f;
const vec3 kSkyColor = vec3(0.5f, 0.7f, 1.0f) * 1.5f;

struct Scene {
  const char* name;
  const char* modelFile;
  const char* textureDir;
  const char* cacheFile;
  vec3 cameraPos;
  vec3 cameraTarget;
};

const Scene kScenes[] = {
    {"sponza", "src/Sponza/sponza.obj", "src/Sponza/", "cache3.data", vec3(-25, 10, -1), vec3(10, 10, 0)},
    {"exterior", "src/bistro/Exterior/exterior.obj", "src/bistro/Exterior/", "cache.data", vec3(-100, 40, -47), vec3(0, 35, 0)},
    {"interior", "src/bistro/Interior/interior.obj", "src/bistro/Interior/", "cache_interior.data", vec3(-25, 10, -1), vec3(10, 10, 0)},
};

#define PER_FRAME_GLSL                                            \
  "layout(std430, buffer_reference) readonly buffer PerFrame {\n" \
  "  mat4 proj;\n"                                                \
  "  mat4 view;\n"                                                \
  "  vec4 cameraPos;\n"                                           \
  "  vec4 sunDir;\n"                                              \
  "  vec4 sunColor;\n"                                            \
  "  vec4 skyColor;\n"                                            \
  "};\n"

const char* kCodeMaterialModels = R"(
const uint kModelMetal = 1u;
const uint kModelCloth = 2u;
const uint kModelFoliage = 3u;
const float kModelGloss[4] = float[](0.25, 0.6, 0.15, 0.4);
const float kDielectricF0 = 0.04;
const float kFoliageTransmission = 0.4;
const float kFoliageSpecular = 0.25;
const float kSheenRoughness = 0.5;
)";

const char* kCodeGBufferVS = BISTRO_GLSL PER_FRAME_GLSL R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

layout(push_constant) uniform constants {
  PerFrame perFrame;
  uvec2 materials;
  uvec2 models;
  uint sampler0;
} pc;

layout (location=0) out vec3 v_worldPos;
layout (location=1) out vec3 v_normal;
layout (location=2) out vec2 v_uv;
layout (location=3) flat out uint v_mtlIndex;

void main() {
  v_worldPos = pos * kSceneScale;
  gl_Position = pc.perFrame.proj * pc.perFrame.view * vec4(v_worldPos, 1.0);
  v_normal = unpackOctahedral16(normal);
  v_uv = uv;
  v_mtlIndex = mtlIndex;
}
)";

const char* kCodeGBufferFS = R"(
#extension GL_EXT_buffer_reference : require

layout (location=0) in vec3 v_worldPos;
layout (location=1) in vec3 v_normal;
layout (location=2) in vec2 v_uv;
layout (location=3) flat in uint v_mtlIndex;

layout (location=0) out vec4 out_Position;
layout (location=1) out vec4 out_Normal;
layout (location=2) out vec4 out_Albedo;
layout (location=3) out vec4 out_Shading;
layout (location=4) out vec4 out_Material;

)" BISTRO_GLSL PER_FRAME_GLSL R"(
layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(std430, buffer_reference) readonly buffer Models {
  uint m[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  Materials materials;
  Models models;
  uint sampler0;
} pc;

float perceivedBrightness(vec3 c) {
  return sqrt(0.299 * c.r * c.r + 0.587 * c.g * c.g + 0.114 * c.b * c.b);
}

float solveMetallic(float diffuse, float specular, float oneMinusSpecular) {
  if (specular < kDielectricF0) {
    return 0.0;
  }
  float b = diffuse * oneMinusSpecular / (1.0 - kDielectricF0) + specular - 2.0 * kDielectricF0;
  float c = kDielectricF0 - specular;
  return clamp((-b + sqrt(max(b * b - 4.0 * kDielectricF0 * c, 0.0))) / (2.0 * kDielectricF0), 0.0, 1.0);
}

vec3 perturbNormal(vec3 n, vec3 p, vec2 uv, vec3 tn) {
  vec3 dp1 = dFdx(p);
  vec3 dp2 = dFdy(p);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);
  vec3 dp2perp = cross(dp2, n);
  vec3 dp1perp = cross(n, dp1);
  vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
  float len = max(dot(t, t), dot(b, b));
  if (len < 1e-20) {
    return n;
  }
  float invMax = inversesqrt(len);
  return normalize(mat3(t * invMax, b * invMax, n) * tn);
}

void main() {
  Material m = pc.materials.mtl[v_mtlIndex];
  if (m.texAlpha > 0 && textureBindless2D(m.texAlpha, pc.sampler0, v_uv).r < 0.5) {
    discard;
  }
  vec4 albedo = m.texDiffuse > 0 ? textureBindless2D(m.texDiffuse, pc.sampler0, v_uv) : vec4(pow(m.diffuse.rgb, vec3(1.0 / 2.2)), 1.0);
  if (albedo.a < 0.5) {
    discard;
  }
  vec3 n = normalize(v_normal);
  bool backFacing = dot(n, pc.perFrame.cameraPos.xyz - v_worldPos) < 0.0;
  out_Position = vec4(v_worldPos, 1.0);
  vec3 nf = backFacing ? -n : n;
  out_Normal = vec4(nf, 0.0);
  uint model = pc.models.m[v_mtlIndex];
  vec3 ns = nf;
  if (m.texNormal > 0) {
    vec3 tn = textureBindless2D(m.texNormal, pc.sampler0, v_uv).xyz * 2.0 - 1.0;
    ns = perturbNormal(nf, v_worldPos, v_uv, tn);
  }
  vec3 v = normalize(pc.perFrame.cameraPos.xyz - v_worldPos);
  float nDotV = dot(ns, v);
  if (nDotV < 0.1) {
    ns = normalize(ns + v * (0.1 - nDotV));
  }
  vec3 base = pow(albedo.rgb, vec3(2.2));
  float gloss = kModelGloss[model];
  float metallic = model == kModelMetal ? 1.0 : 0.0;
  if (m.texSpecular > 0) {
    vec4 spec = textureBindless2D(m.texSpecular, pc.sampler0, v_uv);
    gloss = spec.a;
    if (model == kModelMetal) {
      vec3 s = pow(spec.rgb, vec3(2.2));
      float oneMinusSpecular = 1.0 - max(s.r, max(s.g, s.b));
      metallic = solveMetallic(perceivedBrightness(base), perceivedBrightness(s), oneMinusSpecular);
      vec3 fromDiffuse = base * oneMinusSpecular / (1.0 - kDielectricF0) / max(1.0 - metallic, 1e-4);
      vec3 fromSpecular = (s - kDielectricF0 * (1.0 - metallic)) / max(metallic, 1e-4);
      base = clamp(mix(fromDiffuse, fromSpecular, metallic * metallic), 0.0, 1.0);
    }
  }
  out_Albedo = vec4(pow(base, vec3(1.0 / 2.2)), 1.0);
  out_Shading = vec4(ns, clamp(1.0 - gloss, 0.05, 1.0));
  out_Material = vec4(metallic, float(model) / 255.0, 0.0, 1.0);
}
)";

const char* kCodeSceneDeclarations = R"(
#extension GL_EXT_ray_query : require

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler kSamplers[];
layout (set = 0, binding = 2, rgba16f) uniform writeonly image2D kImagesRGBA16F[];
layout (set = 0, binding = 2, r16f) uniform writeonly image2D kImagesR16F[];
layout (set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

)" BISTRO_GLSL PER_FRAME_GLSL R"(
struct Vertex {
  float x, y, z;
  uint uv;
  uint normalAndMaterial;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(std430, buffer_reference) readonly buffer Vertices {
  Vertex vtx[];
};

layout(std430, buffer_reference) readonly buffer Indices {
  uint idx[];
};

layout(std430, buffer_reference) readonly buffer Floats {
  float v[];
};

layout(std430, buffer_reference) buffer FloatsRW {
  float v[];
};

layout(std430, buffer_reference) readonly buffer Uints {
  uint v[];
};

layout(std430, buffer_reference) buffer UintsRW {
  uint v[];
};

layout(std430, buffer_reference) readonly buffer Vec4s {
  vec4 v[];
};

layout(std430, buffer_reference) buffer Vec4sRW {
  vec4 v[];
};

layout(std430, buffer_reference) buffer UVec4sRW {
  uvec4 v[];
};

layout(std430, buffer_reference) buffer Nlg {
  Vec4s points;
  Uints neighbours;
  Uints bakePoints;
  UintsRW support;
  FloatsRW phiA;
  FloatsRW phiB;
  UintsRW histogram;
  Vec4sRW scratch;
  UintsRW labels;
  Vec4sRW bakeLight;
  UintsRW shares;
  FloatsRW coefficients;
  Vec4sRW sh;
  UintsRW flags;
  Floats pcaMean;
  Uints pcaBasis;
  Floats weights;
  UintsRW graphInput;
  Uints graphOutput;
  Floats outputScale;
  UVec4sRW networks;
  Uints bakeOrder;
};
)";

const char* kCodeSceneFunctions = R"(
const float kRayOffset = 0.01;
const float kInvPi = 0.31830988618;

uint tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;
  for (uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }
  return v0;
}

float rnd(inout uint prev) {
  prev = 1664525u * prev + 1013904223u;
  return float(prev & 0x00FFFFFF) / float(0x01000000);
}

vec4 randomRotation(inout uint seed) {
  float u1 = rnd(seed);
  float u2 = 2.0 * 3.14159265 * rnd(seed);
  float u3 = 2.0 * 3.14159265 * rnd(seed);
  return vec4(sqrt(1.0 - u1) * sin(u2), sqrt(1.0 - u1) * cos(u2), sqrt(u1) * sin(u3), sqrt(u1) * cos(u3));
}

vec3 rotate(vec4 q, vec3 v) {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

vec3 fibonacciDirection(uint i, uint count) {
  float k = float(i) + 0.5;
  float phi = 2.0 * 3.14159265 * fract(k * 0.61803398875);
  float z = 1.0 - 2.0 * k / float(count);
  float r = sqrt(max(0.0, 1.0 - z * z));
  return vec3(r * cos(phi), r * sin(phi), z);
}

vec4 sampleTexture(uint id, vec2 uv) {
  return textureLod(nonuniformEXT(sampler2D(kTextures2D[id], kSamplers[pc.sampler0])), uv, 0.0);
}

void triangleAt(uint triangle, vec2 bc, out vec2 uv, out vec3 normal, out uint material) {
  uint i0 = pc.indices.idx[3 * triangle + 0];
  uint i1 = pc.indices.idx[3 * triangle + 1];
  uint i2 = pc.indices.idx[3 * triangle + 2];
  vec3 w = vec3(1.0 - bc.x - bc.y, bc.x, bc.y);
  uv = unpackHalf2x16(pc.vertices.vtx[i0].uv) * w.x + unpackHalf2x16(pc.vertices.vtx[i1].uv) * w.y +
       unpackHalf2x16(pc.vertices.vtx[i2].uv) * w.z;
  normal = unpackOctahedral16(pc.vertices.vtx[i0].normalAndMaterial & 0xFFFFu) * w.x +
           unpackOctahedral16(pc.vertices.vtx[i1].normalAndMaterial & 0xFFFFu) * w.y +
           unpackOctahedral16(pc.vertices.vtx[i2].normalAndMaterial & 0xFFFFu) * w.z;
  material = pc.vertices.vtx[i0].normalAndMaterial >> 16;
}

bool isCandidateOpaque(rayQueryEXT rq) {
  uint triangle = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) + rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
  vec2 uv;
  vec3 normal;
  uint material;
  triangleAt(triangle, rayQueryGetIntersectionBarycentricsEXT(rq, false), uv, normal, material);
  Material m = pc.materials.mtl[material];
  if (m.texAlpha > 0 && sampleTexture(m.texAlpha, uv).r < 0.5) {
    return false;
  }
  return m.texDiffuse == 0 || sampleTexture(m.texDiffuse, uv).a >= 0.5;
}

// the closest hit that is not alpha-tested away
bool traceOpaque(vec3 origin, vec3 dir, float tMax, uint rayFlags, out uint triangle, out vec2 bc, out float t) {
  rayQueryEXT rq;
  rayQueryInitializeEXT(rq, kTLAS[pc.tlas], rayFlags, 0xFF, origin, 0.0, dir, tMax);
  while (rayQueryProceedEXT(rq)) {
    if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT && isCandidateOpaque(rq)) {
      rayQueryConfirmIntersectionEXT(rq);
    }
  }
  if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
    return false;
  }
  triangle = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) + rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
  bc = rayQueryGetIntersectionBarycentricsEXT(rq, true);
  t = rayQueryGetIntersectionTEXT(rq, true);
  return true;
}

bool isOccluded(vec3 origin, vec3 dir, float tMax) {
  uint triangle;
  vec2 bc;
  float t;
  return traceOpaque(origin, dir, tMax, gl_RayFlagsTerminateOnFirstHitEXT, triangle, bc, t);
}

bool traceFacing(vec3 origin, vec3 dir, float tMax, out bool backFacing) {
  backFacing = false;
  uint triangle;
  vec2 bc;
  float t;
  if (!traceOpaque(origin, dir, tMax, gl_RayFlagsNoneEXT, triangle, bc, t)) {
    return false;
  }
  vec2 uv;
  vec3 normal;
  uint material;
  triangleAt(triangle, bc, uv, normal, material);
  backFacing = dot(normal, dir) > 0.0;
  return true;
}

bool traceClosest(vec3 origin, vec3 dir, out vec3 position, out vec3 normal, out vec3 albedo, out bool backFacing) {
  uint triangle;
  vec2 bc;
  float t;
  if (!traceOpaque(origin, dir, 10000.0, gl_RayFlagsNoneEXT, triangle, bc, t)) {
    return false;
  }
  vec2 uv;
  uint material;
  triangleAt(triangle, bc, uv, normal, material);
  normal = normalize(normal);
  backFacing = dot(normal, dir) > 0.0;
  if (backFacing) {
    normal = -normal;
  }
  position = origin + dir * t;
  Material m = pc.materials.mtl[material];
  albedo = m.texDiffuse > 0 ? pow(sampleTexture(m.texDiffuse, uv).rgb, vec3(2.2)) : m.diffuse.rgb;
  return true;
}
)";

const char* kCodeProbeFunctions = R"(
const float kCosineLobe[9] = float[9](3.14159265, 2.09439510, 2.09439510, 2.09439510, 0.78539816, 0.78539816, 0.78539816, 0.78539816, 0.78539816);
const uint kNoProbe = 0xFFFFFFFFu;

void shBasis(vec3 d, out float y[9]) {
  y[0] = 0.282095;
  y[1] = 0.488603 * d.y;
  y[2] = 0.488603 * d.z;
  y[3] = 0.488603 * d.x;
  y[4] = 1.092548 * d.x * d.y;
  y[5] = 1.092548 * d.y * d.z;
  y[6] = 0.315392 * (3.0 * d.z * d.z - 1.0);
  y[7] = 1.092548 * d.x * d.z;
  y[8] = 0.546274 * (d.x * d.x - d.y * d.y);
}

uint probeIndex(ivec3 c, uint level) {
  uvec3 dims = kLevelDims[level];
  return pc.slots.v[kLevelSlots[level] + uint(c.x) + dims.x * (uint(c.y) + dims.y * uint(c.z))];
}

uint probeLevel(uint probe) {
  return pc.cells.v[probe] >> 30;
}

ivec3 probeCellOf(uint probe) {
  uint packed = pc.cells.v[probe];
  uvec3 dims = kLevelDims[packed >> 30];
  uint linear = packed & 0x3FFFFFFFu;
  return ivec3(linear % dims.x, (linear / dims.x) % dims.y, linear / (dims.x * dims.y));
}

vec3 probePosition(ivec3 c, uint level) {
  return kLevelOrigin[level] + vec3(c) * kLevelSpacing[level];
}

void probeCell(vec3 p, uint level, out ivec3 i0, out vec3 f) {
  vec3 u = (p - kLevelOrigin[level]) / kLevelSpacing[level];
  vec3 cell = clamp(floor(u), vec3(0.0), vec3(kLevelDims[level]) - 2.0);
  i0 = ivec3(cell);
  f = clamp(u - cell, vec3(0.0), vec3(1.0));
}

float trilinearWeight(ivec3 d, vec3 f) {
  return (d.x != 0 ? f.x : 1.0 - f.x) * (d.y != 0 ? f.y : 1.0 - f.y) * (d.z != 0 ? f.z : 1.0 - f.z);
}
)";

const char* kCodeNlgFunctions = R"(
void unpackHalf8(uvec4 v, out float f[8]) {
  for (uint i = 0; i != 4u; i++) {
    vec2 p = unpackHalf2x16(v[i]);
    f[2u * i] = p.x;
    f[2u * i + 1u] = p.y;
  }
}

float nlgWeight(uint probe, vec3 x) {
  uint c = probe * kNetworkVec4s;
  float w1[24];
  float row[8];
  for (uint r = 0; r != 3u; r++) {
    unpackHalf8(pc.nlg.networks.v[c + r], row);
    for (uint i = 0; i != 8u; i++) {
      w1[8u * r + i] = row[i];
    }
  }
  float b[8];
  unpackHalf8(pc.nlg.networks.v[c + 3u], b);
  float h1[8];
  for (uint j = 0; j != 8u; j++) {
    float a = w1[3u * j] * x.x + w1[3u * j + 1u] * x.y + w1[3u * j + 2u] * x.z + b[j];
    h1[j] = a > 0.0 ? a : 0.01 * a;
  }
  unpackHalf8(pc.nlg.networks.v[c + 12u], b);
  float h2[8];
  for (uint k = 0; k != 8u; k++) {
    unpackHalf8(pc.nlg.networks.v[c + 4u + k], row);
    float a = b[k];
    for (uint j = 0; j != 8u; j++) {
      a += row[j] * h1[j];
    }
    h2[k] = a > 0.0 ? a : 0.01 * a;
  }
  unpackHalf8(pc.nlg.networks.v[c + 13u], row);
  float w = unpackHalf2x16(pc.nlg.networks.v[c + 14u].x).x;
  for (uint k = 0; k != 8u; k++) {
    w += row[k] * h2[k];
  }
  return w;
}

// `outPending` is the share of the finest level's trilinear weight on probes that are not baked or lit yet
vec3 nlgIrradiance2(vec3 surface, vec3 n, vec3 ns, vec3 r, out vec3 resultR, out float outPending) {
  float y[9];
  shBasis(ns, y);
  float yr[9];
  shBasis(r, yr);
  vec3 result = vec3(0.0);
  resultR = vec3(0.0);
  float remaining = 1.0;
  vec4 fallback = vec4(0.0);
  vec3 fallbackR = vec3(0.0);
  float pending = 0.0;
  for (int level = int(kProbeLevels) - 1; level >= 0 && remaining > 1e-3; level--) {
    float spacing = kLevelSpacing[level];
    vec3 pos = surface + n * (kLookupOffset * spacing);
    ivec3 i0;
    vec3 f;
    probeCell(pos, uint(level), i0, f);
    vec4 sum = vec4(0.0);
    vec3 sumR = vec3(0.0);
    for (uint corner = 0; corner != 8; corner++) {
      ivec3 d = ivec3(corner & 1u, (corner >> 1) & 1u, (corner >> 2) & 1u);
      uint probe = probeIndex(i0 + d, uint(level));
      if (probe == kNoProbe) {
        continue;
      }
      uint flags = pc.nlg.flags.v[probe];
      bool lit = pc.nlg.sh.v[9u * probe + kShUpdates].w > 0.0;
      bool isPending = !lit && (flags & (kProbeHasLighting | kProbeNotBaked)) != 0u;
      if (level == 0 && isPending) {
        pending += trilinearWeight(d, f);
      }
      if ((flags & kProbeHasLighting) == 0u || !lit) {
        continue;
      }
      vec3 local = (pos - probePosition(i0 + d, uint(level))) / spacing;
      if (any(greaterThan(abs(local), vec3(1.0)))) {
        continue;
      }
      vec3 e = vec3(0.0);
      vec3 er = vec3(0.0);
      for (uint i = 0; i != 9; i++) {
        vec3 c = pc.nlg.sh.v[9u * probe + i].rgb;
        e += kCosineLobe[i] * y[i] * c;
        er += kCosineLobe[i] * yr[i] * c;
      }
      e = max(e, vec3(0.0));
      er = max(er, vec3(0.0));
      if (level == 0) {
        float tri = trilinearWeight(d, f);
        fallback += vec4(e * tri, tri);
        fallbackR += er * tri;
      }
      float w = max(nlgWeight(probe, local), 0.0);
      if (w > 0.0) {
        sum += vec4(e * w, w);
        sumR += er * w;
      }
    }
    if (sum.w > 1e-6) {
      float t = level == 0 ? 1.0 : clamp(sum.w / kLevelCoverage, 0.0, 1.0);
      float k = remaining * t / sum.w;
      result += k * sum.rgb;
      resultR += k * sumR;
      remaining *= 1.0 - t;
    }
  }
  if (remaining > 1e-3 && fallback.w > 1e-6) {
    result += remaining * fallback.rgb / fallback.w;
    resultR += remaining * fallbackR / fallback.w;
  }
  outPending = pending > 0.0 ? pending / (pending + fallback.w) : 0.0;
  return result;
}

vec3 nlgIrradiance(vec3 surface, vec3 n) {
  vec3 unusedR;
  float unusedPending;
  return nlgIrradiance2(surface, n, n, n, unusedR, unusedPending);
}
)";

const char* kCodeDirectCS = R"(
layout (local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform constants {
  PerFrame perFrame;
  Materials materials;
  Vertices vertices;
  Indices indices;
  uint tlas;
  uint sampler0;
  uint texPosition;
  uint texNormal;
  uint imageDirect;
  uint texMaterial;
} pc;
)";

const char* kCodeDirectMainCS = R"(
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= kWidth || p.y >= kHeight) {
    return;
  }
  vec4 posWS = texelFetch(kTextures2D[pc.texPosition], ivec2(p), 0);
  float direct = 0.0;
  if (posWS.w > 0.0) {
    vec3 n = normalize(texelFetch(kTextures2D[pc.texNormal], ivec2(p), 0).xyz);
    float cosSun = dot(n, pc.perFrame.sunDir.xyz);
    uint model = uint(texelFetch(kTextures2D[pc.texMaterial], ivec2(p), 0).g * 255.0 + 0.5);
    float side = cosSun > 0.0 ? 1.0 : model == kModelFoliage ? -1.0 : 0.0;
    if (side != 0.0 && !isOccluded(posWS.xyz + side * n * kRayOffset, pc.perFrame.sunDir.xyz, 10000.0)) {
      direct = 1.0;
    }
  }
  imageStore(kImagesR16F[pc.imageDirect], ivec2(p), vec4(direct));
}
)";

const char* kCodeNlgPushConstants = R"(
layout(push_constant) uniform constants {
  PerFrame perFrame;
  Materials materials;
  Vertices vertices;
  Indices indices;
  Uints slots;
  Uints cells;
  Nlg nlg;
  uint tlas;
  uint sampler0;
  uint frameId;
  uint first;
  uint count;
  uint mode;
  float blend;
} pc;
)";

const char* kCodeSupportCS = R"(
layout (local_size_x = 64) in;

void main() {
  uint point = gl_GlobalInvocationID.x;
  uint local = gl_GlobalInvocationID.y;
  if (point >= kSupportPoints || local >= pc.count) {
    return;
  }
  uint probe = pc.nlg.bakeOrder.v[pc.first + local];
  uint level = probeLevel(probe);
  float spacing = kLevelSpacing[level];
  vec3 center = probePosition(probeCellOf(probe), level);
  vec3 offset = pc.nlg.points.v[point].xyz * spacing;
  vec3 p = center + offset;
  uint bits = 0u;
  uint flags = 0u;
  bool backfaces = false;
  float dist = length(offset);
  bool back;
  if (dist <= 1e-6) {
    flags |= 1u;
  } else if (!traceFacing(center, offset / dist, dist, back)) {
    flags |= 1u;
  } else if (back) {
    flags |= 4u;
  }
  for (uint k = 0; k != kSupportNeighbours; k++) {
    uint j = pc.nlg.neighbours.v[point * kSupportNeighbours + k];
    vec3 d = pc.nlg.points.v[j].xyz * spacing - offset;
    float len = max(length(d), 1e-6);
    if (!traceFacing(p, d / len, len, back)) {
      bits |= 1u << k;
    } else if (back) {
      backfaces = true;
    }
  }
  if (backfaces) {
    flags |= 2u;
  }
  uint at = (local * kSupportPoints + point) * 2u;
  pc.nlg.support.v[at] = bits;
  pc.nlg.support.v[at + 1u] = flags;
}
)";

const char* kCodeLightCS = R"(
#extension GL_KHR_shader_subgroup_arithmetic : require

layout (local_size_x = kBakePoints) in;

shared float partial[kBakePoints / 4u][27];

float shareOf(uint probe, uint point) {
  uint pair = pc.nlg.shares.v[(probe * kBakePoints + point) / 2u];
  vec2 v = unpackHalf2x16(pair);
  return (point & 1u) != 0u ? v.y : v.x;
}

void main() {
  uint local = gl_WorkGroupID.x;
  uint point = gl_LocalInvocationID.x;
  if (local >= pc.count) {
    return;
  }
  uint probe = pc.nlg.bakeOrder.v[pc.first + local];
  uint flags = pc.nlg.flags.v[probe];
  if (pc.mode != 0u && (flags & kProbeHasLighting) == 0u) {
    return;
  }
  float updates = pc.mode != 0u ? pc.nlg.sh.v[9u * probe + kShUpdates].w : 0.0;
  uint level = probeLevel(probe);
  vec3 origin = probePosition(probeCellOf(probe), level) + pc.nlg.points.v[pc.nlg.bakePoints.v[point]].xyz * kLevelSpacing[level];
  float share = pc.mode == 0u ? 1.0 : shareOf(probe, point);
  vec3 sh[9];
  for (uint i = 0; i != 9; i++) {
    sh[i] = vec3(0.0);
  }
  float back = 0.0;
  if (share > 0.0) {
    uint seed = tea(probe * kBakePoints + point, pc.frameId);
    vec4 q = randomRotation(seed);
    vec3 sunDir = pc.perFrame.sunDir.xyz;
    for (uint r = 0; r != kBakeRays; r++) {
      vec3 dir = rotate(q, fibonacciDirection(r, kBakeRays));
      vec3 hitPos;
      vec3 hitNormal;
      vec3 albedo;
      bool backFacing;
      vec3 radiance;
      if (!traceClosest(origin, dir, hitPos, hitNormal, albedo, backFacing)) {
        radiance = pc.perFrame.skyColor.rgb;
      } else if (backFacing) {
        back += 1.0;
        continue;
      } else {
        radiance = vec3(0.0);
        float cosSun = dot(hitNormal, sunDir);
        if (cosSun > 0.0 && !isOccluded(hitPos + hitNormal * kRayOffset, sunDir, 10000.0)) {
          radiance = albedo * kInvPi * pc.perFrame.sunColor.rgb * cosSun;
        }
        if (pc.mode != 0u) {
          radiance += albedo * kInvPi * nlgIrradiance(hitPos, hitNormal);
        }
      }
      float y[9];
      shBasis(dir, y);
      for (uint i = 0; i != 9; i++) {
        sh[i] += radiance * y[i];
      }
    }
  }
  float scale = 4.0 * 3.14159265 / float(kBakeRays);
  if (pc.mode == 0u) {
    pc.nlg.bakeLight.v[local * kBakePoints + point] = vec4(sh[0] * scale, back / float(kBakeRays));
    return;
  }
  for (uint i = 0; i != 9; i++) {
    vec3 sum = subgroupAdd(share * sh[i] * scale);
    if (subgroupElect()) {
      partial[gl_SubgroupID][3 * i + 0] = sum.r;
      partial[gl_SubgroupID][3 * i + 1] = sum.g;
      partial[gl_SubgroupID][3 * i + 2] = sum.b;
    }
  }
  barrier();
  float blend = max(pc.blend, 1.0 / (updates + 1.0));
  if (point < 9u) {
    vec4 value = vec4(0.0);
    for (uint k = 0; k != gl_NumSubgroups; k++) {
      value.rgb += vec3(partial[k][3 * point + 0], partial[k][3 * point + 1], partial[k][3 * point + 2]);
    }
    vec4 old = pc.nlg.sh.v[9u * probe + point];
    vec4 next = mix(old, value, blend);
    next.w = point == kShUpdates ? min(updates + 1.0, 1e6) : next.w;
    pc.nlg.sh.v[9u * probe + point] = next;
  }
}
)";

const char* kCodeQuantizeCS = R"(
layout (local_size_x = 256) in;

void main() {
  uint word = gl_GlobalInvocationID.x;
  if (word >= pc.count * (kSupportPoints / 4u)) {
    return;
  }
  uint packed = 0u;
  for (uint k = 0; k != 4u; k++) {
    float phi = pc.nlg.phiA.v[4u * word + k];
    int q = clamp(int(round(phi * 255.0)) + kGeneratorInputZeroPoint, -128, 127);
    packed |= uint(q & 0xFF) << (8u * k);
  }
  pc.nlg.graphInput.v[word] = packed;
}
)";

const char* kCodePackCS = R"(
layout (local_size_x = 64) in;

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= pc.count * kNetworkVec4s) {
    return;
  }
  uint probe = pc.nlg.bakeOrder.v[pc.first + index / kNetworkVec4s];
  uint h = 8u * (index % kNetworkVec4s);
  uint c = probe * kCoefficients;
  uvec4 packed;
  for (uint i = 0; i != 4u; i++) {
    float a = h + 2u * i < kCoefficients ? pc.nlg.coefficients.v[c + h + 2u * i] : 0.0;
    float b = h + 2u * i + 1u < kCoefficients ? pc.nlg.coefficients.v[c + h + 2u * i + 1u] : 0.0;
    packed[i] = packHalf2x16(vec2(a, b));
  }
  pc.nlg.networks.v[probe * kNetworkVec4s + index % kNetworkVec4s] = packed;
}
)";

const char* kCodeDequantizeCS = R"(
layout (local_size_x = 256) in;

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= pc.count * kCoefficients) {
    return;
  }
  uint o = index % kCoefficients;
  pc.nlg.coefficients.v[pc.nlg.bakeOrder.v[pc.first + index / kCoefficients] * kCoefficients + o] =
      float(int(pc.nlg.graphOutput.v[index])) * pc.nlg.outputScale.v[o];
}
)";

const char* kCodeApplyCS = R"(
layout (local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform constants {
  PerFrame perFrame;
  Uints slots;
  Uints cells;
  Nlg nlg;
  uint texPosition;
  uint texNormal;
  uint imageIndirect;
  uint texShading;
  uint imageIndirectSpec;
  uint texMaterial;
} pc;
)";

const char* kCodeApplyMainCS = R"(
void main() {
  uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= kWidth || p.y >= kHeight) {
    return;
  }
  vec4 posWS = texelFetch(kTextures2D[pc.texPosition], ivec2(p), 0);
  if (posWS.w <= 0.0) {
    imageStore(kImagesRGBA16F[pc.imageIndirect], ivec2(p), vec4(0.0, 0.0, 0.0, 1.0));
    imageStore(kImagesRGBA16F[pc.imageIndirectSpec], ivec2(p), vec4(0.0, 0.0, 0.0, 1.0));
    return;
  }
  vec3 n = normalize(texelFetch(kTextures2D[pc.texNormal], ivec2(p), 0).xyz);
  vec3 ns = normalize(texelFetch(kTextures2D[pc.texShading], ivec2(p), 0).xyz);
  uint model = uint(texelFetch(kTextures2D[pc.texMaterial], ivec2(p), 0).g * 255.0 + 0.5);
  vec3 r = model == kModelFoliage ? -ns : reflect(normalize(posWS.xyz - pc.perFrame.cameraPos.xyz), ns);
  vec3 er;
  float pending;
  vec3 e = nlgIrradiance2(posWS.xyz, n, ns, r, er, pending);
  // until the bake and the first lighting pass reach its probes, a pixel gets the constant sky ambient instead of their light
  vec3 ambient = kAmbientSky * 3.14159265 * pc.perFrame.skyColor.rgb;
  e = mix(e, ambient, pending);
  er = mix(er, ambient, pending);
  imageStore(kImagesRGBA16F[pc.imageIndirect], ivec2(p), vec4(e, 1.0));
  imageStore(kImagesRGBA16F[pc.imageIndirectSpec], ivec2(p), vec4(er, 1.0));
}
)";

const char* kCodeFunctionsCS = R"(
layout (local_size_x = kFunctionThreads) in;

const uint kPointsPerThread = kSupportPoints / kFunctionThreads;
const uint kNoOwner = 0x7C00FFFFu;

shared uint sValid[kSupportPoints / 32u];
shared uint sSeen[kSupportPoints / 32u];
shared uint sReceived[kSupportPoints / 32u];
shared uint sNew[kSupportPoints / 32u];
shared uint sCount[kFunctionThreads];
shared uint sIndex[kFunctionThreads];
shared float sSums[kFunctionThreads * kLightingClusters];
shared vec3 sCentres[kLightingClusters];
shared uint sShares[kBakePoints];
shared uint sFlag;

uint t;

void sync() {
  memoryBarrierBuffer();
  barrier();
}

bool validAt(uint i) {
  return (sValid[i >> 5] & (1u << (i & 31u))) != 0u;
}

bool receivedAt(uint i) {
  return (sReceived[i >> 5] & (1u << (i & 31u))) != 0u;
}

bool newAt(uint i) {
  return (sNew[i >> 5] & (1u << (i & 31u))) != 0u;
}

void setReceived(uint i) {
  atomicOr(sReceived[i >> 5], 1u << (i & 31u));
}

void setNew(uint i) {
  atomicOr(sNew[i >> 5], 1u << (i & 31u));
}

float trilinear(vec3 p) {
  vec3 a = max(1.0 - abs(p), vec3(0.0));
  return a.x * a.y * a.z;
}

uint reduceCount() {
  barrier();
  if (t == 0u) {
    uint sum = 0u;
    for (uint k = 0; k != kFunctionThreads; k++) {
      sum += sCount[k];
    }
    sFlag = sum;
  }
  barrier();
  uint result = sFlag;
  barrier();
  return result;
}

uint supportBits(uint at) {
  return pc.mode != 0u ? 0xFFFFFFFFu : pc.nlg.support.v[at * 2u];
}

float phiAt(uint parity, uint at) {
  return parity == 0u ? pc.nlg.phiA.v[at] : pc.nlg.phiB.v[at];
}

void setPhi(uint parity, uint at, float w) {
  if (parity == 0u) {
    pc.nlg.phiB.v[at] = w;
  } else {
    pc.nlg.phiA.v[at] = w;
  }
}

vec3 feature(uint at) {
  return log2(max(pc.nlg.scratch.v[at].rgb, vec3(0.0)) + 1e-3);
}

uint nearestCentre(vec3 f) {
  uint best = 0u;
  float bestDistance = 1e30;
  for (uint c = 0; c != kLightingClusters; c++) {
    vec3 d = f - sCentres[c];
    float distance = dot(d, d);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = c;
    }
  }
  return best;
}

void insideStart(uint local, uint base, uint probe) {
  for (uint k = 0; k != kPointsPerThread; k++) {
    pc.nlg.scratch.v[base + t + k * kFunctionThreads] = vec4(0.0);
  }
  sReceived[t] = 0u;
  sync();
  {
    uint bp = pc.nlg.bakePoints.v[t];
    vec4 light = pc.nlg.bakeLight.v[local * kBakePoints + t];
    if (validAt(bp) && light.w < kProbeBackfaceLimit) {
      pc.nlg.scratch.v[base + bp] = vec4(light.rgb, 0.0);
      setReceived(bp);
    }
  }
  sync();
  for (uint pass = 0; pass != 48; pass++) {
    sNew[t] = sReceived[t];
    if (t == 0u) {
      sFlag = 0u;
    }
    barrier();
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      if (receivedAt(i)) {
        continue;
      }
      uint bits = supportBits(base + i);
      vec3 sum = vec3(0.0);
      float den = 0.0;
      for (uint n = 0; n != kSupportNeighbours; n++) {
        if ((bits & (1u << n)) == 0u) {
          continue;
        }
        uint j = pc.nlg.neighbours.v[i * kSupportNeighbours + n];
        if (receivedAt(j)) {
          sum += pc.nlg.scratch.v[base + j].rgb;
          den += 1.0;
        }
      }
      if (den > 0.0) {
        pc.nlg.scratch.v[base + i] = vec4(sum / den, 0.0);
        setNew(i);
        sFlag = 1u;
      }
    }
    sync();
    sReceived[t] = sNew[t];
    uint changed = sFlag;
    barrier();
    if (changed == 0u) {
      break;
    }
  }
  uint lit = 0u;
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    if (receivedAt(i) && validAt(i)) {
      lit++;
    }
  }
  sCount[t] = lit;
  if (reduceCount() < kLightingClusters) {
    for (uint k = 0; k != kPointsPerThread; k++) {
      pc.nlg.phiA.v[base + t + k * kFunctionThreads] = 0.0;
    }
    sReceived[t] = 0u;
    sync();
    return;
  }
  if (t == 0u) {
    uint seed = tea(probe, 0x6b6du);
    uint chosen[kLightingClusters];
    for (uint c = 0; c != kLightingClusters; c++) {
      uint i = min(uint(rnd(seed) * float(kSupportPoints)), kSupportPoints - 1u);
      for (uint step = 0; step != kSupportPoints; step++) {
        bool taken = false;
        for (uint d = 0; d != c; d++) {
          taken = taken || chosen[d] == i;
        }
        if (!taken && receivedAt(i) && validAt(i)) {
          break;
        }
        i = (i + 1u) % kSupportPoints;
      }
      chosen[c] = i;
      sCentres[c] = feature(base + i);
    }
  }
  barrier();
  for (uint iteration = 0; iteration != 10; iteration++) {
    float sums[4u * kLightingClusters];
    for (uint m = 0; m != 4u * kLightingClusters; m++) {
      sums[m] = 0.0;
    }
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      if (!(receivedAt(i) && validAt(i))) {
        continue;
      }
      vec3 f = feature(base + i);
      uint c = nearestCentre(f);
      sums[4u * c + 0u] += f.x;
      sums[4u * c + 1u] += f.y;
      sums[4u * c + 2u] += f.z;
      sums[4u * c + 3u] += 1.0;
    }
    vec4 total = vec4(0.0);
    for (uint m = 0; m != 4u; m++) {
      for (uint c = 0; c != kLightingClusters; c++) {
        sSums[t * kLightingClusters + c] = sums[4u * c + m];
      }
      barrier();
      if (t < kLightingClusters) {
        for (uint k = 0; k != kFunctionThreads; k++) {
          total[m] += sSums[k * kLightingClusters + t];
        }
      }
      barrier();
    }
    if (t < kLightingClusters && total.w > 0.0) {
      sCentres[t] = total.xyz / total.w;
    }
    barrier();
  }
  {
    uint counts[kLightingClusters];
    for (uint c = 0; c != kLightingClusters; c++) {
      counts[c] = 0u;
    }
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      if (receivedAt(i) && validAt(i)) {
        counts[nearestCentre(feature(base + i))]++;
      }
    }
    for (uint c = 0; c != kLightingClusters; c++) {
      sSums[t * kLightingClusters + c] = float(counts[c]);
    }
    barrier();
    if (t == 0u) {
      uint largest = 0u;
      float largestCount = -1.0;
      for (uint c = 0; c != kLightingClusters; c++) {
        float count = 0.0;
        for (uint k = 0; k != kFunctionThreads; k++) {
          count += sSums[k * kLightingClusters + c];
        }
        if (count > largestCount) {
          largestCount = count;
          largest = c;
        }
      }
      sFlag = largest;
    }
    barrier();
  }
  uint largest = sFlag;
  sNew[t] = 0u;
  barrier();
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    bool member = receivedAt(i) && validAt(i) && nearestCentre(feature(base + i)) == largest;
    if (member) {
      setNew(i);
    }
    pc.nlg.labels.v[base + i] = member ? i : kSupportPoints;
  }
  sync();
  for (uint pass = 0; pass != 96; pass++) {
    if (t == 0u) {
      sFlag = 0u;
    }
    barrier();
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      if (!newAt(i)) {
        continue;
      }
      uint current = pc.nlg.labels.v[base + i];
      uint label = current;
      uint bits = supportBits(base + i);
      for (uint n = 0; n != kSupportNeighbours; n++) {
        if ((bits & (1u << n)) == 0u) {
          continue;
        }
        uint j = pc.nlg.neighbours.v[i * kSupportNeighbours + n];
        if (newAt(j)) {
          label = min(label, pc.nlg.labels.v[base + j]);
        }
      }
      if (label < current) {
        pc.nlg.labels.v[base + i] = label;
        sFlag = 1u;
      }
    }
    sync();
    uint changed = sFlag;
    barrier();
    if (changed == 0u) {
      break;
    }
  }
  for (uint k = 0; k != kPointsPerThread; k++) {
    pc.nlg.histogram.v[base + t + k * kFunctionThreads] = 0u;
  }
  sync();
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    if (newAt(i)) {
      atomicAdd(pc.nlg.histogram.v[base + pc.nlg.labels.v[base + i]], 1u);
    }
  }
  sync();
  {
    uint bestCount = 0u;
    uint bestIndex = 0u;
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      uint count = pc.nlg.histogram.v[base + i];
      if (count > bestCount || (count == bestCount && count > 0u && i < bestIndex)) {
        bestCount = count;
        bestIndex = i;
      }
    }
    sCount[t] = bestCount;
    sIndex[t] = bestIndex;
    barrier();
    if (t == 0u) {
      uint best = 0u;
      uint bestAt = 0u;
      for (uint k = 0; k != kFunctionThreads; k++) {
        if (sCount[k] > best || (sCount[k] == best && best > 0u && sIndex[k] < bestAt)) {
          best = sCount[k];
          bestAt = sIndex[k];
        }
      }
      sFlag = bestAt;
    }
    barrier();
  }
  uint component = sFlag;
  sReceived[t] = 0u;
  barrier();
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    float w = newAt(i) && pc.nlg.labels.v[base + i] == component ? trilinear(pc.nlg.points.v[i].xyz) : 0.0;
    pc.nlg.phiA.v[base + i] = w;
    if (w > 0.0) {
      setReceived(i);
    }
  }
  sync();
}

void diffuse(uint base, uint parity) {
  sNew[t] = sReceived[t];
  barrier();
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    float w = phiAt(parity, base + i);
    if (!receivedAt(i) && validAt(i)) {
      uint bits = supportBits(base + i);
      vec3 pi = pc.nlg.points.v[i].xyz;
      float num = 0.0;
      float den = 0.0;
      for (uint n = 0; n != kSupportNeighbours; n++) {
        if ((bits & (1u << n)) == 0u) {
          continue;
        }
        uint j = pc.nlg.neighbours.v[i * kSupportNeighbours + n];
        if (!receivedAt(j)) {
          continue;
        }
        vec3 d = pc.nlg.points.v[j].xyz - pi;
        float g = exp(-dot(d, d) / (2.0 * kDiffusionSigma * kDiffusionSigma));
        num += g * phiAt(parity, base + j);
        den += g;
      }
      if (den > 0.0) {
        w = num / den;
        setNew(i);
      }
    }
    setPhi(parity, base + i, w);
  }
  sync();
  sReceived[t] = sNew[t];
  barrier();
}

void attributeSupport(uint local, uint base, uint probe, bool inside) {
  for (uint k = 0; k != kPointsPerThread; k++) {
    pc.nlg.labels.v[base + t + k * kFunctionThreads] = kNoOwner;
  }
  sync();
  {
    uint bp = pc.nlg.bakePoints.v[t];
    if (validAt(bp) && (pc.mode == 2u || pc.nlg.bakeLight.v[local * kBakePoints + t].w < kProbeBackfaceLimit)) {
      pc.nlg.labels.v[base + bp] = t;
    }
  }
  sync();
  for (uint pass = 0; pass != 64; pass++) {
    if (t == 0u) {
      sFlag = 0u;
    }
    barrier();
    for (uint k = 0; k != kPointsPerThread; k++) {
      uint i = t + k * kFunctionThreads;
      if (!validAt(i)) {
        continue;
      }
      uint current = pc.nlg.labels.v[base + i];
      uint best = current;
      uint bits = supportBits(base + i);
      vec3 pi = pc.nlg.points.v[i].xyz;
      for (uint n = 0; n != kSupportNeighbours; n++) {
        if ((bits & (1u << n)) == 0u) {
          continue;
        }
        uint j = pc.nlg.neighbours.v[i * kSupportNeighbours + n];
        if (!validAt(j)) {
          continue;
        }
        uint pj = pc.nlg.labels.v[base + j];
        if ((pj & 0xFFFFu) == 0xFFFFu) {
          continue;
        }
        float dj = unpackHalf2x16(pj >> 16).x + distance(pc.nlg.points.v[j].xyz, pi);
        uint candidate = (packHalf2x16(vec2(dj, 0.0)) << 16) | (pj & 0xFFFFu);
        best = min(best, candidate);
      }
      if (best < current) {
        pc.nlg.labels.v[base + i] = best;
        sFlag = 1u;
      }
    }
    sync();
    uint changed = sFlag;
    barrier();
    if (changed == 0u) {
      break;
    }
  }
  sShares[t] = 0u;
  barrier();
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    if (!validAt(i)) {
      continue;
    }
    uint owner = pc.nlg.labels.v[base + i] & 0xFFFFu;
    if (owner == 0xFFFFu) {
      continue;
    }
    float w = pc.nlg.phiA.v[base + i];
    if (w > 0.0) {
      atomicAdd(sShares[owner], uint(w * 65536.0 + 0.5));
    }
  }
  barrier();
  if (t == 0u) {
    uint sum = 0u;
    for (uint b = 0; b != kBakePoints; b++) {
      sum += sShares[b];
    }
    sFlag = sum;
  }
  barrier();
  uint total = sFlag;
  if (t < kBakePoints / 2u) {
    vec2 v = total > 0u ? vec2(float(sShares[2u * t]), float(sShares[2u * t + 1u])) / float(total) : vec2(0.0);
    pc.nlg.shares.v[(probe * kBakePoints) / 2u + t] = packHalf2x16(v);
  }
  if (t == 0u) {
    pc.nlg.flags.v[probe] = (total > 0u ? kProbeHasLighting : 0u) | (inside ? kProbeInside : 0u);
  }
}

void copyReference(uint probe, bool shares) {
  if (shares && t < kBakePoints / 2u) {
    pc.nlg.shares.v[(probe * kBakePoints) / 2u + t] = pc.nlg.shares.v[(kNumProbes * kBakePoints) / 2u + t];
  }
  if (t < kCoefficients) {
    pc.nlg.coefficients.v[probe * kCoefficients + t] = pc.nlg.coefficients.v[kNumProbes * kCoefficients + t];
  }
  if (shares && t == 0u) {
    pc.nlg.flags.v[probe] = kProbeHasLighting;
  }
}

void main() {
  uint local = gl_WorkGroupID.x;
  t = gl_LocalInvocationID.x;
  if (local >= pc.count) {
    return;
  }
  uint probe = pc.nlg.bakeOrder.v[pc.first + local];
  uint base = local * kSupportPoints;
  if (pc.mode == 1u) {
    sCount[t] = pc.nlg.bakeLight.v[local * kBakePoints + t].w < kProbeBackfaceLimit ? 1u : 0u;
    bool allSeeds = reduceCount() == kBakePoints;
    copyReference(probe, allSeeds);
    if (allSeeds) {
      return;
    }
  }
  sValid[t] = 0u;
  sSeen[t] = 0u;
  barrier();
  uint centreBack = 0u;
  for (uint k = 0; k != kPointsPerThread; k++) {
    uint i = t + k * kFunctionThreads;
    uint flags = pc.mode != 0u ? 1u : pc.nlg.support.v[(base + i) * 2u + 1u];
    bool valid = (flags & 2u) == 0u;
    bool seen = valid && (flags & 1u) != 0u;
    if ((flags & 4u) != 0u) {
      centreBack++;
    }
    if (valid) {
      atomicOr(sValid[i >> 5], 1u << (i & 31u));
    }
    if (seen) {
      atomicOr(sSeen[i >> 5], 1u << (i & 31u));
    }
    pc.nlg.phiA.v[base + i] = seen ? trilinear(pc.nlg.points.v[i].xyz) : 0.0;
  }
  sCount[t] = centreBack;
  bool inside = reduceCount() * 2u > kSupportPoints;
  sReceived[t] = sSeen[t];
  sync();
  if (inside) {
    insideStart(local, base, probe);
  }
  for (uint pass = 0; pass != kDiffusionPasses; pass++) {
    diffuse(base, pass & 1u);
  }
  attributeSupport(local, base, probe, inside);
}
)";

const char* kCodeGeneratorCS = R"(
layout (local_size_x = kGeneratorThreads) in;

shared float sX[kGeneratorProbes][kGeneratorThreads];
shared float sH[kGeneratorProbes][kGeneratorWidth];

float leakyRelu(float a) {
  return a > 0.0 ? a : 0.01 * a;
}

void main() {
  uint t = gl_LocalInvocationID.x;
  uint firstLocal = gl_WorkGroupID.x * kGeneratorProbes;
  if (firstLocal >= pc.count) {
    return;
  }
  float acc[kGeneratorProbes][2];
  for (uint p = 0; p != kGeneratorProbes; p++) {
    acc[p][0] = 0.0;
    acc[p][1] = 0.0;
  }
  for (uint chunk = 0; chunk < kSupportPoints; chunk += kGeneratorThreads) {
    float m = pc.nlg.pcaMean.v[chunk + t];
    for (uint p = 0; p != kGeneratorProbes; p++) {
      sX[p][t] = pc.nlg.phiA.v[(firstLocal + p) * kSupportPoints + chunk + t] - m;
    }
    barrier();
    for (uint k = 0; k != kGeneratorThreads; k++) {
      vec2 b = unpackHalf2x16(pc.nlg.pcaBasis.v[(chunk + k) * (kPcaComponents / 2u) + t]);
      for (uint p = 0; p != kGeneratorProbes; p++) {
        float x = sX[p][k];
        acc[p][0] += x * b.x;
        acc[p][1] += x * b.y;
      }
    }
    barrier();
  }
  for (uint p = 0; p != kGeneratorProbes; p++) {
    sH[p][2u * t] = acc[p][0];
    sH[p][2u * t + 1u] = acc[p][1];
  }
  barrier();
  uint offset = 0u;
  for (uint layer = 0; layer != kGeneratorLayers + 1u; layer++) {
    bool last = layer == kGeneratorLayers;
    uint outputs = last ? kCoefficients : kGeneratorWidth;
    uint o0 = t;
    uint o1 = t + kGeneratorThreads;
    uint biasAt = offset + kGeneratorWidth * outputs;
    for (uint p = 0; p != kGeneratorProbes; p++) {
      acc[p][0] = o0 < outputs ? pc.nlg.weights.v[biasAt + o0] : 0.0;
      acc[p][1] = o1 < outputs ? pc.nlg.weights.v[biasAt + o1] : 0.0;
    }
    for (uint i = 0; i != kGeneratorWidth; i++) {
      float w0 = o0 < outputs ? pc.nlg.weights.v[offset + i * outputs + o0] : 0.0;
      float w1 = o1 < outputs ? pc.nlg.weights.v[offset + i * outputs + o1] : 0.0;
      for (uint p = 0; p != kGeneratorProbes; p++) {
        float h = sH[p][i];
        acc[p][0] += w0 * h;
        acc[p][1] += w1 * h;
      }
    }
    barrier();
    for (uint p = 0; p != kGeneratorProbes; p++) {
      if (last) {
        if (o0 < outputs && firstLocal + p < pc.count) {
          pc.nlg.coefficients.v[pc.nlg.bakeOrder.v[pc.first + firstLocal + p] * kCoefficients + o0] = acc[p][0];
        }
      } else if (layer == 0u) {
        sH[p][o0] = leakyRelu(acc[p][0]);
        sH[p][o1] = leakyRelu(acc[p][1]);
      } else {
        sH[p][o0] += leakyRelu(acc[p][0]);
        sH[p][o1] += leakyRelu(acc[p][1]);
      }
    }
    barrier();
    offset = biasAt + outputs;
  }
}
)";

const char* kCodeGeneratorCooperativeExtensions = R"(
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
)";

// the generator as f16 cooperative matrices accumulated in f32: per layer over the probes of a batch, a dispatch of partial
// sums over slices of k and one that sums them and finishes the layer, the activations in f16 (the A matrices of the next
// layer) and the skip connections summed in f32
const char* kCodeGeneratorCooperativePushConstants = R"(
layout(std430, buffer_reference) readonly buffer Halves {
  float16_t v[];
};

layout(std430, buffer_reference) buffer HalvesRW {
  float16_t v[];
};

layout(push_constant) uniform constants {
  Halves inputs;      // [kBakeBatch][k]
  Halves weights;     // [k][n]
  Floats biases;      // [n]
  FloatsRW stream;    // [kBakeBatch][kGeneratorWidth]
  FloatsRW partials;  // [splits][kBakeBatch][n]
  HalvesRW outputs;   // [kBakeBatch][n]
  Nlg nlg;
  uint first;
  uint count;
  uint k;
  uint n;
  uint mode;
  uint splits;
} pc;

// f32 to f16 rounded to nearest even on every driver, as some round toward zero: the value is rounded to 11 significant bits
// (or to the f16 subnormal step) first, which converts exactly
float16_t toHalf(float x) {
  if (abs(x) < 6.103515625e-5) {
    return float16_t(roundEven(x * 16777216.0) / 16777216.0);
  }
  const uint u = floatBitsToUint(x);
  return float16_t(uintBitsToFloat((u + 0xFFFu + ((u >> 13) & 1u)) & ~0x1FFFu));
}
)";

const char* kCodeGeneratorCooperativeInputCS = R"(
layout (local_size_x = 256) in;

void main() {
  const uint index = gl_GlobalInvocationID.x;
  if (index >= pc.count * kSupportPoints) {
    return;
  }
  pc.outputs.v[index] = toHalf(pc.nlg.phiA.v[index] - pc.nlg.pcaMean.v[index % kSupportPoints]);
}
)";

const char* kCodeGeneratorCooperativeCS = R"(
layout (local_size_x = 64) in;

#define MatA coopmat<float16_t, gl_ScopeSubgroup, kTileM, kTileK, gl_MatrixUseA>
#define MatB coopmat<float16_t, gl_ScopeSubgroup, kTileK, kTileN, gl_MatrixUseB>
#define MatC coopmat<float, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator>

void main() {
  const uint blocksN = pc.n / (kTileN * kBlock);
  const uint blocks = (pc.count + kTileM * kBlock - 1u) / (kTileM * kBlock) * blocksN;
  const uint slice = pc.k / pc.splits;
  for (uint t = gl_SubgroupID; t < kTilesPerGroup; t += gl_NumSubgroups) {
    // the blocks of a slice are next to each other, sharing its rows of the weights in the cache
    const uint job = gl_WorkGroupID.x * kTilesPerGroup + t;
    if (job >= blocks * pc.splits) {
      break;
    }
    const uint split = job / blocks;
    const uint block = job % blocks;
    // the rows past the batch multiply what the buffers hold there, and are left out by the second pass
    const uint m0 = block / blocksN * kTileM * kBlock;
    const uint n0 = block % blocksN * kTileN * kBlock;
    MatC c[kBlock][kBlock];
    [[unroll]] for (uint i = 0u; i != kBlock; i++) {
      [[unroll]] for (uint j = 0u; j != kBlock; j++) {
        c[i][j] = MatC(0.0);
      }
    }
    for (uint k = split * slice; k != (split + 1u) * slice; k += kTileK) {
      MatA a[kBlock];
      MatB b[kBlock];
      [[unroll]] for (uint i = 0u; i != kBlock; i++) {
        coopMatLoad(a[i], pc.inputs.v, (m0 + i * kTileM) * pc.k + k, pc.k, gl_CooperativeMatrixLayoutRowMajor);
        coopMatLoad(b[i], pc.weights.v, k * pc.n + n0 + i * kTileN, pc.n, gl_CooperativeMatrixLayoutRowMajor);
      }
      [[unroll]] for (uint i = 0u; i != kBlock; i++) {
        [[unroll]] for (uint j = 0u; j != kBlock; j++) {
          c[i][j] = coopMatMulAdd(a[i], b[j], c[i][j]);
        }
      }
    }
    [[unroll]] for (uint i = 0u; i != kBlock; i++) {
      [[unroll]] for (uint j = 0u; j != kBlock; j++) {
        const uint at = (split * kBakeBatch + m0 + i * kTileM) * pc.n + n0 + j * kTileN;
        coopMatStore(c[i][j], pc.partials.v, at, pc.n, gl_CooperativeMatrixLayoutRowMajor);
      }
    }
  }
}
)";

const char* kCodeGeneratorCooperativeFinishCS = R"(
layout (local_size_x = 256) in;

float leakyRelu(float a) {
  return a > 0.0 ? a : 0.01 * a;
}

void main() {
  const uint index = gl_GlobalInvocationID.x;
  if (index >= pc.count * pc.n) {
    return;
  }
  const uint o = index % pc.n;
  float z = 0.0;
  for (uint split = 0u; split != pc.splits; split++) {
    z += pc.partials.v[split * kBakeBatch * pc.n + index];
  }
  // n is kGeneratorWidth but in the output layer, so index addresses the activations and the stream
  if (pc.mode == kCooperativePca) {
    pc.outputs.v[index] = toHalf(z);
  } else if (pc.mode == kCooperativeOutput) {
    if (o < kCoefficients) {
      pc.nlg.coefficients.v[pc.nlg.bakeOrder.v[pc.first + index / pc.n] * kCoefficients + o] = z + pc.biases.v[o];
    }
  } else {
    const float h = (pc.mode == kCooperativeSkip ? pc.stream.v[index] : 0.0) + leakyRelu(z + pc.biases.v[o]);
    pc.stream.v[index] = h;
    pc.outputs.v[index] = toHalf(h);
  }
}
)";

const char* kCodePresentPushConstants = R"(
#extension GL_EXT_buffer_reference : require

layout(std430, buffer_reference) buffer Exposure {
  vec4 v;
};

)" PER_FRAME_GLSL R"(
layout(push_constant) uniform constants {
  PerFrame perFrame;
  Exposure exposureBuffer;
  vec2 clipRotation;
  uint texPosition;
  uint texAlbedo;
  uint texDirect;
  uint texIndirect;
  uint texShading;
  uint texMaterial;
  uint texIndirectSpec;
  uint gi;
  uint textures;
  float exposure;
  uint autoExposure;
  float exposureAdapt;
} pc;
)";

const char* kCodePresentColor = R"(
const float kPi = 3.14159265;

vec2 envBrdf(float roughness, float nDotV) {
  vec4 r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(r.x * r.x, exp2(-9.28 * nDotV)) * r.x + r.y;
  return vec2(-1.04, 1.04) * a004 + r.zw;
}

vec3 fresnelSchlick(vec3 f0, float vDotH) {
  return f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
}

vec3 ggxFresnel(vec3 f0, float roughness, float nDotV, vec2 ab) {
  vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
  vec3 fssEss = (f0 + fr * pow(1.0 - nDotV, 5.0)) * ab.x + ab.y;
  float ems = 1.0 - ab.x - ab.y;
  vec3 favg = f0 + (1.0 - f0) / 21.0;
  return fssEss + ems * fssEss * favg / (1.0 - favg * ems);
}

float sheenLambdaHelper(float x, float alphaG) {
  float oneMinusAlphaSq = (1.0 - alphaG) * (1.0 - alphaG);
  float a = mix(21.5473, 25.3245, oneMinusAlphaSq);
  float b = mix(3.82987, 3.32435, oneMinusAlphaSq);
  float c = mix(0.19823, 0.16801, oneMinusAlphaSq);
  float d = mix(-1.97760, -1.27393, oneMinusAlphaSq);
  float e = mix(-4.32054, -4.85967, oneMinusAlphaSq);
  return a / (1.0 + b * pow(x, c)) + d * x + e;
}

float sheenLambda(float cosTheta, float alphaG) {
  return cosTheta < 0.5 ? exp(sheenLambdaHelper(cosTheta, alphaG))
                        : exp(2.0 * sheenLambdaHelper(0.5, alphaG) - sheenLambdaHelper(1.0 - cosTheta, alphaG));
}

float sheenVisibility(float nDotL, float nDotV, float roughness) {
  float alphaG = roughness * roughness;
  return clamp(1.0 / ((1.0 + sheenLambda(nDotV, alphaG) + sheenLambda(nDotL, alphaG)) * (4.0 * nDotV * nDotL)), 0.0, 1.0);
}

float charlieDistribution(float roughness, float nDotH) {
  float invR = 1.0 / (roughness * roughness);
  return (2.0 + invR) * pow(max(1.0 - nDotH * nDotH, 0.0), invR * 0.5) / (2.0 * kPi);
}

float sheenAlbedo(float nDotV) {
  return max(0.28304 * exp(-nDotV) + 0.52437 * exp(-3.0 * nDotV) + 0.01844 * exp(-9.0 * nDotV) - 0.10116, 0.0);
}

bool sceneColor(ivec2 q, out vec3 color) {
  vec4 posWS = texelFetch(kTextures2D[pc.texPosition], q, 0);
  if (posWS.w <= 0.0) {
    color = pc.perFrame.skyColor.rgb;
    return false;
  }
  bool gi = pc.gi != 0u;
  // without GI, or before any probe is baked, the constant sky ambient
  vec3 constantAmbient = kAmbientSky * kPi * pc.perFrame.skyColor.rgb;
  vec3 indirect = gi ? texelFetch(kTextures2D[pc.texIndirect], q, 0).rgb : constantAmbient;
  vec3 base = pc.textures != 0u ? pow(texelFetch(kTextures2D[pc.texAlbedo], q, 0).rgb, vec3(2.2)) : vec3(1.0);
  vec4 shading = texelFetch(kTextures2D[pc.texShading], q, 0);
  vec3 n = normalize(shading.xyz);
  float roughness = shading.w;
  vec4 material = texelFetch(kTextures2D[pc.texMaterial], q, 0);
  float metallic = material.r;
  uint model = uint(material.g * 255.0 + 0.5);
  float transmission = model == kModelFoliage ? kFoliageTransmission : 0.0;
  float specularWeight = model == kModelFoliage ? kFoliageSpecular : 1.0;
  float shadow = texelFetch(kTextures2D[pc.texDirect], q, 0).r;
  vec3 v = normalize(pc.perFrame.cameraPos.xyz - posWS.xyz);
  vec3 l = pc.perFrame.sunDir.xyz;
  float nDotV = max(dot(n, v), 1e-4);
  float cosL = dot(n, l);
  float nDotL = clamp(cosL, 0.0, 1.0);
  vec3 h = normalize(v + l);
  float nDotH = clamp(dot(n, h), 0.0, 1.0);
  float vDotH = clamp(dot(v, h), 0.0, 1.0);
  float vDotHDiffuse = cosL < 0.0 ? clamp(dot(v, normalize(reflect(l, n) + v)), 0.0, 1.0) : vDotH;
  float a = max(roughness, 0.1);
  float a2 = a * a * a * a;
  float d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  float distribution = a2 / (kPi * d * d);
  float visibility = 0.5 / max(nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2) + nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2), 1e-5);
  vec3 specular = vec3(distribution * visibility * nDotL);
  vec3 diffuse = base * ((1.0 - transmission) * nDotL + transmission * clamp(-cosL, 0.0, 1.0)) / kPi;
  vec3 fDielectric = fresnelSchlick(vec3(kDielectricF0), vDotHDiffuse) * specularWeight;
  vec3 fMetal = fresnelSchlick(base, vDotH);
  vec3 sun = pc.perFrame.sunColor.rgb * shadow;
  vec3 direct = sun * mix(mix(diffuse, specular, fDielectric), fMetal * specular, metallic);
  vec3 sheenColor = sqrt(base);
  float sheenMax = max(sheenColor.r, max(sheenColor.g, sheenColor.b));
  if (model == kModelCloth) {
    float scaling = 1.0 - sheenMax * sheenAlbedo(nDotV);
    vec3 sheen = vec3(0.0);
    if (nDotL > 0.0) {
      float s = charlieDistribution(kSheenRoughness, nDotH) * sheenVisibility(nDotL, nDotV, kSheenRoughness);
      sheen = sun * nDotL * s * sheenColor;
      scaling = min(scaling, 1.0 - sheenMax * sheenAlbedo(nDotL));
    }
    direct = sheen + direct * scaling;
  }
  vec3 irradiance = indirect / kPi;
  vec3 indirectSpec = gi ? texelFetch(kTextures2D[pc.texIndirectSpec], q, 0).rgb / kPi : constantAmbient / kPi;
  vec3 fDiffuse = base * irradiance;
  vec3 fSpecular = indirectSpec;
  if (model == kModelFoliage) {
    fDiffuse = mix(fDiffuse, base * indirectSpec, transmission);
    fSpecular = irradiance;
  }
  vec2 ab = envBrdf(roughness, nDotV);
  vec3 ambient = mix(mix(fDiffuse, fSpecular, ggxFresnel(vec3(kDielectricF0), roughness, nDotV, ab) * specularWeight),
                     ggxFresnel(base, roughness, nDotV, ab) * fSpecular, metallic);
  if (model == kModelCloth) {
    float e = sheenAlbedo(nDotV);
    ambient = sheenColor * e * irradiance + ambient * (1.0 - sheenMax * e);
  }
  color = direct + ambient;
  return true;
}
)";

const char* kCodeExposureCS = R"(
layout (local_size_x = 256) in;

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];

const uint kMeterW = 64;
const uint kMeterH = 36;
const float kExposureKey = 0.18;

shared float sLogSum[256];
shared float sCount[256];
)";

const char* kCodeExposureMainCS = R"(
void main() {
  uint tid = gl_LocalInvocationIndex;
  float sum = 0.0;
  float count = 0.0;
  for (uint i = tid; i < kMeterW * kMeterH; i += 256u) {
    vec2 uv = (vec2(i % kMeterW, i / kMeterW) + 0.5) / vec2(kMeterW, kMeterH);
    vec3 c;
    if (sceneColor(ivec2(uv * vec2(kWidth, kHeight)), c)) {
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      if (l > 1e-6) {
        sum += log2(l);
        count += 1.0;
      }
    }
  }
  sLogSum[tid] = sum;
  sCount[tid] = count;
  barrier();
  for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sLogSum[tid] += sLogSum[tid + s];
      sCount[tid] += sCount[tid + s];
    }
    barrier();
  }
  if (tid == 0u && sCount[0] > 0.0) {
    vec4 prev = pc.exposureBuffer.v;
    float target = clamp(log2(kExposureKey) - sLogSum[0] / sCount[0], -10.0, 10.0);
    float ev = prev.y > 0.0 ? mix(prev.z, target, pc.exposureAdapt) : target;
    pc.exposureBuffer.v = vec4(exp2(ev), 1.0, ev, 0.0);
  }
}
)";

const char* kCodePresentVS = FULLSCREEN_TRIANGLE_GLSL R"(
layout (location=0) out vec2 uv;

void main() {
  gl_Position = fullscreenTriangle(pc.clipRotation, uv);
}
)";

const char* kCodePresentFS = R"(
layout (location=0) in vec2 uv;

layout (location=0) out vec4 out_FragColor;

vec3 tonemap(vec3 color) {
  const float startCompression = 0.8 - 0.04;
  const float desaturation = 0.15;
  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;
  float peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return color;
  }
  const float d = 1.0 - startCompression;
  float newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;
  float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, vec3(newPeak), g);
}

void main() {
  ivec2 q = clamp(ivec2(uv * vec2(kWidth, kHeight)), ivec2(0), ivec2(kWidth - 1, kHeight - 1));
  vec3 color;
  sceneColor(q, color);
  float exposure = pc.autoExposure != 0u ? pc.exposureBuffer.v.x * pc.exposure : pc.exposure;
  out_FragColor = vec4(tonemap(exposure * color), 1.0);
}
)";

enum GPUTimestamp {
  GPUTimestamp_Begin = 0,
  GPUTimestamp_GBuffer,
  GPUTimestamp_Direct,
  GPUTimestamp_Lighting,
  GPUTimestamp_Bake,
  GPUTimestamp_GI,
  GPUTimestamp_Present,
  GPUTimestamp_NUM,
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  vec4 cameraPos;
  vec4 sunDir;
  vec4 sunColor;
  vec4 skyColor;
};

struct GBufferPushConstants {
  uint64_t perFrame;
  uint64_t materials;
  uint64_t models;
  uint32_t sampler0;
};

struct DirectPushConstants {
  uint64_t perFrame;
  uint64_t materials;
  uint64_t vertices;
  uint64_t indices;
  uint32_t tlas;
  uint32_t sampler0;
  uint32_t texPosition;
  uint32_t texNormal;
  uint32_t imageDirect;
  uint32_t texMaterial;
};

struct NlgPushConstants {
  uint64_t perFrame;
  uint64_t materials;
  uint64_t vertices;
  uint64_t indices;
  uint64_t slots;
  uint64_t cells;
  uint64_t nlg;
  uint32_t tlas;
  uint32_t sampler0;
  uint32_t frameId;
  uint32_t first;
  uint32_t count;
  uint32_t mode;
  float blend;
};

struct GeneratorCooperativePushConstants {
  uint64_t inputs;
  uint64_t weights;
  uint64_t biases;
  uint64_t stream;
  uint64_t partials;
  uint64_t outputs;
  uint64_t nlg;
  uint32_t first;
  uint32_t count;
  uint32_t k;
  uint32_t n;
  uint32_t mode;
  uint32_t splits;
};

struct ApplyPushConstants {
  uint64_t perFrame;
  uint64_t slots;
  uint64_t cells;
  uint64_t nlg;
  uint32_t texPosition;
  uint32_t texNormal;
  uint32_t imageIndirect;
  uint32_t texShading;
  uint32_t imageIndirectSpec;
  uint32_t texMaterial;
};

struct PresentPushConstants {
  uint64_t perFrame;
  uint64_t exposureBuffer;
  vec2 clipRotation;
  uint32_t texPosition;
  uint32_t texAlbedo;
  uint32_t texDirect;
  uint32_t texIndirect;
  uint32_t texShading;
  uint32_t texMaterial;
  uint32_t texIndirectSpec;
  uint32_t gi;
  uint32_t textures;
  float exposure;
  uint32_t autoExposure;
  float exposureAdapt;
};

struct NlgBuffers {
  uint64_t points;
  uint64_t neighbours;
  uint64_t bakePoints;
  uint64_t support;
  uint64_t phiA;
  uint64_t phiB;
  uint64_t histogram;
  uint64_t scratch;
  uint64_t labels;
  uint64_t bakeLight;
  uint64_t shares;
  uint64_t coefficients;
  uint64_t sh;
  uint64_t flags;
  uint64_t pcaMean;
  uint64_t pcaBasis;
  uint64_t weights;
  uint64_t graphInput;
  uint64_t graphOutput;
  uint64_t outputScale;
  uint64_t networks;
  uint64_t bakeOrder;
};

// a layer of the generator's int8 TOSA graph: the RESCALEs it lacks are empty (`writeIntegerGraph()` of `train_nlg.py`)
struct GraphLayer {
  uint32_t numInputs = 0;
  uint32_t numOutputs = 0;
  std::vector<int8_t> weights;
  std::vector<int32_t> biases;
  std::vector<int32_t> fromStream;
  std::vector<int8_t> fromStreamShift;
  std::vector<int32_t> toActivation;
  std::vector<int8_t> toActivationShift;
  std::vector<int32_t> toStream;
  std::vector<int8_t> toStreamShift;
};

struct Generator {
  std::vector<vec4> points;
  std::vector<uint32_t> neighbours;
  std::vector<uint32_t> bakePoints;
  std::vector<float> mean;
  std::vector<uint32_t> basis;
  std::vector<float> weights;
  std::vector<float> outputScale;
  std::vector<GraphLayer> graphLayers;
  std::vector<int8_t> graphTable;
};

static_assert(kFunctionThreads == kBakePoints);
static_assert(kFunctionThreads == kSupportPoints / 32);
static_assert(kPcaComponents == kGeneratorWidth);
static_assert(kDiffusionPasses % 2 == 0);
static_assert(sizeof(NlgPushConstants) <= 128 && sizeof(ApplyPushConstants) <= 128 && sizeof(PresentPushConstants) <= 128);
static_assert(sizeof(GeneratorCooperativePushConstants) <= 128 && kCooperativeOutputs >= kCoefficients);

bool loadGenerator(const std::vector<uint8_t>& blob, Generator& out) {
  const uint32_t* header = (const uint32_t*)blob.data();
  if (blob.size() < 32) {
    return false;
  }
  const bool isKnownVersion = header[1] == kGeneratorVersion || header[1] == kGeneratorVersion - 1;
  const uint32_t numGraphLayers = header[1] == kGeneratorVersion ? header[7] : 0;
  if (header[0] != kGeneratorMagic || !isKnownVersion || header[2] != kSupportPoints || header[3] != kPcaComponents ||
      header[4] != kGeneratorWidth || header[5] != kGeneratorLayers || header[6] != kCoefficients ||
      (numGraphLayers && numGraphLayers != kGraphLayers)) {
    return false;
  }
  const size_t floatSize = 32 + sizeof(float) * 3 * kSupportPoints + sizeof(float) * kSupportPoints +
                           sizeof(uint16_t) * kSupportPoints * kPcaComponents + sizeof(float) * kGeneratorWeights +
                           sizeof(float) * kCoefficients;
  if (blob.size() < floatSize) {
    return false;
  }
  const float* at = (const float*)(blob.data() + 32);
  out.points.resize(kSupportPoints);
  for (uint32_t i = 0; i != kSupportPoints; i++) {
    out.points[i] = vec4(at[3 * i], at[3 * i + 1], at[3 * i + 2], 0.0f);
  }
  at += 3 * kSupportPoints;
  out.mean.assign(at, at + kSupportPoints);
  at += kSupportPoints;
  out.basis.assign((const uint32_t*)at, (const uint32_t*)at + kSupportPoints * kPcaComponents / 2);
  at += kSupportPoints * kPcaComponents / 2;
  out.weights.assign(at, at + kGeneratorWeights);
  at += kGeneratorWeights;
  out.outputScale.assign(at, at + kCoefficients);

  // the int8 graph: per layer the weights and biases, then fromStream, toActivation and toStream where the layer has them
  const uint8_t* data = blob.data() + floatSize;
  const uint8_t* end = blob.data() + blob.size();
  auto take = [&data, end](void* dst, size_t bytes) -> bool {
    if (bytes > size_t(end - data)) {
      return false;
    }
    memcpy(dst, data, bytes);
    data += bytes;
    return true;
  };
  out.graphLayers.resize(numGraphLayers);
  for (uint32_t i = 0; i != numGraphLayers; i++) {
    GraphLayer& layer = out.graphLayers[i];
    const bool isLast = i + 1 == numGraphLayers;
    const uint32_t numFromStream = i >= 2 ? kGeneratorWidth : 0;
    layer.numInputs = i ? kGeneratorWidth : kSupportPoints;
    layer.numOutputs = isLast ? kCoefficients : kGeneratorWidth;
    const uint32_t numToActivation = isLast ? 0 : layer.numOutputs;
    const uint32_t numToStream = i && !isLast ? kGeneratorWidth : 0;
    layer.weights.resize(size_t(layer.numOutputs) * layer.numInputs);
    layer.biases.resize(layer.numOutputs);
    layer.fromStream.resize(numFromStream);
    layer.fromStreamShift.resize(numFromStream);
    layer.toActivation.resize(numToActivation);
    layer.toActivationShift.resize(numToActivation);
    layer.toStream.resize(numToStream);
    layer.toStreamShift.resize(numToStream);
    const bool isRead =
        take(layer.weights.data(), layer.weights.size()) && take(layer.biases.data(), sizeof(int32_t) * layer.biases.size()) &&
        take(layer.fromStream.data(), sizeof(int32_t) * numFromStream) && take(layer.fromStreamShift.data(), numFromStream) &&
        take(layer.toActivation.data(), sizeof(int32_t) * numToActivation) && take(layer.toActivationShift.data(), numToActivation) &&
        take(layer.toStream.data(), sizeof(int32_t) * numToStream) && take(layer.toStreamShift.data(), numToStream);
    if (!isRead) {
      return false;
    }
  }
  out.graphTable.resize(numGraphLayers ? kGraphTableSize : 0);
  return take(out.graphTable.data(), out.graphTable.size()) && data == end;
}

// the generator's int8 graph over [1, kBakeBatch, 1, channels] as `train_nlg.py` evaluates it: the PCA, then LeakyReLU layers
// (a TABLE) summed into an int32 stream by the skip connections, then the output layer, whose int32 result a compute pass scales
std::vector<uint32_t> buildGeneratorGraph(const Generator& generator) {
  const std::vector<GraphLayer>& layers = generator.graphLayers;
  const int8_t* table = generator.graphTable.data();
  TosaGraphBuilder graph(1, kBakeBatch, kSupportPoints);
  auto activation = [&graph, table](uint32_t value, const GraphLayer& layer) -> uint32_t {
    const uint32_t sum = graph.conv2d(value, layer.numOutputs, 0);
    return graph.table(graph.rescale(sum, false, layer.toActivation.data(), layer.toActivationShift.data(), 0), table);
  };
  const GraphLayer& pca = layers.front();
  const uint32_t z = graph.rescale(graph.conv2d(graph.input(), pca.numOutputs, kGeneratorInputZeroPoint),
                                   false,
                                   pca.toActivation.data(),
                                   pca.toActivationShift.data(),
                                   0);
  uint32_t stream = graph.rescale(activation(z, layers[1]), true, layers[1].toStream.data(), layers[1].toStreamShift.data(), 0);
  for (size_t i = 2; i + 1 < layers.size(); i++) {
    const uint32_t x = graph.rescale(stream, false, layers[i].fromStream.data(), layers[i].fromStreamShift.data(), 0);
    stream = graph.add(stream, graph.rescale(activation(x, layers[i]), true, layers[i].toStream.data(), layers[i].toStreamShift.data(), 0));
  }
  const GraphLayer& last = layers.back();
  const uint32_t x = graph.rescale(stream, false, last.fromStream.data(), last.fromStreamShift.data(), 0);
  return graph.build(graph.conv2d(x, last.numOutputs, 0));
}

bool triangleOverlapsCube(const vec3& a, const vec3& b, const vec3& c, float h) {
  for (uint32_t i = 0; i != 3; i++) {
    if (std::min(a[i], std::min(b[i], c[i])) > h || std::max(a[i], std::max(b[i], c[i])) < -h) {
      return false;
    }
  }
  const vec3 edges[3] = {b - a, c - b, a - c};
  for (const vec3& e : edges) {
    const vec3 axes[3] = {vec3(0.0f, -e.z, e.y), vec3(e.z, 0.0f, -e.x), vec3(-e.y, e.x, 0.0f)};
    for (const vec3& n : axes) {
      const float pa = glm::dot(n, a);
      const float pb = glm::dot(n, b);
      const float pc = glm::dot(n, c);
      const float r = h * (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
      if (std::min(pa, std::min(pb, pc)) > r || std::max(pa, std::max(pb, pc)) < -r) {
        return false;
      }
    }
  }
  const vec3 n = glm::cross(edges[0], edges[1]);
  return std::abs(glm::dot(n, a)) <= h * (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
}

void buildSupportGraph(Generator& g) {
  g.neighbours.assign(size_t(kSupportPoints) * kSupportNeighbours, 0u);
  const uint32_t numThreads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  for (uint32_t w = 0; w != numThreads; w++) {
    threads.emplace_back([&g, w, numThreads]() {
      std::vector<std::pair<float, uint32_t>> nearest(kSupportPoints);
      for (uint32_t i = w; i < kSupportPoints; i += numThreads) {
        const vec3 pi = vec3(g.points[i]);
        for (uint32_t j = 0; j != kSupportPoints; j++) {
          const vec3 d = vec3(g.points[j]) - pi;
          nearest[j] = {i == j ? 1e30f : glm::dot(d, d), j};
        }
        std::partial_sort(nearest.begin(), nearest.begin() + kSupportNeighbours, nearest.end());
        for (uint32_t k = 0; k != kSupportNeighbours; k++) {
          g.neighbours[size_t(i) * kSupportNeighbours + k] = nearest[k].second;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  LRandom rng;
  std::vector<vec3> centres(kBakePoints);
  for (vec3& centre : centres) {
    centre = vec3(g.points[uint32_t(rng.random() * float(kSupportPoints)) % kSupportPoints]);
  }
  std::vector<uint32_t> owner(kSupportPoints, 0u);
  for (uint32_t iteration = 0; iteration != 32; iteration++) {
    for (uint32_t i = 0; i != kSupportPoints; i++) {
      float best = 1e30f;
      for (uint32_t c = 0; c != kBakePoints; c++) {
        const vec3 d = vec3(g.points[i]) - centres[c];
        const float dist = glm::dot(d, d);
        if (dist < best) {
          best = dist;
          owner[i] = c;
        }
      }
    }
    std::vector<vec3> sums(kBakePoints, vec3(0.0f));
    std::vector<uint32_t> counts(kBakePoints, 0u);
    for (uint32_t i = 0; i != kSupportPoints; i++) {
      sums[owner[i]] += vec3(g.points[i]);
      counts[owner[i]]++;
    }
    for (uint32_t c = 0; c != kBakePoints; c++) {
      if (counts[c]) {
        centres[c] = sums[c] / float(counts[c]);
      }
    }
  }
  g.bakePoints.assign(kBakePoints, 0u);
  std::vector<uint8_t> taken(kSupportPoints, 0);
  for (uint32_t c = 0; c != kBakePoints; c++) {
    float best = 1e30f;
    for (uint32_t i = 0; i != kSupportPoints; i++) {
      if (taken[i]) {
        continue;
      }
      const vec3 d = vec3(g.points[i]) - centres[c];
      const float dist = glm::dot(d, d);
      if (dist < best) {
        best = dist;
        g.bakePoints[c] = i;
      }
    }
    taken[g.bakePoints[c]] = 1;
  }
}

enum {
  kBakeFileMagic = 0x4B42474E,
  kBakeFileVersion = 1,
};

struct BakeFileHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t key;
  uint32_t numProbes;
  uint32_t networkWords;
  uint32_t shareWords;
  uint32_t reserved;
};

uint64_t hashBytes(uint64_t h, const void* data, size_t size) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (size_t i = 0; i != size; i++) {
    h = (h ^ bytes[i]) * 1099511628211ull;
  }
  return h;
}

template<typename T>
uint64_t hashVector(uint64_t h, const std::vector<T>& v) {
  return hashBytes(h, v.data(), sizeof(T) * v.size());
}

bool g_ToggleGI = false;
bool g_ToggleTextures = false;

enum MaterialModel : uint32_t {
  MaterialModel_Dielectric = 0,
  MaterialModel_Metal,
  MaterialModel_Cloth,
  MaterialModel_Foliage,
};

uint32_t materialModel(const char* name) {
  std::string s(name);
  std::transform(s.begin(), s.end(), s.begin(), [](char c) { return c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c; });
  const auto has = [&s](const char* token) { return s.find(token) != std::string::npos; };
  if (has("fabric") || has("curtain")) {
    return MaterialModel_Cloth;
  }
  if (s == "leaf" || (has("foliage") && (has("leaf") || has("leaves") || has("hedge") || has("flower")))) {
    return MaterialModel_Foliage;
  }
  const char* metals[] = {"metal",
                          "bronze",
                          "chain",
                          "chrome",
                          "manhole",
                          "rollup_door",
                          "lantern",
                          "bollard",
                          "streetpivot",
                          "flagpole",
                          "vase_hanging",
                          "electricbox"};
  for (const char* metal : metals) {
    if (has(metal)) {
      return MaterialModel_Metal;
    }
  }
  return MaterialModel_Dielectric;
}

vec3 sunDirection(float azimuth, float elevation) {
  return vec3(std::cos(elevation) * std::cos(azimuth), std::sin(elevation), std::cos(elevation) * std::sin(azimuth));
}

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      // the size of the monitor unless `--width` and `--height` say otherwise
      .width = 0,
      .height = 0,
      .resizable = false,
      .contextConfig =
          {
              .enableValidationGpuAV = false,
              .swapchainUseSurfacePreTransform = true,
              .layers = {"VK_LAYER_ML_Graph_Emulation", "VK_LAYER_ML_Tensor_Emulation"},
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  // nothing depends on the resolution but the G-buffer and the screen-space passes: the probes and the networks live in world space
  const uint32_t renderWidth = (uint32_t)app.width_;
  const uint32_t renderHeight = (uint32_t)app.height_;

  int sceneIndex = 1;
  bool showGI = true;
  bool showTextures = false;
  bool texturesRequested = false; // textures are loaded the first time `showTextures` is set
  float sunElevation = 60.0f; // degrees from horizontal
  float sunAzimuth = 135.0f; // degrees around Y axis
  float exposure = 2.5f;
  float exposureCompensation = 1.5f;
  bool autoExposure = true;
  float probeSpacing = 6.0f;
  uint32_t probeLevels = 2;
  bool measureTime = false;
  bool showUI = true;
  bool rebake = false;
  int preferredPath = -1; // the data graph, else the compute shader

  for (int i = 1; i < argc; i++) {
    const char* arg = argv[i];
    const char* next = i + 1 < argc ? argv[i + 1] : nullptr;
    if (!strcmp(arg, "--scene") && next) {
      for (int s = 0; s != (int)LVK_ARRAY_NUM_ELEMENTS(kScenes); s++) {
        sceneIndex = !strcmp(next, kScenes[s].name) ? s : sceneIndex;
      }
      i++;
    } else if (!strcmp(arg, "--no-gi")) {
      showGI = false;
    } else if (!strcmp(arg, "--textures")) {
      showTextures = true;
    } else if (!strcmp(arg, "--sun-azimuth") && next) {
      sunAzimuth = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--sun-elevation") && next) {
      sunElevation = (float)atof(next);
      i++;
    } else if (!strcmp(arg, "--probe-levels") && next) {
      probeLevels = std::clamp<uint32_t>((uint32_t)atoi(next), 1, kMaxProbeLevels);
      i++;
    } else if (!strcmp(arg, "--probe-spacing") && next) {
      probeSpacing = std::max((float)atof(next), 0.5f);
      i++;
    } else if (!strcmp(arg, "--no-auto-exposure")) {
      autoExposure = false;
    } else if (!strcmp(arg, "--timings")) {
      measureTime = true;
    } else if (!strcmp(arg, "--no-ui")) {
      showUI = false;
    } else if (!strcmp(arg, "--rebake")) {
      rebake = true;
    } else if (!strcmp(arg, "--generator") && next) {
      preferredPath = !strcmp(next, "compute")      ? GeneratorPath_Compute
                      : !strcmp(next, "data-graph") ? GeneratorPath_DataGraph
                      : !strcmp(next, "cm")         ? GeneratorPath_Cooperative
                                                    : preferredPath;
      i++;
    }
  }

  const Scene& scene = kScenes[sceneIndex];

  app.cfg_.initialCameraPos = scene.cameraPos;
  app.cfg_.initialCameraTarget = scene.cameraTarget;

  app.positioner_.lookAt(app.cfg_.initialCameraPos, app.cfg_.initialCameraTarget, app.cfg_.initialCameraUpVector);

  lvk::IContext* ctx = app.ctx_.get();

  const bool hasRayQuery = ctx->supportsRayQuery();

  // the start-up takes a while (the probe lattice, the acceleration structures, the shaders), so every step is logged
  const double startupTime = glfwGetTime();
  auto logStartup = [startupTime](const char* step) { LLOGL("Start-up, %5.1f s: %s\n", glfwGetTime() - startupTime, step); };

  logStartup("loading the mesh");
  {
    const std::string cacheFileName = app.folderContentRoot_ + scene.cacheFile;

    if (!loadFromCache(app, cacheFileName.c_str())) {
      if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), scene.modelFile))) {
        LVK_ASSERT_MSG(false, "Cannot load 3D model. Run `deploy_content.py` before running this app.");
        VULKAN_APP_EXIT();
      }
    }
  }

  vec3 sceneMin = vec3(1e30f);
  vec3 sceneMax = vec3(-1e30f);

  for (const VertexData& v : vertexData_) {
    sceneMin = glm::min(sceneMin, v.position * kSceneScale);
    sceneMax = glm::max(sceneMax, v.position * kSceneScale);
  }

  const vec3 sceneSize = sceneMax - sceneMin;
  const float maxSize = std::max(sceneSize.x, std::max(sceneSize.y, sceneSize.z));
  probeSpacing = std::max(probeSpacing, maxSize / float(kMaxProbesPerAxis - 2));
  struct ProbeLevel {
    vec3 origin;
    float spacing;
    uint32_t dims[3];
    uint32_t slots;
  };
  std::vector<ProbeLevel> levels;
  std::vector<uint32_t> probeSlots;
  std::vector<uint32_t> probeCells;
  std::vector<uint32_t> complexProbes;
  std::vector<uint32_t> freeProbes;

  for (uint32_t level = 0; level != probeLevels; level++) {
    const float spacing = probeSpacing / float(1u << level);
    char step[64];
    snprintf(step, sizeof(step), "placing the probes of level %u of %u", level + 1, probeLevels);
    logStartup(step);
    auto latticeSize = [&](uint32_t axis) { return uint32_t(std::floor(sceneSize[axis] / spacing)) + 2; };
    const ProbeLevel l = {
        .origin = sceneMin - 0.5f * spacing,
        .spacing = spacing,
        .dims = {latticeSize(0), latticeSize(1), latticeSize(2)},
        .slots = (uint32_t)probeSlots.size(),
    };
    const size_t numLattice = size_t(l.dims[0]) * l.dims[1] * l.dims[2];
    if (level && probeSlots.size() + numLattice > kMaxLatticePoints) {
      LLOGW("Probe level %u (%.3f apart) would need %zu lattice points: %u levels\n", level, spacing, numLattice, level);
      break;
    }
    // 0: far from geometry, 1: near a triangle's bounds, 2: the cube of one spacing around the probe touches a triangle
    std::vector<uint8_t> marks(numLattice, 0);
    // every thread marks its own z-slices, so the threads never write the same mark
    const uint32_t numThreads = std::clamp(std::thread::hardware_concurrency(), 1u, l.dims[2]);
    std::vector<std::thread> threads;
    for (uint32_t w = 0; w != numThreads; w++) {
      threads.emplace_back([&, w]() {
        const vec3 maxCell = vec3(float(l.dims[0] - 1), float(l.dims[1] - 1), float(l.dims[2] - 1));
        const float half = kFreeSpaceExtent * spacing;
        for (size_t t = 0; t + 2 < indexData_.size(); t += 3) {
          vec3 v[3];
          vec3 lo = vec3(1e30f);
          vec3 hi = vec3(-1e30f);
          for (uint32_t k = 0; k != 3; k++) {
            v[k] = vertexData_[indexData_[t + k]].position * kSceneScale;
            lo = glm::min(lo, v[k]);
            hi = glm::max(hi, v[k]);
          }
          const vec3 a = glm::clamp(glm::floor((lo - l.origin) / spacing) - 1.0f, vec3(0.0f), maxCell);
          const vec3 b = glm::clamp(glm::floor((hi - l.origin) / spacing) + 2.0f, vec3(0.0f), maxCell);
          const uint32_t z0 = (uint32_t)a.z + (w + numThreads - (uint32_t)a.z % numThreads) % numThreads;
          for (uint32_t z = z0; z <= (uint32_t)b.z; z += numThreads) {
            for (uint32_t y = (uint32_t)a.y; y <= (uint32_t)b.y; y++) {
              for (uint32_t x = (uint32_t)a.x; x <= (uint32_t)b.x; x++) {
                uint8_t& m = marks[x + size_t(l.dims[0]) * (y + size_t(l.dims[1]) * z)];
                if (m != 2) {
                  const vec3 p = l.origin + vec3(float(x), float(y), float(z)) * spacing;
                  m = triangleOverlapsCube(v[0] - p, v[1] - p, v[2] - p, half) ? 2 : 1;
                }
              }
            }
          }
        }
      });
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
    probeSlots.resize(probeSlots.size() + numLattice, ~0u);
    for (size_t i = 0; i != numLattice; i++) {
      const uint8_t m = marks[i];
      if (m == 0 || (level && m != 2)) {
        continue;
      }
      probeSlots[l.slots + i] = (uint32_t)probeCells.size();
      (m == 2 ? complexProbes : freeProbes).push_back((uint32_t)probeCells.size());
      probeCells.push_back(uint32_t(i) | (level << 30));
    }
    levels.push_back(l);
  }
  std::vector<uint32_t> bakeOrder = complexProbes;
  bakeOrder.insert(bakeOrder.end(), freeProbes.begin(), freeProbes.end());
  bakeOrder.push_back((uint32_t)probeCells.size());
  uint32_t numComplexProbes = (uint32_t)complexProbes.size();
  const ProbeLevel& level0 = levels[0];

  const uint32_t numProbes = (uint32_t)probeCells.size();

  LLOGL("Scene `%s`: %.1f x %.1f x %.1f, %u x %u x %u probes %.2f apart, %zu levels, %u probes, %u in free space\n",
        scene.name,
        sceneSize.x,
        sceneSize.y,
        sceneSize.z,
        level0.dims[0],
        level0.dims[1],
        level0.dims[2],
        probeSpacing,
        levels.size(),
        numProbes,
        numProbes - numComplexProbes);

  // texture index 0 means "no texture"; loadMaterialTextures() fills them in later, see the render loop
  for (const CachedMaterial& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{.ambient = vec4(mtl.ambient, 1.0f), .diffuse = vec4(mtl.diffuse, 1.0f)});
  }

  auto createStorageBuffer = [&](size_t size, const void* data, const char* name) {
    return ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = size,
        .data = data,
        .debugName = name,
    });
  };

  lvk::Holder<lvk::BufferHandle> sbMaterials =
      createStorageBuffer(sizeof(GPUMaterial) * materials_.size(), materials_.data(), "Buffer: materials");
  std::vector<uint32_t> materialModels(cachedMaterials_.size());
  for (size_t i = 0; i != cachedMaterials_.size(); i++) {
    materialModels[i] = materialModel(cachedMaterials_[i].name);
  }
  lvk::Holder<lvk::BufferHandle> sbMaterialModels =
      createStorageBuffer(sizeof(uint32_t) * materialModels.size(), materialModels.data(), "Buffer: material models");

  const uint8_t accelStructUsage = hasRayQuery ? lvk::BufferUsageBits_AccelStructBuildInputReadOnly : 0;

  lvk::Holder<lvk::BufferHandle> vb0 = ctx->createBuffer({
      .usage = uint8_t(lvk::BufferUsageBits_Vertex | lvk::BufferUsageBits_Storage | accelStructUsage),
      .storage = lvk::StorageType_Device,
      .size = sizeof(VertexData) * vertexData_.size(),
      .data = vertexData_.data(),
      .debugName = "Buffer: vertex",
  });
  lvk::Holder<lvk::BufferHandle> ib0 = ctx->createBuffer({
      .usage = uint8_t(lvk::BufferUsageBits_Index | lvk::BufferUsageBits_Storage | accelStructUsage),
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

  const uint32_t numVertices = (uint32_t)vertexData_.size();
  const uint32_t numTriangles = numIndices / 3;

  SceneAccelStructs accel;
  lvk::Holder<lvk::BufferHandle> ibOpaqueFirst;
  lvk::BufferHandle ibTrace = ib0;
  std::vector<uint32_t> traceIndices;
  std::vector<uint16_t> triangleMaterials;
  bool opaqueSplitDone = !hasRayQuery;

  auto splitOpaqueGeometry = [&]() {
    if (opaqueSplitDone || !texturesRequested || numRemainingMaterialTextures()) {
      return;
    }
    opaqueSplitDone = true;
    std::vector<uint32_t> sorted;
    sorted.reserve(traceIndices.size());
    uint32_t numOpaque = 0;
    for (uint32_t pass = 0; pass != 2; pass++) {
      for (uint32_t t = 0; t != numTriangles; t++) {
        const uint16_t m = triangleMaterials[t];
        const bool alphaTested = m >= materialsAlphaTested_.size() || materialsAlphaTested_[m];
        if (alphaTested == (pass == 1)) {
          sorted.insert(sorted.end(), traceIndices.begin() + 3 * size_t(t), traceIndices.begin() + 3 * size_t(t) + 3);
        }
      }
      if (pass == 0) {
        numOpaque = uint32_t(sorted.size() / 3);
      }
    }
    traceIndices = {};
    triangleMaterials = {};
    ibOpaqueFirst = ctx->createBuffer({
        .usage = uint8_t(lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly),
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * sorted.size(),
        .data = sorted.data(),
        .debugName = "Buffer: index (opaque first)",
    });
    ibTrace = ibOpaqueFirst;
    accel = createSceneAccelStructs(ctx, vb0, numVertices, ibTrace, numTriangles, numOpaque, kSceneScale);
    LLOGL("Ray tracing: %u of %u triangles opaque\n", numOpaque, numTriangles);
  };

  if (hasRayQuery) {
    logStartup("building the acceleration structures");
    // untextured nothing is alpha-tested; splitOpaqueGeometry() rebuilds them once the textures are in
    accel = createSceneAccelStructs(ctx, vb0, numVertices, ib0, numTriangles, numTriangles, kSceneScale);
    triangleMaterials.resize(numTriangles);
    for (uint32_t t = 0; t != numTriangles; t++) {
      triangleMaterials[t] = vertexData_[indexData_[3 * size_t(t)]].mtlIndex;
    }
    traceIndices = std::move(indexData_);
  } else {
    LLOGW("Ray queries are not supported: only the G-buffer is available\n");
  }

  vertexData_ = {};
  indexData_ = {};

  Generator generator;
  bool nlgReady = false;
  const std::string generatorBinPath = app.folderContentRoot_ + "src/neural/nlg/nlg.bin";

  logStartup("loading the generator");
  {
    const std::vector<uint8_t> blob = app.loadFile(generatorBinPath.c_str());
    if (blob.empty()) {
      LLOGW("Cannot load `%s`: the Neural Light Grid is not available\n", generatorBinPath.c_str());
    } else if (!loadGenerator(blob, generator)) {
      LLOGW("`%s` is not a generator of %u support points, PCA %u, %u layers %u wide to %u coefficients\n",
            generatorBinPath.c_str(),
            kSupportPoints,
            kPcaComponents,
            kGeneratorLayers,
            kGeneratorWidth,
            kCoefficients);
    } else {
      logStartup("connecting the support points");
      buildSupportGraph(generator);
      nlgReady = hasRayQuery;
      for (const vec4& p : generator.points) {
        if (std::max(std::abs(p.x), std::max(std::abs(p.y), std::abs(p.z))) > kFreeSpaceExtent) {
          LLOGW("The support points leave the cube of one spacing: every probe is baked in full\n");
          numComplexProbes = numProbes;
          break;
        }
      }
    }
  }

  bool graphReady = false;

  if (nlgReady && ctx->supportsTensorsARM() && ctx->supportsDataGraphARM()) {
    graphReady = !generator.graphLayers.empty();
    if (!graphReady) {
      LLOGW("`%s` has no integer graph: the generator cannot run as a data graph\n", generatorBinPath.c_str());
    }
  } else if (nlgReady) {
    LLOGW("VK_ARM_tensors and VK_ARM_data_graph are not available: the generator cannot run as a data graph\n");
  }

  const CooperativeMatrixShape tile =
      nlgReady ? getCooperativeMatrixShape(ctx, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR)
               : CooperativeMatrixShape{};
  const bool isTiled = kBakeBatch % (tile.m * kCooperativeBlock) == 0 && kCooperativeOutputs % (tile.n * kCooperativeBlock) == 0 &&
                       kSupportPoints % (tile.k * kCooperativePcaSplits) == 0 && kGeneratorWidth % (tile.k * kCooperativeSplits) == 0;
  const bool cooperativeReady = tile.m && isTiled;
  if (nlgReady && !cooperativeReady) {
    LLOGW("f16 cooperative matrices (VK_KHR_cooperative_matrix) are not available: the generator cannot run on them\n");
  }

  std::string constants;
  auto addConst = [&constants](const std::string& declaration, const std::string& value) {
    constants += "const " + declaration + " = " + value + ";\n";
  };
  auto addUint = [&addConst](const char* name, uint32_t value) { addConst(std::string("uint ") + name, std::to_string(value)); };
  auto addFloat = [&addConst](const char* name, float value) {
    char text[32];
    snprintf(text, sizeof(text), "%.8f", value);
    addConst(std::string("float ") + name, text);
  };
  addUint("kWidth", renderWidth);
  addUint("kHeight", renderHeight);
  addUint("kNumProbes", numProbes);
  addFloat("kProbeBackfaceLimit", kProbeBackfaceLimit);
  addFloat("kSceneScale", kSceneScale);
  addUint("kSupportPoints", kSupportPoints);
  addUint("kSupportNeighbours", kSupportNeighbours);
  addUint("kBakePoints", kBakePoints);
  addUint("kBakeRays", kBakeRays);
  addUint("kPcaComponents", kPcaComponents);
  addUint("kGeneratorWidth", kGeneratorWidth);
  addUint("kGeneratorLayers", kGeneratorLayers);
  addUint("kGeneratorProbes", kGeneratorProbes);
  addUint("kGeneratorThreads", kGeneratorThreads);
  addUint("kFunctionThreads", kFunctionThreads);
  addUint("kCoefficients", kCoefficients);
  addUint("kNetworkVec4s", kNetworkVec4s);
  addFloat("kDiffusionSigma", kDiffusionSigma);
  addUint("kDiffusionPasses", kDiffusionPasses);
  addUint("kLightingClusters", kLightingClusters);
  addFloat("kLookupOffset", kLookupOffset);
  addFloat("kLevelCoverage", kLevelCoverage);
  addFloat("kAmbientSky", kAmbientSky);
  addUint("kProbeHasLighting", kProbeHasLighting);
  addUint("kProbeInside", kProbeInside);
  addUint("kProbeNotBaked", kProbeNotBaked);
  addUint("kShUpdates", kShUpdates);
  addConst("int kGeneratorInputZeroPoint", std::to_string(kGeneratorInputZeroPoint));
  addUint("kCooperativePca", kCooperativePca);
  addUint("kCooperativeFirst", kCooperativeFirst);
  addUint("kCooperativeSkip", kCooperativeSkip);
  addUint("kCooperativeOutput", kCooperativeOutput);
  std::string levelDims;
  std::string levelOrigins;
  std::string levelSpacings;
  std::string levelSlots;
  for (const ProbeLevel& l : levels) {
    const char* sep = levelDims.empty() ? "" : ", ";
    char item[128];
    snprintf(item, sizeof(item), "%suvec3(%u, %u, %u)", sep, l.dims[0], l.dims[1], l.dims[2]);
    levelDims += item;
    snprintf(item, sizeof(item), "%svec3(%.8f, %.8f, %.8f)", sep, l.origin.x, l.origin.y, l.origin.z);
    levelOrigins += item;
    snprintf(item, sizeof(item), "%s%.8f", sep, l.spacing);
    levelSpacings += item;
    snprintf(item, sizeof(item), "%s%uu", sep, l.slots);
    levelSlots += item;
  }
  const std::string numLevels = std::to_string(levels.size());
  addConst("uint kProbeLevels", numLevels);
  addConst("uvec3 kLevelDims[" + numLevels + "]", "uvec3[](" + levelDims + ")");
  addConst("vec3 kLevelOrigin[" + numLevels + "]", "vec3[](" + levelOrigins + ")");
  addConst("float kLevelSpacing[" + numLevels + "]", "float[](" + levelSpacings + ")");
  addConst("uint kLevelSlots[" + numLevels + "]", "uint[](" + levelSlots + ")");
  constants += kCodeMaterialModels;
  const std::string codeGBufferVS = constants + kCodeGBufferVS;
  const std::string codeGBufferFS = constants + kCodeGBufferFS;
  const std::string codePresentVS = constants + kCodePresentPushConstants + kCodePresentVS;
  const std::string codePresentFS = constants + kCodePresentPushConstants + kCodePresentColor + kCodePresentFS;
  const std::string codeScene = constants + kCodeSceneDeclarations;
  const std::string codeNlg = codeScene + kCodeNlgPushConstants;
  const std::string codeNlgProbes = codeNlg + kCodeSceneFunctions + kCodeProbeFunctions;
  const std::string codeCooperative =
      kCodeGeneratorCooperativeExtensions + constants + "const uint kTileM = " + std::to_string(tile.m) +
      ";\nconst uint kTileN = " + std::to_string(tile.n) + ";\nconst uint kTileK = " + std::to_string(tile.k) +
      ";\nconst uint kTilesPerGroup = " + std::to_string(kCooperativeTilesPerGroup) +
      ";\nconst uint kBlock = " + std::to_string(kCooperativeBlock) + ";\nconst uint kBakeBatch = " + std::to_string(kBakeBatch) + ";\n" +
      kCodeSceneDeclarations + kCodeGeneratorCooperativePushConstants;

  logStartup("compiling the shaders");
  lvk::Holder<lvk::ShaderModuleHandle> smGBufferVert =
      ctx->createShaderModule({codeGBufferVS.c_str(), lvk::Stage_Vert, "Shader Module: G-buffer (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smGBufferFrag =
      ctx->createShaderModule({codeGBufferFS.c_str(), lvk::Stage_Frag, "Shader Module: G-buffer (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentVert =
      ctx->createShaderModule({codePresentVS.c_str(), lvk::Stage_Vert, "Shader Module: present (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> smPresentFrag =
      ctx->createShaderModule({codePresentFS.c_str(), lvk::Stage_Frag, "Shader Module: present (frag)"});

  struct ComputeProgram {
    lvk::Holder<lvk::ShaderModuleHandle> sm;
    lvk::Holder<lvk::ComputePipelineHandle> pipeline;
  };
  auto createProgram = [ctx, &logStartup](bool enabled, const std::string& code, const char* debugName) {
    if (!enabled) {
      return ComputeProgram{};
    }
    logStartup(debugName);
    lvk::Holder<lvk::ShaderModuleHandle> sm = ctx->createShaderModule({code.c_str(), lvk::Stage_Comp, debugName});
    lvk::Holder<lvk::ComputePipelineHandle> pipeline = ctx->createComputePipeline({.smComp = sm});
    return ComputeProgram{.sm = std::move(sm), .pipeline = std::move(pipeline)};
  };

  const ComputeProgram programExposure =
      createProgram(true,
                    constants + kCodePresentPushConstants + kCodeExposureCS + kCodePresentColor + kCodeExposureMainCS,
                    "Shader Module: exposure (comp)");
  const ComputeProgram programDirect =
      createProgram(hasRayQuery, codeScene + kCodeDirectCS + kCodeSceneFunctions + kCodeDirectMainCS, "Shader Module: direct (comp)");
  const ComputeProgram programApply = createProgram(
      nlgReady, codeScene + kCodeApplyCS + kCodeProbeFunctions + kCodeNlgFunctions + kCodeApplyMainCS, "Shader Module: NLG apply (comp)");
  const ComputeProgram programSupport = createProgram(nlgReady, codeNlgProbes + kCodeSupportCS, "Shader Module: NLG support (comp)");
  const ComputeProgram programLight =
      createProgram(nlgReady, codeNlgProbes + kCodeNlgFunctions + kCodeLightCS, "Shader Module: NLG lighting (comp)");
  const ComputeProgram programFunctions = createProgram(nlgReady, codeNlgProbes + kCodeFunctionsCS, "Shader Module: NLG functions (comp)");
  const ComputeProgram programGenerator = createProgram(nlgReady, codeNlg + kCodeGeneratorCS, "Shader Module: NLG generator (comp)");
  const ComputeProgram programPack = createProgram(nlgReady, codeNlg + kCodePackCS, "Shader Module: NLG pack (comp)");
  const ComputeProgram programQuantize = createProgram(graphReady, codeNlg + kCodeQuantizeCS, "Shader Module: NLG quantize (comp)");
  const ComputeProgram programDequantize = createProgram(graphReady, codeNlg + kCodeDequantizeCS, "Shader Module: NLG dequantize (comp)");
  const ComputeProgram programCooperativeInput = createProgram(
      cooperativeReady, codeCooperative + kCodeGeneratorCooperativeInputCS, "Shader Module: NLG generator input (cooperative matrices)");
  const ComputeProgram programCooperative =
      createProgram(cooperativeReady, codeCooperative + kCodeGeneratorCooperativeCS, "Shader Module: NLG generator (cooperative matrices)");
  const ComputeProgram programCooperativeFinish = createProgram(
      cooperativeReady, codeCooperative + kCodeGeneratorCooperativeFinishCS, "Shader Module: NLG generator finish (cooperative matrices)");

  const lvk::Format formatPosition = lvk::Format_RGBA_F32;
  const lvk::Format formatNormal = lvk::Format_RGBA_F16;
  const lvk::Format formatAlbedo = lvk::Format_RGBA_UN8;
  const lvk::Format formatDepth = lvk::Format_Z_F32;

  lvk::Holder<lvk::RenderPipelineHandle> pipelineGBuffer = ctx->createRenderPipeline({
      .vertexInput =
          {
              .attributes = {{.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
                             {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
                             {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
                             {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)}},
              .inputBindings = {{.stride = sizeof(VertexData)}},
          },
      .smVert = smGBufferVert,
      .smFrag = smGBufferFrag,
      .color = {{.format = formatPosition},
                {.format = formatNormal},
                {.format = formatAlbedo},
                {.format = formatNormal},
                {.format = formatAlbedo}},
      .depthFormat = formatDepth,
      .cullMode = VK_CULL_MODE_NONE,
      .debugName = "Pipeline: G-buffer",
  });
  lvk::Holder<lvk::RenderPipelineHandle> pipelinePresent = ctx->createRenderPipeline({
      .smVert = smPresentVert,
      .smFrag = smPresentFrag,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .cullMode = VK_CULL_MODE_NONE,
      .debugName = "Pipeline: present",
  });
  const vec4 exposureInit = vec4(0.0f);
  lvk::Holder<lvk::BufferHandle> exposureBuffer = createStorageBuffer(sizeof(vec4), &exposureInit, "Buffer: exposure");
  lvk::Holder<lvk::SamplerHandle> samplerMaterials = ctx->createSampler({
      .mipMap = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .debugName = "Sampler: materials",
  });

  const lvk::Dimensions dimensions = {renderWidth, renderHeight, 1};

  // sampled by the passes after the one that writes them
  auto createTarget = [&](lvk::Format format, uint8_t usage, const char* debugName) {
    return ctx->createTexture({
        .format = format,
        .dimensions = dimensions,
        .usage = (uint8_t)(usage | lvk::TextureUsageBits_Sampled),
        .debugName = debugName,
    });
  };
  lvk::Holder<lvk::TextureHandle> texPosition = createTarget(formatPosition, lvk::TextureUsageBits_Attachment, "G-buffer: position");
  lvk::Holder<lvk::TextureHandle> texNormal = createTarget(formatNormal, lvk::TextureUsageBits_Attachment, "G-buffer: normal");
  lvk::Holder<lvk::TextureHandle> texAlbedo = createTarget(formatAlbedo, lvk::TextureUsageBits_Attachment, "G-buffer: albedo");
  lvk::Holder<lvk::TextureHandle> texShading =
      createTarget(formatNormal, lvk::TextureUsageBits_Attachment, "G-buffer: shading normal and roughness");
  lvk::Holder<lvk::TextureHandle> texMaterial =
      createTarget(formatAlbedo, lvk::TextureUsageBits_Attachment, "G-buffer: metallic and material model");
  lvk::Holder<lvk::TextureHandle> texDepth = ctx->createTexture({
      .format = formatDepth,
      .dimensions = dimensions,
      .usage = lvk::TextureUsageBits_Attachment,
      .debugName = "G-buffer: depth",
  });
  lvk::Holder<lvk::TextureHandle> texDirect = createTarget(lvk::Format_R_F16, lvk::TextureUsageBits_Storage, "Direct sunlight");
  if (!hasRayQuery) {
    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();
    buf.cmdClearColorImage(texDirect, {.float32 = {1.0f, 1.0f, 1.0f, 1.0f}});
    ctx->submit(buf);
  }
  lvk::Holder<lvk::TextureHandle> texIndirect = createTarget(lvk::Format_RGBA_F16, lvk::TextureUsageBits_Storage, "Indirect irradiance");
  lvk::Holder<lvk::TextureHandle> texIndirectSpec =
      createTarget(lvk::Format_RGBA_F16, lvk::TextureUsageBits_Storage, "Indirect irradiance around the reflection");
  lvk::Holder<lvk::BufferHandle> probeSlotsBuffer =
      createStorageBuffer(sizeof(uint32_t) * probeSlots.size(), probeSlots.data(), "Buffer: probe slots");
  lvk::Holder<lvk::BufferHandle> probeCellsBuffer =
      createStorageBuffer(sizeof(uint32_t) * probeCells.size(), probeCells.data(), "Buffer: probe cells");

  lvk::Holder<lvk::BufferHandle> nlgPoints;
  lvk::Holder<lvk::BufferHandle> nlgNeighbours;
  lvk::Holder<lvk::BufferHandle> nlgBakePoints;
  lvk::Holder<lvk::BufferHandle> nlgSupport;
  lvk::Holder<lvk::BufferHandle> nlgPhiA;
  lvk::Holder<lvk::BufferHandle> nlgPhiB;
  lvk::Holder<lvk::BufferHandle> nlgScratch;
  lvk::Holder<lvk::BufferHandle> nlgLabels;
  lvk::Holder<lvk::BufferHandle> nlgBakeLight;
  lvk::Holder<lvk::BufferHandle> nlgShares;
  lvk::Holder<lvk::BufferHandle> nlgCoefficients;
  lvk::Holder<lvk::BufferHandle> nlgNetworks;
  lvk::Holder<lvk::BufferHandle> nlgBakeOrder;
  lvk::Holder<lvk::BufferHandle> nlgSh;
  lvk::Holder<lvk::BufferHandle> nlgFlags;
  lvk::Holder<lvk::BufferHandle> nlgPcaMean;
  lvk::Holder<lvk::BufferHandle> nlgPcaBasis;
  lvk::Holder<lvk::BufferHandle> nlgWeights;
  lvk::Holder<lvk::BufferHandle> nlgStruct;
  lvk::Holder<lvk::BufferHandle> nlgGraphInput;
  lvk::Holder<lvk::BufferHandle> nlgGraphOutput;
  lvk::Holder<lvk::BufferHandle> nlgOutputScale;
  lvk::Holder<lvk::TensorHandle> nlgTensorInput;
  lvk::Holder<lvk::TensorHandle> nlgTensorOutput;
  lvk::Holder<lvk::DataGraphPipelineHandle> pipelineGraph;
  lvk::Holder<lvk::BufferHandle> nlgCooperativeInput;
  lvk::Holder<lvk::BufferHandle> nlgCooperativeActivations[2];
  lvk::Holder<lvk::BufferHandle> nlgCooperativeStream;
  lvk::Holder<lvk::BufferHandle> nlgCooperativePartials;
  lvk::Holder<lvk::BufferHandle> nlgCooperativeWeights;
  lvk::Holder<lvk::BufferHandle> nlgCooperativeBiases;
  const size_t nlgShSize = sizeof(vec4) * 9 * numProbes;
  const size_t nlgFlagsSize = sizeof(uint32_t) * numProbes;

  if (nlgReady) {
    nlgPoints = createStorageBuffer(sizeof(vec4) * kSupportPoints, generator.points.data(), "Buffer: NLG support points");
    nlgNeighbours =
        createStorageBuffer(sizeof(uint32_t) * generator.neighbours.size(), generator.neighbours.data(), "Buffer: NLG neighbours");
    nlgBakePoints = createStorageBuffer(sizeof(uint32_t) * kBakePoints, generator.bakePoints.data(), "Buffer: NLG bake points");
    nlgSupport = createStorageBuffer(sizeof(uint32_t) * 2 * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG support visibility");
    nlgPhiA = createStorageBuffer(sizeof(float) * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG functions A");
    nlgPhiB = createStorageBuffer(sizeof(float) * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG functions B");
    nlgScratch = createStorageBuffer(sizeof(vec4) * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG lighting of the support");
    nlgLabels = createStorageBuffer(sizeof(uint32_t) * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG labels");
    nlgBakeLight = createStorageBuffer(sizeof(vec4) * kBakePoints * kBakeBatch, nullptr, "Buffer: NLG bake point lighting");
    nlgShares = createStorageBuffer(sizeof(uint32_t) * (kBakePoints / 2) * (numProbes + 1), nullptr, "Buffer: NLG shares");
    nlgCoefficients = createStorageBuffer(sizeof(float) * kCoefficients * (numProbes + 1), nullptr, "Buffer: NLG coefficients");
    nlgBakeOrder = createStorageBuffer(sizeof(uint32_t) * bakeOrder.size(), bakeOrder.data(), "Buffer: NLG bake order");
    nlgNetworks = createStorageBuffer(sizeof(uint32_t) * 4 * kNetworkVec4s * numProbes, nullptr, "Buffer: NLG networks (f16)");
    nlgSh = createStorageBuffer(nlgShSize, nullptr, "Buffer: NLG lighting");
    nlgFlags = createStorageBuffer(nlgFlagsSize + sizeof(uint32_t), nullptr, "Buffer: NLG flags");
    nlgPcaMean = createStorageBuffer(sizeof(float) * kSupportPoints, generator.mean.data(), "Buffer: NLG PCA mean");
    nlgPcaBasis = createStorageBuffer(sizeof(uint32_t) * generator.basis.size(), generator.basis.data(), "Buffer: NLG PCA basis");
    nlgWeights = createStorageBuffer(sizeof(float) * generator.weights.size(), generator.weights.data(), "Buffer: NLG generator");
    nlgOutputScale = createStorageBuffer(sizeof(float) * kCoefficients, generator.outputScale.data(), "Buffer: NLG output scale");
    if (graphReady) {
      const lvk::TensorDesc descInput = {.format = lvk::Format_R_I8, .rank = 4, .dimensions = {1, kBakeBatch, 1, kSupportPoints}};
      const lvk::TensorDesc descOutput = {.format = lvk::Format_R_I32, .rank = 4, .dimensions = {1, kBakeBatch, 1, kCoefficients}};
      lvk::Result result;
      nlgTensorInput = createTensorWithBuffer(ctx, descInput, nlgGraphInput, "NLG: graph input", &result);
      if (result.isOk()) {
        nlgTensorOutput = createTensorWithBuffer(ctx, descOutput, nlgGraphOutput, "NLG: graph output", &result);
      }
      if (result.isOk()) {
        logStartup("creating the generator's data graph");
        const std::vector<uint32_t> spirv = buildGeneratorGraph(generator);
        const lvk::Holder<lvk::ShaderModuleHandle> smGraph =
            ctx->createShaderModule({spirv.data(), spirv.size() * sizeof(uint32_t), lvk::Stage_DataGraph, "Shader Module: NLG generator"});
        std::vector<lvk::DataGraphConstant> graphConstants;
        for (uint32_t i = 0; i != generator.graphLayers.size(); i++) {
          const GraphLayer& layer = generator.graphLayers[i];
          graphConstants.push_back({.id = 2 * i,
                                    .format = lvk::Format_R_I8,
                                    .rank = 4,
                                    .dimensions = {layer.numOutputs, 1, 1, layer.numInputs},
                                    .data = layer.weights.data()});
          graphConstants.push_back(
              {.id = 2 * i + 1, .format = lvk::Format_R_I32, .rank = 1, .dimensions = {layer.numOutputs}, .data = layer.biases.data()});
        }
        // the pipeline is described by the tensors it gets bound to, which may be optimally tiled copies of the buffer-aliasing ones
        const lvk::TensorDesc boundInput = ctx->getTensorDesc(nlgTensorInput);
        const lvk::TensorDesc boundOutput = ctx->getTensorDesc(nlgTensorOutput);
        pipelineGraph = ctx->createDataGraphPipeline(
            {
                .smGraph = smGraph,
                .inputs = {&boundInput, 1},
                .outputs = {&boundOutput, 1},
                .constants = {graphConstants.data(), graphConstants.size()},
                .debugName = "NLG: generator",
            },
            &result);
      }
      if (!result.isOk()) {
        LLOGW("Cannot create the generator's data graph (%s): it runs as a compute shader\n", result.message);
        graphReady = false;
      }
    }
    if (cooperativeReady) {
      // the float layers as f16 [inputs][outputs], the output layer padded to kCooperativeOutputs, and their f32 biases, a
      // layer every kGeneratorWidth
      std::vector<uint16_t> weights(size_t(kGeneratorLayers) * kGeneratorWidth * kGeneratorWidth + kGeneratorWidth * kCooperativeOutputs);
      std::vector<float> biases(size_t(kGeneratorLayers + 1) * kGeneratorWidth);
      size_t offset = 0;
      for (uint32_t layer = 0; layer != kGeneratorLayers + 1; layer++) {
        const uint32_t outputs = layer == kGeneratorLayers ? kCoefficients : kGeneratorWidth;
        const uint32_t padded = layer == kGeneratorLayers ? kCooperativeOutputs : kGeneratorWidth;
        uint16_t* dst = weights.data() + size_t(layer) * kGeneratorWidth * kGeneratorWidth;
        for (uint32_t i = 0; i != kGeneratorWidth; i++) {
          for (uint32_t o = 0; o != outputs; o++) {
            dst[i * padded + o] = glm::packHalf1x16(generator.weights[offset + i * outputs + o]);
          }
        }
        offset += size_t(kGeneratorWidth) * outputs;
        std::copy(
            generator.weights.begin() + offset, generator.weights.begin() + offset + outputs, biases.begin() + layer * kGeneratorWidth);
        offset += outputs;
      }
      nlgCooperativeInput =
          createStorageBuffer(sizeof(uint16_t) * kSupportPoints * kBakeBatch, nullptr, "Buffer: NLG generator input (f16)");
      nlgCooperativeActivations[0] =
          createStorageBuffer(sizeof(uint16_t) * kGeneratorWidth * kBakeBatch, nullptr, "Buffer: NLG generator activations 0 (f16)");
      nlgCooperativeActivations[1] =
          createStorageBuffer(sizeof(uint16_t) * kGeneratorWidth * kBakeBatch, nullptr, "Buffer: NLG generator activations 1 (f16)");
      nlgCooperativeStream = createStorageBuffer(sizeof(float) * kGeneratorWidth * kBakeBatch, nullptr, "Buffer: NLG generator stream");
      nlgCooperativePartials = createStorageBuffer(
          sizeof(float) * kGeneratorWidth * kBakeBatch * kCooperativePcaSplits, nullptr, "Buffer: NLG generator partial sums");
      nlgCooperativeWeights = createStorageBuffer(sizeof(uint16_t) * weights.size(), weights.data(), "Buffer: NLG generator (f16)");
      nlgCooperativeBiases = createStorageBuffer(sizeof(float) * biases.size(), biases.data(), "Buffer: NLG generator biases");
    }
    const NlgBuffers addresses = {
        .points = ctx->gpuAddress(nlgPoints),
        .neighbours = ctx->gpuAddress(nlgNeighbours),
        .bakePoints = ctx->gpuAddress(nlgBakePoints),
        .support = ctx->gpuAddress(nlgSupport),
        .phiA = ctx->gpuAddress(nlgPhiA),
        .phiB = ctx->gpuAddress(nlgPhiB),
        .histogram = ctx->gpuAddress(nlgPhiB),
        .scratch = ctx->gpuAddress(nlgScratch),
        .labels = ctx->gpuAddress(nlgLabels),
        .bakeLight = ctx->gpuAddress(nlgBakeLight),
        .shares = ctx->gpuAddress(nlgShares),
        .coefficients = ctx->gpuAddress(nlgCoefficients),
        .sh = ctx->gpuAddress(nlgSh),
        .flags = ctx->gpuAddress(nlgFlags),
        .pcaMean = ctx->gpuAddress(nlgPcaMean),
        .pcaBasis = ctx->gpuAddress(nlgPcaBasis),
        .weights = ctx->gpuAddress(nlgWeights),
        .graphInput = graphReady ? ctx->gpuAddress(nlgGraphInput) : 0ull,
        .graphOutput = graphReady ? ctx->gpuAddress(nlgGraphOutput) : 0ull,
        .outputScale = ctx->gpuAddress(nlgOutputScale),
        .networks = ctx->gpuAddress(nlgNetworks),
        .bakeOrder = ctx->gpuAddress(nlgBakeOrder),
    };
    nlgStruct = createStorageBuffer(sizeof(NlgBuffers), &addresses, "Buffer: NLG");
  }

  // the data graph if it is available, else the compute shader, unless `--generator` names another: the cooperative matrices
  // only run when asked for, as some drivers expose them without the hardware (an AMD RDNA2 iGPU takes seconds per batch)
  const bool pathReady[] = {nlgReady, graphReady, cooperativeReady};
  int generatorPath = graphReady ? GeneratorPath_DataGraph : GeneratorPath_Compute;
  generatorPath = preferredPath >= 0 && pathReady[preferredPath] ? preferredPath : generatorPath;
  if (nlgReady) {
    const char* names[] = {"a compute shader", "a data graph", "cooperative matrices"};
    LLOGL("Neural Light Grid: the generator runs as %s\n", names[generatorPath]);
  }

  GPUStageTimer gpuTimer;
  gpuTimer.create(ctx, GPUTimestamp_NUM);

  const mat4 proj = glm::perspective(glm::radians(kCameraFov), float(renderWidth) / float(renderHeight), kCameraNear, kCameraFar);

  const lvk::Dimensions groupsFull = {(renderWidth + 7) / 8, (renderHeight + 7) / 8, 1};

  auto renderGBuffer = [&](lvk::ICommandBuffer& buf) {
    buf.cmdBeginRendering(
        lvk::RenderPass{
            .color = {
                {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}},
                {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}},
                {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}},
                {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 1.0f, 1.0f}},
                {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}},
            },
            .depth = {.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .clearDepth = 1.0f},
        },
        lvk::Framebuffer{
            .color = {{.texture = texPosition},
                      {.texture = texNormal},
                      {.texture = texAlbedo},
                      {.texture = texShading},
                      {.texture = texMaterial}},
            .depthStencil = {.texture = texDepth},
        });
    buf.cmdBindRenderPipeline(pipelineGBuffer);
    buf.cmdBindDepthState({.compareOp = VK_COMPARE_OP_LESS, .isDepthWriteEnabled = true});
    buf.cmdBindVertexBuffer(0, vb0);
    buf.cmdBindIndexBuffer(ib0, VK_INDEX_TYPE_UINT32);
    buf.cmdPushConstants(GBufferPushConstants{
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .materials = ctx->gpuAddress(sbMaterials),
        .models = ctx->gpuAddress(sbMaterialModels),
        .sampler0 = samplerMaterials.index(),
    });
    buf.cmdDrawIndexed(numIndices);
    buf.cmdEndRendering();
  };

  auto direct = [&](lvk::ICommandBuffer& buf) {
    buf.cmdBindComputePipeline(programDirect.pipeline);
    buf.cmdPushConstants(DirectPushConstants{
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .materials = ctx->gpuAddress(sbMaterials),
        .vertices = ctx->gpuAddress(vb0),
        .indices = ctx->gpuAddress(ibTrace),
        .tlas = accel.TLAS.index(),
        .sampler0 = samplerMaterials.index(),
        .texPosition = texPosition.index(),
        .texNormal = texNormal.index(),
        .imageDirect = texDirect.index(),
        .texMaterial = texMaterial.index(),
    });
    buf.cmdDispatch(groupsFull, {.sampledImages = {texPosition, texNormal, texMaterial}, .storageImages = {texDirect}});
  };

  // `frameId` seeds the ray rotations of the lighting passes; the bake always uses 0
  auto nlgPushConstants = [&](uint32_t first, uint32_t count, uint32_t mode, float blend = 0.0f, uint32_t frameId = 0) {
    return NlgPushConstants{
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .materials = ctx->gpuAddress(sbMaterials),
        .vertices = ctx->gpuAddress(vb0),
        .indices = ctx->gpuAddress(ibTrace),
        .slots = ctx->gpuAddress(probeSlotsBuffer),
        .cells = ctx->gpuAddress(probeCellsBuffer),
        .nlg = ctx->gpuAddress(nlgStruct),
        .tlas = accel.TLAS.index(),
        .sampler0 = samplerMaterials.index(),
        .frameId = frameId,
        .first = first,
        .count = count,
        .mode = mode,
        .blend = blend,
    };
  };

  auto functions = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count, uint32_t mode) {
    buf.cmdBindComputePipeline(programFunctions.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, mode));
    buf.cmdDispatch({count, 1, 1},
                    {.buffers = {nlgSupport, nlgBakeLight, nlgPhiA, nlgPhiB, nlgScratch, nlgLabels, nlgShares, nlgFlags, nlgCoefficients}});
  };

  auto generateCooperative = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count) {
    buf.cmdBindComputePipeline(programCooperativeInput.pipeline);
    buf.cmdPushConstants(GeneratorCooperativePushConstants{
        .outputs = ctx->gpuAddress(nlgCooperativeInput),
        .nlg = ctx->gpuAddress(nlgStruct),
        .count = count,
    });
    buf.cmdDispatch({(count * kSupportPoints + 255) / 256, 1, 1}, {.buffers = {nlgPhiA, nlgCooperativeInput}});
    // the PCA, the hidden layers and the output layer, each the partial sums over slices of k, then their sum finished into its
    // outputs; the activations alternate between two buffers
    for (uint32_t layer = 0; layer != kGeneratorLayers + 2; layer++) {
      const bool isPca = layer == 0;
      const bool isOutput = layer == kGeneratorLayers + 1;
      const uint32_t hidden = isPca ? 0 : layer - 1; // the index of the float layer, the output layer last
      const lvk::BufferHandle inputs = isPca ? nlgCooperativeInput : nlgCooperativeActivations[hidden % 2];
      const lvk::BufferHandle outputs = isOutput ? nlgCoefficients : nlgCooperativeActivations[layer % 2];
      const uint32_t k = isPca ? kSupportPoints : kGeneratorWidth;
      const uint32_t n = isOutput ? kCooperativeOutputs : kGeneratorWidth;
      const uint32_t mode = isPca ? kCooperativePca : isOutput ? kCooperativeOutput : layer == 1 ? kCooperativeFirst : kCooperativeSkip;
      const GeneratorCooperativePushConstants pc = {
          .inputs = ctx->gpuAddress(inputs),
          .weights = isPca ? ctx->gpuAddress(nlgPcaBasis)
                           : ctx->gpuAddress(nlgCooperativeWeights, sizeof(uint16_t) * kGeneratorWidth * kGeneratorWidth * hidden),
          .biases = ctx->gpuAddress(nlgCooperativeBiases, sizeof(float) * kGeneratorWidth * hidden),
          .stream = ctx->gpuAddress(nlgCooperativeStream),
          .partials = ctx->gpuAddress(nlgCooperativePartials),
          .outputs = ctx->gpuAddress(outputs),
          .nlg = ctx->gpuAddress(nlgStruct),
          .first = first,
          .count = count,
          .k = k,
          .n = n,
          .mode = mode,
          .splits = uint32_t(isPca ? kCooperativePcaSplits : kCooperativeSplits),
      };
      const uint32_t blockM = tile.m * kCooperativeBlock;
      const uint32_t jobs = (count + blockM - 1) / blockM * (n / (tile.n * kCooperativeBlock)) * pc.splits;
      buf.cmdBindComputePipeline(programCooperative.pipeline);
      buf.cmdPushConstants(pc);
      buf.cmdDispatch({(jobs + kCooperativeTilesPerGroup - 1) / kCooperativeTilesPerGroup, 1, 1},
                      {.buffers = {inputs, nlgCooperativePartials}});
      buf.cmdBindComputePipeline(programCooperativeFinish.pipeline);
      buf.cmdPushConstants(pc);
      buf.cmdDispatch({(count * n + 255) / 256, 1, 1}, {.buffers = {nlgCooperativePartials, outputs, nlgCooperativeStream}});
    }
  };

  auto generate = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count) {
    if (generatorPath == GeneratorPath_Cooperative) {
      generateCooperative(buf, first, count);
      return;
    }
    if (generatorPath == GeneratorPath_DataGraph) {
      buf.cmdBindComputePipeline(programQuantize.pipeline);
      buf.cmdPushConstants(nlgPushConstants(first, count, 0));
      buf.cmdDispatch({(count * (kSupportPoints / 4) + 255) / 256, 1, 1}, {.buffers = {nlgPhiA, nlgGraphInput}});
      const lvk::TensorHandle inputs[] = {nlgTensorInput};
      const lvk::TensorHandle outputs[] = {nlgTensorOutput};
      buf.cmdDispatchDataGraph(pipelineGraph, inputs, outputs);
      buf.cmdBindComputePipeline(programDequantize.pipeline);
      buf.cmdPushConstants(nlgPushConstants(first, count, 0));
      buf.cmdDispatch({(count * kCoefficients + 255) / 256, 1, 1},
                      {.buffers = {nlgGraphOutput, nlgCoefficients, nlgOutputScale}, .tensors = {nlgTensorOutput}});
      return;
    }
    buf.cmdBindComputePipeline(programGenerator.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 0));
    buf.cmdDispatch({(count + kGeneratorProbes - 1) / kGeneratorProbes, 1, 1},
                    {.buffers = {nlgPhiA, nlgCoefficients, nlgPcaMean, nlgPcaBasis, nlgWeights}});
  };

  auto bakeBatch = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count) {
    buf.cmdBindComputePipeline(programSupport.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 0));
    buf.cmdDispatch({kSupportPoints / 64, count, 1},
                    {.buffers = {nlgSupport, nlgPoints, nlgNeighbours, probeSlotsBuffer, probeCellsBuffer}});
    buf.cmdBindComputePipeline(programLight.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 0));
    buf.cmdDispatch({count, 1, 1}, {.buffers = {nlgBakeLight, nlgFlags, nlgSh, nlgCoefficients}});
    functions(buf, first, count, 0);
    generate(buf, first, count);
  };

  auto bakeFreeBatch = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count, bool withReference) {
    buf.cmdBindComputePipeline(programLight.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 0));
    buf.cmdDispatch({count, 1, 1}, {.buffers = {nlgBakeLight, nlgFlags, nlgSh, nlgCoefficients}});
    if (withReference) {
      functions(buf, numProbes, 1, 2);
      generate(buf, numProbes, 1);
    }
    functions(buf, first, count, 1);
  };

  auto pack = [&](lvk::ICommandBuffer& buf, uint32_t first, uint32_t count) {
    buf.cmdBindComputePipeline(programPack.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 0));
    buf.cmdDispatch({(count * kNetworkVec4s + 63) / 64, 1, 1}, {.buffers = {nlgCoefficients, nlgNetworks}});
  };

  auto bakeNext = [&](lvk::ICommandBuffer& buf, uint32_t first) -> uint32_t {
    const bool complex = first < numComplexProbes;
    const uint32_t count = std::min<uint32_t>(kBakeBatch, (complex ? numComplexProbes : numProbes) - first);
    if (complex) {
      bakeBatch(buf, first, count);
    } else {
      bakeFreeBatch(buf, first, count, first == numComplexProbes);
    }
    pack(buf, first, count);
    return count;
  };

  auto updateNlg = [&](lvk::ICommandBuffer& buf, uint32_t frameId, uint32_t first, uint32_t count, float blend) {
    buf.cmdBindComputePipeline(programLight.pipeline);
    buf.cmdPushConstants(nlgPushConstants(first, count, 1, blend, frameId));
    buf.cmdDispatch({count, 1, 1}, {.buffers = {nlgSh, nlgFlags, nlgShares, nlgNetworks, nlgBakeLight}});
  };

  auto apply = [&](lvk::ICommandBuffer& buf) {
    buf.cmdBindComputePipeline(programApply.pipeline);
    buf.cmdPushConstants(ApplyPushConstants{
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .slots = ctx->gpuAddress(probeSlotsBuffer),
        .cells = ctx->gpuAddress(probeCellsBuffer),
        .nlg = ctx->gpuAddress(nlgStruct),
        .texPosition = texPosition.index(),
        .texNormal = texNormal.index(),
        .imageIndirect = texIndirect.index(),
        .texShading = texShading.index(),
        .imageIndirectSpec = texIndirectSpec.index(),
        .texMaterial = texMaterial.index(),
    });
    buf.cmdDispatch(groupsFull,
                    {.sampledImages = {texPosition, texNormal, texShading, texMaterial},
                     .storageImages = {texIndirect, texIndirectSpec},
                     .buffers = {nlgSh, nlgFlags, nlgNetworks}});
  };

  auto perFrameFor = [&](const mat4& viewMatrix, float azimuth, float elevation) {
    return PerFrame{
        .proj = proj,
        .view = viewMatrix,
        .cameraPos = glm::inverse(viewMatrix)[3],
        .sunDir = vec4(sunDirection(azimuth, elevation), 0.0f),
        .sunColor = vec4(kSunColor, 0.0f),
        .skyColor = vec4(kSkyColor, 0.0f),
    };
  };

  uint32_t bakedProbes = 0;

  // the bake depends on the textures (alpha testing, the albedo of the bounce), so a textured and an untextured one are kept
  bool bakeTextured = false;
  auto bakeFilePath = [&]() -> std::string {
    return app.folderContentRoot_ + "nlg_" + scene.name + (bakeTextured ? "" : "_untextured") + ".bake";
  };
  const uint32_t networkWords = 4 * kNetworkVec4s * numProbes;
  const uint32_t shareWords = (kBakePoints / 2) * numProbes;
  // the baked networks depend on the sun they were baked under: k-means over the baked lighting picks their inputs
  float bakeAzimuth = 0.0f;
  float bakeElevation = 0.0f;
  auto bakeKey = [&]() {
    const uint32_t constants[] = {kBakeFileVersion,
                                  numProbes,
                                  kSupportPoints,
                                  kBakePoints,
                                  kCoefficients,
                                  uint32_t(generatorPath),
                                  bakeTextured ? 1u : 0u,
                                  numVertices,
                                  numIndices};
    const float sun[] = {bakeAzimuth, bakeElevation};
    uint64_t h = hashBytes(14695981039346656037ull, constants, sizeof(constants));
    h = hashBytes(h, sun, sizeof(sun));
    h = hashBytes(h, scene.name, strlen(scene.name));
    h = hashVector(h, levels);
    h = hashVector(h, probeCells);
    h = hashVector(h, generator.points);
    h = hashVector(h, generator.bakePoints);
    h = hashVector(h, generator.weights);
    h = hashVector(h, generator.outputScale);
    return h;
  };

  auto countBakeStats = [&](const std::vector<uint32_t>& flags) {
    uint32_t probesInside = 0;
    uint32_t probesUnlit = 0;
    for (uint32_t i = 0; i != numProbes; i++) {
      const uint32_t f = flags[i];
      probesInside += (f & kProbeInside) ? 1 : 0;
      probesUnlit += (f & kProbeHasLighting) ? 0 : 1;
    }
    LLOGL("Neural Light Grid: %u probes baked, %u inside geometry, %u without lighting\n", numProbes, probesInside, probesUnlit);
  };

  auto saveBake = [&]() {
    std::vector<uint32_t> flags(numProbes);
    std::vector<uint32_t> networks(networkWords);
    std::vector<uint32_t> shares(shareWords);
    lvk::Holder<lvk::BufferHandle> readback = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_HostVisible,
        .size = sizeof(uint32_t) * std::max(networkWords, shareWords),
        .debugName = "Buffer: NLG bake readback",
    });
    auto readBack = [&](lvk::BufferHandle src, std::vector<uint32_t>& dst) {
      lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();
      buf.cmdCopyBuffer(src, readback, 0, 0, sizeof(uint32_t) * dst.size());
      ctx->wait(ctx->submit(buf));
      ctx->download(readback, dst.data(), sizeof(uint32_t) * dst.size(), 0);
    };
    readBack(nlgFlags, flags);
    readBack(nlgNetworks, networks);
    readBack(nlgShares, shares);
    countBakeStats(flags);
    const std::string path = bakeFilePath();
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) {
      LLOGW("Cannot write `%s`\n", path.c_str());
      return;
    }
    const BakeFileHeader header = {
        .magic = kBakeFileMagic,
        .version = kBakeFileVersion,
        .key = bakeKey(),
        .numProbes = numProbes,
        .networkWords = networkWords,
        .shareWords = shareWords,
    };
    const bool written = fwrite(&header, sizeof(header), 1, file) == 1 && fwrite(flags.data(), nlgFlagsSize, 1, file) == 1 &&
                         fwrite(networks.data(), sizeof(uint32_t) * networkWords, 1, file) == 1 &&
                         fwrite(shares.data(), sizeof(uint32_t) * shareWords, 1, file) == 1;
    fclose(file);
    if (written) {
      LLOGL("Saved the baked probes to `%s`\n", path.c_str());
    } else {
      LLOGW("Cannot write `%s`\n", path.c_str());
      remove(path.c_str());
    }
  };

  auto loadBake = [&]() -> bool {
    if (rebake) {
      return false;
    }
    const std::string path = bakeFilePath();
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
      return false;
    }
    BakeFileHeader header = {};
    std::vector<uint32_t> flags(numProbes);
    std::vector<uint32_t> networks(networkWords);
    std::vector<uint32_t> shares(shareWords);
    const bool valid = fread(&header, sizeof(header), 1, file) == 1 && header.magic == kBakeFileMagic &&
                       header.version == kBakeFileVersion && header.key == bakeKey() && header.numProbes == numProbes &&
                       header.networkWords == networkWords && header.shareWords == shareWords &&
                       fread(flags.data(), nlgFlagsSize, 1, file) == 1 &&
                       fread(networks.data(), sizeof(uint32_t) * networkWords, 1, file) == 1 &&
                       fread(shares.data(), sizeof(uint32_t) * shareWords, 1, file) == 1;
    fclose(file);
    if (!valid) {
      LLOGW("`%s` is from another lattice, generator or sun: baking again\n", path.c_str());
      return false;
    }
    ctx->upload(nlgFlags, flags.data(), nlgFlagsSize, 0);
    ctx->upload(nlgNetworks, networks.data(), sizeof(uint32_t) * networkWords, 0);
    ctx->upload(nlgShares, shares.data(), sizeof(uint32_t) * shareWords, 0);
    LLOGL("Loaded the baked probes from `%s`\n", path.c_str());
    countBakeStats(flags);
    bakedProbes = numProbes;
    return true;
  };

  bool nlgStale = true;
  // the textured bake starts from the untextured probes: their networks, flags and lighting stay until it replaces them
  bool nlgKeepProbes = false;
  bool nlgLitOnce = false;
  uint32_t nlgUpdates = 0; // probes lit since the lighting started over
  uint32_t nlgCursor = 0; // the lighting goes over the baked probes in the bake order, so it follows the bake
  const uint32_t nlgConvergeUpdates = kNlgConvergePasses * numProbes;

  // the lighting gets as many probes per frame as fit into kNlgTimeBoxUs, after its GPU time per probe
  lvk::Holder<lvk::QueryPoolHandle> nlgTimingQueries = ctx->createQueryPool(2 * kNlgTimingSlots, "Query pool: NLG lighting");
  uint32_t nlgTimedProbes[kNlgTimingSlots] = {};
  uint32_t nlgBudget = kNlgBudget;
  double nlgMsPerProbe = 0.0;

  // the textured lighting averages into the untextured one, which already has the bounces the warm-up passes build up
  bool nlgAverageOld = false;
  auto nlgBlend = [&](uint32_t pass) {
    return nlgAverageOld ? 1.0f / float(pass + 2) : pass < kNlgWarmupPasses ? 1.0f : 1.0f / float(pass - kNlgWarmupPasses + 2);
  };
  // bakes the probes again, the old ones lighting the scene until the new ones replace them
  auto rebakeOverOldProbes = [&]() {
    nlgStale = true;
    nlgKeepProbes = bakedProbes != 0;
    nlgAverageOld = bakedProbes != 0;
    nlgUpdates = 0;
    nlgLitOnce = false;
  };

  uint64_t frameIndex = 0;
  float lastAzimuth = -1.0f;
  float lastElevation = -1.0f;

#if !defined(ANDROID)
#if LVK_WITH_GLFW
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    g_ToggleGI |= key == GLFW_KEY_G && action == GLFW_PRESS;
    g_ToggleTextures |= key == GLFW_KEY_T && action == GLFW_PRESS;
  });
#elif LVK_WITH_SDL3
  app.addKeyCallback([](SDL_Window* window, SDL_KeyboardEvent* event) {
    g_ToggleGI |= event->key == SDLK_G && event->down;
    g_ToggleTextures |= event->key == SDLK_T && event->down;
  });
#endif
#endif // !ANDROID

  logStartup("done");

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    showGI ^= g_ToggleGI;
    showTextures ^= g_ToggleTextures;
    g_ToggleGI = false;
    g_ToggleTextures = false;

    // start streaming textures in asynchronously the first time they are enabled
    if (showTextures && !texturesRequested) {
      texturesRequested = true;
      loadMaterialTextures(app, scene.textureDir, true);
    }

    splitOpaqueGeometry();

    const mat4 viewMatrix = app.camera_.getViewMatrix();
    const float azimuth = glm::radians(sunAzimuth);
    const float elevation = glm::radians(sunElevation);
    const uint32_t remainingTextures = numRemainingMaterialTextures();

    // once the textures are in, the untextured bake is replaced by a textured one
    if (texturesRequested && !remainingTextures && !bakeTextured) {
      bakeTextured = true;
      rebakeOverOldProbes();
    }

    // moving the sun before the bake is done restarts the bake: the baked networks depend on the sun
    if (azimuth != lastAzimuth || elevation != lastElevation) {
      nlgUpdates = 0;
      nlgAverageOld = false;
      nlgStale = nlgStale || bakedProbes < numProbes;
    }
    lastAzimuth = azimuth;
    lastElevation = elevation;

    const bool nlgConverging = bakedProbes != 0 && (bakedProbes < numProbes || nlgUpdates < nlgConvergeUpdates);

    // read back the GPU time of the lighting kNlgTimingSlots frames ago, long finished by now
    const uint32_t timingSlot = frameIndex % kNlgTimingSlots;
    if (nlgTimedProbes[timingSlot]) {
      uint64_t timestamps[2] = {};
      ctx->getQueryPoolResults(nlgTimingQueries, 2 * timingSlot, 2, sizeof(timestamps), timestamps, sizeof(timestamps[0]));
      const double msPerProbe = double(timestamps[1] - timestamps[0]) * ctx->getTimestampPeriodToMs() / double(nlgTimedProbes[timingSlot]);
      // a slower part of the lattice raises the estimate at once, a faster one lowers it gradually; the cost of the next
      // probes varies, so aim at 3/4 of the time box
      nlgMsPerProbe = std::max(msPerProbe, glm::mix(nlgMsPerProbe, msPerProbe, 0.05));
      const double fit = 0.00075 * double(kNlgTimeBoxUs) / std::max(nlgMsPerProbe, 1e-9);
      nlgBudget = std::clamp(uint32_t(std::min(fit, 2.0 * nlgBudget)), uint32_t(kNlgBudgetMin), uint32_t(kNlgBudgetMax));
      nlgTimedProbes[timingSlot] = 0;
    }

    lvk::ICommandBuffer& buf = ctx->acquireCommandBuffer();

    processLoadedMaterialTextures(buf, sbMaterials);

    buf.cmdUpdateBuffer(bufPerFrame, perFrameFor(viewMatrix, azimuth, elevation));

    gpuTimer.beginFrame(buf, frameIndex);
    auto timestamp = [&](GPUTimestamp t) { gpuTimer.timestamp(buf, t); };

    buf.cmdPushDebugGroupLabel("G-buffer", 0xff0000ff);
    renderGBuffer(buf);
    buf.cmdPopDebugGroupLabel();

    timestamp(GPUTimestamp_GBuffer);

    if (hasRayQuery) {
      buf.cmdPushDebugGroupLabel("Direct sunlight", 0xff00ffff);
      direct(buf);
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_Direct);

    const bool canBake = nlgReady && remainingTextures == 0;

    if (canBake && nlgStale) {
      if (!nlgKeepProbes) {
        buf.cmdFillBuffer(nlgSh, 0, nlgShSize, 0);
      }
      bakedProbes = 0;
      bakeAzimuth = azimuth;
      bakeElevation = elevation;
      nlgStale = false;
      if (!loadBake() && !nlgKeepProbes) {
        buf.cmdFillBuffer(nlgFlags, 0, nlgFlagsSize, kProbeNotBaked);
      }
    }

    if (canBake && nlgConverging && bakedProbes) {
      buf.cmdPushDebugGroupLabel("Neural Light Grid lighting", 0xffff00ff);
      nlgCursor = nlgCursor < bakedProbes ? nlgCursor : 0;
      const uint32_t count = std::min(nlgBudget, bakedProbes - nlgCursor);
      buf.cmdResetQueryPool(nlgTimingQueries, 2 * timingSlot, 2);
      buf.cmdWriteTimestamp(nlgTimingQueries, 2 * timingSlot);
      updateNlg(buf, (uint32_t)frameIndex, nlgCursor, count, nlgBlend(nlgUpdates / numProbes));
      buf.cmdWriteTimestamp(nlgTimingQueries, 2 * timingSlot + 1);
      nlgTimedProbes[timingSlot] = count;
      nlgCursor = (nlgCursor + count) % bakedProbes;
      nlgUpdates += count;
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_Lighting);

    bool bakeCompleted = false;

    if (canBake && bakedProbes < numProbes) {
      buf.cmdPushDebugGroupLabel("Neural Light Grid bake", 0xffff8000);
      bakedProbes += bakeNext(buf, bakedProbes);
      bakeCompleted = bakedProbes == numProbes;
      if (bakeCompleted) {
        nlgUpdates = 0;
      }
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_Bake);

    nlgKeepProbes = nlgKeepProbes && bakedProbes < numProbes;
    const bool hasIndirect = nlgReady && (bakedProbes != 0 || nlgKeepProbes);

    if (hasIndirect) {
      buf.cmdPushDebugGroupLabel("Neural Light Grid", 0xff00ff00);
      apply(buf);
      buf.cmdPopDebugGroupLabel();
    }

    timestamp(GPUTimestamp_GI);

    const lvk::Dimensions dimSwapchain = ctx->getDimensions(views[0].colorTexture);
    const lvk::Framebuffer framebuffer = {.color = {{.texture = views[0].colorTexture}}};

    const PresentPushConstants presentPushConstants = {
        .perFrame = ctx->gpuAddress(bufPerFrame),
        .exposureBuffer = ctx->gpuAddress(exposureBuffer),
        .clipRotation = getClipRotation(ctx),
        .texPosition = texPosition.index(),
        .texAlbedo = texAlbedo.index(),
        .texDirect = texDirect.index(),
        .texIndirect = texIndirect.index(),
        .texShading = texShading.index(),
        .texMaterial = texMaterial.index(),
        .texIndirectSpec = texIndirectSpec.index(),
        .gi = showGI && hasIndirect ? 1u : 0u,
        .textures = showTextures ? 1u : 0u,
        .exposure = autoExposure ? exp2f(exposureCompensation) : exposure,
        .autoExposure = autoExposure ? 1u : 0u,
        .exposureAdapt = 1.0f - expf(-deltaSeconds / (nlgLitOnce || !nlgReady ? kExposureTau : kExposureTauFast)),
    };
    lvk::TextureHandle presentInputs[] = {texIndirect, texIndirectSpec, texPosition, texAlbedo, texDirect, texShading, texMaterial};
    if (autoExposure) {
      buf.cmdBindComputePipeline(programExposure.pipeline);
      buf.cmdPushConstants(presentPushConstants);
      buf.cmdDispatch({1, 1, 1}, {.sampledImages = presentInputs, .buffers = {exposureBuffer}});
    }

    buf.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE}}},
                          framebuffer,
                          {.sampledImages = presentInputs, .buffers = {exposureBuffer}});
    buf.cmdBindRenderPipeline(pipelinePresent);
    buf.cmdPushConstants(presentPushConstants);
    buf.cmdDraw(3);

    app.imgui_->beginFrame(framebuffer);
    if (nlgReady && bakedProbes == numProbes && nlgUpdates >= nlgConvergeUpdates) {
      nlgLitOnce = true;
    }
    if (showUI) {
      drawMaterialTexturesLoadingProgress();
      if (nlgReady && !remainingTextures && !nlgLitOnce) {
        const bool baking = bakedProbes < numProbes;
        const float progress =
            std::clamp(baking ? float(bakedProbes) / float(numProbes) : float(nlgUpdates) / float(nlgConvergeUpdates), 0.0f, 1.0f);
        const ImVec2 size(ImGui::GetIO().DisplaySize.x, 32.0f);
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        drawList->AddRectFilled(ImVec2(0, 0), size, ImGui::GetColorU32(ImGuiCol_FrameBg));
        drawList->AddRectFilled(ImVec2(0, 0), ImVec2(size.x * progress, size.y), ImGui::GetColorU32(ImGuiCol_PlotHistogram));
        char text[96] = {};
        if (baking) {
          snprintf(text, sizeof(text), "Baking probes... %u / %u", bakedProbes, numProbes);
        } else {
          snprintf(text, sizeof(text), "Lighting probes... %.0f%%", progress * 100.0f);
        }
        const ImVec2 textSize = ImGui::CalcTextSize(text);
        drawList->AddText(ImVec2(0.5f * (size.x - textSize.x), 0.5f * (size.y - textSize.y)), ImGui::GetColorU32(ImGuiCol_Text), text);
      }
      ImGui::Begin("Neural Light Grid", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      if (!nlgReady) {
        ImGui::TextDisabled(hasRayQuery ? "the generator is not available" : "ray queries are not available");
      }
      ImGui::SeparatorText("Generator");
      // each is grayed out without what it runs on; the probes are baked again with the new one
      const int oldPath = generatorPath;
      const char* pathNames[] = {"Compute shader", "VK_ARM_tensors + VK_ARM_data_graph", "VK_KHR_cooperative_matrix"};
      for (int path = GeneratorPath_Compute; path <= GeneratorPath_Cooperative; path++) {
        ImGui::BeginDisabled(!pathReady[path]);
        ImGui::RadioButton(pathNames[path], &generatorPath, path);
        ImGui::EndDisabled();
      }
      if (generatorPath != oldPath) {
        rebakeOverOldProbes();
      }
      ImGui::Separator();
      ImGui::Checkbox("Enable GI", &showGI);
      ImGui::Checkbox("Enable textures", &showTextures);
      ImGui::Separator();
      ImGui::SliderFloat("Sun Elevation", &sunElevation, 10.0f, 85.0f, "%.0f deg");
      ImGui::SliderFloat("Sun Azimuth", &sunAzimuth, 0.0f, 360.0f, "%.0f deg");
      ImGui::Separator();
      ImGui::Checkbox("Auto exposure", &autoExposure);
      if (autoExposure) {
        ImGui::SliderFloat("Exposure, EV", &exposureCompensation, -4.0f, 4.0f);
      } else {
        ImGui::SliderFloat("Exposure", &exposure, 0.1f, 4.0f);
      }
      if (nlgReady && nlgLitOnce) {
        if (bakedProbes < numProbes) {
          ImGui::Text("Baked %u / %u probes%s", bakedProbes, numProbes, remainingTextures ? " (waiting for the textures)" : "");
        }
        if (nlgConverging) {
          ImGui::Text("Lighting: pass %u of %u, %u probes per frame",
                      std::min<uint32_t>(nlgUpdates / numProbes + 1, kNlgConvergePasses),
                      kNlgConvergePasses,
                      std::min(nlgBudget, numProbes));
          ImGui::Text("          %u rays at each of %u points", kBakeRays, kBakePoints);
        }
      }
      ImGui::End();
      const GPUStageTimer::Row rows[] = {
          {"Frame", GPUTimestamp_Begin, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)},
          {"G-buffer", GPUTimestamp_GBuffer, ImVec4(0.4f, 0.7f, 1.0f, 1.0f)},
          {"Sunlight", GPUTimestamp_Direct, ImVec4(1.0f, 0.85f, 0.3f, 1.0f)},
          {"Lighting", GPUTimestamp_Lighting, ImVec4(1.0f, 0.5f, 0.2f, 1.0f)},
          {"Bake", GPUTimestamp_Bake, ImVec4(1.0f, 0.4f, 0.9f, 1.0f)},
          {"GI", GPUTimestamp_GI, ImVec4(0.4f, 1.0f, 0.5f, 1.0f)},
          {"Present", GPUTimestamp_Present, ImVec4(0.75f, 0.6f, 1.0f, 1.0f)},
      };
      gpuTimer.drawWindow(rows);
      app.drawFPS();
    }
    app.imgui_->endFrame(buf);

    buf.cmdEndRendering();

    timestamp(GPUTimestamp_Present);

    const lvk::SubmitHandle submitHandle = ctx->submit(buf, views[0].colorTexture);

    if (bakeCompleted) {
      ctx->wait(submitHandle);
      saveBake();
    }

    gpuTimer.endFrame(ctx);
    if (measureTime && gpuTimer.isMeasuring && frameIndex % 5 == 0) {
      const double* stageTime = gpuTimer.stageTime;
      LLOGL("GPU time, ms: frame %.2f | G-buffer %.2f, sunlight %.2f, lighting %.2f, bake %.2f, GI %.2f, present %.2f\n",
            gpuTimer.frameTime,
            stageTime[GPUTimestamp_GBuffer],
            stageTime[GPUTimestamp_Direct],
            stageTime[GPUTimestamp_Lighting],
            stageTime[GPUTimestamp_Bake],
            stageTime[GPUTimestamp_GI],
            stageTime[GPUTimestamp_Present]);
    }

    frameIndex++;
  });

  ctx->wait({});

  cancelLoadingMaterialTextures();

  VULKAN_APP_EXIT();
}
