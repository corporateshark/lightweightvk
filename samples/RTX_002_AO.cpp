/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Bistro.h"
#include "VulkanApp.h"

// scene navigation
#define USE_SPONZA 0

#if USE_SPONZA
#define MODEL_PATH "src/Sponza/sponza.obj"
#define CACHE_FILE_NAME "cache3.data"
vec3 lightDir_ = normalize(vec3(-0.5f, 0.85f, -0.05f));
#else
#define MODEL_PATH "src/bistro/Exterior/exterior.obj"
#define CACHE_FILE_NAME "cache2.data"
vec3 lightDir_ = normalize(vec3(0.032f, 0.835f, 0.549f));
#endif

#if defined(ANDROID)
constexpr int kNumSamplesMSAA = 2;
constexpr float kFramebufferScalar = 1.0f;
#else
constexpr int kNumSamplesMSAA = 4;
constexpr int kFramebufferScalar = 1;
#endif // ANDROID

#if defined(ANDROID)
constexpr uint32_t kHashMapSize = 2 * 1024 * 1024; // 2M entries on mobile (24 MB across 3 buffers)
#else
constexpr uint32_t kHashMapSize = 8 * 1024 * 1024; // 8M entries on desktop (96 MB across 3 buffers)
#endif

const char* codeFullscreenSlang = R"(
struct PushConstants {
  uint tex;
};

[[vk::push_constant]] PushConstants pc;

struct VSOutput {
  float2 uv : TEXCOORD0;
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID) {
  VSOutput out;
  out.uv = float2((vertexID << 1) & 2, vertexID & 2);
  out.position = float4(out.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
  return out;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
  return textureBindless2D(pc.tex, 0, input.uv);
}
)";

const char* codeZPrepassSlang = R"(
struct PerFrame {
  float4x4 proj;
  float4x4 view;
};

struct PerObject {
  float4x4 model;
  float4x4 normal;
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  uint64_t padding; // materials
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
float4 vertexMain(float3 pos : POSITION) : SV_Position {
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;

  return proj * view * model * float4(pos, 1.0);
}

[shader("fragment")]
void fragmentMain() {
  // empty fragment shader for Z-prepass
}
)";

const char* codeSlang = R"(
struct Material {
  float4 ambient;
  float4 diffuse;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
  float4x4 light;
};

struct PerObject {
  float4x4 model;
  float4x4 normal;
};

struct Materials {
  Material mtl[];
};

struct AOHashChecksums { uint v[]; };
struct AOSpatialData { uint v[]; };
struct AOHashTime { uint v[]; };

[[vk::constant_id(0)]] const bool kEnableSpatialHash = true;

struct PushConstants {
  float4 lightDir;
  PerFrame* perFrame;
  PerObject* perObject;
  Materials* materials;
  uint tlas;
  bool enableShadows;
  bool enableAO;
  int aoSamples;
  float aoRadius;
  float aoPower;
  uint frameId;
  AOHashChecksums* hashChecksums;
  AOSpatialData* spatialData;
  AOHashTime* hashTime;
  float sp;
  float smin;
  uint maxSamples;
  uint hashMapSize;
  float resolutionY;
  bool enableFiltering;
};

[[vk::push_constant]] PushConstants pc;

struct PerVertex {
  float3 worldPos;
  float3 normal;
  float2 uv;
  float4 Ka;
  float4 Kd;
};

struct VSOutput {
  PerVertex vtx : TEXCOORD0;
  float4 position : SV_Position;
};

//
// https://www.shadertoy.com/view/llfcRl
float2 unpackSnorm2x8(uint d) {
  return float2(uint2(d, d >> 8) & 255u) / 127.5 - 1.0;
}

float3 unpackOctahedral16(uint data) {
  float2 v = unpackSnorm2x8(data);
  // https://x.com/Stubbesaurus/status/937994790553227264
  float3 n = float3(v, 1.0 - abs(v.x) - abs(v.y));
  float t = max(-n.z, 0.0);
  n.x += (n.x > 0.0) ? -t : t;
  n.y += (n.y > 0.0) ? -t : t;
  return normalize(n);
}

[shader("vertex")]
VSOutput vertexMain(
  float3 pos : POSITION,
  float2 uv : TEXCOORD0,
  uint normal : TEXCOORD1,
  uint mtlIndex : TEXCOORD2
) {
  VSOutput out;

  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;

  out.position = proj * view * model * float4(pos, 1.0);

  // compute the normal in world-space
  out.vtx.worldPos = (model * float4(pos, 1.0)).xyz;
  out.vtx.normal = normalize((float3x3)pc.perObject->normal * unpackOctahedral16(normal));
  out.vtx.uv = uv;
  out.vtx.Ka = pc.materials->mtl[mtlIndex].ambient;
  out.vtx.Kd = pc.materials->mtl[mtlIndex].diffuse;

  return out;
}

void computeTBN(in float3 n, out float3 x, out float3 y) {
  float yz = -n.y * n.z;
  y = normalize(((abs(n.z) > 0.9999) ? float3(-n.x * n.y, 1.0 - n.y * n.y, yz) :
                                       float3(-n.x * n.z, yz, 1.0 - n.z * n.z)));
  x = cross(y, n);
}

float traceAO(inout RayQuery<RAY_FLAG_NONE> rq, float3 origin, float3 dir) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = dir;
  ray.TMin = 0.0f;
  ray.TMax = pc.aoRadius;

  rq.TraceRayInline(kTLAS[NonUniformResourceIndex(pc.tlas)], RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);

  while (rq.Proceed()) {}

  return (rq.CommittedStatus() != COMMITTED_NOTHING) ? 1.0 : 0.0;
}

// generate a random unsigned int in [0, 2^24) given the previous RNG state using the Numerical Recipes LCG
uint lcg(inout uint prev) {
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint seed) {
  return (float(lcg(seed)) / float(0x01000000));
}

// Generate a random unsigned int from two unsigned int values, using 16 pairs of rounds of the Tiny Encryption Algorithm
uint tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;
  for(uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }
  return v0;
}

// PCG hash (https://www.pcg-random.org/)
uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// xxHash32 for collision detection checksum
uint xxhash32(uint p) {
  const uint PRIME32_2 = 2246822519u;
  const uint PRIME32_3 = 3266489917u;
  const uint PRIME32_4 = 668265263u;
  const uint PRIME32_5 = 374761393u;
  uint h32 = p + PRIME32_5;
  h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> 15u));
  h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
  h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
  return h32 ^ (h32 >> 16u);
}

// Gautron 2020: "Real-Time Ray-Traced Ambient Occlusion of Complex Scenes using Spatial Hashing"
// normalHashPCG/normalHashXXH are precomputed from the quantized normal to avoid redundant work across LODs
uint spatialHashFindOrInsert(float3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  int3 p = int3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  uint bucketIndex = hashKey & ((pc.hashMapSize >> 2u) - 1u); // hashMapSize/4 buckets
  uint baseCell = bucketIndex << 2u;                           // first cell in the bucket
  uint checksum = xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) +
                  xxhash32(uint(p.z) + normalHashXXH))));
  checksum = max(checksum, 1u);
  // linear probing within the bucket (4 contiguous slots = 16 bytes, fits in one cache line)
  for (uint i = 0u; i < 4u; i++) {
    uint cellIndex = baseCell + i;
    uint prev;
    InterlockedCompareExchange(pc.hashChecksums->v[cellIndex], 0u, checksum, prev);
    if (prev == 0u || prev == checksum)
      return cellIndex;
    if (pc.frameId - pc.hashTime->v[cellIndex] > 1) {
      uint dummy;
      InterlockedExchange(pc.hashChecksums->v[cellIndex], checksum, dummy);
      InterlockedExchange(pc.spatialData->v[cellIndex], 0u, dummy);
      InterlockedExchange(pc.hashTime->v[cellIndex], pc.frameId, dummy);
      return cellIndex;
    }
  }
  return 0xFFFFFFFFu;
}

// Read-only hash lookup - no allocation, no atomics. Returns cellIndex or 0xFFFFFFFFu if not found.
uint spatialHashFind(float3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  int3 p = int3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  uint bucketIndex = hashKey & ((pc.hashMapSize >> 2u) - 1u);
  uint baseCell = bucketIndex << 2u;
  uint checksum = xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) + xxhash32(uint(p.z) + normalHashXXH))));
  checksum = max(checksum, 1u);
  for (uint i = 0u; i < 4u; i++) {
    uint stored = pc.hashChecksums->v[baseCell + i];
    if (stored == checksum) return baseCell + i;
    if (stored == 0u) return 0xFFFFFFFFu;
  }
  return 0xFFFFFFFFu;
}

// Adaptive cell size (Gautron 2020, Eq. 2-3): projects to ~pc.sp pixels on screen, quantized to power-of-2.
float computeCellSize(float3 worldPos) {
  float3 camPos = -(mul(float3x3(pc.perFrame->view), pc.perFrame->view[3].xyz));
  float dist = distance(camPos, worldPos);
  // h = dist * tan(fov/2), but fov = 2*atan(1/proj[1][1]), so tan(atan(x))=x and h = dist/proj[1][1]
  float h = dist / pc.perFrame->proj[1][1];
  float sw = pc.sp * (h * 2.0) / pc.resolutionY;
  return exp2(floor(log2(max(sw / pc.smin, 1.0)))) * pc.smin;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input, float4 fragCoord : SV_Position) : SV_Target {
  PerVertex vtx = input.vtx;
  float3 n = normalize(vtx.normal);
  float occlusion = 1.0;

  // ambient occlusion
  if (pc.enableAO) {
    float3 origin = vtx.worldPos + n * 0.001; // avoid self-occlusion
    float3 tangent, bitangent;
    computeTBN(n, tangent, bitangent);
    uint seed = tea(uint(fragCoord.y * 4003.0 + fragCoord.x), pc.frameId); // prime

    if (kEnableSpatialHash) {
      float swd = computeCellSize(vtx.worldPos);

      // precompute normal hash (shared across all LODs)
      int3 nn = int3(floor(n * 3.0));
      uint nhPCG = pcg(uint(nn.x) + pcg(uint(nn.y) + pcg(uint(nn.z))));
      uint nhXXH = xxhash32(uint(nn.x) + xxhash32(uint(nn.y) + xxhash32(uint(nn.z))));

      // always find LOD 0
      uint cell0 = spatialHashFindOrInsert(vtx.worldPos, swd, nhPCG, nhXXH);
      uint data0 = (cell0 != 0xFFFFFFFFu) ? pc.spatialData->v[cell0] : 0u;
      uint samples0 = data0 & 0xFFFFu;

      if (samples0 >= 4u) {
        // --- fast path: LOD 0 has enough data, skip coarser LODs ---
        if (samples0 < pc.maxSamples) {
          float r1 = rnd(seed);
          float r2 = rnd(seed);
          float sq = sqrt(1.0 - r2);
          float phi = 2.0 * 3.141592653589 * r1;
          float3 direction = float3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
          direction = direction.x * tangent + direction.y * bitangent + direction.z * n;
          RayQuery<RAY_FLAG_NONE> rayQuery;
          uint hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
          uint prevData;
          InterlockedAdd(pc.spatialData->v[cell0], (hit << 16u) + 1u, prevData);
        }
        if (pc.hashTime->v[cell0] != pc.frameId) {
          uint dummy;
          InterlockedExchange(pc.hashTime->v[cell0], pc.frameId, dummy);
        }
        if (pc.enableFiltering) {
          // trilinear interpolation across 8 neighboring cells to smooth cell boundaries
          float3 cellPos = vtx.worldPos / swd - 0.5;
          int3 base = int3(floor(cellPos));
          float3 f = fract(cellPos);
          float totalWeight = 0.0;
          float totalAO = 0.0;
          for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
              for (int dx = 0; dx < 2; dx++) {
                float3 neighborPos = (float3(base + int3(dx, dy, dz)) + 0.5) * swd;
                uint ci = spatialHashFind(neighborPos, swd, nhPCG, nhXXH);
                if (ci != 0xFFFFFFFFu) {
                  uint d = pc.spatialData->v[ci];
                  uint s = d & 0xFFFFu;
                  if (s >= 4u) {
                    float w = (dx == 0 ? (1.0 - f.x) : f.x) *
                              (dy == 0 ? (1.0 - f.y) : f.y) *
                              (dz == 0 ? (1.0 - f.z) : f.z);
                    totalWeight += w;
                    totalAO += w * (1.0 - float(d >> 16u) / float(s));
                  }
                }
              }
            }
          }
          occlusion = (totalWeight > 0.0) ? (totalAO / totalWeight) : 1.0;
        } else {
          occlusion = 1.0 - float(data0 >> 16u) / float(samples0);
        }
      } else {
        // --- slow path: LOD 0 not ready, use multi-LOD fallback ---
        uint cells[4];
        cells[0] = cell0;
        float lodSize = swd * 2.0;
        for (int lod = 1; lod < 4; lod++) {
          cells[lod] = spatialHashFindOrInsert(vtx.worldPos, lodSize, nhPCG, nhXXH);
          lodSize *= 2.0;
        }
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float sq = sqrt(1.0 - r2);
        float phi = 2.0 * 3.141592653589 * r1;
        float3 direction = float3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
        direction = direction.x * tangent + direction.y * bitangent + direction.z * n;
        RayQuery<RAY_FLAG_NONE> rayQuery;
        uint hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
        // read spatialData once per LOD, cache for reuse in the render selection below
        uint cachedData[4];
        for (int lod = 0; lod < 4; lod++) {
          cachedData[lod] = 0u;
          if (cells[lod] != 0xFFFFFFFFu) {
            cachedData[lod] = pc.spatialData->v[cells[lod]];
            if ((cachedData[lod] & 0xFFFFu) < pc.maxSamples) {
              uint prevData;
              InterlockedAdd(pc.spatialData->v[cells[lod]], (hit << 16u) + 1u, prevData);
            }
            if (pc.hashTime->v[cells[lod]] != pc.frameId) {
              uint dummy;
              InterlockedExchange(pc.hashTime->v[cells[lod]], pc.frameId, dummy);
            }
          }
        }
        // render from finest LOD with enough samples (coarse-to-fine, reusing cached reads)
        uint renderData = 0u;
        for (int lod = 3; lod >= 0; lod--) {
          if ((cachedData[lod] & 0xFFFFu) >= 4u)
            renderData = cachedData[lod];
        }
        if ((renderData & 0xFFFFu) > 0u)
          occlusion = 1.0 - float(renderData >> 16u) / float(renderData & 0xFFFFu);
      }
    } else {
      // per-pixel AO (original path, spatial hash disabled)
      float occl = 0.0;
      for(int i = 0; i < pc.aoSamples; i++) {
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float sq = sqrt(1.0 - r2);
        float phi = 2 * 3.141592653589 * r1;
        float3 direction = float3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
        direction = direction.x * tangent + direction.y * bitangent + direction.z * n;
        RayQuery<RAY_FLAG_NONE> rayQuery;
        occl += traceAO(rayQuery, origin, direction);
      }
      occlusion = 1 - (occl / pc.aoSamples);
    }
    occlusion = pow(clamp(occlusion, 0, 1), pc.aoPower);
  }

  // directional shadow
  if (pc.enableShadows) {
    RayDesc ray;
    ray.Origin = vtx.worldPos;
    ray.Direction = pc.lightDir.xyz;
    ray.TMin = 0.01;
    ray.TMax = 1000.0;

    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(kTLAS[NonUniformResourceIndex(pc.tlas)], RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, ray);

    while (rq.Proceed()) {}

    if (rq.CommittedStatus() != COMMITTED_NOTHING) occlusion *= 0.5;
  }

  float NdotL1 = clamp(dot(n, normalize(float3(+1, 1, +1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(float3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 1.0 * (NdotL1 + NdotL2); // just make a bit brighter

  return vtx.Ka + vtx.Kd * NdotL * occlusion;
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;
void main() {
  // generate a triangle covering the entire screen
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
   uint tex;
} pc;

void main() {
  out_FragColor = textureBindless2D(pc.tex, 0, uv);
}
)";

const char* kCodeZPrepassVS = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
  mat4 normal;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  vec2 padding; // materials
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeZPrepassFS = R"(
#version 460

void main() {
  // empty fragment shader for Z-prepass
};
)";

const char* kCodeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal; // Octahedral 16-bit https://www.shadertoy.com/view/llfcRl
layout (location=3) in uint mtlIndex;

struct Material {
   vec4 ambient;
   vec4 diffuse;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
  mat4 normal;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(push_constant) uniform constants {
  vec4 lightDir;
  PerFrame perFrame;
  PerObject perObject;
  Materials materials;
  uint tlas;
} pc;

// output
struct PerVertex {
  vec3 worldPos;
  vec3 normal;
  vec2 uv;
  vec4 Ka;
  vec4 Kd;
};
layout (location=0) out PerVertex vtx;
//

// https://www.shadertoy.com/view/llfcRl
vec2 unpackSnorm2x8(uint d) {
  return vec2(uvec2(d, d >> 8) & 255u) / 127.5 - 1.0;
}
vec3 unpackOctahedral16(uint data) {
  vec2 v = unpackSnorm2x8(data);
  // https://x.com/Stubbesaurus/status/937994790553227264
  vec3 n = vec3(v, 1.0 - abs(v.x) - abs(v.y));
  float t = max(-n.z, 0.0);
  n.x += (n.x > 0.0) ? -t : t;
  n.y += (n.y > 0.0) ? -t : t;
  return normalize(n);
}
//

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
  // compute the normal in world-space
  vtx.worldPos = (model * vec4(pos, 1.0)).xyz;
  vtx.normal = normalize(mat3(pc.perObject.normal) * unpackOctahedral16(normal));
  vtx.uv = uv;
  vtx.Ka = pc.materials.mtl[mtlIndex].ambient;
  vtx.Kd = pc.materials.mtl[mtlIndex].diffuse;
}
)";

const char* kCodeFS = R"(
#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_query : require

layout(constant_id = 0) const bool kEnableSpatialHash = true;

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
};

layout(std430, buffer_reference) coherent buffer HashChecksums { uint v[]; };
layout(std430, buffer_reference) coherent buffer SpatialData { uint v[]; };
layout(std430, buffer_reference) coherent buffer HashTime { uint v[]; };

struct PerVertex {
  vec3 worldPos;
  vec3 normal;
  vec2 uv;
  vec4 Ka;
  vec4 Kd;
};

layout(push_constant) uniform constants {
  vec4 lightDir;
  PerFrame perFrame;
  uvec2 dummy0;
  uvec2 dummy1;
  uint tlas;
  bool enableShadows;
  bool enableAO;
  int aoSamples;
  float aoRadius;
  float aoPower;
  uint frameId;
  HashChecksums hashChecksums;
  SpatialData spatialData;
  HashTime hashTime;
  float sp;
  float smin;
  uint maxSamples;
  uint hashMapSize;
  float resolutionY;
  bool enableFiltering;
} pc;

layout (location=0) in PerVertex vtx;

layout (location=0) out vec4 out_FragColor;

void computeTBN(in vec3 n, out vec3 x, out vec3 y) {
  float yz = -n.y * n.z;
  y = normalize(((abs(n.z) > 0.9999) ? vec3(-n.x * n.y, 1.0 - n.y * n.y, yz) : vec3(-n.x * n.z, yz, 1.0 - n.z * n.z)));
  x = cross(y, n);
}

float traceAO(rayQueryEXT rq, vec3 origin, vec3 dir) {
  rayQueryInitializeEXT(rq, kTLAS[pc.tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0f, dir, pc.aoRadius);

  while (rayQueryProceedEXT(rq)) {}

  return (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
}

// generate a random unsigned int in [0, 2^24) given the previous RNG state using the Numerical Recipes LCG
uint lcg(inout uint prev) {
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate a random float in [0, 1) given the previous RNG state
float rnd(inout uint seed) {
  return (float(lcg(seed)) / float(0x01000000));
}

// Generate a random unsigned int from two unsigned int values, using 16 pairs of rounds of the Tiny Encryption Algorithm. See Zafar, Olano, and Curtis,
// "GPU Random Numbers via the Tiny Encryption Algorithm"
uint tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// PCG hash (https://www.pcg-random.org/)
uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// xxHash32 for collision detection checksum
uint xxhash32(uint p) {
  const uint PRIME32_2 = 2246822519u;
  const uint PRIME32_3 = 3266489917u;
  const uint PRIME32_4 = 668265263u;
  const uint PRIME32_5 = 374761393u;
  uint h32 = p + PRIME32_5;
  h32 = PRIME32_4 * ((h32 << 17u) | (h32 >> 15u));
  h32 = PRIME32_2 * (h32 ^ (h32 >> 15u));
  h32 = PRIME32_3 * (h32 ^ (h32 >> 13u));
  return h32 ^ (h32 >> 16u);
}

// Gautron 2020: "Real-Time Ray-Traced Ambient Occlusion of Complex Scenes using Spatial Hashing"
// normalHashPCG/normalHashXXH are precomputed from the quantized normal to avoid redundant work across LODs
uint spatialHashFindOrInsert(vec3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  ivec3 p = ivec3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  uint bucketIndex = hashKey & ((pc.hashMapSize >> 2u) - 1u); // hashMapSize/4 buckets
  uint baseCell = bucketIndex << 2u;                           // first cell in the bucket
  uint checksum = xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) +
                  xxhash32(uint(p.z) + normalHashXXH))));
  checksum = max(checksum, 1u); // 0 = empty sentinel
  // linear probing within the bucket (4 contiguous slots = 16 bytes, fits in one cache line)
  for (uint i = 0u; i < 4u; i++) {
    uint cellIndex = baseCell + i;
    uint prev = atomicCompSwap(pc.hashChecksums.v[cellIndex], 0u, checksum);
    if (prev == 0u || prev == checksum)
      return cellIndex;
    if (pc.frameId - pc.hashTime.v[cellIndex] > 1) {
      atomicExchange(pc.hashChecksums.v[cellIndex], checksum);
      atomicExchange(pc.spatialData.v[cellIndex], 0u);
      atomicExchange(pc.hashTime.v[cellIndex], pc.frameId);
      return cellIndex;
    }
  }
  return 0xFFFFFFFFu;
}

// Read-only hash lookup - no allocation, no atomics. Returns cellIndex or 0xFFFFFFFFu if not found.
uint spatialHashFind(vec3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  ivec3 p = ivec3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  uint bucketIndex = hashKey & ((pc.hashMapSize >> 2u) - 1u);
  uint baseCell = bucketIndex << 2u;
  uint checksum = xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) + xxhash32(uint(p.z) + normalHashXXH))));
  checksum = max(checksum, 1u);
  for (uint i = 0u; i < 4u; i++) {
    uint stored = pc.hashChecksums.v[baseCell + i];
    if (stored == checksum) return baseCell + i;
    if (stored == 0u) return 0xFFFFFFFFu;
  }
  return 0xFFFFFFFFu;
}

// Adaptive cell size (Gautron 2020, Eq. 2-3): projects to ~pc.sp pixels on screen, quantized to power-of-2.
float computeCellSize(vec3 worldPos) {
  vec3 camPos = -(transpose(mat3(pc.perFrame.view)) * pc.perFrame.view[3].xyz);
  float dist = distance(camPos, worldPos);
  // h = dist * tan(fov/2), but fov = 2*atan(1/proj[1][1]), so tan(atan(x))=x and h = dist/proj[1][1]
  float h = dist / pc.perFrame.proj[1][1];
  float sw = pc.sp * (h * 2.0) / pc.resolutionY;
  return exp2(floor(log2(max(sw / pc.smin, 1.0)))) * pc.smin;
}

void main() {
  vec3 n = normalize(vtx.normal);

  float occlusion = 1.0;

  // ambient occlusion
  if (pc.enableAO)
  {
    vec3 origin = vtx.worldPos + n * 0.001; // avoid self-occlusion

    vec3 tangent, bitangent;
    computeTBN(n, tangent, bitangent);

    uint seed = tea(uint(gl_FragCoord.y * 4003.0 + gl_FragCoord.x), pc.frameId); // prime

    if (kEnableSpatialHash) {
      float swd = computeCellSize(vtx.worldPos);

      // precompute normal hash (shared across all LODs)
      ivec3 nn = ivec3(floor(n * 3.0));
      uint nhPCG = pcg(uint(nn.x) + pcg(uint(nn.y) + pcg(uint(nn.z))));
      uint nhXXH = xxhash32(uint(nn.x) + xxhash32(uint(nn.y) + xxhash32(uint(nn.z))));

      // always find LOD 0
      uint cell0 = spatialHashFindOrInsert(vtx.worldPos, swd, nhPCG, nhXXH);
      uint data0 = (cell0 != 0xFFFFFFFFu) ? pc.spatialData.v[cell0] : 0u;
      uint samples0 = data0 & 0xFFFFu;

      if (samples0 >= 4u) {
        // --- fast path: LOD 0 has enough data, skip coarser LODs ---
        if (samples0 < pc.maxSamples) {
          // trace 1 ray, update LOD 0 only
          float r1 = rnd(seed);
          float r2 = rnd(seed);
          float sq = sqrt(1.0 - r2);
          float phi = 2.0 * 3.141592653589 * r1;
          vec3 direction = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
          direction = direction.x * tangent + direction.y * bitangent + direction.z * n;
          rayQueryEXT rayQuery;
          uint hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
          atomicAdd(pc.spatialData.v[cell0], (hit << 16u) + 1u);
        }
        if (pc.hashTime.v[cell0] != pc.frameId)
          atomicExchange(pc.hashTime.v[cell0], pc.frameId);
        if (pc.enableFiltering) {
          // trilinear interpolation across 8 neighboring cells to smooth cell boundaries
          vec3 cellPos = vtx.worldPos / swd - 0.5;
          ivec3 base = ivec3(floor(cellPos));
          vec3 f = fract(cellPos);
          float totalWeight = 0.0;
          float totalAO = 0.0;
          for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
              for (int dx = 0; dx < 2; dx++) {
                vec3 neighborPos = (vec3(base + ivec3(dx, dy, dz)) + 0.5) * swd;
                uint ci = spatialHashFind(neighborPos, swd, nhPCG, nhXXH);
                if (ci != 0xFFFFFFFFu) {
                  uint d = pc.spatialData.v[ci];
                  uint s = d & 0xFFFFu;
                  if (s >= 4u) {
                    float w = (dx == 0 ? (1.0 - f.x) : f.x) *
                              (dy == 0 ? (1.0 - f.y) : f.y) *
                              (dz == 0 ? (1.0 - f.z) : f.z);
                    totalWeight += w;
                    totalAO += w * (1.0 - float(d >> 16u) / float(s));
                  }
                }
              }
            }
          }
          occlusion = (totalWeight > 0.0) ? (totalAO / totalWeight) : 1.0;
        } else {
          occlusion = 1.0 - float(data0 >> 16u) / float(samples0);
        }
      } else {
        // --- slow path: LOD 0 not ready, use multi-LOD fallback ---
        uint cells[4];
        cells[0] = cell0;
        float lodSize = swd * 2.0;
        for (int lod = 1; lod < 4; lod++) {
          cells[lod] = spatialHashFindOrInsert(vtx.worldPos, lodSize, nhPCG, nhXXH);
          lodSize *= 2.0;
        }
        // trace 1 ray, update ALL LODs
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float sq = sqrt(1.0 - r2);
        float phi = 2.0 * 3.141592653589 * r1;
        vec3 direction = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
        direction = direction.x * tangent + direction.y * bitangent + direction.z * n;
        rayQueryEXT rayQuery;
        uint hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
        // read spatialData once per LOD, cache for reuse in the render selection below
        uint cachedData[4];
        for (int lod = 0; lod < 4; lod++) {
          cachedData[lod] = 0u;
          if (cells[lod] != 0xFFFFFFFFu) {
            cachedData[lod] = pc.spatialData.v[cells[lod]];
            if ((cachedData[lod] & 0xFFFFu) < pc.maxSamples)
              atomicAdd(pc.spatialData.v[cells[lod]], (hit << 16u) + 1u);
            if (pc.hashTime.v[cells[lod]] != pc.frameId)
              atomicExchange(pc.hashTime.v[cells[lod]], pc.frameId);
          }
        }
        // render from finest LOD with enough samples (coarse-to-fine, reusing cached reads)
        uint renderData = 0u;
        for (int lod = 3; lod >= 0; lod--) {
          if ((cachedData[lod] & 0xFFFFu) >= 4u)
            renderData = cachedData[lod];
        }
        if ((renderData & 0xFFFFu) > 0u)
          occlusion = 1.0 - float(renderData >> 16u) / float(renderData & 0xFFFFu);
      }
    } else {
      // per-pixel AO (original path, spatial hash disabled)
      float occl = 0.0;
      for(int i = 0; i < pc.aoSamples; i++) {
        float r1        = rnd(seed);
        float r2        = rnd(seed);
        float sq        = sqrt(1.0 - r2);
        float phi       = 2 * 3.141592653589 * r1;
        vec3  direction = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
        direction       = direction.x * tangent + direction.y * bitangent + direction.z * n;
        rayQueryEXT rayQuery;
        occl += traceAO(rayQuery, origin, direction);
      }
      occlusion = 1 - (occl / pc.aoSamples);
    }
    occlusion = pow(clamp(occlusion, 0, 1), pc.aoPower);
  }
  // directional shadow
  if (pc.enableShadows) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, kTLAS[pc.tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xff, vtx.worldPos, 0.01, pc.lightDir.xyz, +1000.0);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) occlusion *= 0.5;
  }

  float NdotL1 = clamp(dot(n, normalize(vec3(+1, 1, +1))),  0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 1.0 * (NdotL1 + NdotL2); // just make a bit brighter

  out_FragColor = vtx.Ka + vtx.Kd * NdotL * occlusion;
};
)";

lvk::IContext* ctx_ = nullptr;

struct {
  lvk::Holder<lvk::TextureHandle> fbOffscreenColor_;
  lvk::Holder<lvk::TextureHandle> fbOffscreenDepth_;
  lvk::Holder<lvk::TextureHandle> fbOffscreenResolve_;
  lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
  lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
  lvk::Holder<lvk::ShaderModuleHandle> smMeshVertZPrepass_;
  lvk::Holder<lvk::ShaderModuleHandle> smMeshFragZPrepass_;
  lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
  lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_[2]; // [0] = no spatial hash, [1] = spatial hash
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshZPrepass_;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
  lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
  lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
  lvk::Holder<lvk::BufferHandle> sbInstances_; // storage buffer for TLAS instances
  lvk::Holder<lvk::BufferHandle> ubPerFrame_;
  lvk::Holder<lvk::BufferHandle> ubPerObject_;
  std::vector<lvk::Holder<lvk::AccelStructHandle>> BLAS;
  lvk::Holder<lvk::AccelStructHandle> TLAS;
  lvk::Holder<lvk::BufferHandle> sbHashChecksums_;
  lvk::Holder<lvk::BufferHandle> sbSpatialData_;
  lvk::Holder<lvk::BufferHandle> sbHashTime_;
} res;

bool enableShadows_ = true;
bool enableAO_ = true;

int aoSamples_ = 2;
float aoRadius_ = 16.0f;
float aoPower_ = 2.0f;
bool timeVaryingNoise = true;

bool enableSpatialHash_ = true;
#if defined(ANDROID)
float spatialHashPixelSize_ = 10.0f;
float spatialHashMinCellSize_ = 0.350f;
#else
float spatialHashPixelSize_ = 6.0f;
float spatialHashMinCellSize_ = 0.250f;
#endif
int spatialHashMaxSamples_ = 500;
bool enableFiltering_ = false;

uint32_t frameId = 0;

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
};

struct UniformsPerObject {
  mat4 model;
  mat4 normal;
};

// this goes into our GLSL shaders
struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<GPUMaterial> materials_;

bool initModel(const std::string& folderContentRoot) {
  const std::string cacheFileName = folderContentRoot + CACHE_FILE_NAME;

  if (!loadFromCache(cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(folderContentRoot, cacheFileName.c_str(), MODEL_PATH))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  for (const auto& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{vec4(mtl.ambient, 1.0f), vec4(mtl.diffuse, 1.0f)});
  }
  res.sbMaterials_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GPUMaterial) * materials_.size(),
      .data = materials_.data(),
      .debugName = "Buffer: materials",
  });

  res.vb0_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Vertex | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_Device,
      .size = sizeof(VertexData) * vertexData_.size(),
      .data = vertexData_.data(),
      .debugName = "Buffer: vertex",
  });
  res.ib0_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Index | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_Device,
      .size = sizeof(uint32_t) * indexData_.size(),
      .data = indexData_.data(),
      .debugName = "Buffer: index",
  });

  const glm::mat3x4 transformMatrix(1.0f);

  lvk::Holder<lvk::BufferHandle> transformBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(glm::mat3x4),
      .data = &transformMatrix,
  });

  const uint32_t totalPrimitiveCount = (uint32_t)indexData_.size() / 3;
  lvk::AccelStructDesc blasDesc{
      .type = lvk::AccelStructType_BLAS,
      .geometryType = lvk::AccelStructGeomType_Triangles,
      .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
      .vertexBuffer = res.vb0_,
      .vertexStride = sizeof(VertexData),
      .numVertices = (uint32_t)vertexData_.size(),
      .indexFormat = VK_INDEX_TYPE_UINT32,
      .indexBuffer = res.ib0_,
      .transformBuffer = transformBuffer,
      .buildRange = {.primitiveCount = totalPrimitiveCount},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "BLAS",
  };
  const lvk::AccelStructSizes blasSizes = ctx_->getAccelStructSizes(blasDesc);
  LLOGL("Full model BLAS sizes (bytes):\n   buildScratchSize = %llu,\n   accelerationStructureSize = %llu\n",
        blasSizes.buildScratchSize,
        blasSizes.accelerationStructureSize);
  const uint32_t maxStorageBufferSize = ctx_->getMaxStorageBufferRange();

  // Calculate number of BLAS
  const uint32_t requiredBlasCount = [&blasSizes, maxStorageBufferSize]() {
    const uint32_t count1 = blasSizes.buildScratchSize / maxStorageBufferSize;
    const uint32_t count2 = blasSizes.accelerationStructureSize / maxStorageBufferSize;
    return 1 + (count1 > count2 ? count1 : count2);
  }();
  blasDesc.buildRange.primitiveCount = totalPrimitiveCount / requiredBlasCount;

  LVK_ASSERT(requiredBlasCount > 0);
  LLOGL("maxStorageBufferSize = %u bytes\nNumber of BLAS = %u\n", maxStorageBufferSize, requiredBlasCount);

  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(0.05f)));
  res.BLAS.reserve(requiredBlasCount);

  std::vector<lvk::AccelStructInstance> instances;
  instances.reserve(requiredBlasCount);
  const uint32_t primitiveCount = blasDesc.buildRange.primitiveCount;
  for (uint32_t i = 0; i < totalPrimitiveCount; i += primitiveCount) {
    const int rest = (int)totalPrimitiveCount - i;
    blasDesc.buildRange.primitiveOffset = i * 3 * sizeof(uint32_t);
    blasDesc.buildRange.primitiveCount = (primitiveCount < rest) ? primitiveCount : rest;
    res.BLAS.emplace_back(ctx_->createAccelerationStructure(blasDesc));
    instances.emplace_back(lvk::AccelStructInstance{
        .transform = (const lvk::mat3x4&)transform,
        .instanceCustomIndex = 0,
        .mask = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
        .accelerationStructureReference = ctx_->gpuAddress(res.BLAS.back()),
    });
  }

  // Buffer for instance data
  res.sbInstances_ = ctx_->createBuffer(lvk::BufferDesc{
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance) * instances.size(),
      .data = instances.data(),
      .debugName = "sbInstances_",
  });

  res.TLAS = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = res.sbInstances_,
      .buildRange = {.primitiveCount = (uint32_t)instances.size()},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
  });

  return true;
}

[[nodiscard]] lvk::Framebuffer createOffscreenFramebuffer(uint32_t w, uint32_t h) {
  lvk::TextureDesc descDepth = {
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_UN24,
      .dimensions = {w, h},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(w, h),
      .debugName = "Offscreen framebuffer (d)",
  };
  if (kNumSamplesMSAA > 1) {
    descDepth.usage = lvk::TextureUsageBits_Attachment;
    descDepth.numSamples = kNumSamplesMSAA;
    descDepth.numMipLevels = 1;
  }

  const uint8_t usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage;
  const lvk::Format format = lvk::Format_RGBA_UN8;

  lvk::TextureDesc descColor = {
      .type = lvk::TextureType_2D,
      .format = format,
      .dimensions = {w, h},
      .usage = usage,
      .numMipLevels = lvk::calcNumMipLevels(w, h),
      .debugName = "Offscreen framebuffer (color)",
  };
  if (kNumSamplesMSAA > 1) {
    descColor.usage = lvk::TextureUsageBits_Attachment;
    descColor.numSamples = kNumSamplesMSAA;
    descColor.numMipLevels = 1;
  }

  res.fbOffscreenColor_ = ctx_->createTexture(descColor);
  res.fbOffscreenDepth_ = ctx_->createTexture(descDepth);
  lvk::Framebuffer fb = {
      .color = {{.texture = res.fbOffscreenColor_}},
      .depthStencil = {.texture = res.fbOffscreenDepth_},
  };

  if (kNumSamplesMSAA > 1) {
    res.fbOffscreenResolve_ = ctx_->createTexture({.type = lvk::TextureType_2D,
                                                   .format = format,
                                                   .dimensions = {w, h},
                                                   .usage = usage,
                                                   .debugName = "Offscreen framebuffer (color resolve)"});
    fb.color[0].resolveTexture = res.fbOffscreenResolve_;
  }

  return fb;
}

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 0,
      .height = 0,
#if USE_SPONZA
      .initialCameraPos = vec3(-25, 10, -1),
      .initialCameraTarget = vec3(10, 10, 0),
#else
      .initialCameraPos = vec3(-100, 40, -47),
      .initialCameraTarget = vec3(0, 35, 0),
#endif
  };
  VULKAN_APP_DECLARE(app, cfg);

  ctx_ = app.ctx_.get();

  res.ubPerFrame_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerFrame),
      .debugName = "Buffer: uniforms (per frame)",
  });

  res.ubPerObject_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerObject),
      .debugName = "Buffer: uniforms (per object)",
  });

  lvk::RenderPass renderPassZPrepass_ = {.color = {{
                                             .loadOp = lvk::LoadOp_Clear,
                                             .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_DontCare : lvk::StoreOp_Store,
                                             .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
                                         }},
                                         .depth = {
                                             .loadOp = lvk::LoadOp_Clear,
                                             .storeOp = lvk::StoreOp_Store,
                                             .clearDepth = 1.0f,
                                         }};

  lvk::RenderPass renderPassOffscreen_ = {.color = {{
                                              .loadOp = lvk::LoadOp_Clear,
                                              .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_DontCare : lvk::StoreOp_Store,
                                              .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
                                          }},
                                          .depth = {
                                              .loadOp = lvk::LoadOp_Load,
                                              .storeOp = lvk::StoreOp_DontCare,
                                          }};

#if defined(LVK_DEMO_WITH_SLANG)
  res.smMeshVert_ = ctx_->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
  res.smMeshFrag_ = ctx_->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
  res.smMeshVertZPrepass_ = ctx_->createShaderModule({codeZPrepassSlang, lvk::Stage_Vert, "Shader Module: main zprepass (vert)"});
  res.smMeshFragZPrepass_ = ctx_->createShaderModule({codeZPrepassSlang, lvk::Stage_Frag, "Shader Module: main zprepass (frag)"});
  res.smFullscreenVert_ = ctx_->createShaderModule({codeFullscreenSlang, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  res.smFullscreenFrag_ = ctx_->createShaderModule({codeFullscreenSlang, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
#else
  res.smMeshVert_ = ctx_->createShaderModule({kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  res.smMeshFrag_ = ctx_->createShaderModule({kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  res.smMeshVertZPrepass_ = ctx_->createShaderModule({kCodeZPrepassVS, lvk::Stage_Vert, "Shader Module: main zprepass (vert)"});
  res.smMeshFragZPrepass_ = ctx_->createShaderModule({kCodeZPrepassFS, lvk::Stage_Frag, "Shader Module: main zprepass (frag)"});
  res.smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  res.smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  lvk::Framebuffer fbOffscreen = createOffscreenFramebuffer(app.width_ / kFramebufferScalar, app.height_ / kFramebufferScalar);

  const lvk::VertexInput vertexInput = {
      .attributes = {{.location = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(VertexData, position)},
                     {.location = 1, .format = VK_FORMAT_R16G16_SFLOAT, .offset = offsetof(VertexData, uv)},
                     {.location = 2, .format = VK_FORMAT_R16_UINT, .offset = offsetof(VertexData, normal)},
                     {.location = 3, .format = VK_FORMAT_R16_UINT, .offset = offsetof(VertexData, mtlIndex)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };
  for (uint32_t enableSpatialHash = 0; enableSpatialHash < 2; enableSpatialHash++) {
    res.renderPipelineState_Mesh_[enableSpatialHash] = ctx_->createRenderPipeline(lvk::RenderPipelineDesc{
        .vertexInput = vertexInput,
        .smVert = res.smMeshVert_,
        .smFrag = res.smMeshFrag_,
        .specInfo = {.entries = {{.constantId = 0, .size = sizeof(uint32_t)}},
                     .data = &enableSpatialHash,
                     .dataSize = sizeof(enableSpatialHash)},
        .color = {{.format = ctx_->getFormat(fbOffscreen.color[0].texture)}},
        .depthFormat = ctx_->getFormat(fbOffscreen.depthStencil.texture),
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .samplesCount = kNumSamplesMSAA,
        .debugName = "Pipeline: mesh",
    });
  }

  res.renderPipelineState_MeshZPrepass_ = ctx_->createRenderPipeline(lvk::RenderPipelineDesc{
      .vertexInput =
          {
              .attributes =
                  {
                      {.location = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(VertexData, position)},
                  },
              .inputBindings = {{.stride = sizeof(VertexData)}},
          },
      .smVert = res.smMeshVertZPrepass_,
      .smFrag = res.smMeshFragZPrepass_,
      .color = {{.format = ctx_->getFormat(fbOffscreen.color[0].texture)}},
      .depthFormat = ctx_->getFormat(fbOffscreen.depthStencil.texture),
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .samplesCount = kNumSamplesMSAA,
      .debugName = "Pipeline: mesh z-prepass",
  });

  // fullscreen
  res.renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(lvk::RenderPipelineDesc{
      .smVert = res.smFullscreenVert_,
      .smFrag = res.smFullscreenFrag_,
      .color = {{.format = app.ctx_->getSwapchainFormat()}},
      .cullMode = VK_CULL_MODE_NONE,
      .debugName = "Pipeline: fullscreen",
  });

  // create spatial hash map buffers (zero-initialized)
  {
    res.sbHashChecksums_ = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * kHashMapSize,
        .debugName = "Buffer: AO hash checksums",
    });
    res.sbSpatialData_ = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * kHashMapSize,
        .debugName = "Buffer: AO spatial data",
    });
    res.sbHashTime_ = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * kHashMapSize,
        .debugName = "Buffer: AO hash time",
    });
    lvk::ICommandBuffer& buf = ctx_->acquireCommandBuffer();
    buf.cmdFillBuffer(res.sbHashChecksums_, 0, VK_WHOLE_SIZE, 0);
    buf.cmdFillBuffer(res.sbSpatialData_, 0, VK_WHOLE_SIZE, 0);
    buf.cmdFillBuffer(res.sbHashTime_, 0, VK_WHOLE_SIZE, 0);
    ctx_->submit(buf);
  }

  if (!initModel(app.folderContentRoot_)) {
    VULKAN_APP_EXIT();
  }

  const mat4 modelMatrix = glm::scale(mat4(1.0f), vec3(0.05f));

  const UniformsPerObject perObject = {
      .model = modelMatrix,
      .normal = glm::transpose(glm::inverse(modelMatrix)),
  };

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    buffer.cmdUpdateBuffer(res.ubPerFrame_,
                           UniformsPerFrame{
                               .proj = glm::perspective(float(45.0f * (M_PI / 180.0f)), aspectRatio, 0.5f, 500.0f),
                               .view = app.camera_.getViewMatrix(),
                           });
    buffer.cmdUpdateBuffer(res.ubPerObject_, 0, sizeof(perObject), &perObject);
    buffer.cmdBindVertexBuffer(0, res.vb0_, 0);
    buffer.cmdBindIndexBuffer(res.ib0_, VK_INDEX_TYPE_UINT32);

    // Pass 1: mesh Z-prepass
    {
      buffer.cmdBeginRendering(renderPassZPrepass_, fbOffscreen);
      buffer.cmdPushDebugGroupLabel("Render Mesh ZPrepass", 0xff0000ff);
      buffer.cmdBindRenderPipeline(res.renderPipelineState_MeshZPrepass_);
      struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t materials;
      } pc = {
          .perFrame = ctx_->gpuAddress(res.ubPerFrame_),
          .perObject = ctx_->gpuAddress(res.ubPerObject_),
          .materials = ctx_->gpuAddress(res.sbMaterials_),
      };
      buffer.cmdPushConstants(pc);
      buffer.cmdBindDepthState({.compareOp = VK_COMPARE_OP_LESS, .isDepthWriteEnabled = true});
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      buffer.cmdPopDebugGroupLabel();
      buffer.cmdEndRendering();
    }
    // Pass 2: mesh with RTX
    {
      buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen);
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindRenderPipeline(res.renderPipelineState_Mesh_[enableSpatialHash_ ? 1 : 0]);
      const struct {
        vec4 lightDir;
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t materials;
        uint32_t tlas;
        int enableShadows;
        int enableAO;
        int aoSamples;
        float aoRadius;
        float aoPower;
        uint32_t frameId;
        uint64_t hashChecksums;
        uint64_t spatialData;
        uint64_t hashTime;
        float sp;
        float smin;
        uint32_t maxSamples;
        uint32_t hashMapSize;
        float resolutionY;
        int enableFiltering;
      } pc = {
          .lightDir = vec4(lightDir_, 1.0),
          .perFrame = ctx_->gpuAddress(res.ubPerFrame_),
          .perObject = ctx_->gpuAddress(res.ubPerObject_),
          .materials = ctx_->gpuAddress(res.sbMaterials_),
          .tlas = res.TLAS.index(),
          .enableShadows = enableShadows_ ? 1 : 0,
          .enableAO = enableAO_ ? 1 : 0,
          .aoSamples = aoSamples_,
          .aoRadius = aoRadius_,
          .aoPower = aoPower_,
          .frameId = timeVaryingNoise ? frameId++ : 0,
          .hashChecksums = ctx_->gpuAddress(res.sbHashChecksums_),
          .spatialData = ctx_->gpuAddress(res.sbSpatialData_),
          .hashTime = ctx_->gpuAddress(res.sbHashTime_),
          .sp = spatialHashPixelSize_,
          .smin = spatialHashMinCellSize_,
          .maxSamples = (uint32_t)spatialHashMaxSamples_,
          .hashMapSize = kHashMapSize,
          .resolutionY = (float)height,
          .enableFiltering = enableFiltering_ ? 1 : 0,
      };
      buffer.cmdPushConstants(pc);
      buffer.cmdBindDepthState({.compareOp = VK_COMPARE_OP_EQUAL, .isDepthWriteEnabled = false});
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      buffer.cmdPopDebugGroupLabel();
      buffer.cmdEndRendering();
    }

    // Pass 3: render into the swapchain image
    {
      lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen.color[0].resolveTexture : fbOffscreen.color[0].texture;

      const lvk::Framebuffer fbMain_ = {
          .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
      };

      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
          },
          fbMain_,
          {.textures = {tex, fbOffscreen.color[0].texture}});
      {
        buffer.cmdBindRenderPipeline(res.renderPipelineState_Fullscreen_);
        buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
        buffer.cmdBindDepthState({});
        const struct {
          uint32_t texture;
        } bindings = {
            .texture = tex.index(),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdDraw(3);
        buffer.cmdPopDebugGroupLabel();
        // imGui
        {
          app.imgui_->beginFrame(fbMain_);

          auto imGuiPushFlagsAndStyles = [](bool value) {
            ImGui::BeginDisabled(!value);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * (value ? 1.0f : 0.3f));
          };
          auto imGuiPopFlagsAndStyles = []() {
            ImGui::PopStyleVar();
            ImGui::EndDisabled();
          };
#if !defined(ANDROID)
          const float indentSize = 16.0f;
          ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
          ImGui::Text("W/S/A/D - camera movement");
          ImGui::Text("1/2 - camera up/down");
          ImGui::Text("Shift - fast movement");
          ImGui::Separator();
          ImGui::Checkbox("Ray traced shadows", &enableShadows_);
          ImGui::Indent(indentSize);
          imGuiPushFlagsAndStyles(enableShadows_);
          ImGui::SliderFloat3("Light dir", glm::value_ptr(lightDir_), -1, 1);
          imGuiPopFlagsAndStyles();
          lightDir_ = glm::normalize(lightDir_);
          ImGui::Unindent(indentSize);
          ImGui::Checkbox("Ray traced AO:", &enableAO_);
          ImGui::Indent(indentSize);
          imGuiPushFlagsAndStyles(enableAO_);
          ImGui::Checkbox("Time-varying noise", &timeVaryingNoise);
          ImGui::SliderFloat("AO power", &aoPower_, 1.0f, 2.0f);
          ImGui::SliderFloat("AO radius", &aoRadius_, 0.5f, 16.0f);
          imGuiPushFlagsAndStyles(!enableSpatialHash_);
          ImGui::SliderInt("AO samples", &aoSamples_, 1, 32);
          imGuiPopFlagsAndStyles();
          ImGui::Separator();
          ImGui::Checkbox("Spatial hashing", &enableSpatialHash_);
          ImGui::Indent(indentSize);
          imGuiPushFlagsAndStyles(enableSpatialHash_);
          ImGui::SliderFloat("Pixel size (sp)", &spatialHashPixelSize_, 1.0f, 10.0f);
          ImGui::SliderFloat("Min cell size", &spatialHashMinCellSize_, 0.05f, 0.5f);
          ImGui::SliderInt("Max samples/cell", &spatialHashMaxSamples_, 16, 1000);
          ImGui::Checkbox("Trilinear filtering", &enableFiltering_);
          if (ImGui::Button("Reset hash map")) {
            buffer.cmdFillBuffer(res.sbHashChecksums_, 0, VK_WHOLE_SIZE, 0);
            buffer.cmdFillBuffer(res.sbSpatialData_, 0, VK_WHOLE_SIZE, 0);
            buffer.cmdFillBuffer(res.sbHashTime_, 0, VK_WHOLE_SIZE, 0);
          }
          ImGui::Unindent(indentSize);
          imGuiPopFlagsAndStyles();
          ImGui::Unindent(indentSize);
          imGuiPopFlagsAndStyles();
          ImGui::End();
#endif // !defined(ANDROID)
        }
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
      buffer.cmdEndRendering();

      ctx_->submit(buffer, fbMain_.color[0].texture);
    }
  });

  // destroy all the Vulkan stuff before closing the window
  res = {};

  VULKAN_APP_EXIT();
}
