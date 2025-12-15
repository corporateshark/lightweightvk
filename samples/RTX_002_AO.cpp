/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Bistro.h"
#include "VulkanApp.h"

#include <taskflow/algorithm/for_each.hpp>

// scene navigation
#define USE_SPONZA 0

#if USE_SPONZA
#define MODEL_PATH "src/Sponza/sponza.obj"
#define TEXTURES_PATH "src/Sponza/"
#define CACHE_FILE_NAME "cache3.data"
vec3 lightDir_ = normalize(vec3(-0.5f, 0.85f, -0.05f));
#else
#define MODEL_PATH "src/bistro/Exterior/exterior.obj"
#define TEXTURES_PATH "src/bistro/Exterior/"
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
constexpr uint32_t kHashMapSize = 16 * 1024 * 1024; // 16M entries on mobile (128 MB)
#else
constexpr uint32_t kHashMapSize = 32 * 1024 * 1024; // 32M entries on desktop (256 MB)
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
struct Material {
  float4 ambient;
  float4 diffuse;
  uint texAmbient;
  uint texDiffuse;
  uint texAlpha;
  uint padding;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
};

struct PerObject {
  float4x4 model;
  float4x4 normal;
};

struct Materials {
  Material mtl[];
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  Materials* materials;
  bool enableTextures;
};

[[vk::push_constant]] PushConstants pc;

struct VSOutput {
  float2 uv : TEXCOORD0;
  nointerpolation uint texDiffuse : TEXCOORD1;
  nointerpolation uint texAlpha : TEXCOORD2;
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexMain(float3 pos : POSITION, float2 uv : TEXCOORD0, uint mtlIndex : TEXCOORD1) {
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;

  VSOutput out;
  out.position = proj * view * model * float4(pos, 1.0);
  out.uv = uv;
  // texture index 0 disables alpha testing
  out.texDiffuse = pc.enableTextures ? pc.materials->mtl[mtlIndex].texDiffuse : 0;
  out.texAlpha = pc.enableTextures ? pc.materials->mtl[mtlIndex].texAlpha : 0;
  return out;
}

// alpha testing: the main pass uses VK_COMPARE_OP_EQUAL, so it never shades the texels discarded here
[shader("fragment")]
void fragmentMain(VSOutput input) {
  if (input.texAlpha > 0 && textureBindless2D(input.texAlpha, 0, input.uv).r < 0.5)
    discard;
  if (input.texDiffuse > 0 && textureBindless2D(input.texDiffuse, 0, input.uv).a < 0.5)
    discard;
}
)";

const char* codeSlang = R"(
struct Material {
  float4 ambient;
  float4 diffuse;
  uint texAmbient;
  uint texDiffuse;
  uint texAlpha;
  uint padding;
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

struct Vertex {
  float x, y, z;
  uint uv;
  uint16_t normal;
  uint16_t mtlIndex;
};

struct Vertices {
  Vertex vtx[];
};

struct Indices {
  uint idx[];
};

// 64-bit slot: [63:56] frameLow | [55:32] checksum (0 = empty) | [31:16] hits | [15:0] samples
struct AOHashSlot { uint64_t v[]; };

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
  bool enableTextures; // fits into the padding before the 8-byte aligned `hashSlot`
  AOHashSlot* hashSlot;
  float sp;
  float smin;
  uint maxSamples;
  uint hashMapSize;
  float resolutionY;
  bool enableFiltering;
  Vertices* vertices;
  Indices* indices;
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
  nointerpolation uint texDiffuse : TEXCOORD5;
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
  out.texDiffuse = pc.materials->mtl[mtlIndex].texDiffuse;

  return out;
}

void computeTBN(in float3 n, out float3 x, out float3 y) {
  float yz = -n.y * n.z;
  y = normalize(((abs(n.z) > 0.9999) ? float3(-n.x * n.y, 1.0 - n.y * n.y, yz) :
                                       float3(-n.x * n.z, yz, 1.0 - n.z * n.z)));
  x = cross(y, n);
}

float2 unpackHalf2(uint d) {
  return float2(f16tof32(d), f16tof32(d >> 16));
}

// alpha testing of a candidate ray query hit (the BLAS is not opaque); textures are sampled at LOD 0
bool isCandidateOpaque(inout RayQuery<RAY_FLAG_NONE> rq) {
  if (!pc.enableTextures)
    return true;
  // instanceCustomIndex = first triangle of the BLAS chunk
  uint index = 3 * (rq.CandidateInstanceID() + rq.CandidatePrimitiveIndex());
  uint i0 = pc.indices->idx[index + 0];
  uint i1 = pc.indices->idx[index + 1];
  uint i2 = pc.indices->idx[index + 2];
  float2 bc = rq.CandidateTriangleBarycentrics();
  float2 uv = unpackHalf2(pc.vertices->vtx[i0].uv) * (1.0 - bc.x - bc.y) +
              unpackHalf2(pc.vertices->vtx[i1].uv) * bc.x +
              unpackHalf2(pc.vertices->vtx[i2].uv) * bc.y;
  Material mtl = pc.materials->mtl[uint(pc.vertices->vtx[i0].mtlIndex)];
  if (mtl.texAlpha > 0 && textureBindless2DLod(mtl.texAlpha, 0, uv, 0).r < 0.5)
    return false;
  if (mtl.texDiffuse > 0 && textureBindless2DLod(mtl.texDiffuse, 0, uv, 0).a < 0.5)
    return false;
  return true;
}

// returns true if anything opaque was hit
bool traceRayQuery(inout RayQuery<RAY_FLAG_NONE> rq, RayDesc ray) {
  rq.TraceRayInline(kTLAS[NonUniformResourceIndex(pc.tlas)], RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);

  while (rq.Proceed()) {
    if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && isCandidateOpaque(rq))
      rq.CommitNonOpaqueTriangleHit();
  }

  return rq.CommittedStatus() != COMMITTED_NOTHING;
}

float traceAO(inout RayQuery<RAY_FLAG_NONE> rq, float3 origin, float3 dir) {
  RayDesc ray;
  ray.Origin = origin;
  ray.Direction = dir;
  ray.TMin = 0.0f;
  ray.TMax = pc.aoRadius;

  return traceRayQuery(rq, ray) ? 1.0 : 0.0;
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

float3 sampleCosineHemisphere(inout uint seed, float3 tangent, float3 bitangent, float3 n) {
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);
  float phi = 2.0 * 3.141592653589 * r1;
  float3 d = float3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
  return d.x * tangent + d.y * bitangent + d.z * n;
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

void bumpFrame(uint cellIndex, uint frameLow) {
  uint64_t cur;
  InterlockedCompareExchange(pc.hashSlot->v[cellIndex], 0, 0, cur);
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    if (uint(cur >> 56) == frameLow) return;
    uint64_t fresh = ((cur << 8) >> 8) | (uint64_t(frameLow) << 56); // clear high byte, OR new frameLow
    uint64_t prev;
    InterlockedCompareExchange(pc.hashSlot->v[cellIndex], cur, fresh, prev);
    if (prev == cur) return;
    cur = prev;
  }
}

// CAS-loop halve of samples and hits. Seeded with prevData so we never CAS against literal 0 (Adreno link bug).
void halve(uint cellIndex, uint64_t cur) {
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    uint s = uint(cur) & 0xFFFFu;
    uint h = min((uint(cur) >> 16u) & 0xFFFFu, s);
    uint newLow = ((h >> 1u) << 16u) | (s >> 1u);
    uint64_t newSlot = (cur & 0xFFFFFFFF00000000ull) | uint64_t(newLow);
    uint64_t prev;
    InterlockedCompareExchange(pc.hashSlot->v[cellIndex], cur, newSlot, prev);
    if (prev == cur) return;
    cur = prev;
  }
}

// CAS-loop undo of our atomicAdd: subtract from low 32 bits only, preserve high 32.
void undoAdd(uint cellIndex, uint64_t cur, uint hit) {
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    uint cur_hits = (uint(cur) >> 16u) & 0xFFFFu;
    uint dec_hits = (hit > 0u && cur_hits > 0u) ? 1u : 0u;
    uint dec = 1u | (dec_hits << 16u);
    uint64_t newSlot = (cur & 0xFFFFFFFF00000000ull) | uint64_t(uint(cur) - dec);
    uint64_t prev;
    InterlockedCompareExchange(pc.hashSlot->v[cellIndex], cur, newSlot, prev);
    if (prev == cur) return;
    cur = prev;
  }
}

void accumulateAndMaybeHalve(uint cellIndex, uint hit) {
  uint64_t addend = uint64_t((hit << 16u) + 1u);
  uint64_t prevData;
  InterlockedAdd(pc.hashSlot->v[cellIndex], addend, prevData);
  uint prevSamples = uint(prevData) & 0xFFFFu;
  // Cell already at cap when we arrived — undo so samples can't race past cap.
  if (prevSamples >= pc.maxSamples) {
    undoAdd(cellIndex, prevData, hit);
    return;
  }
  // Only the thread landing exactly on cap halves — avoids cascading halves.
  if (prevSamples + 1u == pc.maxSamples) halve(cellIndex, prevData);
}

bool isCellReady(uint dataLow) {
  return (dataLow & 0xFFFFu) >= 4u;
}

// Returns -1.0 when the cell isn't ready yet, otherwise visibility in [0,1] (1 = visible).
float cellVisibility(uint dataLow) {
  uint samples = dataLow & 0xFFFFu;
  uint hits = min((dataLow >> 16u) & 0xFFFFu, samples);
  if (samples < 4u) return -1.0;
  return float(samples - hits) / float(samples);
}

// Hash the (cell-position, cell-size, normal) tuple to a bucket base index and a 24-bit checksum.
// Gautron 2020: "Real-Time Ray-Traced Ambient Occlusion of Complex Scenes using Spatial Hashing"
// normalHashPCG/normalHashXXH are precomputed from the quantized normal to avoid redundant work across LODs
void computeBucket(float3 position, float cellSize, uint normalHashPCG, uint normalHashXXH, out uint baseCell, out uint checksum) {
  int3 p = int3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  baseCell = (hashKey & ((pc.hashMapSize >> 2u) - 1u)) << 2u;
  checksum = max(xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) + xxhash32(uint(p.z) + normalHashXXH)))) & 0xFFFFFFu, 1u);
}

uint spatialHashFindOrInsert(float3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  uint baseCell, checksum;
  computeBucket(position, cellSize, normalHashPCG, normalHashXXH, baseCell, checksum);
  // Phase 1: look for our key already in the bucket - prevents a stale slot's eviction from creating a duplicate when
  // our key is already living at a later slot in the same bucket
  for (uint i = 0u; i < 4u; i++) {
    if ((uint(pc.hashSlot->v[baseCell + i] >> 32) & 0xFFFFFFu) == checksum) return baseCell + i;
  }
  // Phase 2: install empty slot first, evict a stale cell otherwise
  uint frameLow = pc.frameId & 0xFFu;
  uint64_t newSlot = (uint64_t(frameLow) << 56) | (uint64_t(checksum) << 32);
  uint64_t empty = 0ull;
  // linear probing within the bucket (4 contiguous slots = 32 bytes, fits in one cache line)
  for (uint i = 0u; i < 4u; i++) {
    uint cellIndex = baseCell + i;
    uint64_t prev;
    InterlockedCompareExchange(pc.hashSlot->v[cellIndex], empty, newSlot, prev);
    if (prev == empty || (uint(prev >> 32) & 0xFFFFFFu) == checksum)
      return cellIndex;
    uint storedFrame = uint(prev >> 56);
    uint age = (frameLow - storedFrame) & 0xFFu; // wraps every 256 frames; safe for any threshold << 256
    if (age > 3u) { // tolerate a few missed bumpFrame()s (best-effort)
      uint64_t claim;
      InterlockedCompareExchange(pc.hashSlot->v[cellIndex], prev, newSlot, claim);
      // won, or another thread raced ahead and installed *our* key in the same slot
      if (claim == prev || (uint(claim >> 32) & 0xFFFFFFu) == checksum)
        return cellIndex;
    }
  }
  return 0xFFFFFFFFu;
}

// Read-only hash lookup - no allocation, no atomics. Returns cellIndex or 0xFFFFFFFFu if not found.
uint spatialHashFind(float3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  uint baseCell, checksum;
  computeBucket(position, cellSize, normalHashPCG, normalHashXXH, baseCell, checksum);
  for (uint i = 0u; i < 4u; i++) {
    uint stored = uint(pc.hashSlot->v[baseCell + i] >> 32) & 0xFFFFFFu;
    if (stored == checksum) return baseCell + i;
    if (stored == 0u) return 0xFFFFFFFFu;
  }
  return 0xFFFFFFFFu;
}

// Adaptive cell size (Gautron 2020, Eq. 2-3): projects to ~pc.sp pixels on screen, quantized to power-of-2.
float computeCellSize(float3 worldPos) {
  float3 camPos = -(transpose(float3x3(pc.perFrame->view)) * pc.perFrame->view[3].xyz);
  float dist = distance(camPos, worldPos);
  float h = dist / pc.perFrame->proj[1][1]; // h = dist * tan(fov/2), but fov = 2*atan(1/proj[1][1]), so tan(atan(x))=x and h = dist/proj[1][1]
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

      uint frameLow = pc.frameId & 0xFFu;
      // always find LOD 0
      uint cell0 = spatialHashFindOrInsert(vtx.worldPos, swd, nhPCG, nhXXH);
      uint data0 = (cell0 != 0xFFFFFFFFu) ? uint(pc.hashSlot->v[cell0]) : 0u;
      uint samples0 = data0 & 0xFFFFu;
      bool ready0 = isCellReady(data0);

      // trace one ray, shared across all LODs that need a sample this frame
      uint hit = 0u;
      bool lod0NeedsRay = cell0 != 0xFFFFFFFFu && samples0 < pc.maxSamples;
      if (!ready0 || lod0NeedsRay) {
        float3 direction = sampleCosineHemisphere(seed, tangent, bitangent, n);
        RayQuery<RAY_FLAG_NONE> rayQuery;
        hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
      }

      // accumulate to LOD 0
      if (cell0 != 0xFFFFFFFFu) {
        if (lod0NeedsRay) accumulateAndMaybeHalve(cell0, hit);
        bumpFrame(cell0, frameLow);
      }

      // LOD 0 not converged: also fill coarser LODs so the fallback render has data
      uint cachedData[4] = { data0, 0u, 0u, 0u };
      if (!ready0) {
        float lodSize = swd * 2.0;
        for (int lod = 1; lod < 4; lod++) {
          uint ci = spatialHashFindOrInsert(vtx.worldPos, lodSize, nhPCG, nhXXH);
          if (ci != 0xFFFFFFFFu) {
            cachedData[lod] = uint(pc.hashSlot->v[ci]);
            if ((cachedData[lod] & 0xFFFFu) < pc.maxSamples)
              accumulateAndMaybeHalve(ci, hit);
            bumpFrame(ci, frameLow);
          }
          lodSize *= 2.0;
        }
      }

      if (ready0 && pc.enableFiltering) {
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
                float vis = cellVisibility(uint(pc.hashSlot->v[ci]));
                if (vis >= 0.0) {
                  float w = (dx == 0 ? (1.0 - f.x) : f.x) *
                            (dy == 0 ? (1.0 - f.y) : f.y) *
                            (dz == 0 ? (1.0 - f.z) : f.z);
                  totalWeight += w;
                  totalAO += w * vis;
                }
              }
            }
          }
        }
        occlusion = (totalWeight > 0.0) ? (totalAO / totalWeight) : 1.0;
      } else {
        // render from finest LOD with enough samples (coarse-to-fine, reusing cached reads)
        uint renderData = ready0 ? data0 : 0u;
        if (!ready0) {
          for (int lod = 3; lod >= 0; lod--) {
            if (isCellReady(cachedData[lod]))
              renderData = cachedData[lod];
          }
        }
        if (renderData != 0u) {
          float vis = cellVisibility(renderData);
          if (vis >= 0.0) occlusion = vis;
        }
      }
    } else {
      // per-pixel AO (original path, spatial hash disabled)
      float occl = 0.0;
      for(int i = 0; i < pc.aoSamples; i++) {
        float3 direction = sampleCosineHemisphere(seed, tangent, bitangent, n);
        RayQuery<RAY_FLAG_NONE> rayQuery;
        occl += traceAO(rayQuery, origin, direction);
      }
      occlusion = 1 - (occl / pc.aoSamples);
    }
    occlusion = pow(clamp(occlusion, 0, 1), pc.aoPower);
  }

  // directional shadow; surfaces facing away from the light are always shadowed
  if (pc.enableShadows && dot(n, pc.lightDir.xyz) <= 0.0) {
    occlusion *= 0.5;
  } else if (pc.enableShadows) {
    RayDesc ray;
    ray.Origin = vtx.worldPos;
    ray.Direction = pc.lightDir.xyz;
    ray.TMin = 0.01;
    ray.TMax = 1000.0;

    RayQuery<RAY_FLAG_NONE> rq;
    if (traceRayQuery(rq, ray)) occlusion *= 0.5;
  }

  float NdotL1 = clamp(dot(n, normalize(float3(+1, 1, +1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(float3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 1.0 * (NdotL1 + NdotL2); // just make a bit brighter

  // texture index 0 is the black dummy texture: treat it as "no texture"
  float4 albedo = pc.enableTextures && input.texDiffuse > 0 ? textureBindless2D(input.texDiffuse, 0, vtx.uv) : float4(1.0);

  return albedo * (vtx.Ka + vtx.Kd * NdotL * occlusion);
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
layout (location=1) in vec2 uv;
layout (location=2) in uint mtlIndex;

struct Material {
   vec4 ambient;
   vec4 diffuse;
   uint texAmbient;
   uint texDiffuse;
   uint texAlpha;
   uint padding;
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
  PerFrame perFrame;
  PerObject perObject;
  Materials materials;
  bool enableTextures;
} pc;

layout (location=0) out vec2 v_uv;
layout (location=1) flat out uint v_texDiffuse;
layout (location=2) flat out uint v_texAlpha;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
  v_uv = uv;
  // texture index 0 disables alpha testing
  v_texDiffuse = pc.enableTextures ? pc.materials.mtl[mtlIndex].texDiffuse : 0;
  v_texAlpha = pc.enableTextures ? pc.materials.mtl[mtlIndex].texAlpha : 0;
}
)";

// alpha testing: the main pass uses VK_COMPARE_OP_EQUAL, so it never shades the texels discarded here
const char* kCodeZPrepassFS = R"(
layout (location=0) in vec2 v_uv;
layout (location=1) flat in uint v_texDiffuse;
layout (location=2) flat in uint v_texAlpha;

void main() {
  if (v_texAlpha > 0 && textureBindless2D(v_texAlpha, 0, v_uv).r < 0.5)
    discard;
  if (v_texDiffuse > 0 && textureBindless2D(v_texDiffuse, 0, v_uv).a < 0.5)
    discard;
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
   uint texAmbient;
   uint texDiffuse;
   uint texAlpha;
   uint padding;
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
layout (location=5) flat out uint texDiffuse;
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
  texDiffuse = pc.materials.mtl[mtlIndex].texDiffuse;
}
)";

const char* kCodeFS = R"(
#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_shader_16bit_storage : require

layout(constant_id = 0) const bool kEnableSpatialHash = true;

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];
layout(set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {
  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv);
}

vec4 textureBindless2DLod(uint textureid, uint samplerid, vec2 uv, float lod) {
  return textureLod(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv, lod);
}

struct Material {
   vec4 ambient;
   vec4 diffuse;
   uint texAmbient;
   uint texDiffuse;
   uint texAlpha;
   uint padding;
};

struct Vertex {
  float x, y, z;
  uint uv;
  uint16_t normal;
  uint16_t mtlIndex;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
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

// 64-bit slot: [63:56] frameLow | [55:32] checksum (0 = empty) | [31:16] hits | [15:0] samples
layout(std430, buffer_reference) coherent buffer HashSlot { uint64_t v[]; };

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
  Materials materials;
  uint tlas;
  bool enableShadows;
  bool enableAO;
  int aoSamples;
  float aoRadius;
  float aoPower;
  uint frameId;
  bool enableTextures; // fits into the padding before the 8-byte aligned `hashSlot`
  HashSlot hashSlot;
  float sp;
  float smin;
  uint maxSamples;
  uint hashMapSize;
  float resolutionY;
  bool enableFiltering;
  Vertices vertices;
  Indices indices;
} pc;

layout (location=0) in PerVertex vtx;
layout (location=5) flat in uint texDiffuse;

layout (location=0) out vec4 out_FragColor;

void computeTBN(in vec3 n, out vec3 x, out vec3 y) {
  float yz = -n.y * n.z;
  y = normalize(((abs(n.z) > 0.9999) ? vec3(-n.x * n.y, 1.0 - n.y * n.y, yz) : vec3(-n.x * n.z, yz, 1.0 - n.z * n.z)));
  x = cross(y, n);
}

// alpha testing of a candidate ray query hit (the BLAS is not opaque); textures are sampled at LOD 0
bool isCandidateOpaque(rayQueryEXT rq) {
  if (!pc.enableTextures)
    return true;
  // instanceCustomIndex = first triangle of the BLAS chunk
  uint index = 3 * (rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false) + rayQueryGetIntersectionPrimitiveIndexEXT(rq, false));
  uint i0 = pc.indices.idx[index + 0];
  uint i1 = pc.indices.idx[index + 1];
  uint i2 = pc.indices.idx[index + 2];
  vec2 bc = rayQueryGetIntersectionBarycentricsEXT(rq, false);
  vec2 uv = unpackHalf2x16(pc.vertices.vtx[i0].uv) * (1.0 - bc.x - bc.y) +
            unpackHalf2x16(pc.vertices.vtx[i1].uv) * bc.x +
            unpackHalf2x16(pc.vertices.vtx[i2].uv) * bc.y;
  Material mtl = pc.materials.mtl[uint(pc.vertices.vtx[i0].mtlIndex)];
  if (mtl.texAlpha > 0 && textureBindless2DLod(mtl.texAlpha, 0, uv, 0).r < 0.5)
    return false;
  if (mtl.texDiffuse > 0 && textureBindless2DLod(mtl.texDiffuse, 0, uv, 0).a < 0.5)
    return false;
  return true;
}

// returns true if anything opaque was hit
bool traceRayQuery(rayQueryEXT rq, vec3 origin, float tmin, vec3 dir, float tmax) {
  rayQueryInitializeEXT(rq, kTLAS[pc.tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, tmin, dir, tmax);

  while (rayQueryProceedEXT(rq)) {
    if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT && isCandidateOpaque(rq))
      rayQueryConfirmIntersectionEXT(rq);
  }

  return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

float traceAO(rayQueryEXT rq, vec3 origin, vec3 dir) {
  return traceRayQuery(rq, origin, 0.0f, dir, pc.aoRadius) ? 1.0 : 0.0;
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

vec3 sampleCosineHemisphere(inout uint seed, vec3 tangent, vec3 bitangent, vec3 n) {
  float r1 = rnd(seed);
  float r2 = rnd(seed);
  float sq = sqrt(1.0 - r2);
  float phi = 2.0 * 3.141592653589 * r1;
  vec3 d = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
  return d.x * tangent + d.y * bitangent + d.z * n;
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

void bumpFrame(uint cellIndex, uint frameLow) {
  uint64_t cur = atomicCompSwap(pc.hashSlot.v[cellIndex], 0, 0);
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    if (uint(cur >> 56) == frameLow) return;
    uint64_t fresh = (cur << 8) >> 8; // clear high byte
    fresh |= uint64_t(frameLow) << 56;
    uint64_t prev = atomicCompSwap(pc.hashSlot.v[cellIndex], cur, fresh);
    if (prev == cur) return;
    cur = prev;
  }
}

// CAS-loop halve of samples and hits. Seeded with prevData so we never CAS against literal 0 (Adreno link bug).
void halve(uint cellIndex, uint64_t cur) {
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    uint s = uint(cur) & 0xFFFFu;
    uint h = min((uint(cur) >> 16u) & 0xFFFFu, s);
    uint newLow = ((h >> 1u) << 16u) | (s >> 1u);
    uint64_t newSlot = (cur & 0xFFFFFFFF00000000ul) | uint64_t(newLow);
    uint64_t prev = atomicCompSwap(pc.hashSlot.v[cellIndex], cur, newSlot);
    if (prev == cur) return;
    cur = prev;
  }
}

// CAS-loop undo of our atomicAdd: subtract from low 32 bits only, preserve high 32.
void undoAdd(uint cellIndex, uint64_t cur, uint hit) {
  for (uint attempt = 0u; attempt < 4u; attempt++) {
    uint cur_hits = (uint(cur) >> 16u) & 0xFFFFu;
    uint dec_hits = (hit > 0u && cur_hits > 0u) ? 1u : 0u;
    uint dec = 1u | (dec_hits << 16u);
    uint64_t newSlot = (cur & 0xFFFFFFFF00000000ul) | uint64_t(uint(cur) - dec);
    uint64_t prev = atomicCompSwap(pc.hashSlot.v[cellIndex], cur, newSlot);
    if (prev == cur) return;
    cur = prev;
  }
}

void accumulateAndMaybeHalve(uint cellIndex, uint hit) {
  uint64_t addend = uint64_t((hit << 16u) + 1u);
  uint64_t prevData = atomicAdd(pc.hashSlot.v[cellIndex], addend);
  uint prevSamples = uint(prevData) & 0xFFFFu;
  // Cell already at cap when we arrived — undo so samples can't race past cap.
  if (prevSamples >= pc.maxSamples) {
    undoAdd(cellIndex, prevData, hit);
    return;
  }
  // Only the thread landing exactly on cap halves — avoids cascading halves.
  if (prevSamples + 1u == pc.maxSamples) halve(cellIndex, prevData);
}

bool isCellReady(uint dataLow) {
  return (dataLow & 0xFFFFu) >= 4u;
}

// Returns -1.0 when the cell isn't ready yet, otherwise visibility in [0,1] (1 = visible).
float cellVisibility(uint dataLow) {
  uint samples = dataLow & 0xFFFFu;
  uint hits = min((dataLow >> 16u) & 0xFFFFu, samples);
  if (samples < 4u) return -1.0;
  return float(samples - hits) / float(samples);
}

// Hash the (cell-position, cell-size, normal) tuple to a bucket base index and a 24-bit checksum.
// Gautron 2020: "Real-Time Ray-Traced Ambient Occlusion of Complex Scenes using Spatial Hashing"
// normalHashPCG/normalHashXXH are precomputed from the quantized normal to avoid redundant work across LODs
void computeBucket(vec3 position, float cellSize, uint normalHashPCG, uint normalHashXXH, out uint baseCell, out uint checksum) {
  ivec3 p = ivec3(floor(position / cellSize));
  uint cs = uint(cellSize * 10000.0);
  uint hashKey = pcg(cs + pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(p.z) + normalHashPCG))));
  baseCell = (hashKey & ((pc.hashMapSize >> 2u) - 1u)) << 2u;
  checksum = max(xxhash32(cs + xxhash32(uint(p.x) + xxhash32(uint(p.y) + xxhash32(uint(p.z) + normalHashXXH)))) & 0xFFFFFFu, 1u);
}

uint spatialHashFindOrInsert(vec3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  uint baseCell, checksum;
  computeBucket(position, cellSize, normalHashPCG, normalHashXXH, baseCell, checksum);
  // Phase 1: look for our key already in the bucket: prevents a stale slot's eviction from creating a duplicate when 
  // our key is already living at a later slot in the same bucket
  for (uint i = 0u; i < 4u; i++) {
    if ((uint(pc.hashSlot.v[baseCell + i] >> 32) & 0xFFFFFFu) == checksum) return baseCell + i;
  }
  // Phase 2: install — empty slot first, evict a stale cell otherwise
  uint frameLow = pc.frameId & 0xFFu;
  uint64_t newSlot = (uint64_t(frameLow) << 56) | (uint64_t(checksum) << 32);
  uint64_t empty = uint64_t(0);
  // linear probing within the bucket (4 contiguous slots = 32 bytes, fits in one cache line)
  for (uint i = 0u; i < 4u; i++) {
    uint cellIndex = baseCell + i;
    uint64_t prev = atomicCompSwap(pc.hashSlot.v[cellIndex], empty, newSlot);
    if (prev == empty || (uint(prev >> 32) & 0xFFFFFFu) == checksum)
      return cellIndex;
    uint storedFrame = uint(prev >> 56);
    uint age = (frameLow - storedFrame) & 0xFFu; // wraps every 256 frames; safe for any threshold << 256
    if (age > 3u) { // tolerate a few missed bumpFrame()s (best-effort)
      uint64_t claim = atomicCompSwap(pc.hashSlot.v[cellIndex], prev, newSlot);
      // won, or another thread raced ahead and installed *our* key in the same slot
      if (claim == prev || (uint(claim >> 32) & 0xFFFFFFu) == checksum)
        return cellIndex;
    }
  }
  return 0xFFFFFFFFu;
}

// Read-only hash lookup: no allocation, no atomics. Returns cellIndex or 0xFFFFFFFFu if not found.
uint spatialHashFind(vec3 position, float cellSize, uint normalHashPCG, uint normalHashXXH) {
  uint baseCell, checksum;
  computeBucket(position, cellSize, normalHashPCG, normalHashXXH, baseCell, checksum);
  for (uint i = 0u; i < 4u; i++) {
    uint stored = uint(pc.hashSlot.v[baseCell + i] >> 32) & 0xFFFFFFu;
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

      uint frameLow = pc.frameId & 0xFFu;
      // always find LOD 0
      uint cell0 = spatialHashFindOrInsert(vtx.worldPos, swd, nhPCG, nhXXH);
      uint data0 = (cell0 != 0xFFFFFFFFu) ? uint(pc.hashSlot.v[cell0]) : 0u;
      uint samples0 = data0 & 0xFFFFu;
      bool ready0 = isCellReady(data0);

      // trace one ray, shared across all LODs that need a sample this frame
      uint hit = 0u;
      bool lod0NeedsRay = cell0 != 0xFFFFFFFFu && samples0 < pc.maxSamples;
      if (!ready0 || lod0NeedsRay) {
        vec3 direction = sampleCosineHemisphere(seed, tangent, bitangent, n);
        rayQueryEXT rayQuery;
        hit = traceAO(rayQuery, origin, direction) > 0.0 ? 1u : 0u;
      }

      // accumulate to LOD 0
      if (cell0 != 0xFFFFFFFFu) {
        if (lod0NeedsRay) accumulateAndMaybeHalve(cell0, hit);
        bumpFrame(cell0, frameLow);
      }

      // LOD 0 not converged: also fill coarser LODs so the fallback render has data
      uint cachedData[4] = uint[4](data0, 0u, 0u, 0u);
      if (!ready0) {
        float lodSize = swd * 2.0;
        for (int lod = 1; lod < 4; lod++) {
          uint ci = spatialHashFindOrInsert(vtx.worldPos, lodSize, nhPCG, nhXXH);
          if (ci != 0xFFFFFFFFu) {
            cachedData[lod] = uint(pc.hashSlot.v[ci]);
            if ((cachedData[lod] & 0xFFFFu) < pc.maxSamples)
              accumulateAndMaybeHalve(ci, hit);
            bumpFrame(ci, frameLow);
          }
          lodSize *= 2.0;
        }
      }

      if (ready0 && pc.enableFiltering) {
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
                float vis = cellVisibility(uint(pc.hashSlot.v[ci]));
                if (vis >= 0.0) {
                  float w = (dx == 0 ? (1.0 - f.x) : f.x) *
                            (dy == 0 ? (1.0 - f.y) : f.y) *
                            (dz == 0 ? (1.0 - f.z) : f.z);
                  totalWeight += w;
                  totalAO += w * vis;
                }
              }
            }
          }
        }
        occlusion = (totalWeight > 0.0) ? (totalAO / totalWeight) : 1.0;
      } else {
        // render from finest LOD with enough samples (coarse-to-fine, reusing cached reads)
        uint renderData = ready0 ? data0 : 0u;
        if (!ready0) {
          for (int lod = 3; lod >= 0; lod--) {
            if (isCellReady(cachedData[lod]))
              renderData = cachedData[lod];
          }
        }
        if (renderData != 0u) {
          float vis = cellVisibility(renderData);
          if (vis >= 0.0) occlusion = vis;
        }
      }
    } else {
      // per-pixel AO (original path, spatial hash disabled)
      float occl = 0.0;
      for(int i = 0; i < pc.aoSamples; i++) {
        vec3 direction = sampleCosineHemisphere(seed, tangent, bitangent, n);
        rayQueryEXT rayQuery;
        occl += traceAO(rayQuery, origin, direction);
      }
      occlusion = 1 - (occl / pc.aoSamples);
    }
    occlusion = pow(clamp(occlusion, 0, 1), pc.aoPower);
  }
  // directional shadow; surfaces facing away from the light are always shadowed
  if (pc.enableShadows && dot(n, pc.lightDir.xyz) <= 0.0) {
    occlusion *= 0.5;
  } else if (pc.enableShadows) {
    rayQueryEXT rq;
    if (traceRayQuery(rq, vtx.worldPos, 0.01, pc.lightDir.xyz, +1000.0)) occlusion *= 0.5;
  }

  float NdotL1 = clamp(dot(n, normalize(vec3(+1, 1, +1))),  0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 1.0 * (NdotL1 + NdotL2); // just make a bit brighter

  // texture index 0 is the black dummy texture: treat it as "no texture"
  vec4 albedo = pc.enableTextures && texDiffuse > 0 ? textureBindless2D(texDiffuse, 0, vtx.uv) : vec4(1.0);

  out_FragColor = albedo * (vtx.Ka + vtx.Kd * NdotL * occlusion);
};
)";

// Reads back the alpha test input of a material (see isCandidateOpaque()) as UNORM8 texels for meshopt_opacityMapRasterize()
const char* kCodeAlphaReadback = R"(
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 1) uniform sampler kSamplers[];

layout(std430, buffer_reference) writeonly buffer Texels {
  uint texels[]; // 4 texels per element
};

layout(push_constant) uniform constants {
  Texels dst;
  uint texAlpha;
  uint texDiffuse;
  uint width;
  uint height;
} pc;

float sampleAlpha(uint tex, vec2 uv, int channel) {
  float lod = log2(float(textureSize(sampler2D(kTextures2D[tex], kSamplers[0]), 0).x) / float(pc.width));
  return textureLod(sampler2D(kTextures2D[tex], kSamplers[0]), uv, lod)[channel];
}

void main() {
  uint pitch = (pc.width + 3) / 4;
  uvec2 p = gl_GlobalInvocationID.xy;

  if (p.x >= pitch || p.y >= pc.height)
    return;

  uint texels = 0;
  for (uint i = 0; i != 4; i++) {
    uint x = 4 * p.x + i;
    vec2 uv = (vec2(x, p.y) + 0.5) / vec2(pc.width, pc.height);
    float alpha = 1.0; // the row padding beyond `width` stays opaque
    if (pc.texAlpha > 0 && x < pc.width)
      alpha = min(alpha, sampleAlpha(pc.texAlpha, uv, 0));
    if (pc.texDiffuse > 0 && x < pc.width)
      alpha = min(alpha, sampleAlpha(pc.texDiffuse, uv, 3));
    texels |= uint(alpha * 255.0 + 0.5) << (8 * i);
  }
  pc.dst.texels[p.y * pitch + p.x] = texels;
}
)";

// GLSL ray queries must opt in to opacity micromaps (VUID-vkCmdDraw-micromap-11636); goes before the first non-preprocessor token
std::string enableOpacityMicromapsInRayQueries(const char* code) {
  std::string source(code);
  const size_t pos = source.find("layout(constant_id");
  LVK_ASSERT(pos != std::string::npos);
  source.insert(pos,
                "#extension GL_EXT_opacity_micromap_ray_query_mode : require\n"
                "const bool gl_EnableOpacityMicromapEXT = true;\n");
  return source;
}

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
  // created on first use: opting in to opacity micromaps slows down the shader compilation
  lvk::Holder<lvk::ShaderModuleHandle> smMeshFragOpacityMicromaps_;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshOpacityMicromaps_[2];
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshZPrepass_;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
  lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
  lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
  lvk::Holder<lvk::BufferHandle> ubPerFrame_;
  lvk::Holder<lvk::BufferHandle> ubPerObject_;
  std::vector<lvk::Holder<lvk::AccelStructHandle>> BLAS;
  lvk::Holder<lvk::AccelStructHandle> TLAS;
  lvk::Holder<lvk::BufferHandle> sbHashSlot_;
  lvk::Holder<lvk::BufferHandle> sbAlphaTestedIndices_;
  lvk::Holder<lvk::AccelStructHandle> opacityMicromap_;
  lvk::Holder<lvk::AccelStructHandle> TLASOpacityMicromaps;
} res;

bool enableShadows_ = true;
bool enableAO_ = true;
bool enableTextures_ = false;
bool texturesRequested_ = false; // textures are loaded the first time `enableTextures_` is set
bool opacityBaked_ = false;
bool supportsOpacityMicromaps_ = false;
// off by default: much slower than the plain alpha test on GPUs without OMM hardware (e.g. NVIDIA before Ada)
bool enableOpacityMicromaps_ = false;

// as in zeux/niagara: bake from mip 1, target micro-triangle edges of 3 texels of mip 0
constexpr uint32_t kOpacityMicromapTextureMip = 1;
constexpr int kOpacityMicromapMaxSubdivisionLevel = 6;
constexpr float kOpacityMicromapTargetEdge = 3.0f / (1 << kOpacityMicromapTextureMip);

int aoSamples_ = 2;
float aoRadius_ = 8.0f;
float aoPower_ = 1.0f;
bool timeVaryingNoise = true;

bool enableSpatialHash_ = true;
#if defined(ANDROID)
float spatialHashPixelSize_ = 8.0f;
float spatialHashMinCellSize_ = 0.250f;
#else
float spatialHashPixelSize_ = 4.0f;
float spatialHashMinCellSize_ = 0.150f;
#endif
int spatialHashMaxSamples_ = 192;
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

// split into chunks whose scratch buffers fit into maxStorageBufferRange; `instanceCustomIndex` is the first triangle of each chunk
void createBLAS(const char* debugName,
                lvk::BufferHandle indexBuffer,
                uint32_t numTriangles,
                bool opaque,
                lvk::AccelStructHandle opacityMicromap,
                lvk::BufferHandle opacityMicromapIndices,
                std::vector<lvk::AccelStructInstance>& outInstances) {
  const glm::mat3x4 transformMatrix(1.0f);

  lvk::Holder<lvk::BufferHandle> transformBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(glm::mat3x4),
      .data = &transformMatrix,
  });

  lvk::AccelStructDesc blasDesc{
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometryFlags = opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0u, // non-opaque: ray queries do alpha testing
      .vertexFormat = lvk::VertexFormat_Float3,
      .vertexBuffer = res.vb0_,
      .vertexStride = sizeof(VertexData),
      .numVertices = (uint32_t)vertexData_.size(),
      .indexFormat = VK_INDEX_TYPE_UINT32,
      .indexBuffer = indexBuffer,
      .transformBuffer = transformBuffer,
      .micromap = opacityMicromap,
      .micromapIndexBuffer = opacityMicromapIndices,
      .buildRange = {.primitiveCount = numTriangles},
      .buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .debugName = debugName,
  };
  const VkAccelerationStructureBuildSizesInfoKHR blasSizes = ctx_->getAccelStructSizes(blasDesc);
  LLOGL("%s sizes (bytes):\n   buildScratchSize = %llu,\n   accelerationStructureSize = %llu\n",
        debugName,
        blasSizes.buildScratchSize,
        blasSizes.accelerationStructureSize);
  // the scratch buffer is a storage buffer (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674)
  const uint32_t maxStorageBufferSize = ctx_->getMaxStorageBufferRange();
  const uint32_t requiredBlasCount = 1 + uint32_t(blasSizes.buildScratchSize / maxStorageBufferSize);
  const uint32_t primitiveCount = (numTriangles + requiredBlasCount - 1) / requiredBlasCount;

  LLOGL("maxStorageBufferSize = %u bytes\nNumber of BLAS = %u\n", maxStorageBufferSize, requiredBlasCount);

  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(0.05f)));

  for (uint32_t i = 0; i < numTriangles; i += primitiveCount) {
    LVK_ASSERT_MSG(i < (1u << 24), "instanceCustomIndex has only 24 bits");
    blasDesc.buildRange.primitiveOffset = i * 3 * sizeof(uint32_t);
    blasDesc.buildRange.primitiveCount = std::min(primitiveCount, numTriangles - i);
    blasDesc.micromapIndexOffset = i * sizeof(int32_t);
    res.BLAS.emplace_back(ctx_->createAccelerationStructure(blasDesc));
    outInstances.emplace_back(lvk::AccelStructInstance{
        .transform = (const VkTransformMatrixKHR&)transform,
        .instanceCustomIndex = i,
        .mask = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        .accelerationStructureReference = ctx_->gpuAddress(res.BLAS.back()),
    });
  }
}

[[nodiscard]] lvk::Holder<lvk::AccelStructHandle> createTLAS(const std::vector<lvk::AccelStructInstance>& instances,
                                                             const char* debugName) {
  lvk::Holder<lvk::BufferHandle> instancesBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance) * instances.size(),
      .data = instances.data(),
      .debugName = "Buffer: TLAS instances",
  });

  return ctx_->createAccelerationStructure({
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .instancesBuffer = instancesBuffer,
      .buildRange = {.primitiveCount = (uint32_t)instances.size()},
      .buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .debugName = debugName,
  });
}

bool initModel(VulkanApp& app) {
  const std::string cacheFileName = app.folderContentRoot_ + CACHE_FILE_NAME;

  if (!loadFromCache(app, cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), MODEL_PATH))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  // texture index 0 means "no texture"; loadMaterialTextures() fills them in later, see the render loop
  for (const CachedMaterial& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{.ambient = vec4(mtl.ambient, 1.0f), .diffuse = vec4(mtl.diffuse, 1.0f)});
  }
  res.sbMaterials_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GPUMaterial) * materials_.size(),
      .data = materials_.data(),
      .debugName = "Buffer: materials",
  });

  // vertices and indices are also read by the ray query alpha test via buffer device addresses
  res.vb0_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Vertex | lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_Device,
      .size = sizeof(VertexData) * vertexData_.size(),
      .data = vertexData_.data(),
      .debugName = "Buffer: vertex",
  });
  res.ib0_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Index | lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_Device,
      .size = sizeof(uint32_t) * indexData_.size(),
      .data = indexData_.data(),
      .debugName = "Buffer: index",
  });

  // opacity is unknown until the textures are loaded
  std::vector<lvk::AccelStructInstance> instances;
  createBLAS("BLAS", res.ib0_, (uint32_t)indexData_.size() / 3, false, {}, {}, instances);
  res.TLAS = createTLAS(instances, "TLAS");

  return true;
}

struct MaterialOpacity {
  std::vector<uint32_t> triangles;
  uint32_t texAlpha = 0;
  uint32_t texDiffuse = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pitch = 0; // in bytes
  std::vector<uint8_t> texels;
  std::vector<uint8_t> levels;
  std::vector<uint32_t> offsets; // into `data`
  std::vector<uint8_t> data;
  std::vector<int32_t> indices; // per triangle, into `levels`/`offsets`
};

// opacity baking: startBakingOpacity() -> background taskflow -> finishBakingOpacity()
struct {
  std::vector<MaterialOpacity> materials;
  std::vector<vec2> uvs;
  // compacted opacity micromap entries
  std::vector<uint8_t> levels;
  std::vector<uint32_t> offsets; // into `data`
  std::vector<uint8_t> data;
  size_t numEntries = 0;
  std::vector<uint32_t> opaqueIndices;
  std::vector<uint32_t> alphaTestedIndices;
  std::vector<int32_t> alphaTestedOpacityMicromapIndices;
  tf::Taskflow taskflow;
  std::atomic<bool> ready = false;
  double startTime = 0;
} opacityBake_;

// Merges the opacity micromap entries of all materials and splits triangles into opaque and alpha-tested ones
void compactOpacity() {
  const uint32_t numTriangles = (uint32_t)indexData_.size() / 3;

  // per triangle: an index into `levels`/`offsets`, or `VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_KHR` for materials without alpha
  std::vector<int32_t> ommIndices(numTriangles, VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_KHR);
  std::vector<uint8_t>& levels = opacityBake_.levels;
  std::vector<uint32_t>& offsets = opacityBake_.offsets;
  std::vector<uint8_t>& data = opacityBake_.data;

  for (const MaterialOpacity& mtl : opacityBake_.materials) {
    if (mtl.indices.empty()) {
      continue; // fully opaque
    }
    const int32_t firstEntry = (int32_t)levels.size();
    const uint32_t dataOffset = (uint32_t)data.size();
    levels.insert(levels.end(), mtl.levels.begin(), mtl.levels.end());
    data.insert(data.end(), mtl.data.begin(), mtl.data.end());
    for (uint32_t offset : mtl.offsets) {
      offsets.push_back(dataOffset + offset);
    }
    for (size_t t = 0; t != mtl.triangles.size(); t++) {
      ommIndices[mtl.triangles[t]] = firstEntry + mtl.indices[t];
    }
  }

  // merges identical entries and replaces uniform ones with special indices
  opacityBake_.numEntries = meshopt_opacityMapCompact(
      data.data(), data.size(), levels.data(), offsets.data(), levels.size(), ommIndices.data(), ommIndices.size(), 4);

  std::vector<uint32_t>& opaqueIndices = opacityBake_.opaqueIndices;
  std::vector<uint32_t>& alphaTestedIndices = opacityBake_.alphaTestedIndices;
  std::vector<int32_t>& alphaTestedOpacityMicromapIndices = opacityBake_.alphaTestedOpacityMicromapIndices;

  for (uint32_t t = 0; t != numTriangles; t++) {
    const uint32_t* triangle = &indexData_[3 * t];
    if (ommIndices[t] == VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_KHR) {
      opaqueIndices.insert(opaqueIndices.end(), triangle, triangle + 3);
    } else {
      alphaTestedIndices.insert(alphaTestedIndices.end(), triangle, triangle + 3);
      alphaTestedOpacityMicromapIndices.push_back(ommIndices[t]);
    }
  }
}

// Reads back the alpha of all alpha-tested materials on the GPU and starts baking their opacity micromaps in the background
void startBakingOpacity() {
  LVK_PROFILER_FUNCTION();

  opacityBake_.startTime = glfwGetTime();

  const uint32_t numTriangles = (uint32_t)indexData_.size() / 3;

  std::unordered_map<uint32_t, lvk::Dimensions> textureDimensions; // by bindless index
  for (const std::pair<const std::string, lvk::Holder<lvk::TextureHandle>>& t : texturesCache_) {
    textureDimensions[t.second.index()] = ctx_->getDimensions(t.second);
  }

  std::vector<std::vector<uint32_t>> trianglesPerMaterial(materials_.size());
  for (uint32_t t = 0; t != numTriangles; t++) {
    trianglesPerMaterial[vertexData_[indexData_[3 * t]].mtlIndex].push_back(t);
  }

  std::vector<MaterialOpacity>& materials = opacityBake_.materials;
  size_t maxSize = 0;

  for (size_t m = 0; m != materials_.size(); m++) {
    const uint32_t texAlpha = textureDimensions.contains(materials_[m].texAlpha) ? materials_[m].texAlpha : 0;
    const uint32_t texDiffuse = textureDimensions.contains(materials_[m].texDiffuse) ? materials_[m].texDiffuse : 0;
    if (trianglesPerMaterial[m].empty() || (!texAlpha && !texDiffuse))
      continue;

    const lvk::Dimensions dim = textureDimensions[texAlpha ? texAlpha : texDiffuse];
    const uint32_t width = std::max(dim.width >> kOpacityMicromapTextureMip, 1u);
    const uint32_t height = std::max(dim.height >> kOpacityMicromapTextureMip, 1u);
    const uint32_t pitch = 4 * ((width + 3) / 4);
    maxSize = std::max(maxSize, (size_t)pitch * height);

    materials.push_back({
        .triangles = std::move(trianglesPerMaterial[m]),
        .texAlpha = texAlpha,
        .texDiffuse = texDiffuse,
        .width = width,
        .height = height,
        .pitch = pitch,
    });
  }

  lvk::Holder<lvk::ShaderModuleHandle> smComp =
      ctx_->createShaderModule({kCodeAlphaReadback, lvk::Stage_Comp, "Shader Module: alpha readback (comp)"});
  lvk::Holder<lvk::ComputePipelineHandle> pipeline = ctx_->createComputePipeline({
      .smComp = smComp,
      .debugName = "Pipeline: alpha readback",
  });
  const size_t readbackSize = std::max(maxSize, size_t(64 * 1024 * 1024));
  lvk::Holder<lvk::BufferHandle> sbTexels = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = readbackSize,
      .debugName = "Buffer: alpha readback",
  });

  // read back as many materials per submit as fit into the readback buffer
  for (size_t first = 0; first != materials.size();) {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();
    buffer.cmdBindComputePipeline(pipeline);
    size_t end = first;
    size_t offset = 0;
    for (; end != materials.size() && offset + (size_t)materials[end].pitch * materials[end].height <= readbackSize; end++) {
      const MaterialOpacity& mtl = materials[end];
      const struct {
        uint64_t dst;
        uint32_t texAlpha;
        uint32_t texDiffuse;
        uint32_t width;
        uint32_t height;
      } pc = {
          .dst = ctx_->gpuAddress(sbTexels, offset),
          .texAlpha = mtl.texAlpha,
          .texDiffuse = mtl.texDiffuse,
          .width = mtl.width,
          .height = mtl.height,
      };
      buffer.cmdPushConstants(pc);
      buffer.cmdDispatch({.width = (mtl.pitch / 4 + 15) / 16, .height = (mtl.height + 15) / 16});
      offset += (size_t)mtl.pitch * mtl.height;
    }
    ctx_->wait(ctx_->submit(buffer));
    ctx_->invalidateMappedMemory(sbTexels, 0, offset);
    const uint8_t* texels = ctx_->getMappedPtr(sbTexels);
    for (; first != end; first++) {
      const size_t size = (size_t)materials[first].pitch * materials[first].height;
      materials[first].texels.assign(texels, texels + size);
      texels += size;
    }
  }

  // unpack UVs, measure and rasterize the materials in parallel, then compact
  tf::Task unpack = opacityBake_.taskflow.emplace([]() {
    opacityBake_.uvs.resize(vertexData_.size());
    for (size_t v = 0; v != vertexData_.size(); v++) {
      opacityBake_.uvs[v] = glm::unpackHalf2x16(vertexData_[v].uv);
    }
  });
  tf::Task rasterize = opacityBake_.taskflow.for_each_index(size_t(0), materials.size(), size_t(1), [](size_t i) {
    MaterialOpacity& mtl = opacityBake_.materials[i];
    // skip fully opaque textures (meshoptimizer treats alpha > 0.5 as opaque)
    if (*std::min_element(mtl.texels.begin(), mtl.texels.end()) > 128) {
      mtl.texels = {};
      return;
    }
    const std::vector<vec2>& uvs = opacityBake_.uvs;
    const size_t numTriangles = mtl.triangles.size();
    std::vector<uint32_t> indices(3 * numTriangles);
    for (size_t t = 0; t != numTriangles; t++) {
      memcpy(&indices[3 * t], &indexData_[3 * mtl.triangles[t]], 3 * sizeof(uint32_t));
    }
    std::vector<uint32_t> sources(numTriangles);
    mtl.levels.resize(numTriangles);
    mtl.indices.resize(numTriangles);
    const size_t numEntries = meshopt_opacityMapMeasure(mtl.levels.data(),
                                                        sources.data(),
                                                        mtl.indices.data(),
                                                        indices.data(),
                                                        indices.size(),
                                                        &uvs[0].x,
                                                        uvs.size(),
                                                        sizeof(vec2),
                                                        mtl.width,
                                                        mtl.height,
                                                        kOpacityMicromapMaxSubdivisionLevel,
                                                        kOpacityMicromapTargetEdge);
    mtl.levels.resize(numEntries);
    mtl.offsets.resize(numEntries);
    for (size_t e = 0; e != numEntries; e++) {
      mtl.offsets[e] = (uint32_t)mtl.data.size();
      mtl.data.resize(mtl.data.size() + meshopt_opacityMapEntrySize(mtl.levels[e], 4));
    }
    for (size_t e = 0; e != numEntries; e++) {
      const uint32_t* triangle = &indices[3 * sources[e]];
      meshopt_opacityMapRasterize(&mtl.data[mtl.offsets[e]],
                                  mtl.levels[e],
                                  4,
                                  &uvs[triangle[0]].x,
                                  &uvs[triangle[1]].x,
                                  &uvs[triangle[2]].x,
                                  mtl.texels.data(),
                                  1,
                                  mtl.pitch,
                                  mtl.width,
                                  mtl.height);
    }
    mtl.texels = {};
  });
  tf::Task compact = opacityBake_.taskflow.emplace(compactOpacity);
  unpack.precede(rasterize);
  rasterize.precede(compact);
  loaderPool_->run(opacityBake_.taskflow, []() { opacityBake_.ready = true; });
}

// Splits triangles into opaque and alpha-tested BLASes, builds the opacity micromap, and rebuilds the TLASes
void finishBakingOpacity() {
  LVK_PROFILER_FUNCTION();

  const std::vector<uint8_t>& levels = opacityBake_.levels;
  const std::vector<uint32_t>& offsets = opacityBake_.offsets;
  const std::vector<uint8_t>& data = opacityBake_.data;
  const size_t numEntries = opacityBake_.numEntries;
  const std::vector<uint32_t>& opaqueIndices = opacityBake_.opaqueIndices;
  const std::vector<uint32_t>& alphaTestedIndices = opacityBake_.alphaTestedIndices;
  const std::vector<int32_t>& alphaTestedOpacityMicromapIndices = opacityBake_.alphaTestedOpacityMicromapIndices;

  const uint32_t numOpaque = (uint32_t)opaqueIndices.size() / 3;
  const uint32_t numAlphaTested = (uint32_t)alphaTestedIndices.size() / 3;
  const size_t dataSize = numEntries ? offsets[numEntries - 1] + meshopt_opacityMapEntrySize(levels[numEntries - 1], 4) : 0;

  if (supportsOpacityMicromaps_ && numEntries) {
    std::vector<VkMicromapTriangleKHR> ommTriangles(numEntries);
    std::vector<VkMicromapUsageKHR> usageCounts;
    for (size_t e = 0; e != numEntries; e++) {
      ommTriangles[e] = {.dataOffset = offsets[e], .subdivisionLevel = levels[e], .format = VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR};
      if (usageCounts.size() <= levels[e]) {
        usageCounts.resize(levels[e] + 1);
      }
      usageCounts[levels[e]].count++;
    }
    for (size_t level = 0; level != usageCounts.size(); level++) {
      usageCounts[level].subdivisionLevel = (uint32_t)level;
      usageCounts[level].format = VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR;
    }
    std::erase_if(usageCounts, [](const VkMicromapUsageKHR& u) { return u.count == 0; });

    lvk::Holder<lvk::BufferHandle> sbTriangles = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
        .storage = lvk::StorageType_Device,
        .size = sizeof(VkMicromapTriangleKHR) * ommTriangles.size(),
        .data = ommTriangles.data(),
        .debugName = "Buffer: opacity micromap triangles",
    });
    lvk::Holder<lvk::BufferHandle> sbData = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
        .storage = lvk::StorageType_Device,
        .size = dataSize,
        .data = data.data(),
        .debugName = "Buffer: opacity micromap data",
    });
    res.opacityMicromap_ = ctx_->createAccelerationStructure({
        .type = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR,
        .micromapData = sbData,
        .micromapTriangles = sbTriangles,
        .micromapUsageCounts = usageCounts.data(),
        .micromapNumUsageCounts = (uint32_t)usageCounts.size(),
        .buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .debugName = "Opacity micromap",
    });
  }

  res.BLAS.clear();

  std::vector<lvk::AccelStructInstance> instances;

  if (numOpaque) {
    lvk::Holder<lvk::BufferHandle> sbOpaqueIndices = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * opaqueIndices.size(),
        .data = opaqueIndices.data(),
        .debugName = "Buffer: opaque indices",
    });
    createBLAS("BLAS (opaque)", sbOpaqueIndices, numOpaque, true, {}, {}, instances);
  }

  std::vector<lvk::AccelStructInstance> instancesOpacityMicromaps = instances;

  if (numAlphaTested) {
    res.sbAlphaTestedIndices_ = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint32_t) * alphaTestedIndices.size(),
        .data = alphaTestedIndices.data(),
        .debugName = "Buffer: alpha-tested indices",
    });
    createBLAS("BLAS (alpha-tested)", res.sbAlphaTestedIndices_, numAlphaTested, false, {}, {}, instances);
    if (res.opacityMicromap_.valid()) {
      lvk::Holder<lvk::BufferHandle> sbAlphaTestedOpacityMicromapIndices = ctx_->createBuffer({
          .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
          .storage = lvk::StorageType_Device,
          .size = sizeof(int32_t) * alphaTestedOpacityMicromapIndices.size(),
          .data = alphaTestedOpacityMicromapIndices.data(),
          .debugName = "Buffer: alpha-tested opacity micromap indices",
      });
      createBLAS("BLAS (alpha-tested, opacity micromap)",
                 res.sbAlphaTestedIndices_,
                 numAlphaTested,
                 false,
                 res.opacityMicromap_,
                 sbAlphaTestedOpacityMicromapIndices,
                 instancesOpacityMicromaps);
    }
  }

  // a BLAS with a micromap attached is slower even with micromaps disabled, hence a separate TLAS
  res.TLAS = createTLAS(instances, "TLAS");
  if (res.opacityMicromap_.valid()) {
    res.TLASOpacityMicromaps = createTLAS(instancesOpacityMicromaps, "TLAS (opacity micromaps)");
  }

  LLOGL("Opacity baked in %u ms: %u opaque triangles, %u alpha-tested triangles, %u opacity micromap entries (%zu bytes)\n",
        (uint32_t)(1000.0 * (glfwGetTime() - opacityBake_.startTime)),
        numOpaque,
        numAlphaTested,
        (uint32_t)numEntries,
        dataSize);

  // release the CPU-side data
  opacityBake_.materials = {};
  opacityBake_.uvs = {};
  opacityBake_.levels = {};
  opacityBake_.offsets = {};
  opacityBake_.data = {};
  opacityBake_.opaqueIndices = {};
  opacityBake_.alphaTestedIndices = {};
  opacityBake_.alphaTestedOpacityMicromapIndices = {};

  opacityBaked_ = true;
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

  lvk::RenderPass renderPassZPrepass_ = {
      .color = {{
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = kNumSamplesMSAA > 1 ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
      }},
      .depth = {
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .clearDepth = 1.0f,
      }};

  lvk::RenderPass renderPassOffscreen_ = {
      .color = {{
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = kNumSamplesMSAA > 1 ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
          .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
      }},
      .depth = {
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
          .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
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
  // GLSL only: Slang cannot emit the OpacityMicromapIdKHR execution mode
  supportsOpacityMicromaps_ = ctx_->supportsOpacityMicromap();
  res.smMeshFrag_ = ctx_->createShaderModule({kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  res.smMeshVertZPrepass_ = ctx_->createShaderModule({kCodeZPrepassVS, lvk::Stage_Vert, "Shader Module: main zprepass (vert)"});
  res.smMeshFragZPrepass_ = ctx_->createShaderModule({kCodeZPrepassFS, lvk::Stage_Frag, "Shader Module: main zprepass (frag)"});
  res.smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  res.smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  lvk::Framebuffer fbOffscreen = createOffscreenFramebuffer(app.width_ / kFramebufferScalar, app.height_ / kFramebufferScalar);

  const lvk::VertexInput vertexInput = {
      .attributes = {{.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
                     {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
                     {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
                     {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };
  const lvk::Format colorFormat = ctx_->getFormat(fbOffscreen.color[0].texture);
  const lvk::Format depthFormat = ctx_->getFormat(fbOffscreen.depthStencil.texture);
  auto createMeshPipelines = [&vertexInput, colorFormat, depthFormat](lvk::ShaderModuleHandle smFrag,
                                                                     lvk::Holder<lvk::RenderPipelineHandle>* outPipelines) {
    for (uint32_t enableSpatialHash = 0; enableSpatialHash < 2; enableSpatialHash++) {
      outPipelines[enableSpatialHash] = ctx_->createRenderPipeline(lvk::RenderPipelineDesc{
          .vertexInput = vertexInput,
          .smVert = res.smMeshVert_,
          .smFrag = smFrag,
          .specInfo = {.entries = {{.constantId = 0, .size = sizeof(uint32_t)}},
                       .data = &enableSpatialHash,
                       .dataSize = sizeof(enableSpatialHash)},
          .color = {{.format = colorFormat}},
          .depthFormat = depthFormat,
          .cullMode = VK_CULL_MODE_BACK_BIT,
          .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
          .samplesCount = kNumSamplesMSAA,
          .debugName = "Pipeline: mesh",
      });
    }
  };
  createMeshPipelines(res.smMeshFrag_, res.renderPipelineState_Mesh_);

  res.renderPipelineState_MeshZPrepass_ = ctx_->createRenderPipeline(lvk::RenderPipelineDesc{
      .vertexInput =
          {
              .attributes =
                  {
                      {.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
                      {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
                      {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)},
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

  // create spatial hash map buffer (zero-initialized)
  {
    res.sbHashSlot_ = ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(uint64_t) * kHashMapSize,
        .debugName = "Buffer: AO hash slot",
    });
    lvk::ICommandBuffer& buf = ctx_->acquireCommandBuffer();
    buf.cmdFillBuffer(res.sbHashSlot_, 0, VK_WHOLE_SIZE, 0);
    ctx_->submit(buf);
  }

  if (!initModel(app)) {
    VULKAN_APP_EXIT();
  }

  const mat4 modelMatrix = glm::scale(mat4(1.0f), vec3(0.05f));

  const UniformsPerObject perObject = {
      .model = modelMatrix,
      .normal = glm::transpose(glm::inverse(modelMatrix)),
  };

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    bool resetHashMap = false;

    // start streaming textures in asynchronously the first time they are enabled
    if (enableTextures_ && !texturesRequested_) {
      texturesRequested_ = true;
      loadMaterialTextures(app, TEXTURES_PATH);
    }

    // all textures are on the GPU (the last update was submitted with the previous frame); with opacity micromaps supported, wait
    // until they are enabled for the first time
    if (texturesRequested_ && opacityBake_.taskflow.empty() && !numRemainingMaterialTextures() &&
        (enableOpacityMicromaps_ || !supportsOpacityMicromaps_)) {
      startBakingOpacity();
    }
    if (opacityBake_.ready.exchange(false)) {
      finishBakingOpacity();
    }

    const bool useOpacityMicromaps = enableTextures_ && enableOpacityMicromaps_ && res.TLASOpacityMicromaps.valid();
    const lvk::AccelStructHandle tlas = useOpacityMicromaps ? res.TLASOpacityMicromaps : res.TLAS;
    if (useOpacityMicromaps && res.smMeshFragOpacityMicromaps_.empty()) {
      res.smMeshFragOpacityMicromaps_ = ctx_->createShaderModule(
          {enableOpacityMicromapsInRayQueries(kCodeFS).c_str(), lvk::Stage_Frag, "Shader Module: main, opacity micromaps (frag)"});
      createMeshPipelines(res.smMeshFragOpacityMicromaps_, res.renderPipelineState_MeshOpacityMicromaps_);
    }
    const lvk::BufferHandle alphaTestedIndices = res.sbAlphaTestedIndices_.valid() ? res.sbAlphaTestedIndices_ : res.ib0_;

    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    processLoadedMaterialTextures(buffer, res.sbMaterials_);

    buffer.cmdUpdateBuffer(res.ubPerFrame_,
                           UniformsPerFrame{
                               .proj = glm::perspective(float(45.0f * (M_PI / 180.0f)), views[0].aspectRatio, 0.5f, 500.0f),
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
      const struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t materials;
        uint32_t enableTextures;
      } pc = {
          .perFrame = ctx_->gpuAddress(res.ubPerFrame_),
          .perObject = ctx_->gpuAddress(res.ubPerObject_),
          .materials = ctx_->gpuAddress(res.sbMaterials_),
          .enableTextures = enableTextures_ ? 1u : 0u,
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
      const lvk::Holder<lvk::RenderPipelineHandle>* pipelines =
          useOpacityMicromaps ? res.renderPipelineState_MeshOpacityMicromaps_ : res.renderPipelineState_Mesh_;
      buffer.cmdBindRenderPipeline(pipelines[enableSpatialHash_ ? 1 : 0]);
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
        int enableTextures;
        uint64_t hashSlot;
        float sp;
        float smin;
        uint32_t maxSamples;
        uint32_t hashMapSize;
        float resolutionY;
        int enableFiltering;
        uint64_t vertices;
        uint64_t indices;
      } pc = {
          .lightDir = vec4(lightDir_, 1.0),
          .perFrame = ctx_->gpuAddress(res.ubPerFrame_),
          .perObject = ctx_->gpuAddress(res.ubPerObject_),
          .materials = ctx_->gpuAddress(res.sbMaterials_),
          .tlas = tlas.index(),
          .enableShadows = enableShadows_ ? 1 : 0,
          .enableAO = enableAO_ ? 1 : 0,
          .aoSamples = aoSamples_,
          .aoRadius = aoRadius_,
          .aoPower = aoPower_,
          .frameId = timeVaryingNoise ? frameId++ : 0,
          .enableTextures = enableTextures_ ? 1 : 0,
          .hashSlot = ctx_->gpuAddress(res.sbHashSlot_),
          .sp = spatialHashPixelSize_,
          .smin = spatialHashMinCellSize_,
          .maxSamples = (uint32_t)spatialHashMaxSamples_,
          .hashMapSize = kHashMapSize,
          .resolutionY = views[0].viewport.height,
          .enableFiltering = enableFiltering_ ? 1 : 0,
          .vertices = ctx_->gpuAddress(res.vb0_),
          .indices = ctx_->gpuAddress(alphaTestedIndices),
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
              .color = {{.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                         .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
          },
          fbMain_,
          {.sampledImages = {tex, fbOffscreen.color[0].texture}});
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
          ImGui::Checkbox("Textures (and alpha testing)", &enableTextures_);
          ImGui::Indent(indentSize);
          imGuiPushFlagsAndStyles(enableTextures_ && texturesRequested_ && !numRemainingMaterialTextures() && supportsOpacityMicromaps_);
          const char* labelOpacityMicromaps = !supportsOpacityMicromaps_                  ? "Opacity micromaps (not supported)"
                                            : enableOpacityMicromaps_ && !opacityBaked_ ? "Opacity micromaps (baking...)"
                                                                                        : "Opacity micromaps";
          ImGui::Checkbox(labelOpacityMicromaps, &enableOpacityMicromaps_);
          imGuiPopFlagsAndStyles();
          ImGui::Unindent(indentSize);
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
          ImGui::SliderFloat("AO power", &aoPower_, 0.5f, 2.0f);
          ImGui::SliderFloat("AO radius", &aoRadius_, 1.0f, 16.0f);
          imGuiPushFlagsAndStyles(!enableSpatialHash_);
          ImGui::SliderInt("AO samples", &aoSamples_, 1, 32);
          imGuiPopFlagsAndStyles();
          ImGui::Separator();
          ImGui::Checkbox("Spatial hashing", &enableSpatialHash_);
          ImGui::Indent(indentSize);
          imGuiPushFlagsAndStyles(enableSpatialHash_);
          ImGui::SliderFloat("Pixel size (sp)", &spatialHashPixelSize_, 1.0f, 10.0f);
          ImGui::SliderFloat("Min cell size", &spatialHashMinCellSize_, 0.05f, 0.5f);
          ImGui::SliderInt("Max samples/cell", &spatialHashMaxSamples_, 16, 250);
          ImGui::Checkbox("Trilinear filtering", &enableFiltering_);
          resetHashMap = ImGui::Button("Reset hash map");
          ImGui::Unindent(indentSize);
          imGuiPopFlagsAndStyles();
          ImGui::Unindent(indentSize);
          imGuiPopFlagsAndStyles();
          ImGui::End();
#else
          const float screenWidth = ImGui::GetIO().DisplaySize.x;
          const float screenHeight = ImGui::GetIO().DisplaySize.y;
          const float buttonWidth = screenWidth * 0.1f;
          const float buttonHeight = buttonWidth * 0.5f;
          ImGui::SetNextWindowPos(ImVec2(10.0f, screenHeight - 2.0f * buttonHeight - 30.0f), ImGuiCond_Always);
          ImGui::Begin("##movement",
                       nullptr,
                       ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_AlwaysAutoResize |
                           ImGuiWindowFlags_NoNavInputs);
          ImGui::Button("Forward", ImVec2(buttonWidth, buttonHeight));
          app.positioner_.movement_.forward_ = ImGui::IsItemActive();
          ImGui::Button("Backward", ImVec2(buttonWidth, buttonHeight));
          app.positioner_.movement_.backward_ = ImGui::IsItemActive();
          ImGui::End();
#endif // !defined(ANDROID)
          drawMaterialTexturesLoadingProgress();
        }
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
      buffer.cmdEndRendering();

      if (resetHashMap) {
        buffer.cmdFillBuffer(res.sbHashSlot_, 0, VK_WHOLE_SIZE, 0);
      }

      ctx_->submit(buffer, fbMain_.color[0].texture);
    }
  });

  // destroy all the Vulkan stuff before closing the window
  cancelLoadingMaterialTextures();
  res = {};

  VULKAN_APP_EXIT();
}
