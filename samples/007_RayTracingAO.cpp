/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdio.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <ldrutils/lutils/ScopeExit.h>

#include <meshoptimizer.h>
#include <shared/Camera.h>
#include <shared/UtilsFPS.h>
#include <fast_obj.h>

#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <lvk/HelpersImGui.h>
#include <lvk/LVK.h>

constexpr uint32_t kMeshCacheVersion = 0xC0DE000A;
constexpr int kNumSamplesMSAA = 4;
#if defined(NDEBUG)
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif // NDEBUG

std::string folderThirdParty;
std::string folderContentRoot;

std::unique_ptr<lvk::ImGuiRenderer> imgui_;

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
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  Materials materials;
} pc;

// output
layout (location=0) flat out Material mtl;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  mtl = pc.materials.mtl[mtlIndex];
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeZPrepassFS = R"(
#version 460

struct Material {
  vec4 ambient;
  vec4 diffuse;
};

layout (location=0) flat in Material mtl;

layout(push_constant) uniform constants {
  uvec2 perFrame;
} pc;

void main() {
  vec4 Ka = mtl.ambient;
  vec4 Kd = mtl.diffuse;
  if (Kd.a < 0.5)
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
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
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
};
layout (location=0) out PerVertex vtx;
layout (location=5) flat out Material mtl;
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
  mtl = pc.materials.mtl[mtlIndex];
  gl_Position = proj * view * model * vec4(pos, 1.0);

  // Compute the normal in world-space
  mat3 norm_matrix = transpose(inverse(mat3(model)));
  vtx.worldPos = (model * vec4(pos, 1.0)).xyz;
  vtx.normal = normalize(norm_matrix * unpackOctahedral16(normal));
  vtx.uv = uv;
}
)";

const char* kCodeFS = R"(
#version 460
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_ray_query : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
};

struct Material {
  vec4 ambient;
  vec4 diffuse;
};

struct PerVertex {
  vec3 worldPos;
  vec3 normal;
  vec2 uv;
};

layout(push_constant) uniform constants {
  vec4 lightDir;
  PerFrame perFrame;
  uvec2 dummy0;
  uvec2 dummy1;
  uint tlas;
  bool enableShadows;
  bool enableAO;
  bool aoDistanceBased;
  int aoSamples;
  float aoRadius;
  float aoPower;
  uint frameId;
} pc;

layout (location=0) in PerVertex vtx;
layout (location=5) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;

void computeTBN(in vec3 n, out vec3 x, out vec3 y) {
  float yz = -n.y * n.z;
  y = normalize(((abs(n.z) > 0.9999) ? vec3(-n.x * n.y, 1.0 - n.y * n.y, yz) : vec3(-n.x * n.z, yz, 1.0 - n.z * n.z)));
  x = cross(y, n);
}

float traceAO(rayQueryEXT rq, vec3 origin, vec3 dir) {
  uint flags = pc.aoDistanceBased ? gl_RayFlagsTerminateOnFirstHitEXT : gl_RayFlagsNoneEXT;

  rayQueryInitializeEXT(rq, kTLAS[pc.tlas], flags, 0xFF, origin, 0.0f, dir, pc.aoRadius);

  while (rayQueryProceedEXT(rq)) {}

  if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
    if (pc.aoDistanceBased) return 1;
    float length = 1.0 - (rayQueryGetIntersectionTEXT(rq, true) / pc.aoRadius);
    return length;
  }

  return 0;
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

void main() {
  vec4 Ka = mtl.ambient;
  vec4 Kd = mtl.diffuse;
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);

  float occlusion = 1.0;

  // ambient occlusion
  if (pc.enableAO)
  {
    vec3 origin = vtx.worldPos + n * 0.001; // avoid self-occlusion

    vec3 tangent, bitangent;
    computeTBN(n, tangent, bitangent);

    uint seed = tea(uint(gl_FragCoord.y * 4003.0 + gl_FragCoord.x), pc.frameId); // prime

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
    occlusion = pow(clamp(occlusion, 0, 1), pc.aoPower);
  }
  // directional shadow
  if (pc.enableShadows) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, kTLAS[pc.tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xff, vtx.worldPos, 0.01, pc.lightDir.xyz, +1000.0);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) occlusion *= 0.5;
  }

  out_FragColor = Ka + Kd * occlusion;
};
)";

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

int width_ = 0;
int height_ = 0;
FramesPerSecondCounter fps_;

constexpr uint32_t kNumBufferedFrames = 3;

std::unique_ptr<lvk::IContext> ctx_;
lvk::Framebuffer fbMain_; // swapchain
lvk::Framebuffer fbOffscreen_;
lvk::Holder<lvk::TextureHandle> fbOffscreenColor_;
lvk::Holder<lvk::TextureHandle> fbOffscreenDepth_;
lvk::Holder<lvk::TextureHandle> fbOffscreenResolve_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshVertZPrepass_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFragZPrepass_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshZPrepass_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
lvk::Holder<lvk::BufferHandle> sbInstances_; // storage buffer for TLAS instances
std::vector<lvk::Holder<lvk::BufferHandle>> ubPerFrame_, ubPerObject_;
lvk::RenderPass renderPassOffscreen_;
lvk::RenderPass renderPassZPrepass_;
lvk::RenderPass renderPassMain_;
lvk::Holder<lvk::AccelStructHandle> BLAS;
lvk::Holder<lvk::AccelStructHandle> TLAS;

// scene navigation
CameraPositioner_FirstPerson positioner_(vec3(-100, 40, -47), vec3(0, 35, 0), vec3(0, 1, 0));
Camera camera_(positioner_);
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;

bool enableShadows_ = true;
bool enableAO_ = true;

int aoSamples_ = 4;
bool aoDistanceBased_ = false;
float aoRadius_ = 8.0f;
float aoPower_ = 1.0f;
bool timeVaryingNoise = false;

uint32_t frameId = 0;

vec3 lightDir_ = normalize(vec3(0.032f, 0.835f, 0.549f));

struct VertexData {
  vec3 position;
  uint32_t uv; // hvec2
  uint16_t normal; // Octahedral 16-bit https://www.shadertoy.com/view/llfcRl
  uint16_t mtlIndex;
};

static_assert(sizeof(VertexData) == 5 * sizeof(uint32_t));

vec2 msign(vec2 v) {
  return vec2(v.x >= 0.0 ? 1.0f : -1.0f, v.y >= 0.0 ? 1.0f : -1.0f);
}

// https://www.shadertoy.com/view/llfcRl
uint16_t packSnorm2x8(vec2 v) {
  glm::uvec2 d = glm::uvec2(round(127.5f + v * 127.5f));
  return d.x | (d.y << 8u);
}

// https://www.shadertoy.com/view/llfcRl
uint16_t packOctahedral16(vec3 n) {
  n /= (abs(n.x) + abs(n.y) + abs(n.z));
  return ::packSnorm2x8((n.z >= 0.0) ? vec2(n.x, n.y) : (vec2(1.0) - abs(vec2(n.y, n.x))) * msign(vec2(n)));
}

std::vector<VertexData> vertexData_;
std::vector<uint32_t> indexData_;

struct UniformsPerFrame {
  mat4 proj;
  mat4 view;
} perFrame_;

struct UniformsPerObject {
  mat4 model;
};
#define MAX_MATERIAL_NAME 128

struct CachedMaterial {
  char name[MAX_MATERIAL_NAME] = {};
  vec3 ambient = vec3(0.0f);
  vec3 diffuse = vec3(0.0f);
};

// this goes into our GLSL shaders
struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<CachedMaterial> cachedMaterials_;
std::vector<GPUMaterial> materials_;

bool initModel();
void createPipelines();
void createOffscreenFramebuffer();

bool init() {
  for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
    ubPerFrame_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                              .storage = lvk::StorageType_HostVisible,
                                              .size = sizeof(UniformsPerFrame),
                                              .debugName = "Buffer: uniforms (per frame)"},
                                             nullptr));
    ubPerObject_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                               .storage = lvk::StorageType_HostVisible,
                                               .size = sizeof(UniformsPerObject),
                                               .debugName = "Buffer: uniforms (per object)"},
                                              nullptr));
  }

  renderPassZPrepass_ = {.color = {{
                             .loadOp = lvk::LoadOp_Clear,
                             .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_MsaaResolve : lvk::StoreOp_Store,
                             .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
                         }},
                         .depth = {
                             .loadOp = lvk::LoadOp_Clear,
                             .storeOp = lvk::StoreOp_Store,
                             .clearDepth = 1.0f,
                         }};

  renderPassOffscreen_ = {.color = {{
                              .loadOp = lvk::LoadOp_Clear,
                              .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_MsaaResolve : lvk::StoreOp_Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
                          }},
                          .depth = {
                              .loadOp = lvk::LoadOp_Load,
                              .storeOp = lvk::StoreOp_DontCare,
                          }};

  renderPassMain_ = {
      .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
  };

  fbMain_ = {
      .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
  };

  createOffscreenFramebuffer();
  createPipelines();

  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, (folderThirdParty + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), float(height_) / 70.0f);

  if (!initModel()) {
    return false;
  }

  return true;
}

void destroy() {
  imgui_ = nullptr;

  vb0_ = nullptr;
  ib0_ = nullptr;
  sbMaterials_ = nullptr;
  ubPerFrame_.clear();
  ubPerObject_.clear();
  smMeshVert_ = nullptr;
  smMeshFrag_ = nullptr;
  smMeshVertZPrepass_ = nullptr;
  smMeshFragZPrepass_ = nullptr;
  smFullscreenVert_ = nullptr;
  smFullscreenFrag_ = nullptr;
  renderPipelineState_Mesh_ = nullptr;
  renderPipelineState_MeshZPrepass_ = nullptr;
  renderPipelineState_Fullscreen_ = nullptr;
  ctx_->destroy(fbMain_);
  fbOffscreenColor_ = nullptr;
  fbOffscreenDepth_ = nullptr;
  fbOffscreenResolve_ = nullptr;
  TLAS = nullptr;
  BLAS = nullptr;
  sbInstances_ = nullptr;
  ctx_ = nullptr;
}

bool loadAndCache(const char* cacheFileName) {
  LVK_PROFILER_FUNCTION();

  // load 3D model and cache it
  LLOGL("Loading `exterior.obj`... It can take a while in debug builds...\n");

  fastObjMesh* mesh = fast_obj_read((folderContentRoot + "src/bistro/Exterior/exterior.obj").c_str());
  SCOPE_EXIT {
    if (mesh)
      fast_obj_destroy(mesh);
  };

  if (!LVK_VERIFY(mesh)) {
    LVK_ASSERT_MSG(false, "Did you read the tutorial at the top of this file?");
    return false;
  }

  uint32_t vertexCount = 0;

  for (uint32_t i = 0; i < mesh->face_count; ++i)
    vertexCount += mesh->face_vertices[i];

  vertexData_.reserve(vertexCount);

  uint32_t vertexIndex = 0;

  for (uint32_t face = 0; face < mesh->face_count; face++) {
    for (uint32_t v = 0; v < mesh->face_vertices[face]; v++) {
      LVK_ASSERT(v < 3);
      const fastObjIndex gi = mesh->indices[vertexIndex++];

      const float* p = &mesh->positions[gi.p * 3];
      const float* n = &mesh->normals[gi.n * 3];
      const float* t = &mesh->texcoords[gi.t * 2];

      vertexData_.push_back({
          .position = vec3(p[0], p[1], p[2]),
          .uv = glm::packHalf2x16(vec2(t[0], t[1])),
          .normal = packOctahedral16(vec3(n[0], n[1], n[2])),
          .mtlIndex = (uint16_t)mesh->face_materials[face],
      });
    }
  }

  // repack the mesh as described in https://github.com/zeux/meshoptimizer
  {
    // 1. Generate an index buffer
    const size_t indexCount = vertexData_.size();
    std::vector<uint32_t> remap(indexCount);
    const size_t vertexCount =
        meshopt_generateVertexRemap(remap.data(), nullptr, indexCount, vertexData_.data(), indexCount, sizeof(VertexData));
    // 2. Remap vertices
    std::vector<VertexData> remappedVertices;
    indexData_.resize(indexCount);
    remappedVertices.resize(vertexCount);
    meshopt_remapIndexBuffer(indexData_.data(), nullptr, indexCount, &remap[0]);
    meshopt_remapVertexBuffer(remappedVertices.data(), vertexData_.data(), indexCount, sizeof(VertexData), remap.data());
    vertexData_ = remappedVertices;
    // 3. Optimize for the GPU vertex cache reuse and overdraw
    meshopt_optimizeVertexCache(indexData_.data(), indexData_.data(), indexCount, vertexCount);
    meshopt_optimizeOverdraw(
        indexData_.data(), indexData_.data(), indexCount, &vertexData_[0].position.x, vertexCount, sizeof(VertexData), 1.05f);
    meshopt_optimizeVertexFetch(vertexData_.data(), indexData_.data(), indexCount, vertexData_.data(), vertexCount, sizeof(VertexData));
  }

  // loop over materials
  for (uint32_t mtlIdx = 0; mtlIdx != mesh->material_count; mtlIdx++) {
    const fastObjMaterial& m = mesh->materials[mtlIdx];
    CachedMaterial mtl;
    mtl.ambient = vec3(m.Ka[0], m.Ka[1], m.Ka[2]);
    mtl.diffuse = vec3(m.Kd[0], m.Kd[1], m.Kd[2]);
    LVK_ASSERT(strlen(m.name) < MAX_MATERIAL_NAME);
    strcat(mtl.name, m.name);
    cachedMaterials_.push_back(mtl);
  }

  LLOGL("Caching mesh...\n");

  FILE* cacheFile = fopen(cacheFileName, "wb");
  if (!cacheFile) {
    return false;
  }
  const uint32_t numMaterials = (uint32_t)cachedMaterials_.size();
  const uint32_t numVertices = (uint32_t)vertexData_.size();
  const uint32_t numIndices = (uint32_t)indexData_.size();
  fwrite(&kMeshCacheVersion, sizeof(kMeshCacheVersion), 1, cacheFile);
  fwrite(&numMaterials, sizeof(numMaterials), 1, cacheFile);
  fwrite(&numVertices, sizeof(numVertices), 1, cacheFile);
  fwrite(&numIndices, sizeof(numIndices), 1, cacheFile);
  fwrite(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile);
  fwrite(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile);
  fwrite(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile);
  return fclose(cacheFile) == 0;
}

bool loadFromCache(const char* cacheFileName) {
  FILE* cacheFile = fopen(cacheFileName, "rb");
  SCOPE_EXIT {
    if (cacheFile) {
      fclose(cacheFile);
    }
  };
  if (!cacheFile) {
    return false;
  }
#define CHECK_READ(expected, read) \
  if ((read) != (expected)) {      \
    return false;                  \
  }
  uint32_t versionProbe = 0;
  CHECK_READ(1, fread(&versionProbe, sizeof(versionProbe), 1, cacheFile));
  if (versionProbe != kMeshCacheVersion) {
    LLOGL("Cache file has wrong version id\n");
    return false;
  }
  uint32_t numMaterials = 0;
  uint32_t numVertices = 0;
  uint32_t numIndices = 0;
  CHECK_READ(1, fread(&numMaterials, sizeof(numMaterials), 1, cacheFile));
  CHECK_READ(1, fread(&numVertices, sizeof(numVertices), 1, cacheFile));
  CHECK_READ(1, fread(&numIndices, sizeof(numIndices), 1, cacheFile));
  cachedMaterials_.resize(numMaterials);
  vertexData_.resize(numVertices);
  indexData_.resize(numIndices);
  CHECK_READ(numMaterials, fread(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile));
  CHECK_READ(numVertices, fread(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile));
  CHECK_READ(numIndices, fread(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile));
#undef CHECK_READ
  return true;
}

bool initModel() {
  const std::string cacheFileName = folderContentRoot + "cache2.data";

  if (!loadFromCache(cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(cacheFileName.c_str()))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  for (const auto& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{vec4(mtl.ambient, 1.0f), vec4(mtl.diffuse, 1.0f)});
  }
  sbMaterials_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                     .storage = lvk::StorageType_Device,
                                     .size = sizeof(GPUMaterial) * materials_.size(),
                                     .data = materials_.data(),
                                     .debugName = "Buffer: materials"},
                                    nullptr);

  vb0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Vertex | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(VertexData) * vertexData_.size(),
                             .data = vertexData_.data(),
                             .debugName = "Buffer: vertex"},
                            nullptr);
  ib0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Index | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(uint32_t) * indexData_.size(),
                             .data = indexData_.data(),
                             .debugName = "Buffer: index"},
                            nullptr);

  const glm::mat3x4 transformMatrix(1.0f);

  lvk::Holder<lvk::BufferHandle> transformBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(glm::mat3x4),
      .data = &transformMatrix,
  });

  BLAS = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_BLAS,
      .geometryType = lvk::AccelStructGeomType_Triangles,
      .vertexFormat = lvk::VertexFormat::Float3,
      .vertexBuffer = vb0_,
      .vertexStride = sizeof(VertexData),
      .numVertices = (uint32_t)vertexData_.size(),
      .indexFormat = lvk::IndexFormat_UI32,
      .indexBuffer = ib0_,
      .transformBuffer = transformBuffer,
      .buildRange = {.primitiveCount = (uint32_t)indexData_.size() / 3},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "BLAS",
  });

  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(0.05f)));

  const lvk::AccelStructInstance instance{
      // clang-format off
      .transform = (const lvk::mat3x4&)transform,
      // clang-format on
      .instanceCustomIndex = 0,
      .mask = 0xff,
      .instanceShaderBindingTableRecordOffset = 0,
      .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
      .accelerationStructureReference = ctx_->gpuAddress(BLAS),
  };

  // Buffer for instance data
  sbInstances_ = ctx_->createBuffer(lvk::BufferDesc{
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance),
      .data = &instance,
      .debugName = "sbInstances_",
  });

  TLAS = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = sbInstances_,
      .buildRange = {.primitiveCount = 1},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
  });

  return true;
}

void createPipelines() {
  if (renderPipelineState_Mesh_.valid()) {
    return;
  }

  smMeshVert_ = ctx_->createShaderModule({kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  smMeshFrag_ = ctx_->createShaderModule({kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  smMeshVertZPrepass_ = ctx_->createShaderModule({kCodeZPrepassVS, lvk::Stage_Vert, "Shader Module: main zprepass (vert)"});
  smMeshFragZPrepass_ = ctx_->createShaderModule({kCodeZPrepassFS, lvk::Stage_Frag, "Shader Module: main zprepass (frag)"});
  smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});

  {
    const lvk::VertexInput vdesc = {
        .attributes =
            {
                {.location = 0, .format = lvk::VertexFormat::Float3, .offset = offsetof(VertexData, position)},
                {.location = 1, .format = lvk::VertexFormat::HalfFloat2, .offset = offsetof(VertexData, uv)},
                {.location = 2, .format = lvk::VertexFormat::UShort1, .offset = offsetof(VertexData, normal)},
                {.location = 3, .format = lvk::VertexFormat::UShort1, .offset = offsetof(VertexData, mtlIndex)},
            },
        .inputBindings = {{.stride = sizeof(VertexData)}},
    };

    lvk::RenderPipelineDesc desc = {
        .vertexInput = vdesc,
        .smVert = smMeshVert_,
        .smFrag = smMeshFrag_,
        .color = {{.format = ctx_->getFormat(fbOffscreen_.color[0].texture)}},
        .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
        .cullMode = lvk::CullMode_Back,
        .frontFace = lvk::WindingMode_CCW,
        .samplesCount = kNumSamplesMSAA,
        .debugName = "Pipeline: mesh",
    };

    renderPipelineState_Mesh_ = ctx_->createRenderPipeline(desc, nullptr);
  }
  {
    const lvk::VertexInput vdesc = {
        .attributes =
            {
                {.location = 0, .format = lvk::VertexFormat::Float3, .offset = offsetof(VertexData, position)},
                {.location = 3, .format = lvk::VertexFormat::UShort1, .offset = offsetof(VertexData, mtlIndex)},
            },
        .inputBindings = {{.stride = sizeof(VertexData)}},
    };
    lvk::RenderPipelineDesc desc = {
        .vertexInput = vdesc,
        .smVert = smMeshVertZPrepass_,
        .smFrag = smMeshFragZPrepass_,
        .color = {{.format = ctx_->getFormat(fbOffscreen_.color[0].texture)}},
        .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
        .cullMode = lvk::CullMode_Back,
        .frontFace = lvk::WindingMode_CCW,
        .samplesCount = kNumSamplesMSAA,
        .debugName = "Pipeline: mesh z-prepass",
    };

    renderPipelineState_MeshZPrepass_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  // fullscreen
  {
    const lvk::RenderPipelineDesc desc = {
        .smVert = smFullscreenVert_,
        .smFrag = smFullscreenFrag_,
        .color = {{.format = ctx_->getFormat(fbMain_.color[0].texture)}},
        .depthFormat = ctx_->getFormat(fbMain_.depthStencil.texture),
        .cullMode = lvk::CullMode_None,
        .debugName = "Pipeline: fullscreen",
    };
    renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(desc, nullptr);
  }
}

void createOffscreenFramebuffer() {
  const uint32_t w = width_;
  const uint32_t h = height_;
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

  fbOffscreenColor_ = ctx_->createTexture(descColor);
  fbOffscreenDepth_ = ctx_->createTexture(descDepth);
  lvk::Framebuffer fb = {
      .color = {{.texture = fbOffscreenColor_}},
      .depthStencil = {.texture = fbOffscreenDepth_},
  };

  if (kNumSamplesMSAA > 1) {
    fbOffscreenResolve_ = ctx_->createTexture({.type = lvk::TextureType_2D,
                                               .format = format,
                                               .dimensions = {w, h},
                                               .usage = usage,
                                               .debugName = "Offscreen framebuffer (color resolve)"});
    fb.color[0].resolveTexture = fbOffscreenResolve_;
  }

  fbOffscreen_ = fb;
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
  createOffscreenFramebuffer();
}

void render(double delta, uint32_t frameIndex) {
  LVK_PROFILER_FUNCTION();

  if (!width_ && !height_)
    return;

  lvk::TextureHandle nativeDrawable = ctx_->getCurrentSwapchainTexture();
  fbMain_.color[0].texture = nativeDrawable;

  // imGui
  {
    imgui_->beginFrame(fbMain_);

    auto imGuiPushFlagsAndStyles = [](bool value) {
      ImGui::BeginDisabled(!value);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * (value ? 1.0f : 0.3f));
    };
    auto imGuiPopFlagsAndStyles = []() {
      ImGui::PopStyleVar();
      ImGui::EndDisabled();
    };

    const float indentSize = 16.0f;
    ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S/A/D - camera movement");
    ImGui::Text("1/2 - camera up/down");
    ImGui::Text("Shift - fast movement");
    ImGui::Separator();
    ImGui::Checkbox("Time-varying noise", &timeVaryingNoise);
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
    ImGui::Checkbox("Distance based AO", &aoDistanceBased_);
    ImGui::SliderFloat("AO radius", &aoRadius_, 0.5f, 16.0f);
    ImGui::SliderFloat("AO power", &aoPower_, 1.0f, 2.0f);
    ImGui::SliderInt("AO samples", &aoSamples_, 1, 32);
    ImGui::Unindent(indentSize);
    imGuiPopFlagsAndStyles();

    ImGui::End();

    // a nice FPS counter
    {
      const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
      const ImGuiViewport* v = ImGui::GetMainViewport();
      LVK_ASSERT(v);
      ImGui::SetNextWindowPos(
          {
              v->WorkPos.x + v->WorkSize.x - 15.0f,
              v->WorkPos.y + 15.0f,
          },
          ImGuiCond_Always,
          {1.0f, 0.0f});
      ImGui::SetNextWindowBgAlpha(0.30f);
      ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x, 0));
      if (ImGui::Begin("##FPS", nullptr, flags)) {
        ImGui::Text("FPS : %i", (int)fps_.getFPS());
        ImGui::Text("Ms  : %.1f", 1000.0 / fps_.getFPS());
      }
      ImGui::End();
    }
  }

  positioner_.update(delta, mousePos_, mousePressed_);

  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = (float)width_ / (float)height_;

  perFrame_ = UniformsPerFrame{
      .proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f),
      .view = camera_.getViewMatrix(),
  };
  ctx_->upload(ubPerFrame_[frameIndex], &perFrame_, sizeof(perFrame_));

  UniformsPerObject perObject;

  perObject.model = glm::scale(mat4(1.0f), vec3(0.05f));

  ctx_->upload(ubPerObject_[frameIndex], &perObject, sizeof(perObject));

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  buffer.cmdBindVertexBuffer(0, vb0_, 0);
  buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);

  // Pass 1: mesh z-prepass
  {
    buffer.cmdBeginRendering(renderPassZPrepass_, fbOffscreen_);
    buffer.cmdPushDebugGroupLabel("Render Mesh ZPrepass", 0xff0000ff);
    buffer.cmdBindRenderPipeline(renderPipelineState_MeshZPrepass_);
    struct {
      uint64_t perFrame;
      uint64_t perObject;
      uint64_t materials;
    } pc = {
        .perFrame = ctx_->gpuAddress(ubPerFrame_[frameIndex]),
        .perObject = ctx_->gpuAddress(ubPerObject_[frameIndex]),
        .materials = ctx_->gpuAddress(sbMaterials_),
    };
    buffer.cmdPushConstants(pc);
    buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
    buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
    buffer.cmdPopDebugGroupLabel();
    buffer.cmdEndRendering();
  }
  // Pass 2: mesh with RTX
  {
    buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen_);
    buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
    buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
    struct {
      vec4 lightDir;
      uint64_t perFrame;
      uint64_t perObject;
      uint64_t materials;
      uint32_t tlas;
      int enableShadows;
      int enableAO;
      int aoDistanceBased;
      int aoSamples;
      float aoRadius;
      float aoPower;
      uint32_t frameId;
    } pc = {
        .lightDir = vec4(lightDir_, 1.0),
        .perFrame = ctx_->gpuAddress(ubPerFrame_[frameIndex]),
        .perObject = ctx_->gpuAddress(ubPerObject_[frameIndex]),
        .materials = ctx_->gpuAddress(sbMaterials_),
        .tlas = TLAS.index(),
        .enableShadows = enableShadows_ ? 1 : 0,
        .enableAO = enableAO_ ? 1 : 0,
        .aoDistanceBased = aoDistanceBased_ ? 1 : 0,
        .aoSamples = aoSamples_,
        .aoRadius = aoRadius_,
        .aoPower = aoPower_,
        .frameId = timeVaryingNoise ? frameId++ : 0,
    };
    buffer.cmdPushConstants(pc);
    buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Equal, .isDepthWriteEnabled = false});
    buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
    buffer.cmdPopDebugGroupLabel();
    buffer.cmdEndRendering();
  }

  ctx_->submit(buffer);

  // Pass 3: render into the swapchain image
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen_.color[0].resolveTexture : fbOffscreen_.color[0].texture;

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassMain_, fbMain_, {.textures = {tex}});
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
      buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
      buffer.cmdBindDepthState({});
      struct {
        uint32_t texture;
      } bindings = {
          .texture = tex.index(),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDraw(3);
      buffer.cmdPopDebugGroupLabel();

      imgui_->endFrame(buffer);
    }
    buffer.cmdEndRendering();

    ctx_->submit(buffer, fbMain_.color[0].texture);
  }
}

inline ImVec4 toVec4(const vec4& c) {
  return ImVec4(c.x, c.y, c.z, c.w);
}

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  // find the content folder
  {
    using namespace std::filesystem;
    path subdir("third-party/content/");
    path dir = current_path();
    // find the content somewhere above our current build directory
    while (dir != current_path().root_path() && !exists(dir / subdir)) {
      dir = dir.parent_path();
    }
    if (!exists(dir / subdir)) {
      printf("Cannot find the content directory. Run `deploy_content.py` before running this app.");
      LVK_ASSERT(false);
      return EXIT_FAILURE;
    }
    folderThirdParty = (dir / path("third-party/deps/src/")).string();
    folderContentRoot = (dir / subdir).string();
  }

  GLFWwindow* window = lvk::initWindow("Vulkan Bistro", width_, height_);
  ctx_ = lvk::createVulkanContextWithSwapchain(
      window, width_, height_, {.enableValidation = kEnableValidationLayers}, lvk::HWDeviceType_Discrete);
  if (!ctx_) {
    return EXIT_FAILURE;
  }

  if (!init()) {
    return EXIT_FAILURE;
  }

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    resize();
  });

  glfwSetCursorPosCallback(window, [](auto* window, double x, double y) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width && height) {
      mousePos_ = vec2(x / width, 1.0f - y / height);
      ImGui::GetIO().MousePos = ImVec2(x, y);
    }
  });

  glfwSetMouseButtonCallback(window, [](auto* window, int button, int action, int mods) {
    if (!ImGui::GetIO().WantCaptureMouse) {
      if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed_ = (action == GLFW_PRESS);
      }
    } else {
      // release the mouse
      mousePressed_ = false;
    }
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    const ImGuiMouseButton_ imguiButton = (button == GLFW_MOUSE_BUTTON_LEFT)
                                              ? ImGuiMouseButton_Left
                                              : (button == GLFW_MOUSE_BUTTON_RIGHT ? ImGuiMouseButton_Right : ImGuiMouseButton_Middle);
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)xpos, (float)ypos);
    io.MouseDown[imguiButton] = action == GLFW_PRESS;
  });

  glfwSetScrollCallback(window, [](GLFWwindow* window, double dx, double dy) {
    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH = (float)dx;
    io.MouseWheel = (float)dy;
  });

  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int mods) {
    const bool pressed = action != GLFW_RELEASE;
    if (key == GLFW_KEY_ESCAPE && pressed) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_ESCAPE && pressed)
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_W) {
      positioner_.movement_.forward_ = pressed;
    }
    if (key == GLFW_KEY_S) {
      positioner_.movement_.backward_ = pressed;
    }
    if (key == GLFW_KEY_A) {
      positioner_.movement_.left_ = pressed;
    }
    if (key == GLFW_KEY_D) {
      positioner_.movement_.right_ = pressed;
    }
    if (key == GLFW_KEY_1) {
      positioner_.movement_.up_ = pressed;
    }
    if (key == GLFW_KEY_2) {
      positioner_.movement_.down_ = pressed;
    }
    if (mods & GLFW_MOD_SHIFT) {
      positioner_.movement_.fastSpeed_ = pressed;
    }
    if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
      positioner_.movement_.fastSpeed_ = pressed;
    }
    if (key == GLFW_KEY_SPACE) {
      positioner_.setUpVector(vec3(0.0f, 1.0f, 0.0f));
    }
  });

  double prevTime = glfwGetTime();
  uint32_t frameIndex = 0;

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const double newTime = glfwGetTime();
    const double delta = newTime - prevTime;
    prevTime = newTime;

    if (!width_ || !height_)
      continue;

    fps_.tick(delta);

    render(delta, frameIndex);

    frameIndex = (frameIndex + 1) % kNumBufferedFrames;
  }

  // destroy all the Vulkan stuff before closing the window
  destroy();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
#else
double getCurrentTimestamp() {
  timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1.0e-9 * t.tv_nsec;
}

extern "C" {
void handle_cmd(android_app* app, int32_t cmd) {
  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (app->window != nullptr) {
      width_ = ANativeWindow_getWidth(app->window);
      height_ = ANativeWindow_getHeight(app->window);
      ctx_ = lvk::createVulkanContextWithSwapchain(app->window,
                                                   width_,
                                                   height_,
                                                   {
                                                     .enableValidation = kEnableValidationLayers,
                                                     .enableAccelerationStructure = true,
                                                     .enableRayQuery = true,
                                                   });
      if (!init()) {
        LLOGW("Failed to initialize the app\n");
        std::terminate();
      }
    }
    break;
  case APP_CMD_TERM_WINDOW:
    destroy();
    break;
  }
}

void resize_callback(ANativeActivity* activity, ANativeWindow* window) {
  int w = ANativeWindow_getWidth(window);
  int h = ANativeWindow_getHeight(window);
  if (width_ != w || height_ != h) {
    width_ = w;
    height_ = h;
    if (ctx_) {
      resize();
    }
  }
}

void android_main(android_app* app) {
  minilog::initialize(nullptr, {.threadNames = false});
  app->onAppCmd = handle_cmd;
  app->activity->callbacks->onNativeWindowResized = resize_callback;

  // find the content folder
  {
    using namespace std::filesystem;
    if (const char* externalStorage = std::getenv("EXTERNAL_STORAGE")) {
      folderThirdParty = (std::filesystem::path(externalStorage) / "LVK" / "deps" / "src").string() + "/";
      folderContentRoot = (std::filesystem::path(externalStorage) / "LVK" / "content").string() + "/";
      if (!exists(folderThirdParty) || !exists(folderContentRoot)) {
        LLOGW("Cannot find the content directory. Run `deploy_content_android.py` before running this app.\n");
        LVK_ASSERT(false);
        std::terminate();
      }
    } else {
      LLOGW("Cannot find EXTERNAL_STORAGE.\n");
      LVK_ASSERT(false);
      std::terminate();
    }
  }

  fps_.printFPS_ = false;

  double prevTime = getCurrentTimestamp();
  uint32_t frameIndex = 0;

  int events = 0;
  android_poll_source* source = nullptr;
  do {
    double newTime = getCurrentTimestamp();
    double delta = newTime - prevTime;
    if (fps_.tick(delta)) {
      LLOGL("FPS: %.1f\n", fps_.getFPS());
    }
    prevTime = newTime;
    if (ctx_) {
      render(delta, frameIndex);
    }
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source) {
        source->process(app, source);
      }
    }
    frameIndex = (frameIndex + 1) % kNumBufferedFrames;
  } while (!app->destroyRequested);
}
} // extern "C"
#endif
