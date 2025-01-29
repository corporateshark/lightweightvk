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
#if defined(NDEBUG)
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif // NDEBUG

#if defined(ANDROID)
constexpr int kFramebufferScalar = 2;
#endif

std::string folderThirdParty;
std::string folderContentRoot;

std::unique_ptr<lvk::ImGuiRenderer> imgui_;

#define UBOS_AND_PUSH_CONSTANTS R"(
struct Material {
  vec4 ambient;
  vec4 diffuse;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

struct Vertex {
  float x, y, z;
  uint uv;
  uint16_t normal; // Octahedral 16-bit https://www.shadertoy.com/view/llfcRl
  uint16_t mtlIndex;
};

layout(std430, buffer_reference) readonly buffer Vertices {
  Vertex vtx[];
};

layout(std430, buffer_reference) readonly buffer Indices {
  uint idx[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 viewInverse;
  mat4 projInverse;
};

layout(push_constant) uniform constants {
  vec4 lightDir;
  PerFrame perFrame;
  Materials materials;
  Indices indices;
  Vertices vertices;
  uint outTexture;
  uint tlas;
};
)"

const char* codeRayGen = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require

layout (set = 0, binding = 2, rgba8) uniform image2D kTextures2DInOut[];
layout (set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];
)"
UBOS_AND_PUSH_CONSTANTS
R"(
layout(location = 0) rayPayloadEXT vec4 payload;

const float tmin = 0.001;
const float tmax = 500.0;

void main() {
  vec2 pixelCenter = gl_LaunchIDEXT.xy + vec2(0.5);
  vec2 d = 2.0 * (pixelCenter / gl_LaunchSizeEXT.xy) - 1.0;

  vec4 origin = perFrame.viewInverse * vec4(0,0,0,1);
  vec4 target = perFrame.projInverse * vec4(d, 1, 1);
  vec4 direction = perFrame.viewInverse * vec4(normalize(target.xyz), 0);

  payload = vec4(0.0, 0.0, 0.0, 1.0);

  traceRayEXT(kTLAS[tlas], gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

  imageStore(kTextures2DInOut[outTexture], ivec2(gl_LaunchIDEXT.xy), payload);
}
)";

const char* codeMiss = R"(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 payload;

void main() {
  payload = vec4(1.0, 1.0, 1.0, 1.0);
})";

const char* codeMissShadow = R"(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT bool isShadowed;

void main() {
  isShadowed = false;
})";

const char* codeClosestHit = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require

const float tmin = 0.001;
const float tmax = 500.0;

layout (set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

layout(location = 0) rayPayloadInEXT vec4 payload;
layout(location = 1) rayPayloadEXT bool isShadowed;

hitAttributeEXT vec2 attribs;
)"
UBOS_AND_PUSH_CONSTANTS
R"(

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
  const vec3 baryCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

  uint index = 3 * gl_PrimitiveID;
  ivec3 triangleIndex = ivec3(indices.idx[index + 0], indices.idx[index + 1], indices.idx[index + 2]);

  vec3 nrm0 = unpackOctahedral16(uint(vertices.vtx[triangleIndex.x].normal));
  vec3 nrm1 = unpackOctahedral16(uint(vertices.vtx[triangleIndex.y].normal));
  vec3 nrm2 = unpackOctahedral16(uint(vertices.vtx[triangleIndex.z].normal));
  vec3 normal = normalize(nrm0 * baryCoords.x + nrm1 * baryCoords.y + nrm2 * baryCoords.z);
  vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT));

  Material mat = materials.mtl[uint(vertices.vtx[triangleIndex.x].mtlIndex)];

  float shadow = 1.0;
  if (dot(lightDir.xyz, worldNormal) > 0) {
    isShadowed = true;
    vec3 hitPoint = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    traceRayEXT(kTLAS[tlas], gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                0xff, 0, 0, 1, hitPoint, tmin, lightDir.xyz, tmax, 1);
    if (isShadowed) {
      shadow = 0.6;
    }
  }

  float NdotL1 = clamp(dot(worldNormal, normalize(vec3(-1, 1,+1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(worldNormal, normalize(vec3(-1, 1,-1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1 + NdotL2);

  payload = vec4(mat.diffuse.rgb * shadow * max(NdotL, 0.0), mat.diffuse.a);
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
  out_FragColor = textureBindless2D(pc.tex, 0, vec2(uv.x, 1.0 - uv.y));
}
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
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smRaygen_;
lvk::Holder<lvk::ShaderModuleHandle> smMiss_;
lvk::Holder<lvk::ShaderModuleHandle> smMissShadow_;
lvk::Holder<lvk::ShaderModuleHandle> smHit_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
lvk::Holder<lvk::BufferHandle> sbInstances_; // storage buffer for TLAS instances
std::vector<lvk::Holder<lvk::BufferHandle>> ubPerFrame_;
lvk::RenderPass renderPassMain_;
std::vector<lvk::Holder<lvk::AccelStructHandle>> BLAS_;
lvk::Holder<lvk::AccelStructHandle> TLAS_;
lvk::Holder<lvk::TextureHandle> rayTracingOutputImage_;
lvk::Holder<lvk::RayTracingPipelineHandle> rayTracingPipeline_;

// scene navigation
#define USE_SPONZA 0

#if USE_SPONZA
#define MODEL_PATH "src/Sponza/sponza.obj"
CameraPositioner_FirstPerson positioner_(vec3(-25, 10, -1), vec3(10, 10, 0), vec3(0, 1, 0));
vec3 lightDir_ = normalize(vec3(0.05f, 1.0f, 0.01f));
#else
#define MODEL_PATH "src/bistro/Exterior/exterior.obj"
CameraPositioner_FirstPerson positioner_(vec3(-100, 40, -47), vec3(0, 35, 0), vec3(0, 1, 0));
vec3 lightDir_ = normalize(vec3(0.032f, 0.835f, 0.549f));
#endif
Camera camera_(positioner_);
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;

uint32_t frameId = 0;

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
  mat4 viewInverse;
  mat4 projInverse;
} perFrame_;

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
void createRayTracingOutputImage();

bool init() {
  for (uint32_t i = 0; i != kNumBufferedFrames; i++) {
    ubPerFrame_.push_back(ctx_->createBuffer({.usage = lvk::BufferUsageBits_Uniform,
                                              .storage = lvk::StorageType_HostVisible,
                                              .size = sizeof(UniformsPerFrame),
                                              .debugName = "Buffer: uniforms (per frame)"},
                                             nullptr));
  }

  renderPassMain_ = {
      .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
  };

  fbMain_ = {
      .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
  };

  createRayTracingOutputImage();
  createPipelines();

  float fontSizePixels = float(height_) / 70.0f;
#if defined(ANDROID)
  fontSizePixels *= kFramebufferScalar;
#endif
  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, (folderThirdParty + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), fontSizePixels);

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
  smFullscreenVert_ = nullptr;
  smFullscreenFrag_ = nullptr;
  smRaygen_ = nullptr;
  smHit_ = nullptr;
  smMiss_ = nullptr;
  smMissShadow_ = nullptr;
  renderPipelineState_Fullscreen_ = nullptr;
  rayTracingPipeline_ = nullptr;
  rayTracingOutputImage_ = nullptr;
  ctx_->destroy(fbMain_);
  TLAS_ = nullptr;
  BLAS_.clear();
  sbInstances_ = nullptr;
  ctx_ = nullptr;
}

bool loadAndCache(const char* cacheFileName) {
  LVK_PROFILER_FUNCTION();

  // load 3D model and cache it
  LLOGL("Loading `%s`... It can take a while in debug builds...\n", MODEL_PATH);

  fastObjMesh* mesh = fast_obj_read((folderContentRoot + MODEL_PATH).c_str());
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

  vb0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(VertexData) * vertexData_.size(),
                             .data = vertexData_.data(),
                             .debugName = "Buffer: vertex"},
                            nullptr);
  ib0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage | lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
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

  const auto totalPrimitiveCount = (uint32_t)indexData_.size() / 3;
  lvk::AccelStructDesc blasDesc{
      .type = lvk::AccelStructType_BLAS,
      .geometryType = lvk::AccelStructGeomType_Triangles,
      .vertexFormat = lvk::VertexFormat::Float3,
      .vertexBuffer = vb0_,
      .vertexStride = sizeof(VertexData),
      .numVertices = (uint32_t)vertexData_.size(),
      .indexFormat = lvk::IndexFormat_UI32,
      .indexBuffer = ib0_,
      .transformBuffer = transformBuffer,
      .buildRange = {.primitiveCount = totalPrimitiveCount},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "BLAS",
  };
  auto blasSizes = ctx_->getAccelStructSizes(blasDesc);
  LLOGL("Full model BLAS sizes buildScratchSize = %llu bytes, accelerationStructureSize = %llu\n",
        blasSizes.buildScratchSize, blasSizes.accelerationStructureSize);
  const auto maxStorageBufferSize = ctx_->getMaxStorageBufferSize();

  // Calculate number of BLAS
  uint32_t requiredBlasCount = 1;
  if (maxStorageBufferSize != -1) {
    requiredBlasCount = blasSizes.buildScratchSize / maxStorageBufferSize;
    const auto cnt = blasSizes.accelerationStructureSize / maxStorageBufferSize;
    if (cnt > requiredBlasCount)
      requiredBlasCount = cnt;
    requiredBlasCount++;

    blasDesc.buildRange.primitiveCount = totalPrimitiveCount / requiredBlasCount;
  }

  LVK_ASSERT(requiredBlasCount > 0);
  LLOGL("maxStorageBufferSize = %d bytes, number of BLAS = %d\n", maxStorageBufferSize, requiredBlasCount);

  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(0.05f)));
  BLAS_.reserve(requiredBlasCount);

  std::vector<lvk::AccelStructInstance> instances;
  instances.reserve(requiredBlasCount);
  const auto primitiveCount = blasDesc.buildRange.primitiveCount;
  for (int i = 0; i < totalPrimitiveCount; i += (int)primitiveCount) {
    const auto rest = (int)totalPrimitiveCount - i;
    blasDesc.buildRange.primitiveOffset = (uint32_t)i * 3 * sizeof(uint32_t);
    blasDesc.buildRange.primitiveCount = (primitiveCount < rest) ? primitiveCount : rest;
    BLAS_.emplace_back(ctx_->createAccelerationStructure(blasDesc));
    instances.emplace_back(lvk::AccelStructInstance{
        // clang-format off
        .transform = (const lvk::mat3x4&)transform,
        // clang-format on
        .instanceCustomIndex = 0,
        .mask = 0xff,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
        .accelerationStructureReference = ctx_->gpuAddress(BLAS_.back()),
    });
  }

  // Buffer for instance data
  sbInstances_ = ctx_->createBuffer(lvk::BufferDesc{
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance) * instances.size(),
      .data = instances.data(),
      .debugName = "sbInstances_",
  });

  TLAS_ = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = sbInstances_,
      .buildRange = {.primitiveCount = (uint32_t)instances.size()},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
  });

  return true;
}

void createPipelines() {
  if (!rayTracingPipeline_.valid()) {
    smRaygen_ = ctx_->createShaderModule(
        {codeRayGen, lvk::Stage_RayGen, "Shader Module: main (raygen)"});
    smMiss_ = ctx_->createShaderModule({codeMiss, lvk::Stage_Miss, "Shader Module: main (miss)"});
    smMissShadow_ = ctx_->createShaderModule({codeMissShadow, lvk::Stage_Miss, "Shader Module: main (miss shadow)"});
    smHit_ = ctx_->createShaderModule(
        {codeClosestHit, lvk::Stage_ClosestHit, "Shader Module: main (closesthit)"});

    rayTracingPipeline_ = ctx_->createRayTracingPipeline(lvk::RayTracingPipelineDesc{
        .smRayGen = {smRaygen_},
        .smClosestHit = {smHit_},
        .smMiss = {smMiss_, smMissShadow_},
    });
  }

  if (!renderPipelineState_Fullscreen_.valid()) {
    smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
    smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
    const lvk::RenderPipelineDesc desc = {
        .smVert = smFullscreenVert_,
        .smFrag = smFullscreenFrag_,
        .color = {{.format = ctx_->getFormat(fbMain_.color[0].texture)}},
        .cullMode = lvk::CullMode_None,
        .debugName = "Pipeline: fullscreen",
    };
    renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(desc, nullptr);
  }
}

void createRayTracingOutputImage() {
  const lvk::TextureDesc descColor = {
      .type = lvk::TextureType_2D,
      .format = lvk::Format_BGRA_UN8,
      .dimensions = {(uint32_t)width_, (uint32_t)height_},
      .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
      .debugName = "Ray-Tracing Output Image",
  };
  rayTracingOutputImage_ = ctx_->createTexture(descColor);
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
  createRayTracingOutputImage();
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
      .viewInverse = glm::inverse(camera_.getViewMatrix()),
      .projInverse = glm::inverse(glm::perspective(fov, aspectRatio, 0.5f, 500.0f)),
  };
  ctx_->upload(ubPerFrame_[frameIndex], &perFrame_, sizeof(perFrame_));

  // Pass 1: ray-trace the scene
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    struct {
        vec4 lightDir;
        uint64_t perFrame;
        uint64_t materials;
        uint64_t indices;
        uint64_t vertices;
        uint32_t outTexture;
        uint32_t tlas;
    } pc = {
        .lightDir = vec4(lightDir_, 0.0f),
        .perFrame = ctx_->gpuAddress(ubPerFrame_[frameIndex]),
        .materials = ctx_->gpuAddress(sbMaterials_),
        .indices = ctx_->gpuAddress(ib0_),
        .vertices = ctx_->gpuAddress(vb0_),
        .outTexture = rayTracingOutputImage_.index(),
        .tlas = TLAS_.index(),
    };

    buffer.cmdBindRayTracingPipeline(rayTracingPipeline_);
    buffer.cmdPushConstants(pc);
    buffer.cmdTraceRays((uint32_t)width_, (uint32_t)height_, 1,
                        {.textures = {lvk::TextureHandle(rayTracingOutputImage_)}});
    ctx_->submit(buffer);
  }

  // Pass 2: render into the swapchain image
  {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassMain_, fbMain_, {.textures = {rayTracingOutputImage_}});
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
      buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
      buffer.cmdBindDepthState({});
      struct {
        uint32_t texture;
      } bindings = {
          .texture = rayTracingOutputImage_.index(),
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
      width_ = ANativeWindow_getWidth(app->window) / kFramebufferScalar;
      height_ = ANativeWindow_getHeight(app->window) / kFramebufferScalar;
      ctx_ = lvk::createVulkanContextWithSwapchain(app->window,
                                                   width_,
                                                   height_,
                                                   {
                                                     .enableValidation = kEnableValidationLayers,
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
  int w = ANativeWindow_getWidth(window) / kFramebufferScalar;
  int h = ANativeWindow_getHeight(window) / kFramebufferScalar;
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
