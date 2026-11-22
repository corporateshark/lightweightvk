/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm Neural Super Sampling (NSS) on top of VK_ARM_tensors and VK_ARM_data_graph, fully bindless.
 *
 * The window opens at the size of the monitor (`--width W --height H`, or the shorthands `--1080p` and `--4k`), and the
 * Bistro scene is rendered at half of it with subpixel jitter into color, depth and motion vector targets and upscaled 2x to
 * it by the NSS v1 pipeline, presented 1:1 (Arm ships the network shaped for 960x540, and the sample reshapes it at start-up):
 *
 *   depth scatter (compute) -> preprocess (compute, writes the int8 input tensor) -> data graph (VK_ARM_data_graph,
 *   the neural network from a VGF file) -> postprocess (compute, kernel prediction filter + temporal accumulation), plus a
 *   feedback-to-texture copy (compute) where the int8 feedback tensor cannot alias its texture
 *
 * The NSS algorithm comes from a fork of the Arm Neural Graphics SDK that binds NSS resources bindlessly, deployed by
 * `deploy_deps.py`; the compute entry points and the shader permutation are compiled into this file. The Arm ML Emulation
 * Layer for Vulkan is used when LightweightVK was configured with `LVK_WITH_ML_EMULATION_LAYER=ON` (run with
 * `--no-ml-emulation` to disable it). `--no-nss` starts with NSS disabled,
 * `--camera-rotate` rotates the camera continuously (temporal stability test; combine with `--screenshot-frame`),
 * `--ao-samples N` sets the number of ambient occlusion rays per pixel (6 by default, 0 disables it), `--timings` also prints
 * the GPU time of every stage to the log.
 * `--full-res` starts with the scene rendered at the output resolution and NSS off, which is what NSS is compared
 * against; the "F" key and the render mode combo box switch between the two at run time.
 *
 * 1) Run the script "deploy_deps.py" from the LightweightVK root folder.
 * 2) Run the script "deploy_content.py" from the LightweightVK root folder.
 * 3) Run this app.
 */

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdio.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#define FFX_CPU 1
#include <neural-graphics-sdk/sdk/include/FidelityFX/gpu/nss/ffx_nss_resources.h>
#undef FFX_CPU

#include <lvk/HelpersImGui.h>
#include <lvk/HelpersVgf.h>
#include <lvk/LVK.h>

#include "Bistro.h"
#include "VulkanApp.h"

#define MODEL_PATH "src/bistro/Exterior/exterior.obj"

uint32_t kRenderWidth = 1280;
uint32_t kRenderHeight = 720;
uint32_t kOutputWidth = 2560;
uint32_t kOutputHeight = 1440;
const char* kNssModelFile = "src/neural/nss/2_nss-960x540-v1_0_1.vgf"; // reshaped at start-up for other resolutions
constexpr float kCameraNear = 0.5f;
constexpr float kCameraFar = 500.0f;
constexpr float kCameraFovY = float(45.0f * (M_PI / 180.0f));
constexpr float kSceneScale = 0.05f;

std::vector<std::string> folderShadersNSS;

const VulkanApp* app_ = nullptr;

// clang-format off

#define PER_FRAME_GLSL                                            \
  "layout(std430, buffer_reference) readonly buffer PerFrame {\n" \
  "  mat4 proj;\n"                                                \
  "  mat4 projNoJitter;\n"                                        \
  "  mat4 view;\n"                                                \
  "  mat4 viewProjPrev;\n"                                        \
  "  mat4 skyboxViewProjPrev;\n"                                  \
  "  mat4 light;\n"                                               \
  "  uint texSkyboxRadiance;\n"                                   \
  "  uint texSkyboxIrradiance;\n"                                 \
  "  uint texShadow;\n"                                           \
  "  uint sampler0;\n"                                            \
  "  uint samplerShadow0;\n"                                      \
  "};\n"

#define MOTION_VECTOR_GLSL                                 \
  "vec2 clipToUv(vec4 clip) {\n"                           \
  "  const vec2 ndc = clip.xy / clip.w;\n"                 \
  "  return vec2(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y);\n" \
  "}\n"                                                    \
  "vec2 motionVector(vec4 clipCur, vec4 clipPrev) {\n"     \
  "  return clipToUv(clipPrev) - clipToUv(clipCur);\n"     \
  "}\n"

const char* kAOPushConstants = R"(
  uint tlas;
  uint aoSamples;
  uint frameId;
  float aoRadius;
  float aoPower;
)";

const char* kAODisabled = R"(
float ambientOcclusion(vec3 worldPos, vec3 n, uint tlas, uint samples, uint frameId, float radius, float power) {
  return 1.0;
}
)";

const char* kAOEnabled = R"(
float traceAO(rayQueryEXT rq, uint tlas, vec3 origin, vec3 dir, float radius) {
  rayQueryInitializeEXT(rq, kTLAS[tlas], gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, origin, 0.0, dir, radius);
  while (rayQueryProceedEXT(rq)) {}
  return (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) ? 1.0 : 0.0;
}

uint lcg(inout uint prev) {
  prev = 1664525u * prev + 1013904223u;
  return prev & 0x00FFFFFFu;
}

float rnd(inout uint seed) {
  return float(lcg(seed)) / float(0x01000000);
}

uint tea(uint val0, uint val1) {
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;
  for (uint n = 0; n < 16; n++) {
    s0 += 0x9e3779b9u;
    v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
    v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
  }
  return v0;
}

void computeTBN(in vec3 n, out vec3 x, out vec3 y) {
  const float yz = -n.y * n.z;
  y = normalize((abs(n.z) > 0.9999) ? vec3(-n.x * n.y, 1.0 - n.y * n.y, yz) : vec3(-n.x * n.z, yz, 1.0 - n.z * n.z));
  x = cross(y, n);
}

vec3 sampleCosineHemisphere(inout uint seed, vec3 tangent, vec3 bitangent, vec3 n) {
  const float r1 = rnd(seed);
  const float r2 = rnd(seed);
  const float sq = sqrt(1.0 - r2);
  const float phi = 2.0 * 3.141592653589 * r1;
  const vec3 d = vec3(cos(phi) * sq, sin(phi) * sq, sqrt(r2));
  return d.x * tangent + d.y * bitangent + d.z * n;
}

float ambientOcclusion(vec3 worldPos, vec3 n, uint tlas, uint samples, uint frameId, float radius, float power) {
  if (samples == 0)
    return 1.0;
  const vec3 origin = worldPos + n * 0.001;
  vec3 tangent, bitangent;
  computeTBN(n, tangent, bitangent);
  uint seed = tea(uint(gl_FragCoord.y * 4003.0 + gl_FragCoord.x), frameId);
  float occlusion = 0.0;
  for (uint i = 0; i < samples; i++) {
    rayQueryEXT rq;
    occlusion += traceAO(rq, tlas, origin, sampleCosineHemisphere(seed, tangent, bitangent, n), radius);
  }
  return pow(clamp(1.0 - occlusion / float(samples), 0.0, 1.0), power);
}
)";

const char* kCodeVSHead = PER_FRAME_GLSL BISTRO_GLSL R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

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
)";

const char* kCodeVSTail = R"(
} pc;

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
  vec3 worldPos;
};
layout (location=0) out PerVertex vtx;
layout (location=4) out vec4 clipCur;
layout (location=5) out vec4 clipPrev;
layout (location=6) flat out Material mtl;

void main() {
  mat4 model = pc.perObject.model;
  mtl = pc.materials.mtl[mtlIndex];
  const vec4 worldPos = model * vec4(pos, 1.0);
  const vec4 viewPos = pc.perFrame.view * worldPos;
  gl_Position = pc.perFrame.proj * viewPos;
  clipCur = pc.perFrame.projNoJitter * viewPos;
  clipPrev = pc.perFrame.viewProjPrev * worldPos;

  vtx.normal = normalize(mat3(pc.perObject.normal) * unpackOctahedral16(normal));
  vtx.uv = uv;
  vtx.shadowCoords = pc.perFrame.light * worldPos;
  vtx.worldPos = worldPos.xyz;
}
)";

const char* kCodeFSHead = PER_FRAME_GLSL MOTION_VECTOR_GLSL BISTRO_GLSL R"(
struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
  vec3 worldPos;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  uvec2 perObject;
  uvec2 materials;
)";

const char* kCodeFSTail = R"(
} pc;

layout (location=0) in PerVertex vtx;
layout (location=4) in vec4 clipCur;
layout (location=5) in vec4 clipPrev;
layout (location=6) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;
layout (location=1) out vec2 out_Motion;

float PCF3(vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame.texShadow).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(pc.perFrame.texShadow, pc.perFrame.samplerShadow0, uvw + size * vec3(u, v, 0));
  return shadow / 9;
}

float shadow(vec4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float depthBias = -0.00005;
    float shadowSample = PCF3(vec3(s.x, 1.0 - s.y, s.z + depthBias));
    return mix(0.3, 1.0, shadowSample);
  }
  return 1.0;
}

void main() {
  vec4 alpha = textureBindless2D(mtl.texAlpha, pc.perFrame.sampler0, vtx.uv);
  if (mtl.texAlpha > 0 && alpha.r < 0.5)
    discard;
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, pc.perFrame.sampler0, vtx.uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, pc.perFrame.sampler0, vtx.uv);
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = textureBindlessCube(pc.perFrame.texSkyboxIrradiance, pc.perFrame.sampler0, n) * Kd * (vec4(1.0) - f0);
  const float ao = ambientOcclusion(vtx.worldPos, n, pc.tlas, pc.aoSamples, pc.frameId, pc.aoRadius, pc.aoPower);
  out_FragColor = vec4((Ka + diffuse * shadow(vtx.shadowCoords) * ao).rgb, 1.0);
  out_Motion = motionVector(clipCur, clipPrev);
};
)";

const char* kShadowVS = PER_FRAME_GLSL R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;

void main() {
  gl_Position = pc.perFrame.proj * pc.perFrame.view * pc.perObject.model * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kSkyboxVS = PER_FRAME_GLSL R"(
layout (location=0) out vec3 textureCoords;
layout (location=1) out vec4 clipCur;
layout (location=2) out vec4 clipPrev;

const vec3 positions[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

const int indices[36] = int[36](
	0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
);

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;

void main() {
  mat4 view = pc.perFrame.view;
  view = mat4(view[0], view[1], view[2], vec4(0, 0, 0, 1));
  vec3 pos = positions[indices[gl_VertexIndex]];
  gl_Position = (pc.perFrame.proj * view * vec4(pos, 1.0)).xyww;
  clipCur = pc.perFrame.projNoJitter * view * vec4(pos, 1.0);
  clipPrev = pc.perFrame.skyboxViewProjPrev * vec4(pos, 1.0);

  textureCoords = pos;
}

)";
const char* kSkyboxFS = PER_FRAME_GLSL MOTION_VECTOR_GLSL R"(
layout (location=0) in vec3 textureCoords;
layout (location=1) in vec4 clipCur;
layout (location=2) in vec4 clipPrev;
layout (location=0) out vec4 out_FragColor;
layout (location=1) out vec2 out_Motion;

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

void main() {
  out_FragColor = vec4(textureBindlessCube(pc.perFrame.texSkyboxRadiance, pc.perFrame.sampler0, textureCoords).rgb, 1.0);
  out_Motion = motionVector(clipCur, clipPrev);
}
)";

const char* kCodeFullscreenVS = FULLSCREEN_TRIANGLE_GLSL R"(
layout (location=0) out vec2 uv;

layout(push_constant) uniform constants {
  vec2 clipRotation;
  uint tex;
  uint smp;
} pc;

void main() {
  gl_Position = fullscreenTriangle(pc.clipRotation, uv);
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
  vec2 clipRotation;
  uint tex;
  uint smp;
  vec2 uvScale;
} pc;

void main() {
  out_FragColor = vec4(textureBindless2D(pc.tex, pc.smp, uv * pc.uvScale).rgb, 1.0);
}
)";

#define NSS_CONFIG_GLSL R"(
#define FFX_GPU 1
#define FFX_GLSL 1
#define FFX_HALF 1

#define QUANTIZED 1
#define REVERSE_Z 0
#define NSS_SHADER_QUALITY_MODE 0
#define SCALE_PRESET_MODE NSS_SCALE_PRESET_X2
#define NSS_SUPPORT_TENSOR 0
#define NSS_BINDLESS 1
#define MANAGE_HISTORY 0
)"

const char* kNssDepthScatter = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_DEPTH 1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 2
#define NSS_BIND_UAV_DEPTH_TM1 3
#define NSS_BIND_CB_NSS 4

#include "ffx_nss_depth_scatter.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  DepthScatter(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssPreprocess = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_COLOR_JITTERED 0
#define NSS_BIND_SRV_INPUT_DEPTH 1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 2
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR 4
#define NSS_BIND_SRV_INPUT_DEPTH_TM1 5
#define NSS_BIND_SRV_LUMA_DERIV_TM1 6
#define NSS_BIND_PREPROCESS_INPUT_TENSOR 8
#define NSS_BIND_UAV_LUMA_DERIV 9
#define NSS_BIND_UAV_NEAREST_DEPTH_COORD 10
#define NSS_BIND_CB_NSS 12

#define NSS_PREPROCESS 1

#include "ffx_nss_preprocess.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  Preprocess(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssPostprocess = NSS_CONFIG_GLSL R"(

#define NSS_BIND_SRV_INPUT_COLOR_JITTERED 0
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 1
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 2
#define NSS_BIND_KPN_TENSOR 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR 4
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD 5
#define NSS_BIND_UAV_UPSCALED_OUTPUT 7
#define NSS_BIND_CB_NSS 9

#define NSS_POSTPROCESS 1

#include "ffx_nss_postprocess.h"

layout(local_size_x = FFX_NSS_THREAD_GROUP_WIDTH, local_size_y = FFX_NSS_THREAD_GROUP_HEIGHT, local_size_z = FFX_NSS_THREAD_GROUP_DEPTH) in;

void main() {
  Postprocess(int32_t2(gl_GlobalInvocationID.xy));
}
)";

const char* kNssFeedbackToTexture = NSS_CONFIG_GLSL R"(
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 2, rgba8_snorm) uniform writeonly image2D kImagesRGBA8Snorm[];

layout(std430, buffer_reference) readonly buffer FeedbackTensor {
  i8vec4 texels[];
};

layout(push_constant) uniform PushConstants {
  FeedbackTensor tensor;
  uint image;
  uint width;
  uint height;
} pc;

void main() {
  const uvec2 p = gl_GlobalInvocationID.xy;
  if (p.x >= pc.width || p.y >= pc.height) {
    return;
  }
  const vec4 snorm = max(vec4(pc.tensor.texels[p.y * pc.width + p.x]) / 127.0, vec4(-1.0));
  imageStore(kImagesRGBA8Snorm[pc.image], ivec2(p), snorm);
}
)";

// clang-format on

lvk::IContext* ctx_ = nullptr;

// the scene resources, destroyed by destroyScene()
struct SceneResources {
  lvk::Holder<lvk::BufferHandle> vb0, ib0;
  SceneAccelStructs accel;
  lvk::Holder<lvk::BufferHandle> sbMaterials;
  lvk::Holder<lvk::BufferHandle> ubPerFrame, ubPerFrameShadow, ubPerObject;
  lvk::Holder<lvk::TextureHandle> texShadowMap;
  lvk::Holder<lvk::TextureHandle> texColorLR, texMotionLR, texDepthLR;
  lvk::Holder<lvk::TextureHandle> skyboxReference, skyboxIrradiance;
  lvk::Holder<lvk::SamplerHandle> sampler, samplerShadow, samplerLinearClamp;
  lvk::Holder<lvk::ShaderModuleHandle> smMeshVert, smMeshFrag;
  lvk::Holder<lvk::ShaderModuleHandle> smShadowVert, smShadowFrag;
  lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert, smFullscreenFrag;
  lvk::Holder<lvk::ShaderModuleHandle> smSkyboxVert, smSkyboxFrag;
  lvk::Holder<lvk::RenderPipelineHandle> pipelineMesh, pipelineShadow, pipelineSkybox, pipelineFullscreen;
} res_;

uint32_t sceneWidth_ = kRenderWidth;
uint32_t sceneHeight_ = kRenderHeight;
enum RenderMode {
  RenderMode_FullResolution = 0,
  RenderMode_HalfResolution,
  RenderMode_HalfResolutionNSS,
  RenderMode_NUM_MODES,
};
int renderMode_ = RenderMode_HalfResolutionNSS;
bool rayQuery_ = false;
int aoSamples_ = 6;
float aoRadius_ = 8.0f;
float aoPower_ = 1.0f;

const lvk::RenderPass renderPassOffscreen_ = {
    .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}},
              {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
    .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
};
const lvk::RenderPass renderPassMain_ = {
    .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
};
const lvk::RenderPass renderPassShadow_ = {
    .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
};
const lvk::DepthState depthState_ = {.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true};
const lvk::DepthState depthStateLEqual_ = {.compareOp = lvk::CompareOp_LessEqual, .isDepthWriteEnabled = true};

bool isShadowMapDirty_ = true;

struct UniformsPerFrame {
  mat4 proj;
  mat4 projNoJitter;
  mat4 view;
  mat4 viewProjPrev;
  mat4 skyboxViewProjPrev;
  mat4 light;
  uint32_t texSkyboxRadiance = 0;
  uint32_t texSkyboxIrradiance = 0;
  uint32_t texShadow = 0;
  uint32_t sampler = 0;
  uint32_t samplerShadow = 0;
};

struct UniformsPerObject {
  mat4 model;
  mat4 normal;
};

bool initModel(VulkanApp& app) {
  const std::string cacheFileName = app.folderContentRoot_ + "cache.data";

  if (!loadFromCache(app, cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(app, cacheFileName.c_str(), MODEL_PATH))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  loadMaterialTextures(app, "src/bistro/Exterior/");

  res_.sbMaterials = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                         .storage = lvk::StorageType_Device,
                                         .size = sizeof(GPUMaterial) * materials_.size(),
                                         .data = materials_.data(),
                                         .debugName = "Buffer: materials"},
                                        nullptr);

  const uint8_t accelStructUsage = rayQuery_ ? lvk::BufferUsageBits_AccelStructBuildInputReadOnly : 0;
  res_.vb0 = ctx_->createBuffer({.usage = uint8_t(lvk::BufferUsageBits_Vertex | accelStructUsage),
                                 .storage = lvk::StorageType_Device,
                                 .size = sizeof(VertexData) * vertexData_.size(),
                                 .data = vertexData_.data(),
                                 .debugName = "Buffer: vertex"},
                                nullptr);
  res_.ib0 = ctx_->createBuffer({.usage = uint8_t(lvk::BufferUsageBits_Index | accelStructUsage),
                                 .storage = lvk::StorageType_Device,
                                 .size = sizeof(uint32_t) * indexData_.size(),
                                 .data = indexData_.data(),
                                 .debugName = "Buffer: index"},
                                nullptr);
  if (!rayQuery_) {
    return true;
  }
  const uint32_t numTriangles = uint32_t(indexData_.size()) / 3;
  res_.accel = createSceneAccelStructs(ctx_, res_.vb0, uint32_t(vertexData_.size()), res_.ib0, numTriangles, numTriangles, kSceneScale);
  LLOGL("Ray-traced ambient occlusion: %zu BLAS over %u triangles\n", res_.accel.BLAS.size(), numTriangles);
  return res_.accel.TLAS.valid();
}

void createOffscreenFramebuffer() {
  res_.texColorLR = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_F16,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR color (jittered)",
  });
  res_.texMotionLR = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RG_F16,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR motion vectors",
  });
  res_.texDepthLR = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_F32,
      .dimensions = {sceneWidth_, sceneHeight_},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "LR depth",
  });
}

void createPipelines() {
  const lvk::VertexInput vdesc = {
      .attributes =
          {
              {.location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)},
              {.location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv)},
              {.location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal)},
              {.location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex)},
          },
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  const lvk::VertexInput vdescs = {
      .attributes = {{.format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  const std::string codeVert = std::string(kCodeVSHead) + kAOPushConstants + kCodeVSTail;
  const std::string codeFrag = std::string(rayQuery_ ? kAOEnabled : kAODisabled) + kCodeFSHead + kAOPushConstants + kCodeFSTail;
  res_.smMeshVert = ctx_->createShaderModule({codeVert.c_str(), lvk::Stage_Vert, "Shader Module: main (vert)"});
  res_.smMeshFrag = ctx_->createShaderModule({codeFrag.c_str(), lvk::Stage_Frag, "Shader Module: main (frag)"});
  res_.smShadowVert = ctx_->createShaderModule({kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)"});
  res_.smShadowFrag = ctx_->createShaderModule({kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
  res_.smFullscreenVert = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  res_.smFullscreenFrag = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
  res_.smSkyboxVert = ctx_->createShaderModule({kSkyboxVS, lvk::Stage_Vert, "Shader Module: skybox (vert)"});
  res_.smSkyboxFrag = ctx_->createShaderModule({kSkyboxFS, lvk::Stage_Frag, "Shader Module: skybox (frag)"});

  res_.pipelineMesh = ctx_->createRenderPipeline(
      {
          .vertexInput = vdesc,
          .smVert = res_.smMeshVert,
          .smFrag = res_.smMeshFrag,
          .color = {{.format = ctx_->getFormat(res_.texColorLR)}, {.format = ctx_->getFormat(res_.texMotionLR)}},
          .depthFormat = ctx_->getFormat(res_.texDepthLR),
          .cullMode = lvk::CullMode_Back,
          .frontFace = lvk::WindingMode_CCW,
          .debugName = "Pipeline: mesh",
      },
      nullptr);

  res_.pipelineShadow = ctx_->createRenderPipeline(
      {
          .vertexInput = vdescs,
          .smVert = res_.smShadowVert,
          .smFrag = res_.smShadowFrag,
          .depthFormat = ctx_->getFormat(res_.texShadowMap),
          .cullMode = lvk::CullMode_None,
          .debugName = "Pipeline: shadow",
      },
      nullptr);

  res_.pipelineFullscreen = ctx_->createRenderPipeline(
      {
          .smVert = res_.smFullscreenVert,
          .smFrag = res_.smFullscreenFrag,
          .color = {{.format = ctx_->getSwapchainFormat()}},
          .cullMode = lvk::CullMode_None,
          .debugName = "Pipeline: fullscreen",
      },
      nullptr);

  res_.pipelineSkybox = ctx_->createRenderPipeline(
      {
          .smVert = res_.smSkyboxVert,
          .smFrag = res_.smSkyboxFrag,
          .color = {{.format = ctx_->getFormat(res_.texColorLR)}, {.format = ctx_->getFormat(res_.texMotionLR)}},
          .depthFormat = ctx_->getFormat(res_.texDepthLR),
          .cullMode = lvk::CullMode_Front,
          .frontFace = lvk::WindingMode_CCW,
          .debugName = "Pipeline: skybox",
      },
      nullptr);
}

bool initScene(VulkanApp& app) {
  rayQuery_ = ctx_->supportsRayQuery();
  if (!rayQuery_) {
    LLOGW("VK_KHR_ray_query is not available: rendering without ambient occlusion\n");
  }

  auto createUniformBuffer = [](size_t size, const char* debugName) {
    return ctx_->createBuffer({
        .usage = lvk::BufferUsageBits_Uniform,
        .storage = lvk::StorageType_HostVisible,
        .size = size,
        .debugName = debugName,
    });
  };
  res_.ubPerFrame = createUniformBuffer(sizeof(UniformsPerFrame), "Buffer: uniforms (per frame)");
  res_.ubPerFrameShadow = createUniformBuffer(sizeof(UniformsPerFrame), "Buffer: uniforms (per frame shadow)");
  res_.ubPerObject = createUniformBuffer(sizeof(UniformsPerObject), "Buffer: uniforms (per object)");

  res_.sampler = ctx_->createSampler({
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
  });
  res_.samplerShadow = ctx_->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .depthCompareOp = lvk::CompareOp_LessEqual,
      .depthCompareEnabled = true,
      .debugName = "Sampler: shadow",
  });
  res_.samplerLinearClamp = ctx_->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .debugName = "Sampler: linear clamp",
  });

  res_.texShadowMap = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_UN16,
      .dimensions = {4096, 4096},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .debugName = "Shadow map",
  });
  createOffscreenFramebuffer();
  createPipelines();

  // `loadMaterialTextures()` in initModel() flips stb_image vertically, so the skybox has to be loaded first
  loadSkyboxTextures(app, res_.skyboxReference, res_.skyboxIrradiance);

  return initModel(app);
}

void destroyScene() {
  LLOGL("Waiting for the loader thread to exit...\n");
  cancelLoadingMaterialTextures();

  res_ = {};
}

struct NssConstants {
  float deviceToViewDepth[4];
  float jitterOffset[4];
  float jitterOffsetTm1[4];
  float scaleFactor[4];
  int32_t outputDims[2];
  int32_t inputDims[2];
  float invOutputDims[2];
  float invInputDims[2];
  int32_t depthTm1Size[2];
  float invDepthTm1Size[2];
  int32_t inputTensorSize[2];
  float inputTensorSizeRcp[2];
  int32_t kpnDimension[2];
  float motionVectorScale[2];
  float paddingScale[2];
  float depthClipRequiredSepScale;
  float depthClipPower;
  float kpnScale[2];
  int32_t debugViewMode;
  float notHistoryReset;
  float exposure[2];
  int32_t indexModulo[2];
  int32_t reducedInputModulo[2];
  int32_t lutOffset[2];
};

static_assert(sizeof(NssConstants) == 208);

struct NssPushConstants {
  uint64_t constants;
  uint32_t samplerPoint;
  uint32_t samplerLinear;
  uint32_t texColor;
  uint32_t texDepth;
  uint32_t texMotion;
  uint32_t texHistory;
  uint32_t texFeedback;
  uint32_t texLumaDerivTm1;
  uint32_t texDepthTm1;
  uint32_t texNearestDepthCoord;
  uint32_t imgDepthTm1;
  uint32_t imgLumaDeriv;
  uint32_t imgNearestDepthCoord;
  uint32_t imgOutput;
  uint64_t tensorInput;
  uint64_t tensorKpn;
};

std::string expandIncludes(const std::string& text, std::vector<std::string>& included, bool& ok);

std::string loadShaderWithIncludes(const std::string& fileName, std::vector<std::string>& included, bool& ok) {
  included.push_back(std::filesystem::path(fileName).filename().string());

  std::vector<uint8_t> blob;
  for (const std::string& root : folderShadersNSS) {
    blob = app_->loadFile((root + fileName).c_str());
    if (!blob.empty()) {
      break;
    }
  }
  if (blob.empty()) {
    LLOGW("Cannot open shader `%s`\n", fileName.c_str());
    ok = false;
    return {};
  }
  return expandIncludes(std::string(blob.begin(), blob.end()), included, ok);
}

std::string expandIncludes(const std::string& text, std::vector<std::string>& included, bool& ok) {
  std::string result;
  size_t lineStart = 0;
  while (lineStart <= text.size()) {
    size_t lineEnd = text.find('\n', lineStart);
    if (lineEnd == std::string::npos) {
      lineEnd = text.size();
    }
    std::string line = text.substr(lineStart, lineEnd - lineStart);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    lineStart = lineEnd + 1;
    const size_t pos = line.find("#include \"");
    if (pos != std::string::npos && line.find_first_not_of(" \t") == pos) {
      const size_t start = pos + strlen("#include \"");
      const size_t end = line.find('"', start);
      const std::string includeName = line.substr(start, end - start);
      const std::string includeFileName = std::filesystem::path(includeName).filename().string();
      if (std::find(included.begin(), included.end(), includeFileName) == included.end()) {
        result += "// >>> " + includeName + "\n";
        result += loadShaderWithIncludes(includeName, included, ok);
        result += "// <<< " + includeName + "\n";
      }
      continue;
    }
    result += line;
    result += '\n';
  }
  return result;
}

struct NeuralSuperSampling {
  enum {
    kAlignment = FFX_NSS_RESOURCE_ALIGNMENT,
    kThreadGroup = FFX_NSS_THREAD_GROUP_WIDTH,
    kKpnChannels = 36,
    kInputChannels = 12,
    kFeedbackChannels = 4,
    kJitterPhaseCount = 8 * 2 * 2,
  };
  static_assert(FFX_NSS_THREAD_GROUP_WIDTH == FFX_NSS_THREAD_GROUP_HEIGHT, "The dispatches below assume square thread groups");
  inline static uint32_t kGraphWidth = 0;
  inline static uint32_t kGraphHeight = 0;
  inline static uint32_t kKpnWidth = 0;
  inline static uint32_t kKpnHeight = 0;
  inline static uint32_t kDepthTm1Width = 0;
  inline static uint32_t kDepthTm1Height = 0;

  static void configure() {
    kGraphWidth = (kRenderWidth + kAlignment - 1) / kAlignment * kAlignment;
    kGraphHeight = (kRenderHeight + kAlignment - 1) / kAlignment * kAlignment;
    kKpnWidth = kGraphWidth / 4;
    kKpnHeight = kGraphHeight / 4;
    kDepthTm1Width = kRenderWidth / 2;
    kDepthTm1Height = kRenderHeight / 2;
  }

  // Arm publishes the network shaped for 960x540. For another resolution the shapes come from TOSA inference over the
  // graph: CONV2D and RESIZE move H and W, CONCAT moves the channels, RESCALE and TABLE keep the shape. Every operator
  // result type has a shape constant of its own, which gets the inferred shape in place; nothing else is touched
  static bool reshapeGraph(const uint32_t* words,
                           uint32_t numWords,
                           const glm::ivec4& inputShape,
                           std::vector<uint32_t>& outWords,
                           glm::ivec4 (&outShapes)[2]) {
    enum {
      SpvOpExtInst = 12,
      SpvOpTypeInt = 21,
      SpvOpConstant = 43,
      SpvOpConstantComposite = 44,
      SpvOpConstantNull = 46,
      SpvOpTypeTensorARM = 4163,
      SpvOpGraphConstantARM = 4181,
      SpvOpGraphInputARM = 4184,
      SpvOpGraphSetOutputARM = 4185,
      SpvOpConstantCompositeReplicateEXT = 4461,
      TOSA_CONV2D = 2,
      TOSA_TABLE = 30,
      TOSA_CONCAT = 54,
      TOSA_RESIZE = 63,
      TOSA_RESCALE = 65,
    };
    if (numWords < 5) {
      return false;
    }
    const uint32_t bound = words[3];
    std::vector<uint32_t> definitions(bound, 0); // the offset of the instruction defining an id
    std::vector<glm::ivec4> shapes(bound, glm::ivec4(0)); // x == 0 for an id without a shape
    uint32_t typeUint = 0;
    uint32_t endOfTypeUint = 0;
    uint32_t outputs[2] = {};

    auto opcode = [&](uint32_t id) -> uint32_t { return words[definitions[id]] & 0xFFFF; };
    // the i-th element of a constant scalar, tensor or null
    auto value = [&](uint32_t id, uint32_t i) -> int32_t {
      const uint32_t offset = definitions[id];
      const uint32_t scalar = opcode(id) == SpvOpConstantComposite               ? words[offset + 3 + i]
                              : opcode(id) == SpvOpConstantCompositeReplicateEXT ? words[offset + 3]
                                                                                 : id;
      return opcode(id) == SpvOpConstantNull ? 0 : (int32_t)words[definitions[scalar] + 3];
    };
    auto shapeOfType = [&](uint32_t type) -> glm::ivec4 {
      const uint32_t shape = words[definitions[type] + 4];
      return glm::ivec4(value(shape, 0), value(shape, 1), value(shape, 2), value(shape, 3));
    };

    for (uint32_t offset = 5; offset < numWords;) {
      const uint32_t length = words[offset] >> 16;
      const uint32_t op = words[offset] & 0xFFFF;
      if (!length || offset + length > numWords) {
        return false;
      }
      const bool isType = op == SpvOpTypeInt || op == SpvOpTypeTensorARM;
      const bool isValue = op == SpvOpConstant || op == SpvOpConstantComposite || op == SpvOpConstantNull ||
                           op == SpvOpConstantCompositeReplicateEXT || op == SpvOpGraphConstantARM || op == SpvOpGraphInputARM ||
                           op == SpvOpExtInst;
      const uint32_t id = isType ? words[offset + 1] : isValue ? words[offset + 2] : 0;
      if ((isType || isValue) && (!id || id >= bound)) {
        return false;
      }
      definitions[id] = isType || isValue ? offset : 0;
      if (op == SpvOpTypeInt && words[offset + 2] == 32 && !words[offset + 3]) {
        typeUint = id;
        endOfTypeUint = offset + length;
      }
      if (op == SpvOpGraphInputARM) {
        shapes[id] = inputShape;
      }
      if (op == SpvOpGraphSetOutputARM) {
        const int32_t index = value(words[offset + 2], 0);
        if (index < 0 || index > 1) {
          return false;
        }
        outputs[index] = words[offset + 1];
      }
      if (op == SpvOpExtInst) {
        const uint32_t* operands = words + offset + 5;
        const uint32_t instruction = words[offset + 4];
        const uint32_t source = instruction == TOSA_CONV2D    ? operands[5]
                                : instruction == TOSA_RESIZE  ? operands[1]
                                : instruction == TOSA_RESCALE ? operands[5]
                                : instruction == TOSA_CONCAT  ? operands[1]
                                                              : operands[0];
        const glm::ivec4 src = shapes[source];
        if (!src.x) {
          LLOGW("NSS: TOSA operator %u reads a value of unknown shape\n", instruction);
          return false;
        }
        if (instruction == TOSA_CONV2D) {
          const glm::ivec4 kernel = shapeOfType(words[definitions[operands[6]] + 1]);
          glm::ivec4 result(src.x, 0, 0, kernel.x);
          for (int32_t i = 0; i != 2; i++) {
            const int32_t padded =
                src[1 + i] + value(operands[0], 2 * i) + value(operands[0], 2 * i + 1) - value(operands[2], i) * (kernel[1 + i] - 1) - 1;
            const int32_t stride = value(operands[1], i);
            if (padded < 0 || padded % stride) {
              LLOGW("NSS: CONV2D on %ix%i does not produce whole dimensions: pick a resolution the network divides\n", src.z, src.y);
              return false;
            }
            result[1 + i] = padded / stride + 1;
          }
          shapes[id] = result;
        } else if (instruction == TOSA_RESIZE) {
          glm::ivec4 result = src;
          for (int32_t i = 0; i != 2; i++) {
            const int32_t scaled = (src[1 + i] - 1) * value(operands[2], 2 * i) - value(operands[3], i) + value(operands[4], i);
            const int32_t denominator = value(operands[2], 2 * i + 1);
            if (scaled < 0 || scaled % denominator) {
              LLOGW("NSS: RESIZE on %ix%i does not produce whole dimensions\n", src.z, src.y);
              return false;
            }
            result[1 + i] = scaled / denominator + 1;
          }
          shapes[id] = result;
        } else if (instruction == TOSA_CONCAT) {
          const int32_t axis = value(operands[0], 0);
          glm::ivec4 result = src;
          for (uint32_t i = 2; i != length - 5; i++) {
            result[axis] += shapes[operands[i]][axis];
          }
          shapes[id] = result;
        } else if (instruction == TOSA_RESCALE || instruction == TOSA_TABLE) {
          shapes[id] = src;
        } else {
          LLOGW("NSS: unexpected TOSA operator %u\n", instruction);
          return false;
        }
      }
      offset += length;
    }
    if (!typeUint || !outputs[0] || !outputs[1]) {
      return false;
    }

    // the shape constants of the result types, rewritten with the uint constants declared before them or new ones declared
    // right after the uint type
    std::vector<glm::uvec2> constants; // (value, id)
    for (uint32_t id = 0; id != bound; id++) {
      const bool isUint = definitions[id] && opcode(id) == SpvOpConstant && words[definitions[id] + 1] == typeUint;
      if (isUint) {
        constants.push_back(glm::uvec2(words[definitions[id] + 3], id));
      }
    }
    std::vector<uint32_t> newConstants;
    uint32_t nextId = bound;
    auto constantUint = [&](uint32_t v, uint32_t user) -> uint32_t {
      for (const glm::uvec2& c : constants) {
        const bool isDeclaredBefore = c.y >= bound || definitions[c.y] < definitions[user];
        if (c.x == v && isDeclaredBefore) {
          return c.y;
        }
      }
      constants.push_back(glm::uvec2(v, nextId));
      newConstants.insert(newConstants.end(), {(4u << 16) | SpvOpConstant, typeUint, nextId, v});
      return nextId++;
    };
    std::vector<glm::ivec4> newShapes(bound, glm::ivec4(0));
    for (uint32_t id = 0; id != bound; id++) {
      if (!shapes[id].x) {
        continue;
      }
      const uint32_t type = words[definitions[id] + 1];
      const bool isShapedTensor = opcode(type) == SpvOpTypeTensorARM && (words[definitions[type]] >> 16) == 5;
      const uint32_t shape = isShapedTensor ? words[definitions[type] + 4] : 0;
      const bool isShape = shape && opcode(shape) == SpvOpConstantComposite && (words[definitions[shape]] >> 16) == 7;
      const bool isConflicting = isShape && newShapes[shape].x && newShapes[shape] != shapes[id];
      if (!isShape || isConflicting) {
        LLOGW("NSS: the graph does not give every result a shape of its own\n");
        return false;
      }
      newShapes[shape] = shapes[id];
    }
    outWords.assign(words, words + numWords);
    for (uint32_t id = 0; id != bound; id++) {
      for (int32_t i = 0; newShapes[id].x && i != 4; i++) {
        outWords[definitions[id] + 3 + i] = constantUint(newShapes[id][i], id);
      }
    }
    outWords.insert(outWords.begin() + endOfTypeUint, newConstants.begin(), newConstants.end());
    outWords[3] = nextId;
    outShapes[0] = shapes[outputs[0]];
    outShapes[1] = shapes[outputs[1]];
    return true;
  }

  bool supported = false;
  bool resetHistory = true;
  float exposure = 1.0f;
  uint32_t frameIndex = 0;
  vec2 jitter = vec2(0.0f);
  vec2 jitterPrev = vec2(0.0f);

  lvk::Holder<lvk::BufferHandle> bufTensorInput;
  lvk::Holder<lvk::BufferHandle> bufTensorKpn;
  lvk::Holder<lvk::BufferHandle> bufTensorFeedback;
  lvk::Holder<lvk::TextureHandle> texFeedback;
  lvk::Holder<lvk::TensorHandle> tensorInput;
  lvk::Holder<lvk::TensorHandle> tensorKpn;
  lvk::Holder<lvk::TensorHandle> tensorFeedback;
  bool feedbackAliasing = false;
  lvk::Holder<lvk::TextureHandle> texLumaDeriv[2];
  lvk::Holder<lvk::TextureHandle> texNearestDepthCoord;
  lvk::Holder<lvk::TextureHandle> texDepthTm1;
  lvk::Holder<lvk::TextureHandle> texOutput[2];
  lvk::Holder<lvk::SamplerHandle> samplerPoint;
  lvk::Holder<lvk::BufferHandle> bufConstants;
  lvk::Holder<lvk::ShaderModuleHandle> smDepthScatter;
  lvk::Holder<lvk::ShaderModuleHandle> smPreprocess;
  lvk::Holder<lvk::ShaderModuleHandle> smPostprocess;
  lvk::Holder<lvk::ShaderModuleHandle> smFeedbackToTexture;
  lvk::Holder<lvk::ComputePipelineHandle> pipelineDepthScatter;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePreprocess;
  lvk::Holder<lvk::ComputePipelineHandle> pipelinePostprocess;
  lvk::Holder<lvk::ComputePipelineHandle> pipelineFeedbackToTexture;
  lvk::Holder<lvk::DataGraphPipelineHandle> pipelineGraph;
  lvk::TensorHandle graphInputs[1] = {};
  lvk::TensorHandle graphOutputs[2] = {};

  static float halton(int32_t index, int32_t base) {
    float f = 1.0f;
    float result = 0.0f;
    for (int32_t i = index; i > 0; i /= base) {
      f /= (float)base;
      result += f * (float)(i % base);
    }
    return result;
  }

  static vec2 jitterOffset(uint32_t frame) {
    const int32_t index = (int32_t)(frame % kJitterPhaseCount) + 1;
    return vec2(halton(index, 2) - 0.5f, halton(index, 3) - 0.5f);
  }

  bool init(lvk::IContext* ctx) {
    supported = false;

    if (!ctx->supportsTensorsARM() || !ctx->supportsDataGraphARM()) {
      LLOGW("NSS requires VK_ARM_tensors and VK_ARM_data_graph (configure with LVK_WITH_ML_EMULATION_LAYER=ON to emulate them)\n");
      return false;
    }

    lvk::VgfModel vgf;
    const std::vector<uint8_t> blobVGF = app_->loadFile((app_->folderContentRoot_ + kNssModelFile).c_str());
    if (blobVGF.empty() || !vgf.load(blobVGF.data(), blobVGF.size())) {
      LLOGW("NSS: cannot load `%s`. Run `deploy_content.py`.\n", kNssModelFile);
      return false;
    }
    const bool isInt8 = vgf.getNumInputs() == 1 && vgf.getNumOutputs() == 2 && vgf.getInput(0).format == lvk::Format_R_I8 &&
                        vgf.getOutput(0).format == lvk::Format_R_I8 && vgf.getOutput(1).format == lvk::Format_R_I8;
    std::vector<uint32_t> spirv;
    glm::ivec4 outputShapes[2] = {};
    if (!isInt8 || !reshapeGraph((const uint32_t*)vgf.getSpirv(),
                                 (uint32_t)(vgf.getSpirvSize() / sizeof(uint32_t)),
                                 glm::ivec4(1, kGraphHeight, kGraphWidth, kInputChannels),
                                 spirv,
                                 outputShapes)) {
      LLOGW("NSS: cannot reshape the graph for %ux%u\n", kGraphWidth, kGraphHeight);
      return false;
    }
    const int64_t dimsInput[] = {1, kGraphHeight, kGraphWidth, kInputChannels};
    const int64_t dimsKpn[] = {1, kKpnHeight, kKpnWidth, kKpnChannels};
    const int64_t dimsFeedback[] = {1, kGraphHeight, kGraphWidth, kFeedbackChannels};
    const glm::ivec4 shapeKpn(1, kKpnHeight, kKpnWidth, kKpnChannels);
    const glm::ivec4 shapeFeedback(1, kGraphHeight, kGraphWidth, kFeedbackChannels);
    const uint32_t indexKpn = outputShapes[0] == shapeKpn ? 0 : 1;
    const uint32_t indexFeedback = 1 - indexKpn;
    if (outputShapes[indexKpn] != shapeKpn || outputShapes[indexFeedback] != shapeFeedback) {
      LLOGW("NSS: unexpected graph interface (this sample expects an int8 NSS v1 model that upscales two times)\n");
      return false;
    }

    lvk::Result result;
    auto graphTensorDesc = [](const int64_t (&dims)[4]) -> lvk::TensorDesc {
      return {
          .format = lvk::Format_R_I8,
          .rank = 4,
          .dimensions = {dims[0], dims[1], dims[2], dims[3]},
          .usage = lvk::TensorUsageBits_DataGraph,
      };
    };
    tensorInput = createTensorWithBuffer(ctx, graphTensorDesc(dimsInput), bufTensorInput, "NSS: input tensor", &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the input tensor: %s\n", result.message);
      return false;
    }
    tensorKpn = createTensorWithBuffer(ctx, graphTensorDesc(dimsKpn), bufTensorKpn, "NSS: KPN coefficients", &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the KPN tensor: %s\n", result.message);
      return false;
    }
    feedbackAliasing = ctx->getTensorFormatSupport(lvk::Format_R_I8, lvk::TensorTiling_Optimal) & lvk::TensorUsageBits_ImageAliasing;
    if (feedbackAliasing) {
      texFeedback = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_TensorAliasing,
          .debugName = "NSS: temporal feedback (texture)",
      });
      tensorFeedback = ctx->createTensorTextureAliasing(texFeedback, graphTensorDesc(dimsFeedback), "NSS: temporal feedback", &result);
      feedbackAliasing = result.isOk();
      if (!feedbackAliasing) {
        LLOGW("NSS: cannot alias the feedback texture: %s\n", result.message);
        texFeedback = nullptr;
      }
    }
    if (!feedbackAliasing) {
      tensorFeedback = createTensorWithBuffer(ctx, graphTensorDesc(dimsFeedback), bufTensorFeedback, "NSS: temporal feedback", &result);
      if (!result.isOk()) {
        LLOGW("NSS: cannot create the feedback tensor: %s\n", result.message);
        return false;
      }
      texFeedback = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = "NSS: temporal feedback (texture)",
      });
    }
    LLOGL("NSS: temporal feedback via %s\n", feedbackAliasing ? "a texture aliased by the tensor" : "a buffer copied into a texture");

    for (uint32_t i = 0; i != 2; i++) {
      texLumaDeriv[i] = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = i ? "NSS: luma derivative 1" : "NSS: luma derivative 0",
      });
      texOutput[i] = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_F16,
          .dimensions = {kOutputWidth, kOutputHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = i ? "NSS: upscaled output 1" : "NSS: upscaled output 0",
      });
    }
    texNearestDepthCoord = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_R_UN8,
        .dimensions = {kGraphWidth, kGraphHeight},
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
        .debugName = "NSS: nearest depth coord",
    });
    texDepthTm1 = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_R_UI32,
        .dimensions = {kDepthTm1Width, kDepthTm1Height},
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
        .debugName = "NSS: reconstructed previous depth",
    });
    samplerPoint = ctx->createSampler({
        .minFilter = lvk::SamplerFilter_Nearest,
        .magFilter = lvk::SamplerFilter_Nearest,
        .wrapU = lvk::SamplerWrap_Clamp,
        .wrapV = lvk::SamplerWrap_Clamp,
        .debugName = "NSS: point clamp",
    });
    bufConstants = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(NssConstants),
        .debugName = "NSS: constants",
    });

    struct ComputePass {
      const char* code;
      lvk::Holder<lvk::ShaderModuleHandle>* sm;
      lvk::Holder<lvk::ComputePipelineHandle>* pipeline;
      const char* debugName;
    };
    const ComputePass passes[] = {
        {kNssDepthScatter, &smDepthScatter, &pipelineDepthScatter, "NSS: depth scatter"},
        {kNssPreprocess, &smPreprocess, &pipelinePreprocess, "NSS: preprocess"},
        {kNssPostprocess, &smPostprocess, &pipelinePostprocess, "NSS: postprocess"},
        {kNssFeedbackToTexture, &smFeedbackToTexture, &pipelineFeedbackToTexture, "NSS: feedback to texture"},
    };
    for (const ComputePass& pass : passes) {
      const double start = glfwGetTime();
      std::vector<std::string> included;
      bool ok = true;
      const std::string source = expandIncludes(pass.code, included, ok);
      if (!ok || source.empty()) {
        return false;
      }
      lvk::ShaderModuleDesc desc(source.c_str(), lvk::Stage_Comp, pass.debugName);
      desc.optimizeSPIRV = false;
      *pass.sm = ctx->createShaderModule(desc);
      if (pass.sm->empty()) {
        LLOGW("NSS: cannot compile `%s`\n", pass.debugName);
        return false;
      }
      *pass.pipeline = ctx->createComputePipeline({.smComp = *pass.sm, .debugName = pass.debugName});
      LLOGL("NSS: compiled `%s` in %.1f ms\n", pass.debugName, 1000.0 * (glfwGetTime() - start));
    }

    graphInputs[0] = tensorInput;
    graphOutputs[indexKpn] = tensorKpn;
    graphOutputs[indexFeedback] = tensorFeedback;
    const double start = glfwGetTime();
    const lvk::Holder<lvk::ShaderModuleHandle> smGraph =
        ctx->createShaderModule({spirv.data(), spirv.size() * sizeof(uint32_t), lvk::Stage_DataGraph, "NSS: data graph"});
    // the pipeline is described by the tensors it gets bound to, which may be optimally tiled copies of the buffer-aliasing ones
    const lvk::TensorDesc boundInputs[] = {ctx->getTensorDesc(graphInputs[0])};
    const lvk::TensorDesc boundOutputs[] = {ctx->getTensorDesc(graphOutputs[0]), ctx->getTensorDesc(graphOutputs[1])};
    pipelineGraph = ctx->createDataGraphPipeline(
        {
            .smGraph = smGraph,
            .entryPoint = vgf.getEntryPoint(),
            .inputs = {boundInputs, 1},
            .outputs = {boundOutputs, 2},
            .constants = vgf.getConstants(),
            .debugName = "NSS: data graph",
        },
        &result);
    if (!result.isOk()) {
      LLOGW("NSS: cannot create the data graph pipeline: %s\n", result.message);
      return false;
    }
    LLOGL("NSS: data graph pipeline created in %.1f ms\n", 1000.0 * (glfwGetTime() - start));

    supported = true;
    resetHistory = true;
    return true;
  }

  lvk::TextureHandle history() const {
    return texOutput[(frameIndex + 1) & 1];
  }
  lvk::TextureHandle output() const {
    return texOutput[frameIndex & 1];
  }

  NssConstants makeConstants(bool reset) const {
    const float fMin = std::min(kCameraNear, kCameraFar);
    const float fMax = std::max(kCameraNear, kCameraFar);
    const float fQ = fMax / (fMin - fMax);
    const float aspect = (float)kRenderWidth / (float)kRenderHeight;
    const float cotHalfFovY = cosf(0.5f * kCameraFovY) / sinf(0.5f * kCameraFovY);
    const float tanHalfFovX = 1.0f / (cotHalfFovY / aspect);
    const float tanHalfFovY = 1.0f / cotHalfFovY;

    const float halfViewport = sqrtf(float(kRenderWidth * kRenderWidth + kRenderHeight * kRenderHeight));
    const float cornerViewDirLen = sqrtf(tanHalfFovX * tanHalfFovX + tanHalfFovY * tanHalfFovY + 1.0f);
    const float kReferenceViewportLength = 2202.9071700822983f;
    const float resolutionFactor = std::min(std::max(halfViewport / kReferenceViewportLength, 0.0f), 1.0f);

    const vec2 scale = vec2((float)kOutputWidth / kRenderWidth, (float)kOutputHeight / kRenderHeight);
    const int32_t indexModulo[2] = {(int32_t)(kOutputWidth / kRenderWidth), (int32_t)(kOutputHeight / kRenderHeight)};
    // the output pixel the jitter lands in, relative to the unjittered one, wrapped into [0, indexModulo)
    auto lutOffset = [](float jitter, float scale, int32_t modulo) -> int32_t {
      const int32_t base = (int32_t)floorf(0.5f * scale);
      const int32_t jittered = (int32_t)floorf((jitter + 0.5f) * scale);
      const int32_t d = (jittered - base) % modulo;
      return d < 0 ? d + modulo : d;
    };

    const vec2 jitterTm1 = reset ? jitter : jitterPrev;
    return {
        .deviceToViewDepth = {-1.0f * fQ, fQ * fMin, tanHalfFovX, tanHalfFovY},
        .jitterOffset = {jitter.x, jitter.y, jitter.x / kRenderWidth, jitter.y / kRenderHeight},
        .jitterOffsetTm1 = {jitterTm1.x, jitterTm1.y, jitterTm1.x / kRenderWidth, jitterTm1.y / kRenderHeight},
        .scaleFactor = {scale.x, scale.y, (float)kRenderWidth / kOutputWidth, (float)kRenderHeight / kOutputHeight},
        .outputDims = {(int32_t)kOutputWidth, (int32_t)kOutputHeight},
        .inputDims = {(int32_t)kRenderWidth, (int32_t)kRenderHeight},
        .invOutputDims = {1.0f / kOutputWidth, 1.0f / kOutputHeight},
        .invInputDims = {1.0f / kRenderWidth, 1.0f / kRenderHeight},
        .depthTm1Size = {(int32_t)kDepthTm1Width, (int32_t)kDepthTm1Height},
        .invDepthTm1Size = {1.0f / kDepthTm1Width, 1.0f / kDepthTm1Height},
        .inputTensorSize = {(int32_t)kGraphWidth, (int32_t)kGraphHeight},
        .inputTensorSizeRcp = {1.0f / kGraphWidth, 1.0f / kGraphHeight},
        .kpnDimension = {(int32_t)kKpnWidth, (int32_t)kKpnHeight},
        .motionVectorScale = {(float)kRenderWidth, (float)kRenderHeight},
        .paddingScale = {(float)kRenderWidth / kGraphWidth, (float)kRenderHeight / kGraphHeight},
        .depthClipRequiredSepScale = 1.37e-05f * cornerViewDirLen * halfViewport,
        .depthClipPower = 1.0f + 2.0f * resolutionFactor,
        .kpnScale = {(float)kKpnWidth / kGraphWidth, (float)kKpnHeight / kGraphHeight},
        .debugViewMode = 0,
        .notHistoryReset = reset ? 0.0f : 1.0f,
        .exposure = {exposure, 1.0f / exposure},
        .indexModulo = {indexModulo[0], indexModulo[1]},
        .reducedInputModulo = {1, 1},
        .lutOffset = {lutOffset(jitter.x, scale.x, indexModulo[0]), lutOffset(jitter.y, scale.y, indexModulo[1])},
    };
  }

  void dispatch(lvk::IContext* ctx, lvk::ICommandBuffer& buffer) {
    const bool reset = resetHistory;
    resetHistory = false;

    const NssConstants constants = makeConstants(reset);
    buffer.cmdUpdateBuffer(bufConstants, 0, sizeof(constants), &constants);

    const lvk::TextureHandle lumaDerivTm1 = texLumaDeriv[(frameIndex + 1) & 1];
    const lvk::TextureHandle lumaDeriv = texLumaDeriv[frameIndex & 1];
    const lvk::TextureHandle texHistory = history();
    const lvk::TextureHandle texOut = output();

    if (reset) {
      buffer.cmdPushDebugGroupLabel("NSS: reset history", 0xff0000ff);
      const lvk::TextureHandle clearTargets[] = {
          texLumaDeriv[0], texLumaDeriv[1], texNearestDepthCoord, texOutput[0], texOutput[1], texFeedback};
      for (const lvk::TextureHandle tex : clearTargets) {
        buffer.cmdClearColorImage(tex, {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}});
      }
      buffer.cmdPopDebugGroupLabel();
    }

    const NssPushConstants pc = {
        .constants = ctx->gpuAddress(bufConstants),
        .samplerPoint = samplerPoint.index(),
        .samplerLinear = res_.samplerLinearClamp.index(),
        .texColor = res_.texColorLR.index(),
        .texDepth = res_.texDepthLR.index(),
        .texMotion = res_.texMotionLR.index(),
        .texHistory = texHistory.index(),
        .texFeedback = texFeedback.index(),
        .texLumaDerivTm1 = lumaDerivTm1.index(),
        .texDepthTm1 = texDepthTm1.index(),
        .texNearestDepthCoord = texNearestDepthCoord.index(),
        .imgDepthTm1 = texDepthTm1.index(),
        .imgLumaDeriv = lumaDeriv.index(),
        .imgNearestDepthCoord = texNearestDepthCoord.index(),
        .imgOutput = texOut.index(),
        .tensorInput = ctx->gpuAddress(bufTensorInput),
        .tensorKpn = ctx->gpuAddress(bufTensorKpn),
    };

    auto groups = [](uint32_t size) -> uint32_t { return (size + kThreadGroup - 1) / kThreadGroup; };

    buffer.cmdPushDebugGroupLabel("NSS: depth scatter", 0xff0000ff);
    buffer.cmdClearColorImage(texDepthTm1, {.uint32 = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff}});
    buffer.cmdBindComputePipeline(pipelineDepthScatter);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kDepthTm1Width), groups(kDepthTm1Height), 1},
                       {
                           .sampledImages = {res_.texDepthLR, res_.texMotionLR},
                           .storageImages = {texDepthTm1},
                           .buffers = {bufConstants},
                       });
    buffer.cmdPopDebugGroupLabel();

    buffer.cmdPushDebugGroupLabel("NSS: preprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePreprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch(
        {groups(kGraphWidth), groups(kGraphHeight), 1},
        {
            .sampledImages = {res_.texColorLR, res_.texDepthLR, res_.texMotionLR, texHistory, texFeedback, lumaDerivTm1, texDepthTm1},
            .storageImages = {lumaDeriv, texNearestDepthCoord},
            .buffers = {bufConstants},
            .tensors = {tensorInput},
        });
    buffer.cmdPopDebugGroupLabel();

    buffer.cmdPushDebugGroupLabel("NSS: data graph", 0xff0000ff);
    buffer.cmdDispatchDataGraph(pipelineGraph, graphInputs, graphOutputs);
    buffer.cmdPopDebugGroupLabel();

    if (!feedbackAliasing) {
      buffer.cmdPushDebugGroupLabel("NSS: feedback to texture", 0xff0000ff);
      buffer.cmdBindComputePipeline(pipelineFeedbackToTexture);
      const struct {
        uint64_t tensor;
        uint32_t image;
        uint32_t width;
        uint32_t height;
      } pcFeedback = {
          .tensor = ctx->gpuAddress(bufTensorFeedback),
          .image = texFeedback.index(),
          .width = kGraphWidth,
          .height = kGraphHeight,
      };
      buffer.cmdPushConstants(pcFeedback);
      buffer.cmdDispatch({groups(kGraphWidth), groups(kGraphHeight), 1},
                         {
                             .storageImages = {texFeedback},
                             .tensors = {tensorFeedback},
                         });
      buffer.cmdPopDebugGroupLabel();
    }

    buffer.cmdPushDebugGroupLabel("NSS: postprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePostprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kOutputWidth), groups(kOutputHeight), 1},
                       {
                           .sampledImages = {res_.texColorLR, res_.texMotionLR, texHistory, texFeedback, texNearestDepthCoord},
                           .storageImages = {texOut},
                           .buffers = {bufConstants},
                           .tensors = {tensorKpn, tensorFeedback},
                       });
    buffer.cmdPopDebugGroupLabel();
  }
};

NeuralSuperSampling nss_;

enum GPUTimestamp {
  GPUTimestamp_Begin = 0,
  GPUTimestamp_Scene,
  GPUTimestamp_NSS,
  GPUTimestamp_Present,
  GPUTimestamp_NUM,
};
GPUStageTimer gpuTimer_;

// N toggles NSS, R resets its history, F toggles the full-resolution scene
void onKey(bool toggleNSS, bool resetHistory, bool toggleFullResolution) {
  if (ImGui::GetIO().WantCaptureKeyboard) {
    return;
  }
  if (toggleNSS && nss_.supported) {
    renderMode_ = renderMode_ == RenderMode_HalfResolutionNSS ? RenderMode_HalfResolution : RenderMode_HalfResolutionNSS;
  }
  if (toggleFullResolution) {
    renderMode_ = renderMode_ == RenderMode_FullResolution ? (nss_.supported ? RenderMode_HalfResolutionNSS : RenderMode_HalfResolution)
                                                           : RenderMode_FullResolution;
  }
  nss_.resetHistory |= toggleNSS || resetHistory || toggleFullResolution;
}

// half of the window, upscaled 2x; an odd window size leaves one row or column of the output out of the window
void selectResolution(uint32_t windowWidth, uint32_t windowHeight) {
  kRenderWidth = (windowWidth + 1) / 2;
  kRenderHeight = (windowHeight + 1) / 2;
  kOutputWidth = kRenderWidth * 2;
  kOutputHeight = kRenderHeight * 2;
  sceneWidth_ = kRenderWidth;
  sceneHeight_ = kRenderHeight;
  NeuralSuperSampling::configure();
}

VULKAN_APP_MAIN {
  // the size of the monitor unless `--width` and `--height` (or the shorthands) say otherwise
  uint32_t windowHeight = 0;
  for (int i = 1; i < argc; i++) {
    windowHeight = !strcmp(argv[i], "--1080p") ? 1080 : !strcmp(argv[i], "--4k") ? 2160 : windowHeight;
  }

  const VulkanAppConfig cfg{
      .width = (int)(windowHeight * 16 / 9),
      .height = (int)windowHeight,
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

  selectResolution((uint32_t)app.width_, (uint32_t)app.height_);

  bool cameraRotate = false;
  bool measureTime = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--no-nss")) {
      renderMode_ = RenderMode_HalfResolution;
    } else if (!strcmp(argv[i], "--ao-samples") && i + 1 < argc) {
      aoSamples_ = std::clamp(atoi(argv[++i]), 0, 16);
    } else if (!strcmp(argv[i], "--camera-rotate")) {
      cameraRotate = true;
    } else if (!strcmp(argv[i], "--timings")) {
      measureTime = true;
    } else if (!strcmp(argv[i], "--full-res")) {
      renderMode_ = RenderMode_FullResolution;
      sceneWidth_ = kOutputWidth;
      sceneHeight_ = kOutputHeight;
    }
  }
  uint32_t frameCounter = 0;

  ctx_ = app.ctx_.get();
  app_ = &app;
  const std::filesystem::path sdk =
      std::filesystem::path(app.folderThirdParty_) / "neural-graphics-sdk" / "sdk" / "include" / "FidelityFX" / "gpu";
  folderShadersNSS = {
      (sdk / "").string(),
      (sdk / "nss" / "").string(),
  };

  if (kEnableTextureCompression) {
    LLOGL("Compressing textures... It can take a while in debug builds...(needs to be done once)\n");
  }

  if (!initScene(app)) {
#if defined(ANDROID)
    return;
#else
    return EXIT_FAILURE;
#endif // ANDROID
  }

  nss_.init(ctx_);
  if (!nss_.supported && renderMode_ == RenderMode_HalfResolutionNSS) {
    renderMode_ = RenderMode_HalfResolution;
  }

  gpuTimer_.create(ctx_, GPUTimestamp_NUM);

#if !defined(ANDROID)
#if LVK_WITH_GLFW
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    if (action == GLFW_PRESS) {
      onKey(key == GLFW_KEY_N, key == GLFW_KEY_R, key == GLFW_KEY_F);
    }
  });
#elif LVK_WITH_SDL3
  app.addKeyCallback([](SDL_Window* window, SDL_KeyboardEvent* event) {
    if (event->down && !event->repeat) {
      onKey(event->key == SDLK_N, event->key == SDLK_R, event->key == SDLK_F);
    }
  });
#endif
#endif // !ANDROID

  mat4 viewProjPrev = mat4(1.0f);
  mat4 skyboxViewProjPrev = mat4(1.0f);

  app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const RenderView& view = views[0];

    if (cameraRotate) {
      const float angle = float(frameCounter) * 0.25f * float(M_PI / 180.0);
      const vec3 dir0 = glm::normalize(cfg.initialCameraTarget - cfg.initialCameraPos);
      const vec3 dir = glm::rotate(glm::mat4(1.0f), angle, vec3(0.0f, 1.0f, 0.0f)) * vec4(dir0, 0.0f);
      app.positioner_.lookAt(cfg.initialCameraPos, cfg.initialCameraPos + dir, vec3(0.0f, 1.0f, 0.0f));
    }

    const bool isFullResScene = renderMode_ == RenderMode_FullResolution;
    const uint32_t sceneWidth = isFullResScene ? kOutputWidth : kRenderWidth;
    const uint32_t sceneHeight = isFullResScene ? kOutputHeight : kRenderHeight;
    if (sceneWidth != sceneWidth_ || sceneHeight != sceneHeight_) {
      sceneWidth_ = sceneWidth;
      sceneHeight_ = sceneHeight;
      createOffscreenFramebuffer();
      nss_.resetHistory = true;
    }

    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    processLoadedMaterialTextures(buffer, res_.sbMaterials);

    const bool useNSS = nss_.supported && renderMode_ == RenderMode_HalfResolutionNSS;
    nss_.jitterPrev = nss_.jitter;
    nss_.jitter = useNSS ? NeuralSuperSampling::jitterOffset(nss_.frameIndex) : vec2(0.0f);
    const float aspectRatio = (float)sceneWidth_ / (float)sceneHeight_;
    const mat4 projNoJitter = glm::perspectiveRH_ZO(kCameraFovY, aspectRatio, kCameraNear, kCameraFar);
    const mat4 proj =
        glm::translate(mat4(1.0f), vec3(-2.0f * nss_.jitter.x / sceneWidth_, 2.0f * nss_.jitter.y / sceneHeight_, 0.0f)) * projNoJitter;
    const mat4 viewMatrix = app.camera_.getViewMatrix();
    const mat4 skyboxView = mat4(viewMatrix[0], viewMatrix[1], viewMatrix[2], vec4(0, 0, 0, 1));
    if (!frameCounter) {
      viewProjPrev = projNoJitter * viewMatrix;
      skyboxViewProjPrev = projNoJitter * skyboxView;
    }

    const mat4 shadowProj = glm::perspective(float(60.0f * (M_PI / 180.0f)), 1.0f, 10.0f, 4000.0f);
    const mat4 shadowView = mat4(vec4(0.772608519f, 0.532385886f, -0.345892131f, 0),
                                 vec4(0, 0.544812560f, 0.838557839f, 0),
                                 vec4(0.634882748f, -0.647876859f, 0.420926809f, 0),
                                 vec4(-58.9244843f, -30.4530792f, -508.410126f, 1.0f));
    const mat4 scaleBias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

    const UniformsPerFrame perFrame = {
        .proj = proj,
        .projNoJitter = projNoJitter,
        .view = viewMatrix,
        .viewProjPrev = viewProjPrev,
        .skyboxViewProjPrev = skyboxViewProjPrev,
        .light = scaleBias * shadowProj * shadowView,
        .texSkyboxRadiance = res_.skyboxReference.index(),
        .texSkyboxIrradiance = res_.skyboxIrradiance.index(),
        .texShadow = res_.texShadowMap.index(),
        .sampler = res_.sampler.index(),
        .samplerShadow = res_.samplerShadow.index(),
    };

    const mat4 modelMatrix = glm::scale(mat4(1.0f), vec3(kSceneScale));

    const UniformsPerObject perObject = {
        .model = modelMatrix,
        .normal = glm::transpose(glm::inverse(modelMatrix)),
    };

    buffer.cmdUpdateBuffer(res_.ubPerFrame, 0, sizeof(perFrame), &perFrame);
    buffer.cmdUpdateBuffer(res_.ubPerObject, 0, sizeof(perObject), &perObject);

    if (isShadowMapDirty_) {
      const UniformsPerFrame perFrameShadow{
          .proj = shadowProj,
          .view = shadowView,
      };
      buffer.cmdUpdateBuffer(res_.ubPerFrameShadow, 0, sizeof(perFrameShadow), &perFrameShadow);
      buffer.cmdBeginRendering(renderPassShadow_, {.depthStencil = {.texture = res_.texShadowMap}});
      {
        buffer.cmdBindRenderPipeline(res_.pipelineShadow);
        buffer.cmdPushDebugGroupLabel("Render Shadows", 0xff0000ff);
        buffer.cmdBindDepthState(depthState_);
        buffer.cmdBindVertexBuffer(0, res_.vb0, 0);
        const struct {
          uint64_t perFrame;
          uint64_t perObject;
        } bindings = {
            .perFrame = ctx_->gpuAddress(res_.ubPerFrameShadow),
            .perObject = ctx_->gpuAddress(res_.ubPerObject),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdBindIndexBuffer(res_.ib0, lvk::IndexFormat_UI32);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
        buffer.cmdPopDebugGroupLabel();
      }
      buffer.cmdEndRendering();
      buffer.cmdTransitionToShaderReadOnly({res_.texShadowMap}, {});
      isShadowMapDirty_ = false;
    }

    gpuTimer_.beginFrame(buffer, frameCounter);

    {
      const lvk::Framebuffer fbOffscreen = {
          .color = {{.texture = res_.texColorLR}, {.texture = res_.texMotionLR}},
          .depthStencil = {.texture = res_.texDepthLR},
      };
      buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen);
      {
        buffer.cmdBindRenderPipeline(res_.pipelineMesh);
        buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
        buffer.cmdBindDepthState(depthState_);
        buffer.cmdBindVertexBuffer(0, res_.vb0, 0);

        const struct {
          uint64_t perFrame;
          uint64_t perObject;
          uint64_t materials;
          uint32_t tlas;
          uint32_t aoSamples;
          uint32_t frameId;
          float aoRadius;
          float aoPower;
        } bindings = {
            .perFrame = ctx_->gpuAddress(res_.ubPerFrame),
            .perObject = ctx_->gpuAddress(res_.ubPerObject),
            .materials = ctx_->gpuAddress(res_.sbMaterials),
            .tlas = res_.accel.TLAS.empty() ? 0u : res_.accel.TLAS.index(),
            .aoSamples = res_.accel.TLAS.empty() ? 0u : uint32_t(aoSamples_),
            .frameId = frameCounter + 1,
            .aoRadius = aoRadius_,
            .aoPower = aoPower_,
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdBindIndexBuffer(res_.ib0, lvk::IndexFormat_UI32);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
        buffer.cmdPopDebugGroupLabel();

        buffer.cmdBindRenderPipeline(res_.pipelineSkybox);
        buffer.cmdPushDebugGroupLabel("Render Skybox", 0x00ff00ff);
        buffer.cmdBindDepthState(depthStateLEqual_);
        buffer.cmdDraw(3 * 6 * 2);
        buffer.cmdPopDebugGroupLabel();
      }
      buffer.cmdEndRendering();
      gpuTimer_.timestamp(buffer, GPUTimestamp_Scene);
    }

    if (useNSS) {
      nss_.dispatch(ctx_, buffer);
    }
    gpuTimer_.timestamp(buffer, GPUTimestamp_NSS);

    {
      const lvk::TextureHandle tex = useNSS ? nss_.output() : lvk::TextureHandle(res_.texColorLR);
      const lvk::Framebuffer fbMain = {.color = {{.texture = view.colorTexture}}};

      buffer.cmdBeginRendering(renderPassMain_, fbMain, {.sampledImages = {tex}});
      {
        buffer.cmdBindRenderPipeline(res_.pipelineFullscreen);
        buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
        // the window in the orientation of the output: the swapchain of a quarter-turned surface is transposed
        const lvk::Dimensions dim = ctx_->getDimensions(view.colorTexture);
        const bool isQuarterTurn = lvk::isQuarterTurn(ctx_->getSwapchainSurfaceTransform());
        const vec2 window = isQuarterTurn ? vec2(dim.height, dim.width) : vec2(dim.width, dim.height);
        const struct {
          vec2 clipRotation;
          uint32_t tex;
          uint32_t smp;
          vec2 uvScale;
        } bindings = {
            .clipRotation = getClipRotation(ctx_),
            .tex = tex.index(),
            .smp = res_.samplerLinearClamp.index(),
            .uvScale = window / vec2(kOutputWidth, kOutputHeight),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdDraw(3);
        buffer.cmdPopDebugGroupLabel();

        app.imgui_->beginFrame(fbMain);
        drawMaterialTexturesLoadingProgress();
        ImGui::Begin("Neural Super Sampling", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
        ImGui::Text("W/S/A/D - camera movement, Shift - fast movement");
        ImGui::Text("N - toggle NSS, R - reset history, F - full-resolution scene");
        ImGui::Separator();
        char modes[RenderMode_NUM_MODES][64] = {};
        snprintf(modes[RenderMode_FullResolution], sizeof(modes[0]) - 1, "Full-resolution (%ux%u)", kOutputWidth, kOutputHeight);
        snprintf(modes[RenderMode_HalfResolution], sizeof(modes[0]) - 1, "Half-resolution (%ux%u)", kRenderWidth, kRenderHeight);
        snprintf(modes[RenderMode_HalfResolutionNSS],
                 sizeof(modes[0]) - 1,
                 "Half-resolution + NSS (%ux%u -> %ux%u)",
                 kRenderWidth,
                 kRenderHeight,
                 kOutputWidth,
                 kOutputHeight);
        const char* items[RenderMode_NUM_MODES] = {modes[0], modes[1], modes[2]};
        const int numItems = nss_.supported ? RenderMode_NUM_MODES : RenderMode_NUM_MODES - 1;
        // wide enough for the longest mode
        float itemsWidth = 0.0f;
        for (const char* item : items) {
          itemsWidth = std::max(itemsWidth, ImGui::CalcTextSize(item).x);
        }
        ImGui::SetNextItemWidth(itemsWidth + 2.0f * ImGui::GetStyle().FramePadding.x + ImGui::GetFrameHeight());
        if (ImGui::Combo("Rendering", &renderMode_, items, numItems)) {
          nss_.resetHistory = true;
        }
        ImGui::Separator();
        if (rayQuery_) {
          ImGui::SliderInt("Ray-traced AO samples per pixel", &aoSamples_, 0, 16);
          ImGui::SliderFloat("AO radius", &aoRadius_, 1.0f, 16.0f);
          ImGui::SliderFloat("AO power", &aoPower_, 0.5f, 2.0f);
        } else {
          ImGui::Text("VK_KHR_ray_query is not available: no ambient occlusion");
        }
        ImGui::Separator();
        if (nss_.supported) {
          if (ImGui::Button("Reset history")) {
            nss_.resetHistory = true;
          }
          ImGui::SliderFloat("Exposure", &nss_.exposure, 0.1f, 8.0f);
        } else {
          ImGui::Text("VK_ARM_tensors / VK_ARM_data_graph are not available: showing the %ux%u input", kRenderWidth, kRenderHeight);
        }
        ImGui::End();
        const GPUStageTimer::Row rows[] = {
            {"Frame", GPUTimestamp_Begin, ImVec4(1.0f, 1.0f, 1.0f, 1.0f)},
            {"Scene", GPUTimestamp_Scene, ImVec4(0.4f, 0.7f, 1.0f, 1.0f)},
            {"NSS", GPUTimestamp_NSS, ImVec4(1.0f, 0.5f, 0.2f, 1.0f)},
            {"Present", GPUTimestamp_Present, ImVec4(0.75f, 0.6f, 1.0f, 1.0f)},
        };
        gpuTimer_.drawWindow(rows);
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
      buffer.cmdEndRendering();
    }
    gpuTimer_.timestamp(buffer, GPUTimestamp_Present);

    ctx_->submit(buffer, view.colorTexture);

    gpuTimer_.endFrame(ctx_);

    if (measureTime && gpuTimer_.isMeasuring && frameCounter % 120 == 0) {
      LLOGL("GPU time, ms: frame %.2f | scene %.2f, NSS %.2f, present %.2f\n",
            gpuTimer_.frameTime,
            gpuTimer_.stageTime[GPUTimestamp_Scene],
            gpuTimer_.stageTime[GPUTimestamp_NSS],
            gpuTimer_.stageTime[GPUTimestamp_Present]);
    }

    frameCounter++;
    viewProjPrev = projNoJitter * viewMatrix;
    skyboxViewProjPrev = projNoJitter * skyboxView;
    if (useNSS) {
      nss_.frameIndex++;
    }
  });

  nss_ = {};
  destroyScene();
  gpuTimer_ = {};
  ctx_ = nullptr;

  VULKAN_APP_EXIT();
}
