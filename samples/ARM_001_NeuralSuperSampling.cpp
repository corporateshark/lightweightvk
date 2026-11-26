/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm Neural Super Sampling (NSS) on top of VK_ARM_tensors and VK_ARM_data_graph, or VK_KHR_cooperative_matrix, fully bindless.
 *
 * The window opens at the size of the monitor (`--width W --height H`, or the shorthands `--1080p` and `--4k`), and the
 * Bistro scene is rendered at half of it with subpixel jitter into color, depth and motion vector targets and upscaled 2x to
 * it by the NSS v1 pipeline, presented 1:1 (Arm ships the network shaped for 960x540, and the sample reshapes it at start-up):
 *
 *   depth scatter (compute) -> preprocess (compute, writes the int8 input tensor) -> data graph (VK_ARM_data_graph,
 *   the neural network from a VGF file) -> postprocess (compute, kernel prediction filter + temporal accumulation), plus a
 *   feedback-to-texture copy (compute) where the int8 feedback tensor cannot alias its texture
 *
 * The same network also runs as int8 cooperative matrices (VK_KHR_cooperative_matrix) with no VK_ARM_* extension, bit for bit
 * what the data graph computes: the sample turns the graph into compute dispatches, one per CONV2D with the RESCALE and TABLE
 * after it and the RESIZE or CONCAT before it folded in. Both are in the UI, each grayed out without the extensions it runs on;
 * `--nss-cm` starts with the cooperative matrices.
 *
 * The NSS algorithm comes from a fork of the Arm Neural Graphics SDK that binds NSS resources bindlessly, deployed by
 * `deploy_deps.py`; the compute entry points and the shader permutation are compiled into this file. The Arm ML Emulation
 * Layer for Vulkan is used when LightweightVK was configured with `LVK_WITH_ML_EMULATION_LAYER=ON` (run with
 * `--no-ml-emulation` to disable it). `--no-nss` starts with NSS disabled,
 * `--camera-rotate` rotates the camera continuously (temporal stability test; combine with `--screenshot-frame`),
 * `--ao-samples N` sets the number of ambient occlusion rays per pixel (4 by default, 1 on Android, 0 disables it), `--timings` also prints
 * the GPU time of every stage to the log, and every 600 frames that of every NSS pass and layer.
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
// vec4: a vec2 output next to ray queries corrupts out_FragColor.rg on Adreno
layout (location=1) out vec4 out_Motion;

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
  out_Motion = vec4(motionVector(clipCur, clipPrev), 0.0, 0.0);
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
#if defined(ANDROID)
int aoSamples_ = 1;
#else
int aoSamples_ = 4;
#endif // ANDROID
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

// a layer of the network on VK_KHR_cooperative_matrix: a CONV2D (int8 into int32) with the RESCALE after it and the TABLE after
// that, reading the RESIZE (nearest, 2x) or the CONCAT (of channels) before it straight from the activations they combine; the
// input zero point is folded into the biases, so the staged int8 input is the A matrices as it is, and the padding outside the
// input is the zero point itself. A subgroup computes tiles of kTileM pixels by kRows rows of the output from the input rows it
// staged itself, so no subgroup ever waits for another; bit for bit what the data graph computes
const char* kNssCooperativeExtensions = R"(
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
)";

const char* kNssCooperativeCS = R"(
layout (local_size_x = 64) in;

layout(std430, buffer_reference) readonly buffer Words {
  uint v[];
};
layout(std430, buffer_reference) readonly buffer Weights {
  int8_t v[];
};
layout(std430, buffer_reference) writeonly buffer Activations {
  int8_t v[];
};
layout(std430, buffer_reference) readonly buffer Params {
  int v[];
};

layout(push_constant) uniform constants {
  Words source0;
  Words source1;
  Activations destination;
  Weights weights; // per kernel row, B = kRowK x kPaddedOutputs int8, row-major
  Params params;   // the folded biases, the multipliers, the shifts (kPaddedOutputs each), then the table
} pc;

const uint kSpan = (kTileM - 1u) * kStride + kKernelWidth; // the input pixels of a staged row
const uint kStagedRows = (kRows - 1u) * kStride + kKernelHeight; // of the kRows output rows of a tile
// a staged pixel has its channels padded to 16 bytes, so the taps of a kernel row are one contiguous row of A, K = kRowK wide;
// the K padding reaches into the next pixels (and kTileK bytes past the staged rows), and meets the zero rows of B
const uint kStaged = kStagedRows * kSpan * kStagedChannels + kTileK;
const uint kRowStep = kStride * kSpan * kStagedChannels; // from the staged rows of an output row to those of the next one
const uint kMaxSubgroups = 64u / kMinSubgroupSize;

shared int8_t sA[kMaxSubgroups * kStaged];
shared int sC[kMaxSubgroups * kTileM * kTileN];

// TOSA RESCALE with scale32 and single rounding, (value * multiplier + 2^(shift - 1)) >> shift in 64 bits, saturated to int8
int rescale(int value, int multiplier, int shift, int zeroPoint) {
  int hi;
  int lo;
  imulExtended(value, multiplier, hi, lo);
  uint carry;
  const uint low = uaddCarry(uint(lo), shift <= 32 ? 1u << (shift - 1) : 0u, carry);
  const int high = hi + int(carry) + (shift > 32 ? 1 << (shift - 33) : 0);
  int result;
  if (shift >= 32) {
    result = high >> (shift - 32);
  } else {
    const int top = high >> shift;
    const int bottom = int((low >> shift) | (uint(high) << (32 - shift)));
    result = top == (bottom >> 31) ? bottom : (high < 0 ? -1024 : 1024);
  }
  return clamp(clamp(result, -1024, 1024) + zeroPoint, -128, 127);
}

void storeOutput(uint pixel, uint channel, int acc, int multiplier, int shift) {
  const int q = rescale(acc, multiplier, shift, kOutputZeroPoint);
  // the no-op mask keeps Mali-G925 from faulting
  pc.destination.v[pixel * kOutputs + channel] = int8_t(kHasTable ? pc.params.v[kTableAt + (uint(q + 128) & 255u)] : q);
}

// the RESCALE (and the TABLE) of the accumulators of an output row, which C holds, into the output
void storeOutputs(uint firstPixel, uint ox0, uint n, uint cAt) {
  if (gl_SubgroupSize % kTileN == 0u) {
    // an invocation always gets the same output channel, so it loads its parameters once
    const uint col = gl_SubgroupInvocationID % kTileN;
    const uint o = n + col;
    const int bias = pc.params.v[o];
    const int multiplier = pc.params.v[kPaddedOutputs + o];
    const int shift = pc.params.v[2u * kPaddedOutputs + o];
    for (uint r = gl_SubgroupInvocationID / kTileN; r < kTileM; r += gl_SubgroupSize / kTileN) {
      if (o < kOutputs && ox0 + r < kOutputWidth) {
        storeOutput(firstPixel + r, o, sC[cAt + r * kTileN + col] + bias, multiplier, shift);
      }
    }
  } else {
    for (uint i = gl_SubgroupInvocationID; i < kTileM * kTileN; i += gl_SubgroupSize) {
      const uint r = i / kTileN;
      const uint o = n + i % kTileN;
      if (o < kOutputs && ox0 + r < kOutputWidth) {
        storeOutput(firstPixel + r, o, sC[cAt + i] + pc.params.v[o], pc.params.v[kPaddedOutputs + o], pc.params.v[2u * kPaddedOutputs + o]);
      }
    }
  }
}

void main() {
  const uint tilesPerRow = (kOutputWidth + kTileM - 1u) / kTileM;
  const uint stagedAt = gl_SubgroupID * kStaged;
  const uint cAt = gl_SubgroupID * kTileM * kTileN;
  for (uint t = gl_SubgroupID; t < kTilesPerGroup; t += gl_NumSubgroups) {
    const uint tile = gl_WorkGroupID.x * kTilesPerGroup + t;
    const uint oy0 = tile / tilesPerRow * kRows;
    if (oy0 >= kOutputHeight) {
      break;
    }
    const uint ox0 = tile % tilesPerRow * kTileM;

    // the input rows of the tile: the zero point outside the input, the channels of the sources (concatenated) inside it; the
    // padded channels meet the zero rows of B, so they are never written
    const int iy0 = int(oy0 * kStride) - kPadTop;
    const int ix0 = int(ox0 * kStride) - kPadLeft;
    for (uint w = gl_SubgroupInvocationID; w < kStagedRows * kSpan * (kChannels / 4u); w += gl_SubgroupSize) {
      const uint c4 = w % (kChannels / 4u);
      const uint pixel = w / (kChannels / 4u);
      const int iy = iy0 + int(pixel / kSpan);
      const int ix = ix0 + int(pixel % kSpan);
      uint word = kInputZeroPointWord;
      if (iy >= 0 && iy < int(kInputHeight) && ix >= 0 && ix < int(kInputWidth)) {
        const uint source = (uint(iy) >> kUpsample) * (kInputWidth >> kUpsample) + (uint(ix) >> kUpsample);
        word = c4 < kChannels0 / 4u ? pc.source0.v[source * (kChannels0 / 4u) + c4]
                                    : pc.source1.v[source * (kChannels1 / 4u) + c4 - kChannels0 / 4u];
      }
      const uint at = stagedAt + pixel * kStagedChannels + 4u * c4;
      [[unroll]] for (uint b = 0u; b != 4u; b++) {
        sA[at + b] = int8_t(bitfieldExtract(int(word), int(8u * b), 8));
      }
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();

    // kRows output rows at a time, so that a B tile is loaded once for kRows A tiles; the accumulators are 4 variables, since
    // some drivers are much slower with an array of cooperative matrices
    [[unroll]] for (uint n = 0u; n != kPaddedOutputs; n += kTileN) {
      coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator> c0 = coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator>(0);
      coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator> c1 = coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator>(0);
      coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator> c2 = coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator>(0);
      coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator> c3 = coopmat<int, gl_ScopeSubgroup, kTileM, kTileN, gl_MatrixUseAccumulator>(0);
      [[unroll]] for (uint ky = 0u; ky != kKernelHeight; ky++) {
        [[unroll]] for (uint k = 0u; k != kRowK; k += kTileK) {
          coopmat<int8_t, gl_ScopeSubgroup, kTileK, kTileN, gl_MatrixUseB> b;
          coopMatLoad(b, pc.weights.v, (ky * kRowK + k) * kPaddedOutputs + n, kPaddedOutputs, gl_CooperativeMatrixLayoutRowMajor);
          const uint aAt = stagedAt + ky * kSpan * kStagedChannels + k;
          coopmat<int8_t, gl_ScopeSubgroup, kTileM, kTileK, gl_MatrixUseA> a;
          coopMatLoad(a, sA, aAt, kStride * kStagedChannels, gl_CooperativeMatrixLayoutRowMajor);
          c0 = coopMatMulAdd(a, b, c0);
          if (kRows > 1u) {
            coopMatLoad(a, sA, aAt + kRowStep, kStride * kStagedChannels, gl_CooperativeMatrixLayoutRowMajor);
            c1 = coopMatMulAdd(a, b, c1);
          }
          if (kRows > 2u) {
            coopMatLoad(a, sA, aAt + 2u * kRowStep, kStride * kStagedChannels, gl_CooperativeMatrixLayoutRowMajor);
            c2 = coopMatMulAdd(a, b, c2);
            coopMatLoad(a, sA, aAt + 3u * kRowStep, kStride * kStagedChannels, gl_CooperativeMatrixLayoutRowMajor);
            c3 = coopMatMulAdd(a, b, c3);
          }
        }
      }
      // a tile at the bottom of the output computes rows past it too, which are left out
      [[unroll]] for (uint row = 0u; row != kRows; row++) {
        if (oy0 + row >= kOutputHeight) {
          break;
        }
        if (row == 0u) {
          coopMatStore(c0, sC, cAt, kTileN, gl_CooperativeMatrixLayoutRowMajor);
        } else if (row == 1u) {
          coopMatStore(c1, sC, cAt, kTileN, gl_CooperativeMatrixLayoutRowMajor);
        } else if (row == 2u) {
          coopMatStore(c2, sC, cAt, kTileN, gl_CooperativeMatrixLayoutRowMajor);
        } else {
          coopMatStore(c3, sC, cAt, kTileN, gl_CooperativeMatrixLayoutRowMajor);
        }
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        storeOutputs((oy0 + row) * kOutputWidth + ox0, ox0, n, cAt);
        subgroupMemoryBarrierShared();
        subgroupBarrier();
      }
    }
  }
}
)";

struct NssCooperativePushConstants {
  uint64_t source0;
  uint64_t source1;
  uint64_t destination;
  uint64_t weights;
  uint64_t params;
};

enum NssPath {
  NssPath_DataGraph = 0, // VK_ARM_tensors + VK_ARM_data_graph
  NssPath_Cooperative, // VK_KHR_cooperative_matrix
};

struct NeuralSuperSampling {
  enum {
    kAlignment = FFX_NSS_RESOURCE_ALIGNMENT,
    kThreadGroup = FFX_NSS_THREAD_GROUP_WIDTH,
    kKpnChannels = 36,
    kInputChannels = 12,
    kFeedbackChannels = 4,
    kJitterPhaseCount = 8 * 2 * 2,
    kCooperativeTilesPerGroup = 4, // of `kNssCooperativeCS`
    kCooperativeMaxMultiRowTile = 16 * 16, // larger tiles compute one output row
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

  // the network for the cooperative matrix path: a layer is a CONV2D with the RESCALE after it and the TABLE after that, reading
  // the RESIZE (nearest, 2x) or the CONCAT (of channels) before it straight from the activations they combine
  struct CooperativeLayer {
    uint32_t sources[2] = {}; // activations, 0 being the input of the graph
    uint32_t sourceChannels[2] = {};
    bool upsample = false;
    glm::ivec4 inputShape = glm::ivec4(0); // of the CONV2D, after the RESIZE
    glm::ivec4 outputShape = glm::ivec4(0);
    glm::ivec4 kernelShape = glm::ivec4(0); // OHWI
    uint32_t stride = 0;
    int32_t padTop = 0;
    int32_t padLeft = 0;
    int32_t inputZeroPoint = 0;
    int32_t outputZeroPoint = 0;
    std::vector<int32_t> weights;
    std::vector<int32_t> biases;
    std::vector<int32_t> multipliers;
    std::vector<int32_t> shifts;
    std::vector<int32_t> table; // empty without a TABLE
    uint32_t output = 0; // activation
  };

  // the layers of a (reshaped) graph and the shapes of its activations; false for anything else than what `kNssCooperativeCS`
  // computes exactly
  static bool buildCooperativeLayers(const std::vector<uint32_t>& words,
                                     ldr::Span<const lvk::DataGraphConstant> constants,
                                     std::vector<CooperativeLayer>& outLayers,
                                     std::vector<glm::ivec4>& outActivations,
                                     uint32_t (&outOutputs)[2]) {
    enum {
      SpvOpExtInst = 12,
      SpvOpTypeInt = 21,
      SpvOpConstantTrue = 41,
      SpvOpConstantFalse = 42,
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
      kAccumulatorInt32 = 1,
      kResizeNearest = 1,
      kRoundingSingle = 1,
      kTableSize = 256,
    };
    const uint32_t bound = words.size() > 5 ? words[3] : 0;
    std::vector<uint32_t> definitions(bound, 0);
    std::vector<uint32_t> operators; // the offsets of the TOSA operators, in order
    uint32_t input = 0;
    uint32_t outputs[2] = {};
    for (uint32_t offset = 5; offset < words.size();) {
      const uint32_t length = words[offset] >> 16;
      const uint32_t op = words[offset] & 0xFFFF;
      if (!length || offset + length > words.size()) {
        return false;
      }
      const bool isType = op == SpvOpTypeInt || op == SpvOpTypeTensorARM;
      const bool isValue = op == SpvOpConstant || op == SpvOpConstantComposite || op == SpvOpConstantNull || op == SpvOpConstantTrue ||
                           op == SpvOpConstantFalse || op == SpvOpConstantCompositeReplicateEXT || op == SpvOpGraphConstantARM ||
                           op == SpvOpGraphInputARM || op == SpvOpExtInst;
      const uint32_t id = isType ? words[offset + 1] : isValue ? words[offset + 2] : 0;
      if (id >= bound) {
        return false;
      }
      definitions[id] = id ? offset : definitions[id];
      input = op == SpvOpGraphInputARM ? id : input;
      if (op == SpvOpExtInst) {
        operators.push_back(offset);
      }
      if (op == SpvOpGraphSetOutputARM) {
        const uint32_t index = words[definitions[words[offset + 2]] + 3];
        if (index > 1) {
          return false;
        }
        outputs[index] = words[offset + 1];
      }
      offset += length;
    }

    auto opcode = [&](uint32_t id) -> uint32_t { return words[definitions[id]] & 0xFFFF; };
    auto typeOf = [&](uint32_t id) -> uint32_t { return words[definitions[id] + 1]; };
    // a scalar constant, sign-extended from the width of its type
    auto scalar = [&](uint32_t id) -> int32_t {
      if (opcode(id) != SpvOpConstant) {
        return opcode(id) == SpvOpConstantTrue ? 1 : 0;
      }
      const uint32_t width = words[definitions[typeOf(id)] + 2];
      const uint32_t v = words[definitions[id] + 3];
      return width >= 32 ? (int32_t)v : (int32_t)(v << (32 - width)) >> (32 - width);
    };
    // the shape of a tensor type, its unused trailing dimensions 1
    auto shapeOf = [&](uint32_t type) -> glm::ivec4 {
      const uint32_t rank = (uint32_t)scalar(words[definitions[type] + 3]);
      const uint32_t shape = words[definitions[type] + 4];
      glm::ivec4 result(1);
      for (uint32_t i = 0; i != std::min(rank, 4u); i++) {
        result[i] = opcode(shape) == SpvOpConstantComposite ? scalar(words[definitions[shape] + 3 + i]) : 1;
      }
      return result;
    };
    auto countOf = [&](uint32_t type) -> size_t {
      const glm::ivec4 shape = shapeOf(type);
      return size_t(shape.x) * shape.y * shape.z * shape.w;
    };
    // the elements of a constant (scalar, tensor or graph constant) as int32
    auto ints = [&](uint32_t id) -> std::vector<int32_t> {
      const uint32_t offset = definitions[id];
      const uint32_t op = opcode(id);
      if (op == SpvOpConstantComposite) {
        std::vector<int32_t> result;
        for (uint32_t i = 3; i != (words[offset] >> 16); i++) {
          result.push_back(scalar(words[offset + i]));
        }
        return result;
      }
      if (op == SpvOpConstantNull) {
        return std::vector<int32_t>(countOf(typeOf(id)), 0);
      }
      if (op == SpvOpConstantCompositeReplicateEXT) {
        return std::vector<int32_t>(countOf(typeOf(id)), scalar(words[offset + 3]));
      }
      if (op == SpvOpGraphConstantARM) {
        for (const lvk::DataGraphConstant& c : constants) {
          if (c.id != words[offset + 3]) {
            continue;
          }
          size_t count = 1;
          for (uint32_t i = 0; i != c.rank; i++) {
            count *= (size_t)c.dimensions[i];
          }
          std::vector<int32_t> result(count);
          for (size_t i = 0; i != count; i++) {
            result[i] = c.format == lvk::Format_R_I32   ? ((const int32_t*)c.data)[i]
                        : c.format == lvk::Format_R_I16 ? ((const int16_t*)c.data)[i]
                                                        : ((const int8_t*)c.data)[i];
          }
          return result;
        }
        return {};
      }
      return {scalar(id)};
    };

    std::vector<uint32_t> readers(bound, 0); // how many operators read a value
    for (const uint32_t offset : operators) {
      const uint32_t instruction = words[offset + 4];
      const uint32_t first = instruction == TOSA_CONV2D || instruction == TOSA_RESCALE ? 10
                             : instruction == TOSA_RESIZE                              ? 6
                             : instruction == TOSA_CONCAT                              ? 6
                                                                                       : 5;
      const uint32_t last = instruction == TOSA_CONCAT ? (words[offset] >> 16) : first + 1;
      for (uint32_t i = first; i != last; i++) {
        readers[words[offset + i]]++;
      }
    }

    std::vector<int32_t> activationOf(bound, -1);
    std::vector<int32_t> layerOf(bound, -1); // of a CONV2D result
    std::vector<int32_t> resized(bound, -1); // the activation a RESIZE result upsamples
    std::vector<glm::ivec2> concatenated(bound, glm::ivec2(-1)); // the activations a CONCAT result joins
    outLayers.clear();
    outActivations = {shapeOf(typeOf(input))};
    activationOf[input] = 0;
    for (const uint32_t offset : operators) {
      const uint32_t type = words[offset + 1];
      const uint32_t result = words[offset + 2];
      const uint32_t instruction = words[offset + 4];
      const uint32_t* operands = &words[offset + 5];
      const uint32_t numOperands = (words[offset] >> 16) - 5;
      if (instruction == TOSA_CONV2D) {
        const uint32_t source = operands[5];
        CooperativeLayer layer;
        const bool isActivation = activationOf[source] >= 0;
        const bool isResized = resized[source] >= 0;
        const bool isConcatenated = concatenated[source].x >= 0;
        if (!isActivation && !isResized && !isConcatenated) {
          return false;
        }
        layer.sources[0] = isActivation ? activationOf[source] : isResized ? resized[source] : concatenated[source].x;
        layer.sources[1] = isConcatenated ? concatenated[source].y : layer.sources[0];
        layer.sourceChannels[0] = outActivations[layer.sources[0]].w;
        layer.sourceChannels[1] = isConcatenated ? outActivations[layer.sources[1]].w : 0;
        layer.upsample = isResized;
        layer.inputShape = shapeOf(typeOf(source));
        layer.outputShape = shapeOf(type);
        layer.kernelShape = shapeOf(typeOf(operands[6]));
        const std::vector<int32_t> pad = ints(operands[0]);
        const std::vector<int32_t> stride = ints(operands[1]);
        const std::vector<int32_t> dilation = ints(operands[2]);
        layer.weights = ints(operands[6]);
        layer.biases = ints(operands[7]);
        layer.inputZeroPoint = ints(operands[8])[0];
        const uint32_t channels = layer.sourceChannels[0] + layer.sourceChannels[1];
        const bool isSupported = pad.size() == 4 && stride.size() == 2 && stride[0] == stride[1] && stride[0] > 0 && dilation.size() == 2 &&
                                 dilation[0] == 1 && dilation[1] == 1 && scalar(operands[3]) == kAccumulatorInt32 &&
                                 ints(operands[9])[0] == 0 && layer.kernelShape.w == (int32_t)channels &&
                                 layer.inputShape.w == (int32_t)channels && layer.sourceChannels[0] % 4 == 0 &&
                                 layer.sourceChannels[1] % 4 == 0 && layer.weights.size() == countOf(typeOf(operands[6])) &&
                                 layer.biases.size() == (size_t)layer.kernelShape.x;
        if (!isSupported) {
          return false;
        }
        layer.stride = (uint32_t)stride[0];
        layer.padTop = pad[0];
        layer.padLeft = pad[2];
        layerOf[result] = (int32_t)outLayers.size();
        outLayers.push_back(std::move(layer));
      } else if (instruction == TOSA_RESCALE) {
        const uint32_t source = operands[5];
        if (layerOf[source] < 0 || readers[source] != 1) {
          return false;
        }
        CooperativeLayer& layer = outLayers[layerOf[source]];
        layer.multipliers = ints(operands[6]);
        layer.shifts = ints(operands[7]);
        layer.outputZeroPoint = ints(operands[9])[0];
        const bool isSupported = scalar(operands[0]) == 1 && scalar(operands[1]) == kRoundingSingle && scalar(operands[2]) == 1 &&
                                 !scalar(operands[3]) && !scalar(operands[4]) && ints(operands[8])[0] == 0 &&
                                 layer.multipliers.size() == (size_t)layer.kernelShape.x &&
                                 layer.shifts.size() == (size_t)layer.kernelShape.x && shapeOf(type) == layer.outputShape;
        if (!isSupported) {
          return false;
        }
        layer.output = (uint32_t)outActivations.size();
        activationOf[result] = (int32_t)layer.output;
        outActivations.push_back(layer.outputShape);
      } else if (instruction == TOSA_TABLE) {
        const uint32_t source = operands[0];
        const int32_t activation = activationOf[source];
        CooperativeLayer* producer = nullptr;
        for (CooperativeLayer& layer : outLayers) {
          producer = activation > 0 && layer.output == (uint32_t)activation ? &layer : producer;
        }
        if (!producer || readers[source] != 1 || !producer->table.empty()) {
          return false;
        }
        producer->table = ints(operands[1]);
        if (producer->table.size() != kTableSize) {
          return false;
        }
        activationOf[result] = activation;
      } else if (instruction == TOSA_RESIZE) {
        // only a 2x nearest upsampling, which reads the input at (x >> 1, y >> 1)
        const int32_t activation = activationOf[operands[1]];
        const std::vector<int32_t> scale = ints(operands[2]); // y_n, y_d, x_n, x_d
        const std::vector<int32_t> offsets = ints(operands[3]);
        if (activation < 0 || scalar(operands[0]) != kResizeNearest || scale.size() != 4 || offsets.size() != 2) {
          return false;
        }
        const glm::ivec4 from = outActivations[activation];
        const glm::ivec4 to = shapeOf(type);
        for (int32_t axis = 0; axis != 2; axis++) {
          const int32_t n = scale[2 * axis];
          const int32_t d = scale[2 * axis + 1];
          const int32_t size = from[1 + axis];
          if (to[1 + axis] != 2 * size || n <= 0 || d <= 0) {
            return false;
          }
          for (int32_t o = 0; o != to[1 + axis]; o++) {
            const int32_t v = o * d + offsets[axis];
            const int32_t floor = v >= 0 ? v / n : -((-v + n - 1) / n);
            const int32_t i = std::clamp(floor + (2 * (v - floor * n) >= n ? 1 : 0), 0, size - 1);
            if (i != o / 2) {
              return false;
            }
          }
        }
        resized[result] = activation;
      } else if (instruction == TOSA_CONCAT) {
        const bool isSupported = numOperands == 3 && ints(operands[0])[0] == 3 && activationOf[operands[1]] >= 0 &&
                                 activationOf[operands[2]] >= 0;
        if (!isSupported) {
          return false;
        }
        concatenated[result] = glm::ivec2(activationOf[operands[1]], activationOf[operands[2]]);
      } else {
        return false;
      }
    }
    for (uint32_t i = 0; i != 2; i++) {
      if (!outputs[i] || activationOf[outputs[i]] <= 0) {
        return false;
      }
      outOutputs[i] = (uint32_t)activationOf[outputs[i]];
    }
    for (const CooperativeLayer& layer : outLayers) {
      if (layer.multipliers.empty()) {
        return false;
      }
    }
    return true;
  }

  // the staged input rows and the C tiles of a workgroup of `kNssCooperativeCS`
  static uint32_t getCooperativeSharedMemory(const CooperativeLayer& layer, const CooperativeMatrixShape& tile, uint32_t rows) {
    const uint32_t span = (tile.m - 1) * layer.stride + uint32_t(layer.kernelShape.z);
    const uint32_t stagedRows = (rows - 1) * layer.stride + uint32_t(layer.kernelShape.y);
    const uint32_t stagedChannels = (uint32_t(layer.kernelShape.w) + 15) / 16 * 16;
    return 64 / tile.minSubgroupSize * (stagedRows * span * stagedChannels + tile.k + uint32_t(sizeof(int32_t)) * tile.m * tile.n);
  }

  // output rows per tile: 4 to load a B tile once for 4 A tiles, fewer if shared memory is short, 0 if even 1 row does not fit;
  // 1 for tiles larger than 16x16 (Adreno's 64x16 is slow and wrong with several accumulators)
  static uint32_t getCooperativeRows(const CooperativeLayer& layer, const CooperativeMatrixShape& tile) {
    for (uint32_t rows = 4; rows; rows /= 2) {
      const bool isFewEnough = tile.m * tile.n <= kCooperativeMaxMultiRowTile;
      const bool isSharedMemoryEnough = getCooperativeSharedMemory(layer, tile, rows) <= tile.maxSharedMemorySize;
      if ((isFewEnough || rows == 1) && isSharedMemoryEnough) {
        return rows;
      }
    }
    return 0;
  }

  // the B matrices (per kernel row, its taps' channels padded to 16, padded to K, by the outputs padded to N, zeros in the padding)
  // and the parameters (the biases with the input zero point folded in, the multipliers, the shifts, then the table) of a layer,
  // appended to `weights` and `params`; returns the GLSL constants of its `kNssCooperativeCS`
  static std::string buildCooperativeLayer(const CooperativeLayer& layer,
                                           const CooperativeMatrixShape& tile,
                                           std::vector<int8_t>& weights,
                                           std::vector<int32_t>& params) {
    auto roundUp = [](uint32_t value, uint32_t multiple) { return (value + multiple - 1) / multiple * multiple; };
    const uint32_t outputs = (uint32_t)layer.kernelShape.x;
    const uint32_t kernelHeight = (uint32_t)layer.kernelShape.y;
    const uint32_t kernelWidth = (uint32_t)layer.kernelShape.z;
    const uint32_t taps = kernelHeight * kernelWidth;
    const uint32_t channels = (uint32_t)layer.kernelShape.w;
    const uint32_t stagedChannels = roundUp(channels, 16);
    const uint32_t rowK = roundUp(kernelWidth * stagedChannels, tile.k);
    const uint32_t paddedOutputs = roundUp(outputs, tile.n);
    const size_t weightsAt = weights.size();
    weights.resize(weightsAt + size_t(kernelHeight) * rowK * paddedOutputs, 0);
    const size_t paramsAt = params.size();
    params.resize(paramsAt + 3 * size_t(paddedOutputs), 0);
    for (uint32_t o = 0; o != outputs; o++) {
      int32_t sum = 0;
      for (uint32_t tap = 0; tap != taps; tap++) {
        for (uint32_t c = 0; c != channels; c++) {
          const int32_t w = layer.weights[(size_t(o) * taps + tap) * channels + c];
          const size_t row = size_t(tap / kernelWidth) * rowK + tap % kernelWidth * stagedChannels + c;
          weights[weightsAt + row * paddedOutputs + o] = (int8_t)w;
          sum += w;
        }
      }
      params[paramsAt + o] = layer.biases[o] - layer.inputZeroPoint * sum;
      params[paramsAt + paddedOutputs + o] = layer.multipliers[o];
      params[paramsAt + 2 * paddedOutputs + o] = layer.shifts[o];
    }
    params.insert(params.end(), layer.table.begin(), layer.table.end());
    const uint32_t zeroPoint = uint32_t(layer.inputZeroPoint) & 0xFF;
    struct Constant {
      const char* name;
      int64_t value;
    };
    const Constant values[] = {
        {"kTilesPerGroup", kCooperativeTilesPerGroup},
        {"kTileM", tile.m},
        {"kTileN", tile.n},
        {"kTileK", tile.k},
        {"kRows", getCooperativeRows(layer, tile)},
        {"kMinSubgroupSize", tile.minSubgroupSize},
        {"kKernelHeight", layer.kernelShape.y},
        {"kKernelWidth", layer.kernelShape.z},
        {"kStride", layer.stride},
        {"kInputWidth", layer.inputShape.z},
        {"kInputHeight", layer.inputShape.y},
        {"kUpsample", layer.upsample ? 1 : 0},
        {"kChannels", channels},
        {"kChannels0", layer.sourceChannels[0]},
        {"kChannels1", layer.sourceChannels[1]},
        {"kStagedChannels", stagedChannels},
        {"kRowK", rowK},
        {"kOutputWidth", layer.outputShape.z},
        {"kOutputHeight", layer.outputShape.y},
        {"kOutputs", outputs},
        {"kPaddedOutputs", paddedOutputs},
        {"kInputZeroPointWord", zeroPoint * 0x01010101u},
        {"kTableAt", 3 * paddedOutputs},
    };
    std::string code;
    for (const Constant& v : values) {
      code += std::string("const uint ") + v.name + " = " + std::to_string(v.value) + "u;\n";
    }
    code += "const int kPadTop = " + std::to_string(layer.padTop) + ";\nconst int kPadLeft = " + std::to_string(layer.padLeft) +
            ";\nconst int kOutputZeroPoint = " + std::to_string(layer.outputZeroPoint) +
            ";\nconst bool kHasTable = " + (layer.table.empty() ? "false" : "true") + ";\n";
    return code;
  }

  bool supported = false;
  bool graphReady = false;
  bool cooperativeReady = false;
  int path = NssPath_DataGraph;
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
  struct CooperativeDispatch {
    lvk::Holder<lvk::ShaderModuleHandle> sm;
    lvk::Holder<lvk::ComputePipelineHandle> pipeline;
    uint32_t sources[2] = {};
    uint32_t output = 0;
    uint32_t groups = 0;
    size_t weightsAt = 0;
    size_t paramsAt = 0;
    char name[96] = {};
  };
  std::vector<CooperativeDispatch> cooperativeDispatches;
  std::vector<lvk::Holder<lvk::BufferHandle>> cooperativeActivations;
  std::vector<lvk::BufferHandle> activationBuffers; // of every activation, the tensors' buffers included
  lvk::Holder<lvk::BufferHandle> cooperativeWeights;
  lvk::Holder<lvk::BufferHandle> cooperativeParams;
  // the GPU time of every pass (every layer of the cooperative matrices), with `--timings`
  enum {
    kPassSlots = 4,
    kMaxPassTimestamps = 24,
  };
  bool profilePasses = false;
  lvk::Holder<lvk::QueryPoolHandle> queryPoolPasses;
  uint32_t passSlotTimestamps[kPassSlots] = {}; // written into a slot and not read back yet
  const char* passSlotNames[kPassSlots][kMaxPassTimestamps] = {};
  uint32_t numPassTimestamps = 0;
  double passTimes[kMaxPassTimestamps] = {}; // ms, summed over passFrames
  const char* passNames[kMaxPassTimestamps] = {};
  uint32_t numPassTimes = 0;
  uint32_t passFrames = 0;

  // reads back the timestamps of the slot this frame writes, 4 frames old, then writes timestamp 0
  void beginPasses(lvk::IContext* ctx, lvk::ICommandBuffer& buffer) {
    if (!profilePasses) {
      return;
    }
    const uint32_t slot = frameIndex % kPassSlots;
    const uint32_t num = passSlotTimestamps[slot];
    if (num) {
      uint64_t timestamps[kMaxPassTimestamps] = {};
      ctx->getQueryPoolResults(queryPoolPasses, slot * kMaxPassTimestamps, num, sizeof(uint64_t) * num, timestamps, sizeof(uint64_t));
      if (num != numPassTimes) {
        passFrames = 0;
        numPassTimes = num;
        std::fill(passTimes, passTimes + kMaxPassTimestamps, 0.0);
      }
      for (uint32_t i = 1; i != num; i++) {
        passTimes[i] += double(timestamps[i] - timestamps[i - 1]) * ctx->getTimestampPeriodToMs();
        passNames[i] = passSlotNames[slot][i];
      }
      passFrames++;
    }
    passSlotTimestamps[slot] = 0;
    buffer.cmdResetQueryPool(queryPoolPasses, slot * kMaxPassTimestamps, kMaxPassTimestamps);
    numPassTimestamps = 0;
    markPass(buffer, nullptr);
  }
  void markPass(lvk::ICommandBuffer& buffer, const char* name) {
    if (!profilePasses) {
      return;
    }
    LVK_ASSERT(numPassTimestamps < kMaxPassTimestamps);
    const uint32_t slot = frameIndex % kPassSlots;
    passSlotNames[slot][numPassTimestamps] = name;
    buffer.cmdWriteTimestamp(queryPoolPasses, slot * kMaxPassTimestamps + numPassTimestamps++);
  }
  void endPasses() {
    if (profilePasses) {
      passSlotTimestamps[frameIndex % kPassSlots] = numPassTimestamps;
    }
  }
  // the mean of every pass since the last call
  void logPassTimes() {
    if (!passFrames) {
      return;
    }
    double total = 0.0;
    for (uint32_t i = 1; i != numPassTimes; i++) {
      total += passTimes[i];
    }
    LLOGL("NSS GPU time per pass, ms (mean of %u frames, total %.2f):\n", passFrames, total / passFrames);
    for (uint32_t i = 1; i != numPassTimes; i++) {
      LLOGL("  %6.2f  %4.1f%%  %s\n", passTimes[i] / passFrames, 100.0 * passTimes[i] / total, passNames[i]);
    }
    passFrames = 0;
    std::fill(passTimes, passTimes + kMaxPassTimestamps, 0.0);
  }

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
    graphReady = false;
    cooperativeReady = false;

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

    auto graphTensorDesc = [](const int64_t (&dims)[4]) -> lvk::TensorDesc {
      return {
          .format = lvk::Format_R_I8,
          .rank = 4,
          .dimensions = {dims[0], dims[1], dims[2], dims[3]},
          .usage = lvk::TensorUsageBits_DataGraph,
      };
    };
    // the tensors' buffers: the data graph reaches them through the tensors aliasing them, the cooperative matrices directly
    auto createTensorBuffer = [ctx, &graphTensorDesc](const int64_t (&dims)[4], const char* debugName) -> lvk::Holder<lvk::BufferHandle> {
      return ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = ctx->getTensorBufferSize(graphTensorDesc(dims)),
          .debugName = debugName,
      });
    };
    bufTensorInput = createTensorBuffer(dimsInput, "NSS: input tensor");
    bufTensorKpn = createTensorBuffer(dimsKpn, "NSS: KPN coefficients");
    bufTensorFeedback = createTensorBuffer(dimsFeedback, "NSS: temporal feedback");

    // the same network as int8 cooperative matrices in compute shaders, which needs no VK_ARM_* extension; 16-wide tiles pad
    // its 4 to 64 outputs the least
    const CooperativeMatrixShape tile =
        getCooperativeMatrixShape(ctx, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR, 16);
    std::vector<CooperativeLayer> layers;
    std::vector<glm::ivec4> activations;
    uint32_t cooperativeOutputs[2] = {};
    const bool isComputable = tile.m && buildCooperativeLayers(spirv, vgf.getConstants(), layers, activations, cooperativeOutputs);
    uint32_t sharedMemory = 0;
    for (const CooperativeLayer& layer : layers) {
      sharedMemory = std::max(sharedMemory, getCooperativeSharedMemory(layer, tile, 1));
    }
    const bool isSharedMemoryEnough = sharedMemory <= tile.maxSharedMemorySize;
    cooperativeReady = isComputable && isSharedMemoryEnough && activations[0] == glm::ivec4(1, kGraphHeight, kGraphWidth, kInputChannels) &&
                       activations[cooperativeOutputs[indexKpn]] == shapeKpn &&
                       activations[cooperativeOutputs[indexFeedback]] == shapeFeedback;
    if (!cooperativeReady) {
      LLOGW(!tile.m         ? "NSS: int8 cooperative matrices (VK_KHR_cooperative_matrix) are not available\n"
            : !isComputable ? "NSS: the graph has operators the cooperative matrix path does not compute\n"
                            : "NSS: the cooperative matrix path needs more shared memory than the device has\n");
    }

    const bool hasTensors = ctx->supportsTensorsARM() && ctx->supportsDataGraphARM();
    if (!hasTensors) {
      LLOGW(
          "NSS: VK_ARM_tensors and VK_ARM_data_graph are not available (configure with LVK_WITH_ML_EMULATION_LAYER=ON to emulate them)\n");
    }
    lvk::Result result;
    bool tensorsReady = hasTensors;
    if (tensorsReady) {
      tensorInput = ctx->createTensorBufferAliasing(bufTensorInput, 0, graphTensorDesc(dimsInput), "NSS: input tensor", &result);
      tensorsReady = result.isOk();
    }
    if (tensorsReady) {
      tensorKpn = ctx->createTensorBufferAliasing(bufTensorKpn, 0, graphTensorDesc(dimsKpn), "NSS: KPN coefficients", &result);
      tensorsReady = result.isOk();
    }
    if (hasTensors && !tensorsReady) {
      LLOGW("NSS: cannot create the tensors: %s\n", result.message);
    }
    // the cooperative matrices always write the feedback into its buffer and copy it into the texture
    feedbackAliasing = tensorsReady &&
                       (ctx->getTensorFormatSupport(lvk::Format_R_I8, lvk::TensorTiling_Optimal) & lvk::TensorUsageBits_ImageAliasing);
    if (feedbackAliasing) {
      texFeedback = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageFlags(lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_TensorAliasing |
                                          (cooperativeReady ? lvk::TextureUsageBits_Storage : 0)),
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
      if (tensorsReady) {
        tensorFeedback =
            ctx->createTensorBufferAliasing(bufTensorFeedback, 0, graphTensorDesc(dimsFeedback), "NSS: temporal feedback", &result);
        tensorsReady = result.isOk();
        if (!tensorsReady) {
          LLOGW("NSS: cannot create the feedback tensor: %s\n", result.message);
        }
      }
      texFeedback = ctx->createTexture({
          .type = lvk::TextureType_2D,
          .format = lvk::Format_RGBA_SN8,
          .dimensions = {kGraphWidth, kGraphHeight},
          .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
          .debugName = "NSS: temporal feedback (texture)",
      });
    }
    if (tensorsReady) {
      LLOGL("NSS: temporal feedback via %s\n", feedbackAliasing ? "a texture aliased by the tensor" : "a buffer copied into a texture");
    }

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

    if (tensorsReady) {
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
      graphReady = result.isOk();
      if (graphReady) {
        LLOGL("NSS: data graph pipeline created in %.1f ms\n", 1000.0 * (glfwGetTime() - start));
      } else {
        LLOGW("NSS: cannot create the data graph pipeline: %s\n", result.message);
      }
    }

    if (cooperativeReady) {
      const double start = glfwGetTime();
      // the activations the layers write; the input and the outputs of the graph are the tensors' buffers
      activationBuffers.assign(activations.size(), {});
      activationBuffers[0] = bufTensorInput;
      activationBuffers[cooperativeOutputs[indexKpn]] = bufTensorKpn;
      activationBuffers[cooperativeOutputs[indexFeedback]] = bufTensorFeedback;
      for (uint32_t i = 1; i != activations.size(); i++) {
        if (activationBuffers[i].valid()) {
          continue;
        }
        const glm::ivec4 shape = activations[i];
        cooperativeActivations.push_back(ctx->createBuffer({
            .usage = lvk::BufferUsageBits_Storage,
            .storage = lvk::StorageType_Device,
            .size = size_t(shape.x) * shape.y * shape.z * shape.w,
            .debugName = "NSS: activation",
        }));
        activationBuffers[i] = cooperativeActivations.back();
      }
      std::vector<int8_t> weights;
      std::vector<int32_t> params;
      for (const CooperativeLayer& layer : layers) {
        const size_t weightsAt = weights.size();
        const size_t paramsAt = sizeof(int32_t) * params.size();
        const std::string code = kNssCooperativeExtensions + buildCooperativeLayer(layer, tile, weights, params) + kNssCooperativeCS;
        const uint32_t rows = getCooperativeRows(layer, tile);
        CooperativeDispatch dispatch = {
            .sm = ctx->createShaderModule({code.c_str(), lvk::Stage_Comp, "Shader Module: NSS cooperative matrices (comp)"}),
            .sources = {layer.sources[0], layer.sources[1]},
            .output = layer.output,
            .groups = (uint32_t(layer.outputShape.z + tile.m - 1) / tile.m * ((uint32_t(layer.outputShape.y) + rows - 1) / rows) +
                       kCooperativeTilesPerGroup - 1) /
                      kCooperativeTilesPerGroup,
            .weightsAt = weightsAt,
            .paramsAt = paramsAt,
        };
        snprintf(dispatch.name,
                 sizeof(dispatch.name),
                 "layer %u: %ux%u/%u%s, %u -> %u channels, %ux%u, %ux%u tiles, %u rows",
                 (uint32_t)cooperativeDispatches.size(),
                 (uint32_t)layer.kernelShape.y,
                 (uint32_t)layer.kernelShape.z,
                 (uint32_t)layer.stride,
                 layer.upsample ? " upsampled" : "",
                 (uint32_t)layer.kernelShape.w,
                 (uint32_t)layer.kernelShape.x,
                 (uint32_t)layer.outputShape.z,
                 (uint32_t)layer.outputShape.y,
                 tile.m,
                 tile.n,
                 rows);
        dispatch.pipeline = ctx->createComputePipeline({.smComp = dispatch.sm, .debugName = "NSS: cooperative matrices"});
        cooperativeDispatches.push_back(std::move(dispatch));
      }
      cooperativeWeights = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = weights.size(),
          .data = weights.data(),
          .debugName = "NSS: cooperative matrix weights",
      });
      cooperativeParams = ctx->createBuffer({
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = sizeof(int32_t) * params.size(),
          .data = params.data(),
          .debugName = "NSS: cooperative matrix parameters",
      });
      LLOGL("NSS: %u layers as %ux%ux%u int8 cooperative matrices, created in %.1f ms\n",
            (uint32_t)layers.size(),
            tile.m,
            tile.n,
            tile.k,
            1000.0 * (glfwGetTime() - start));
    }

    queryPoolPasses = ctx->createQueryPool(kPassSlots * kMaxPassTimestamps, "Query pool: NSS passes");
    supported = graphReady || cooperativeReady;
    path = graphReady ? NssPath_DataGraph : NssPath_Cooperative;
    resetHistory = true;
    return supported;
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
    beginPasses(ctx, buffer);
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
    markPass(buffer, "depth scatter");

    // the cooperative matrices read and write the tensors' buffers directly
    const bool isGraph = path == NssPath_DataGraph;
    lvk::TensorHandle tensorsInput[] = {tensorInput};
    lvk::TensorHandle tensorsOutput[] = {tensorKpn, tensorFeedback};
    lvk::TensorHandle tensorsFeedback[] = {tensorFeedback};

    buffer.cmdPushDebugGroupLabel("NSS: preprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePreprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch(
        {groups(kGraphWidth), groups(kGraphHeight), 1},
        {
            .sampledImages = {res_.texColorLR, res_.texDepthLR, res_.texMotionLR, texHistory, texFeedback, lumaDerivTm1, texDepthTm1},
            .storageImages = {lumaDeriv, texNearestDepthCoord},
            .buffers = {bufConstants, bufTensorInput},
            .tensors = {tensorsInput, isGraph ? 1u : 0u},
        });
    buffer.cmdPopDebugGroupLabel();
    markPass(buffer, "preprocess");

    if (isGraph) {
      buffer.cmdPushDebugGroupLabel("NSS: data graph", 0xff0000ff);
      buffer.cmdDispatchDataGraph(pipelineGraph, graphInputs, graphOutputs);
      buffer.cmdPopDebugGroupLabel();
      markPass(buffer, "data graph");
    } else {
      buffer.cmdPushDebugGroupLabel("NSS: cooperative matrices", 0xff0000ff);
      for (const CooperativeDispatch& layer : cooperativeDispatches) {
        const lvk::BufferHandle source0 = activationBuffers[layer.sources[0]];
        const lvk::BufferHandle source1 = activationBuffers[layer.sources[1]];
        const lvk::BufferHandle destination = activationBuffers[layer.output];
        buffer.cmdBindComputePipeline(layer.pipeline);
        buffer.cmdPushConstants(NssCooperativePushConstants{
            .source0 = ctx->gpuAddress(source0),
            .source1 = ctx->gpuAddress(source1),
            .destination = ctx->gpuAddress(destination),
            .weights = ctx->gpuAddress(cooperativeWeights, layer.weightsAt),
            .params = ctx->gpuAddress(cooperativeParams, layer.paramsAt),
        });
        buffer.cmdDispatch({layer.groups, 1, 1}, {.buffers = {source0, source1, destination}});
        markPass(buffer, layer.name);
      }
      buffer.cmdPopDebugGroupLabel();
    }

    if (!isGraph || !feedbackAliasing) {
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
                             .buffers = {bufTensorFeedback},
                             .tensors = {tensorsFeedback, isGraph ? 1u : 0u},
                         });
      buffer.cmdPopDebugGroupLabel();
      markPass(buffer, "feedback to texture");
    }

    buffer.cmdPushDebugGroupLabel("NSS: postprocess", 0xff0000ff);
    buffer.cmdBindComputePipeline(pipelinePostprocess);
    buffer.cmdPushConstants(pc);
    buffer.cmdDispatch({groups(kOutputWidth), groups(kOutputHeight), 1},
                       {
                           .sampledImages = {res_.texColorLR, res_.texMotionLR, texHistory, texFeedback, texNearestDepthCoord},
                           .storageImages = {texOut},
                           .buffers = {bufConstants, bufTensorKpn},
                           .tensors = {tensorsOutput, isGraph ? 2u : 0u},
                       });
    buffer.cmdPopDebugGroupLabel();
    markPass(buffer, "postprocess");
    endPasses();
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
      .framebufferScalar = 1, // NSS upscales to the native resolution
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
  bool preferCooperative = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--no-nss")) {
      renderMode_ = RenderMode_HalfResolution;
    } else if (!strcmp(argv[i], "--ao-samples") && i + 1 < argc) {
      aoSamples_ = std::clamp(atoi(argv[++i]), 0, 16);
    } else if (!strcmp(argv[i], "--camera-rotate")) {
      cameraRotate = true;
    } else if (!strcmp(argv[i], "--timings")) {
      measureTime = true;
    } else if (!strcmp(argv[i], "--nss-cm")) {
      preferCooperative = true;
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

  nss_.profilePasses = measureTime;
  nss_.init(ctx_);
  // `--nss-cm` prefers the cooperative matrices to the data graph
  nss_.path = preferCooperative && nss_.cooperativeReady ? NssPath_Cooperative : nss_.path;
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
    const mat4 projNoJitter = glm::perspective(kCameraFovY, aspectRatio, kCameraNear, kCameraFar);
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
        ImGui::Text("W/S/A/D - camera movement");
        ImGui::Text("1/2 - camera up/down");
        ImGui::Text("Shift - fast movement");
        ImGui::Text("Space - reset camera");
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
        // the network on either, each grayed out without the extensions it runs on
        const float indentSize = 16.0f;
        ImGui::Indent(indentSize);
        ImGui::BeginDisabled(!nss_.graphReady || renderMode_ != RenderMode_HalfResolutionNSS);
        ImGui::RadioButton("NSS (VK_ARM_tensors + VK_ARM_data_graph)", &nss_.path, NssPath_DataGraph);
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!nss_.cooperativeReady || renderMode_ != RenderMode_HalfResolutionNSS);
        ImGui::RadioButton("NSS (VK_KHR_cooperative_matrix)", &nss_.path, NssPath_Cooperative);
        ImGui::EndDisabled();
        ImGui::Unindent(indentSize);
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
          ImGui::Text("NSS needs VK_ARM_tensors + VK_ARM_data_graph or VK_KHR_cooperative_matrix: showing the %ux%u input",
                      kRenderWidth,
                      kRenderHeight);
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
    if (measureTime && frameCounter % 600 == 0) {
      nss_.logPassTimes();
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
