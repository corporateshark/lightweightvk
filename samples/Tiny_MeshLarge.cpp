/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A brief tutorial how to run this beast:
 *
 * 1) Run the script "deploy_deps.py" from the LightweightVK root folder.
 * 2) Run the script "deploy_content.py" from the LightweightVK root folder.
 *    NOTE: In case of Android, script is "deploy_content_android.py", see README.md
 * 3) Run this app.
 *
 */

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <stdio.h>
#include <thread>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

#include <ktx-software/lib/gl_format.h>
#include <ktx-software/lib/vkformat_enum.h>
#include <ktx.h>
#include <ldrutils/lmath/Colors.h>
#include <ldrutils/lutils/ScopeExit.h>

#include <fast_obj.h>
#include <meshoptimizer.h>
#include <shared/Camera.h>
#include <shared/UtilsCubemap.h>
#include <shared/UtilsFPS.h>
#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>
#include <taskflow/taskflow.hpp>

#include <implot/implot.h>
#include <lvk/HelpersImGui.h>
#include <lvk/LVK.h>

#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

constexpr uint32_t kMeshCacheVersion = 0xC0DE000A;
#if !defined(__APPLE__)
constexpr int kNumSamplesMSAA = 8;
#else
constexpr int kNumSamplesMSAA = 4;
#endif
#if defined(__APPLE__) || defined(ANDROID)
constexpr bool kEnableCompression = false;
#else
constexpr bool kEnableCompression = true;
#endif
constexpr bool kPreferIntegratedGPU = false;
#if defined(NDEBUG)
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif // NDEBUG

std::string folderThirdParty;
std::string folderContentRoot;

std::unique_ptr<lvk::ImGuiRenderer> imgui_;

enum GPUTimestamp {
  GPUTimestamp_BeginSceneRendering = 0,
  GPUTimestamp_EndSceneRendering,

  GPUTimestamp_BeginComputePass,
  GPUTimestamp_EndComputePass,

  GPUTimestamp_BeginPresent,
  GPUTimestamp_EndPresent,

  GPUTimestamp_NUM_TIMESTAMPS
};
lvk::Holder<lvk::QueryPoolHandle> queryPoolTimestamps_;
uint64_t pipelineTimestamps[GPUTimestamp_NUM_TIMESTAMPS] = {};
double timestampBeginRendering = 0;
double timestampEndRendering = 0;

// https://github.com/KhronosGroup/MoltenVK/issues/2106
// TODO: After fix, use the only shader with binding = 2 to be compatible with Vulkan image layout on other platforms.
#ifdef __APPLE__
const char* kCodeComputeTest = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0, rgba8) uniform image2D kTextures2DInOut[];

layout(push_constant) uniform constants {
   uint tex;
   uint width;
   uint height;
} pc;

void main() {
   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

   if (pos.x < pc.width && pos.y < pc.height) {
     vec4 pixel = imageLoad(kTextures2DInOut[pc.tex], pos);
     float luminance = dot(pixel, vec4(0.299, 0.587, 0.114, 0.0)); // https://www.w3.org/TR/AERT/#color-contrast
     imageStore(kTextures2DInOut[pc.tex], pos, vec4(vec3(luminance), 1.0));
   }
}
)";
#else
const char* kCodeComputeTest = R"(
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 2, rgba8) uniform image2D kTextures2DInOut[];

layout(push_constant) uniform constants {
   uint tex;
   uint width;
   uint height;
} pc;

void main() {
   ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

   if (pos.x < pc.width && pos.y < pc.height) {
     vec4 pixel = imageLoad(kTextures2DInOut[pc.tex], pos);
     float luminance = dot(pixel, vec4(0.299, 0.587, 0.114, 0.0)); // https://www.w3.org/TR/AERT/#color-contrast
     imageStore(kTextures2DInOut[pc.tex], pos, vec4(vec3(luminance), 1.0));
   }
}
)";
#endif

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

const char* kCodeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal; // Octahedral 16-bit https://www.shadertoy.com/view/llfcRl
layout (location=3) in uint mtlIndex;

struct Material {
   vec4 ambient;
   vec4 diffuse;
   int texAmbient;
   int texDiffuse;
   int texAlpha;
   int padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
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
struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
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
  mat4 light = pc.perFrame.light;
  mtl = pc.materials.mtl[mtlIndex];
  gl_Position = proj * view * model * vec4(pos, 1.0);

  // Compute the normal in world-space
  mat3 norm_matrix = transpose(inverse(mat3(model)));
  vtx.normal = normalize(norm_matrix * unpackOctahedral16(normal));
  vtx.uv = uv;
  vtx.shadowCoords = light * model * vec4(pos, 1.0);
}
)";

const char* kCodeVS_Wireframe = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
   PerObject perObject;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeFS_Wireframe = R"(
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(1.0);
};
)";

const char* kCodeFS = R"(

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

struct Material {
  vec4 ambient;
  vec4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 shadowCoords;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

layout (location=0) in PerVertex vtx;
layout (location=5) flat in Material mtl;

layout (location=0) out vec4 out_FragColor;

layout (constant_id = 0) const bool bDrawNormals = false;

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
  float NdotL1 = clamp(dot(n, normalize(vec3(-1, 1,+1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(vec3(-1, 1,-1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1+NdotL2);
  // IBL diffuse
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = textureBindlessCube(pc.perFrame.texSkyboxIrradiance, pc.perFrame.sampler0, n) * Kd * (vec4(1.0) - f0);
  out_FragColor = bDrawNormals ?
    vec4(0.5 * (n+vec3(1.0)), 1.0) :
    Ka + diffuse * shadow(vtx.shadowCoords);
};
)";

const char* kShadowVS = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kSkyboxVS = R"(
layout (location=0) out vec3 textureCoords;

const vec3 positions[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

const int indices[36] = int[36](
	0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
);

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

layout(push_constant) uniform constants
{
	PerFrame perFrame;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  // discard translation
  view = mat4(view[0], view[1], view[2], vec4(0, 0, 0, 1));
  mat4 transform = proj * view;
  vec3 pos = positions[indices[gl_VertexIndex]];
  gl_Position = (transform * vec4(pos, 1.0)).xyww;

  // skybox
  textureCoords = pos;
}

)";
const char* kSkyboxFS = R"(
layout (location=0) in vec3 textureCoords;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

void main() {
  out_FragColor = textureBindlessCube(pc.perFrame.texSkyboxRadiance, pc.perFrame.sampler0, textureCoords);
}
)";

using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

int width_ = 0;
int height_ = 0;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;
lvk::Framebuffer fbMain_; // swapchain
lvk::Framebuffer fbOffscreen_;
lvk::Holder<lvk::TextureHandle> fbOffscreenColor_;
lvk::Holder<lvk::TextureHandle> fbOffscreenDepth_;
lvk::Holder<lvk::TextureHandle> fbOffscreenResolve_;
lvk::Framebuffer fbShadowMap_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowVert_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxVert_;
lvk::Holder<lvk::ShaderModuleHandle> smSkyboxFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smGrayscaleComp_;
lvk::Holder<lvk::ComputePipelineHandle> computePipelineState_Grayscale_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshNormals_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshWireframe_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Shadow_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Skybox_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;
lvk::Holder<lvk::BufferHandle> vb0_, ib0_; // buffers for vertices and indices
lvk::Holder<lvk::BufferHandle> sbMaterials_; // storage buffer for materials
lvk::Holder<lvk::BufferHandle> ubPerFrame_, ubPerFrameShadow_, ubPerObject_;
lvk::Holder<lvk::SamplerHandle> sampler_;
lvk::Holder<lvk::SamplerHandle> samplerShadow_;
lvk::Holder<lvk::TextureHandle> textureDummyWhite_;
lvk::Holder<lvk::TextureHandle> skyboxTextureReference_;
lvk::Holder<lvk::TextureHandle> skyboxTextureIrradiance_;
lvk::RenderPass renderPassOffscreen_;
lvk::RenderPass renderPassMain_;
lvk::RenderPass renderPassShadow_;
lvk::DepthState depthState_;
lvk::DepthState depthStateLEqual_;

// scene navigation
CameraPositioner_FirstPerson positioner_(vec3(-100, 40, -47), vec3(0, 35, 0), vec3(0, 1, 0));
Camera camera_(positioner_);
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;
bool enableComputePass_ = false;
bool enableWireframe_ = false;
bool showPerfStats_ = false;
bool drawNormals_ = false;

bool isShadowMapDirty_ = true;

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
  mat4 light;
  uint32_t texSkyboxRadiance = 0;
  uint32_t texSkyboxIrradiance = 0;
  uint32_t texShadow = 0;
  uint32_t sampler = 0;
  uint32_t samplerShadow = 0;
} perFrame_;

struct UniformsPerObject {
  mat4 model;
};
#define MAX_MATERIAL_NAME 128

struct CachedMaterial {
  char name[MAX_MATERIAL_NAME] = {};
  vec3 ambient = vec3(0.0f);
  vec3 diffuse = vec3(0.0f);
  char ambient_texname[MAX_MATERIAL_NAME] = {};
  char diffuse_texname[MAX_MATERIAL_NAME] = {};
  char alpha_texname[MAX_MATERIAL_NAME] = {};
};

// this goes into our GLSL shaders
struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
  uint32_t texAmbient = 0;
  uint32_t texDiffuse = 0;
  uint32_t texAlpha = 0;
  uint32_t padding[1];
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<CachedMaterial> cachedMaterials_;
std::vector<GPUMaterial> materials_;

struct MaterialTextures {
  lvk::TextureHandle ambient;
  lvk::TextureHandle diffuse;
  lvk::TextureHandle alpha;
};

std::vector<MaterialTextures> textures_; // same indexing as in materials_

struct LoadedImage {
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t channels = 0;
  uint8_t* pixels = nullptr;
  std::string debugName;
  std::string compressedFileName;
};

struct LoadedMaterial {
  size_t idx = 0;
  LoadedImage ambient;
  LoadedImage diffuse;
  LoadedImage alpha;
};

// file name -> LoadedImage
std::mutex imagesCacheMutex_;
std::unordered_map<std::string, LoadedImage> imagesCache_; // accessible only from the loader pool (multiple threads)
std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache_; // accessible the main thread
std::vector<LoadedMaterial> loadedMaterials_;
std::mutex loadedMaterialsMutex_;
std::atomic<bool> loaderShouldExit_ = false;
std::atomic<uint32_t> remainingMaterialsToLoad_ = 0;
std::unique_ptr<tf::Executor> loaderPool_ = std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));

static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static std::string convertFileName(std::string fileName) {
  // generate compressed filename
  const std::string compressedPathPrefix = folderContentRoot;

  if (fileName.find(compressedPathPrefix) == 0) {
    // remove leading path
    fileName = fileName.substr(compressedPathPrefix.length());
  }

  std::replace(fileName.begin(), fileName.end(), ':', '_');
  std::replace(fileName.begin(), fileName.end(), '.', '_');
  std::replace(fileName.begin(), fileName.end(), '/', '_');
  std::replace(fileName.begin(), fileName.end(), '\\', '_');

  // return absolute compressed filename
  return compressedPathPrefix + fileName + ".ktx";
}
static void stringReplaceAll(std::string& s, const std::string& searchString, const std::string& replaceString) {
  size_t pos = 0;
  while ((pos = s.find(searchString, pos)) != std::string::npos) {
    s.replace(pos, searchString.length(), replaceString);
  }
}

bool initModel();
void loadSkyboxTexture();
void loadMaterials();
void createPipelines();
void createShadowMap();
void createOffscreenFramebuffer();

bool init() {
  {
    const uint32_t pixel = 0xFFFFFFFF;
    textureDummyWhite_ = ctx_->createTexture(
        {
            .type = lvk::TextureType_2D,
            .format = lvk::Format_R_UN8,
            .dimensions = {1, 1},
            .usage = lvk::TextureUsageBits_Sampled,
            .swizzle = {lvk::Swizzle_1, lvk::Swizzle_1, lvk::Swizzle_1, lvk::Swizzle_1},
            .data = &pixel,
            .debugName = "dummy 1x1 (white)",
        },
        nullptr);
  }

  ubPerFrame_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerFrame),
      .debugName = "Buffer: uniforms (per frame)",
  });
  ubPerFrameShadow_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerFrame),
      .debugName = "Buffer: uniforms (per frame shadow)",
  });
  ubPerObject_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Uniform,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(UniformsPerObject),
      .debugName = "Buffer: uniforms (per object)",
  });

  depthState_ = {.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true};
  depthStateLEqual_ = {.compareOp = lvk::CompareOp_LessEqual, .isDepthWriteEnabled = true};

  sampler_ = ctx_->createSampler({
      .mipMap = lvk::SamplerMip_Linear,
      .wrapU = lvk::SamplerWrap_Repeat,
      .wrapV = lvk::SamplerWrap_Repeat,
      .debugName = "Sampler: linear",
  });
  samplerShadow_ = ctx_->createSampler({
      .wrapU = lvk::SamplerWrap_Clamp,
      .wrapV = lvk::SamplerWrap_Clamp,
      .depthCompareOp = lvk::CompareOp_LessEqual,
      .depthCompareEnabled = true,
      .debugName = "Sampler: shadow",
  });

  renderPassOffscreen_ = {.color = {{
                              .loadOp = lvk::LoadOp_Clear,
                              .storeOp = kNumSamplesMSAA > 1 ? lvk::StoreOp_MsaaResolve : lvk::StoreOp_Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, 1.0f},
                          }},
                          .depth = {
                              .loadOp = lvk::LoadOp_Clear,
                              .storeOp = lvk::StoreOp_Store,
                              .clearDepth = 1.0f,
                          }};

  renderPassMain_ = {
      .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
  };
  renderPassShadow_ = {
      .color = {},
      .depth = {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f},
  };

  fbMain_ = {
      .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
  };

  createShadowMap();
  createOffscreenFramebuffer();
  createPipelines();

  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, (folderThirdParty + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), float(height_) / 70.0f);

  queryPoolTimestamps_ = ctx_->createQueryPool(GPUTimestamp_NUM_TIMESTAMPS, "queryPoolTimestamps_");

  if (!initModel()) {
    return false;
  }

  loadSkyboxTexture();
  loadMaterials();

  return true;
}

void destroy() {
  imgui_ = nullptr;

  vb0_ = nullptr;
  ib0_ = nullptr;
  sbMaterials_ = nullptr;
  ubPerFrame_ = nullptr;
  ubPerFrameShadow_ = nullptr;
  ubPerObject_ = nullptr;
  smMeshVert_ = nullptr;
  smMeshFrag_ = nullptr;
  smMeshWireframeVert_ = nullptr;
  smMeshWireframeFrag_ = nullptr;
  smShadowVert_ = nullptr;
  smShadowFrag_ = nullptr;
  smFullscreenVert_ = nullptr;
  smFullscreenFrag_ = nullptr;
  smSkyboxVert_ = nullptr;
  smSkyboxFrag_ = nullptr;
  smGrayscaleComp_ = nullptr;
  renderPipelineState_Mesh_ = nullptr;
  renderPipelineState_MeshNormals_ = nullptr;
  renderPipelineState_MeshWireframe_ = nullptr;
  renderPipelineState_Shadow_ = nullptr;
  renderPipelineState_Skybox_ = nullptr;
  renderPipelineState_Fullscreen_ = nullptr;
  computePipelineState_Grayscale_ = nullptr;
  textureDummyWhite_ = nullptr;
  skyboxTextureReference_ = nullptr;
  skyboxTextureIrradiance_ = nullptr;
  textures_.clear();
  texturesCache_.clear();
  sampler_ = nullptr;
  samplerShadow_ = nullptr;
  ctx_->destroy(fbMain_);
  ctx_->destroy(fbShadowMap_);
  fbOffscreenColor_ = nullptr;
  fbOffscreenDepth_ = nullptr;
  fbOffscreenResolve_ = nullptr;
  queryPoolTimestamps_ = nullptr;
  ctx_ = nullptr;

  printf("Waiting for the loader thread to exit...\n");

  loaderPool_ = nullptr;
}

std::string normalizeTextureName(const char* n) {
  if (!n)
    return std::string();
  LVK_ASSERT(strlen(n) < MAX_MATERIAL_NAME);
  std::string name(n);
#if defined(__linux__) || defined(__APPLE__) || defined(ANDROID)
  std::replace(name.begin(), name.end(), '\\', '/');
#endif
  return name;
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
    strcat(mtl.ambient_texname, normalizeTextureName(mesh->textures[m.map_Ka].name).c_str());
    strcat(mtl.diffuse_texname, normalizeTextureName(mesh->textures[m.map_Kd].name).c_str());
    strcat(mtl.alpha_texname, normalizeTextureName(mesh->textures[m.map_d].name).c_str());
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
  const std::string cacheFileName = folderContentRoot + "cache.data";

  if (!loadFromCache(cacheFileName.c_str())) {
    if (!LVK_VERIFY(loadAndCache(cacheFileName.c_str()))) {
      LVK_ASSERT_MSG(false, "Cannot load 3D model");
      return false;
    }
  }

  for (const auto& mtl : cachedMaterials_) {
    materials_.push_back(
        GPUMaterial{vec4(mtl.ambient, 1.0f), vec4(mtl.diffuse, 1.0f), textureDummyWhite_.index(), textureDummyWhite_.index()});
  }

  sbMaterials_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Storage,
                                     .storage = lvk::StorageType_Device,
                                     .size = sizeof(GPUMaterial) * materials_.size(),
                                     .data = materials_.data(),
                                     .debugName = "Buffer: materials"},
                                    nullptr);

  vb0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Vertex,
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(VertexData) * vertexData_.size(),
                             .data = vertexData_.data(),
                             .debugName = "Buffer: vertex"},
                            nullptr);
  ib0_ = ctx_->createBuffer({.usage = lvk::BufferUsageBits_Index,
                             .storage = lvk::StorageType_Device,
                             .size = sizeof(uint32_t) * indexData_.size(),
                             .data = indexData_.data(),
                             .debugName = "Buffer: index"},
                            nullptr);
  return true;
}

void createPipelines() {
  if (renderPipelineState_Mesh_.valid()) {
    return;
  }

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

  // shadow
  const lvk::VertexInput vdescs = {
      .attributes = {{.format = lvk::VertexFormat::Float3, .offset = offsetof(VertexData, position)}},
      .inputBindings = {{.stride = sizeof(VertexData)}},
  };

  smMeshVert_ = ctx_->createShaderModule({kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  smMeshFrag_ = ctx_->createShaderModule({kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  smMeshWireframeVert_ = ctx_->createShaderModule({kCodeVS_Wireframe, lvk::Stage_Vert, "Shader Module: main wireframe (vert)"});
  smMeshWireframeFrag_ = ctx_->createShaderModule({kCodeFS_Wireframe, lvk::Stage_Frag, "Shader Module: main wireframe (frag)"});
  smShadowVert_ = ctx_->createShaderModule({kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)"});
  smShadowFrag_ = ctx_->createShaderModule({kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
  smFullscreenVert_ = ctx_->createShaderModule({kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)"});
  smFullscreenFrag_ = ctx_->createShaderModule({kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)"});
  smSkyboxVert_ = ctx_->createShaderModule({kSkyboxVS, lvk::Stage_Vert, "Shader Module: skybox (vert)"});
  smSkyboxFrag_ = ctx_->createShaderModule({kSkyboxFS, lvk::Stage_Frag, "Shader Module: skybox (frag)"});

  {
    lvk::RenderPipelineDesc desc = {
        .vertexInput = vdesc,
        .smVert = smMeshVert_,
        .smFrag = smMeshFrag_,
        .color = {{.format = ctx_->getFormat(fbOffscreen_.color[0].texture)}},
        .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
        .cullMode = lvk::CullMode_Back,
        .frontFaceWinding = lvk::WindingMode_CCW,
        .samplesCount = kNumSamplesMSAA,
        .debugName = "Pipeline: mesh",
    };

    renderPipelineState_Mesh_ = ctx_->createRenderPipeline(desc, nullptr);

    const uint32_t drawNormals = 1;

    desc.specInfo = {.entries = {{.constantId = 0, .size = sizeof(uint32_t)}}, .data = &drawNormals, .dataSize = sizeof(drawNormals)},

    renderPipelineState_MeshNormals_ = ctx_->createRenderPipeline(desc, nullptr);

    desc.polygonMode = lvk::PolygonMode_Line;
    desc.vertexInput = vdescs; // positions-only
    desc.smVert = smMeshWireframeVert_;
    desc.smFrag = smMeshWireframeFrag_;
    desc.debugName = "Pipeline: mesh (wireframe)";
    renderPipelineState_MeshWireframe_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  // shadow
  renderPipelineState_Shadow_ = ctx_->createRenderPipeline(
      lvk::RenderPipelineDesc{
          .vertexInput = vdescs,
          .smVert = smShadowVert_,
          .smFrag = smShadowFrag_,
          .depthFormat = ctx_->getFormat(fbShadowMap_.depthStencil.texture),
          .cullMode = lvk::CullMode_None,
          .debugName = "Pipeline: shadow",
      },
      nullptr);

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

  // skybox
  {
    const lvk::RenderPipelineDesc desc = {
        .smVert = smSkyboxVert_,
        .smFrag = smSkyboxFrag_,
        .color = {{
            .format = ctx_->getFormat(fbOffscreen_.color[0].texture),
        }},
        .depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
        .cullMode = lvk::CullMode_Front,
        .frontFaceWinding = lvk::WindingMode_CCW,
        .samplesCount = kNumSamplesMSAA,
        .debugName = "Pipeline: skybox",
    };

    renderPipelineState_Skybox_ = ctx_->createRenderPipeline(desc, nullptr);
  }

  smGrayscaleComp_ = ctx_->createShaderModule({kCodeComputeTest, lvk::Stage_Comp, "Shader Module: grayscale (comp)"});

  computePipelineState_Grayscale_ = ctx_->createComputePipeline({.smComp = smGrayscaleComp_}, nullptr);
}

void createShadowMap() {
  const uint32_t w = 4096;
  const uint32_t h = 4096;
  const lvk::TextureDesc desc = {
      .type = lvk::TextureType_2D,
      .format = lvk::Format_Z_UN16,
      .dimensions = {w, h},
      .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(w, h),
      .debugName = "Shadow map",
  };
  fbShadowMap_ = {
      .depthStencil = {.texture = ctx_->createTexture(desc).release()},
  };
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

void showTimeGPU();
double getCurrentTimestamp();
void processLoadedMaterials(lvk::ICommandBuffer& buffer);

void render(double delta) {
  LVK_PROFILER_FUNCTION();

  if (!width_ && !height_)
    return;

  lvk::TextureHandle nativeDrawable = ctx_->getCurrentSwapchainTexture();
  fbMain_.color[0].texture = nativeDrawable;

  // imGui
  {
    imgui_->beginFrame(fbMain_);
    ImGui::ShowDemoWindow();

    ImGui::Begin("Keyboard hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S/A/D - camera movement");
    ImGui::Text("1/2 - camera up/down");
    ImGui::Text("Shift - fast movement");
    ImGui::Text("C - toggle compute shader postprocessing");
    ImGui::Text("N - toggle normals");
    ImGui::Text("T - toggle wireframe");
    ImGui::Text("P - show perf stats");
    ImGui::End();

    if (!textures_[1].diffuse.empty()) {
      ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::Image(textures_[1].diffuse.index(), ImVec2(256, 256));
      ImGui::End();
    }

    if (uint32_t num = remainingMaterialsToLoad_.load(std::memory_order_acquire)) {
      ImGui::SetNextWindowPos(ImVec2(0, 0));
      ImGui::Begin("Loading...", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs);
      ImGui::ProgressBar(1.0f - float(num) / cachedMaterials_.size(), ImVec2(ImGui::GetIO().DisplaySize.x, 32));
      ImGui::End();
    }
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

    if (showPerfStats_)
      showTimeGPU();
  }

  positioner_.update(delta, mousePos_, mousePressed_);

  timestampBeginRendering = getCurrentTimestamp();

  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = (float)width_ / (float)height_;

  const mat4 shadowProj = glm::perspective(float(60.0f * (M_PI / 180.0f)), 1.0f, 10.0f, 4000.0f);
  const mat4 shadowView = mat4(vec4(0.772608519f, 0.532385886f, -0.345892131f, 0),
                               vec4(0, 0.544812560f, 0.838557839f, 0),
                               vec4(0.634882748f, -0.647876859f, 0.420926809f, 0),
                               vec4(-58.9244843f, -30.4530792f, -508.410126f, 1.0f));
  const mat4 scaleBias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

  perFrame_ = UniformsPerFrame{
      .proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f),
      .view = camera_.getViewMatrix(),
      .light = scaleBias * shadowProj * shadowView,
      .texSkyboxRadiance = skyboxTextureReference_.index(),
      .texSkyboxIrradiance = skyboxTextureIrradiance_.index(),
      .texShadow = fbShadowMap_.depthStencil.texture.index(),
      .sampler = sampler_.index(),
      .samplerShadow = samplerShadow_.index(),
  };

  const UniformsPerObject perObject = {
      .model = glm::scale(mat4(1.0f), vec3(0.05f)),
  };

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  processLoadedMaterials(buffer);

  buffer.cmdUpdateBuffer(ubPerFrame_, 0, sizeof(perFrame_), &perFrame_);
  buffer.cmdUpdateBuffer(ubPerObject_, 0, sizeof(perObject), &perObject);

  // Pass 1: shadows
  if (isShadowMapDirty_) {
    const UniformsPerFrame perFrameShadow{
        .proj = shadowProj,
        .view = shadowView,
    };
    buffer.cmdUpdateBuffer(ubPerFrameShadow_, 0, sizeof(perFrameShadow), &perFrameShadow);
    buffer.cmdBeginRendering(renderPassShadow_, fbShadowMap_);
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Shadow_);
      buffer.cmdPushDebugGroupLabel("Render Shadows", 0xff0000ff);
      buffer.cmdBindDepthState(depthState_);
      buffer.cmdBindVertexBuffer(0, vb0_, 0);
      struct {
        uint64_t perFrame;
        uint64_t perObject;
      } bindings = {
          .perFrame = ctx_->gpuAddress(ubPerFrameShadow_),
          .perObject = ctx_->gpuAddress(ubPerObject_),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      buffer.cmdPopDebugGroupLabel();
    }
    buffer.cmdEndRendering();
    buffer.transitionToShaderReadOnly(fbShadowMap_.depthStencil.texture);
    buffer.cmdGenerateMipmap(fbShadowMap_.depthStencil.texture);
    isShadowMapDirty_ = false;
  }

#define GPU_TIMESTAMP(timestamp) buffer.cmdWriteTimestamp(queryPoolTimestamps_, timestamp);

  // Pass 2: mesh
  {
    buffer.cmdResetQueryPool(queryPoolTimestamps_, 0, GPUTimestamp_NUM_TIMESTAMPS);

    GPU_TIMESTAMP(GPUTimestamp_BeginSceneRendering);

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen_);
    {
      // Scene
      buffer.cmdBindRenderPipeline(drawNormals_ ? renderPipelineState_MeshNormals_ : renderPipelineState_Mesh_);
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindDepthState(depthState_);
      buffer.cmdBindVertexBuffer(0, vb0_, 0);

      struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t materials;
      } bindings = {
          .perFrame = ctx_->gpuAddress(ubPerFrame_),
          .perObject = ctx_->gpuAddress(ubPerObject_),
          .materials = ctx_->gpuAddress(sbMaterials_),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
      buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      if (enableWireframe_) {
        buffer.cmdBindRenderPipeline(renderPipelineState_MeshWireframe_);
        buffer.cmdDrawIndexed(static_cast<uint32_t>(indexData_.size()));
      }
      buffer.cmdPopDebugGroupLabel();

      // Skybox
      buffer.cmdBindRenderPipeline(renderPipelineState_Skybox_);
      buffer.cmdPushDebugGroupLabel("Render Skybox", 0x00ff00ff);
      buffer.cmdBindDepthState(depthStateLEqual_);
      buffer.cmdDraw(3 * 6 * 2);
      buffer.cmdPopDebugGroupLabel();
    }
    buffer.cmdEndRendering();

    GPU_TIMESTAMP(GPUTimestamp_EndSceneRendering);
  }

  // Pass 3: compute shader post-processing
  {
    GPU_TIMESTAMP(GPUTimestamp_BeginComputePass);

    if (enableComputePass_) {
      lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen_.color[0].resolveTexture : fbOffscreen_.color[0].texture;

      buffer.cmdBindComputePipeline(computePipelineState_Grayscale_);

      struct {
        uint32_t texture;
        uint32_t width;
        uint32_t height;
      } bindings = {
          .texture = tex.index(),
          .width = (uint32_t)width_,
          .height = (uint32_t)height_,
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDispatchThreadGroups(
          {
              .width = 1 + (uint32_t)width_ / 16,
              .height = 1 + (uint32_t)height_ / 16,
              .depth = 1u,
          },
          {
              .textures = {tex},
          });
    }
    GPU_TIMESTAMP(GPUTimestamp_EndComputePass);
  }

  // Pass 4: render into the swapchain image
  {
    GPU_TIMESTAMP(GPUTimestamp_BeginPresent);

    lvk::TextureHandle tex = kNumSamplesMSAA > 1 ? fbOffscreen_.color[0].resolveTexture : fbOffscreen_.color[0].texture;

    // This will clear the framebuffer
    buffer.cmdBeginRendering(renderPassMain_, fbMain_, {.textures = {tex}});
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
      buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
      buffer.cmdBindDepthState(depthState_);
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

    GPU_TIMESTAMP(GPUTimestamp_EndPresent);
  }

  ctx_->submit(buffer, fbMain_.color[0].texture);

  timestampEndRendering = getCurrentTimestamp();

  // timestamp stats
  if (showPerfStats_) {
    ctx_->getQueryPoolResults(queryPoolTimestamps_,
                              0,
                              LVK_ARRAY_NUM_ELEMENTS(pipelineTimestamps),
                              sizeof(pipelineTimestamps),
                              pipelineTimestamps,
                              sizeof(pipelineTimestamps[0]));
  }
}

void generateCompressedTexture(LoadedImage img) {
  LVK_PROFILER_FUNCTION();

  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  printf("...compressing texture to %s\n", img.compressedFileName.c_str());

  const uint32_t mipmapLevelCount = lvk::calcNumMipLevels(img.w, img.h);

  // create a KTX2 texture for RGBA data
  ktxTextureCreateInfo createInfoKTX2 = {
      .glInternalformat = GL_RGBA8,
      .vkFormat = VK_FORMAT_R8G8B8A8_UNORM,
      .baseWidth = img.w,
      .baseHeight = img.h,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipmapLevelCount,
      .numLayers = 1u,
      .numFaces = 1u,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture2* textureKTX2 = nullptr;
  (void)LVK_VERIFY(ktxTexture2_Create(&createInfoKTX2, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKTX2) == KTX_SUCCESS);

  SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(textureKTX2));
  };

  uint32_t w = img.w;
  uint32_t h = img.h;

  // generate custom mip-pyramid
  for (uint32_t i = 0; i != mipmapLevelCount; ++i) {
    size_t offset = 0;
    ktxTexture_GetImageOffset(ktxTexture(textureKTX2), i, 0, 0, &offset);

    stbir_resize_uint8_linear((const unsigned char*)img.pixels,
                              (int)img.w,
                              (int)img.h,
                              0,
                              ktxTexture_GetData(ktxTexture(textureKTX2)) + offset,
                              w,
                              h,
                              0,
                              STBIR_RGBA);

    h = h > 1 ? h >> 1 : 1;
    w = w > 1 ? w >> 1 : 1;
  }

  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  // compress to Basis and transcode to BC7
  ktxBasisParams params = {
      .structSize = sizeof(params),
      .threadCount = 8,
      .compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL,
      .qualityLevel = 255,
  };
  (void)LVK_VERIFY(ktxTexture2_CompressBasisEx(textureKTX2, &params) == KTX_SUCCESS);
  (void)LVK_VERIFY(ktxTexture2_TranscodeBasis(textureKTX2, KTX_TTF_BC7_RGBA, 0) == KTX_SUCCESS);

  // convert to KTX1
  ktxTextureCreateInfo createInfoKTX1 = {
      .glInternalformat = GL_COMPRESSED_RGBA_BPTC_UNORM,
      .vkFormat = VK_FORMAT_BC7_UNORM_BLOCK,
      .baseWidth = img.w,
      .baseHeight = img.h,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipmapLevelCount,
      .numLayers = 1u,
      .numFaces = 1u,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture1* textureKTX1 = nullptr;
  (void)LVK_VERIFY(ktxTexture1_Create(&createInfoKTX1, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKTX1) == KTX_SUCCESS);

  for (uint32_t i = 0; i != mipmapLevelCount; ++i) {
    size_t offset1 = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(textureKTX1), i, 0, 0, &offset1) == KTX_SUCCESS);
    size_t offset2 = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(textureKTX2), i, 0, 0, &offset2) == KTX_SUCCESS);
    memcpy(ktxTexture_GetData(ktxTexture(textureKTX1)) + offset1,
           ktxTexture_GetData(ktxTexture(textureKTX2)) + offset2,
           ktxTexture_GetImageSize(ktxTexture(textureKTX1), i));
  }

  ktxTexture_WriteToNamedFile(ktxTexture(textureKTX1), img.compressedFileName.c_str());
}

LoadedImage loadImage(const char* fileName, int channels) {
  LVK_PROFILER_FUNCTION();

  if (!fileName || !*fileName) {
    return LoadedImage();
  }

  char debugStr[512] = {0};

  snprintf(debugStr, sizeof(debugStr) - 1, "%s (%i)", fileName, channels);

  const std::string debugName(debugStr);

  {
    std::lock_guard lock(imagesCacheMutex_);

    const auto it = imagesCache_.find(debugName);

    if (it != imagesCache_.end()) {
      LVK_ASSERT(channels == it->second.channels);
      return it->second;
    }
  }

  int w, h;
  uint8_t* pixels = stbi_load(fileName, &w, &h, nullptr, channels);

  const LoadedImage img = {
      .w = (uint32_t)w,
      .h = (uint32_t)h,
      .channels = (uint32_t)channels,
      .pixels = pixels,
      .debugName = debugName,
      .compressedFileName = convertFileName(fileName),
  };

  if (img.pixels && kEnableCompression && (channels != 1) && !std::filesystem::exists(img.compressedFileName.c_str())) {
    generateCompressedTexture(img);
  }

  std::lock_guard lock(imagesCacheMutex_);

  imagesCache_[fileName] = img;

  return img;
}

void loadMaterial(size_t i) {
  LVK_PROFILER_FUNCTION();

  static const std::string pathPrefix = folderContentRoot + "src/bistro/Exterior/";

  SCOPE_EXIT {
    remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
  };

#define LOAD_TEX(result, tex, channels)                                                                                                   \
  const LoadedImage result =                                                                                                              \
      std::string(cachedMaterials_[i].tex).empty() ? LoadedImage() : loadImage((pathPrefix + cachedMaterials_[i].tex).c_str(), channels); \
  if (loaderShouldExit_.load(std::memory_order_acquire)) {                                                                                \
    return;                                                                                                                               \
  }

  LOAD_TEX(ambient, ambient_texname, 4);
  LOAD_TEX(diffuse, diffuse_texname, 4);
  LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

  const LoadedMaterial mtl{i, ambient, diffuse, alpha};

  if (!mtl.ambient.pixels && !mtl.diffuse.pixels) {
    // skip missing textures
    materials_[i].texDiffuse = 0;
  } else {
    std::lock_guard guard(loadedMaterialsMutex_);
    loadedMaterials_.push_back(mtl);
    remainingMaterialsToLoad_.fetch_add(1u, std::memory_order_release);
  }
}

void loadMaterials() {
  stbi_set_flip_vertically_on_load(1);

  remainingMaterialsToLoad_ = (uint32_t)cachedMaterials_.size();

  textures_.resize(cachedMaterials_.size());
  for (size_t i = 0; i != cachedMaterials_.size(); i++) {
    loaderPool_->silent_async([i]() { loadMaterial(i); });
  }
}

lvk::Format ktx2iglTextureFormat(ktx_uint32_t format) {
  switch (format) {
  case GL_RGBA32F:
    return lvk::Format_RGBA_F32;
  case GL_RG16F:
    return lvk::Format_RG_F16;
  default:;
  }
  LVK_ASSERT_MSG(false, "Code should NOT be reached");
  return lvk::Format_RGBA_UN8;
}

void loadCubemapTexture(const std::string& fileNameKTX, lvk::Holder<lvk::TextureHandle>& tex) {
  LVK_PROFILER_FUNCTION();

  ktxTexture1* texture = nullptr;
  (void)LVK_VERIFY(ktxTexture1_CreateFromNamedFile(fileNameKTX.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) == KTX_SUCCESS);
  SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(texture));
  };

  if (!LVK_VERIFY(texture->glInternalformat == GL_RGBA32F)) {
    LVK_ASSERT_MSG(false, "Texture format not supported");
    return;
  }

  const uint32_t width = texture->baseWidth;
  const uint32_t height = texture->baseHeight;

  if (tex.empty()) {
    tex = ctx_->createTexture({
        .type = lvk::TextureType_Cube,
        .format = ktx2iglTextureFormat(texture->glInternalformat),
        .dimensions = {width, height},
        .usage = lvk::TextureUsageBits_Sampled,
        .numMipLevels = lvk::calcNumMipLevels(width, height),
        .data = texture->pData,
        // if compression is enabled, upload all mip-levels
        .dataNumMipLevels = kEnableCompression ? lvk::calcNumMipLevels(width, height) : 1u,
        .generateMipmaps = !kEnableCompression,
        .debugName = fileNameKTX.c_str(),
    });
  }
}

ktxTexture1* bitmapToCube(Bitmap& bmp) {
  LVK_ASSERT(bmp.comp_ == 3); // RGB
  LVK_ASSERT(bmp.type_ == eBitmapType_Cube);
  LVK_ASSERT(bmp.fmt_ == eBitmapFormat_Float);

  const int w = bmp.w_;
  const int h = bmp.h_;

  const uint32_t mipLevels = lvk::calcNumMipLevels(w, h);

  ktxTextureCreateInfo createInfo = {
      .glInternalformat = GL_RGBA32F,
      .vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
      .baseWidth = static_cast<uint32_t>(w),
      .baseHeight = static_cast<uint32_t>(h),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipLevels,
      .numLayers = 1u,
      .numFaces = 6u,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture1* texture = nullptr;
  (void)LVK_VERIFY(ktxTexture1_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) == KTX_SUCCESS);

  const int numFacePixels = w * h;

  for (size_t face = 0; face != 6; face++) {
    const vec3* src = reinterpret_cast<vec3*>(bmp.data_.data()) + face * numFacePixels;
    size_t offset = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(texture), 0, 0, face, &offset) == KTX_SUCCESS);
    float* dst = (float*)(texture->pData + offset);
    for (int y = 0; y != h; y++) {
      for (int x = 0; x != w; x++) {
        const vec4 rgba = vec4(src[x + y * w], 1.0f);
        memcpy(dst, &rgba, sizeof(rgba));
        dst += 4;
      }
    }
  }

  return texture;
}

void generateMipmaps(const std::string& outFilename, ktxTexture1* cubemap) {
  LVK_PROFILER_FUNCTION();

  LLOGL("Generating mipmaps");

  LVK_ASSERT(cubemap);

  uint32_t prevWidth = cubemap->baseWidth;
  uint32_t prevHeight = cubemap->baseHeight;

  for (uint32_t face = 0; face != 6; face++) {
    LLOGL(".");
    for (uint32_t miplevel = 1; miplevel < cubemap->numLevels; miplevel++) {
      LLOGL(":");
      const uint32_t width = prevWidth > 1 ? prevWidth >> 1 : 1;
      const uint32_t height = prevHeight > 1 ? prevWidth >> 1 : 1;

      size_t prevOffset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel - 1, 0, face, &prevOffset) == KTX_SUCCESS);
      size_t offset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel, 0, face, &offset) == KTX_SUCCESS);

      stbir_resize_float_linear(reinterpret_cast<const float*>(cubemap->pData + prevOffset),
                                prevWidth,
                                prevHeight,
                                0,
                                reinterpret_cast<float*>(cubemap->pData + offset),
                                width,
                                height,
                                0,
                                STBIR_RGBA);

      prevWidth = width;
      prevHeight = height;
    }
    prevWidth = cubemap->baseWidth;
    prevHeight = cubemap->baseHeight;
  }

  LLOGL("\n");
  ktxTexture_WriteToNamedFile(ktxTexture(cubemap), outFilename.c_str());
}

void processCubemap(const std::string& inFilename, const std::string& outFilenameEnv, const std::string& outFilenameIrr) {
  LVK_PROFILER_FUNCTION();

  int sourceWidth, sourceHeight;
  float* pxs = stbi_loadf(inFilename.c_str(), &sourceWidth, &sourceHeight, nullptr, 3);
  SCOPE_EXIT {
    if (pxs) {
      stbi_image_free(pxs);
    }
  };

  if (!LVK_VERIFY(pxs != nullptr)) {
    LVK_ASSERT_MSG(false, "Did you read the tutorial at the top of Tiny_MeshLarge.cpp?");
    return;
  }

  // Environment map
  {
    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(sourceWidth, sourceHeight, 3, eBitmapFormat_Float, pxs));
    ktxTexture1* cube = bitmapToCube(bmp);
    generateMipmaps(outFilenameEnv, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }

  // Irradiance map
  {
    constexpr int dstW = 256;
    constexpr int dstH = 128;

    std::vector<vec3> out(dstW * dstH);
    convolveDiffuse((vec3*)pxs, sourceWidth, sourceHeight, dstW, dstH, out.data(), 1024);

    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(dstW, dstH, 3, eBitmapFormat_Float, out.data()));
    ktxTexture1* cube = bitmapToCube(bmp);
    generateMipmaps(outFilenameIrr, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }
}

void loadSkyboxTexture() {
  LVK_PROFILER_FUNCTION();

  const std::string skyboxFileName{"immenstadter_horn_2k"};
  const std::string skyboxSubdir{"src/skybox_hdr/"};

  const std::string fileNameRefKTX = folderContentRoot + skyboxFileName + "_ReferenceMap.ktx";
  const std::string fileNameIrrKTX = folderContentRoot + skyboxFileName + "_IrradianceMap.ktx";

  if (!std::filesystem::exists(fileNameRefKTX) || !std::filesystem::exists(fileNameIrrKTX)) {
    const std::string inFilename = folderContentRoot + skyboxSubdir + skyboxFileName + ".hdr";
    LLOGL("Cubemap in KTX format not found. Extracting from HDR file `%s`...\n", inFilename.c_str());

    processCubemap(inFilename, fileNameRefKTX, fileNameIrrKTX);
  }

  loadCubemapTexture(fileNameRefKTX, skyboxTextureReference_);
  loadCubemapTexture(fileNameIrrKTX, skyboxTextureIrradiance_);
}

lvk::Format formatFromChannels(uint32_t channels) {
  if (channels == 1) {
    return lvk::Format_R_UN8;
  }

  if (channels == 4) {
    return kEnableCompression ? lvk::Format_BC7_RGBA : lvk::Format_RGBA_UN8;
  }

  return lvk::Format_Invalid;
}

lvk::TextureHandle createTexture(const LoadedImage& img) {
  if (!img.pixels) {
    return {};
  }

  const auto it = texturesCache_.find(img.debugName);

  if (it != texturesCache_.end()) {
    return it->second;
  }

  const bool hasCompressedTexture = kEnableCompression && img.channels == 4 && std::filesystem::exists(img.compressedFileName.c_str());

  const void* initialData = img.pixels;
  uint32_t initialDataNumMipLevels = 1u;

  ktxTexture* texture = nullptr;

  if (hasCompressedTexture) {
    // uploading the texture
    if (!LVK_VERIFY(ktxTexture_CreateFromNamedFile(img.compressedFileName.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) ==
                    KTX_SUCCESS)) {
      printf("Failed to load %s\n", img.compressedFileName.c_str());
      assert(0);
    }
    initialData = texture->pData;
    initialDataNumMipLevels = lvk::calcNumMipLevels(img.w, img.h);
  }
  SCOPE_EXIT {
    if (texture)
      ktxTexture_Destroy(ktxTexture(texture));
  };

  // No mip-maps come from files on Apple and Android platforms, we need to generate them.
#if defined(__APPLE__) || defined(ANDROID)
  const bool generateMipmaps = true;
#else
  const bool generateMipmaps = !hasCompressedTexture;
#endif

  lvk::Holder<lvk::TextureHandle> tex = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = formatFromChannels(img.channels),
      .dimensions = {img.w, img.h},
      .usage = lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(img.w, img.h),
      .swizzle = (img.channels == 1) ? lvk::ComponentMapping{lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R}
                                     : lvk::ComponentMapping{},
      .data = initialData,
      .dataNumMipLevels = initialDataNumMipLevels,
      .generateMipmaps = generateMipmaps,
      .debugName = img.debugName.c_str(),
  });

  lvk::TextureHandle handle = tex;

  texturesCache_[img.debugName] = std::move(tex);

  return handle;
}

void processLoadedMaterials(lvk::ICommandBuffer& buffer) {
  LoadedMaterial mtl;

  {
    std::lock_guard guard(loadedMaterialsMutex_);
    if (loadedMaterials_.empty()) {
      return;
    } else {
      mtl = loadedMaterials_.back();
      loadedMaterials_.pop_back();
      remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
    }
  }

  {
    MaterialTextures tex;

    tex.ambient = createTexture(mtl.ambient);
    tex.diffuse = createTexture(mtl.diffuse);
    tex.alpha = createTexture(mtl.alpha);

    // update GPU materials
    materials_[mtl.idx].texAmbient = tex.ambient.index();
    materials_[mtl.idx].texDiffuse = tex.diffuse.index();
    materials_[mtl.idx].texAlpha = tex.alpha.index();
    textures_[mtl.idx] = std::move(tex);
  }
  LVK_ASSERT(materials_[mtl.idx].texAmbient >= 0);
  LVK_ASSERT(materials_[mtl.idx].texDiffuse >= 0);
  LVK_ASSERT(materials_[mtl.idx].texAlpha >= 0);
  buffer.cmdUpdateBuffer(sbMaterials_, 0, sizeof(GPUMaterial) * materials_.size(), materials_.data());
}

inline ImVec4 toVec4(const vec4& c) {
  return ImVec4(c.x, c.y, c.z, c.w);
}

void showTimeGPU() {
#if defined(LVK_WITH_IMPLOT)
  const double toMS = ctx_->getTimestampPeriodToMs();
  auto getTimespan = [toMS](GPUTimestamp begin) -> double {
    return double(pipelineTimestamps[begin + 1] - pipelineTimestamps[begin]) * toMS;
  };
  struct sTimeStats {
    enum size { kNumTimelines = 5 };
    struct MinMax {
      float vmin = FLT_MAX;
      float vmax = 0.0f;
    };
    double add(uint32_t pass, const char* name, double value) {
      LVK_ASSERT(pass < kNumTimelines);
      names[pass] = name;
      const float prev = timelines[pass].empty() ? (float)value : timelines[pass].back();
      timelines[pass].push_back(0.9 * prev + 0.1 * value);
      if (timelines[pass].size() > 128)
        timelines[pass].erase(timelines[pass].begin());
      avg[pass] = value;
      return (float)value;
    }
    void updateMinMax() {
      for (uint32_t i = 0; i != kNumTimelines; i++) {
        float minT = FLT_MAX;
        float maxT = 0.0f;
        for (float v : timelines[i]) {
          if (v < minT)
            minT = v;
          if (v > maxT)
            maxT = v;
        }
        minmax[i] = {minT, maxT};
      }
    }
    std::vector<float> timelines[kNumTimelines] = {};
    MinMax minmax[kNumTimelines] = {};
    float avg[kNumTimelines] = {};
    const char* names[kNumTimelines] = {};
    const vec4 colors[kNumTimelines] = {LC_Red, LC_Green, LC_Green, LC_LightBlue, LC_Red};
  };
  static sTimeStats stats;

  const double timeScene = stats.add(1, " Scene", getTimespan(GPUTimestamp_BeginSceneRendering));
  const double timeCompute = stats.add(2, " Compute", getTimespan(GPUTimestamp_BeginComputePass));
  const double timePresent = stats.add(3, " Present", getTimespan(GPUTimestamp_BeginPresent));

  const double timeGPU = timeScene + timeCompute + timePresent;
  stats.add(0, "GPU", timeGPU);
  const double timeCPU = stats.add(4, "CPU", (timestampEndRendering - timestampBeginRendering) * 1000);
  stats.updateMinMax();

  char text[128];
  snprintf(text,
           sizeof(text),
           "GPU: %6.02f ms   (Scene: %.02f   Compute: %.02f   Present: %.02f)",
           timeGPU,
           timeScene,
           timeCompute,
           timePresent);

  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav;
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::SetNextWindowPos({20, height_ * 0.8f}, ImGuiCond_Appearing);
  ImGui::SetNextWindowSize({width_ * 0.4f, 0});
  ImGui::Begin("GPU Stats", nullptr, flags);
  ImGui::Text("%s", text);

  auto Sparkline = [](const char* id, const float* values, int count, float min_v, float max_v, const ImVec4& col, const ImVec2& size) {
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
    ImPlot::SetNextAxesLimits(0, count - 1, min_v, max_v, ImGuiCond_Always);
    if (ImPlot::BeginPlot(id, size, ImPlotFlags_CanvasOnly)) {
      ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
      ImPlot::PushStyleColor(ImPlotCol_Line, col);
      ImPlot::PlotLine(id, values, count, 1, 0);
      ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
      ImPlot::PlotShaded(id, values, count, 0, 1, 0);
      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();
      ImPlot::EndPlot();
    }
    ImPlot::PopStyleVar();
  };

  auto RowLeadIn = [](const char* stage, float value, const ImVec4& color) {
    ImGui::TableSetColumnIndex(0);
    ImGui::TextColored(color, "%s", stage);
    ImGui::TableSetColumnIndex(1);
    ImGui::TextColored(color, "%6.02f", value);
    ImGui::TableSetColumnIndex(2);
  };

  if (ImGui::BeginTable("##table", 3, ImGuiTableFlags_None, ImVec2(-1, 0))) {
    const ImGuiTableColumnFlags flags = ImGuiTableColumnFlags_NoSort;
    ImGui::TableSetupColumn("Stage", flags);
    ImGui::TableSetupColumn("Time (ms)", flags);
    ImGui::TableSetupColumn("Graph", flags | ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    for (uint32_t i = 0; i != sTimeStats::kNumTimelines; i++) {
      ImGui::TableNextRow();
      const ImVec4 color = toVec4(stats.colors[i]);
      RowLeadIn(stats.names[i], stats.avg[i], color);
      if (stats.avg[i] > 0.01) {
        ImGui::PushID(i);
        Sparkline("##spark",
                  stats.timelines[i].data(),
                  stats.timelines[i].size(),
                  stats.minmax[i].vmin * 0.8f,
                  stats.minmax[i].vmax * 1.2f,
                  color,
                  ImVec2(-1, 30));
        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }

  ImGui::End();
#endif // LVK_WITH_IMPLOT
}

#if !defined(ANDROID)
double getCurrentTimestamp() {
  return glfwGetTime();
}

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
  ctx_ = lvk::createVulkanContextWithSwapchain(window,
                                               width_,
                                               height_,
                                               {
                                                   .enableValidation = kEnableValidationLayers,
                                               },
                                               kPreferIntegratedGPU ? lvk::HWDeviceType_Integrated : lvk::HWDeviceType_Discrete);
  if (!ctx_) {
    return EXIT_FAILURE;
  }

  if (kEnableCompression) {
    printf("Compressing textures... It can take a while in debug builds...(needs to be done once)\n");
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
      loaderShouldExit_.store(true, std::memory_order_release);
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_N && pressed) {
      drawNormals_ = !drawNormals_;
    }
    if (key == GLFW_KEY_C && pressed) {
      enableComputePass_ = !enableComputePass_;
    }
    if (key == GLFW_KEY_T && pressed) {
      enableWireframe_ = !enableWireframe_;
    }
    if (key == GLFW_KEY_P && pressed) {
      showPerfStats_ = !showPerfStats_;
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
    if (key == GLFW_KEY_F9 && action == GLFW_PRESS) {
      ktxTextureCreateInfo createInfo = {
          .glInternalformat = GL_RGBA8,
          .vkFormat = VK_FORMAT_B8G8R8A8_UNORM,
          .baseWidth = static_cast<uint32_t>(width_),
          .baseHeight = static_cast<uint32_t>(height_),
          .baseDepth = 1u,
          .numDimensions = 2u,
          .numLevels = 1u,
          .numLayers = 1u,
          .numFaces = 1u,
          .generateMipmaps = KTX_FALSE,
      };

      ktxTexture1* texture = nullptr;
      (void)LVK_VERIFY(ktxTexture1_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) == KTX_SUCCESS);
      ctx_->download(ctx_->getCurrentSwapchainTexture(), {.dimensions = {(uint32_t)width_, (uint32_t)height_}}, texture->pData);
      ktxTexture_WriteToNamedFile(ktxTexture(texture), "screenshot.ktx");
      ktxTexture_Destroy(ktxTexture(texture));
    }
  });

  double prevTime = getCurrentTimestamp();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const double newTime = getCurrentTimestamp();
    const double delta = newTime - prevTime;
    prevTime = newTime;

    if (!width_ || !height_)
      continue;

    fps_.tick(delta);

    render(delta);
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
                                                   },
                                                   kPreferIntegratedGPU ? lvk::HWDeviceType_Integrated : lvk::HWDeviceType_Discrete);
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
      render(delta);
    }
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source) {
        source->process(app, source);
      }
    }
  } while (!app->destroyRequested);
}
} // extern "C"
#endif
