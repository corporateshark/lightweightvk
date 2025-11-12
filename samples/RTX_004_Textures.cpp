/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <filesystem>

#include <ldrutils/lutils/ScopeExit.h>
#include <stb/stb_image.h>

// we are going to use raw Vulkan here to initialize VK_KHR_ray_tracing_position_fetch
#include <lvk/vulkan/VulkanUtils.h>

const char* codeSlang = R"(
[[vk::binding(2, 0)]] RWTexture2D<float4> kTextures2DInOut[];

struct Camera {
  float4x4 viewInverse;
  float4x4 projInverse;
};

struct PushConstants {
  Camera* cam;
  float2 dim;
  uint outTexture;
  uint texBackground;
  uint texObject;
  uint tlas;
  float time;
};
[[vk::push_constant]] PushConstants pc;

struct RayPayload {
  float3 color;
};

[shader("raygeneration")]
void rayGenMain() {
  uint3 launchID = DispatchRaysIndex();
  uint3 launchSize = DispatchRaysDimensions();

  float2 pixelCenter = float2(launchID.xy) + float2(0.5, 0.5);
  float2 d = 2.0 * (pixelCenter / float2(launchSize.xy)) - 1.0;

  float4 origin    = pc.cam->viewInverse * float4(0, 0, 0, 1);
  float4 target    = pc.cam->projInverse * float4(d, 1, 1);
  float4 direction = pc.cam->viewInverse * float4(normalize(target.xyz), 0);

  RayDesc ray;
  ray.Origin = origin.xyz;
  ray.Direction = direction.xyz;
  ray.TMin = 0.1;
  ray.TMax = 500.0;

  RayPayload payload = { float3(0.0, 0.0, 0.0) };

  TraceRay(
    kTLAS[NonUniformResourceIndex(pc.tlas)],
    RAY_FLAG_FORCE_OPAQUE,
    0xff,
    0,
    0,
    0,
    ray,
    payload
  );

  kTextures2DInOut[NonUniformResourceIndex(pc.outTexture)][launchID.xy] = float4(payload.color, 1.0);
}

[shader("miss")]
void missMain(inout RayPayload payload) {
  float2 uv = float2(DispatchRaysIndex().xy) / pc.dim;
  payload.color = textureBindless2DLod(pc.texBackground, 0, uv, 0).rgb;
}

float4 triplanar(uint tex, float3 worldPos, float3 normal) {
  // generate weights, show texture on both sides of the object (positive and negative)
  float3 weights = abs(normal);
  // make the transition sharper
  weights = pow(weights, float3(8.0, 8.0, 8.0));
  // make sure the sum of all components is 1
  weights = weights / (weights.x + weights.y + weights.z);

  // sample the texture for 3 different projections
  float4 cXY = textureBindless2DLod(tex, 0, worldPos.xy, 0);
  float4 cZY = textureBindless2DLod(tex, 0, worldPos.zy, 0);
  float4 cXZ = textureBindless2DLod(tex, 0, worldPos.xz, 0);

  // combine the projected colors
  return cXY * weights.z + cZY * weights.x + cXZ * weights.y;
}

[shader("closesthit")]
void closestHitMain(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
  float3 pos0 = HitTriangleVertexPosition(0);
  float3 pos1 = HitTriangleVertexPosition(1);
  float3 pos2 = HitTriangleVertexPosition(2);

  float3 baryCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                             attribs.barycentrics.x,
                             attribs.barycentrics.y);
  float3 pos = pos0 * baryCoords.x + pos1 * baryCoords.y + pos2 * baryCoords.z;

  // triplanar mapping in object-space; for our icosahedron, object-space position and normal vectors are the same
  payload.color = triplanar(pc.texObject, pos, normalize(pos)).rgb;
}
)";

const char* codeRayGen = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 2, rgba8) uniform image2D kTextures2DInOut[];
layout (set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];

layout(std430, buffer_reference) readonly buffer Camera {
  mat4 viewInverse;
  mat4 projInverse;
};

layout(push_constant) uniform constants {
  Camera cam;
  vec2 dim;
  uint outTexture;
  uint texBackground;
  uint texObject;
  uint tlas;
  float time;
};

layout(location = 0) rayPayloadEXT vec3 payload;

const float tmin = 0.1;
const float tmax = 500.0;

void main() {
  vec2 pixelCenter = gl_LaunchIDEXT.xy + vec2(0.5);
  vec2 d = 2.0 * (pixelCenter / gl_LaunchSizeEXT.xy) - 1.0;

  vec4 origin = cam.viewInverse * vec4(0,0,0,1);
  vec4 target = cam.projInverse * vec4(d, 1, 1);
  vec4 direction = cam.viewInverse * vec4(normalize(target.xyz), 0);

  payload = vec3(0.0);

  traceRayEXT(kTLAS[tlas], gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

  imageStore(kTextures2DInOut[outTexture], ivec2(gl_LaunchIDEXT.xy), vec4(payload, 1.0));
}
)";

const char* codeMiss = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler kSamplers[];

vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {
  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv);
}

layout(location = 0) rayPayloadInEXT vec3 payload;

layout(push_constant) uniform constants {
  vec2 cam; // just an opaque buffer reference - no access required
  vec2 dim;
  uint outTexture;
  uint texBackground;
  uint texObject;
  uint tlas;
  float time;
};

void main() {
  vec2 uv = gl_LaunchIDEXT.xy / dim;
  payload = textureBindless2D(texBackground, 0, uv).rgb;
})";

const char* codeClosestHit = R"(
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_ray_tracing_position_fetch : require

layout (set = 0, binding = 0) uniform texture2D kTextures2D[];
layout (set = 0, binding = 1) uniform sampler kSamplers[];

vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {
  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv);
}

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

layout(push_constant) uniform constants {
  vec2 cam; // just an opaque buffer reference - no access required
  vec2 dim;
  uint outTexture;
  uint texBackground;
  uint texObject;
  uint tlas;
  float time;
};

vec4 triplanar(uint tex, vec3 worldPos, vec3 normal) {
  // generate weights, show texture on both sides of the object (positive and negative)
  vec3 weights = abs(normal);
  // make the transition sharper
  weights = pow(weights, vec3(8.0));
  // make sure the sum of all components is 1
  weights = weights / (weights.x + weights.y + weights.z);

  // sample the texture for 3 different projections
  vec4 cXY = textureBindless2D(tex, 0, worldPos.xy);
  vec4 cZY = textureBindless2D(tex, 0, worldPos.zy);
  vec4 cXZ = textureBindless2D(tex, 0, worldPos.xz);

  // combine the projected colors
  return cXY * weights.z + cZY * weights.x + cXZ * weights.y;
}

void main() {
  vec3 pos0 = gl_HitTriangleVertexPositionsEXT[0];
  vec3 pos1 = gl_HitTriangleVertexPositionsEXT[1];
  vec3 pos2 = gl_HitTriangleVertexPositionsEXT[2];

  vec3 baryCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  vec3 pos = pos0 * baryCoords.x + pos1 * baryCoords.y + pos2 * baryCoords.z;

  // triplanar mapping in object-space; for our icosahedron, object-space position and normal vectors are the same
  payload = triplanar(texObject, pos, normalize(pos)).rgb;
}
)";

lvk::IContext* ctx_ = nullptr;

struct Resources {
  lvk::Holder<lvk::AccelStructHandle> BLAS;
  lvk::Holder<lvk::AccelStructHandle> TLAS;

  lvk::Holder<lvk::BufferHandle> vertexBuffer;
  lvk::Holder<lvk::BufferHandle> indexBuffer;
  lvk::Holder<lvk::BufferHandle> instancesBuffer;

  lvk::Holder<lvk::TextureHandle> storageImage;
  lvk::Holder<lvk::TextureHandle> texBackground;
  lvk::Holder<lvk::TextureHandle> texObject;

  lvk::Holder<lvk::ShaderModuleHandle> raygen_;
  lvk::Holder<lvk::ShaderModuleHandle> miss_;
  lvk::Holder<lvk::ShaderModuleHandle> hit_;

  lvk::Holder<lvk::BufferHandle> ubo;

  lvk::Holder<lvk::RayTracingPipelineHandle> pipeline;
} res;

void createBottomLevelAccelerationStructure() {
  struct Vertex {
    float pos[3];
  };
  const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
  const Vertex vertices[] = {
      {-1, t, 0},
      {1, t, 0},
      {-1, -t, 0},
      {1, -t, 0},

      {0, -1, t},
      {0, 1, t},
      {0, -1, -t},
      {0, 1, -t},

      {t, 0, -1},
      {t, 0, 1},
      {-t, 0, -1},
      {-t, 0, 1},
  };

  const uint32_t indices[] = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
                              3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};

  const glm::mat3x4 transformMatrix(1.0f);

  res.vertexBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(vertices),
      .data = vertices,
  });
  res.indexBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(indices),
      .data = indices,
  });
  lvk::Holder<lvk::BufferHandle> transformBuffer = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(glm::mat3x4),
      .data = &transformMatrix,
  });

  res.BLAS = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_BLAS,
      .geometryType = lvk::AccelStructGeomType_Triangles,
      .vertexFormat = lvk::VertexFormat::Float3,
      .vertexBuffer = res.vertexBuffer,
      .numVertices = (uint32_t)LVK_ARRAY_NUM_ELEMENTS(vertices),
      .indexFormat = lvk::IndexFormat_UI32,
      .indexBuffer = res.indexBuffer,
      .transformBuffer = transformBuffer,
      .buildRange = {.primitiveCount = (uint32_t)LVK_ARRAY_NUM_ELEMENTS(indices) / 3},
      .debugName = "BLAS",
  });
}

void createTopLevelAccelerationStructure() {
  const lvk::AccelStructInstance instance{
      // clang-format off
      .transform = {.matrix = {1.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 1.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 1.0f, 0.0f}},
      // clang-format on
      .instanceCustomIndex = 0,
      .mask = 0xff,
      .instanceShaderBindingTableRecordOffset = 0,
      .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
      .accelerationStructureReference = ctx_->gpuAddress(res.BLAS),
  };

  // Buffer for instance data
  res.instancesBuffer = ctx_->createBuffer(lvk::BufferDesc{
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_Device,
      .size = sizeof(lvk::AccelStructInstance),
      .data = &instance,
      .debugName = "instanceBuffer",
  });

  res.TLAS = ctx_->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = res.instancesBuffer,
      .buildRange = {.primitiveCount = 1},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace | lvk::AccelStructBuildFlagBits_AllowUpdate,
  });
}

lvk::Holder<lvk::TextureHandle> createTextureFromFile(VulkanApp& app, const char* fileName) {
  using namespace std::filesystem;
  path dir = app.folderContentRoot_;
  int32_t texWidth = 0;
  int32_t texHeight = 0;
  int32_t channels = 0;
  uint8_t* pixels = stbi_load((dir / path(fileName)).string().c_str(), &texWidth, &texHeight, &channels, 4);
  SCOPE_EXIT {
    stbi_image_free(pixels);
  };
  if (!pixels) {
    LVK_ASSERT_MSG(false, "Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
    LLOGW("Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
    std::terminate();
  }
  return ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_RGBA_UN8,
      .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = pixels,
      .debugName = fileName,
  });
}

VULKAN_APP_MAIN {
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR positionFetchFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
      .rayTracingPositionFetch = VK_TRUE,
  };
  const VulkanAppConfig cfg{
      .width = -80,
      .height = -80,
      .resizable = true,
      .contextConfig =
          {
              .extensionsDevice = {"VK_KHR_ray_tracing_position_fetch"},
              .extensionsDeviceFeatures = &positionFetchFeatures,
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  ctx_ = app.ctx_.get();

  createBottomLevelAccelerationStructure();
  createTopLevelAccelerationStructure();

  const struct UniformData {
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
  } uniformData = {
      .viewInverse = glm::inverse(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -5.0f))),
      .projInverse = glm::inverse(glm::perspective(glm::radians(60.0f), (float)app.width_ / (float)app.height_, 0.1f, 1000.0f)),
  };

  res.ubo = ctx_->createBuffer(lvk::BufferDesc{
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(uniformData),
      .data = &uniformData,
      .debugName = "cameraBuffer",
  });

  res.storageImage = ctx_->createTexture(
      lvk::TextureDesc{
          .type = lvk::TextureType_2D,
          .format = lvk::Format_BGRA_UN8,
          .dimensions = {(uint32_t)app.width_, (uint32_t)app.height_, 1u},
          .numLayers = 1,
          .numSamples = 1,
          .usage = lvk::TextureUsageBits_Storage,
      },
      "storageImage");

  res.texBackground = createTextureFromFile(app, "src/bistro/BuildingTextures/wood_polished_01_diff.png");
  res.texObject = createTextureFromFile(app, "src/bistro/BuildingTextures/Cobble_02B_Diff.png");

#if defined(LVK_DEMO_WITH_SLANG)
  res.raygen_ = ctx_->createShaderModule({codeSlang, lvk::Stage_RayGen, "Shader Module: main (raygen)"});
  res.miss_ = ctx_->createShaderModule({codeSlang, lvk::Stage_Miss, "Shader Module: main (miss)"});
  res.hit_ = ctx_->createShaderModule({codeSlang, lvk::Stage_ClosestHit, "Shader Module: main (closesthit)"});
#else
  res.raygen_ = ctx_->createShaderModule({codeRayGen, lvk::Stage_RayGen, "Shader Module: main (raygen)"});
  res.miss_ = ctx_->createShaderModule({codeMiss, lvk::Stage_Miss, "Shader Module: main (miss)"});
  res.hit_ = ctx_->createShaderModule({codeClosestHit, lvk::Stage_ClosestHit, "Shader Module: main (closesthit)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  res.pipeline = ctx_->createRayTracingPipeline(lvk::RayTracingPipelineDesc{
      .smRayGen = {lvk::ShaderModuleHandle(res.raygen_)},
      .smClosestHit = {lvk::ShaderModuleHandle(res.hit_)},
      .smMiss = {lvk::ShaderModuleHandle(res.miss_)},
  });

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

    const glm::mat3x4 transformMatrix = glm::rotate(glm::mat4(1.0f), (float)glfwGetTime(), glm::vec3(1, 1, 1));

    buffer.cmdUpdateBuffer(res.instancesBuffer, offsetof(lvk::AccelStructInstance, transform), sizeof(transformMatrix), &transformMatrix);

    struct {
      uint64_t camBuffer;
      vec2 dim;
      uint32_t outTexture;
      uint32_t texBackground;
      uint32_t texObject;
      uint32_t tlas;
      float time;
    } pc = {
        .camBuffer = ctx_->gpuAddress(res.ubo),
        .dim = vec2(width, height),
        .outTexture = res.storageImage.index(),
        .texBackground = res.texBackground.index(),
        .texObject = res.texObject.index(),
        .tlas = res.TLAS.index(),
        .time = (float)glfwGetTime(),
    };

    buffer.cmdUpdateTLAS(res.TLAS, res.instancesBuffer);
    buffer.cmdBindRayTracingPipeline(res.pipeline);
    buffer.cmdPushConstants(pc);
    buffer.cmdTraceRays(width, height, 1, {.textures = {lvk::TextureHandle(res.storageImage)}});
    buffer.cmdCopyImage(res.storageImage, ctx_->getCurrentSwapchainTexture(), ctx_->getDimensions(ctx_->getCurrentSwapchainTexture()));
    lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
    };
    buffer.cmdBeginRendering(lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Load, .storeOp = lvk::StoreOp_Store}}}, framebuffer);
    app.imgui_->beginFrame(framebuffer);
    app.drawFPS();
    app.imgui_->endFrame(buffer);
    buffer.cmdEndRendering();
    ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
  });

  // destroy all the Vulkan stuff before closing the window
  res = {};

  VULKAN_APP_EXIT();
}
