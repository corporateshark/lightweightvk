/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <shared/UtilsFPS.h>

#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <lvk/LVK.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>

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
  uint outTexture;
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

layout(location = 0) rayPayloadInEXT vec3 payload;

layout(push_constant) uniform constants {
  vec2 cam; // just an opaque buffer reference - no access required
  uint outTexture;
  uint tlas;
  float time;
};

vec2 rotate(vec2 v, float angle) {
  mat2 r = mat2(cos(angle), -sin(angle),
                sin(angle),  cos(angle));
  return r * (v-0.5*gl_LaunchSizeEXT.xy);
}

void main() {
  vec2 uv = rotate(gl_LaunchIDEXT.xy, 0.2 * time);
  vec2 pos = floor(uv / 64.0);
  payload = vec3(mod(pos.x + mod(pos.y, 2.0), 2.0));
})";

const char* codeClosestHit = R"(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 attribs;

void main() {
  const vec3 baryCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  payload = baryCoords;
}
)";

int width_ = -80;
int height_ = -80;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;

struct Resources {
  lvk::Holder<lvk::AccelStructHandle> BLAS;
  lvk::Holder<lvk::AccelStructHandle> TLAS;

  lvk::Holder<lvk::BufferHandle> vertexBuffer;
  lvk::Holder<lvk::BufferHandle> indexBuffer;
  lvk::Holder<lvk::BufferHandle> instancesBuffer;

  lvk::Holder<lvk::TextureHandle> storageImage;

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
      .storage = lvk::StorageType_HostVisible,
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

void init() {
  createBottomLevelAccelerationStructure();
  createTopLevelAccelerationStructure();

  struct UniformData {
    glm::mat4 viewInverse = glm::inverse(glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, -5.0f)));
    glm::mat4 projInverse = glm::inverse(glm::perspective(glm::radians(60.0f), (float)width_ / (float)height_, 0.1f, 1000.0f));
  } uniformData;

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
          .dimensions = {(uint32_t)width_, (uint32_t)height_, 1u},
          .numLayers = 1,
          .numSamples = 1,
          .usage = lvk::TextureUsageBits_Storage,
      },
      "storageImage");

  res.raygen_ = ctx_->createShaderModule({codeRayGen, lvk::Stage_RayGen, "Shader Module: main (raygen)"});
  res.miss_ = ctx_->createShaderModule({codeMiss, lvk::Stage_Miss, "Shader Module: main (miss)"});
  res.hit_ = ctx_->createShaderModule({codeClosestHit, lvk::Stage_ClosestHit, "Shader Module: main (closesthit)"});

  res.pipeline = ctx_->createRayTracingPipeline(lvk::RayTracingPipelineDesc{
      .smRayGen = {res.raygen_},
      .smClosestHit = {res.hit_},
      .smMiss = {res.miss_},
  });
}

void destroy() {
  res = {};
  ctx_ = nullptr;
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
}

double glfwGetTime();

void render() {
  if (!width_ || !height_) {
    return;
  }

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  const glm::mat3x4 transformMatrix = glm::rotate(glm::mat4(1.0f), (float)glfwGetTime(), glm::vec3(1, 1, 1));
  ctx_->upload(res.instancesBuffer, &transformMatrix, sizeof(transformMatrix), offsetof(lvk::AccelStructInstance, transform));

  struct {
    uint64_t camBuffer;
    uint32_t outTexture;
    uint32_t tlas;
    float time;
  } pc = {
      .camBuffer = ctx_->gpuAddress(res.ubo),
      .outTexture = res.storageImage.index(),
      .tlas = res.TLAS.index(),
      .time = (float)glfwGetTime(),
  };

  buffer.cmdUpdateTLAS(res.TLAS, res.instancesBuffer);
  buffer.cmdBindRayTracingPipeline(res.pipeline);
  buffer.cmdPushConstants(pc);
  buffer.cmdTraceRays((uint32_t)width_, (uint32_t)height_, 1, {.textures = {lvk::TextureHandle(res.storageImage)}});
  buffer.cmdCopyImage(res.storageImage, ctx_->getCurrentSwapchainTexture(), ctx_->getDimensions(ctx_->getCurrentSwapchainTexture()));

  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan Hello Ray Tracing", width_, height_, true);

  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {
#if defined(NDEBUG)
    .enableValidation = false,
#else
    .enableValidation = true,
#endif
  });
  if (!ctx_) {
    return 255;
  }
  init();

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    resize();
  });

  double prevTime = glfwGetTime();

  // main loop
  while (!glfwWindowShouldClose(window)) {
    const double newTime = glfwGetTime();
    fps_.tick(newTime - prevTime);
    prevTime = newTime;
    render();
    glfwPollEvents();
  }

  // destroy all the Vulkan stuff before closing the window
  destroy();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
#else
double glfwGetTime() {
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
                                                     .enableValidation = false,
                                                     .enableAccelerationStructure = true,
                                                     .enableRayTracingPipeline = true,
                                                   });
      init();
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

  fps_.printFPS_ = false;

  double prevTime = glfwGetTime();

  int events = 0;
  android_poll_source* source = nullptr;
  do {
    double newTime = glfwGetTime();
    double delta = newTime - prevTime;
    if (fps_.tick(delta)) {
      LLOGL("FPS: %.1f\n", fps_.getFPS());
    }
    prevTime = newTime;
    if (ctx_) {
      render();
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
