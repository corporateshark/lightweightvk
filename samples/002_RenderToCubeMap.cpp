/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <cassert>
#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <lvk/LVK.h>

#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <shared/UtilsFPS.h>

const char* codeTriangleVS = R"(
#version 460
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
	vec2(-0.6, -0.6),
	vec2( 0.6, -0.6),
	vec2( 0.0,  0.6)
);
const vec3 col[6] = vec3[6](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0),
   vec3(1.0, 0.0, 1.0),
	vec3(1.0, 1.0, 0.0),
	vec3(0.0, 1.0, 1.0)
);
layout(push_constant) uniform constants {
  uint face;
  float time;
} pc;

void main() {
	gl_Position = vec4(pos[gl_VertexIndex] * (1.5 + sin(pc.time)) * 0.5, 0.0, 1.0);
	color = col[pc.face];
}
)";

const char* codeTriangleFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;

void main() {
	out_FragColor = vec4(color, 1.0);
};
)";

const char* codeVS = R"(
layout (location=0) out vec3 dir;
layout (location=1) out flat uint textureId;

const vec3 vertices[8] = vec3[8](
	vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
	vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

layout(push_constant) uniform constants {
  mat4 mvp;
  uint texture0;
} pc;

void main() {
  vec3 v = vertices[gl_VertexIndex];
  gl_Position = pc.mvp * vec4(v, 1.0);
  dir = v;
  textureId = pc.texture0;
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 dir;
layout (location=1) in flat uint textureId;
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = textureBindlessCube(textureId, 0, normalize(dir));
};
)";

using glm::mat4;
using glm::vec3;
using glm::vec4;

int width_ = 0;
int height_ = 0;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;
lvk::Holder<lvk::ShaderModuleHandle> vert_;
lvk::Holder<lvk::ShaderModuleHandle> frag_;
lvk::Holder<lvk::ShaderModuleHandle> vertTriangle_;
lvk::Holder<lvk::ShaderModuleHandle> fragTriangle_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
lvk::Holder<lvk::BufferHandle> ib0_;
lvk::Holder<lvk::TextureHandle> texture_;

static uint16_t indexData[36] = {0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
                                 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3};

void init() {
  ib0_ = ctx_->createBuffer({
      .usage = lvk::BufferUsageBits_Index,
      .storage = lvk::StorageType_Device,
      .size = sizeof(indexData),
      .data = indexData,
      .debugName = "Buffer: index",
  });

  texture_ = ctx_->createTexture({
      .type = lvk::TextureType_Cube,
      .format = lvk::Format_BGRA_UN8,
      .dimensions = {512, 512},
      .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment,
      .debugName = "CubeMap",
  });

  vert_ = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  frag_ = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  vertTriangle_ = ctx_->createShaderModule({codeTriangleVS, lvk::Stage_Vert, "Shader Module: triangle (vert)"});
  fragTriangle_ = ctx_->createShaderModule({codeTriangleFS, lvk::Stage_Frag, "Shader Module: triangle (frag)"});

  renderPipelineState_Mesh_ = ctx_->createRenderPipeline({
      .smVert = vert_,
      .smFrag = frag_,
      .color = {{.format = ctx_->getSwapchainFormat()}},
      .cullMode = lvk::CullMode_Back,
      .frontFace = lvk::WindingMode_CW,
      .debugName = "Pipeline: mesh",
  });
  renderPipelineState_Triangle_ = ctx_->createRenderPipeline({
      .smVert = vertTriangle_,
      .smFrag = fragTriangle_,
      .color = {{.format = ctx_->getFormat(texture_)}},
      .debugName = "Pipeline: triangle",
  });
}

void destroy() {
  ib0_ = nullptr;
  vert_ = nullptr;
  frag_ = nullptr;
  vertTriangle_ = nullptr;
  fragTriangle_ = nullptr;
  renderPipelineState_Mesh_ = nullptr;
  renderPipelineState_Triangle_ = nullptr;
  texture_ = nullptr;
  ctx_ = nullptr;
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
}

void render(float time) {
  LVK_PROFILER_FUNCTION();

  if (!width_ || !height_) {
    return;
  }

  const float fov = float(45.0f * (M_PI / 180.0f));
  const float aspectRatio = (float)width_ / (float)height_;
  const mat4 proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 500.0f);
  const mat4 view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 5.0f));
  const mat4 model = glm::rotate(mat4(1.0f), time, glm::normalize(vec3(1.0f, 1.0f, 1.0f)));

  // Command buffers (1-N per thread): create, submit and forget
  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  buffer.cmdPushDebugGroupLabel("Render to Cube Map", 0xff0000ff);
  for (uint8_t face = 0; face != 6; face++) {
    buffer.cmdBeginRendering({.color = {{
                                  .loadOp = lvk::LoadOp_Clear,
                                  .storeOp = lvk::StoreOp_Store,
                                  .layer = face,
                                  .clearColor = {0.3f, 0.3f, 0.3f, 1.0f},
                              }}},
                             {.color = {{.texture = texture_}}});
    buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
    const struct {
      uint32_t face;
      float time;
    } pc = {
        .face = face,
        .time = 10.0f * time,
    };
    buffer.cmdPushConstants(pc);
    buffer.cmdDraw(3);
    buffer.cmdEndRendering();
  }
  buffer.cmdPopDebugGroupLabel();

  buffer.cmdBeginRendering({.color = {{
                                .loadOp = lvk::LoadOp_Clear,
                                .storeOp = lvk::StoreOp_Store,
                                .clearColor = {1.0f, 1.0f, 1.0f, 1.0f},
                            }}},
                           {.color = {{.texture = ctx_->getCurrentSwapchainTexture()}}},
                           {.textures = {lvk::TextureHandle(texture_)}});
  {
    buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
    buffer.cmdBindViewport({0.0f, 0.0f, (float)width_, (float)height_, 0.0f, +1.0f});
    buffer.cmdBindScissorRect({0, 0, (uint32_t)width_, (uint32_t)height_});
    buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
    buffer.cmdBindDepthState({});
    buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
    struct {
      mat4 mvp;
      uint32_t texture;
    } bindings = {
        .mvp = proj * view * model,
        .texture = texture_.index(),
    };
    buffer.cmdPushConstants(bindings);
    buffer.cmdDrawIndexed(3 * 6 * 2);
    buffer.cmdPopDebugGroupLabel();
  }
  buffer.cmdEndRendering();

  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan Mesh", width_, height_, true);

  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {});
  if (!ctx_) {
    return 1;
  }

  init();

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    resize();
  });

  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
  });

  double prevTime = glfwGetTime();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    const double newTime = glfwGetTime();
    fps_.tick(newTime - prevTime);
    prevTime = newTime;
    render((float)newTime);
    glfwPollEvents();
  }

  // destroy all the Vulkan stuff before closing the window
  destroy();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
#else
extern "C" {
void handle_cmd(android_app* app, int32_t cmd) {
  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (app->window != nullptr) {
      width_ = ANativeWindow_getWidth(app->window);
      height_ = ANativeWindow_getHeight(app->window);
      ctx_ = lvk::createVulkanContextWithSwapchain(app->window, width_, height_, {});
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

  timespec prevTime = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &prevTime);

  int events = 0;
  android_poll_source* source = nullptr;
  do {
    timespec newTime = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &newTime);
    double newTimeSec = ((double)newTime.tv_sec + 1.0e-9 * newTime.tv_nsec);
    if (fps_.tick(newTimeSec - ((double)prevTime.tv_sec + 1.0e-9 * prevTime.tv_nsec))) {
      LLOGL("FPS: %.1f\n", fps_.getFPS());
    }
    prevTime = newTime;
    if (ctx_) {
      render((float)newTimeSec);
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