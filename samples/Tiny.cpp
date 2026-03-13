/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <shared/UtilsFPS.h>

#include <lvk/LVK.h>
#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

const char* codeVS = R"(
#version 460
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
	vec2(-0.6, -0.4),
	vec2( 0.6, -0.4),
	vec2( 0.0,  0.6)
);
const vec3 col[3] = vec3[3](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0)
);
void main() {
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	color = col[gl_VertexIndex];
}
)";

const char* codeFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor0;
layout (location=1) out vec4 out_FragColor1;

void main() {
	out_FragColor0 = vec4(color, 1.0);
	out_FragColor1 = vec4(1.0, 1.0, 0.0, 1.0);
};
)";

int width_ = 800;
int height_ = 600;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;

struct VulkanObjects {
  void init();
  void destroy();
  void resize();
  void createFramebuffer();
  void render();
  lvk::Framebuffer fb_ = {};
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
  lvk::Holder<lvk::ShaderModuleHandle> vert_;
  lvk::Holder<lvk::ShaderModuleHandle> frag_;
} vk;

void VulkanObjects::init() {
  createFramebuffer();

  vert_ = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  frag_ = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});

  renderPipelineState_Triangle_ = ctx_->createRenderPipeline(
      {
          .smVert = vert_,
          .smFrag = frag_,
          .color = {{ctx_->getFormat(fb_.color[0].texture)},
                    {ctx_->getFormat(fb_.color[1].texture)},
                    {ctx_->getFormat(fb_.color[2].texture)},
                    {ctx_->getFormat(fb_.color[3].texture)}},
      },
      nullptr);
  LVK_ASSERT(renderPipelineState_Triangle_.valid());
}

void VulkanObjects::destroy() {
  ctx_->destroy(vk.fb_.color[1].texture);
  ctx_->destroy(vk.fb_.color[2].texture);
  ctx_->destroy(vk.fb_.color[3].texture);
  vk = {};

  ctx_ = nullptr;
}

void VulkanObjects::resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
  createFramebuffer();
}

void VulkanObjects::createFramebuffer() {
  lvk::TextureHandle texSwapchain = ctx_->getCurrentSwapchainTexture();

  {
    const lvk::TextureDesc desc = {
        .type = lvk::TextureType_2D,
        .format = ctx_->getFormat(texSwapchain),
        .dimensions = ctx_->getDimensions(texSwapchain),
        .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
    };

    fb_ = {.color = {{.texture = texSwapchain},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C1").release()},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C2").release()},
                     {.texture = ctx_->createTexture(desc, "Framebuffer C3").release()}}};
  }
}

void VulkanObjects::render() {
  if (!width_ || !height_) {
    return;
  }

  fb_.color[0].texture = ctx_->getCurrentSwapchainTexture();

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  // This will clear the framebuffer
  buffer.cmdBeginRendering(
      {.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}},
                 {.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 0.0f, 0.0f, 1.0f}},
                 {.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 1.0f, 0.0f, 1.0f}},
                 {.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 0.0f, 1.0f, 1.0f}}}},
      fb_);
  {
    buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
    buffer.cmdBindViewport({0.0f, 0.0f, (float)width_, (float)height_, 0.0f, +1.0f});
    buffer.cmdBindScissorRect({0, 0, (uint32_t)width_, (uint32_t)height_});
    buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
    buffer.cmdDraw(3);
    buffer.cmdPopDebugGroupLabel();
  }
  buffer.cmdEndRendering();
  ctx_->submit(buffer, fb_.color[0].texture);
}

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
#if defined(LVK_WITH_MINILOG)
  minilog::initialize(nullptr, {.threadNames = false});
#endif // LVK_WITH_MINILOG

  GLFWwindow* window = lvk::initWindow("Vulkan Triangle", width_, height_, true);
  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {});
  if (!ctx_) {
    return 1;
  }
  vk.init();

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    vk.resize();
  });

  double prevTime = glfwGetTime();

  // main loop
  while (!glfwWindowShouldClose(window)) {
    const double newTime = glfwGetTime();
    fps_.tick(newTime - prevTime);
    prevTime = newTime;
    vk.render();
    glfwPollEvents();
  }

  // destroy all the Vulkan stuff before closing the window
  vk.destroy();

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
      vk.init();
    }
    break;
  case APP_CMD_TERM_WINDOW:
    vk.destroy();
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
      vk.resize();
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
    if (fps_.tick(((double)newTime.tv_sec + 1.0e-9 * newTime.tv_nsec) - 
                  ((double)prevTime.tv_sec + 1.0e-9 * prevTime.tv_nsec))) {
      LLOGL("FPS: %.1f\n", fps_.getFPS());
    }
    prevTime = newTime;
    if (ctx_) {
      vk.render();
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
