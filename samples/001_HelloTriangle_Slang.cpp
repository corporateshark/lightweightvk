/*
* LightweightVK
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
layout (location=0) out vec4 out_FragColor;

void main() {
	out_FragColor = vec4(color, 1.0);
};
)";

int width_ = 800;
int height_ = 600;
FramesPerSecondCounter fps_;

lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
std::unique_ptr<lvk::IContext> ctx_;
lvk::Holder<lvk::ShaderModuleHandle> vert_;
lvk::Holder<lvk::ShaderModuleHandle> frag_;

void init() {
  vert_ = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  frag_ = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});

  renderPipelineState_Triangle_ = ctx_->createRenderPipeline(
      {
          .smVert = vert_,
          .smFrag = frag_,
          .color = {{.format = ctx_->getSwapchainFormat()}},
      },
      nullptr);

  LVK_ASSERT(renderPipelineState_Triangle_.valid());
}

void destroy() {
  vert_ = nullptr;
  frag_ = nullptr;
  renderPipelineState_Triangle_ = nullptr;
  ctx_ = nullptr;
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
}

void render() {
  if (!width_ || !height_) {
    return;
  }

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  // This will clear the framebuffer
  buffer.cmdBeginRendering(
      {.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}}},
      {.color = {{.texture = ctx_->getCurrentSwapchainTexture()}}});
  buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
  buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
  buffer.cmdDraw(3);
  buffer.cmdPopDebugGroupLabel();
  buffer.cmdEndRendering();
  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan Hello Triangle", width_, height_, true);

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
    fps_.tick(((double)newTime.tv_sec + 1.0e-9 * newTime.tv_nsec) - 
              ((double)prevTime.tv_sec + 1.0e-9 * prevTime.tv_nsec));
    LLOGL("FPS: %.1f\n", fps_.getFPS());
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
