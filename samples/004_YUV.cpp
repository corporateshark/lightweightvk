/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <shared/UtilsFPS.h>

#include <lvk/LVK.h>
#include <lvk/HelpersImGui.h>
#include <ldrutils/lutils/ScopeExit.h>
#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <cstddef>
#include <filesystem>
#include <vector>

std::filesystem::path getPathToContentFolder();

const char* codeVS = R"(
#version 460
layout (location=0) out vec2 uv;
const vec2 pos[4] = vec2[4](
	vec2(-1.0, -1.0),
   vec2(-1.0, +1.0),
	vec2(+1.0, -1.0),
	vec2(+1.0, +1.0)
);
void main() {
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	uv = pos[gl_VertexIndex] * 0.5 + 0.5;
   uv.y = 1.0-uv.y;
}
)";

const char* codeFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout (constant_id = 0) const uint textureId = 0;

void main() {
  out_FragColor = texture(kSamplerYUV[textureId], uv);
};
)";

int width_ = 0;
int height_ = 0;
FramesPerSecondCounter fps_;
size_t currentDemo_ = 0;

std::unique_ptr<lvk::IContext> ctx_;

// demonstrate different YUV formats
struct YUVFormatDemo {
  const char* name;
  lvk::Format format;
  lvk::Holder<lvk::TextureHandle> texture;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState;
};

struct Resources {
  std::unique_ptr<lvk::ImGuiRenderer> imgui;
  lvk::Holder<lvk::ShaderModuleHandle> vert;
  lvk::Holder<lvk::ShaderModuleHandle> frag;
  std::vector<YUVFormatDemo> demos;
};

Resources res_;

void createDemo(const char* name, lvk::Format format, const char* fileName) {
  using namespace std::filesystem;
  path dir = getPathToContentFolder();
  int32_t texWidth = 1920;
  int32_t texHeight = 1080;
  FILE* file = fopen((dir / path(fileName)).string().c_str(), "rb");
  SCOPE_EXIT {
    if (file) {
      fclose(file);
    }
  };
  fseek(file, 0, SEEK_END);
  const uint32_t length = ftell(file);
  fseek(file, 0, SEEK_SET);

  LVK_ASSERT_MSG(file && length, "Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
  if (!file || !length) {
    printf("Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
    std::terminate();
  }

  LVK_ASSERT(length == texWidth * texHeight * 3 / 2);

  std::vector<uint8_t> pixels(length);
  fread(pixels.data(), 1, length, file);

  lvk::Holder<lvk::TextureHandle> texture = ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = format,
      .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = pixels.data(),
      .debugName = name,
  });

  const uint32_t textureId = texture.index();

  res_.demos.push_back({
      .name = name,
      .format = format,
      .texture = std::move(texture),
      .renderPipelineState = ctx_->createRenderPipeline({
          .topology = lvk::Topology_TriangleStrip,
          .smVert = res_.vert,
          .smFrag = res_.frag,
          .specInfo = {.entries = {{.constantId = 0, .size = sizeof(uint32_t)}}, .data = &textureId, .dataSize = sizeof(textureId)},
          .color = {{.format = ctx_->getSwapchainFormat()}},
          .debugName = name,
      }),
  });
}

void init(lvk::LVKwindow* window) {
  res_.imgui = std::make_unique<lvk::ImGuiRenderer>(*ctx_, window, nullptr, float(height_) / 70.0f);

  res_.vert = ctx_->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  res_.frag = ctx_->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});

  createDemo("YUV NV12", lvk::Format_YUV_NV12, "igl-samples/output_frame_900.nv12.yuv");
  createDemo("YUV 420p", lvk::Format_YUV_420p, "igl-samples/output_frame_900.420p.yuv");
}

void destroy() {
  res_ = {};
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

  const lvk::Framebuffer framebuffer = {
      .color = {{.texture = ctx_->getCurrentSwapchainTexture()}},
  };

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_DontCare}}}, framebuffer);

  if (!res_.demos.empty()) {
    const YUVFormatDemo& demo = res_.demos[currentDemo_];
    buffer.cmdBindRenderPipeline(demo.renderPipelineState);
    buffer.cmdDraw(4);
    {
      res_.imgui->beginFrame(framebuffer);
      const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
      ImGui::SetNextWindowPos({15.0f, 15.0f});
      ImGui::SetNextWindowBgAlpha(0.30f);
      ImGui::Begin("##FormatYUV", nullptr, flags);
      ImGui::Text("%s", demo.name);
      ImGui::Text("Press any key to change");
      ImGui::End();
      res_.imgui->endFrame(buffer);
    }
  }

  buffer.cmdEndRendering();

  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

#if !defined(ANDROID)
std::filesystem::path getPathToContentFolder() {
  using namespace std::filesystem;
  path dir = current_path();
  const char* contentFolder = "third-party/content/src/";
  while (dir != current_path().root_path() && !exists(dir / path(contentFolder))) {
    dir = dir.parent_path();
  }
  return dir / path(contentFolder);
}

int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan YUV", width_, height_, true);

  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {});
  if (!ctx_) {
    return 1;
  }
  init(window);

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    resize();
  });
  glfwSetMouseButtonCallback(window, [](auto* window, int button, int action, int mods) {
    if (action == GLFW_PRESS && !res_.demos.empty()) {
      currentDemo_ = (currentDemo_ + 1) % res_.demos.size();
    }
  });
  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_T && action == GLFW_PRESS) {
      currentDemo_ = 0;
      if (!res_.demos.empty())
        res_.demos.pop_back();
    } else if (action == GLFW_PRESS && !res_.demos.empty()) {
      currentDemo_ = (currentDemo_ + 1) % res_.demos.size();
    }
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
std::filesystem::path getPathToContentFolder() {
  if (const char* externalStorage = std::getenv("EXTERNAL_STORAGE")) {
    return std::filesystem::path(externalStorage) / "LVK" / "content" / "src";
  }
  return {};
}

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
    if (fps_.tick(((double)newTime.tv_sec + 1.0e-9 * newTime.tv_nsec) - 
                  ((double)prevTime.tv_sec + 1.0e-9 * prevTime.tv_nsec))) {
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
