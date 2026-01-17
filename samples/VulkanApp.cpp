/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Based on https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition/blob/main/shared/VulkanApp.cpp

#include "VulkanApp.h"

#include <filesystem>
#include <vector>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#if defined(ANDROID)
#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <time.h>

double glfwGetTime() {
  timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1.0e-9 * t.tv_nsec;
}

static const char* cmdToString(int32_t cmd) {
#define CMD(cmd) \
  case cmd:      \
    return #cmd
  switch (cmd) {
    CMD(APP_CMD_INPUT_CHANGED);
    CMD(APP_CMD_INIT_WINDOW);
    CMD(APP_CMD_TERM_WINDOW);
    CMD(APP_CMD_WINDOW_RESIZED);
    CMD(APP_CMD_WINDOW_REDRAW_NEEDED);
    CMD(APP_CMD_CONTENT_RECT_CHANGED);
    CMD(APP_CMD_GAINED_FOCUS);
    CMD(APP_CMD_LOST_FOCUS);
    CMD(APP_CMD_CONFIG_CHANGED);
    CMD(APP_CMD_LOW_MEMORY);
    CMD(APP_CMD_START);
    CMD(APP_CMD_RESUME);
    CMD(APP_CMD_SAVE_STATE);
    CMD(APP_CMD_PAUSE);
    CMD(APP_CMD_STOP);
    CMD(APP_CMD_DESTROY);
  }
#undef CMD
  return "";
}

extern "C" {

static void handle_cmd(android_app* androidApp, int32_t cmd) {
  VulkanApp* app = (VulkanApp*)androidApp->userData;

  LLOGD("handle_cmd(%s)", cmdToString(cmd));

  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (androidApp->window) {
      app->width_ = ANativeWindow_getWidth(androidApp->window) / app->cfg_.framebufferScalar;
      app->height_ = ANativeWindow_getHeight(androidApp->window) / app->cfg_.framebufferScalar;
      if (!app->ctx_)
        app->ctx_ = lvk::createVulkanContextWithSwapchain(androidApp->window, app->width_, app->height_, app->cfg_.contextConfig);
    }
    return;
  case APP_CMD_TERM_WINDOW:
    app->ctx_ = nullptr;
    return;
  }
}

static int32_t handle_input(android_app* androidApp, AInputEvent* event) {
  VulkanApp* app = (VulkanApp*)androidApp->userData;

  if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
    int32_t action = AMotionEvent_getAction(event);
    int32_t actionMasked = action & AMOTION_EVENT_ACTION_MASK;

    float x = AMotionEvent_getX(event, 0);
    float y = AMotionEvent_getY(event, 0);

    // Store both normalized [0,1] and pixel coordinates
    float normalizedX = x / (app->width_ * app->cfg_.framebufferScalar);
    float normalizedY = 1.0f - (y / (app->height_ * app->cfg_.framebufferScalar));

    // Update ImGui
    ImGuiIO& io = ImGui::GetIO();

    switch (actionMasked) {
    case AMOTION_EVENT_ACTION_DOWN:
      // Update position BEFORE setting mouse down for proper hit testing
      io.MousePos = ImVec2(x / app->cfg_.framebufferScalar, y / app->cfg_.framebufferScalar);
      io.MouseDown[0] = true;

      // Always update position
      app->mouseState_.pos.x = normalizedX;
      app->mouseState_.pos.y = normalizedY;

      // Check if touch started on ImGui window AFTER updating both position and down state
      app->imguiCapturedTouch_ = io.WantCaptureMouse;

      // Set pressedLeft immediately if not captured
      if (!app->imguiCapturedTouch_) {
        app->mouseState_.pressedLeft = true;
      }

      LLOGD("Touch down: %.2f, %.2f (ImGui captured: %d)", x, y, app->imguiCapturedTouch_);
      return 1;

    case AMOTION_EVENT_ACTION_MOVE:
      io.MousePos = ImVec2(x / app->cfg_.framebufferScalar, y / app->cfg_.framebufferScalar);
      app->mouseState_.pos.x = normalizedX;
      app->mouseState_.pos.y = normalizedY;
      return 1;

    case AMOTION_EVENT_ACTION_UP:
    case AMOTION_EVENT_ACTION_CANCEL:
      // Keep position valid for the up event so ImGui can process the click
      io.MousePos = ImVec2(x / app->cfg_.framebufferScalar, y / app->cfg_.framebufferScalar);
      io.MouseDown[0] = false;

      app->mouseState_.pressedLeft = false;
      app->imguiCapturedTouch_ = false;
      LLOGD("Touch up: %.2f, %.2f", x, y);
      return 1;
    }
  }

  return 0;
}

static void resize_callback(ANativeActivity* activity, ANativeWindow* window) {
  LLOGD("resize_callback()");

  VulkanApp* app = (VulkanApp*)activity->instance;
  const int w = ANativeWindow_getWidth(window) / app->cfg_.framebufferScalar;
  const int h = ANativeWindow_getHeight(window) / app->cfg_.framebufferScalar;
  if (app->width_ != w || app->height_ != h) {
    app->width_ = w;
    app->height_ = h;
    if (app->ctx_) {
      app->ctx_->recreateSwapchain(w, h);
      app->depthTexture_.reset();
      LLOGD("Swapchain recreated");
    }
  }

  LLOGD("resize_callback()<-");
}

} // extern "C"
#else
double glfwGetTime() {
  return (double)SDL_GetTicks() * 0.001;
}
#endif // ANDROID

#if defined(ANDROID)
VulkanApp::VulkanApp(android_app* androidApp, const VulkanAppConfig& cfg) : androidApp_(androidApp), cfg_(cfg) {
  const char* logFileName = nullptr;
#else
VulkanApp::VulkanApp(int argc, char* argv[], const VulkanAppConfig& cfg) : cfg_(cfg) {
  const char* logFileName = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--headless")) {
      cfg_.contextConfig.enableHeadlessSurface = true;
    } else if (!strcmp(argv[i], "--log-file")) {
      if (i + 1 < argc) {
        logFileName = argv[++i];
      } else {
        LLOGW("Specify a file name for `--log-file <filename>`");
      }
    } else if (!strcmp(argv[i], "--screenshot-frame")) {
      if (i + 1 < argc) {
        cfg_.screenshotFrameNumber = strtoull(argv[++i], nullptr, 10);
      } else {
        LLOGW("Specify a frame number for `--screenshot-frame <framenumber>`");
      }
    } else if (!strcmp(argv[i], "--screenshot-file")) {
      if (i + 1 < argc) {
        cfg_.screenshotFileName = argv[++i];
      } else {
        LLOGW("Specify a file name for `--screenshot-file <filename>`");
      }
    }
  }
#endif // ANDROID
  minilog::initialize(logFileName,
                      {
                          .logLevelPrintToConsole = cfg_.contextConfig.enableHeadlessSurface ? minilog::Debug : minilog::Log,
                          .threadNames = false,
                      });

  // we use minilog
  fpsCounter_.printFPS_ = false;

  // find the content folder
  {
    using namespace std::filesystem;
#if defined(ANDROID)
    if (const char* externalStorage = std::getenv("EXTERNAL_STORAGE")) {
      folderThirdParty_ = (path(externalStorage) / "LVK" / "deps" / "src").string() + "/";
      folderContentRoot_ = (path(externalStorage) / "LVK" / "content").string() + "/";
    }
#else
    path subdir("third-party/content/");
    path dir = current_path();
    // find the content somewhere above our current build directory
    while (dir != current_path().root_path() && !exists(dir / subdir)) {
      dir = dir.parent_path();
    }
    if (!exists(dir / subdir)) {
      LLOGW("Cannot find the content directory. Run `deploy_content.py` before running this app.");
      LVK_ASSERT(false);
    }
    folderThirdParty_ = (dir / path("third-party/deps/src/")).string();
    folderContentRoot_ = (dir / subdir).string();
#endif // ANDROID
  }

#if defined(ANDROID)
  androidApp_->userData = this;
  androidApp_->onAppCmd = handle_cmd;
  androidApp_->onInputEvent = handle_input;

  int events = 0;
  android_poll_source* source = nullptr;

  LLOGD("Waiting for an Android window...");

  while (!androidApp_->destroyRequested && !ctx_) {
    // poll until a Window is created
    if (ALooper_pollOnce(1, nullptr, &events, (void**)&source) >= 0) {
      if (source)
        source->process(androidApp_, source);
    }
  }

  LLOGD("...Android window ready!");

  if (!ctx_)
    return;

  androidApp_->activity->instance = this;
  androidApp_->activity->callbacks->onNativeWindowResized = resize_callback;
#else
  width_ = cfg_.width;
  height_ = cfg_.height;

  window_ = lvk::initWindow("Simple example", width_, height_, cfg_.resizable, cfg_.contextConfig.enableHeadlessSurface);

  ctx_ = lvk::createVulkanContextWithSwapchain(window_, width_, height_, cfg_.contextConfig);
#endif // ANDROID

  // initialize ImGUi after GLFW callbacks have been installed
  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, window_, (folderThirdParty_ + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), 30.0f);
}

VulkanApp::~VulkanApp() {
  imgui_ = nullptr;
  depthTexture_ = nullptr;
  ctx_ = nullptr;
#if !defined(ANDROID)
  if (window_) {
    SDL_DestroyWindow(window_);
  }
  SDL_Quit();
#endif // !ANDROID
}

lvk::Format VulkanApp::getDepthFormat() const {
  return ctx_->getFormat(getDepthTexture());
}

lvk::TextureHandle VulkanApp::getDepthTexture() const {
  if (depthTexture_.empty()) {
    depthTexture_ = ctx_->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_Z_F32,
        .dimensions = {(uint32_t)width_, (uint32_t)height_},
        .usage = lvk::TextureUsageBits_Attachment,
        .debugName = "Depth buffer",
    });
  }

  return depthTexture_;
}

void VulkanApp::run(DrawFrameFunc drawFrame) {
  double timeStamp = glfwGetTime();
  float deltaSeconds = 0.0f;

#if defined(ANDROID)
  int events = 0;
  android_poll_source* source = nullptr;
  do {
    const double newTimeStamp = glfwGetTime();
    deltaSeconds = static_cast<float>(newTimeStamp - timeStamp);
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }
    timeStamp = newTimeStamp;
    if (ctx_) {
      const float ratio = width_ / (float)height_;

      const bool justPressed = mouseState_.pressedLeft && !imguiLastPressedLeft_;

      positioner_.update(
          deltaSeconds, mouseState_.pos, ImGui::GetIO().WantCaptureMouse ? false : (mouseState_.pressedLeft && !justPressed));

      // clear ImGui hover state one frame after touch ends
      if (imguiClearMouseNextFrame_) {
        ImGui::GetIO().MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
        imguiClearMouseNextFrame_ = false;
      }

      imguiClearMouseNextFrame_ = !mouseState_.pressedLeft && imguiLastPressedLeft_;
      imguiLastPressedLeft_ = mouseState_.pressedLeft;

      drawFrame((uint32_t)width_, (uint32_t)height_, ratio, deltaSeconds);
    }
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source) {
        source->process(androidApp_, source);
      }
    }
  } while (!androidApp_->destroyRequested);
#else
  bool running = true;
  SDL_Event event;

  while (cfg_.contextConfig.enableHeadlessSurface || running) {
    const double newTimeStamp = glfwGetTime();
    deltaSeconds = static_cast<float>(newTimeStamp - timeStamp);
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }
    timeStamp = newTimeStamp;

    while (SDL_PollEvent(&event)) {
      ImGuiIO& io = ImGui::GetIO();

      switch (event.type) {
      case SDL_EVENT_QUIT:
        running = false;
        break;

      case SDL_EVENT_WINDOW_RESIZED:
      case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
        int w, h;
        SDL_GetWindowSizeInPixels(window_, &w, &h);
        if (width_ != w || height_ != h) {
          width_ = w;
          height_ = h;
          ctx_->recreateSwapchain(width_, height_);
          depthTexture_.reset();
        }
        break;
      }

      case SDL_EVENT_MOUSE_BUTTON_DOWN:
      case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (event.button.button == SDL_BUTTON_LEFT) {
          mouseState_.pressedLeft = (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        }

        ImGuiMouseButton imguiButton = ImGuiMouseButton_Left;
        if (event.button.button == SDL_BUTTON_RIGHT)
          imguiButton = ImGuiMouseButton_Right;
        else if (event.button.button == SDL_BUTTON_MIDDLE)
          imguiButton = ImGuiMouseButton_Middle;

        io.MousePos = ImVec2((float)event.button.x, (float)event.button.y);
        io.MouseDown[imguiButton] = (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);

        for (auto& cb : callbacksMouseButton) {
          cb(window_, &event.button);
        }
        break;
      }

      case SDL_EVENT_MOUSE_WHEEL:
        io.MouseWheelH = event.wheel.x;
        io.MouseWheel = event.wheel.y;
        break;

      case SDL_EVENT_MOUSE_MOTION:
        io.MousePos = ImVec2((float)event.motion.x, (float)event.motion.y);
        mouseState_.pos.x = static_cast<float>(event.motion.x / (float)width_);
        mouseState_.pos.y = 1.0f - static_cast<float>(event.motion.y / (float)height_);
        break;

      case SDL_EVENT_KEY_DOWN:
      case SDL_EVENT_KEY_UP: {
        bool pressed = (event.type == SDL_EVENT_KEY_DOWN);
        SDL_Keycode key = event.key.key;

        if (key == SDLK_ESCAPE && pressed)
          running = false;
        if (key == SDLK_W)
          positioner_.movement_.forward_ = pressed;
        if (key == SDLK_S)
          positioner_.movement_.backward_ = pressed;
        if (key == SDLK_A)
          positioner_.movement_.left_ = pressed;
        if (key == SDLK_D)
          positioner_.movement_.right_ = pressed;
        if (key == SDLK_1)
          positioner_.movement_.up_ = pressed;
        if (key == SDLK_2)
          positioner_.movement_.down_ = pressed;

        positioner_.movement_.fastSpeed_ = (event.key.mod & SDL_KMOD_SHIFT) != 0;

        if (key == SDLK_SPACE && pressed) {
          positioner_.lookAt(cfg_.initialCameraPos, cfg_.initialCameraTarget, cfg_.initialCameraUpVector);
        }

        for (auto& cb : callbacksKey) {
          cb(window_, &event.key);
        }
        break;
      }
      }
    }

    if (!ctx_ || !width_ || !height_)
      continue;

    const float ratio = width_ / (float)height_;

    positioner_.update(deltaSeconds, mouseState_.pos, ImGui::GetIO().WantCaptureMouse ? false : mouseState_.pressedLeft);

    lvk::TextureHandle tex = ctx_->getCurrentSwapchainTexture();

    drawFrame((uint32_t)width_, (uint32_t)height_, ratio, deltaSeconds);

    if (cfg_.screenshotFrameNumber == ++frameCount_) {
      ctx_->wait({});
      const lvk::Dimensions dim = ctx_->getDimensions(tex);
      const lvk::Format format = ctx_->getFormat(tex);
      LLOGL("Saving screenshot...%ux%u\n", dim.width, dim.height);
      if (format != lvk::Format_BGRA_UN8 && format != lvk::Format_BGRA_SRGB8 && format != lvk::Format_RGBA_UN8 &&
          format != lvk::Format_RGBA_SRGB8) {
        LLOGW("Unsupported pixel format %u\n", (uint32_t)format);
        break;
      }
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      ctx_->download(tex, {.dimensions = {dim.width, dim.height}}, pixelsRGBA.data());
      if (format == lvk::Format_BGRA_UN8 || format == lvk::Format_BGRA_SRGB8) {
        // swap R-B
        for (uint32_t i = 0; i < pixelsRGBA.size(); i += 4) {
          std::swap(pixelsRGBA[i + 0], pixelsRGBA[i + 2]);
        }
      }
      // convert to RGB
      for (uint32_t i = 0; i < pixelsRGB.size() / 3; i++) {
        pixelsRGB[3 * i + 0] = pixelsRGBA[4 * i + 0];
        pixelsRGB[3 * i + 1] = pixelsRGBA[4 * i + 1];
        pixelsRGB[3 * i + 2] = pixelsRGBA[4 * i + 2];
      }
      stbi_write_png(cfg_.screenshotFileName, (int)dim.width, (int)dim.height, 3, pixelsRGB.data(), 0);
      break;
    }
  }
#endif // ANDROID

  LLOGD("Terminating app...");
}

void VulkanApp::drawFPS() {
  if (const ImGuiViewport* v = ImGui::GetMainViewport()) {
    ImGui::SetNextWindowPos({v->WorkPos.x + v->WorkSize.x - 15.0f, v->WorkPos.y + 15.0f}, ImGuiCond_Always, {1.0f, 0.0f});
  }
  ImGui::SetNextWindowBgAlpha(0.30f);
  ImGui::SetNextWindowSize(ImVec2(ImGui::CalcTextSize("FPS : _______").x, 0));
  if (ImGui::Begin("##FPS",
                   nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                       ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove)) {
    ImGui::Text("FPS : %i", (int)fpsCounter_.getFPS());
    ImGui::Text("Ms  : %.1f", fpsCounter_.getFPS() > 0 ? 1000.0 / fpsCounter_.getFPS() : 0);
  }
  ImGui::End();
}
