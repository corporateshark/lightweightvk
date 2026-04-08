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
#elif LVK_WITH_SDL3
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
    } else if (!strcmp(argv[i], "--width")) {
      if (i + 1 < argc) {
        cfg_.width = (int)strtol(argv[++i], nullptr, 10);
      } else {
        LLOGW("Specify a value for `--width <pixels>`");
      }
    } else if (!strcmp(argv[i], "--height")) {
      if (i + 1 < argc) {
        cfg_.height = (int)strtol(argv[++i], nullptr, 10);
      } else {
        LLOGW("Specify a value for `--height <pixels>`");
      }
    }
  }
#endif // ANDROID
#if defined(LVK_WITH_MINILOG)
  minilog::initialize(logFileName,
                      {
                          .logLevelPrintToConsole = cfg_.contextConfig.enableHeadlessSurface ? minilog::Debug : minilog::Log,
                          .threadNames = false,
                      });
#endif

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
#if LVK_WITH_OPENXR
  if (cfg_.enableOpenXR) {
#if defined(LVK_WITH_GLFW)
    glfwInit();
#endif // LVK_WITH_GLFW
    initOpenXR();
    initXrSession();
    initXrSwapchains();
    width_ = xrSwapchains_[0].width;
    height_ = xrSwapchains_[0].height;
    xrLastTimeStamp_ = glfwGetTime();
  } else
#endif // LVK_WITH_OPENXR
  {
    width_ = cfg_.width;
    height_ = cfg_.height;
    window_ = lvk::initWindow("Simple example", width_, height_, cfg_.resizable, cfg_.contextConfig.enableHeadlessSurface);
    ctx_ = lvk::createVulkanContextWithSwapchain(window_, width_, height_, cfg_.contextConfig);
  }
#endif // ANDROID

#if LVK_WITH_GLFW
  if (window_) {
    glfwSetWindowUserPointer(window_, this);

    glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* window, int width, int height) {
      VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
      if (app->width_ == width && app->height_ == height)
        return;
      app->width_ = width;
      app->height_ = height;
      app->ctx_->recreateSwapchain(width, height);
      app->depthTexture_.reset();
    });
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* window, int button, int action, int mods) {
      VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
      if (button == GLFW_MOUSE_BUTTON_LEFT) {
        app->mouseState_.pressedLeft = action == GLFW_PRESS;
      }
      for (auto& cb : app->callbacksMouseButton) {
        cb(window, button, action, mods);
      }
    });
    glfwSetCursorPosCallback(window_, [](GLFWwindow* window, double x, double y) {
      VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
      const ImVec2 size = ImGui::GetIO().DisplaySize;
      app->mouseState_.pos.x = static_cast<float>(x / size.x);
      app->mouseState_.pos.y = 1.0f - static_cast<float>(y / size.y);
    });
    glfwSetKeyCallback(window_, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
      VulkanApp* app = (VulkanApp*)glfwGetWindowUserPointer(window);
      const bool pressed = action != GLFW_RELEASE && !ImGui::GetIO().WantCaptureKeyboard;
      if (key == GLFW_KEY_ESCAPE && pressed)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
      if (key == GLFW_KEY_W)
        app->positioner_.movement_.forward_ = pressed;
      if (key == GLFW_KEY_S)
        app->positioner_.movement_.backward_ = pressed;
      if (key == GLFW_KEY_A)
        app->positioner_.movement_.left_ = pressed;
      if (key == GLFW_KEY_D)
        app->positioner_.movement_.right_ = pressed;
      if (key == GLFW_KEY_1)
        app->positioner_.movement_.up_ = pressed;
      if (key == GLFW_KEY_2)
        app->positioner_.movement_.down_ = pressed;

      app->positioner_.movement_.fastSpeed_ = (mods & GLFW_MOD_SHIFT) != 0;

      if (key == GLFW_KEY_SPACE) {
        app->positioner_.lookAt(app->cfg_.initialCameraPos, app->cfg_.initialCameraTarget, app->cfg_.initialCameraUpVector);
      }
      for (auto& cb : app->callbacksKey) {
        cb(window, key, scancode, action, mods);
      }
    });
  }
#endif // LVK_WITH_GLFW

  // initialize ImGUi after GLFW callbacks have been installed
  imgui_ = std::make_unique<lvk::ImGuiRenderer>(
      *ctx_, window_, (folderThirdParty_ + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(), 30.0f);
}

VulkanApp::~VulkanApp() {
  imgui_ = nullptr;
  depthTexture_ = nullptr;
#if LVK_WITH_OPENXR
  if (cfg_.enableOpenXR) {
    if (ctx_) {
      ctx_->wait({});
    }
    destroyXrSwapchains();
    if (xrAppSpace_) {
      xrDestroySpace(xrAppSpace_);
    }
    if (xrSession_) {
      xrDestroySession(xrSession_);
    }
    ctx_ = nullptr;
    if (xrInstance_) {
      xrDestroyInstance(xrInstance_);
    }
#if LVK_WITH_GLFW
    glfwTerminate();
#endif // LVK_WITH_GLFW
    return;
  }
#endif // LVK_WITH_OPENXR
  ctx_ = nullptr;
#if LVK_WITH_GLFW
  glfwDestroyWindow(window_);
  glfwTerminate();
#elif LVK_WITH_SDL3
  if (window_) {
    SDL_DestroyWindow(window_);
  }
  SDL_Quit();
#endif
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
#if LVK_WITH_OPENXR
  if (cfg_.enableOpenXR) {
    while (!xrShouldQuit_) {
      pollXrEvents();
      if (xrShouldQuit_)
        break;
      if (!renderXrFrame(drawFrame))
        break;
    }
    return;
  }
#endif // LVK_WITH_OPENXR

  double timeStamp = glfwGetTime();

#if defined(ANDROID)
  const float kTimeQuantum = 0.02f;
  double accTime = 0;
  int events = 0;
  android_poll_source* source = nullptr;
  do {
    const double newTimeStamp = glfwGetTime();
    const float deltaSeconds = static_cast<float>(newTimeStamp - timeStamp);
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }
    timeStamp = newTimeStamp;
    // simulation: tick in fixed quanta
    accTime += deltaSeconds;
    while (accTime >= kTimeQuantum) {
      accTime -= kTimeQuantum;
      simulatedTime_ += kTimeQuantum;
    }
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

      const RenderView view = {
          .viewport = {0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f},
          .scissorRect = {0, 0, (uint32_t)width_, (uint32_t)height_},
          .colorTexture = ctx_->getCurrentSwapchainTexture(),
          .aspectRatio = ratio,
      };
      drawFrame({&view, 1}, deltaSeconds);
    }
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source) {
        source->process(androidApp_, source);
      }
    }
  } while (!androidApp_->destroyRequested);
#elif LVK_WITH_GLFW
  const float kTimeQuantum = 0.02f;
  double accTime = 0;
  while (cfg_.contextConfig.enableHeadlessSurface || !glfwWindowShouldClose(window_)) {
    const double newTimeStamp = glfwGetTime();
    const float deltaSeconds = cfg_.screenshotFrameNumber ? kTimeQuantum : static_cast<float>(newTimeStamp - timeStamp);
    timeStamp = newTimeStamp;

    if (window_) {
      glfwGetFramebufferSize(window_, &width_, &height_);

      glfwPollEvents();
    }

    if (!ctx_ || !width_ || !height_)
      continue;

    // simulation: tick in fixed quanta
    accTime += deltaSeconds;
    while (accTime >= kTimeQuantum) {
      accTime -= kTimeQuantum;
      simulatedTime_ += kTimeQuantum;
      positioner_.update(kTimeQuantum, mouseState_.pos, ImGui::GetIO().WantCaptureMouse ? false : mouseState_.pressedLeft);
    }
    // FPS measurement: real time
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }

    const float ratio = width_ / (float)height_;

    const RenderView view = {
        .viewport = {0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f},
        .scissorRect = {0, 0, (uint32_t)width_, (uint32_t)height_},
        .colorTexture = ctx_->getCurrentSwapchainTexture(),
        .aspectRatio = ratio,
    };

    drawFrame({&view, 1}, deltaSeconds);

    if (cfg_.screenshotFrameNumber == ++frameCount_) {
      ctx_->wait({});
      const lvk::Dimensions dim = ctx_->getDimensions(view.colorTexture);
      const lvk::Format format = ctx_->getFormat(view.colorTexture);
      LLOGL("Saving screenshot...%ux%u\n", dim.width, dim.height);
      if (format != lvk::Format_BGRA_UN8 && format != lvk::Format_BGRA_SRGB8 && format != lvk::Format_RGBA_UN8 &&
          format != lvk::Format_RGBA_SRGB8) {
        LLOGW("Unsupported pixel format %u\n", (uint32_t)format);
        break;
      }
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      ctx_->download(view.colorTexture, {.dimensions = {dim.width, dim.height}}, pixelsRGBA.data());
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
#elif LVK_WITH_SDL3
  bool running = true;
  SDL_Event event;

  const float kTimeQuantum = 0.02f;
  double accTime = 0;
  while (cfg_.contextConfig.enableHeadlessSurface || running) {
    const double newTimeStamp = glfwGetTime();
    const float deltaSeconds = cfg_.screenshotFrameNumber ? kTimeQuantum : static_cast<float>(newTimeStamp - timeStamp);
    timeStamp = newTimeStamp;

    if (!ctx_ || !width_ || !height_)
      continue;

    // simulation: tick in fixed quanta
    accTime += deltaSeconds;
    while (accTime >= kTimeQuantum) {
      accTime -= kTimeQuantum;
      simulatedTime_ += kTimeQuantum;
      positioner_.update(kTimeQuantum, mouseState_.pos, ImGui::GetIO().WantCaptureMouse ? false : mouseState_.pressedLeft);
    }
    // FPS measurement: real time
    if (fpsCounter_.tick(deltaSeconds)) {
      LLOGL("FPS: %.1f\n", fpsCounter_.getFPS());
    }

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

    const RenderView view = {
        .viewport = {0.0f, 0.0f, (float)width_, (float)height_, 0.0f, 1.0f},
        .scissorRect = {0, 0, (uint32_t)width_, (uint32_t)height_},
        .colorTexture = ctx_->getCurrentSwapchainTexture(),
        .aspectRatio = ratio,
    };

    drawFrame({&view, 1}, deltaSeconds);

    if (cfg_.screenshotFrameNumber == ++frameCount_) {
      ctx_->wait({});
      const lvk::Dimensions dim = ctx_->getDimensions(view.colorTexture);
      const lvk::Format format = ctx_->getFormat(view.colorTexture);
      LLOGL("Saving screenshot...%ux%u\n", dim.width, dim.height);
      if (format != lvk::Format_BGRA_UN8 && format != lvk::Format_BGRA_SRGB8 && format != lvk::Format_RGBA_UN8 &&
          format != lvk::Format_RGBA_SRGB8) {
        LLOGW("Unsupported pixel format %u\n", (uint32_t)format);
        break;
      }
      std::vector<uint8_t> pixelsRGBA(dim.width * dim.height * 4);
      std::vector<uint8_t> pixelsRGB(dim.width * dim.height * 3);
      ctx_->download(view.colorTexture, {.dimensions = {dim.width, dim.height}}, pixelsRGBA.data());
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
#endif // ANDROID / LVK_WITH_GLFW / LVK_WITH_SDL3

  LLOGD("Terminating app...");
}

double VulkanApp::getSimulatedTime() const {
  return simulatedTime_;
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

#if LVK_WITH_OPENXR

void VulkanApp::initOpenXR() {
  const char* extensions[] = {
      XR_KHR_VULKAN_ENABLE_EXTENSION_NAME,
  };

  const XrInstanceCreateInfo instanceCI = {
      .type = XR_TYPE_INSTANCE_CREATE_INFO,
      .applicationInfo =
          {
              .applicationName = "LVK OpenXR Samples",
              .applicationVersion = 1,
              .engineName = "LightweightVK",
              .engineVersion = 1,
              .apiVersion = XR_API_VERSION_1_1,
          },
      .enabledExtensionCount = 1,
      .enabledExtensionNames = extensions,
  };

  const XrResult result = xrCreateInstance(&instanceCI, &xrInstance_);
  if (XR_FAILED(result)) {
    LLOGW("Failed to create OpenXR instance (%s). Is an OpenXR runtime available?\n", lvk::xrResultToString(result));
    LVK_ASSERT(false);
    return;
  }

  XrInstanceProperties instanceProps = {.type = XR_TYPE_INSTANCE_PROPERTIES};
  XR_ASSERT(xrGetInstanceProperties(xrInstance_, &instanceProps));
  LLOGL("OpenXR Runtime: %s v%u.%u.%u\n",
        instanceProps.runtimeName,
        XR_VERSION_MAJOR(instanceProps.runtimeVersion),
        XR_VERSION_MINOR(instanceProps.runtimeVersion),
        XR_VERSION_PATCH(instanceProps.runtimeVersion));

  // get system (HMD)
  const XrSystemGetInfo systemGI = {
      .type = XR_TYPE_SYSTEM_GET_INFO,
      .formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY,
  };
  {
    const XrResult sysResult = xrGetSystem(xrInstance_, &systemGI, &xrSystemId_);
    if (sysResult == XR_ERROR_FORM_FACTOR_UNAVAILABLE) {
      LLOGW("No HMD found. Make sure your Quest 3 is connected and Quest Link is active.\n");
      exit(1);
    }
    if (XR_FAILED(sysResult)) {
      LLOGW("OpenXR error: xrGetSystem() returned %s (%d)\n", lvk::xrResultToString(sysResult), (int)sysResult);
      LVK_ASSERT_MSG(false, "xrGetSystem() failed");
    }
  }

  XrSystemProperties systemProps = {.type = XR_TYPE_SYSTEM_PROPERTIES};
  XR_ASSERT(xrGetSystemProperties(xrInstance_, xrSystemId_, &systemProps));
  LLOGL("OpenXR System: %s (vendorId=%u)\n", systemProps.systemName, systemProps.vendorId);

  // OpenXR extension function pointers
  PFN_xrGetVulkanInstanceExtensionsKHR xrGetVulkanInstanceExtensions = nullptr;
  PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensions = nullptr;

  // load extension functions
  XR_ASSERT(xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanInstanceExtensionsKHR", (PFN_xrVoidFunction*)&xrGetVulkanInstanceExtensions));
  XR_ASSERT(xrGetInstanceProcAddr(xrInstance_, "xrGetVulkanDeviceExtensionsKHR", (PFN_xrVoidFunction*)&xrGetVulkanDeviceExtensions));

  // replace spaces with null terminators in-place so that pointers into the string are valid C strings
  auto parseExtensionString = [](std::string& storage, std::vector<const char*>& outPtrs) {
    outPtrs.clear();
    char* p = storage.data();
    const char* const end = p + storage.size();
    while (p < end) {
      while (p < end && (isblank(*p) || *p == '\0'))
        *p++ = '\0';
      if (p < end)
        outPtrs.push_back(p);
      while (p < end && (!isblank(*p) && *p != '\0'))
        p++;
    }
  };

  std::vector<const char*> xrInstanceExtPtrs;
  std::vector<const char*> xrDeviceExtPtrs;

  // get required Vulkan instance extensions from XR runtime
  uint32_t numInstanceExts = 0;
  XR_ASSERT(xrGetVulkanInstanceExtensions(xrInstance_, xrSystemId_, 0, &numInstanceExts, nullptr));
  if (numInstanceExts) {
    xrVulkanExtsInstance_.resize(numInstanceExts);
    XR_ASSERT(xrGetVulkanInstanceExtensions(xrInstance_, xrSystemId_, numInstanceExts, &numInstanceExts, xrVulkanExtsInstance_.data()));
    while (!xrVulkanExtsInstance_.empty() && xrVulkanExtsInstance_.back() == '\0') {
      xrVulkanExtsInstance_.pop_back();
    }
    LLOGL("OpenXR required Vulkan instance extensions: %s\n", xrVulkanExtsInstance_.c_str());
    parseExtensionString(xrVulkanExtsInstance_, xrInstanceExtPtrs);
  }

  // get required Vulkan device extensions from XR runtime
  uint32_t numDeviceExts = 0;
  XR_ASSERT(xrGetVulkanDeviceExtensions(xrInstance_, xrSystemId_, 0, &numDeviceExts, nullptr));
  if (numDeviceExts) {
    xrVulkanExtsDevice_.resize(numDeviceExts);
    XR_ASSERT(xrGetVulkanDeviceExtensions(xrInstance_, xrSystemId_, numDeviceExts, &numDeviceExts, xrVulkanExtsDevice_.data()));
    while (!xrVulkanExtsDevice_.empty() && xrVulkanExtsDevice_.back() == '\0') {
      xrVulkanExtsDevice_.pop_back();
    }
    LLOGL("OpenXR required Vulkan device extensions: %s\n", xrVulkanExtsDevice_.c_str());
    parseExtensionString(xrVulkanExtsDevice_, xrDeviceExtPtrs);
  }

  // query view configuration
  uint32_t numViews = 0;
  XR_ASSERT(xrEnumerateViewConfigurationViews(xrInstance_, xrSystemId_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &numViews, nullptr));
  xrConfigViews_.resize(numViews, {.type = XR_TYPE_VIEW_CONFIGURATION_VIEW});
  XR_ASSERT(xrEnumerateViewConfigurationViews(
      xrInstance_, xrSystemId_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, numViews, &numViews, xrConfigViews_.data()));

  LVK_ASSERT(numViews == 2);

  LLOGL("OpenXR view config: %ux%u (recommended), %ux%u (max)\n",
        xrConfigViews_[0].recommendedImageRectWidth,
        xrConfigViews_[0].recommendedImageRectHeight,
        xrConfigViews_[0].maxImageRectWidth,
        xrConfigViews_[0].maxImageRectHeight);

  // add XR-required Vulkan instance extensions
  {
    uint32_t idx = 0;
    while (idx < lvk::kMaxCustomExtensions && cfg_.contextConfig.extensionsInstance[idx])
      idx++;
    for (const char* ext : xrInstanceExtPtrs) {
      if (LVK_VERIFY(idx < lvk::kMaxCustomExtensions))
        cfg_.contextConfig.extensionsInstance[idx++] = ext;
    }
  }

  // add XR-required Vulkan device extensions
  {
    uint32_t idx = 0;
    while (idx < lvk::kMaxCustomExtensions && cfg_.contextConfig.extensionsDevice[idx])
      idx++;
    for (const char* ext : xrDeviceExtPtrs) {
      if (LVK_VERIFY(idx < lvk::kMaxCustomExtensions))
        cfg_.contextConfig.extensionsDevice[idx++] = ext;
    }
  }

  ctx_ = createVulkanContextXR(xrInstance_, xrSystemId_, xrGetInstanceProcAddr, cfg_.contextConfig);
}

void VulkanApp::initXrSession() {
  const lvk::VulkanContext* vkCtx = static_cast<lvk::VulkanContext*>(ctx_.get());

  const XrGraphicsBindingVulkanKHR graphicsBinding = {
      .type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR,
      .instance = vkCtx->getVkInstance(),
      .physicalDevice = vkCtx->getVkPhysicalDevice(),
      .device = vkCtx->getVkDevice(),
      .queueFamilyIndex = vkCtx->deviceQueues_.graphicsQueueFamilyIndex,
      .queueIndex = 0,
  };

  const XrSessionCreateInfo sessionCI = {
      .type = XR_TYPE_SESSION_CREATE_INFO,
      .next = &graphicsBinding,
      .systemId = xrSystemId_,
  };

  XR_ASSERT(xrCreateSession(xrInstance_, &sessionCI, &xrSession_));
  LLOGL("OpenXR session created\n");

  const XrReferenceSpaceCreateInfo spaceCI = {
      .type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
      .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL,
      .poseInReferenceSpace =
          {
              .orientation = {.x = 0, .y = 0, .z = 0, .w = 1},
              .position = {.x = 0, .y = 0, .z = 0},
          },
  };
  XR_ASSERT(xrCreateReferenceSpace(xrSession_, &spaceCI, &xrAppSpace_));
}

void VulkanApp::initXrSwapchains() {
  lvk::VulkanContext* const vkCtx = static_cast<lvk::VulkanContext*>(ctx_.get());

  uint32_t numFormats = 0;
  XR_ASSERT(xrEnumerateSwapchainFormats(xrSession_, 0, &numFormats, nullptr));
  std::vector<int64_t> formats(numFormats);
  XR_ASSERT(xrEnumerateSwapchainFormats(xrSession_, numFormats, &numFormats, formats.data()));

  LVK_ASSERT(!formats.empty());

  // select the first available format from a preference list
  auto selectFormat = [&formats](const std::vector<VkFormat>& preferred, VkFormat fallback) -> VkFormat {
    for (const VkFormat p : preferred) {
      for (const int64_t a : formats) {
        if (a == p)
          return p;
      }
    }
    return fallback;
  };

  const VkFormat vkColorFormat = selectFormat(
      {
          // TODO: add HDR formats based on ContextConfig::swapchainRequestedColorSpace
          VK_FORMAT_R8G8B8A8_SRGB,
          VK_FORMAT_B8G8R8A8_SRGB,
          VK_FORMAT_R8G8B8A8_UNORM,
          VK_FORMAT_B8G8R8A8_UNORM,
      },
      VkFormat(formats[0]));
  LLOGD("OpenXR color format: VkFormat %d\n", (int)vkColorFormat);

  const VkFormat vkDepthFormat = selectFormat(
      {
          VK_FORMAT_D32_SFLOAT,
          VK_FORMAT_D24_UNORM_S8_UINT,
          VK_FORMAT_D16_UNORM,
      },
      VK_FORMAT_UNDEFINED);

  for (uint32_t eye = 0; eye < 2; eye++) {
    const uint32_t width = xrConfigViews_[eye].recommendedImageRectWidth;
    const uint32_t height = xrConfigViews_[eye].recommendedImageRectHeight;

    // color swapchain
    {
      const XrSwapchainCreateInfo swapchainCI = {
          .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
          .usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT,
          .format = vkColorFormat,
          .sampleCount = 1,
          .width = width,
          .height = height,
          .faceCount = 1,
          .arraySize = 1,
          .mipCount = 1,
      };

      XrSwapchainData& sc = xrSwapchains_[eye];
      XR_ASSERT(xrCreateSwapchain(xrSession_, &swapchainCI, &sc.swapchain));
      sc.format = vkColorFormat;
      sc.width = width;
      sc.height = height;

      uint32_t numImages = 0;
      XR_ASSERT(xrEnumerateSwapchainImages(sc.swapchain, 0, &numImages, nullptr));
      sc.images.resize(numImages, {.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
      XR_ASSERT(xrEnumerateSwapchainImages(sc.swapchain, numImages, &numImages, (XrSwapchainImageBaseHeader*)sc.images.data()));

      sc.textures.resize(numImages);
      for (uint32_t i = 0; i < numImages; i++) {
        char debugNameImage[256];
        char debugNameView[256];
        snprintf(debugNameImage, sizeof(debugNameImage), "Image: XR eye%u color %u", eye, i);
        snprintf(debugNameView, sizeof(debugNameView), "Image View: XR eye%u color %u", eye, i);

        lvk::VulkanImage image = {
            .vkImage_ = sc.images[i].image,
            .vkUsageFlags_ = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .vkExtent_ = {.width = width, .height = height, .depth = 1},
            .vkType_ = VK_IMAGE_TYPE_2D,
            .vkImageFormat_ = vkColorFormat,
            .isOwningVkImage_ = false,
        };

        lvk::setDebugObjectName(vkCtx->getVkDevice(), VK_OBJECT_TYPE_IMAGE, (uint64_t)image.vkImage_, debugNameImage);

        image.imageView_ = image.createImageView(vkCtx->getVkDevice(),
                                                 VK_IMAGE_VIEW_TYPE_2D,
                                                 vkColorFormat,
                                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                                 0,
                                                 VK_REMAINING_MIP_LEVELS,
                                                 0,
                                                 1,
                                                 {},
                                                 nullptr,
                                                 debugNameView);

        sc.textures[i] = vkCtx->texturesPool_.create(std::move(image));
      }
    }

    // depth swapchain
    if (vkDepthFormat != VK_FORMAT_UNDEFINED) {
      const XrSwapchainCreateInfo depthCI = {
          .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
          .usageFlags = XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
          .format = vkDepthFormat,
          .sampleCount = 1,
          .width = width,
          .height = height,
          .faceCount = 1,
          .arraySize = 1,
          .mipCount = 1,
      };

      XrDepthSwapchainData& dsc = xrDepthSwapchains_[eye];
      XR_ASSERT(xrCreateSwapchain(xrSession_, &depthCI, &dsc.swapchain));
      dsc.width = width;
      dsc.height = height;

      uint32_t numImages = 0;
      XR_ASSERT(xrEnumerateSwapchainImages(dsc.swapchain, 0, &numImages, nullptr));
      dsc.images.resize(numImages, {.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
      XR_ASSERT(xrEnumerateSwapchainImages(dsc.swapchain, numImages, &numImages, (XrSwapchainImageBaseHeader*)dsc.images.data()));

      dsc.textures.resize(numImages);
      for (uint32_t i = 0; i < numImages; i++) {
        char debugNameImage[256];
        char debugNameView[256];
        snprintf(debugNameImage, sizeof(debugNameImage), "Image: XR eye%u depth %u", eye, i);
        snprintf(debugNameView, sizeof(debugNameView), "Image View: XR eye%u depth %u", eye, i);

        lvk::VulkanImage image = {
            .vkImage_ = dsc.images[i].image,
            .vkUsageFlags_ = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .vkExtent_ = {.width = width, .height = height, .depth = 1},
            .vkType_ = VK_IMAGE_TYPE_2D,
            .vkImageFormat_ = vkDepthFormat,
            .isOwningVkImage_ = false,
            .isDepthFormat_ = lvk::VulkanImage::isDepthFormat(vkDepthFormat),
            .isStencilFormat_ = lvk::VulkanImage::isStencilFormat(vkDepthFormat),
        };

        lvk::setDebugObjectName(vkCtx->getVkDevice(), VK_OBJECT_TYPE_IMAGE, (uint64_t)image.vkImage_, debugNameImage);

        const VkImageAspectFlags aspect = image.getImageAspectFlags();
        image.imageView_ = image.createImageView(vkCtx->getVkDevice(),
                                                 VK_IMAGE_VIEW_TYPE_2D,
                                                 vkDepthFormat,
                                                 aspect,
                                                 0,
                                                 VK_REMAINING_MIP_LEVELS,
                                                 0,
                                                 1,
                                                 {},
                                                 nullptr,
                                                 debugNameView);

        dsc.textures[i] = vkCtx->texturesPool_.create(std::move(image));
      }
    }
  }

  LLOGL("OpenXR swapchains created: %ux%u per eye, %u images\n",
        xrSwapchains_[0].width,
        xrSwapchains_[0].height,
        (uint32_t)xrSwapchains_[0].images.size());
}

void VulkanApp::destroyXrSwapchains() {
  if (ctx_) {
    for (uint32_t eye = 0; eye < 2; eye++) {
      for (lvk::TextureHandle tex : xrSwapchains_[eye].textures) {
        ctx_->destroy(tex);
      }
      xrSwapchains_[eye].textures.clear();

      for (lvk::TextureHandle tex : xrDepthSwapchains_[eye].textures) {
        ctx_->destroy(tex);
      }
      xrDepthSwapchains_[eye].textures.clear();
    }
  }

  for (uint32_t eye = 0; eye < 2; eye++) {
    if (xrSwapchains_[eye].swapchain) {
      xrDestroySwapchain(xrSwapchains_[eye].swapchain);
      xrSwapchains_[eye].swapchain = XR_NULL_HANDLE;
    }
    if (xrDepthSwapchains_[eye].swapchain) {
      xrDestroySwapchain(xrDepthSwapchains_[eye].swapchain);
      xrDepthSwapchains_[eye].swapchain = XR_NULL_HANDLE;
    }
  }
}

void VulkanApp::pollXrEvents() {
  XrEventDataBuffer event = {.type = XR_TYPE_EVENT_DATA_BUFFER};

  while (xrPollEvent(xrInstance_, &event) == XR_SUCCESS) {
    switch (event.type) {
    case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
      const XrEventDataSessionStateChanged* stateEvent = (XrEventDataSessionStateChanged*)&event;
      xrSessionState_ = stateEvent->state;
      LLOGL("OpenXR session state: %s\n", lvk::xrSessionStateToString(xrSessionState_));
      switch (xrSessionState_) {
      case XR_SESSION_STATE_READY: {
        const XrSessionBeginInfo beginInfo = {
            .type = XR_TYPE_SESSION_BEGIN_INFO,
            .primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
        };
        XR_ASSERT(xrBeginSession(xrSession_, &beginInfo));
        xrSessionRunning_ = true;
        LLOGL("OpenXR session started\n");
        break;
      }
      case XR_SESSION_STATE_STOPPING:
        XR_ASSERT(xrEndSession(xrSession_));
        xrSessionRunning_ = false;
        LLOGL("OpenXR session stopped\n");
        break;
      case XR_SESSION_STATE_EXITING:
      case XR_SESSION_STATE_LOSS_PENDING:
        xrShouldQuit_ = true;
        break;
      default:
        break;
      }
      break;
    }
    case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
      xrShouldQuit_ = true;
      break;
    default:
      break;
    }
    event = {.type = XR_TYPE_EVENT_DATA_BUFFER};
  }
}

mat4 VulkanApp::xrCreateProjectionMatrix(const XrFovf& fov, float nearZ, float farZ) {
  const float tanL = tanf(fov.angleLeft);
  const float tanR = tanf(fov.angleRight);
  const float tanU = tanf(fov.angleUp);
  const float tanD = tanf(fov.angleDown);

  const float tanW = tanR - tanL;
  const float tanH = tanU - tanD;

  // clang-format off
  return mat4(
             2.0f / tanW,                  0.0f,                              0.0f,   0.0f,
                    0.0f,           2.0f / tanH,                              0.0f,   0.0f,
    (tanR + tanL) / tanW,  (tanU + tanD) / tanH,            -farZ / (farZ - nearZ),  -1.0f,
                    0.0f,                  0.0f,  -(farZ * nearZ) / (farZ - nearZ),   0.0f);
  // clang-format on
}

mat4 VulkanApp::xrCreateViewMatrix(const XrPosef& pose) {
  const glm::quat q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  const vec3 pos(pose.position.x, pose.position.y, pose.position.z);

  const mat4 rot = glm::mat4_cast(glm::inverse(q));

  return rot * glm::translate(mat4(1.0f), -pos);
}

bool VulkanApp::renderXrFrame(DrawFrameFunc& drawFrame) {
  if (!xrSessionRunning_) {
    return true;
  }

  const XrFrameWaitInfo frameWaitInfo = {.type = XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState frameState = {.type = XR_TYPE_FRAME_STATE};
  XR_ASSERT(xrWaitFrame(xrSession_, &frameWaitInfo, &frameState));

  // timing (always update to avoid delta spikes when shouldRender is false)
  const double now = glfwGetTime();
  const float deltaSeconds = static_cast<float>(now - xrLastTimeStamp_);
  xrLastTimeStamp_ = now;

  const XrFrameBeginInfo frameBeginInfo = {.type = XR_TYPE_FRAME_BEGIN_INFO};
  XR_ASSERT(xrBeginFrame(xrSession_, &frameBeginInfo));

  if (!frameState.shouldRender) {
    // drain the GPU queue so the runtime's xrEndFrame() compositing doesn't race with prior submissions
    ctx_->wait({});
    const XrFrameEndInfo frameEndInfo = {
        .type = XR_TYPE_FRAME_END_INFO,
        .displayTime = frameState.predictedDisplayTime,
        .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
    };
    XR_ASSERT(xrEndFrame(xrSession_, &frameEndInfo));
    return true;
  }

  const XrViewLocateInfo viewLocateInfo = {
      .type = XR_TYPE_VIEW_LOCATE_INFO,
      .viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
      .displayTime = frameState.predictedDisplayTime,
      .space = xrAppSpace_,
  };

  XrViewState viewState = {.type = XR_TYPE_VIEW_STATE};
  XrView xrViews[2] = {{.type = XR_TYPE_VIEW}, {.type = XR_TYPE_VIEW}};
  uint32_t numViews = 2;
  XR_ASSERT(xrLocateViews(xrSession_, &viewLocateInfo, &viewState, 2, &numViews, xrViews));

  // acquire all swapchain images
  uint32_t colorImageIndices[2] = {};
  uint32_t depthImageIndices[2] = {};
  bool hasDepth[2] = {};
  RenderView renderViews[2] = {};

  for (uint32_t eye = 0; eye != 2; eye++) {
    const XrSwapchainImageAcquireInfo acquireInfo = {.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XR_ASSERT(xrAcquireSwapchainImage(xrSwapchains_[eye].swapchain, &acquireInfo, &colorImageIndices[eye]));

    const XrSwapchainImageWaitInfo waitInfo = {
        .type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
        .timeout = XR_INFINITE_DURATION,
    };
    XR_ASSERT(xrWaitSwapchainImage(xrSwapchains_[eye].swapchain, &waitInfo));

    hasDepth[eye] = (xrDepthSwapchains_[eye].swapchain);
    if (hasDepth[eye]) {
      XR_ASSERT(xrAcquireSwapchainImage(xrDepthSwapchains_[eye].swapchain, &acquireInfo, &depthImageIndices[eye]));
      XR_ASSERT(xrWaitSwapchainImage(xrDepthSwapchains_[eye].swapchain, &waitInfo));
    }

    // populate render views
    const float w = (float)xrSwapchains_[eye].width;
    const float h = (float)xrSwapchains_[eye].height;
    renderViews[eye] = {
        .proj = xrCreateProjectionMatrix(xrViews[eye].fov, 0.1f, 100.0f),
        .view = xrCreateViewMatrix(xrViews[eye].pose),
        .viewport = {0.0f, 0.0f, w, h, 0.0f, 1.0f},
        .scissorRect = {0, 0, (uint32_t)w, (uint32_t)h},
        .colorTexture = xrSwapchains_[eye].textures[colorImageIndices[eye]],
        .depthTexture = hasDepth[eye] ? xrDepthSwapchains_[eye].textures[depthImageIndices[eye]] : lvk::TextureHandle{},
        .aspectRatio = w / h,
    };
  }

  // call the sample's draw function
  drawFrame(renderViews, deltaSeconds);

  // wait for GPU before releasing swapchain images
  ctx_->wait({});

  // release swapchain images and fill composition layer
  XrCompositionLayerProjectionView projectionViews[2] = {
      {.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
      {.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
  };

  for (uint32_t eye = 0; eye < 2; eye++) {
    const XrSwapchainImageReleaseInfo releaseInfo = {.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    XR_ASSERT(xrReleaseSwapchainImage(xrSwapchains_[eye].swapchain, &releaseInfo));
    if (hasDepth[eye]) {
      XR_ASSERT(xrReleaseSwapchainImage(xrDepthSwapchains_[eye].swapchain, &releaseInfo));
    }

    projectionViews[eye] = {
        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
        .pose = xrViews[eye].pose,
        .fov = xrViews[eye].fov,
        .subImage =
            {
                .swapchain = xrSwapchains_[eye].swapchain,
                .imageRect =
                    {
                        .offset = {0, 0},
                        .extent = {(int32_t)xrSwapchains_[eye].width, (int32_t)xrSwapchains_[eye].height},
                    },
            },
    };
  }

  const XrCompositionLayerProjection projectionLayer = {
      .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION,
      .space = xrAppSpace_,
      .viewCount = 2,
      .views = projectionViews,
  };
  const XrCompositionLayerBaseHeader* layers[] = {(XrCompositionLayerBaseHeader*)&projectionLayer};

  const XrFrameEndInfo frameEndInfo = {
      .type = XR_TYPE_FRAME_END_INFO,
      .displayTime = frameState.predictedDisplayTime,
      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
      .layerCount = 1,
      .layers = layers,
  };
  XR_ASSERT(xrEndFrame(xrSession_, &frameEndInfo));

  return true;
}

#endif // LVK_WITH_OPENXR
