/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Based on https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition/blob/main/shared/VulkanApp.h

#pragma once

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cmath>

#include <functional>

#include <lvk/HelpersImGui.h>
#include <lvk/LVK.h>

#if LVK_WITH_OPENXR
#include <lvk/vulkan/XrUtils.h>
#endif // LVK_WITH_OPENXR

// clang-format off
#if defined(ANDROID)
#  include <android_native_app_glue.h>
#  include <jni.h>
double glfwGetTime();
#else
#  include <GLFW/glfw3.h>
#endif
// clang-format on

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <shared/Camera.h>
#include <shared/UtilsFPS.h>

// clang-format off
#if defined(ANDROID)
#  define VULKAN_APP_MAIN void android_main(android_app* androidApp)
#  define VULKAN_APP_DECLARE(app, config) VulkanApp app(androidApp, config)
#  define VULKAN_APP_EXIT() return
#else
#  define VULKAN_APP_MAIN int main(int argc, char* argv[])
#  define VULKAN_APP_DECLARE(app, config) VulkanApp app(argc, argv, config)
#  define VULKAN_APP_EXIT() return 0
#endif
// clang-format on

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

struct RenderView {
  mat4 proj; // OpenXR projection
  mat4 view; // OpenXR view (head space to eye space)
  lvk::Viewport viewport;
  lvk::ScissorRect scissorRect;
  lvk::TextureHandle colorTexture;
  lvk::TextureHandle depthTexture;
  float aspectRatio = 1.0f;
};

using DrawFrameFunc = std::function<void(lvk::Span<const RenderView> views, float deltaSeconds)>;

struct VulkanAppConfig {
  int width = -95; // 95% horizontally
  int height = -90; // 90% vertically
#if defined(ANDROID)
  int framebufferScalar = 2;
#else
  int framebufferScalar = 1;
#endif // ANDROID
  bool resizable = false;
  vec3 initialCameraPos = vec3(0.0f, 0.0f, -2.5f);
  vec3 initialCameraTarget = vec3(0.0f, 0.0f, 0.0f);
  vec3 initialCameraUpVector = vec3(0.0f, 1.0f, 0.0f);
  uint64_t screenshotFrameNumber = 0; // frames start from 1
  const char* screenshotFileName = "screenshot.png";
  lvk::ContextConfig contextConfig;
#if LVK_WITH_OPENXR
  bool enableOpenXR = false;
#endif // LVK_WITH_OPENXR
};

class VulkanApp {
 public:
#if defined(ANDROID)
  explicit VulkanApp(android_app* androidApp, const VulkanAppConfig& cfg = {});
#else
  explicit VulkanApp(int argc, char* argv[], const VulkanAppConfig& cfg = {});
#endif // ANDROID
  virtual ~VulkanApp();

  virtual void run(DrawFrameFunc drawFrame);
  virtual void drawFPS();

  double getSimulatedTime() const;

  lvk::Format getDepthFormat() const;
  lvk::TextureHandle getDepthTexture() const;
#if !defined(ANDROID)
  void addMouseButtonCallback(GLFWmousebuttonfun cb) {
    callbacksMouseButton.push_back(cb);
  }
  void addKeyCallback(GLFWkeyfun cb) {
    callbacksKey.push_back(cb);
  }
#endif // ANDROID
#if LVK_WITH_OPENXR
  lvk::TextureHandle xrSwapchainTexture(uint32_t eye, uint32_t imageIndex) const {
    LVK_ASSERT(eye < 2 && imageIndex < xrSwapchains_[eye].textures.size());
    return xrSwapchains_[eye].textures[imageIndex];
  }
  lvk::TextureHandle xrDepthSwapchainTexture(uint32_t eye, uint32_t imageIndex) const {
    LVK_ASSERT(eye < 2 && imageIndex < xrDepthSwapchains_[eye].textures.size());
    return xrDepthSwapchains_[eye].textures[imageIndex];
  }
#endif // LVK_WITH_OPENXR
 public:
  std::string folderThirdParty_;
  std::string folderContentRoot_;
  int width_ = 0;
  int height_ = 0;
#if defined(ANDROID)
  android_app* androidApp_ = nullptr;
  bool imguiCapturedTouch_ = false;
  bool imguiLastPressedLeft_ = false;
  bool imguiClearMouseNextFrame_ = false;
#endif // ANDROID
  std::unique_ptr<lvk::IContext> ctx_;
  mutable lvk::Holder<lvk::TextureHandle> depthTexture_;
  FramesPerSecondCounter fpsCounter_ = FramesPerSecondCounter(0.5f);
  std::unique_ptr<lvk::ImGuiRenderer> imgui_;

  lvk::LVKwindow* window_ = nullptr; // when declared before imgui_, Vivo X200 Pro crashes...

  VulkanAppConfig cfg_ = {};

  CameraPositioner_FirstPerson positioner_ = {cfg_.initialCameraPos, cfg_.initialCameraTarget, cfg_.initialCameraUpVector};
  Camera camera_ = Camera(positioner_);

  struct MouseState {
    vec2 pos = vec2(0.0f);
    bool pressedLeft = false;
  } mouseState_;

 protected:
#if !defined(ANDROID)
  std::vector<GLFWmousebuttonfun> callbacksMouseButton;
  std::vector<GLFWkeyfun> callbacksKey;
#endif // ANDROID

  uint64_t frameCount_ = 0;
  double simulatedTime_ = 0.0;

#if LVK_WITH_OPENXR
  void initOpenXR();
  void initXrSession();
  void initXrSwapchains();
  void destroyXrSwapchains();
  void pollXrEvents();
  bool renderXrFrame(DrawFrameFunc& drawFrame);

  static mat4 xrCreateProjectionMatrix(const XrFovf& fov, float nearZ, float farZ);
  static mat4 xrCreateViewMatrix(const XrPosef& pose);

  // OpenXR state
  XrInstance xrInstance_ = XR_NULL_HANDLE;
  XrSystemId xrSystemId_ = XR_NULL_SYSTEM_ID;
  XrSession xrSession_ = XR_NULL_HANDLE;
  XrSpace xrAppSpace_ = XR_NULL_HANDLE;
  XrSessionState xrSessionState_ = XR_SESSION_STATE_UNKNOWN;
  bool xrSessionRunning_ = false;
  bool xrShouldQuit_ = false;
  double xrLastTimeStamp_ = 0;

  // extension strings storage (must outlive ContextConfig)
  std::string xrVulkanExtsInstance_;
  std::string xrVulkanExtsDevice_;

  // per-eye swapchain data
  struct XrSwapchainData {
    XrSwapchain swapchain = XR_NULL_HANDLE;
    int64_t format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<XrSwapchainImageVulkanKHR> images;
    std::vector<lvk::TextureHandle> textures;
  };
  XrSwapchainData xrSwapchains_[2] = {}; // left and right eye

  struct XrDepthSwapchainData {
    XrSwapchain swapchain = XR_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<XrSwapchainImageVulkanKHR> images;
    std::vector<lvk::TextureHandle> textures;
  };
  XrDepthSwapchainData xrDepthSwapchains_[2] = {};

  std::vector<XrViewConfigurationView> xrConfigViews_;
#endif // LVK_WITH_OPENXR
};
