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

using DrawFrameFunc = std::function<void(uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds)>;

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
#else
  GLFWwindow* window_ = nullptr;
#endif // ANDROID
  std::unique_ptr<lvk::IContext> ctx_;
  mutable lvk::Holder<lvk::TextureHandle> depthTexture_;
  FramesPerSecondCounter fpsCounter_ = FramesPerSecondCounter(0.5f);
  std::unique_ptr<lvk::ImGuiRenderer> imgui_;

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
};
