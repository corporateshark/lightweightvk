/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "LVK.h"

#include <assert.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#if LVK_WITH_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// clang-format off
#ifdef _WIN32
#  define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__linux__)
#  if defined(LVK_WITH_WAYLAND)
#    define GLFW_EXPOSE_NATIVE_WAYLAND
#  else
#    define GLFW_EXPOSE_NATIVE_X11
#  endif
#elif __APPLE__
#  define GLFW_EXPOSE_NATIVE_COCOA
#else
#  error Unsupported OS
#endif
// clang-format on

#include <GLFW/glfw3native.h>
#endif // LVK_WITH_GLFW

#include <lvk/vulkan/VulkanClasses.h>

namespace {

struct TextureFormatProperties {
  const lvk::Format format = lvk::Format_Invalid;
  const uint8_t bytesPerBlock : 5 = 1;
  const uint8_t blockWidth : 3 = 1;
  const uint8_t blockHeight : 3 = 1;
  const uint8_t minBlocksX : 2 = 1;
  const uint8_t minBlocksY : 2 = 1;
  const bool depth : 1 = false;
  const bool stencil : 1 = false;
  const bool compressed : 1 = false;
  const uint8_t numPlanes : 2 = 1;
};

#define PROPS(fmt, bpb, ...) \
  TextureFormatProperties { .format = lvk::Format_##fmt, .bytesPerBlock = bpb, ##__VA_ARGS__ }

static constexpr TextureFormatProperties properties[] = {
    PROPS(Invalid, 1),
    PROPS(R_UN8, 1),
    PROPS(R_UI16, 2),
    PROPS(R_UI32, 4),
    PROPS(R_UN16, 2),
    PROPS(R_F16, 2),
    PROPS(R_F32, 4),
    PROPS(RG_UN8, 2),
    PROPS(RG_UI16, 4),
    PROPS(RG_UI32, 8),
    PROPS(RG_UN16, 4),
    PROPS(RG_F16, 4),
    PROPS(RG_F32, 8),
    PROPS(RGBA_UN8, 4),
    PROPS(RGBA_UI32, 16),
    PROPS(RGBA_F16, 8),
    PROPS(RGBA_F32, 16),
    PROPS(RGBA_SRGB8, 4),
    PROPS(BGRA_UN8, 4),
    PROPS(BGRA_SRGB8, 4),
    PROPS(A2B10G10R10_UN, 4),
    PROPS(A2R10G10B10_UN, 4),
    PROPS(ETC2_RGB8, 8, .blockWidth = 4, .blockHeight = 4, .compressed = true),
    PROPS(ETC2_SRGB8, 8, .blockWidth = 4, .blockHeight = 4, .compressed = true),
    PROPS(BC7_RGBA, 16, .blockWidth = 4, .blockHeight = 4, .compressed = true),
    PROPS(Z_UN16, 2, .depth = true),
    PROPS(Z_UN24, 3, .depth = true),
    PROPS(Z_F32, 4, .depth = true),
    PROPS(Z_UN24_S_UI8, 4, .depth = true, .stencil = true),
    PROPS(Z_F32_S_UI8, 5, .depth = true, .stencil = true),
    PROPS(YUV_NV12, 24, .blockWidth = 4, .blockHeight = 4, .compressed = true, .numPlanes = 2), // Subsampled 420
    PROPS(YUV_420p, 24, .blockWidth = 4, .blockHeight = 4, .compressed = true, .numPlanes = 3), // Subsampled 420
};

} // namespace

#if __APPLE__ && LVK_WITH_GLFW
void* createCocoaWindowView(GLFWwindow* window);
#endif

static_assert(sizeof(TextureFormatProperties) <= sizeof(uint32_t));
static_assert(LVK_ARRAY_NUM_ELEMENTS(properties) == lvk::Format_YUV_420p + 1);

bool lvk::isDepthOrStencilFormat(lvk::Format format) {
  return properties[format].depth || properties[format].stencil;
}

uint32_t lvk::getNumImagePlanes(lvk::Format format) {
  return properties[format].numPlanes;
}

uint32_t lvk::getVertexFormatSize(lvk::VertexFormat format) {
  // clang-format off
#define SIZE4(LVKBaseType, BaseType)           \
  case VertexFormat::LVKBaseType##1: return sizeof(BaseType) * 1u; \
  case VertexFormat::LVKBaseType##2: return sizeof(BaseType) * 2u; \
  case VertexFormat::LVKBaseType##3: return sizeof(BaseType) * 3u; \
  case VertexFormat::LVKBaseType##4: return sizeof(BaseType) * 4u;
#define SIZE2_4_NORM(LVKBaseType, BaseType)           \
  case VertexFormat::LVKBaseType##2Norm: return sizeof(BaseType) * 2u; \
  case VertexFormat::LVKBaseType##4Norm: return sizeof(BaseType) * 4u;

  // clang-format on

  switch (format) {
    SIZE4(Float, float);
    SIZE4(Byte, uint8_t);
    SIZE4(UByte, uint8_t);
    SIZE4(Short, uint16_t);
    SIZE4(UShort, uint16_t);
    SIZE2_4_NORM(Byte, uint8_t);
    SIZE2_4_NORM(UByte, uint8_t);
    SIZE2_4_NORM(Short, uint16_t);
    SIZE2_4_NORM(UShort, uint16_t);
    SIZE4(Int, uint32_t);
    SIZE4(UInt, uint32_t);
    SIZE4(HalfFloat, uint16_t);
  case VertexFormat::Int_2_10_10_10_REV:
    return sizeof(uint32_t);
  default:
    assert(false);
    return 0;
  }
#undef SIZE4
#undef SIZE2_4_NORM
}

uint32_t lvk::getTextureBytesPerLayer(uint32_t width, uint32_t height, lvk::Format format, uint32_t level) {
  const uint32_t levelWidth = std::max(width >> level, 1u);
  const uint32_t levelHeight = std::max(height >> level, 1u);

  const TextureFormatProperties props = properties[format];

  if (!props.compressed) {
    return props.bytesPerBlock * levelWidth * levelHeight;
  }

  const uint32_t blockWidth = std::max((uint32_t)props.blockWidth, 1u);
  const uint32_t blockHeight = std::max((uint32_t)props.blockHeight, 1u);
  const uint32_t widthInBlocks = (levelWidth + props.blockWidth - 1) / props.blockWidth;
  const uint32_t heightInBlocks = (levelHeight + props.blockHeight - 1) / props.blockHeight;
  return widthInBlocks * heightInBlocks * props.bytesPerBlock;
}

uint32_t lvk::getTextureBytesPerPlane(uint32_t width, uint32_t height, lvk::Format format, uint32_t plane) {
  const TextureFormatProperties props = properties[format];

  LVK_ASSERT(plane < props.numPlanes);

  switch (format) {
  case Format_YUV_NV12:
    return width * height / (plane + 1);
  case Format_YUV_420p:
    return width * height / (plane ? 4 : 1);
  default:;
  }

  return getTextureBytesPerLayer(width, height, format, 0);
}

bool lvk::Assert(bool cond, const char* file, int line, const char* format, ...) {
  if (!cond) {
    va_list ap;
    va_start(ap, format);
    LLOGW("[LVK] Assertion failed in %s:%d: ", file, line);
    MINILOG_LOG_PROC(minilog::Warning, format, ap);
    LLOGW("\n");
    va_end(ap);
    assert(false);
  }
  return cond;
}

void lvk::destroy(lvk::IContext* ctx, lvk::ComputePipelineHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::RenderPipelineHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::RayTracingPipelineHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::ShaderModuleHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::SamplerHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::BufferHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::TextureHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::QueryPoolHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

void lvk::destroy(lvk::IContext* ctx, lvk::AccelStructHandle handle) {
  if (ctx) {
    ctx->destroy(handle);
  }
}

// Logs GLSL shaders with line numbers annotation
void lvk::logShaderSource(const char* text) {
  uint32_t line = 0;
  uint32_t numChars = 0;
  const char* lineStart = text;

  while (text && *text) {
    if (*text == '\n') {
      if (numChars) {
#if defined(MINILOG_RAW_OUTPUT)
        LLOGL("(%3u) %.*s\n", ++line, numChars, lineStart);
#else
        LLOGL("(%3u) %.*s", ++line, numChars, lineStart);
#endif // MINILOG_RAW_OUTPUT
      } else {
#if defined(MINILOG_RAW_OUTPUT)
        LLOGL("(%3u)\n", ++line);
#else
        LLOGL("(%3u)", ++line);
#endif // MINILOG_RAW_OUTPUT
      }
      numChars = 0;
      lineStart = text + 1;
    } else if (*text == '\r') {
      // skip it to support Windows/UNIX EOLs
      numChars = 0;
      lineStart = text + 1;
    } else {
      numChars++;
    }
    text++;
  }
  if (numChars) {
#if defined(MINILOG_RAW_OUTPUT)
    LLOGL("(%3u) %.*s\n", ++line, numChars, lineStart);
#else
    LLOGL("(%3u) %.*s", ++line, numChars, lineStart);
#endif // MINILOG_RAW_OUTPUT
  }
  LLOGL("\n");
}

uint32_t lvk::VertexInput::getVertexSize() const {
  uint32_t vertexSize = 0;
  for (uint32_t i = 0; i < LVK_VERTEX_ATTRIBUTES_MAX && attributes[i].format != VertexFormat::Invalid; i++) {
    LVK_ASSERT_MSG(attributes[i].offset == vertexSize, "Unsupported vertex attributes format");
    vertexSize += lvk::getVertexFormatSize(attributes[i].format);
  }
  return vertexSize;
}

#if LVK_WITH_GLFW
GLFWwindow* lvk::initWindow(const char* windowTitle, int& outWidth, int& outHeight, bool resizable) {
  if (!glfwInit()) {
    return nullptr;
  }

  const bool wantsWholeArea = outWidth <= 0 || outHeight <= 0;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, wantsWholeArea || !resizable ? GLFW_FALSE : GLFW_TRUE);

  // render full screen without overlapping taskbar
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();

  int x = 0;
  int y = 0;
  int w = outWidth;
  int h = outHeight;

  if (wantsWholeArea) {
    int areaW = 0;
    int areaH = 0;
    glfwGetMonitorWorkarea(monitor, &x, &y, &areaW, &areaH);
    auto getPercent = [](int value, int percent) {
      assert(percent > 0 && percent <= 100);
      return static_cast<int>(static_cast<float>(value) * static_cast<float>(percent) / 100.0f);
    };
    if (outWidth < 0) {
      w = getPercent(areaW, -outWidth);
      x = (areaW - w) / 2;
    } else {
      w = areaW;
    }
    if (outHeight < 0) {
      h = getPercent(areaH, -outHeight);
      y = (areaH - h) / 2;
    } else {
      h = areaH;
    }
  }

  GLFWwindow* window = glfwCreateWindow(w, h, windowTitle, nullptr, nullptr);

  if (!window) {
    glfwTerminate();
    return nullptr;
  }

  if (wantsWholeArea) {
    glfwSetWindowPos(window, x, y);
  }

  glfwGetWindowSize(window, &outWidth, &outHeight);

  glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
  });

  glfwSetErrorCallback([](int error, const char* description) { printf("GLFW Error (%i): %s\n", error, description); });

  return window;
}
#endif // LVK_WITH_GLFW

#if LVK_WITH_GLFW || defined(ANDROID)
std::unique_ptr<lvk::IContext> lvk::createVulkanContextWithSwapchain(LVKwindow* window,
                                                                     uint32_t width,
                                                                     uint32_t height,
                                                                     const lvk::ContextConfig& cfg,
                                                                     lvk::HWDeviceType preferredDeviceType,
                                                                     int selectedDevice) {
  using namespace lvk;

  std::unique_ptr<VulkanContext> ctx;

#if defined(_WIN32)
  ctx = std::make_unique<VulkanContext>(cfg, (void*)glfwGetWin32Window(window));
#elif defined(ANDROID)
  ctx = std::make_unique<VulkanContext>(cfg, (void*)window);
#elif defined(__linux__)
#if defined(LVK_WITH_WAYLAND)
  wl_surface* waylandWindow = glfwGetWaylandWindow(window);
  if (!waylandWindow) {
    LVK_ASSERT_MSG(false, "Wayland window not found");
    return nullptr;
  }
  ctx = std::make_unique<VulkanContext>(cfg, (void*)waylandWindow, (void*)glfwGetWaylandDisplay());
#else
  ctx = std::make_unique<VulkanContext>(cfg, (void*)glfwGetX11Window(window), (void*)glfwGetX11Display());
#endif
#elif defined(__APPLE__)
  ctx = std::make_unique<VulkanContext>(cfg, createCocoaWindowView(window));
#else
#error Unsupported OS
#endif

  HWDeviceDesc devices[16];
  const uint32_t numDevices = ctx->queryDevices(devices, LVK_ARRAY_NUM_ELEMENTS(devices));

  if (!numDevices) {
    LVK_ASSERT_MSG(false, "GPU is not found");
    return nullptr;
  }

  if (selectedDevice < 0) {
    selectedDevice = [preferredDeviceType, &devices, numDevices]() -> int {
      // define device type priority order
      HWDeviceType priority[4] = {preferredDeviceType};
      {
        int index = 1;
        for (int type = HWDeviceType_Integrated; type <= HWDeviceType_Software; type++) {
          if (type != preferredDeviceType)
            priority[index++] = (HWDeviceType)type;
        }
      }
      // search devices in priority order
      for (HWDeviceType type : priority) {
        for (uint32_t i = 0; i < numDevices; i++) {
          if (devices[i].type == type)
            return (int)i;
        }
      }
      return 0;
    }();
  }

  if (selectedDevice >= numDevices) {
    LVK_ASSERT_MSG(false, "Invalid device index");
    return nullptr;
  }

  Result res = ctx->initContext(devices[selectedDevice]);

  if (!res.isOk()) {
    LVK_ASSERT_MSG(false, "createVulkanContextWithSwapchain() failed");
    return nullptr;
  }

  if (width > 0 && height > 0) {
    res = ctx->initSwapchain(width, height);
    if (!res.isOk()) {
      LVK_ASSERT_MSG(false, "initSwapchain() failed");
      return nullptr;
    }
  }

  return std::move(ctx);
}
#endif // LVK_WITH_GLFW || defined(ANDROID)
