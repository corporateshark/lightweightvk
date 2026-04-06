/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(LVK_WITH_OPENXR) || !LVK_WITH_OPENXR
#error OpenXR is not enabled (define LVK_WITH_OPENXR)
#endif // !defined(LVK_WITH_OPENXR) || !LVK_WITH_OPENXR

#include "VulkanClasses.h"

#if !defined(XR_USE_GRAPHICS_API_VULKAN)
#define XR_USE_GRAPHICS_API_VULKAN
#endif // !defined(XR_USE_GRAPHICS_API_VULKAN)
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#define XR_CHECK(func)                                                                                                              \
  {                                                                                                                                 \
    const XrResult xrResult_ = func;                                                                                                \
    if (XR_FAILED(xrResult_)) {                                                                                                     \
      LLOGW("OpenXR error: %s:%i\n  %s\n  %s (%d)\n", __FILE__, __LINE__, #func, lvk::xrResultToString(xrResult_), (int)xrResult_); \
      LVK_ASSERT_MSG(false, "OpenXR call failed");                                                                                  \
    }                                                                                                                               \
  }

namespace lvk {

const char* xrResultToString(XrResult result);
const char* xrSessionStateToString(XrSessionState state);

std::unique_ptr<lvk::IContext> createVulkanContextXR(XrInstance xrInstance,
                                                     XrSystemId xrSystemId,
                                                     PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDevice,
                                                     const lvk::ContextConfig& ctxCfg);
} // namespace lvk
