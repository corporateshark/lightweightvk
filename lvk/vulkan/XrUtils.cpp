/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "XrUtils.h"

// clang-format off
const char* lvk::xrResultToString(XrResult result) {
  switch (result) {
  case XR_SUCCESS:                                    return "XR_SUCCESS";
  case XR_TIMEOUT_EXPIRED:                            return "XR_TIMEOUT_EXPIRED";
  case XR_SESSION_LOSS_PENDING:                       return "XR_SESSION_LOSS_PENDING";
  case XR_EVENT_UNAVAILABLE:                          return "XR_EVENT_UNAVAILABLE";
  case XR_SPACE_BOUNDS_UNAVAILABLE:                   return "XR_SPACE_BOUNDS_UNAVAILABLE";
  case XR_SESSION_NOT_FOCUSED:                        return "XR_SESSION_NOT_FOCUSED";
  case XR_FRAME_DISCARDED:                            return "XR_FRAME_DISCARDED";
  case XR_ERROR_VALIDATION_FAILURE:                   return "XR_ERROR_VALIDATION_FAILURE";
  case XR_ERROR_RUNTIME_FAILURE:                      return "XR_ERROR_RUNTIME_FAILURE";
  case XR_ERROR_OUT_OF_MEMORY:                        return "XR_ERROR_OUT_OF_MEMORY";
  case XR_ERROR_API_VERSION_UNSUPPORTED:              return "XR_ERROR_API_VERSION_UNSUPPORTED";
  case XR_ERROR_INITIALIZATION_FAILED:                return "XR_ERROR_INITIALIZATION_FAILED";
  case XR_ERROR_FUNCTION_UNSUPPORTED:                 return "XR_ERROR_FUNCTION_UNSUPPORTED";
  case XR_ERROR_FEATURE_UNSUPPORTED:                  return "XR_ERROR_FEATURE_UNSUPPORTED";
  case XR_ERROR_EXTENSION_NOT_PRESENT:                return "XR_ERROR_EXTENSION_NOT_PRESENT";
  case XR_ERROR_LIMIT_REACHED:                        return "XR_ERROR_LIMIT_REACHED";
  case XR_ERROR_SIZE_INSUFFICIENT:                    return "XR_ERROR_SIZE_INSUFFICIENT";
  case XR_ERROR_HANDLE_INVALID:                       return "XR_ERROR_HANDLE_INVALID";
  case XR_ERROR_INSTANCE_LOST:                        return "XR_ERROR_INSTANCE_LOST";
  case XR_ERROR_SESSION_RUNNING:                      return "XR_ERROR_SESSION_RUNNING";
  case XR_ERROR_SESSION_NOT_RUNNING:                  return "XR_ERROR_SESSION_NOT_RUNNING";
  case XR_ERROR_SESSION_LOST:                         return "XR_ERROR_SESSION_LOST";
  case XR_ERROR_SYSTEM_INVALID:                       return "XR_ERROR_SYSTEM_INVALID";
  case XR_ERROR_PATH_INVALID:                         return "XR_ERROR_PATH_INVALID";
  case XR_ERROR_PATH_COUNT_EXCEEDED:                  return "XR_ERROR_PATH_COUNT_EXCEEDED";
  case XR_ERROR_PATH_FORMAT_INVALID:                  return "XR_ERROR_PATH_FORMAT_INVALID";
  case XR_ERROR_PATH_UNSUPPORTED:                     return "XR_ERROR_PATH_UNSUPPORTED";
  case XR_ERROR_LAYER_INVALID:                        return "XR_ERROR_LAYER_INVALID";
  case XR_ERROR_LAYER_LIMIT_EXCEEDED:                 return "XR_ERROR_LAYER_LIMIT_EXCEEDED";
  case XR_ERROR_SWAPCHAIN_RECT_INVALID:               return "XR_ERROR_SWAPCHAIN_RECT_INVALID";
  case XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED:         return "XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED";
  case XR_ERROR_ACTION_TYPE_MISMATCH:                 return "XR_ERROR_ACTION_TYPE_MISMATCH";
  case XR_ERROR_SESSION_NOT_READY:                    return "XR_ERROR_SESSION_NOT_READY";
  case XR_ERROR_SESSION_NOT_STOPPING:                 return "XR_ERROR_SESSION_NOT_STOPPING";
  case XR_ERROR_TIME_INVALID:                         return "XR_ERROR_TIME_INVALID";
  case XR_ERROR_REFERENCE_SPACE_UNSUPPORTED:          return "XR_ERROR_REFERENCE_SPACE_UNSUPPORTED";
  case XR_ERROR_FILE_ACCESS_ERROR:                    return "XR_ERROR_FILE_ACCESS_ERROR";
  case XR_ERROR_FILE_CONTENTS_INVALID:                return "XR_ERROR_FILE_CONTENTS_INVALID";
  case XR_ERROR_FORM_FACTOR_UNSUPPORTED:              return "XR_ERROR_FORM_FACTOR_UNSUPPORTED";
  case XR_ERROR_FORM_FACTOR_UNAVAILABLE:              return "XR_ERROR_FORM_FACTOR_UNAVAILABLE";
  case XR_ERROR_API_LAYER_NOT_PRESENT:                return "XR_ERROR_API_LAYER_NOT_PRESENT";
  case XR_ERROR_CALL_ORDER_INVALID:                   return "XR_ERROR_CALL_ORDER_INVALID";
  case XR_ERROR_GRAPHICS_DEVICE_INVALID:              return "XR_ERROR_GRAPHICS_DEVICE_INVALID";
  case XR_ERROR_POSE_INVALID:                         return "XR_ERROR_POSE_INVALID";
  case XR_ERROR_INDEX_OUT_OF_RANGE:                   return "XR_ERROR_INDEX_OUT_OF_RANGE";
  case XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED:  return "XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED";
  case XR_ERROR_ENVIRONMENT_BLEND_MODE_UNSUPPORTED:   return "XR_ERROR_ENVIRONMENT_BLEND_MODE_UNSUPPORTED";
  case XR_ERROR_NAME_DUPLICATED:                      return "XR_ERROR_NAME_DUPLICATED";
  case XR_ERROR_NAME_INVALID:                         return "XR_ERROR_NAME_INVALID";
  case XR_ERROR_ACTIONSET_NOT_ATTACHED:               return "XR_ERROR_ACTIONSET_NOT_ATTACHED";
  case XR_ERROR_ACTIONSETS_ALREADY_ATTACHED:          return "XR_ERROR_ACTIONSETS_ALREADY_ATTACHED";
  case XR_ERROR_LOCALIZED_NAME_DUPLICATED:            return "XR_ERROR_LOCALIZED_NAME_DUPLICATED";
  case XR_ERROR_LOCALIZED_NAME_INVALID:               return "XR_ERROR_LOCALIZED_NAME_INVALID";
  case XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING:   return "XR_ERROR_GRAPHICS_REQUIREMENTS_CALL_MISSING";
  case XR_ERROR_RUNTIME_UNAVAILABLE:                  return "XR_ERROR_RUNTIME_UNAVAILABLE";
  default:                                            return "XR_UNKNOWN";
  }
}
// clang-format on

// clang-format off
const char* lvk::xrSessionStateToString(XrSessionState state) {
  switch (state) {
  case XR_SESSION_STATE_UNKNOWN:      return "UNKNOWN";
  case XR_SESSION_STATE_IDLE:         return "IDLE";
  case XR_SESSION_STATE_READY:        return "READY";
  case XR_SESSION_STATE_SYNCHRONIZED: return "SYNCHRONIZED";
  case XR_SESSION_STATE_VISIBLE:      return "VISIBLE";
  case XR_SESSION_STATE_FOCUSED:      return "FOCUSED";
  case XR_SESSION_STATE_STOPPING:     return "STOPPING";
  case XR_SESSION_STATE_LOSS_PENDING: return "LOSS_PENDING";
  case XR_SESSION_STATE_EXITING:      return "EXITING";
  default:                            return "INVALID";
  }
}
// clang-format on

std::unique_ptr<lvk::IContext> lvk::createVulkanContextXR(XrInstance xrInstance,
                                                          XrSystemId xrSystemId,
                                                          PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr,
                                                          const lvk::ContextConfig& ctxCfg) {
  LVK_ASSERT(xrGetInstanceProcAddr);

  PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirements = nullptr;
  PFN_xrGetVulkanGraphicsDeviceKHR xrGetVulkanGraphicsDevice = nullptr;

  XR_ASSERT(xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsRequirementsKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsRequirements));
  XR_ASSERT(xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsDeviceKHR", (PFN_xrVoidFunction*)&xrGetVulkanGraphicsDevice));

  LVK_ASSERT(xrGetVulkanGraphicsRequirements);
  LVK_ASSERT(xrGetVulkanGraphicsDevice);

  // get Vulkan graphics requirements (must be called before creating Vulkan device per spec)
  XrGraphicsRequirementsVulkanKHR graphicsReqs = {.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
  XR_ASSERT(xrGetVulkanGraphicsRequirements(xrInstance, xrSystemId, &graphicsReqs));

  LVK_ASSERT_MSG(graphicsReqs.minApiVersionSupported <= XR_MAKE_VERSION(1, 3, 0), "Unsupported Vulkan version required by OpenXR runtime");

  std::unique_ptr<lvk::VulkanContext> ctx = std::make_unique<lvk::VulkanContext>(ctxCfg, nullptr);

  VkPhysicalDevice devices[16];
  const uint32_t numDevices = ctx->queryDevices(devices, nullptr, 16);
  LVK_ASSERT_MSG(numDevices > 0, "No GPU found");

  // get the physical device OpenXR wants
  VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
  XR_ASSERT(xrGetVulkanGraphicsDevice(xrInstance, xrSystemId, ctx->getVkInstance(), &vkPhysicalDevice));

  // find the matching device index
  const uint32_t selectedDevice = [&devices, numDevices, vkPhysicalDevice]() -> uint32_t {
    for (uint32_t i = 0; i != numDevices; i++) {
      if (devices[i] == vkPhysicalDevice) {
        return i;
      }
    }
    return 0;
  }();

  if (selectedDevice >= numDevices) {
    LVK_ASSERT_MSG(false, "Invalid device index");
    return nullptr;
  }

  const lvk::Result res = ctx->initContext(devices[selectedDevice]);
  LVK_ASSERT(res.isOk());

  if (ctx->getVkPhysicalDevice() != vkPhysicalDevice) {
    LLOGW("LVK selected a different physical device than OpenXR requested\n");
  }

  return ctx;
}
