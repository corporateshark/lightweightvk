/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanUtils.h"

#include <igl/vulkan/Common.h>
#include <igl/vulkan/VulkanContext.h>

#include <shader/ShaderCompile.h>

lvk::Result lvk::getResultFromVkResult(VkResult result) {
  if (result == VK_SUCCESS) {
    return Result();
  }

  Result res(Result::Code::RuntimeError, ivkGetVulkanResultString(result));

  switch (result) {
  case VK_ERROR_OUT_OF_HOST_MEMORY:
  case VK_ERROR_OUT_OF_DEVICE_MEMORY:
  case VK_ERROR_OUT_OF_POOL_MEMORY:
  case VK_ERROR_TOO_MANY_OBJECTS:
    res.code = Result::Code::ArgumentOutOfRange;
    return res;
  default:;
    // skip other Vulkan error codes
  }
  return res;
}

VkFormat lvk::formatToVkFormat(lvk::Format format) {
  using TextureFormat = ::lvk::Format;
  switch (format) {
  case lvk::Format_Invalid:
    return VK_FORMAT_UNDEFINED;
  case lvk::Format_R_UN8:
    return VK_FORMAT_R8_UNORM;
  case lvk::Format_R_UN16:
    return VK_FORMAT_R16_UNORM;
  case lvk::Format_R_F16:
    return VK_FORMAT_R16_SFLOAT;
  case lvk::Format_R_UI16:
    return VK_FORMAT_R16_UINT;
  case lvk::Format_RG_UN8:
    return VK_FORMAT_R8G8_UNORM;
  case lvk::Format_RG_UN16:
    return VK_FORMAT_R16G16_UNORM;
  case lvk::Format_BGRA_UN8:
    return VK_FORMAT_B8G8R8A8_UNORM;
  case lvk::Format_RGBA_UN8:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case lvk::Format_RGBA_SRGB8:
    return VK_FORMAT_R8G8B8A8_SRGB;
  case lvk::Format_BGRA_SRGB8:
    return VK_FORMAT_B8G8R8A8_SRGB;
  case lvk::Format_RG_F16:
    return VK_FORMAT_R16G16_SFLOAT;
  case lvk::Format_RG_F32:
    return VK_FORMAT_R32G32_SFLOAT;
  case lvk::Format_RG_UI16:
    return VK_FORMAT_R16G16_UINT;
  case lvk::Format_R_F32:
    return VK_FORMAT_R32_SFLOAT;
  case lvk::Format_RGBA_F16:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case lvk::Format_RGBA_UI32:
    return VK_FORMAT_R32G32B32A32_UINT;
  case lvk::Format_RGBA_F32:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case lvk::Format_ETC2_RGB8:
    return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
  case lvk::Format_ETC2_SRGB8:
    return VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK;
  case lvk::Format_BC7_RGBA:
    return VK_FORMAT_BC7_UNORM_BLOCK;
  case lvk::Format_Z_UN16:
    return VK_FORMAT_D16_UNORM;
  case lvk::Format_Z_UN24:
    return VK_FORMAT_D24_UNORM_S8_UINT;
  case lvk::Format_Z_F32:
    return VK_FORMAT_D32_SFLOAT;
  case lvk::Format_Z_UN24_S_UI8:
    return VK_FORMAT_D24_UNORM_S8_UINT;
  }
#if defined(_MSC_VER)
  IGL_ASSERT_MSG(false, "TextureFormat value not handled: %d", (int)format);
  return VK_FORMAT_UNDEFINED;
#endif // _MSC_VER
}

lvk::Format lvk::vkFormatToFormat(VkFormat format) {
  switch (format) {
  case VK_FORMAT_UNDEFINED:
    return Format_Invalid;
  case VK_FORMAT_R8_UNORM:
    return Format_R_UN8;
  case VK_FORMAT_R16_UNORM:
    return Format_R_UN16;
  case VK_FORMAT_R16_SFLOAT:
    return Format_R_F16;
  case VK_FORMAT_R16_UINT:
    return Format_R_UI16;
  case VK_FORMAT_R8G8_UNORM:
    return Format_RG_UN8;
  case VK_FORMAT_B8G8R8A8_UNORM:
    return Format_BGRA_UN8;
  case VK_FORMAT_R8G8B8A8_UNORM:
    return Format_RGBA_UN8;
  case VK_FORMAT_R8G8B8A8_SRGB:
    return Format_RGBA_SRGB8;
  case VK_FORMAT_B8G8R8A8_SRGB:
    return Format_BGRA_SRGB8;
  case VK_FORMAT_R16G16_UNORM:
    return Format_RG_UN16;
  case VK_FORMAT_R16G16_SFLOAT:
    return Format_RG_F16;
  case VK_FORMAT_R32G32_SFLOAT:
    return Format_RG_F32;
  case VK_FORMAT_R16G16_UINT:
    return Format_RG_UI16;
  case VK_FORMAT_R32_SFLOAT:
    return Format_R_F32;
  case VK_FORMAT_R16G16B16A16_SFLOAT:
    return Format_RGBA_F16;
  case VK_FORMAT_R32G32B32A32_UINT:
    return Format_RGBA_UI32;
  case VK_FORMAT_R32G32B32A32_SFLOAT:
    return Format_RGBA_F32;
  case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
    return Format_ETC2_RGB8;
  case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
    return Format_ETC2_SRGB8;
  case VK_FORMAT_D16_UNORM:
    return Format_Z_UN16;
  case VK_FORMAT_BC7_UNORM_BLOCK:
    return Format_BC7_RGBA;
  case VK_FORMAT_X8_D24_UNORM_PACK32:
    return Format_Z_UN24;
  case VK_FORMAT_D24_UNORM_S8_UINT:
    return Format_Z_UN24_S_UI8;
  case VK_FORMAT_D32_SFLOAT:
    return Format_Z_F32;
  default:;
  }
  IGL_ASSERT_MSG(false, "VkFormat value not handled: %d", (int)format);
  return Format_Invalid;
}

VkSemaphore lvk::createSemaphore(VkDevice device, const char* debugName) {
  const VkSemaphoreCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .flags = 0,
  };
  VkSemaphore semaphore = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateSemaphore(device, &ci, nullptr, &semaphore));
  VK_ASSERT(
      ivkSetDebugObjectName(device, VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)semaphore, debugName));
  return semaphore;
}

VkFence lvk::createFence(VkDevice device, const char* debugName) {
  const VkFenceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .flags = 0,
  };
  VkFence fence = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateFence(device, &ci, nullptr, &fence));
  VK_ASSERT(ivkSetDebugObjectName(device, VK_OBJECT_TYPE_FENCE, (uint64_t)fence, debugName));
  return fence;
}

uint32_t lvk::findQueueFamilyIndex(VkPhysicalDevice physDev, VkQueueFlags flags) {
  using lvk::vulkan::DeviceQueues;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> props(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, props.data());

  auto findDedicatedQueueFamilyIndex = [&props](VkQueueFlags require,
                                                VkQueueFlags avoid) -> uint32_t {
    for (uint32_t i = 0; i != props.size(); i++) {
      const bool isSuitable = (props[i].queueFlags & require) == require;
      const bool isDedicated = (props[i].queueFlags & avoid) == 0;
      if (props[i].queueCount && isSuitable && isDedicated)
        return i;
    }
    return DeviceQueues::INVALID;
  };

  // dedicated queue for compute
  if (flags & VK_QUEUE_COMPUTE_BIT) {
    const uint32_t q = findDedicatedQueueFamilyIndex(flags, VK_QUEUE_GRAPHICS_BIT);
    if (q != DeviceQueues::INVALID)
      return q;
  }

  // dedicated queue for transfer
  if (flags & VK_QUEUE_TRANSFER_BIT) {
    const uint32_t q = findDedicatedQueueFamilyIndex(flags, VK_QUEUE_GRAPHICS_BIT);
    if (q != DeviceQueues::INVALID)
      return q;
  }

  // any suitable
  return findDedicatedQueueFamilyIndex(flags, 0);
}

VmaAllocator lvk::createVmaAllocator(VkPhysicalDevice physDev,
                                     VkDevice device,
                                     VkInstance instance,
                                     uint32_t apiVersion) {
  const VmaVulkanFunctions funcs = {
      .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
      .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
      .vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties,
      .vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties,
      .vkAllocateMemory = vkAllocateMemory,
      .vkFreeMemory = vkFreeMemory,
      .vkMapMemory = vkMapMemory,
      .vkUnmapMemory = vkUnmapMemory,
      .vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges,
      .vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges,
      .vkBindBufferMemory = vkBindBufferMemory,
      .vkBindImageMemory = vkBindImageMemory,
      .vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements,
      .vkGetImageMemoryRequirements = vkGetImageMemoryRequirements,
      .vkCreateBuffer = vkCreateBuffer,
      .vkDestroyBuffer = vkDestroyBuffer,
      .vkCreateImage = vkCreateImage,
      .vkDestroyImage = vkDestroyImage,
      .vkCmdCopyBuffer = vkCmdCopyBuffer,
      .vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2,
      .vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2,
      .vkBindBufferMemory2KHR = vkBindBufferMemory2,
      .vkBindImageMemory2KHR = vkBindImageMemory2,
      .vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2,
      .vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements,
      .vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements,
  };

  const VmaAllocatorCreateInfo ci = {
      .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice = physDev,
      .device = device,
      .preferredLargeHeapBlockSize = 0,
      .pAllocationCallbacks = nullptr,
      .pDeviceMemoryCallbacks = nullptr,
      .pHeapSizeLimit = nullptr,
      .pVulkanFunctions = &funcs,
      .instance = instance,
      .vulkanApiVersion = apiVersion,
  };
  VmaAllocator vma = VK_NULL_HANDLE;
  VK_ASSERT(vmaCreateAllocator(&ci, &vma));
  return vma;
}

namespace {

VkFilter samplerFilterToVkFilter(lvk::SamplerFilter filter) {
  switch (filter) {
  case lvk::SamplerFilter_Nearest:
    return VK_FILTER_NEAREST;
  case lvk::SamplerFilter_Linear:
    return VK_FILTER_LINEAR;
  }
  IGL_ASSERT_MSG(false, "SamplerFilter value not handled: %d", (int)filter);
  return VK_FILTER_LINEAR;
}

VkSamplerMipmapMode samplerMipMapToVkSamplerMipmapMode(lvk::SamplerMip filter) {
  switch (filter) {
  case lvk::SamplerMip_Disabled:
  case lvk::SamplerMip_Nearest:
    return VK_SAMPLER_MIPMAP_MODE_NEAREST;
  case lvk::SamplerMip_Linear:
    return VK_SAMPLER_MIPMAP_MODE_LINEAR;
  }
  IGL_ASSERT_MSG(false, "SamplerMipMap value not handled: %d", (int)filter);
  return VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode samplerWrapModeToVkSamplerAddressMode(lvk::SamplerWrap mode) {
  switch (mode) {
  case lvk::SamplerWrap_Repeat:
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  case lvk::SamplerWrap_Clamp:
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  case lvk::SamplerWrap_MirrorRepeat:
    return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  }
  IGL_ASSERT_MSG(false, "SamplerWrapMode value not handled: %d", (int)mode);
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

} // namespace 

VkSamplerCreateInfo lvk::samplerStateDescToVkSamplerCreateInfo(const lvk::SamplerStateDesc& desc, const VkPhysicalDeviceLimits& limits) {
  IGL_ASSERT_MSG(desc.mipLodMax >= desc.mipLodMin,
                 "mipLodMax (%d) must be greater than or equal to mipLodMin (%d)",
                 (int)desc.mipLodMax,
                 (int)desc.mipLodMin);

  VkSamplerCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = samplerFilterToVkFilter(desc.magFilter),
      .minFilter = samplerFilterToVkFilter(desc.minFilter),
      .mipmapMode = samplerMipMapToVkSamplerMipmapMode(desc.mipMap),
      .addressModeU = samplerWrapModeToVkSamplerAddressMode(desc.wrapU),
      .addressModeV = samplerWrapModeToVkSamplerAddressMode(desc.wrapV),
      .addressModeW = samplerWrapModeToVkSamplerAddressMode(desc.wrapW),
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 0.0f,
      .compareEnable = desc.depthCompareEnabled ? VK_TRUE : VK_FALSE,
      .compareOp = desc.depthCompareEnabled ? lvk::vulkan::compareOpToVkCompareOp(desc.depthCompareOp) : VK_COMPARE_OP_ALWAYS,
      .minLod = float(desc.mipLodMin),
      .maxLod = desc.mipMap == lvk::SamplerMip_Disabled ? 0.0f : float(desc.mipLodMax),
      .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
      .unnormalizedCoordinates = VK_FALSE,
  };

  if (desc.maxAnisotropic > 1) {
    const bool isAnisotropicFilteringSupported = limits.maxSamplerAnisotropy > 1;
    IGL_ASSERT_MSG(isAnisotropicFilteringSupported, "Anisotropic filtering is not supported by the device.");
    ci.anisotropyEnable = isAnisotropicFilteringSupported ? VK_TRUE : VK_FALSE;

    if (limits.maxSamplerAnisotropy < desc.maxAnisotropic) {
      LLOGL(
          "Supplied sampler anisotropic value greater than max supported by the device, setting to "
          "%.0f",
          static_cast<double>(limits.maxSamplerAnisotropy));
    }
    ci.maxAnisotropy = std::min((float)limits.maxSamplerAnisotropy, (float)desc.maxAnisotropic);
  }

  return ci;
}

lvk::Result lvk::compileShader(VkDevice device,
                               VkShaderStageFlagBits stage,
                               const char* code,
                               VkShaderModule* outShaderModule) {
  IGL_PROFILER_FUNCTION();

  if (!outShaderModule) {
    return Result(Result::Code::ArgumentOutOfRange, "outShaderModule is NULL");
  }

  std::vector<uint32_t> bytecode;
  lvk::Result res = lvk::shader::compile(code, stage, bytecode);

  // Even if compilation+linking is successful, we may have some warnings.
  if (*lvk::shader::getProcessingLog()) {
	  LLOGW(lvk::shader::getProcessingLog());
  }

  if (!res.isOk()) {
	  logShaderSource(code);
	  assert(false);
	  return Result();
  }

  const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytecode.size() * sizeof(uint32_t),
      .pCode = bytecode.data(),
  };
  VK_ASSERT_RETURN(vkCreateShaderModule(device, &ci, nullptr, outShaderModule));

  return Result();
}
