/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define VMA_VULKAN_VERSION 1003000
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

// set to 1 to see very verbose debug console logs with Vulkan commands
#define LVK_VULKAN_PRINT_COMMANDS 0

#if !defined(VK_NO_PROTOTYPES)
#define VK_NO_PROTOTYPES 1
#endif // !defined(VK_NO_PROTOTYPES)

// enable to use VulkanMemoryAllocator (VMA)
#define LVK_VULKAN_USE_VMA 1

#include <cassert>
#include <cstdio>
#include <vector>

#include <volk.h>
#include <vk_mem_alloc.h>
#include <lvk/LVK.h>

#define VK_ASSERT(func)                                            \
  {                                                                \
    const VkResult vk_assert_result = func;                        \
    if (vk_assert_result != VK_SUCCESS) {                          \
      LLOGW("Vulkan API call failed: %s:%i\n  %s\n  %s\n", \
                    __FILE__,                                      \
                    __LINE__,                                      \
                    #func,                                         \
                    lvk::getVulkanResultString(vk_assert_result)); \
      assert(false);                                               \
    }                                                              \
  }

#define VK_ASSERT_RETURN(func)                                     \
  {                                                                \
    const VkResult vk_assert_result = func;                        \
    if (vk_assert_result != VK_SUCCESS) {                          \
      LLOGW("Vulkan API call failed: %s:%i\n  %s\n  %s\n", \
                    __FILE__,                                      \
                    __LINE__,                                      \
                    #func,                                         \
                    lvk::getVulkanResultString(vk_assert_result)); \
      assert(false);                                               \
      return getResultFromVkResult(vk_assert_result);              \
    }                                                              \
  }

typedef struct glslang_resource_s glslang_resource_t;

struct StageAccess {
  VkPipelineStageFlags2 stage;
  VkAccessFlags2 access;
};

namespace lvk {

VkSemaphore createSemaphore(VkDevice device, const char* debugName);
VkSemaphore createSemaphoreTimeline(VkDevice device, uint64_t initialValue, const char* debugName);
VkFence createFence(VkDevice device, const char* debugName);
VmaAllocator createVmaAllocator(VkPhysicalDevice physDev, VkDevice device, VkInstance instance, uint32_t apiVersion);
uint32_t findQueueFamilyIndex(VkPhysicalDevice physDev, VkQueueFlags flags);
VkResult setDebugObjectName(VkDevice device, VkObjectType type, uint64_t handle, const char* name);
VkResult allocateMemory(VkPhysicalDevice physDev,
                        VkDevice device,
                        const VkMemoryRequirements* memRequirements,
                        VkMemoryPropertyFlags props,
                        VkDeviceMemory* outMemory);
VkResult allocateMemory2(VkPhysicalDevice physDev,
                         VkDevice device,
                         const VkMemoryRequirements2* memRequirements,
                         VkMemoryPropertyFlags props,
                         VkDeviceMemory* outMemory);

glslang_resource_t getGlslangResource(const VkPhysicalDeviceLimits& limits);
Result compileShader(VkShaderStageFlagBits stage,
                     const char* code,
                     std::vector<uint8_t>* outSPIRV,
                     const glslang_resource_t* glslLangResource = nullptr);

VkSamplerCreateInfo samplerStateDescToVkSamplerCreateInfo(const lvk::SamplerStateDesc& desc, const VkPhysicalDeviceLimits& limits);
VkDescriptorSetLayoutBinding getDSLBinding(uint32_t binding,
                                           VkDescriptorType descriptorType,
                                           uint32_t descriptorCount,
                                           VkShaderStageFlags stageFlags,
                                           const VkSampler* immutableSamplers = nullptr);
VkSpecializationInfo getPipelineShaderStageSpecializationInfo(lvk::SpecializationConstantDesc desc, VkSpecializationMapEntry* outEntries);
VkPipelineShaderStageCreateInfo getPipelineShaderStageCreateInfo(VkShaderStageFlagBits stage,
                                                                 VkShaderModule shaderModule,
                                                                 const char* entryPoint,
                                                                 const VkSpecializationInfo* specializationInfo);
VkBindImageMemoryInfo getBindImageMemoryInfo(const VkBindImagePlaneMemoryInfo* next, VkImage image, VkDeviceMemory memory);

StageAccess getPipelineStageAccess(VkImageLayout state);

void imageMemoryBarrier(VkCommandBuffer buffer,
                        VkImage image,
                        VkAccessFlags srcAccessMask,
                        VkAccessFlags dstAccessMask,
                        VkImageLayout oldImageLayout,
                        VkImageLayout newImageLayout,
                        VkPipelineStageFlags srcStageMask,
                        VkPipelineStageFlags dstStageMask,
                        VkImageSubresourceRange subresourceRange);
void imageMemoryBarrier2(VkCommandBuffer buffer,
                         VkImage image,
                         StageAccess src,
                         StageAccess dst,
                         VkImageLayout oldImageLayout,
                         VkImageLayout newImageLayout,
                         VkImageSubresourceRange subresourceRange);

VkSampleCountFlagBits getVulkanSampleCountFlags(uint32_t numSamples, VkSampleCountFlags maxSamplesMask);

void setResultFrom(Result* outResult, VkResult result);
Result getResultFromVkResult(VkResult result);
const char* getVulkanResultString(VkResult result);
uint32_t getBytesPerPixel(VkFormat format);
uint32_t getNumImagePlanes(VkFormat format);
lvk::Format vkFormatToFormat(VkFormat format);
VkFormat formatToVkFormat(lvk::Format format);
VkCompareOp compareOpToVkCompareOp(lvk::CompareOp func);
VkExtent2D getImagePlaneExtent(VkExtent2D plane0, lvk::Format format, uint32_t plane);

// raw Vulkan helpers: use this if you want to interop LightweightVK API with your own raw Vulkan API calls
VkDevice getVkDevice(const IContext* ctx);
VkPhysicalDevice getVkPhysicalDevice(const IContext* ctx);
VkCommandBuffer getVkCommandBuffer(const ICommandBuffer& buffer);
VkBuffer getVkBuffer(const IContext* ctx, BufferHandle buffer);
VkImage getVkImage(const IContext* ctx, TextureHandle texture);
VkImageView getVkImageView(const IContext* ctx, TextureHandle texture);
VkShaderModule getVkShaderModule(const IContext* ctx, ShaderModuleHandle shader);
VkDeviceAddress getVkAccelerationStructureDeviceAddress(const IContext* ctx, AccelStructHandle accelStruct);
VkAccelerationStructureKHR getVkAccelerationStructure(const IContext* ctx, AccelStructHandle accelStruct);
VkBuffer getVkBuffer(const IContext* ctx, AccelStructHandle accelStruct);
VkPipeline getVkPipeline(const IContext* ctx, RayTracingPipelineHandle pipeline);
VkPipelineLayout getVkPipelineLayout(const IContext* ctx, RayTracingPipelineHandle pipeline);

} // namespace lvk
