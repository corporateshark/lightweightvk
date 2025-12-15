/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <vector>

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
VkFence createFence(VkDevice device, const char* debugName, bool isSignaled = false);
VmaAllocator createVmaAllocator(VkPhysicalDevice physDev, VkDevice device, VkInstance instance, uint32_t apiVersion);
uint32_t findQueueFamilyIndex(VkPhysicalDevice physDev, VkQueueFlags flags);
VkResult setDebugObjectName(VkDevice device, VkObjectType type, uint64_t handle, const char* name);
VkResult allocateMemory2(VkPhysicalDevice physDev,
                         VkDevice device,
                         const VkMemoryRequirements2* memRequirements,
                         VkMemoryPropertyFlags props,
                         VkDeviceMemory* outMemory);

glslang_resource_t getGlslangResource(const VkPhysicalDeviceLimits& limits);
Result compileShaderGlslang(lvk::ShaderStage stage,
                            const char* code,
                            std::vector<uint8_t>* outSPIRV,
                            const glslang_resource_t* glslLangResource = nullptr);
Result compileShaderSlang(lvk::ShaderStage stage, const char* code, const char* entryPointName, std::vector<uint8_t>* outSPIRV);

VkSamplerCreateInfo samplerStateDescToVkSamplerCreateInfo(const lvk::SamplerStateDesc& desc, const VkPhysicalDeviceLimits& limits);
VkDescriptorSetLayoutBinding getDSLBinding(uint32_t binding,
                                           VkDescriptorType descriptorType,
                                           uint32_t descriptorCount,
                                           VkShaderStageFlags stageFlags,
                                           const VkSampler* immutableSamplers = nullptr);
VkSpecializationInfo getPipelineShaderStageSpecializationInfo(lvk::SpecializationConstantDesc desc, VkSpecializationMapEntry* outEntries);
VkPipelineShaderStageCreateInfo getPipelineShaderStageCreateInfo(VkShaderStageFlagBits stage,
                                                                 const VkShaderModuleCreateInfo& ci,
                                                                 const char* entryPoint,
                                                                 const VkSpecializationInfo* specializationInfo);
VkBindImageMemoryInfo getBindImageMemoryInfo(const VkBindImagePlaneMemoryInfo* next, VkImage image, VkDeviceMemory memory);

StageAccess getPipelineStageAccess(VkImageLayout state);

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
const char* getVkDeviceFaultAddressTypeString(VkDeviceFaultAddressTypeEXT type);
uint32_t getBytesPerPixel(VkFormat format);
uint32_t getNumImagePlanes(VkFormat format);
lvk::Format vkFormatToFormat(VkFormat format);
VkFormat formatToVkFormat(lvk::Format format);
VkExtent2D getImagePlaneExtent(VkExtent2D plane0, lvk::Format format, uint32_t plane);

// raw Vulkan helpers: use this if you want to interop LightweightVK API with your own raw Vulkan API calls
VkDevice getVkDevice(const IContext* ctx);
VkPhysicalDevice getVkPhysicalDevice(const IContext* ctx);
VkCommandBuffer getVkCommandBuffer(const ICommandBuffer& buffer);
VkBuffer getVkBuffer(const IContext* ctx, BufferHandle buffer);
VkImage getVkImage(const IContext* ctx, TextureHandle texture);
VkImageView getVkImageView(const IContext* ctx, TextureHandle texture);
VkDeviceAddress getVkAccelerationStructureDeviceAddress(const IContext* ctx, AccelStructHandle accelStruct);
VkAccelerationStructureKHR getVkAccelerationStructure(const IContext* ctx, AccelStructHandle accelStruct);
VkBuffer getVkBuffer(const IContext* ctx, AccelStructHandle accelStruct);
VkPipeline getVkPipeline(const IContext* ctx, RayTracingPipelineHandle pipeline);
VkPipelineLayout getVkPipelineLayout(const IContext* ctx, RayTracingPipelineHandle pipeline);

VkDeviceSize getBufferSize(const IContext* ctx, lvk::BufferHandle handle);

// properties/limits
const VkPhysicalDeviceProperties2& getVkPhysicalDeviceProperties2(const IContext* ctx);
const VkPhysicalDeviceVulkan11Properties& getVkPhysicalDeviceVulkan11Properties(const IContext* ctx);
const VkPhysicalDeviceVulkan12Properties& getVkPhysicalDeviceVulkan12Properties(const IContext* ctx);
const VkPhysicalDeviceVulkan13Properties& getVkPhysicalDeviceVulkan13Properties(const IContext* ctx);

} // namespace lvk
