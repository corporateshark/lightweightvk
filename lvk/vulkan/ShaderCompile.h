#pragma once

#include <igl/vulkan/VulkanHelpers.h>

namespace lvk {
	struct Result;
}

namespace lvk::shader {
	void initialize();

	void shutdown();

	VkPhysicalDevice getTargetPhysicalDevice();

	void loadTargetPhysicalDeviceLimits(VkPhysicalDevice device, const VkPhysicalDeviceLimits &limits);

	lvk::Result compile(VkDevice device,
						VkShaderStageFlagBits stage,
						const char* code,
						VkShaderModule* outShaderModule);
}