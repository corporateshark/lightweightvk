#pragma once

#include <vector>
#include <vulkan/vulkan.h>

namespace lvk {
	struct Result;
}

namespace lvk::shader {
	void initialize();

	void shutdown();

	VkPhysicalDevice getTargetPhysicalDevice();

	void loadTargetPhysicalDeviceLimits(VkPhysicalDevice device, const VkPhysicalDeviceLimits &limits);

	lvk::Result compile(const char *code, VkShaderStageFlagBits stage, std::vector<uint32_t>& out);

	const char *getProcessingLog();
}