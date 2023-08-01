#pragma once

#include <vector>
#include <vulkan/vulkan.h>

#ifdef _MSC_VER
#ifdef LVK_BUILDING_SHADER_COMPILE
#define LVKDLL __declspec(dllexport)
#else
#define LVKDLL __declspec(dllimport)
#endif
#else
#define LVKDLL
#endif

namespace lvk {
	struct Result;
}

namespace lvk::shader {
	LVKDLL void initialize();

	LVKDLL void shutdown();

	LVKDLL VkPhysicalDevice getTargetPhysicalDevice();

	LVKDLL void loadTargetPhysicalDeviceLimits(VkPhysicalDevice device, const VkPhysicalDeviceLimits &limits);

	LVKDLL lvk::Result compile(const char *code, VkShaderStageFlagBits stage, std::vector<uint32_t>& out);

	LVKDLL const char *getProcessingLog();
}