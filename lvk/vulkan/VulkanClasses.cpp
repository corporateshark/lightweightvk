/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>
#include <vector>

#define VMA_IMPLEMENTATION
#define VOLK_IMPLEMENTATION

#include "VulkanClasses.h"
#include "VulkanUtils.h"

#include <SPIRV-Reflect/spirv_reflect.h>
#include <glslang/Include/glslang_c_interface.h>
#include <ldrutils/lutils/ScopeExit.h>

#ifndef VK_USE_PLATFORM_WIN32_KHR
#include <unistd.h>
#endif

// clang-format off
#if defined(LVK_WITH_TRACY_GPU)
  #include "tracy/TracyVulkan.hpp"
  #define LVK_PROFILER_GPU_ZONE(name, ctx, cmdBuffer, color) TracyVkZoneC(ctx->pimpl_->tracyVkCtx_, cmdBuffer, name, color);
#else
  #define LVK_PROFILER_GPU_ZONE(name, ctx, cmdBuffer, color)
#endif // LVK_WITH_TRACY_GPU
// clang-format on

#if !defined(__APPLE__)
#include <malloc.h>
#endif

uint32_t lvk::VulkanPipelineBuilder::numPipelinesCreated_ = 0;

static_assert(lvk::HWDeviceDesc::LVK_MAX_PHYSICAL_DEVICE_NAME_SIZE == VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
static_assert(lvk::Swizzle_Default == (uint32_t)VK_COMPONENT_SWIZZLE_IDENTITY);
static_assert(lvk::Swizzle_0 == (uint32_t)VK_COMPONENT_SWIZZLE_ZERO);
static_assert(lvk::Swizzle_1 == (uint32_t)VK_COMPONENT_SWIZZLE_ONE);
static_assert(lvk::Swizzle_R == (uint32_t)VK_COMPONENT_SWIZZLE_R);
static_assert(lvk::Swizzle_G == (uint32_t)VK_COMPONENT_SWIZZLE_G);
static_assert(lvk::Swizzle_B == (uint32_t)VK_COMPONENT_SWIZZLE_B);
static_assert(lvk::Swizzle_A == (uint32_t)VK_COMPONENT_SWIZZLE_A);
static_assert(sizeof(lvk::AccelStructInstance) == sizeof(VkAccelerationStructureInstanceKHR));
static_assert(sizeof(lvk::mat3x4) == sizeof(VkTransformMatrixKHR));
static_assert(sizeof(lvk::ClearColorValue) == sizeof(VkClearColorValue));

namespace {

const char* kDefaultValidationLayers[] = {"VK_LAYER_KHRONOS_validation"};

// These bindings should match GLSL declarations injected into shaders in VulkanContext::createShaderModule().
enum Bindings {
  kBinding_Textures = 0,
  kBinding_Samplers = 1,
  kBinding_StorageImages = 2,
  kBinding_YUVImages = 3,
  kBinding_AccelerationStructures = 4,
  kBinding_NumBindings = 5,
};

const uint32_t kDescriptorSet_InputAttachments = 4; // for VkDescriptorSetLayout in getVkPipeline()

VkDeviceSize getAlignedSize(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t getAlignedAddress(uint64_t addr, uint64_t align) {
  const uint64_t offs = addr % align;
  return offs ? addr + (align - offs) : addr;
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msgSeverity,
                                                   [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT msgType,
                                                   const VkDebugUtilsMessengerCallbackDataEXT* cbData,
                                                   void* userData) {
  if (msgSeverity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    return VK_FALSE;
  }

  const bool isError = (msgSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
  const bool isWarning = (msgSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0;

  const size_t len = cbData->pMessage ? strlen(cbData->pMessage) : 128u;

  LVK_ASSERT(len < 65536);

  char* errorName = (char*)alloca(len + 1);
  int object = 0;
  void* handle = nullptr;
  char typeName[128] = {};
  void* messageID = nullptr;

  minilog::eLogLevel level = minilog::Log;
  if (isError) {
    lvk::VulkanContext* ctx = static_cast<lvk::VulkanContext*>(userData);
    level = ctx->config_.terminateOnValidationError ? minilog::FatalError : minilog::Warning;
  }

  if (!isError && !isWarning && cbData->pMessageIdName) {
    if (strcmp(cbData->pMessageIdName, "Loader Message") == 0) {
      return VK_FALSE;
    }
  }

  if (sscanf(cbData->pMessage,
             "Validation Error: [ %[^]] ] Object %i: handle = %p, type = %127s | MessageID = %p",
             errorName,
             &object,
             &handle,
             typeName,
             &messageID) >= 2) {
    const char* message = strrchr(cbData->pMessage, '|') + 1;

    MINILOG_LOG_PROC(level,
                     "%sValidation layer:\n Validation Error: %s \n Object %i: handle = %p, type = %s\n "
                     "MessageID = %p \n%s \n",
                     isError ? "\nERROR:\n" : "",
                     errorName,
                     object,
                     handle,
                     typeName,
                     messageID,
                     message);
  } else {
    MINILOG_LOG_PROC(level, "%sValidation layer:\n%s\n", isError ? "\nERROR:\n" : "", cbData->pMessage);
  }

  if (isError) {
    lvk::VulkanContext* ctx = static_cast<lvk::VulkanContext*>(userData);
    if (ctx->config_.terminateOnValidationError) {
      LVK_ASSERT(false);
      std::terminate();
    }
  }

  return VK_FALSE;
}

VkIndexType indexFormatToVkIndexType(lvk::IndexFormat fmt) {
  switch (fmt) {
  case lvk::IndexFormat_UI8:
    return VK_INDEX_TYPE_UINT8_EXT;
  case lvk::IndexFormat_UI16:
    return VK_INDEX_TYPE_UINT16;
  case lvk::IndexFormat_UI32:
    return VK_INDEX_TYPE_UINT32;
  };
  LVK_ASSERT(false);
  return VK_INDEX_TYPE_NONE_KHR;
}

VkPrimitiveTopology topologyToVkPrimitiveTopology(lvk::Topology t) {
  switch (t) {
  case lvk::Topology_Point:
    return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  case lvk::Topology_Line:
    return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  case lvk::Topology_LineStrip:
    return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  case lvk::Topology_Triangle:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  case lvk::Topology_TriangleStrip:
    return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  case lvk::Topology_Patch:
    return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
  }
  LVK_ASSERT_MSG(false, "Implement Topology = %u", (uint32_t)t);
  return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
}

VkAttachmentLoadOp loadOpToVkAttachmentLoadOp(lvk::LoadOp a) {
  switch (a) {
  case lvk::LoadOp_Invalid:
    LVK_ASSERT(false);
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  case lvk::LoadOp_DontCare:
    return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  case lvk::LoadOp_Load:
    return VK_ATTACHMENT_LOAD_OP_LOAD;
  case lvk::LoadOp_Clear:
    return VK_ATTACHMENT_LOAD_OP_CLEAR;
  case lvk::LoadOp_None:
    return VK_ATTACHMENT_LOAD_OP_NONE_EXT;
  }
  LVK_ASSERT(false);
  return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
}

VkAttachmentStoreOp storeOpToVkAttachmentStoreOp(lvk::StoreOp a) {
  switch (a) {
  case lvk::StoreOp_DontCare:
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  case lvk::StoreOp_Store:
    return VK_ATTACHMENT_STORE_OP_STORE;
  case lvk::StoreOp_MsaaResolve:
    // for MSAA resolve, we have to store data into a special "resolve" attachment
    return VK_ATTACHMENT_STORE_OP_DONT_CARE;
  case lvk::StoreOp_None:
    return VK_ATTACHMENT_STORE_OP_NONE;
  }
  LVK_ASSERT(false);
  return VK_ATTACHMENT_STORE_OP_DONT_CARE;
}

VkResolveModeFlagBits resolveModeToVkResolveModeFlagBits(lvk::ResolveMode mode, VkResolveModeFlags supported) {
  switch (mode) {
  case lvk::ResolveMode_None:
    return VK_RESOLVE_MODE_NONE;
  case lvk::ResolveMode_SampleZero:
    return VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
  case lvk::ResolveMode_Average:
    return supported & VK_RESOLVE_MODE_AVERAGE_BIT ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
  case lvk::ResolveMode_Min:
    return supported & VK_RESOLVE_MODE_MIN_BIT ? VK_RESOLVE_MODE_MIN_BIT : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
  case lvk::ResolveMode_Max:
    return supported & VK_RESOLVE_MODE_MAX_BIT ? VK_RESOLVE_MODE_MAX_BIT : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
  }
  LVK_ASSERT(false);
  return VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
}

VkShaderStageFlagBits shaderStageToVkShaderStage(lvk::ShaderStage stage) {
  switch (stage) {
  case lvk::Stage_Vert:
    return VK_SHADER_STAGE_VERTEX_BIT;
  case lvk::Stage_Tesc:
    return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
  case lvk::Stage_Tese:
    return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
  case lvk::Stage_Geom:
    return VK_SHADER_STAGE_GEOMETRY_BIT;
  case lvk::Stage_Frag:
    return VK_SHADER_STAGE_FRAGMENT_BIT;
  case lvk::Stage_Comp:
    return VK_SHADER_STAGE_COMPUTE_BIT;
  case lvk::Stage_Task:
    return VK_SHADER_STAGE_TASK_BIT_EXT;
  case lvk::Stage_Mesh:
    return VK_SHADER_STAGE_MESH_BIT_EXT;
  case lvk::Stage_RayGen:
    return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  case lvk::Stage_AnyHit:
    return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  case lvk::Stage_ClosestHit:
    return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  case lvk::Stage_Miss:
    return VK_SHADER_STAGE_MISS_BIT_KHR;
  case lvk::Stage_Intersection:
    return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  case lvk::Stage_Callable:
    return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
  };
  LVK_ASSERT(false);
  return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

VkMemoryPropertyFlags storageTypeToVkMemoryPropertyFlags(lvk::StorageType storage) {
  VkMemoryPropertyFlags memFlags{0};

  switch (storage) {
  case lvk::StorageType_Device:
    memFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    break;
  case lvk::StorageType_HostVisible:
    memFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    break;
  case lvk::StorageType_Memoryless:
    memFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
    break;
  }
  return memFlags;
}

VkBuildAccelerationStructureFlagsKHR buildFlagsToVkBuildAccelerationStructureFlags(uint8_t buildFlags) {
  VkBuildAccelerationStructureFlagsKHR flags = 0;

  if (buildFlags & lvk::AccelStructBuildFlagBits_AllowUpdate) {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  }
  if (buildFlags & lvk::AccelStructBuildFlagBits_AllowCompaction) {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  }
  if (buildFlags & lvk::AccelStructBuildFlagBits_PreferFastTrace) {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  }
  if (buildFlags & lvk::AccelStructBuildFlagBits_PreferFastBuild) {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
  }
  if (buildFlags & lvk::AccelStructBuildFlagBits_LowMemory) {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR;
  }

  return flags;
}

VkPolygonMode polygonModeToVkPolygonMode(lvk::PolygonMode mode) {
  switch (mode) {
  case lvk::PolygonMode_Fill:
    return VK_POLYGON_MODE_FILL;
  case lvk::PolygonMode_Line:
    return VK_POLYGON_MODE_LINE;
  case lvk::PolygonMode_Point:
    return VK_POLYGON_MODE_POINT;
  }
  LVK_ASSERT_MSG(false, "Implement a missing polygon fill mode");
  return VK_POLYGON_MODE_FILL;
}

VkBlendFactor blendFactorToVkBlendFactor(lvk::BlendFactor value) {
  switch (value) {
  case lvk::BlendFactor_Zero:
    return VK_BLEND_FACTOR_ZERO;
  case lvk::BlendFactor_One:
    return VK_BLEND_FACTOR_ONE;
  case lvk::BlendFactor_SrcColor:
    return VK_BLEND_FACTOR_SRC_COLOR;
  case lvk::BlendFactor_OneMinusSrcColor:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
  case lvk::BlendFactor_DstColor:
    return VK_BLEND_FACTOR_DST_COLOR;
  case lvk::BlendFactor_OneMinusDstColor:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
  case lvk::BlendFactor_SrcAlpha:
    return VK_BLEND_FACTOR_SRC_ALPHA;
  case lvk::BlendFactor_OneMinusSrcAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  case lvk::BlendFactor_DstAlpha:
    return VK_BLEND_FACTOR_DST_ALPHA;
  case lvk::BlendFactor_OneMinusDstAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
  case lvk::BlendFactor_BlendColor:
    return VK_BLEND_FACTOR_CONSTANT_COLOR;
  case lvk::BlendFactor_OneMinusBlendColor:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
  case lvk::BlendFactor_BlendAlpha:
    return VK_BLEND_FACTOR_CONSTANT_ALPHA;
  case lvk::BlendFactor_OneMinusBlendAlpha:
    return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
  case lvk::BlendFactor_SrcAlphaSaturated:
    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
  case lvk::BlendFactor_Src1Color:
    return VK_BLEND_FACTOR_SRC1_COLOR;
  case lvk::BlendFactor_OneMinusSrc1Color:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
  case lvk::BlendFactor_Src1Alpha:
    return VK_BLEND_FACTOR_SRC1_ALPHA;
  case lvk::BlendFactor_OneMinusSrc1Alpha:
    return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
  default:
    LVK_ASSERT(false);
    return VK_BLEND_FACTOR_ONE; // default for unsupported values
  }
}

VkBlendOp blendOpToVkBlendOp(lvk::BlendOp value) {
  switch (value) {
  case lvk::BlendOp_Add:
    return VK_BLEND_OP_ADD;
  case lvk::BlendOp_Subtract:
    return VK_BLEND_OP_SUBTRACT;
  case lvk::BlendOp_ReverseSubtract:
    return VK_BLEND_OP_REVERSE_SUBTRACT;
  case lvk::BlendOp_Min:
    return VK_BLEND_OP_MIN;
  case lvk::BlendOp_Max:
    return VK_BLEND_OP_MAX;
  }

  LVK_ASSERT(false);
  return VK_BLEND_OP_ADD;
}

VkCullModeFlags cullModeToVkCullMode(lvk::CullMode mode) {
  switch (mode) {
  case lvk::CullMode_None:
    return VK_CULL_MODE_NONE;
  case lvk::CullMode_Front:
    return VK_CULL_MODE_FRONT_BIT;
  case lvk::CullMode_Back:
    return VK_CULL_MODE_BACK_BIT;
  }
  LVK_ASSERT_MSG(false, "Implement a missing cull mode");
  return VK_CULL_MODE_NONE;
}

VkFrontFace windingModeToVkFrontFace(lvk::WindingMode mode) {
  switch (mode) {
  case lvk::WindingMode_CCW:
    return VK_FRONT_FACE_COUNTER_CLOCKWISE;
  case lvk::WindingMode_CW:
    return VK_FRONT_FACE_CLOCKWISE;
  }
  LVK_ASSERT_MSG(false, "Wrong winding order (cannot be more than 2)");
  return VK_FRONT_FACE_CLOCKWISE;
}

VkStencilOp stencilOpToVkStencilOp(lvk::StencilOp op) {
  switch (op) {
  case lvk::StencilOp_Keep:
    return VK_STENCIL_OP_KEEP;
  case lvk::StencilOp_Zero:
    return VK_STENCIL_OP_ZERO;
  case lvk::StencilOp_Replace:
    return VK_STENCIL_OP_REPLACE;
  case lvk::StencilOp_IncrementClamp:
    return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
  case lvk::StencilOp_DecrementClamp:
    return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
  case lvk::StencilOp_Invert:
    return VK_STENCIL_OP_INVERT;
  case lvk::StencilOp_IncrementWrap:
    return VK_STENCIL_OP_INCREMENT_AND_WRAP;
  case lvk::StencilOp_DecrementWrap:
    return VK_STENCIL_OP_DECREMENT_AND_WRAP;
  }
  LVK_ASSERT(false);
  return VK_STENCIL_OP_KEEP;
}

VkVertexInputRate vertexInputRateToVkVertexInputRate(lvk::VertexInputRate rate) {
  switch (rate) {
  case lvk::VertexInputRate_Vertex:
    return VK_VERTEX_INPUT_RATE_VERTEX;
  case lvk::VertexInputRate_Instance:
    return VK_VERTEX_INPUT_RATE_INSTANCE;
  }
  LVK_ASSERT(false);
  return VK_VERTEX_INPUT_RATE_VERTEX;
}

VkFormat vertexFormatToVkFormat(lvk::VertexFormat fmt) {
  using lvk::VertexFormat;
  switch (fmt) {
  case VertexFormat::Invalid:
    LVK_ASSERT(false);
    return VK_FORMAT_UNDEFINED;
  case VertexFormat::Float1:
    return VK_FORMAT_R32_SFLOAT;
  case VertexFormat::Float2:
    return VK_FORMAT_R32G32_SFLOAT;
  case VertexFormat::Float3:
    return VK_FORMAT_R32G32B32_SFLOAT;
  case VertexFormat::Float4:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  case VertexFormat::Byte1:
    return VK_FORMAT_R8_SINT;
  case VertexFormat::Byte2:
    return VK_FORMAT_R8G8_SINT;
  case VertexFormat::Byte3:
    return VK_FORMAT_R8G8B8_SINT;
  case VertexFormat::Byte4:
    return VK_FORMAT_R8G8B8A8_SINT;
  case VertexFormat::UByte1:
    return VK_FORMAT_R8_UINT;
  case VertexFormat::UByte2:
    return VK_FORMAT_R8G8_UINT;
  case VertexFormat::UByte3:
    return VK_FORMAT_R8G8B8_UINT;
  case VertexFormat::UByte4:
    return VK_FORMAT_R8G8B8A8_UINT;
  case VertexFormat::Short1:
    return VK_FORMAT_R16_SINT;
  case VertexFormat::Short2:
    return VK_FORMAT_R16G16_SINT;
  case VertexFormat::Short3:
    return VK_FORMAT_R16G16B16_SINT;
  case VertexFormat::Short4:
    return VK_FORMAT_R16G16B16A16_SINT;
  case VertexFormat::UShort1:
    return VK_FORMAT_R16_UINT;
  case VertexFormat::UShort2:
    return VK_FORMAT_R16G16_UINT;
  case VertexFormat::UShort3:
    return VK_FORMAT_R16G16B16_UINT;
  case VertexFormat::UShort4:
    return VK_FORMAT_R16G16B16A16_UINT;
    // Normalized variants
  case VertexFormat::Byte2Norm:
    return VK_FORMAT_R8G8_SNORM;
  case VertexFormat::Byte4Norm:
    return VK_FORMAT_R8G8B8A8_SNORM;
  case VertexFormat::UByte2Norm:
    return VK_FORMAT_R8G8_UNORM;
  case VertexFormat::UByte4Norm:
    return VK_FORMAT_R8G8B8A8_UNORM;
  case VertexFormat::Short2Norm:
    return VK_FORMAT_R16G16_SNORM;
  case VertexFormat::Short4Norm:
    return VK_FORMAT_R16G16B16A16_SNORM;
  case VertexFormat::UShort2Norm:
    return VK_FORMAT_R16G16_UNORM;
  case VertexFormat::UShort4Norm:
    return VK_FORMAT_R16G16B16A16_UNORM;
  case VertexFormat::Int1:
    return VK_FORMAT_R32_SINT;
  case VertexFormat::Int2:
    return VK_FORMAT_R32G32_SINT;
  case VertexFormat::Int3:
    return VK_FORMAT_R32G32B32_SINT;
  case VertexFormat::Int4:
    return VK_FORMAT_R32G32B32A32_SINT;
  case VertexFormat::UInt1:
    return VK_FORMAT_R32_UINT;
  case VertexFormat::UInt2:
    return VK_FORMAT_R32G32_UINT;
  case VertexFormat::UInt3:
    return VK_FORMAT_R32G32B32_UINT;
  case VertexFormat::UInt4:
    return VK_FORMAT_R32G32B32A32_UINT;
  case VertexFormat::HalfFloat1:
    return VK_FORMAT_R16_SFLOAT;
  case VertexFormat::HalfFloat2:
    return VK_FORMAT_R16G16_SFLOAT;
  case VertexFormat::HalfFloat3:
    return VK_FORMAT_R16G16B16_SFLOAT;
  case VertexFormat::HalfFloat4:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  case VertexFormat::Int_2_10_10_10_REV:
    return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
  }
  LVK_ASSERT(false);
  return VK_FORMAT_UNDEFINED;
}

bool supportsFormat(VkPhysicalDevice physicalDevice, VkFormat format) {
  VkFormatProperties properties;
  vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
  return properties.bufferFeatures != 0 || properties.linearTilingFeatures != 0 || properties.optimalTilingFeatures != 0;
}

std::vector<VkFormat> getCompatibleDepthStencilFormats(lvk::Format format) {
  switch (format) {
  case lvk::Format_Z_UN16:
    return {VK_FORMAT_D16_UNORM, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
  case lvk::Format_Z_UN24:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM_S8_UINT};
  case lvk::Format_Z_F32:
    return {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
  case lvk::Format_Z_UN24_S_UI8:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  case lvk::Format_Z_F32_S_UI8:
    return {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT};
  default:
    return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
  }
  return {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT};
}

bool validateImageLimits(VkImageType imageType,
                         VkSampleCountFlagBits samples,
                         const VkExtent3D& extent,
                         const VkPhysicalDeviceLimits& limits,
                         lvk::Result* outResult) {
  using lvk::Result;

  if (samples != VK_SAMPLE_COUNT_1_BIT && !LVK_VERIFY(imageType == VK_IMAGE_TYPE_2D)) {
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "Multisampling is supported only for 2D images"));
    return false;
  }

  if (imageType == VK_IMAGE_TYPE_2D &&
      !LVK_VERIFY(extent.width <= limits.maxImageDimension2D && extent.height <= limits.maxImageDimension2D)) {
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "2D texture size exceeded"));
    return false;
  }
  if (imageType == VK_IMAGE_TYPE_3D &&
      !LVK_VERIFY(extent.width <= limits.maxImageDimension3D && extent.height <= limits.maxImageDimension3D &&
                  extent.depth <= limits.maxImageDimension3D)) {
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "3D texture size exceeded"));
    return false;
  }

  return true;
}

lvk::Result validateRange(const VkExtent3D& ext, uint32_t numLevels, const lvk::TextureRangeDesc& range) {
  if (!LVK_VERIFY(range.dimensions.width > 0 && range.dimensions.height > 0 || range.dimensions.depth > 0 || range.numLayers > 0 ||
                  range.numMipLevels > 0)) {
    return lvk::Result{lvk::Result::Code::ArgumentOutOfRange, "width, height, depth numLayers, and numMipLevels must be > 0"};
  }
  if (range.mipLevel > numLevels) {
    return lvk::Result{lvk::Result::Code::ArgumentOutOfRange, "range.mipLevel exceeds texture mip-levels"};
  }

  const uint32_t texWidth = std::max(ext.width >> range.mipLevel, 1u);
  const uint32_t texHeight = std::max(ext.height >> range.mipLevel, 1u);
  const uint32_t texDepth = std::max(ext.depth >> range.mipLevel, 1u);

  if (range.dimensions.width > texWidth || range.dimensions.height > texHeight || range.dimensions.depth > texDepth) {
    return lvk::Result{lvk::Result::Code::ArgumentOutOfRange, "range dimensions exceed texture dimensions"};
  }
  if (range.offset.x > texWidth - range.dimensions.width || range.offset.y > texHeight - range.dimensions.height ||
      range.offset.z > texDepth - range.dimensions.depth) {
    return lvk::Result{lvk::Result::Code::ArgumentOutOfRange, "range dimensions exceed texture dimensions"};
  }

  return lvk::Result{};
}

bool isHostVisibleSingleHeapMemory(VkPhysicalDevice physDev) {
  VkPhysicalDeviceMemoryProperties memProperties;

  vkGetPhysicalDeviceMemoryProperties(physDev, &memProperties);

  if (memProperties.memoryHeapCount != 1) {
    return false;
  }

  const uint32_t flag = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memProperties.memoryTypes[i].propertyFlags & flag) == flag) {
      return true;
    }
  }

  return false;
}

void getDeviceExtensionProps(VkPhysicalDevice dev, std::vector<VkExtensionProperties>& props, const char* validationLayer = nullptr) {
  uint32_t numExtensions = 0;
  vkEnumerateDeviceExtensionProperties(dev, validationLayer, &numExtensions, nullptr);
  std::vector<VkExtensionProperties> p(numExtensions);
  vkEnumerateDeviceExtensionProperties(dev, validationLayer, &numExtensions, p.data());
  props.insert(props.end(), p.begin(), p.end());
}

bool hasExtension(const char* ext, const std::vector<VkExtensionProperties>& props) {
  for (const VkExtensionProperties& p : props) {
    if (strcmp(ext, p.extensionName) == 0)
      return true;
  }
  return false;
}

void transitionToColorAttachment(VkCommandBuffer buffer, lvk::VulkanImage* colorTex) {
  if (!LVK_VERIFY(colorTex)) {
    return;
  }

  if (!LVK_VERIFY(!colorTex->isDepthFormat_ && !colorTex->isStencilFormat_)) {
    LVK_ASSERT_MSG(false, "Color attachments cannot have depth/stencil formats");
    return;
  }
  LVK_ASSERT_MSG(colorTex->vkImageFormat_ != VK_FORMAT_UNDEFINED, "Invalid color attachment format");
  colorTex->transitionLayout(buffer,
                             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
}

bool isDepthOrStencilVkFormat(VkFormat format) {
  switch (format) {
  case VK_FORMAT_D16_UNORM:
  case VK_FORMAT_X8_D24_UNORM_PACK32:
  case VK_FORMAT_D32_SFLOAT:
  case VK_FORMAT_S8_UINT:
  case VK_FORMAT_D16_UNORM_S8_UINT:
  case VK_FORMAT_D24_UNORM_S8_UINT:
  case VK_FORMAT_D32_SFLOAT_S8_UINT:
    return true;
  default:
    return false;
  }
  return false;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats,
                                           lvk::ColorSpace requestedColorSpace,
                                           bool hasSwapchainColorspaceExt) {
  LVK_ASSERT(!formats.empty());

  auto isNativeSwapChainBGR = [](const std::vector<VkSurfaceFormatKHR>& formats) -> bool {
    for (const VkSurfaceFormatKHR& fmt : formats) {
      // The preferred format should be the one which is closer to the beginning of the formats
      // container. If BGR is encountered earlier, it should be picked as the format of choice. If RGB
      // happens to be earlier, take it.
      if (fmt.format == VK_FORMAT_R8G8B8A8_UNORM || fmt.format == VK_FORMAT_R8G8B8A8_SRGB ||
          fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) {
        return false;
      }
      if (fmt.format == VK_FORMAT_B8G8R8A8_UNORM || fmt.format == VK_FORMAT_B8G8R8A8_SRGB ||
          fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
        return true;
      }
    }
    return false;
  };

  auto colorSpaceToVkSurfaceFormat = [](lvk::ColorSpace colorSpace, bool isBGR, bool hasSwapchainColorspaceExt) -> VkSurfaceFormatKHR {
    switch (colorSpace) {
    case lvk::ColorSpace_SRGB_NONLINEAR:
      return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    case lvk::ColorSpace_SRGB_EXTENDED_LINEAR:
      if (hasSwapchainColorspaceExt)
        return VkSurfaceFormatKHR{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
      [[fallthrough]];
    case lvk::ColorSpace_HDR10:
      if (hasSwapchainColorspaceExt) {
        return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                  VK_COLOR_SPACE_HDR10_ST2084_EXT};
      }
      [[fallthrough]];
    default:
      // default to normal sRGB non linear.
      return VkSurfaceFormatKHR{isBGR ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_R8G8B8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    }
  };

  const VkSurfaceFormatKHR preferred =
      colorSpaceToVkSurfaceFormat(requestedColorSpace, isNativeSwapChainBGR(formats), hasSwapchainColorspaceExt);

  for (const VkSurfaceFormatKHR& fmt : formats) {
    if (fmt.format == preferred.format && fmt.colorSpace == preferred.colorSpace) {
      return fmt;
    }
  }

  // if we can't find a matching format and color space, fallback on matching only format
  for (const VkSurfaceFormatKHR& fmt : formats) {
    if (fmt.format == preferred.format) {
      return fmt;
    }
  }

  LLOGL("Could not find a native swap chain format that matched our designed swapchain format. Defaulting to first supported format.");

  return formats[0];
}

} // namespace

namespace lvk {

struct DeferredTask {
  DeferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) : task_(std::move(task)), handle_(handle) {}
  std::packaged_task<void()> task_;
  SubmitHandle handle_;
};

struct VulkanContextImpl final {
  // Vulkan Memory Allocator
  VmaAllocator vma_ = VK_NULL_HANDLE;

  lvk::CommandBuffer currentCommandBuffer_;

  std::vector<DeferredTask> deferredTasks_;

  struct YcbcrConversionData {
    VkSamplerYcbcrConversionInfo info;
    lvk::Holder<SamplerHandle> sampler;
  };
  YcbcrConversionData ycbcrConversionData_[256]; // indexed by lvk::Format
  uint32_t numYcbcrSamplers_ = 0;
  // max of all used values of VkSamplerYcbcrConversionImageFormatProperties::combinedImageSamplerDescriptorCount
  uint32_t maxCombinedImageSamplerDescriptorCount_ = 1;

#if defined(LVK_WITH_TRACY_GPU)
  TracyVkCtx tracyVkCtx_ = nullptr;
  VkCommandPool tracyCommandPool_ = VK_NULL_HANDLE;
  VkCommandBuffer tracyCommandBuffer_ = VK_NULL_HANDLE;
#endif // LVK_WITH_TRACY_GPU
};

} // namespace lvk

void lvk::VulkanBuffer::flushMappedMemory(const VulkanContext& ctx, VkDeviceSize offset, VkDeviceSize size) const {
  if (!LVK_VERIFY(isMapped())) {
    return;
  }

  if (LVK_VULKAN_USE_VMA) {
    vmaFlushAllocation((VmaAllocator)ctx.getVmaAllocator(), vmaAllocation_, offset, size);
  } else {
    const VkMappedMemoryRange range = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory_,
        .offset = offset,
        .size = size,
    };
    vkFlushMappedMemoryRanges(ctx.getVkDevice(), 1, &range);
  }
}

void lvk::VulkanBuffer::invalidateMappedMemory(const VulkanContext& ctx, VkDeviceSize offset, VkDeviceSize size) const {
  if (!LVK_VERIFY(isMapped())) {
    return;
  }

  if (LVK_VULKAN_USE_VMA) {
    vmaInvalidateAllocation(static_cast<VmaAllocator>(ctx.getVmaAllocator()), vmaAllocation_, offset, size);
  } else {
    const VkMappedMemoryRange range = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = vkMemory_,
        .offset = offset,
        .size = size,
    };
    vkInvalidateMappedMemoryRanges(ctx.getVkDevice(), 1, &range);
  }
}

void lvk::VulkanBuffer::getBufferSubData(const VulkanContext& ctx, size_t offset, size_t size, void* data) {
  // only host-visible buffers can be downloaded this way
  LVK_ASSERT(mappedPtr_);

  if (!mappedPtr_) {
    return;
  }

  LVK_ASSERT(offset + size <= bufferSize_);

  if (!isCoherentMemory_) {
    invalidateMappedMemory(ctx, offset, size);
  }

  const uint8_t* src = static_cast<uint8_t*>(mappedPtr_) + offset;
  memcpy(data, src, size);
}

void lvk::VulkanBuffer::bufferSubData(const VulkanContext& ctx, size_t offset, size_t size, const void* data) {
  // only host-visible buffers can be uploaded this way
  LVK_ASSERT(mappedPtr_);

  if (!mappedPtr_) {
    return;
  }

  LVK_ASSERT(offset + size <= bufferSize_);

  if (data) {
    memcpy((uint8_t*)mappedPtr_ + offset, data, size);
  } else {
    memset((uint8_t*)mappedPtr_ + offset, 0, size);
  }

  if (!isCoherentMemory_) {
    flushMappedMemory(ctx, offset, size);
  }
}

VkImageView lvk::VulkanImage::createImageView(VkDevice device,
                                              VkImageViewType type,
                                              VkFormat format,
                                              VkImageAspectFlags aspectMask,
                                              uint32_t baseLevel,
                                              uint32_t numLevels,
                                              uint32_t baseLayer,
                                              uint32_t numLayers,
                                              const VkComponentMapping components,
                                              const VkSamplerYcbcrConversionInfo* ycbcr,
                                              const char* debugName) const {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const VkImageViewCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = ycbcr,
      .image = vkImage_,
      .viewType = type,
      .format = format,
      .components = components,
      .subresourceRange = {aspectMask, baseLevel, numLevels ? numLevels : numLevels_, baseLayer, numLayers},
  };
  VkImageView vkView = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateImageView(device, &ci, nullptr, &vkView));
  VK_ASSERT(lvk::setDebugObjectName(device, VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)vkView, debugName));

  return vkView;
}

void lvk::VulkanImage::transitionLayout(VkCommandBuffer commandBuffer,
                                        VkImageLayout newImageLayout,
                                        const VkImageSubresourceRange& subresourceRange) const {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_BARRIER);

  const VkImageLayout oldImageLayout =
      vkImageLayout_ == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
          ? (isDepthAttachment() ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
          : vkImageLayout_;

  if (newImageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) {
    newImageLayout = isDepthAttachment() ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }

  StageAccess src = getPipelineStageAccess(oldImageLayout);
  StageAccess dst = getPipelineStageAccess(newImageLayout);

  if (isDepthAttachment() && isResolveAttachment) {
    // https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#renderpass-resolve-operations
    src.stage |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    dst.stage |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    src.access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    dst.access |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
  }

  const VkImageMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src.stage,
      .srcAccessMask = src.access,
      .dstStageMask = dst.stage,
      .dstAccessMask = dst.access,
      .oldLayout = vkImageLayout_,
      .newLayout = newImageLayout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = vkImage_,
      .subresourceRange = subresourceRange,
  };

  const VkDependencyInfo depInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };

  vkCmdPipelineBarrier2(commandBuffer, &depInfo);

  vkImageLayout_ = newImageLayout;
}

VkImageAspectFlags lvk::VulkanImage::getImageAspectFlags() const {
  VkImageAspectFlags flags = 0;

  flags |= isDepthFormat_ ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
  flags |= isStencilFormat_ ? VK_IMAGE_ASPECT_STENCIL_BIT : 0;
  flags |= !(isDepthFormat_ || isStencilFormat_) ? VK_IMAGE_ASPECT_COLOR_BIT : 0;

  return flags;
}

void lvk::VulkanImage::generateMipmap(VkCommandBuffer commandBuffer) const {
  LVK_PROFILER_FUNCTION();

  // Check if device supports downscaling for color or depth/stencil buffer based on image format
  {
    const uint32_t formatFeatureMask = (VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT);

    const bool hardwareDownscalingSupported = (vkFormatProperties_.optimalTilingFeatures & formatFeatureMask) == formatFeatureMask;

    if (!hardwareDownscalingSupported) {
      LLOGW("Doesn't support hardware downscaling of this image format: %p", vkImageFormat_);
      return;
    }
  }

  // Choose linear filter for color formats if supported by the device, else use nearest filter
  // Choose nearest filter by default for depth/stencil formats
  const VkFilter blitFilter = [](bool isDepthOrStencilFormat, bool imageFilterLinear) {
    if (isDepthOrStencilFormat) {
      return VK_FILTER_NEAREST;
    }
    if (imageFilterLinear) {
      return VK_FILTER_LINEAR;
    }
    return VK_FILTER_NEAREST;
  }(isDepthFormat_ || isStencilFormat_, vkFormatProperties_.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

  const VkImageAspectFlags imageAspectFlags = getImageAspectFlags();

  if (vkCmdBeginDebugUtilsLabelEXT) {
    const VkDebugUtilsLabelEXT utilsLabel = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = "Generate mipmaps",
        .color = {1.0f, 0.75f, 1.0f, 1.0f},
    };
    vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &utilsLabel);
  }

  const VkImageLayout originalImageLayout = vkImageLayout_;

  LVK_ASSERT(originalImageLayout != VK_IMAGE_LAYOUT_UNDEFINED);

  // 0: Transition the first level and all layers into VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  transitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VkImageSubresourceRange{imageAspectFlags, 0, 1, 0, numLayers_});

  for (uint32_t layer = 0; layer < numLayers_; ++layer) {
    int32_t mipWidth = (int32_t)vkExtent_.width;
    int32_t mipHeight = (int32_t)vkExtent_.height;

    for (uint32_t i = 1; i < numLevels_; ++i) {
      // 1: Transition the i-th level to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; it will be copied into from the (i-1)-th layer
      lvk::imageMemoryBarrier2(commandBuffer,
                               vkImage_,
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .access = VK_ACCESS_2_NONE},
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
                               VK_IMAGE_LAYOUT_UNDEFINED, // oldImageLayout
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, // newImageLayout
                               VkImageSubresourceRange{imageAspectFlags, i, 1, layer, 1});

      const int32_t nextLevelWidth = mipWidth > 1 ? mipWidth / 2 : 1;
      const int32_t nextLevelHeight = mipHeight > 1 ? mipHeight / 2 : 1;

      const VkOffset3D srcOffsets[2] = {
          VkOffset3D{0, 0, 0},
          VkOffset3D{mipWidth, mipHeight, 1},
      };
      const VkOffset3D dstOffsets[2] = {
          VkOffset3D{0, 0, 0},
          VkOffset3D{nextLevelWidth, nextLevelHeight, 1},
      };

      // 2: Blit the image from the prev mip-level (i-1) (VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) to the current mip-level (i)
      // (VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
#if LVK_VULKAN_PRINT_COMMANDS
      LLOGL("%p vkCmdBlitImage()\n", commandBuffer);
#endif // LVK_VULKAN_PRINT_COMMANDS
      const VkImageBlit blit = {
          .srcSubresource = VkImageSubresourceLayers{imageAspectFlags, i - 1, layer, 1},
          .srcOffsets = {srcOffsets[0], srcOffsets[1]},
          .dstSubresource = VkImageSubresourceLayers{imageAspectFlags, i, layer, 1},
          .dstOffsets = {dstOffsets[0], dstOffsets[1]},
      };
      vkCmdBlitImage(commandBuffer,
                     vkImage_,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     vkImage_,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     1,
                     &blit,
                     blitFilter);
      // 3: Transition i-th level to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL as it will be read from in the next iteration
      lvk::imageMemoryBarrier2(commandBuffer,
                               vkImage_,
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, /* oldImageLayout */
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, /* newImageLayout */
                               VkImageSubresourceRange{imageAspectFlags, i, 1, layer, 1});

      // Compute the size of the next mip-level
      mipWidth = nextLevelWidth;
      mipHeight = nextLevelHeight;
    }
  }

  // 4: Transition all levels and layers (faces) to their final layout
  lvk::imageMemoryBarrier2(
      commandBuffer,
      vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, // oldImageLayout
      originalImageLayout, // newImageLayout
      VkImageSubresourceRange{imageAspectFlags, 0, numLevels_, 0, numLayers_});

  if (vkCmdEndDebugUtilsLabelEXT) {
    vkCmdEndDebugUtilsLabelEXT(commandBuffer);
  }

  vkImageLayout_ = originalImageLayout;
}

bool lvk::VulkanImage::isDepthFormat(VkFormat format) {
  return (format == VK_FORMAT_D16_UNORM) || (format == VK_FORMAT_X8_D24_UNORM_PACK32) || (format == VK_FORMAT_D32_SFLOAT) ||
         (format == VK_FORMAT_D16_UNORM_S8_UINT) || (format == VK_FORMAT_D24_UNORM_S8_UINT) || (format == VK_FORMAT_D32_SFLOAT_S8_UINT);
}

bool lvk::VulkanImage::isStencilFormat(VkFormat format) {
  return (format == VK_FORMAT_S8_UINT) || (format == VK_FORMAT_D16_UNORM_S8_UINT) || (format == VK_FORMAT_D24_UNORM_S8_UINT) ||
         (format == VK_FORMAT_D32_SFLOAT_S8_UINT);
}

VkImageView lvk::VulkanImage::getOrCreateVkImageViewForFramebuffer(VulkanContext& ctx, uint8_t level, uint16_t layer, uint32_t viewMask) {
  LVK_ASSERT(level < LVK_MAX_MIP_LEVELS);
  LVK_ASSERT(layer < LVK_ARRAY_NUM_ELEMENTS(imageViewForFramebuffer_[0]));
  LVK_ASSERT(!viewMask || layer == 0);

  if (level >= LVK_MAX_MIP_LEVELS || layer >= LVK_ARRAY_NUM_ELEMENTS(imageViewForFramebuffer_[0])) {
    return VK_NULL_HANDLE;
  }

  if (!viewMask && imageViewForFramebuffer_[level][layer] != VK_NULL_HANDLE) {
    return imageViewForFramebuffer_[level][layer];
  }
  if (viewMask && imageViewForFramebufferMultiview_[level] != VK_NULL_HANDLE) {
    return imageViewForFramebufferMultiview_[level];
  }

  char debugNameImageView[320] = {0};
  snprintf(debugNameImageView,
           sizeof(debugNameImageView) - 1,
           "Image View: '%s' imageViewForFramebuffer[%u][%u][%u]",
           debugName_,
           level,
           layer,
           viewMask);

  const uint32_t numViews = viewMask ?
#if defined(_MSC_VER)
                                     _mm_popcnt_u32(viewMask)
#else
                                     __builtin_popcount(viewMask)
#endif
                                     : 1;

  VkImageView view = createImageView(ctx.getVkDevice(),
                                     viewMask ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                                     vkImageFormat_,
                                     getImageAspectFlags(),
                                     level,
                                     1u,
                                     layer,
                                     numViews,
                                     {},
                                     nullptr,
                                     debugNameImageView);

  if (viewMask) {
    imageViewForFramebufferMultiview_[level] = view;
  } else {
    imageViewForFramebuffer_[level][layer] = view;
  }

  return view;
}

lvk::VulkanSwapchain::VulkanSwapchain(VulkanContext& ctx, uint32_t width, uint32_t height)
: ctx_(ctx)
, device_(ctx.vkDevice_)
, graphicsQueue_(ctx.deviceQueues_.graphicsQueue)
, width_(width)
, height_(height) {
  surfaceFormat_ =
      chooseSwapSurfaceFormat(ctx.deviceSurfaceFormats_, ctx.config_.swapchainRequestedColorSpace, ctx.has_EXT_swapchain_colorspace_);

  LVK_ASSERT_MSG(ctx.vkSurface_ != VK_NULL_HANDLE,
                 "You are trying to create a swapchain but your OS surface is empty. Did you want to "
                 "create an offscreen rendering context? If so, set 'width' and 'height' to 0 when you "
                 "create your lvk::IContext");

  VkBool32 queueFamilySupportsPresentation = VK_FALSE;
  VK_ASSERT(vkGetPhysicalDeviceSurfaceSupportKHR(
      ctx.getVkPhysicalDevice(), ctx.deviceQueues_.graphicsQueueFamilyIndex, ctx.vkSurface_, &queueFamilySupportsPresentation));
  LVK_ASSERT_MSG(queueFamilySupportsPresentation == VK_TRUE, "The queue family used with the swapchain does not support presentation");

  auto chooseSwapImageCount = [](const VkSurfaceCapabilitiesKHR& caps) -> uint32_t {
    const uint32_t desired = caps.minImageCount + 1;
    const bool exceeded = caps.maxImageCount > 0 && desired > caps.maxImageCount;
    return exceeded ? caps.maxImageCount : desired;
  };

  auto chooseSwapPresentMode = [](const std::vector<VkPresentModeKHR>& modes) -> VkPresentModeKHR {
#if defined(__linux__) || defined(_M_ARM64)
    if (std::find(modes.cbegin(), modes.cend(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.cend()) {
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
#endif // __linux__
    if (std::find(modes.cbegin(), modes.cend(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.cend()) {
      return VK_PRESENT_MODE_MAILBOX_KHR;
    }
    if (std::find(modes.cbegin(), modes.cend(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.cend()) {
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    if (std::find(modes.cbegin(), modes.cend(), VK_PRESENT_MODE_FIFO_RELAXED_KHR) != modes.cend()) {
      return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
  };

  VkSurfaceCapabilitiesKHR caps = {};
  VK_ASSERT(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.getVkPhysicalDevice(), ctx.vkSurface_, &caps));

  VkFormatProperties props = {};
  vkGetPhysicalDeviceFormatProperties(ctx.getVkPhysicalDevice(), surfaceFormat_.format, &props);

  // trim the image extent
  width_ = width = std::min(width, caps.maxImageExtent.width);
  height_ = height = std::min(height, caps.maxImageExtent.width);

  auto chooseUsageFlags = [](const VkSurfaceCapabilitiesKHR& caps, const VkFormatProperties& props) -> VkImageUsageFlags {
    VkImageUsageFlags usageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    const bool isStorageSupported = (caps.supportedUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT) > 0;
    const bool isTilingOptimalSupported = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) > 0;

    if (isStorageSupported && isTilingOptimalSupported) {
      usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    return usageFlags;
  };

  const VkImageUsageFlags usageFlags = chooseUsageFlags(caps, props);
  const bool isCompositeAlphaOpaqueSupported = (ctx.deviceSurfaceCaps_.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0;
  const VkSwapchainCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = ctx.vkSurface_,
      .minImageCount = chooseSwapImageCount(ctx.deviceSurfaceCaps_),
      .imageFormat = surfaceFormat_.format,
      .imageColorSpace = surfaceFormat_.colorSpace,
      .imageExtent = {.width = width, .height = height},
      .imageArrayLayers = 1,
      .imageUsage = usageFlags,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &ctx.deviceQueues_.graphicsQueueFamilyIndex,
#if defined(ANDROID)
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
#else
      .preTransform = ctx.deviceSurfaceCaps_.currentTransform,
#endif
      .compositeAlpha = isCompositeAlphaOpaqueSupported ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
      .presentMode = chooseSwapPresentMode(ctx.devicePresentModes_),
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
  };
  VK_ASSERT(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

  if (ctx_.has_EXT_hdr_metadata_) {
    const VkHdrMetadataEXT metadata = {
        .sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
        .displayPrimaryRed = {.x = 0.680f, .y = 0.320f},
        .displayPrimaryGreen = {.x = 0.265f, .y = 0.690f},
        .displayPrimaryBlue = {.x = 0.150f, .y = 0.060f},
        .whitePoint = {.x = 0.3127f, .y = 0.3290f},
        .maxLuminance = 80.0f,
        .minLuminance = 0.001f,
        .maxContentLightLevel = 2000.0f,
        .maxFrameAverageLightLevel = 500.0f,
    };
    vkSetHdrMetadataEXT(device_, 1, &swapchain_, &metadata);
  }

  VkImage swapchainImages[LVK_MAX_SWAPCHAIN_IMAGES];
  VK_ASSERT(vkGetSwapchainImagesKHR(device_, swapchain_, &numSwapchainImages_, nullptr));
  if (numSwapchainImages_ > LVK_MAX_SWAPCHAIN_IMAGES) {
    LVK_ASSERT(numSwapchainImages_ <= LVK_MAX_SWAPCHAIN_IMAGES);
    numSwapchainImages_ = LVK_MAX_SWAPCHAIN_IMAGES;
  }
  VK_ASSERT(vkGetSwapchainImagesKHR(device_, swapchain_, &numSwapchainImages_, swapchainImages));

  LVK_ASSERT(numSwapchainImages_ > 0);

  char debugNameImage[256] = {0};
  char debugNameImageView[256] = {0};

  // create images, image views and framebuffers
  for (uint32_t i = 0; i < numSwapchainImages_; i++) {
    acquireSemaphore_[i] = lvk::createSemaphore(device_, "Semaphore: swapchain-acquire");

    if (!ctx_.has_EXT_swapchain_maintenance1_) {
      char debugNameFence[256] = {0};
      snprintf(debugNameFence, sizeof(debugNameFence) - 1, "Fence: swapchain %u", i);
      acquireFence_[i] = lvk::createFence(device_, debugNameFence, true);
    }

    snprintf(debugNameImage, sizeof(debugNameImage) - 1, "Image: swapchain %u", i);
    snprintf(debugNameImageView, sizeof(debugNameImageView) - 1, "Image View: swapchain %u", i);
    VulkanImage image = {
        .vkImage_ = swapchainImages[i],
        .vkUsageFlags_ = usageFlags,
        .vkExtent_ = VkExtent3D{.width = width_, .height = height_, .depth = 1},
        .vkType_ = VK_IMAGE_TYPE_2D,
        .vkImageFormat_ = surfaceFormat_.format,
        .isSwapchainImage_ = true,
        .isOwningVkImage_ = false,
        .isDepthFormat_ = VulkanImage::isDepthFormat(surfaceFormat_.format),
        .isStencilFormat_ = VulkanImage::isStencilFormat(surfaceFormat_.format),
    };

    VK_ASSERT(lvk::setDebugObjectName(device_, VK_OBJECT_TYPE_IMAGE, (uint64_t)image.vkImage_, debugNameImage));

    image.imageView_ = image.createImageView(device_,
                                             VK_IMAGE_VIEW_TYPE_2D,
                                             surfaceFormat_.format,
                                             VK_IMAGE_ASPECT_COLOR_BIT,
                                             0,
                                             VK_REMAINING_MIP_LEVELS,
                                             0,
                                             1,
                                             {},
                                             nullptr,
                                             debugNameImageView);

    swapchainTextures_[i] = ctx_.texturesPool_.create(std::move(image));
  }
}

lvk::VulkanSwapchain::~VulkanSwapchain() {
  for (TextureHandle handle : swapchainTextures_) {
    ctx_.destroy(handle);
  }
  vkDestroySwapchainKHR(device_, swapchain_, nullptr);
  for (VkSemaphore sem : acquireSemaphore_) {
    vkDestroySemaphore(device_, sem, nullptr);
  }
  for (VkFence fence : presentFence_) {
    if (fence)
      vkDestroyFence(device_, fence, nullptr);
  }
  for (VkFence fence : acquireFence_) {
    if (fence)
      vkDestroyFence(device_, fence, nullptr);
  }
}

VkImage lvk::VulkanSwapchain::getCurrentVkImage() const {
  if (LVK_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
    lvk::VulkanImage* tex = ctx_.texturesPool_.get(swapchainTextures_[currentImageIndex_]);
    return tex->vkImage_;
  }
  return VK_NULL_HANDLE;
}

VkImageView lvk::VulkanSwapchain::getCurrentVkImageView() const {
  if (LVK_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
    lvk::VulkanImage* tex = ctx_.texturesPool_.get(swapchainTextures_[currentImageIndex_]);
    return tex->imageView_;
  }
  return VK_NULL_HANDLE;
}

lvk::TextureHandle lvk::VulkanSwapchain::getCurrentTexture() {
  LVK_PROFILER_FUNCTION();

  if (getNextImage_) {
    const VkSemaphoreWaitInfo waitInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &ctx_.timelineSemaphore_,
        .pValues = &timelineWaitValues_[currentImageIndex_],
    };
    VK_ASSERT(vkWaitSemaphores(device_, &waitInfo, UINT64_MAX));

    VkFence acquireFence = VK_NULL_HANDLE;

    if (ctx_.has_EXT_swapchain_maintenance1_) {
      // VK_EXT_swapchain_maintenance1: before acquiring again, wait for the presentation operation to finish
      if (presentFence_[currentImageIndex_]) {
        VK_ASSERT(vkWaitForFences(device_, 1, &presentFence_[currentImageIndex_], VK_TRUE, UINT64_MAX));
        VK_ASSERT(vkResetFences(device_, 1, &presentFence_[currentImageIndex_]));
      }
    } else {
      // without VK_EXT_swapchain_maintenance1: use acquire fences to synchronize semaphore reuse
      VK_ASSERT(vkWaitForFences(device_, 1, &acquireFence_[currentImageIndex_], VK_TRUE, UINT64_MAX));
      VK_ASSERT(vkResetFences(device_, 1, &acquireFence_[currentImageIndex_]));

      acquireFence = acquireFence_[currentImageIndex_];
    }

    VkSemaphore acquireSemaphore = acquireSemaphore_[currentImageIndex_];
    // when timeout is set to UINT64_MAX, we wait until the next image has been acquired
    VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, acquireSemaphore, acquireFence, &currentImageIndex_);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR) {
      VK_ASSERT(r);
    }
    getNextImage_ = false;
    ctx_.immediate_->waitSemaphore(acquireSemaphore);
  }

  if (LVK_VERIFY(currentImageIndex_ < numSwapchainImages_)) {
    return swapchainTextures_[currentImageIndex_];
  }

  return {};
}

const VkSurfaceFormatKHR& lvk::VulkanSwapchain::getSurfaceFormat() const {
  return surfaceFormat_;
}

uint32_t lvk::VulkanSwapchain::getNumSwapchainImages() const {
  return numSwapchainImages_;
}

uint32_t lvk::VulkanSwapchain::getSwapchainCurrentImageIndex() const {
  return currentImageIndex_;
}

lvk::Result lvk::VulkanSwapchain::present(VkSemaphore waitSemaphore) {
  LVK_PROFILER_FUNCTION();

  LVK_PROFILER_ZONE("vkQueuePresent()", LVK_PROFILER_COLOR_PRESENT);
  const VkSwapchainPresentFenceInfoEXT fenceInfo = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT,
      .swapchainCount = 1,
      .pFences = &presentFence_[currentImageIndex_],
  };
  const VkPresentInfoKHR pi = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = ctx_.has_EXT_swapchain_maintenance1_ ? &fenceInfo : nullptr,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &waitSemaphore,
      .swapchainCount = 1u,
      .pSwapchains = &swapchain_,
      .pImageIndices = &currentImageIndex_,
  };
  if (ctx_.has_EXT_swapchain_maintenance1_) {
    if (!presentFence_[currentImageIndex_]) {
      presentFence_[currentImageIndex_] = lvk::createFence(device_, "Fence: present-fence");
    }
  }
  VkResult r = vkQueuePresentKHR(graphicsQueue_, &pi);
  if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR) {
    VK_ASSERT(r);
  }
  LVK_PROFILER_ZONE_END();

  // Ready to call acquireNextImage() on the next getCurrentVulkanTexture();
  getNextImage_ = true;
  currentFrameIndex_++;

  LVK_PROFILER_FRAME(nullptr);

  return Result();
}

lvk::VulkanImmediateCommands::VulkanImmediateCommands(VkDevice device,
                                                      uint32_t queueFamilyIndex,
                                                      bool has_EXT_device_fault,
                                                      const char* debugName)
: device_(device)
, queueFamilyIndex_(queueFamilyIndex)
, has_EXT_device_fault_(has_EXT_device_fault)
, debugName_(debugName) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue_);

  const VkCommandPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queueFamilyIndex,
  };
  VK_ASSERT(vkCreateCommandPool(device, &ci, nullptr, &commandPool_));
  VK_ASSERT(lvk::setDebugObjectName(device, VK_OBJECT_TYPE_COMMAND_POOL, (uint64_t)commandPool_, debugName));

  const VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commandPool_,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };

  for (uint32_t i = 0; i != kMaxCommandBuffers; i++) {
    CommandBufferWrapper& buf = buffers_[i];
    char fenceName[256] = {0};
    char semaphoreName[256] = {0};
    if (debugName) {
      snprintf(fenceName, sizeof(fenceName) - 1, "Fence: %s (cmdbuf %u)", debugName, i);
      snprintf(semaphoreName, sizeof(semaphoreName) - 1, "Semaphore: %s (cmdbuf %u)", debugName, i);
    }
    buf.semaphore_ = lvk::createSemaphore(device, semaphoreName);
    buf.fence_ = lvk::createFence(device, fenceName);
    VK_ASSERT(vkAllocateCommandBuffers(device, &ai, &buf.cmdBufAllocated_));
    buffers_[i].handle_.bufferIndex_ = i;
  }
}

lvk::VulkanImmediateCommands::~VulkanImmediateCommands() {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_DESTROY);

  waitAll();

  for (CommandBufferWrapper& buf : buffers_) {
    // lifetimes of all VkFence objects are managed explicitly we do not use deferredTask() for them
    vkDestroyFence(device_, buf.fence_, nullptr);
    vkDestroySemaphore(device_, buf.semaphore_, nullptr);
  }

  vkDestroyCommandPool(device_, commandPool_, nullptr);
}

void lvk::VulkanImmediateCommands::purge() {
  LVK_PROFILER_FUNCTION();

  const uint32_t numBuffers = static_cast<uint32_t>(LVK_ARRAY_NUM_ELEMENTS(buffers_));

  for (uint32_t i = 0; i != numBuffers; i++) {
    // always start checking with the oldest submitted buffer, then wrap around
    CommandBufferWrapper& buf = buffers_[(i + lastSubmitHandle_.bufferIndex_ + 1) % numBuffers];

    if (buf.cmdBuf_ == VK_NULL_HANDLE || buf.isEncoding_) {
      continue;
    }

    const VkResult result = vkWaitForFences(device_, 1, &buf.fence_, VK_TRUE, 0);

    if (result == VK_SUCCESS) {
      VK_ASSERT(vkResetCommandBuffer(buf.cmdBuf_, VkCommandBufferResetFlags{0}));
      VK_ASSERT(vkResetFences(device_, 1, &buf.fence_));
      buf.cmdBuf_ = VK_NULL_HANDLE;
      numAvailableCommandBuffers_++;
    } else {
      if (result != VK_TIMEOUT) {
        VK_ASSERT(result);
      }
    }
  }
}

const lvk::VulkanImmediateCommands::CommandBufferWrapper& lvk::VulkanImmediateCommands::acquire() {
  LVK_PROFILER_FUNCTION();

  if (!numAvailableCommandBuffers_) {
    purge();
  }

  while (!numAvailableCommandBuffers_) {
    LLOGL("Waiting for command buffers...\n");
    LVK_PROFILER_ZONE("Waiting for command buffers...", LVK_PROFILER_COLOR_WAIT);
    purge();
    LVK_PROFILER_ZONE_END();
  }

  VulkanImmediateCommands::CommandBufferWrapper* current = nullptr;

  // we are ok with any available buffer
  for (CommandBufferWrapper& buf : buffers_) {
    if (buf.cmdBuf_ == VK_NULL_HANDLE) {
      current = &buf;
      break;
    }
  }

  // make clang happy
  assert(current);

  LVK_ASSERT_MSG(numAvailableCommandBuffers_, "No available command buffers");
  LVK_ASSERT_MSG(current, "No available command buffers");
  LVK_ASSERT(current->cmdBufAllocated_ != VK_NULL_HANDLE);

  current->handle_.submitId_ = submitCounter_;
  numAvailableCommandBuffers_--;

  current->cmdBuf_ = current->cmdBufAllocated_;
  current->isEncoding_ = true;
  const VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  VK_ASSERT(vkBeginCommandBuffer(current->cmdBuf_, &bi));

  nextSubmitHandle_ = current->handle_;

  return *current;
}

void lvk::VulkanImmediateCommands::wait(const SubmitHandle handle) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_WAIT);

  if (handle.empty()) {
    vkDeviceWaitIdle(device_);
    return;
  }

  if (isReady(handle)) {
    return;
  }

  if (!LVK_VERIFY(!buffers_[handle.bufferIndex_].isEncoding_)) {
    // we are waiting for a buffer which has not been submitted - this is probably a logic error somewhere in the calling code
    return;
  }

  VK_ASSERT(vkWaitForFences(device_, 1, &buffers_[handle.bufferIndex_].fence_, VK_TRUE, UINT64_MAX));

  purge();
}

void lvk::VulkanImmediateCommands::waitAll() {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_WAIT);

  VkFence fences[kMaxCommandBuffers];

  uint32_t numFences = 0;

  for (const CommandBufferWrapper& buf : buffers_) {
    if (buf.cmdBuf_ != VK_NULL_HANDLE && !buf.isEncoding_) {
      fences[numFences++] = buf.fence_;
    }
  }

  if (numFences) {
    VK_ASSERT(vkWaitForFences(device_, numFences, fences, VK_TRUE, UINT64_MAX));
  }

  purge();
}

bool lvk::VulkanImmediateCommands::isReady(const SubmitHandle handle, bool fastCheckNoVulkan) const {
  LVK_ASSERT(handle.bufferIndex_ < kMaxCommandBuffers);

  if (handle.empty()) {
    // a null handle
    return true;
  }

  const CommandBufferWrapper& buf = buffers_[handle.bufferIndex_];

  if (buf.cmdBuf_ == VK_NULL_HANDLE) {
    // already recycled and not yet reused
    return true;
  }

  if (buf.handle_.submitId_ != handle.submitId_) {
    // already recycled and reused by another command buffer
    return true;
  }

  if (fastCheckNoVulkan) {
    // do not ask the Vulkan API about it, just let it retire naturally (when submitId for this bufferIndex gets incremented)
    return false;
  }

  return vkWaitForFences(device_, 1, &buf.fence_, VK_TRUE, 0) == VK_SUCCESS;
}

lvk::SubmitHandle lvk::VulkanImmediateCommands::submit(const CommandBufferWrapper& wrapper) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_SUBMIT);
  LVK_ASSERT(wrapper.isEncoding_);
  VK_ASSERT(vkEndCommandBuffer(wrapper.cmdBuf_));

  VkSemaphoreSubmitInfo waitSemaphores[] = {{}, {}};
  uint32_t numWaitSemaphores = 0;
  if (waitSemaphore_.semaphore) {
    waitSemaphores[numWaitSemaphores++] = waitSemaphore_;
  }
  if (lastSubmitSemaphore_.semaphore) {
    waitSemaphores[numWaitSemaphores++] = lastSubmitSemaphore_;
  }
  VkSemaphoreSubmitInfo signalSemaphores[] = {
      VkSemaphoreSubmitInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                            .semaphore = wrapper.semaphore_,
                            .stageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT},
      {},
  };
  uint32_t numSignalSemaphores = 1;
  if (signalSemaphore_.semaphore) {
    signalSemaphores[numSignalSemaphores++] = signalSemaphore_;
  }

  LVK_PROFILER_ZONE("vkQueueSubmit2()", LVK_PROFILER_COLOR_SUBMIT);
#if LVK_VULKAN_PRINT_COMMANDS
  LLOGL("%p vkQueueSubmit2()\n\n", wrapper.cmdBuf_);
#endif // LVK_VULKAN_PRINT_COMMANDS
  const VkCommandBufferSubmitInfo bufferSI = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = wrapper.cmdBuf_,
  };
  const VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = numWaitSemaphores,
      .pWaitSemaphoreInfos = waitSemaphores,
      .commandBufferInfoCount = 1u,
      .pCommandBufferInfos = &bufferSI,
      .signalSemaphoreInfoCount = numSignalSemaphores,
      .pSignalSemaphoreInfos = signalSemaphores,
  };
  const VkResult result = vkQueueSubmit2(queue_, 1u, &si, wrapper.fence_);
  if (has_EXT_device_fault_ && result == VK_ERROR_DEVICE_LOST) {
    VkDeviceFaultCountsEXT count = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT,
    };
    vkGetDeviceFaultInfoEXT(device_, &count, nullptr);
    std::vector<VkDeviceFaultAddressInfoEXT> addressInfo(count.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendorInfo(count.vendorInfoCount);
    std::vector<uint8_t> binary(count.vendorBinarySize);
    VkDeviceFaultInfoEXT info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
        .pAddressInfos = addressInfo.data(),
        .pVendorInfos = vendorInfo.data(),
        .pVendorBinaryData = binary.data(),
    };
    vkGetDeviceFaultInfoEXT(device_, &count, &info);
    LLOGW("VK_ERROR_DEVICE_LOST: %s\n", info.description);
    for (const VkDeviceFaultAddressInfoEXT& aInfo : addressInfo) {
      VkDeviceSize lowerAddress = aInfo.reportedAddress & ~(aInfo.addressPrecision - 1);
      VkDeviceSize upperAddress = aInfo.reportedAddress | (aInfo.addressPrecision - 1);
      LLOGW("...address range [ %" PRIx64 ", %" PRIx64 " ]: %s\n",
            lowerAddress,
            upperAddress,
            getVkDeviceFaultAddressTypeString(aInfo.addressType));
    }
    for (const VkDeviceFaultVendorInfoEXT& vInfo : vendorInfo) {
      LLOGW("...caused by `%s` with error code %" PRIx64 " and data %" PRIx64 "\n",
            vInfo.description,
            vInfo.vendorFaultCode,
            vInfo.vendorFaultData);
    }
    const VkDeviceSize binarySize = count.vendorBinarySize;
    if (info.pVendorBinaryData && binarySize >= sizeof(VkDeviceFaultVendorBinaryHeaderVersionOneEXT)) {
      const VkDeviceFaultVendorBinaryHeaderVersionOneEXT* header =
          std::launder(reinterpret_cast<const VkDeviceFaultVendorBinaryHeaderVersionOneEXT*>(info.pVendorBinaryData));
      const char hexDigits[] = "0123456789abcdef";
      char uuid[VK_UUID_SIZE * 2 + 1] = {};
      for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        uuid[i * 2 + 0] = hexDigits[(header->pipelineCacheUUID[i] >> 4) & 0xF];
        uuid[i * 2 + 1] = hexDigits[header->pipelineCacheUUID[i] & 0xF];
      }
      LLOGW("VkDeviceFaultVendorBinaryHeaderVersionOne:");
      LLOGW("   headerSize        : %u\n", header->headerSize);
      LLOGW("   headerVersion     : %u\n", (uint32_t)header->headerVersion);
      LLOGW("   vendorID          : %u\n", header->vendorID);
      LLOGW("   deviceID          : %u\n", header->deviceID);
      LLOGW("   driverVersion     : %u\n", header->driverVersion);
      LLOGW("   pipelineCacheUUID : %s\n", uuid);
      if (header->applicationNameOffset && header->applicationNameOffset < binarySize) {
        LLOGW("   applicationName   : %s\n", (const char*)info.pVendorBinaryData + header->applicationNameOffset);
      }
      LLOGW("   applicationVersion: %i.%i.%i\n",
            VK_API_VERSION_MAJOR(header->applicationVersion),
            VK_API_VERSION_MINOR(header->applicationVersion),
            VK_API_VERSION_PATCH(header->applicationVersion));
      if (header->engineNameOffset && header->engineNameOffset < binarySize) {
        LLOGW("   engineName        : %s\n", (const char*)info.pVendorBinaryData + header->engineNameOffset);
      }
      LLOGW("   engineVersion     : %i.%i.%i\n",
            VK_API_VERSION_MAJOR(header->engineVersion),
            VK_API_VERSION_MINOR(header->engineVersion),
            VK_API_VERSION_PATCH(header->engineVersion));
      LLOGW("   apiVersion        : %i.%i.%i.%i\n",
            VK_API_VERSION_MAJOR(header->apiVersion),
            VK_API_VERSION_MINOR(header->apiVersion),
            VK_API_VERSION_PATCH(header->apiVersion),
            VK_API_VERSION_VARIANT(header->apiVersion));
    }
  }
  VK_ASSERT(result);
  LVK_PROFILER_ZONE_END();

  lastSubmitSemaphore_.semaphore = wrapper.semaphore_;
  lastSubmitHandle_ = wrapper.handle_;
  waitSemaphore_.semaphore = VK_NULL_HANDLE;
  signalSemaphore_.semaphore = VK_NULL_HANDLE;

  // reset
  const_cast<CommandBufferWrapper&>(wrapper).isEncoding_ = false;
  submitCounter_++;

  if (!submitCounter_) {
    // skip the 0 value - when uint32_t wraps around (null SubmitHandle)
    submitCounter_++;
  }

  return lastSubmitHandle_;
}

void lvk::VulkanImmediateCommands::waitSemaphore(VkSemaphore semaphore) {
  LVK_ASSERT(waitSemaphore_.semaphore == VK_NULL_HANDLE);

  waitSemaphore_.semaphore = semaphore;
}

void lvk::VulkanImmediateCommands::signalSemaphore(VkSemaphore semaphore, uint64_t signalValue) {
  LVK_ASSERT(signalSemaphore_.semaphore == VK_NULL_HANDLE);

  signalSemaphore_.semaphore = semaphore;
  signalSemaphore_.value = signalValue;
}

VkSemaphore lvk::VulkanImmediateCommands::acquireLastSubmitSemaphore() {
  return std::exchange(lastSubmitSemaphore_.semaphore, VK_NULL_HANDLE);
}

VkFence lvk::VulkanImmediateCommands::getVkFence(lvk::SubmitHandle handle) const {
  if (handle.empty()) {
    return VK_NULL_HANDLE;
  }

  return buffers_[handle.bufferIndex_].fence_;
}

lvk::SubmitHandle lvk::VulkanImmediateCommands::getLastSubmitHandle() const {
  return lastSubmitHandle_;
}

lvk::SubmitHandle lvk::VulkanImmediateCommands::getNextSubmitHandle() const {
  return nextSubmitHandle_;
}

lvk::VulkanPipelineBuilder::VulkanPipelineBuilder()
: vertexInputState_(VkPipelineVertexInputStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 0,
      .pVertexBindingDescriptions = nullptr,
      .vertexAttributeDescriptionCount = 0,
      .pVertexAttributeDescriptions = nullptr,
  })
, inputAssembly_({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
  })
, tessellationState_({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      .flags = 0,
      .patchControlPoints = 0,
  })
, rasterizationState_({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0.0f,
      .depthBiasClamp = 0.0f,
      .depthBiasSlopeFactor = 0.0f,
      .lineWidth = 1.0f,
  })
, multisampleState_({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 0.0f,
      .pSampleMask = nullptr,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE,
  })
, depthStencilState_({
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = VK_FALSE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
      .depthBoundsTestEnable = VK_FALSE,
      .stencilTestEnable = VK_FALSE,
      .front =
          {
              .failOp = VK_STENCIL_OP_KEEP,
              .passOp = VK_STENCIL_OP_KEEP,
              .depthFailOp = VK_STENCIL_OP_KEEP,
              .compareOp = VK_COMPARE_OP_NEVER,
              .compareMask = 0,
              .writeMask = 0,
              .reference = 0,
          },
      .back =
          {
              .failOp = VK_STENCIL_OP_KEEP,
              .passOp = VK_STENCIL_OP_KEEP,
              .depthFailOp = VK_STENCIL_OP_KEEP,
              .compareOp = VK_COMPARE_OP_NEVER,
              .compareMask = 0,
              .writeMask = 0,
              .reference = 0,
          },
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f,
  }) {}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::dynamicState(VkDynamicState state) {
  LVK_ASSERT(numDynamicStates_ < LVK_MAX_DYNAMIC_STATES);
  dynamicStates_[numDynamicStates_++] = state;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::primitiveTopology(VkPrimitiveTopology topology) {
  inputAssembly_.topology = topology;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::rasterizationSamples(VkSampleCountFlagBits samples, float minSampleShading) {
  multisampleState_.rasterizationSamples = samples;
  multisampleState_.sampleShadingEnable = minSampleShading > 0 ? VK_TRUE : VK_FALSE;
  multisampleState_.minSampleShading = minSampleShading;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::cullMode(VkCullModeFlags mode) {
  rasterizationState_.cullMode = mode;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::frontFace(VkFrontFace mode) {
  rasterizationState_.frontFace = mode;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::polygonMode(VkPolygonMode mode) {
  rasterizationState_.polygonMode = mode;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::vertexInputState(const VkPipelineVertexInputStateCreateInfo& state) {
  vertexInputState_ = state;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::viewMask(uint32_t mask) {
  viewMask_ = mask;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::colorAttachments(const VkPipelineColorBlendAttachmentState* states,
                                                                         const VkFormat* formats,
                                                                         uint32_t numColorAttachments) {
  LVK_ASSERT(states);
  LVK_ASSERT(formats);
  LVK_ASSERT(numColorAttachments <= LVK_ARRAY_NUM_ELEMENTS(colorBlendAttachmentStates_));
  LVK_ASSERT(numColorAttachments <= LVK_ARRAY_NUM_ELEMENTS(colorAttachmentFormats_));
  for (uint32_t i = 0; i != numColorAttachments; i++) {
    colorBlendAttachmentStates_[i] = states[i];
    colorAttachmentFormats_[i] = formats[i];
  }
  numColorAttachments_ = numColorAttachments;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::depthAttachmentFormat(VkFormat format) {
  depthAttachmentFormat_ = format;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::stencilAttachmentFormat(VkFormat format) {
  stencilAttachmentFormat_ = format;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::patchControlPoints(uint32_t numPoints) {
  tessellationState_.patchControlPoints = numPoints;
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::shaderStage(VkPipelineShaderStageCreateInfo stage) {
  if (stage.pNext) {
    LVK_ASSERT(numShaderStages_ < LVK_ARRAY_NUM_ELEMENTS(shaderStages_));
    shaderStages_[numShaderStages_++] = stage;
  }
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::stencilStateOps(VkStencilFaceFlags faceMask,
                                                                        VkStencilOp failOp,
                                                                        VkStencilOp passOp,
                                                                        VkStencilOp depthFailOp,
                                                                        VkCompareOp compareOp) {
  depthStencilState_.stencilTestEnable = depthStencilState_.stencilTestEnable == VK_TRUE || failOp != VK_STENCIL_OP_KEEP ||
                                                 passOp != VK_STENCIL_OP_KEEP || depthFailOp != VK_STENCIL_OP_KEEP ||
                                                 compareOp != VK_COMPARE_OP_ALWAYS
                                             ? VK_TRUE
                                             : VK_FALSE;

  if (faceMask & VK_STENCIL_FACE_FRONT_BIT) {
    VkStencilOpState& s = depthStencilState_.front;
    s.failOp = failOp;
    s.passOp = passOp;
    s.depthFailOp = depthFailOp;
    s.compareOp = compareOp;
  }

  if (faceMask & VK_STENCIL_FACE_BACK_BIT) {
    VkStencilOpState& s = depthStencilState_.back;
    s.failOp = failOp;
    s.passOp = passOp;
    s.depthFailOp = depthFailOp;
    s.compareOp = compareOp;
  }
  return *this;
}

lvk::VulkanPipelineBuilder& lvk::VulkanPipelineBuilder::stencilMasks(VkStencilFaceFlags faceMask,
                                                                     uint32_t compareMask,
                                                                     uint32_t writeMask,
                                                                     uint32_t reference) {
  if (faceMask & VK_STENCIL_FACE_FRONT_BIT) {
    VkStencilOpState& s = depthStencilState_.front;
    s.compareMask = compareMask;
    s.writeMask = writeMask;
    s.reference = reference;
  }

  if (faceMask & VK_STENCIL_FACE_BACK_BIT) {
    VkStencilOpState& s = depthStencilState_.back;
    s.compareMask = compareMask;
    s.writeMask = writeMask;
    s.reference = reference;
  }
  return *this;
}

VkResult lvk::VulkanPipelineBuilder::build(VkDevice device,
                                           VkPipelineCache pipelineCache,
                                           VkPipelineLayout pipelineLayout,
                                           VkPipeline* outPipeline,
                                           const char* debugName) noexcept {
  const VkPipelineDynamicStateCreateInfo dynamicState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = numDynamicStates_,
      .pDynamicStates = dynamicStates_,
  };
  // viewport and scissor can be NULL if the viewport state is dynamic
  // https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkPipelineViewportStateCreateInfo.html
  const VkPipelineViewportStateCreateInfo viewportState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = nullptr,
      .scissorCount = 1,
      .pScissors = nullptr,
  };
  const VkPipelineColorBlendStateCreateInfo colorBlendState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = numColorAttachments_,
      .pAttachments = colorBlendAttachmentStates_,
  };
  const VkPipelineRenderingCreateInfo renderingInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
      .pNext = nullptr,
      .viewMask = viewMask_,
      .colorAttachmentCount = numColorAttachments_,
      .pColorAttachmentFormats = colorAttachmentFormats_,
      .depthAttachmentFormat = depthAttachmentFormat_,
      .stencilAttachmentFormat = stencilAttachmentFormat_,
  };

  const VkGraphicsPipelineCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderingInfo,
      .flags = 0,
      .stageCount = numShaderStages_,
      .pStages = shaderStages_,
      .pVertexInputState = &vertexInputState_,
      .pInputAssemblyState = &inputAssembly_,
      .pTessellationState = &tessellationState_,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterizationState_,
      .pMultisampleState = &multisampleState_,
      .pDepthStencilState = &depthStencilState_,
      .pColorBlendState = &colorBlendState,
      .pDynamicState = &dynamicState,
      .layout = pipelineLayout,
      .renderPass = VK_NULL_HANDLE,
      .subpass = 0,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = -1,
  };

#if defined(ANDROID)
  LLOGD("vkCreateGraphicsPipelines(): %s\n", debugName);
#endif // defined(ANDROID)
  const VkResult result = vkCreateGraphicsPipelines(device, pipelineCache, 1, &ci, nullptr, outPipeline);

  if (!LVK_VERIFY(result == VK_SUCCESS)) {
    return result;
  }

  numPipelinesCreated_++;

  // set debug name
  return lvk::setDebugObjectName(device, VK_OBJECT_TYPE_PIPELINE, (uint64_t)*outPipeline, debugName);
}

lvk::CommandBuffer::CommandBuffer(VulkanContext* ctx) : ctx_(ctx), wrapper_(&ctx_->immediate_->acquire()) {}

lvk::CommandBuffer::~CommandBuffer() {
  // did you forget to call cmdEndRendering()?
  LVK_ASSERT(!isRendering_);
}

void lvk::CommandBuffer::transitionToShaderReadOnly(TextureHandle handle) const {
  LVK_PROFILER_FUNCTION();

  const lvk::VulkanImage& img = *ctx_->texturesPool_.get(handle);

  LVK_ASSERT(!img.isSwapchainImage_);

  // transition only non-multisampled images - MSAA images cannot be accessed from shaders
  if (img.vkSamples_ == VK_SAMPLE_COUNT_1_BIT) {
    const VkImageAspectFlags flags = img.getImageAspectFlags();
    // set the result of the previous render pass
    img.transitionLayout(wrapper_->cmdBuf_,
                         img.isSampledImage() ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
                         VkImageSubresourceRange{flags, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
  }
}

void lvk::CommandBuffer::transitionToRenderingLocalRead(TextureHandle handle) const {
  LVK_PROFILER_FUNCTION();

  const lvk::VulkanImage& img = *ctx_->texturesPool_.get(handle);

  LVK_ASSERT(!img.isSwapchainImage_);
  LVK_ASSERT_MSG(img.vkUsageFlags_ & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                 "Input attachment texture must have VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT (lvk::TextureUsageBits_InputAttachment)");

  const VkImageAspectFlags flags = img.getImageAspectFlags();
  img.transitionLayout(wrapper_->cmdBuf_,
                       VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR,
                       VkImageSubresourceRange{flags, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
}

void lvk::CommandBuffer::cmdBindRayTracingPipeline(lvk::RayTracingPipelineHandle handle) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(!handle.empty() && ctx_->has_KHR_ray_tracing_pipeline_)) {
    return;
  }

  currentPipelineGraphics_ = {};
  currentPipelineCompute_ = {};
  currentPipelineRayTracing_ = handle;

  VkPipeline pipeline = ctx_->getVkPipeline(handle);

  const lvk::RayTracingPipelineState* rtps = ctx_->rayTracingPipelinesPool_.get(handle);

  LVK_ASSERT(rtps);
  LVK_ASSERT(pipeline != VK_NULL_HANDLE);

  if (lastPipelineBound_ != pipeline) {
    lastPipelineBound_ = pipeline;
    vkCmdBindPipeline(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    ctx_->checkAndUpdateDescriptorSets();
    ctx_->bindDefaultDescriptorSets(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtps->pipelineLayout_);
  }
}

void lvk::CommandBuffer::cmdBindComputePipeline(lvk::ComputePipelineHandle handle) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(!handle.empty())) {
    return;
  }

  currentPipelineGraphics_ = {};
  currentPipelineCompute_ = handle;
  currentPipelineRayTracing_ = {};

  VkPipeline pipeline = ctx_->getVkPipeline(handle);

  const lvk::ComputePipelineState* cps = ctx_->computePipelinesPool_.get(handle);

  LVK_ASSERT(cps);
  LVK_ASSERT(pipeline != VK_NULL_HANDLE);

  if (lastPipelineBound_ != pipeline) {
    lastPipelineBound_ = pipeline;
    vkCmdBindPipeline(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    ctx_->checkAndUpdateDescriptorSets();
    ctx_->bindDefaultDescriptorSets(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_COMPUTE, cps->pipelineLayout_);
  }
}

void lvk::CommandBuffer::cmdDispatchThreadGroups(const Dimensions& threadgroupCount, const Dependencies& deps) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDispatchThreadGroups()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DISPATCH);

  LVK_ASSERT(!isRendering_);

  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.textures[i]; i++) {
    useComputeTexture(deps.textures[i], VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }
  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.buffers[i]; i++) {
    const lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(deps.buffers[i]);
    LVK_ASSERT_MSG(buf->vkUsageFlags_ & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   "Did you forget to specify BufferUsageBits_Storage on your buffer?");
    bufferBarrier(deps.buffers[i],
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }

  vkCmdDispatch(wrapper_->cmdBuf_, threadgroupCount.width, threadgroupCount.height, threadgroupCount.depth);
}

void lvk::CommandBuffer::cmdPushDebugGroupLabel(const char* label, uint32_t colorRGBA) const {
  LVK_ASSERT(label);

  if (!label || !vkCmdBeginDebugUtilsLabelEXT) {
    return;
  }
  const VkDebugUtilsLabelEXT utilsLabel = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      .pNext = nullptr,
      .pLabelName = label,
      .color = {float((colorRGBA >> 0) & 0xff) / 255.0f,
                float((colorRGBA >> 8) & 0xff) / 255.0f,
                float((colorRGBA >> 16) & 0xff) / 255.0f,
                float((colorRGBA >> 24) & 0xff) / 255.0f},
  };
  vkCmdBeginDebugUtilsLabelEXT(wrapper_->cmdBuf_, &utilsLabel);
}

void lvk::CommandBuffer::cmdInsertDebugEventLabel(const char* label, uint32_t colorRGBA) const {
  LVK_ASSERT(label);

  if (!label || !vkCmdInsertDebugUtilsLabelEXT) {
    return;
  }
  const VkDebugUtilsLabelEXT utilsLabel = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      .pNext = nullptr,
      .pLabelName = label,
      .color = {float((colorRGBA >> 0) & 0xff) / 255.0f,
                float((colorRGBA >> 8) & 0xff) / 255.0f,
                float((colorRGBA >> 16) & 0xff) / 255.0f,
                float((colorRGBA >> 24) & 0xff) / 255.0f},
  };
  vkCmdInsertDebugUtilsLabelEXT(wrapper_->cmdBuf_, &utilsLabel);
}

void lvk::CommandBuffer::cmdPopDebugGroupLabel() const {
  if (!vkCmdEndDebugUtilsLabelEXT) {
    return;
  }
  vkCmdEndDebugUtilsLabelEXT(wrapper_->cmdBuf_);
}

void lvk::CommandBuffer::useComputeTexture(TextureHandle handle, VkPipelineStageFlags2 dstStage) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_BARRIER);

  LVK_ASSERT(!handle.empty());
  lvk::VulkanImage& tex = *ctx_->texturesPool_.get(handle);

  (void)dstStage; // TODO: add extra dstStage

  if (!tex.isStorageImage() && !tex.isSampledImage()) {
    LVK_ASSERT_MSG(false, "Did you forget to specify TextureUsageBits::Storage or TextureUsageBits::Sampled on your texture?");
    return;
  }

  tex.transitionLayout(wrapper_->cmdBuf_,
                       tex.isStorageImage() ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VkImageSubresourceRange{tex.getImageAspectFlags(), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
}

void lvk::CommandBuffer::bufferBarrier(BufferHandle handle, VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_BARRIER);

  lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(handle);

  VkBufferMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = srcStage,
      .srcAccessMask = 0,
      .dstStageMask = dstStage,
      .dstAccessMask = 0,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buf->vkBuffer_,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
  };

  if (srcStage & VK_PIPELINE_STAGE_2_TRANSFER_BIT) {
    barrier.srcAccessMask |= VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  } else {
    barrier.srcAccessMask |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  }

  if (dstStage & VK_PIPELINE_STAGE_2_TRANSFER_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
  } else {
    barrier.dstAccessMask |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
  }
  if (dstStage & VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
  }
  if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) {
    barrier.dstAccessMask |= VK_ACCESS_2_INDEX_READ_BIT;
  }

  const VkDependencyInfo depInfo = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };

  vkCmdPipelineBarrier2(wrapper_->cmdBuf_, &depInfo);
}

void lvk::CommandBuffer::cmdBeginRendering(const lvk::RenderPass& renderPass, const lvk::Framebuffer& fb, const Dependencies& deps) {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT(!isRendering_);

  isRendering_ = true;
  viewMask_ = renderPass.viewMask;

  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.textures[i]; i++) {
    transitionToShaderReadOnly(deps.textures[i]);
  }
  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.buffers[i]; i++) {
    VkPipelineStageFlags2 dstStageFlags = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    const lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(deps.buffers[i]);
    LVK_ASSERT(buf);
    if ((buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) || (buf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)) {
      dstStageFlags |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
    }
    if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
      dstStageFlags |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }
    bufferBarrier(deps.buffers[i], VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, dstStageFlags);
  }

  const uint32_t numFbColorAttachments = fb.getNumColorAttachments();
  const uint32_t numPassColorAttachments = renderPass.getNumColorAttachments();

  LVK_ASSERT(numPassColorAttachments == numFbColorAttachments);

  framebuffer_ = fb;

  // transition all the color attachments
  for (uint32_t i = 0; i != numFbColorAttachments; i++) {
    if (TextureHandle handle = fb.color[i].texture) {
      lvk::VulkanImage* colorTex = ctx_->texturesPool_.get(handle);
      transitionToColorAttachment(wrapper_->cmdBuf_, colorTex);
    }
    // handle MSAA
    if (TextureHandle handle = fb.color[i].resolveTexture) {
      lvk::VulkanImage* colorResolveTex = ctx_->texturesPool_.get(handle);
      colorResolveTex->isResolveAttachment = true;
      transitionToColorAttachment(wrapper_->cmdBuf_, colorResolveTex);
    }
  }
  // transition depth-stencil attachment
  TextureHandle depthTex = fb.depthStencil.texture;
  if (depthTex) {
    const lvk::VulkanImage& depthImg = *ctx_->texturesPool_.get(depthTex);
    LVK_ASSERT_MSG(depthImg.vkImageFormat_ != VK_FORMAT_UNDEFINED, "Invalid depth attachment format");
    LVK_ASSERT_MSG(depthImg.isDepthFormat_, "Invalid depth attachment format");
    const VkImageAspectFlags flags = depthImg.getImageAspectFlags();
    depthImg.transitionLayout(wrapper_->cmdBuf_,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VkImageSubresourceRange{flags, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
  }
  // handle depth MSAA
  if (TextureHandle handle = fb.depthStencil.resolveTexture) {
    lvk::VulkanImage& depthResolveImg = *ctx_->texturesPool_.get(handle);
    LVK_ASSERT_MSG(depthResolveImg.isDepthFormat_, "Invalid resolve depth attachment format");
    depthResolveImg.isResolveAttachment = true;
    const VkImageAspectFlags flags = depthResolveImg.getImageAspectFlags();
    depthResolveImg.transitionLayout(wrapper_->cmdBuf_,
                                     VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     VkImageSubresourceRange{flags, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
  }

  // calculate and transition input attachments
  {
    uint32_t i = 0;
    while (i != LVK_MAX_COLOR_ATTACHMENTS && deps.inputAttachments[i]) {
      const lvk::TextureHandle handle = deps.inputAttachments[i];
      const lvk::VulkanImage* tex = ctx_->texturesPool_.get(handle);
      LVK_ASSERT(tex);
      transitionToRenderingLocalRead(handle);
      inputAttachments_.imageInfos[i] = {
          .sampler = VK_NULL_HANDLE,
          .imageView = tex->imageView_,
          .imageLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR,
      };
      inputAttachments_.writes[i] = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = VK_NULL_HANDLE, // ignored for push descriptors
          .dstBinding = i,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
          .pImageInfo = &inputAttachments_.imageInfos[i],
      };
      i++;
    }
    inputAttachments_.count = i;
  }

  VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
  uint32_t mipLevel = 0;
  uint32_t fbWidth = 0;
  uint32_t fbHeight = 0;

  VkRenderingAttachmentInfo colorAttachments[LVK_MAX_COLOR_ATTACHMENTS];

  for (uint32_t i = 0; i != numFbColorAttachments; i++) {
    const lvk::Framebuffer::AttachmentDesc& attachment = fb.color[i];
    LVK_ASSERT(!attachment.texture.empty());

    lvk::VulkanImage& colorTexture = *ctx_->texturesPool_.get(attachment.texture);
    const lvk::RenderPass::AttachmentDesc& descColor = renderPass.color[i];
    if (mipLevel && descColor.level) {
      LVK_ASSERT_MSG(descColor.level == mipLevel, "All color attachments should have the same mip-level");
    }
    const VkExtent3D dim = colorTexture.vkExtent_;
    if (fbWidth) {
      LVK_ASSERT_MSG(dim.width == fbWidth, "All attachments should have the same width");
    }
    if (fbHeight) {
      LVK_ASSERT_MSG(dim.height == fbHeight, "All attachments should have the same height");
    }
    mipLevel = descColor.level;
    fbWidth = dim.width;
    fbHeight = dim.height;
    samples = colorTexture.vkSamples_;
    colorAttachments[i] = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = colorTexture.getOrCreateVkImageViewForFramebuffer(*ctx_, descColor.level, descColor.layer, viewMask_),
        .imageLayout = colorTexture.vkImageLayout_, // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = (samples > 1) ? resolveModeToVkResolveModeFlagBits(descColor.resolveMode, VK_RESOLVE_MODE_FLAG_BITS_MAX_ENUM)
                                     : VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = loadOpToVkAttachmentLoadOp(descColor.loadOp),
        .storeOp = storeOpToVkAttachmentStoreOp(descColor.storeOp),
    };
    memcpy(&colorAttachments[i].clearValue.color, &descColor.clearColor, sizeof(descColor.clearColor));
    // handle MSAA
    if (descColor.storeOp == StoreOp_MsaaResolve) {
      LVK_ASSERT(samples > 1);
      LVK_ASSERT_MSG(!attachment.resolveTexture.empty(), "Framebuffer attachment should contain a resolve texture");
      lvk::VulkanImage& colorResolveTexture = *ctx_->texturesPool_.get(attachment.resolveTexture);
      colorAttachments[i].resolveImageView =
          colorResolveTexture.getOrCreateVkImageViewForFramebuffer(*ctx_, descColor.level, descColor.layer, viewMask_);
      colorAttachments[i].resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
  }

  VkRenderingAttachmentInfo depthAttachment = {};

  if (fb.depthStencil.texture) {
    lvk::VulkanImage& depthTexture = *ctx_->texturesPool_.get(fb.depthStencil.texture);
    const RenderPass::AttachmentDesc& descDepth = renderPass.depth;
    LVK_ASSERT_MSG(descDepth.level == mipLevel, "Depth attachment should have the same mip-level as color attachments");
    depthAttachment = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .pNext = nullptr,
        .imageView = depthTexture.getOrCreateVkImageViewForFramebuffer(*ctx_, descDepth.level, descDepth.layer, viewMask_),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .resolveMode = VK_RESOLVE_MODE_NONE,
        .resolveImageView = VK_NULL_HANDLE,
        .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp = loadOpToVkAttachmentLoadOp(descDepth.loadOp),
        .storeOp = storeOpToVkAttachmentStoreOp(descDepth.storeOp),
        .clearValue = {.depthStencil = {.depth = descDepth.clearDepth, .stencil = descDepth.clearStencil}},
    };
    // handle depth MSAA
    if (descDepth.storeOp == StoreOp_MsaaResolve) {
      LVK_ASSERT(depthTexture.vkSamples_ == samples);
      const lvk::Framebuffer::AttachmentDesc& attachment = fb.depthStencil;
      LVK_ASSERT_MSG(!attachment.resolveTexture.empty(), "Framebuffer depth attachment should contain a resolve texture");
      lvk::VulkanImage& depthResolveTexture = *ctx_->texturesPool_.get(attachment.resolveTexture);
      depthAttachment.resolveImageView =
          depthResolveTexture.getOrCreateVkImageViewForFramebuffer(*ctx_, descDepth.level, descDepth.layer, viewMask_);
      depthAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depthAttachment.resolveMode =
          resolveModeToVkResolveModeFlagBits(descDepth.resolveMode, ctx_->vkPhysicalDeviceVulkan12Properties_.supportedDepthResolveModes);
    }
    const VkExtent3D dim = depthTexture.vkExtent_;
    if (fbWidth) {
      LVK_ASSERT_MSG(dim.width == fbWidth, "All attachments should have the same width");
    }
    if (fbHeight) {
      LVK_ASSERT_MSG(dim.height == fbHeight, "All attachments should have the same height");
    }
    mipLevel = descDepth.level;
    fbWidth = dim.width;
    fbHeight = dim.height;
  }

  const uint32_t width = std::max(fbWidth >> mipLevel, 1u);
  const uint32_t height = std::max(fbHeight >> mipLevel, 1u);
  const lvk::Viewport viewport = {0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f};
  const lvk::ScissorRect scissor = {0, 0, width, height};

  VkRenderingAttachmentInfo stencilAttachment = depthAttachment;

  const bool isStencilFormat = renderPass.stencil.loadOp != lvk::LoadOp_Invalid;

  const VkRenderingInfo renderingInfo = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .pNext = nullptr,
      .flags = 0,
      .renderArea = {VkOffset2D{(int32_t)scissor.x, (int32_t)scissor.y}, VkExtent2D{scissor.width, scissor.height}},
      .layerCount = renderPass.layerCount,
      .viewMask = renderPass.viewMask,
      .colorAttachmentCount = numFbColorAttachments,
      .pColorAttachments = colorAttachments,
      .pDepthAttachment = depthTex ? &depthAttachment : nullptr,
      .pStencilAttachment = isStencilFormat ? &stencilAttachment : nullptr,
  };

  cmdBindViewport(viewport);
  cmdBindScissorRect(scissor);
  cmdBindDepthState({});

  ctx_->checkAndUpdateDescriptorSets();

  vkCmdSetDepthCompareOp(wrapper_->cmdBuf_, VK_COMPARE_OP_ALWAYS);
  vkCmdSetDepthBiasEnable(wrapper_->cmdBuf_, VK_FALSE);

  vkCmdBeginRendering(wrapper_->cmdBuf_, &renderingInfo);
}

void lvk::CommandBuffer::cmdEndRendering() {
  LVK_ASSERT(isRendering_);

  isRendering_ = false;

  vkCmdEndRendering(wrapper_->cmdBuf_);

  framebuffer_ = {};
}

void lvk::CommandBuffer::cmdNextSubpass() {
  LVK_ASSERT(isRendering_);

  const VkMemoryBarrier2 memoryBarrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
  };

  const VkDependencyInfo dependencyInfo = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memoryBarrier,
  };

  vkCmdPipelineBarrier2(wrapper_->cmdBuf_, &dependencyInfo);
}

void lvk::CommandBuffer::cmdBindViewport(const Viewport& viewport) {
  // https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
  const VkViewport vp = {
      .x = viewport.x, // float x;
      .y = viewport.height - viewport.y, // float y;
      .width = viewport.width, // float width;
      .height = -viewport.height, // float height;
      .minDepth = viewport.minDepth, // float minDepth;
      .maxDepth = viewport.maxDepth, // float maxDepth;
  };
  vkCmdSetViewport(wrapper_->cmdBuf_, 0, 1, &vp);
}

void lvk::CommandBuffer::cmdBindScissorRect(const ScissorRect& rect) {
  const VkRect2D scissor = {
      VkOffset2D{(int32_t)rect.x, (int32_t)rect.y},
      VkExtent2D{rect.width, rect.height},
  };
  vkCmdSetScissor(wrapper_->cmdBuf_, 0, 1, &scissor);
}

void lvk::CommandBuffer::cmdBindRenderPipeline(lvk::RenderPipelineHandle handle) {
  if (!LVK_VERIFY(!handle.empty())) {
    return;
  }

  currentPipelineGraphics_ = handle;
  currentPipelineCompute_ = {};
  currentPipelineRayTracing_ = {};

  const lvk::RenderPipelineState* rps = ctx_->renderPipelinesPool_.get(handle);

  LVK_ASSERT(rps);

  const bool hasDepthAttachmentPipeline = rps->desc_.depthFormat != Format_Invalid;
  const bool hasDepthAttachmentPass = !framebuffer_.depthStencil.texture.empty();

  if (hasDepthAttachmentPipeline != hasDepthAttachmentPass) {
    LVK_ASSERT(false);
    LLOGW("Make sure your render pass and render pipeline both have matching depth attachments");
  }

  VkPipeline pipeline = ctx_->getVkPipeline(handle, viewMask_);

  LVK_ASSERT(pipeline != VK_NULL_HANDLE);

  if (lastPipelineBound_ != pipeline) {
    lastPipelineBound_ = pipeline;
    vkCmdBindPipeline(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    ctx_->bindDefaultDescriptorSets(wrapper_->cmdBuf_, VK_PIPELINE_BIND_POINT_GRAPHICS, rps->pipelineLayout_);
    if (inputAttachments_.count) {
      vkCmdPushDescriptorSetKHR(wrapper_->cmdBuf_,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                rps->pipelineLayout_,
                                kDescriptorSet_InputAttachments,
                                inputAttachments_.count,
                                inputAttachments_.writes);
    }
  }
}

void lvk::CommandBuffer::cmdBindDepthState(const DepthState& desc) {
  LVK_PROFILER_FUNCTION();

  const VkCompareOp op = compareOpToVkCompareOp(desc.compareOp);
  vkCmdSetDepthWriteEnable(wrapper_->cmdBuf_, desc.isDepthWriteEnabled ? VK_TRUE : VK_FALSE);
  vkCmdSetDepthTestEnable(wrapper_->cmdBuf_, (op != VK_COMPARE_OP_ALWAYS || desc.isDepthWriteEnabled) ? VK_TRUE : VK_FALSE);

#if defined(ANDROID)
  // This is a workaround for the issue.
  // On Android (Mali-G715-Immortalis MC11 v1.r38p1-01eac0.c1a71ccca2acf211eb87c5db5322f569)
  // if depth-stencil texture is not set, call of vkCmdSetDepthCompareOp leads to disappearing of all content.
  if (!framebuffer_.depthStencil.texture) {
    return;
  }
#endif
  vkCmdSetDepthCompareOp(wrapper_->cmdBuf_, op);
}

void lvk::CommandBuffer::cmdBindVertexBuffer(uint32_t index, BufferHandle buffer, uint64_t bufferOffset) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(!buffer.empty())) {
    return;
  }

  lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(buffer);

  LVK_ASSERT(buf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  vkCmdBindVertexBuffers2(wrapper_->cmdBuf_, index, 1, &buf->vkBuffer_, &bufferOffset, nullptr, nullptr);
}

void lvk::CommandBuffer::cmdBindIndexBuffer(BufferHandle indexBuffer, IndexFormat indexFormat, uint64_t indexBufferOffset) {
  lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(indexBuffer);

  LVK_ASSERT(buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

  const VkIndexType type = indexFormatToVkIndexType(indexFormat);
  vkCmdBindIndexBuffer(wrapper_->cmdBuf_, buf->vkBuffer_, indexBufferOffset, type);
}

void lvk::CommandBuffer::cmdPushConstants(const void* data, size_t size, size_t offset) {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT(size % 4 == 0); // VUID-vkCmdPushConstants-size-00369: size must be a multiple of 4

  // check push constant size is within max size
  const VkPhysicalDeviceLimits& limits = ctx_->getVkPhysicalDeviceProperties().limits;
  if (!LVK_VERIFY(size + offset <= limits.maxPushConstantsSize)) {
    LLOGW("Push constants size exceeded %u (max %u bytes)", size + offset, limits.maxPushConstantsSize);
  }

  if (currentPipelineGraphics_.empty() && currentPipelineCompute_.empty() && currentPipelineRayTracing_.empty()) {
    LVK_ASSERT_MSG(false, "No pipeline bound - cannot set push constants");
    return;
  }

  const lvk::RenderPipelineState* stateGraphics = ctx_->renderPipelinesPool_.get(currentPipelineGraphics_);
  const lvk::ComputePipelineState* stateCompute = ctx_->computePipelinesPool_.get(currentPipelineCompute_);
  const lvk::RayTracingPipelineState* stateRayTracing = ctx_->rayTracingPipelinesPool_.get(currentPipelineRayTracing_);

  LVK_ASSERT(stateGraphics || stateCompute || stateRayTracing);

  VkPipelineLayout layout = stateGraphics ? stateGraphics->pipelineLayout_
                                          : (stateCompute ? stateCompute->pipelineLayout_ : stateRayTracing->pipelineLayout_);
  VkShaderStageFlags shaderStageFlags = stateGraphics ? stateGraphics->shaderStageFlags_
                                                      : (stateCompute ? VK_SHADER_STAGE_COMPUTE_BIT : stateRayTracing->shaderStageFlags_);

  vkCmdPushConstants(wrapper_->cmdBuf_, layout, shaderStageFlags, (uint32_t)offset, (uint32_t)size, data);
}

void lvk::CommandBuffer::cmdFillBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, uint32_t data) {
  LVK_PROFILER_FUNCTION();
  LVK_ASSERT(buffer.valid());
  LVK_ASSERT(size);
  LVK_ASSERT(size % 4 == 0);
  LVK_ASSERT(bufferOffset % 4 == 0);

  lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(buffer);

  bufferBarrier(buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  vkCmdFillBuffer(wrapper_->cmdBuf_, buf->vkBuffer_, bufferOffset, size, data);

  VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;

  if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
  }
  if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
  }

  bufferBarrier(buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, dstStage);
}

void lvk::CommandBuffer::cmdCopyBuffer(BufferHandle srcBuffer, BufferHandle dstBuffer, size_t srcOffset, size_t dstOffset, size_t size) {
  LVK_PROFILER_FUNCTION();
  LVK_ASSERT(srcBuffer.valid());
  LVK_ASSERT(dstBuffer.valid());
  LVK_ASSERT(size);
  // Check buffer regions do not overlap
  if (srcBuffer == dstBuffer) {
    if (srcOffset < dstOffset) {
      LVK_ASSERT(dstOffset - srcOffset >= size);
    } else {
      LVK_ASSERT(srcOffset - dstOffset >= size);
    }
  }

  lvk::VulkanBuffer* srcBuf = ctx_->buffersPool_.get(srcBuffer);
  lvk::VulkanBuffer* dstBuf = ctx_->buffersPool_.get(dstBuffer);

  bufferBarrier(srcBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  bufferBarrier(dstBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  const VkBufferCopy2 copyRegion = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
      .srcOffset = srcOffset,
      .dstOffset = dstOffset,
      .size = size,
  };

  const VkCopyBufferInfo2 copyInfo = {
      .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .srcBuffer = srcBuf->vkBuffer_,
      .dstBuffer = dstBuf->vkBuffer_,
      .regionCount = 1,
      .pRegions = &copyRegion,
  };

  vkCmdCopyBuffer2(wrapper_->cmdBuf_, &copyInfo);

  VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;

  if (srcBuf->vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
    srcStage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
  }
  if (srcBuf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    srcStage |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
  }

  VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;

  if (dstBuf->vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
  }
  if (dstBuf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
  }

  bufferBarrier(srcBuffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, srcStage);
  bufferBarrier(dstBuffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, dstStage);
}

void lvk::CommandBuffer::cmdUpdateBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, const void* data) {
  LVK_PROFILER_FUNCTION();
  LVK_ASSERT(buffer.valid());
  LVK_ASSERT(data);
  LVK_ASSERT(size && size <= 65536);
  LVK_ASSERT(size % 4 == 0);
  LVK_ASSERT(bufferOffset % 4 == 0);

  lvk::VulkanBuffer* buf = ctx_->buffersPool_.get(buffer);

  bufferBarrier(buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

  vkCmdUpdateBuffer(wrapper_->cmdBuf_, buf->vkBuffer_, bufferOffset, size, data);

  VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;

  if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
  }
  if (buf->vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    dstStage |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
  }

  bufferBarrier(buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, dstStage);
}

void lvk::CommandBuffer::cmdDraw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t baseInstance) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDraw()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  if (vertexCount == 0) {
    return;
  }

  vkCmdDraw(wrapper_->cmdBuf_, vertexCount, instanceCount, firstVertex, baseInstance);
}

void lvk::CommandBuffer::cmdDrawIndexed(uint32_t indexCount,
                                        uint32_t instanceCount,
                                        uint32_t firstIndex,
                                        int32_t vertexOffset,
                                        uint32_t baseInstance) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawIndexed()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  if (indexCount == 0) {
    return;
  }

  LVK_ASSERT(ctx_->awaitingCreation_ == false);

  vkCmdDrawIndexed(wrapper_->cmdBuf_, indexCount, instanceCount, firstIndex, vertexOffset, baseInstance);
}

void lvk::CommandBuffer::cmdDrawIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, uint32_t drawCount, uint32_t stride) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawIndirect()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  lvk::VulkanBuffer* bufIndirect = ctx_->buffersPool_.get(indirectBuffer);

  LVK_ASSERT(bufIndirect);

  vkCmdDrawIndirect(
      wrapper_->cmdBuf_, bufIndirect->vkBuffer_, indirectBufferOffset, drawCount, stride ? stride : sizeof(VkDrawIndirectCommand));
}

void lvk::CommandBuffer::cmdDrawIndexedIndirect(BufferHandle indirectBuffer,
                                                size_t indirectBufferOffset,
                                                uint32_t drawCount,
                                                uint32_t stride) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawIndexedIndirect()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  lvk::VulkanBuffer* bufIndirect = ctx_->buffersPool_.get(indirectBuffer);

  LVK_ASSERT(bufIndirect);

  vkCmdDrawIndexedIndirect(
      wrapper_->cmdBuf_, bufIndirect->vkBuffer_, indirectBufferOffset, drawCount, stride ? stride : sizeof(VkDrawIndexedIndirectCommand));
}

void lvk::CommandBuffer::cmdDrawIndexedIndirectCount(BufferHandle indirectBuffer,
                                                     size_t indirectBufferOffset,
                                                     BufferHandle countBuffer,
                                                     size_t countBufferOffset,
                                                     uint32_t maxDrawCount,
                                                     uint32_t stride) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawIndexedIndirectCount()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  lvk::VulkanBuffer* bufIndirect = ctx_->buffersPool_.get(indirectBuffer);
  lvk::VulkanBuffer* bufCount = ctx_->buffersPool_.get(countBuffer);

  LVK_ASSERT(bufIndirect);
  LVK_ASSERT(bufCount);

  vkCmdDrawIndexedIndirectCount(wrapper_->cmdBuf_,
                                bufIndirect->vkBuffer_,
                                indirectBufferOffset,
                                bufCount->vkBuffer_,
                                countBufferOffset,
                                maxDrawCount,
                                stride ? stride : sizeof(VkDrawIndexedIndirectCommand));
}

void lvk::CommandBuffer::cmdDrawMeshTasks(const Dimensions& threadgroupCount) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawMeshTasks()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  vkCmdDrawMeshTasksEXT(wrapper_->cmdBuf_, threadgroupCount.width, threadgroupCount.height, threadgroupCount.depth);
}

void lvk::CommandBuffer::cmdDrawMeshTasksIndirect(BufferHandle indirectBuffer,
                                                  size_t indirectBufferOffset,
                                                  uint32_t drawCount,
                                                  uint32_t stride) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawMeshTasksIndirect()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  lvk::VulkanBuffer* bufIndirect = ctx_->buffersPool_.get(indirectBuffer);

  LVK_ASSERT(bufIndirect);

  vkCmdDrawMeshTasksIndirectEXT(wrapper_->cmdBuf_,
                                bufIndirect->vkBuffer_,
                                indirectBufferOffset,
                                drawCount,
                                stride ? stride : sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

void lvk::CommandBuffer::cmdDrawMeshTasksIndirectCount(BufferHandle indirectBuffer,
                                                       size_t indirectBufferOffset,
                                                       BufferHandle countBuffer,
                                                       size_t countBufferOffset,
                                                       uint32_t maxDrawCount,
                                                       uint32_t stride) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdDrawMeshTasksIndirectCount()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_DRAW);

  lvk::VulkanBuffer* bufIndirect = ctx_->buffersPool_.get(indirectBuffer);
  lvk::VulkanBuffer* bufCount = ctx_->buffersPool_.get(countBuffer);

  LVK_ASSERT(bufIndirect);
  LVK_ASSERT(bufCount);

  vkCmdDrawMeshTasksIndirectCountEXT(wrapper_->cmdBuf_,
                                     bufIndirect->vkBuffer_,
                                     indirectBufferOffset,
                                     bufCount->vkBuffer_,
                                     countBufferOffset,
                                     maxDrawCount,
                                     stride ? stride : sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

void lvk::CommandBuffer::cmdTraceRays(uint32_t width, uint32_t height, uint32_t depth, const Dependencies& deps) {
  LVK_PROFILER_FUNCTION();
  LVK_PROFILER_GPU_ZONE("cmdTraceRays()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_RTX);

  lvk::RayTracingPipelineState* rtps = ctx_->rayTracingPipelinesPool_.get(currentPipelineRayTracing_);

  if (!LVK_VERIFY(rtps)) {
    return;
  }

  LVK_ASSERT(!isRendering_);

  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.textures[i]; i++) {
    useComputeTexture(deps.textures[i], VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
  }
  for (uint32_t i = 0; i != Dependencies::LVK_MAX_SUBMIT_DEPENDENCIES && deps.buffers[i]; i++) {
    bufferBarrier(deps.buffers[i],
                  VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
  }

  vkCmdTraceRaysKHR(
      wrapper_->cmdBuf_, &rtps->sbtEntryRayGen, &rtps->sbtEntryMiss, &rtps->sbtEntryHit, &rtps->sbtEntryCallable, width, height, depth);
}

void lvk::CommandBuffer::cmdSetBlendColor(const float color[4]) {
  vkCmdSetBlendConstants(wrapper_->cmdBuf_, color);
}

void lvk::CommandBuffer::cmdSetDepthBias(float constantFactor, float slopeFactor, float clamp) {
  vkCmdSetDepthBias(wrapper_->cmdBuf_, constantFactor, clamp, slopeFactor);
}

void lvk::CommandBuffer::cmdSetDepthBiasEnable(bool enable) {
  vkCmdSetDepthBiasEnable(wrapper_->cmdBuf_, enable ? VK_TRUE : VK_FALSE);
}

void lvk::CommandBuffer::cmdResetQueryPool(QueryPoolHandle pool, uint32_t firstQuery, uint32_t queryCount) {
  VkQueryPool vkPool = *ctx_->queriesPool_.get(pool);

  vkCmdResetQueryPool(wrapper_->cmdBuf_, vkPool, firstQuery, queryCount);
}

void lvk::CommandBuffer::cmdWriteTimestamp(QueryPoolHandle pool, uint32_t query) {
  VkQueryPool vkPool = *ctx_->queriesPool_.get(pool);

  vkCmdWriteTimestamp(wrapper_->cmdBuf_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, vkPool, query);
}

void lvk::CommandBuffer::cmdClearColorImage(TextureHandle tex, const ClearColorValue& value, const TextureLayers& layers) {
  LVK_PROFILER_GPU_ZONE("cmdClearColorImage()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_COPY);

  static_assert(sizeof(ClearColorValue) == sizeof(VkClearColorValue));

  lvk::VulkanImage* img = ctx_->texturesPool_.get(tex);

  if (!LVK_VERIFY(img)) {
    return;
  }

  const VkImageSubresourceRange range = {
      .aspectMask = img->getImageAspectFlags(),
      .baseMipLevel = layers.mipLevel,
      .levelCount = VK_REMAINING_MIP_LEVELS,
      .baseArrayLayer = layers.layer,
      .layerCount = layers.numLayers,
  };

  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      img->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
      img->vkImageLayout_,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      range);

  vkCmdClearColorImage(wrapper_->cmdBuf_,
                       img->vkImage_,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       reinterpret_cast<const VkClearColorValue*>(&value),
                       1,
                       &range);

  // a ternary cascade...
  const VkImageLayout newLayout = img->vkImageLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? (img->isAttachment()     ? VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
                                         : img->isSampledImage() ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                                 : VK_IMAGE_LAYOUT_GENERAL)
                                      : img->vkImageLayout_;

  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      img->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      newLayout,
      range);

  img->vkImageLayout_ = newLayout;
}

void lvk::CommandBuffer::cmdCopyImage(TextureHandle src,
                                      TextureHandle dst,
                                      const Dimensions& extent,
                                      const Offset3D& srcOffset,
                                      const Offset3D& dstOffset,
                                      const TextureLayers& srcLayers,
                                      const TextureLayers& dstLayers) {
  LVK_PROFILER_GPU_ZONE("cmdCopyImage()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_COPY);

  lvk::VulkanImage* imgSrc = ctx_->texturesPool_.get(src);
  lvk::VulkanImage* imgDst = ctx_->texturesPool_.get(dst);

  LVK_ASSERT(imgSrc && imgDst);
  LVK_ASSERT(srcLayers.numLayers == dstLayers.numLayers);

  if (!imgSrc || !imgDst) {
    return;
  }

  const VkImageSubresourceRange rangeSrc = {
      .aspectMask = imgSrc->getImageAspectFlags(),
      .baseMipLevel = srcLayers.mipLevel,
      .levelCount = 1,
      .baseArrayLayer = srcLayers.layer,
      .layerCount = srcLayers.numLayers,
  };
  const VkImageSubresourceRange rangeDst = {
      .aspectMask = imgDst->getImageAspectFlags(),
      .baseMipLevel = dstLayers.mipLevel,
      .levelCount = 1,
      .baseArrayLayer = dstLayers.layer,
      .layerCount = dstLayers.numLayers,
  };

  LVK_ASSERT(imgSrc->vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);

  const VkExtent3D dstExtent = imgDst->vkExtent_;
  const bool coversFullDstImage = dstExtent.width == extent.width && dstExtent.height == extent.height && dstExtent.depth == extent.depth &&
                                  dstOffset.x == 0 && dstOffset.y == 0 && dstOffset.z == 0;

  LVK_ASSERT(coversFullDstImage || imgDst->vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);

  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      imgSrc->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
      imgSrc->vkImageLayout_,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      rangeSrc);
  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      imgDst->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
      coversFullDstImage ? VK_IMAGE_LAYOUT_UNDEFINED : imgDst->vkImageLayout_,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      rangeDst);

  const VkImageCopy regionCopy = {
      .srcSubresource =
          {
              .aspectMask = imgSrc->getImageAspectFlags(),
              .mipLevel = srcLayers.mipLevel,
              .baseArrayLayer = srcLayers.layer,
              .layerCount = srcLayers.numLayers,
          },
      .srcOffset = {.x = srcOffset.x, .y = srcOffset.y, .z = srcOffset.z},
      .dstSubresource =
          {
              .aspectMask = imgDst->getImageAspectFlags(),
              .mipLevel = dstLayers.mipLevel,
              .baseArrayLayer = dstLayers.layer,
              .layerCount = dstLayers.numLayers,
          },
      .dstOffset = {.x = dstOffset.x, .y = dstOffset.y, .z = dstOffset.z},
      .extent = {.width = extent.width, .height = extent.height, .depth = extent.depth},
  };
  const VkImageBlit regionBlit = {
      .srcSubresource = regionCopy.srcSubresource,
      .srcOffsets = {{},
                     {.x = int32_t(srcOffset.x + imgSrc->vkExtent_.width),
                      .y = int32_t(srcOffset.y + imgSrc->vkExtent_.height),
                      .z = int32_t(srcOffset.z + imgSrc->vkExtent_.depth)}},
      .dstSubresource = regionCopy.dstSubresource,
      .dstOffsets = {{},
                     {.x = int32_t(dstOffset.x + imgDst->vkExtent_.width),
                      .y = int32_t(dstOffset.y + imgDst->vkExtent_.height),
                      .z = int32_t(dstOffset.z + imgDst->vkExtent_.depth)}},
  };

  const bool isCompatible = getBytesPerPixel(imgSrc->vkImageFormat_) == getBytesPerPixel(imgDst->vkImageFormat_);

  isCompatible ? vkCmdCopyImage(wrapper_->cmdBuf_,
                                imgSrc->vkImage_,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                imgDst->vkImage_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1,
                                &regionCopy)
               : vkCmdBlitImage(wrapper_->cmdBuf_,
                                imgSrc->vkImage_,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                imgDst->vkImage_,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                1,
                                &regionBlit,
                                VK_FILTER_LINEAR);

  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      imgSrc->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      imgSrc->vkImageLayout_,
      rangeSrc);

  // a ternary cascade...
  const VkImageLayout newLayout = imgDst->vkImageLayout_ == VK_IMAGE_LAYOUT_UNDEFINED
                                      ? (imgDst->isAttachment()     ? VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL
                                         : imgDst->isSampledImage() ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                                    : VK_IMAGE_LAYOUT_GENERAL)
                                      : imgDst->vkImageLayout_;

  lvk::imageMemoryBarrier2(
      wrapper_->cmdBuf_,
      imgDst->vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      newLayout,
      rangeDst);

  imgDst->vkImageLayout_ = newLayout;
}

void lvk::CommandBuffer::cmdGenerateMipmap(TextureHandle handle) {
  if (handle.empty()) {
    return;
  }

  const lvk::VulkanImage* tex = ctx_->texturesPool_.get(handle);

  if (tex->numLevels_ <= 1) {
    return;
  }

  LVK_ASSERT(tex->vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);

  tex->generateMipmap(wrapper_->cmdBuf_);
}

void lvk::CommandBuffer::cmdUpdateTLAS(AccelStructHandle handle, BufferHandle instancesBuffer) {
  LVK_PROFILER_GPU_ZONE("cmdUpdateTLAS()", ctx_, wrapper_->cmdBuf_, LVK_PROFILER_COLOR_CMD_RTX);

  if (handle.empty()) {
    return;
  }

  lvk::AccelerationStructure* as = ctx_->accelStructuresPool_.get(handle);

  const VkAccelerationStructureGeometryKHR accelerationStructureGeometry{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry =
          {
              .instances =
                  {
                      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                      .arrayOfPointers = VK_FALSE,
                      .data = {.deviceAddress = ctx_->gpuAddress(instancesBuffer)},
                  },
          },
      .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
  };

  VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      .geometryCount = 1,
      .pGeometries = &accelerationStructureGeometry,
  };
  VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
  };
  vkGetAccelerationStructureBuildSizesKHR(ctx_->getVkDevice(),
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &accelerationStructureBuildGeometryInfo,
                                          &as->buildRangeInfo.primitiveCount,
                                          &accelerationStructureBuildSizesInfo);

  const uint32_t alignment = ctx_->accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment;
  accelerationStructureBuildSizesInfo.accelerationStructureSize += alignment;
  accelerationStructureBuildSizesInfo.updateScratchSize += alignment;
  accelerationStructureBuildSizesInfo.buildScratchSize += alignment;

  if (!as->scratchBuffer.valid() || getBufferSize(ctx_, as->scratchBuffer) < accelerationStructureBuildSizesInfo.updateScratchSize) {
    LLOGD("Recreating scratch buffer for TLAS update");
    as->scratchBuffer = ctx_->createBuffer(
        lvk::BufferDesc{
            .usage = lvk::BufferUsageBits_Storage,
            .storage = lvk::StorageType_Device,
            .size = accelerationStructureBuildSizesInfo.updateScratchSize,
            .debugName = "scratchBuffer",
        },
        nullptr,
        nullptr);
  }

  const VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
      .srcAccelerationStructure = as->vkHandle,
      .dstAccelerationStructure = as->vkHandle,
      .geometryCount = 1,
      .pGeometries = &accelerationStructureGeometry,
      .scratchData = {.deviceAddress =
                          getAlignedAddress(ctx_->gpuAddress(as->scratchBuffer),
                                            ctx_->accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment)},
  };

  const VkAccelerationStructureBuildRangeInfoKHR* accelerationBuildStructureRangeInfos[] = {&as->buildRangeInfo};

  {
    const VkBufferMemoryBarrier2 barriers[] = {
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .buffer = getVkBuffer(ctx_, handle),
            .size = VK_WHOLE_SIZE,
        },
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
            .buffer = getVkBuffer(ctx_, instancesBuffer),
            .size = VK_WHOLE_SIZE,
        },
    };
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = LVK_ARRAY_NUM_ELEMENTS(barriers),
        .pBufferMemoryBarriers = barriers,
    };
    vkCmdPipelineBarrier2(wrapper_->cmdBuf_, &dependencyInfo);
  }
  vkCmdBuildAccelerationStructuresKHR(wrapper_->cmdBuf_, 1, &accelerationBuildGeometryInfo, accelerationBuildStructureRangeInfos);
  {
    const VkBufferMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .buffer = getVkBuffer(ctx_, handle),
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    const VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(wrapper_->cmdBuf_, &dependencyInfo);
  }
}

lvk::VulkanStagingDevice::VulkanStagingDevice(VulkanContext& ctx) : ctx_(ctx) {
  LVK_PROFILER_FUNCTION();

  const VkDeviceSize maxMemoryAllocationSize = ctx_.vkPhysicalDeviceVulkan11Properties_.maxMemoryAllocationSize;

  LVK_ASSERT(ctx_.config_.maxStagingBufferSize <= maxMemoryAllocationSize);

  // clamped to the max limits
  maxBufferSize_ = std::min(maxMemoryAllocationSize, ctx_.config_.maxStagingBufferSize);
  minBufferSize_ = std::min(minBufferSize_, maxBufferSize_);
}

void lvk::VulkanStagingDevice::bufferSubData(VulkanBuffer& buffer, size_t dstOffset, size_t size, const void* data) {
  LVK_PROFILER_FUNCTION();

  if (buffer.isMapped()) {
    buffer.bufferSubData(ctx_, dstOffset, size, data);
    return;
  }

  lvk::VulkanBuffer* stagingBuffer = ctx_.buffersPool_.get(stagingBuffer_);

  LVK_ASSERT(stagingBuffer);

  while (size) {
    // get next staging buffer free offset
    MemoryRegionDesc desc = getNextFreeOffset((uint32_t)size);
    const uint32_t chunkSize = std::min((uint64_t)size, desc.size_);

    // copy data into staging buffer
    stagingBuffer->bufferSubData(ctx_, desc.offset_, chunkSize, data);

    // do the transfer
    const VkBufferCopy copy = {
        .srcOffset = desc.offset_,
        .dstOffset = dstOffset,
        .size = chunkSize,
    };

    const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper = ctx_.immediate_->acquire();
    vkCmdCopyBuffer(wrapper.cmdBuf_, stagingBuffer->vkBuffer_, buffer.vkBuffer_, 1, &copy);
    VkBufferMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = 0,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer.vkBuffer_,
        .offset = dstOffset,
        .size = chunkSize,
    };
    VkPipelineStageFlags dstMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (buffer.vkUsageFlags_ & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) {
      dstMask |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
      barrier.dstAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }
    if (buffer.vkUsageFlags_ & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) {
      dstMask |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      barrier.dstAccessMask |= VK_ACCESS_INDEX_READ_BIT;
    }
    if (buffer.vkUsageFlags_ & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
      dstMask |= VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
      barrier.dstAccessMask |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    }
    if (buffer.vkUsageFlags_ & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR) {
      dstMask |= VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
      barrier.dstAccessMask |= VK_ACCESS_MEMORY_READ_BIT;
    }
    vkCmdPipelineBarrier(
        wrapper.cmdBuf_, VK_PIPELINE_STAGE_TRANSFER_BIT, dstMask, VkDependencyFlags{}, 0, nullptr, 1, &barrier, 0, nullptr);
    desc.handle_ = ctx_.immediate_->submit(wrapper);
    regions_.push_back(desc);

    size -= chunkSize;
    data = (uint8_t*)data + chunkSize;
    dstOffset += chunkSize;
  }
}

void lvk::VulkanStagingDevice::imageData2D(VulkanImage& image,
                                           const VkRect2D& imageRegion,
                                           uint32_t baseMipLevel,
                                           uint32_t numMipLevels,
                                           uint32_t layer,
                                           uint32_t numLayers,
                                           VkFormat format,
                                           const void* data,
                                           uint32_t bufferRowLength) {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT(numMipLevels <= LVK_MAX_MIP_LEVELS);

  const Format texFormat = vkFormatToFormat(format);

  // divide the width and height by 2 until we get to the size of level 'baseMipLevel'
  const uint32_t width = image.vkExtent_.width >> baseMipLevel;
  const uint32_t height = image.vkExtent_.height >> baseMipLevel;
  const bool coversFullImage = !imageRegion.offset.x && !imageRegion.offset.y && imageRegion.extent.width == width &&
                               imageRegion.extent.height == height;

  LVK_ASSERT(coversFullImage || image.vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);

  if (numMipLevels > 1 || numLayers > 1) {
    LVK_ASSERT(!bufferRowLength);
    LVK_ASSERT_MSG(coversFullImage, "Uploading mip-levels with an image region that is smaller than the base mip-level is not supported");
  }

  // find the storage size for all mip-levels being uploaded
  uint32_t layerStorageSize = 0;
  for (uint32_t i = 0; i < numMipLevels; ++i) {
    const uint32_t mipSize =
        lvk::getTextureBytesPerLayer(bufferRowLength ? bufferRowLength : imageRegion.extent.width, imageRegion.extent.height, texFormat, i);
    layerStorageSize += mipSize;
  }

  const uint32_t storageSize = layerStorageSize * numLayers;

  ensureStagingBufferSize(storageSize);

  LVK_ASSERT(storageSize <= stagingBufferSize_);

  MemoryRegionDesc desc = getNextFreeOffset(storageSize);
  // No support for copying image in multiple smaller chunk sizes. If we get smaller buffer size than storageSize, we will wait for GPU idle
  // and get bigger chunk.
  if (desc.size_ < storageSize) {
    waitAndReset();
    desc = getNextFreeOffset(storageSize);
  }
  LVK_ASSERT(desc.size_ >= storageSize);

  const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper = ctx_.immediate_->acquire();

  lvk::VulkanBuffer* stagingBuffer = ctx_.buffersPool_.get(stagingBuffer_);

  stagingBuffer->bufferSubData(ctx_, desc.offset_, storageSize, data);

  uint32_t offset = 0;

  const uint32_t numPlanes = lvk::getNumImagePlanes(image.vkImageFormat_);

  if (numPlanes > 1) {
    LVK_ASSERT(layer == 0 && baseMipLevel == 0);
    LVK_ASSERT(numLayers == 1 && numMipLevels == 1);
    LVK_ASSERT(imageRegion.offset.x == 0 && imageRegion.offset.y == 0);
    LVK_ASSERT(image.vkType_ == VK_IMAGE_TYPE_2D);
    LVK_ASSERT(image.vkExtent_.width == imageRegion.extent.width && image.vkExtent_.height == imageRegion.extent.height);
  }

  VkImageAspectFlags imageAspect = VK_IMAGE_ASPECT_COLOR_BIT;

  if (numPlanes == 2) {
    imageAspect = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT;
  }
  if (numPlanes == 3) {
    imageAspect = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT;
  }

  // https://registry.khronos.org/KTX/specs/1.0/ktxspec.v1.html
  for (uint32_t mipLevel = 0; mipLevel < numMipLevels; ++mipLevel) {
    for (uint32_t layer = 0; layer != numLayers; layer++) {
      const uint32_t currentMipLevel = baseMipLevel + mipLevel;

      LVK_ASSERT(currentMipLevel < image.numLevels_);
      LVK_ASSERT(mipLevel < image.numLevels_);

      // 1. Transition initial image layout into TRANSFER_DST_OPTIMAL
      lvk::imageMemoryBarrier2(wrapper.cmdBuf_,
                               image.vkImage_,
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .access = VK_ACCESS_2_NONE},
                               StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
                               coversFullImage ? VK_IMAGE_LAYOUT_UNDEFINED : image.vkImageLayout_,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VkImageSubresourceRange{imageAspect, currentMipLevel, 1, layer, 1});

#if LVK_VULKAN_PRINT_COMMANDS
      LLOGL("%p vkCmdCopyBufferToImage()\n", wrapper.cmdBuf_);
#endif // LVK_VULKAN_PRINT_COMMANDS
      // 2. Copy the pixel data from the staging buffer into the image
      uint32_t planeOffset = 0;
      for (uint32_t plane = 0; plane != numPlanes; plane++) {
        const VkExtent2D extent = lvk::getImagePlaneExtent(
            {
                .width = std::max(1u, imageRegion.extent.width >> mipLevel),
                .height = std::max(1u, imageRegion.extent.height >> mipLevel),
            },
            vkFormatToFormat(format),
            plane);
        const VkRect2D region = {
            .offset = {.x = imageRegion.offset.x >> mipLevel, .y = imageRegion.offset.y >> mipLevel},
            .extent = extent,
        };
        const VkBufferImageCopy copy = {
            // the offset for this level is at the start of all mip-levels plus the size of all previous mip-levels being uploaded
            .bufferOffset = desc.offset_ + offset + planeOffset,
            .bufferRowLength = bufferRowLength,
            .bufferImageHeight = 0,
            .imageSubresource =
                VkImageSubresourceLayers{numPlanes > 1 ? VK_IMAGE_ASPECT_PLANE_0_BIT << plane : imageAspect, currentMipLevel, layer, 1},
            .imageOffset = {.x = region.offset.x, .y = region.offset.y, .z = 0},
            .imageExtent = {.width = region.extent.width, .height = region.extent.height, .depth = 1u},
        };
        vkCmdCopyBufferToImage(wrapper.cmdBuf_, stagingBuffer->vkBuffer_, image.vkImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        planeOffset += lvk::getTextureBytesPerPlane(imageRegion.extent.width, imageRegion.extent.height, vkFormatToFormat(format), plane);
      }

      // 3. Transition TRANSFER_DST_OPTIMAL into SHADER_READ_ONLY_OPTIMAL
      lvk::imageMemoryBarrier2(
          wrapper.cmdBuf_,
          image.vkImage_,
          StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
          StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VkImageSubresourceRange{imageAspect, currentMipLevel, 1, layer, 1});

      offset += lvk::getTextureBytesPerLayer(imageRegion.extent.width, imageRegion.extent.height, texFormat, currentMipLevel);
    }
  }

  image.vkImageLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  desc.handle_ = ctx_.immediate_->submit(wrapper);
  regions_.push_back(desc);
}

void lvk::VulkanStagingDevice::imageData3D(VulkanImage& image,
                                           const VkOffset3D& offset,
                                           const VkExtent3D& extent,
                                           VkFormat format,
                                           const void* data) {
  LVK_PROFILER_FUNCTION();
  LVK_ASSERT_MSG(image.numLevels_ == 1, "Can handle only 3D images with exactly 1 mip-level");
  LVK_ASSERT_MSG((offset.x == 0) && (offset.y == 0) && (offset.z == 0), "Can upload only full-size 3D images");
  const uint32_t storageSize = extent.width * extent.height * extent.depth * getBytesPerPixel(format);

  ensureStagingBufferSize(storageSize);

  LVK_ASSERT_MSG(storageSize <= stagingBufferSize_, "No support for copying image in multiple smaller chunk sizes");

  // get next staging buffer free offset
  MemoryRegionDesc desc = getNextFreeOffset(storageSize);

  // No support for copying image in multiple smaller chunk sizes.
  // If we get smaller buffer size than storageSize, we will wait for GPU idle and get a bigger chunk.
  if (desc.size_ < storageSize) {
    waitAndReset();
    desc = getNextFreeOffset(storageSize);
  }

  LVK_ASSERT(desc.size_ >= storageSize);

  lvk::VulkanBuffer* stagingBuffer = ctx_.buffersPool_.get(stagingBuffer_);

  // 1. Copy the pixel data into the host visible staging buffer
  stagingBuffer->bufferSubData(ctx_, desc.offset_, storageSize, data);

  const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper = ctx_.immediate_->acquire();

  // 1. Transition initial image layout into TRANSFER_DST_OPTIMAL
  lvk::imageMemoryBarrier2(wrapper.cmdBuf_,
                           image.vkImage_,
                           StageAccess{.stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .access = VK_ACCESS_2_NONE},
                           StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
                           VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

  // 2. Copy the pixel data from the staging buffer into the image
  const VkBufferImageCopy copy = {
      .bufferOffset = desc.offset_,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = offset,
      .imageExtent = extent,
  };
  vkCmdCopyBufferToImage(wrapper.cmdBuf_, stagingBuffer->vkBuffer_, image.vkImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

  // 3. Transition TRANSFER_DST_OPTIMAL into SHADER_READ_ONLY_OPTIMAL
  lvk::imageMemoryBarrier2(
      wrapper.cmdBuf_,
      image.vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

  image.vkImageLayout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  desc.handle_ = ctx_.immediate_->submit(wrapper);
  regions_.push_back(desc);
}

void lvk::VulkanStagingDevice::getImageData(VulkanImage& image,
                                            const VkOffset3D& offset,
                                            const VkExtent3D& extent,
                                            VkImageSubresourceRange range,
                                            VkFormat format,
                                            void* outData) {
  LVK_ASSERT(image.vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);
  LVK_ASSERT(range.layerCount == 1);

  const uint32_t storageSize = extent.width * extent.height * extent.depth * getBytesPerPixel(format);

  ensureStagingBufferSize(storageSize);

  LVK_ASSERT(storageSize <= stagingBufferSize_);

  // get next staging buffer free offset
  MemoryRegionDesc desc = getNextFreeOffset(storageSize);

  // No support for copying image in multiple smaller chunk sizes.
  // If we get smaller buffer size than storageSize, we will wait for GPU idle and get a bigger chunk.
  if (desc.size_ < storageSize) {
    waitAndReset();
    desc = getNextFreeOffset(storageSize);
  }

  LVK_ASSERT(desc.size_ >= storageSize);

  lvk::VulkanBuffer* stagingBuffer = ctx_.buffersPool_.get(stagingBuffer_);

  const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper1 = ctx_.immediate_->acquire();

  // 1. Transition to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
  lvk::imageMemoryBarrier2(
      wrapper1.cmdBuf_,
      image.vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
      image.vkImageLayout_,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      range);

  // 2.  Copy the pixel data from the image into the staging buffer
  const VkBufferImageCopy copy = {
      .bufferOffset = desc.offset_,
      .bufferRowLength = 0,
      .bufferImageHeight = extent.height,
      .imageSubresource =
          VkImageSubresourceLayers{
              .aspectMask = range.aspectMask,
              .mipLevel = range.baseMipLevel,
              .baseArrayLayer = range.baseArrayLayer,
              .layerCount = range.layerCount,
          },
      .imageOffset = offset,
      .imageExtent = extent,
  };
  vkCmdCopyImageToBuffer(wrapper1.cmdBuf_, image.vkImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer->vkBuffer_, 1, &copy);

  desc.handle_ = ctx_.immediate_->submit(wrapper1);
  regions_.push_back(desc);

  waitAndReset();

  if (!stagingBuffer->isCoherentMemory_) {
    stagingBuffer->invalidateMappedMemory(ctx_, desc.offset_, desc.size_);
  }

  // 3. Copy data from staging buffer into data
  memcpy(outData, stagingBuffer->getMappedPtr() + desc.offset_, storageSize);

  // 4. Transition back to the initial image layout
  const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper2 = ctx_.immediate_->acquire();

  lvk::imageMemoryBarrier2(
      wrapper2.cmdBuf_,
      image.vkImage_,
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .access = VK_ACCESS_2_TRANSFER_READ_BIT},
      StageAccess{.stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT},
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      image.vkImageLayout_,
      range);

  ctx_.immediate_->wait(ctx_.immediate_->submit(wrapper2));
}

void lvk::VulkanStagingDevice::ensureStagingBufferSize(uint32_t sizeNeeded) {
  LVK_PROFILER_FUNCTION();

  const uint32_t alignedSize = std::max(getAlignedSize(sizeNeeded, kStagingBufferAlignment), minBufferSize_);

  sizeNeeded = alignedSize < ctx_.config_.maxStagingBufferSize ? alignedSize : ctx_.config_.maxStagingBufferSize;

  if (!stagingBuffer_.empty()) {
    const bool isEnoughSize = sizeNeeded <= stagingBufferSize_;
    const bool isMaxSize = stagingBufferSize_ == ctx_.config_.maxStagingBufferSize;

    if (isEnoughSize || isMaxSize) {
      return;
    }
  }

  waitAndReset();

  // deallocate the previous staging buffer
  stagingBuffer_ = nullptr;

  // if the combined size of the new staging buffer and the existing one is larger than the limit imposed by some architectures on buffers
  // that are device and host visible, we need to wait for the current buffer to be destroyed before we can allocate a new one
  if ((sizeNeeded + stagingBufferSize_) > ctx_.config_.maxStagingBufferSize) {
    ctx_.waitDeferredTasks();
  }

  stagingBufferSize_ = sizeNeeded;

  char debugName[256] = {0};
  snprintf(debugName, sizeof(debugName) - 1, "Buffer: staging buffer %u", stagingBufferCounter_++);

  stagingBuffer_ = {&ctx_,
                    ctx_.createBuffer(stagingBufferSize_,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                      nullptr,
                                      debugName)};
  LVK_ASSERT(!stagingBuffer_.empty());

  regions_.clear();
  regions_.push_back({0, stagingBufferSize_, SubmitHandle()});
}

lvk::VulkanStagingDevice::MemoryRegionDesc lvk::VulkanStagingDevice::getNextFreeOffset(uint32_t size) {
  LVK_PROFILER_FUNCTION();

  const uint32_t requestedAlignedSize = getAlignedSize(size, kStagingBufferAlignment);

  ensureStagingBufferSize(requestedAlignedSize);

  LVK_ASSERT(!regions_.empty());

  // if we can't find an available region that is big enough to store requestedAlignedSize, return whatever we could find, which will be
  // stored in bestNextIt
  auto bestNextIt = regions_.begin();

  for (auto it = regions_.begin(); it != regions_.end(); ++it) {
    if (ctx_.immediate_->isReady(it->handle_)) {
      // This region is free, but is it big enough?
      if (it->size_ >= requestedAlignedSize) {
        // It is big enough!
        const uint32_t unusedSize = it->size_ - requestedAlignedSize;
        const uint32_t unusedOffset = it->offset_ + requestedAlignedSize;

        // Return this region and add the remaining unused size to the regions_ deque
        SCOPE_EXIT {
          regions_.erase(it);
          if (unusedSize > 0) {
            regions_.insert(regions_.begin(), {unusedOffset, unusedSize, SubmitHandle()});
          }
        };

        return {it->offset_, requestedAlignedSize, SubmitHandle()};
      }
      // cache the largest available region that isn't as big as the one we're looking for
      if (it->size_ > bestNextIt->size_) {
        bestNextIt = it;
      }
    }
  }

  // we found a region that is available that is smaller than the requested size. It's the best we can do
  if (bestNextIt != regions_.end() && ctx_.immediate_->isReady(bestNextIt->handle_)) {
    SCOPE_EXIT {
      regions_.erase(bestNextIt);
    };

    return {bestNextIt->offset_, bestNextIt->size_, SubmitHandle()};
  }

  // nothing was available. Let's wait for the entire staging buffer to become free
  waitAndReset();

  // waitAndReset() adds a region that spans the entire buffer. Since we'll be using part of it, we need to replace it with a used block and
  // an unused portion
  regions_.clear();

  // store the unused size in the deque first...
  const uint64_t unusedSize = stagingBufferSize_ > requestedAlignedSize ? stagingBufferSize_ - requestedAlignedSize : 0;

  if (unusedSize) {
    const uint64_t unusedOffset = stagingBufferSize_ - unusedSize;
    regions_.insert(regions_.begin(), {unusedOffset, unusedSize, SubmitHandle()});
  }

  // ...and then return the smallest free region that can hold the requested size
  return {
      .offset_ = 0,
      .size_ = stagingBufferSize_ - unusedSize,
      .handle_ = SubmitHandle(),
  };
}

void lvk::VulkanStagingDevice::waitAndReset() {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_WAIT);

  for (const MemoryRegionDesc& r : regions_) {
    ctx_.immediate_->wait(r.handle_);
  };

  regions_.clear();
  regions_.push_back({0, stagingBufferSize_, SubmitHandle()});
}

lvk::VulkanContext::VulkanContext(const lvk::ContextConfig& config, void* window, void* display, VkSurfaceKHR surface)
: config_(config)
, vkSurface_(surface) {
  LVK_PROFILER_THREAD("MainThread");

  pimpl_ = std::make_unique<VulkanContextImpl>();

  if (volkInitialize() != VK_SUCCESS) {
    LLOGW("volkInitialize() failed\n");
    exit(255);
  };

  glslang_initialize_process();

  createInstance();

  if (!surface) {
    if (config_.enableHeadlessSurface) {
      createHeadlessSurface();
    } else if (window || display) {
      createSurface(window, display);
    }
  }
}

lvk::VulkanContext::~VulkanContext() {
  LVK_PROFILER_FUNCTION();

  VK_ASSERT(vkDeviceWaitIdle(vkDevice_));

#if defined(LVK_WITH_TRACY_GPU)
  TracyVkDestroy(pimpl_->tracyVkCtx_);
  if (pimpl_->tracyCommandPool_) {
    vkDestroyCommandPool(vkDevice_, pimpl_->tracyCommandPool_, nullptr);
  }
#endif // LVK_WITH_TRACY_GPU

  stagingDevice_.reset(nullptr);
  swapchain_.reset(nullptr); // swapchain has to be destroyed prior to Surface

  vkDestroySemaphore(vkDevice_, timelineSemaphore_, nullptr);

  destroy(dummyTexture_);

  for (VulkanContextImpl::YcbcrConversionData& data : pimpl_->ycbcrConversionData_) {
    if (data.info.conversion != VK_NULL_HANDLE) {
      vkDestroySamplerYcbcrConversion(vkDevice_, data.info.conversion, nullptr);
      data.sampler.reset();
    }
  }

  if (shaderModulesPool_.numObjects()) {
    LLOGW("Leaked %u shader modules\n", shaderModulesPool_.numObjects());
  }
  if (renderPipelinesPool_.numObjects()) {
    LLOGW("Leaked %u render pipelines\n", renderPipelinesPool_.numObjects());
  }
  if (computePipelinesPool_.numObjects()) {
    LLOGW("Leaked %u compute pipelines\n", computePipelinesPool_.numObjects());
  }
  if (samplersPool_.numObjects() > 1) {
    // the dummy value is owned by the context
    LLOGW("Leaked %u samplers\n", samplersPool_.numObjects() - 1);
  }
  if (texturesPool_.numObjects()) {
    LLOGW("Leaked %u textures\n", texturesPool_.numObjects());
  }
  if (buffersPool_.numObjects()) {
    LLOGW("Leaked %u buffers\n", buffersPool_.numObjects());
  }

  // manually destroy the dummy sampler
  vkDestroySampler(vkDevice_, samplersPool_.objects_.front().obj_, nullptr);
  samplersPool_.clear();
  computePipelinesPool_.clear();
  renderPipelinesPool_.clear();
  shaderModulesPool_.clear();
  texturesPool_.clear();

  waitDeferredTasks();

  immediate_.reset(nullptr);

  for (const DescriptorSet& dset : DSets_) {
    vkDestroyDescriptorPool(vkDevice_, dset.vkDPool, nullptr);
    vkDestroyDescriptorSetLayout(vkDevice_, dset.vkDSL, nullptr);
  }
  vkDestroyDescriptorSetLayout(vkDevice_, dslInputAttachments_, nullptr);
  vkDestroySurfaceKHR(vkInstance_, vkSurface_, nullptr);
  vkDestroyPipelineCache(vkDevice_, pipelineCache_, nullptr);

  // Clean up VMA
  if (LVK_VULKAN_USE_VMA) {
    vmaDestroyAllocator(pimpl_->vma_);
  }

  // Device has to be destroyed prior to Instance
  vkDestroyDevice(vkDevice_, nullptr);

  if (vkDebugUtilsMessenger_) {
    vkDestroyDebugUtilsMessengerEXT(vkInstance_, vkDebugUtilsMessenger_, nullptr);
  }

  vkDestroyInstance(vkInstance_, nullptr);

  glslang_finalize_process();

  LLOGL("Vulkan graphics pipelines created: %u\n", VulkanPipelineBuilder::getNumPipelinesCreated());
}

lvk::ICommandBuffer& lvk::VulkanContext::acquireCommandBuffer() {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT_MSG(!pimpl_->currentCommandBuffer_.ctx_, "Cannot acquire more than 1 command buffer simultaneously");

#if defined(_M_ARM64)
  vkDeviceWaitIdle(vkDevice_); // a temporary workaround for Windows on Snapdragon
#endif

  pimpl_->currentCommandBuffer_ = CommandBuffer(this);

  return pimpl_->currentCommandBuffer_;
}

lvk::SubmitHandle lvk::VulkanContext::submit(lvk::ICommandBuffer& commandBuffer, TextureHandle present) {
  LVK_PROFILER_FUNCTION();

  CommandBuffer* vkCmdBuffer = static_cast<CommandBuffer*>(&commandBuffer);

  LVK_ASSERT(vkCmdBuffer);
  LVK_ASSERT(vkCmdBuffer->ctx_);
  LVK_ASSERT(vkCmdBuffer->wrapper_);

#if defined(LVK_WITH_TRACY_GPU)
  TracyVkCollect(pimpl_->tracyVkCtx_, vkCmdBuffer->wrapper_->cmdBuf_);
#endif // LVK_WITH_TRACY_GPU

  if (present) {
    const lvk::VulkanImage& tex = *texturesPool_.get(present);

    LVK_ASSERT(tex.isSwapchainImage_);

    tex.transitionLayout(vkCmdBuffer->wrapper_->cmdBuf_,
                         VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                         VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS});
  }

  const bool shouldPresent = hasSwapchain() && present;

  if (shouldPresent) {
    // if we a presenting a swapchain image, signal our timeline semaphore
    const uint64_t signalValue = swapchain_->currentFrameIndex_ + swapchain_->getNumSwapchainImages();
    // we wait for this value next time we want to acquire this swapchain image
    swapchain_->timelineWaitValues_[swapchain_->currentImageIndex_] = signalValue;
    immediate_->signalSemaphore(timelineSemaphore_, signalValue);
  }

  vkCmdBuffer->lastSubmitHandle_ = immediate_->submit(*vkCmdBuffer->wrapper_);

  if (shouldPresent) {
    swapchain_->present(immediate_->acquireLastSubmitSemaphore());
  }

  processDeferredTasks();

  SubmitHandle handle = vkCmdBuffer->lastSubmitHandle_;

  // assign the last submit handle to all previous "orphan" dsets
  const size_t numSets = DSets_.size();
  for (size_t count = 0; count != numSets; count++) {
    // calculate index wrapping backwards
    const size_t idx = (lastUpdatedDSet_ + numSets - 1 - count) % numSets;
    if (!DSets_[idx].handle_.empty()) {
      // break once we reach a "non-orphan" dset
      break;
    }
    DSets_[idx].handle_ = handle;
  }

  // reset
  pimpl_->currentCommandBuffer_ = {};

  return handle;
}

void lvk::VulkanContext::wait(SubmitHandle handle) {
  immediate_->wait(handle);
}

lvk::Holder<lvk::BufferHandle> lvk::VulkanContext::createBuffer(const BufferDesc& requestedDesc, const char* debugName, Result* outResult) {
  BufferDesc desc = requestedDesc;

  if (debugName && *debugName)
    desc.debugName = debugName;

  if (!useStaging_ && (desc.storage == StorageType_Device)) {
    desc.storage = StorageType_HostVisible;
  }

  // Use staging device to transfer data into the buffer when the storage is private to the device
  VkBufferUsageFlags usageFlags = (desc.storage == StorageType_Device) ? VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                       : 0;

  if (desc.usage == 0) {
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "Invalid buffer usage"));
    return {};
  }

  if (desc.usage & BufferUsageBits_Index) {
    usageFlags |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  }
  if (desc.usage & BufferUsageBits_Vertex) {
    usageFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  }
  if (desc.usage & BufferUsageBits_Uniform) {
    usageFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (desc.usage & BufferUsageBits_Storage) {
    usageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (desc.usage & BufferUsageBits_Indirect) {
    usageFlags |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (desc.usage & BufferUsageBits_ShaderBindingTable) {
    usageFlags |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (desc.usage & BufferUsageBits_AccelStructBuildInputReadOnly) {
    usageFlags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  if (desc.usage & BufferUsageBits_AccelStructStorage) {
    usageFlags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  LVK_ASSERT_MSG(usageFlags, "Invalid buffer usage");

  const VkMemoryPropertyFlags memFlags = storageTypeToVkMemoryPropertyFlags(desc.storage);

  Result result;
  BufferHandle handle = createBuffer(desc.size, usageFlags, memFlags, &result, desc.debugName);

  if (!LVK_VERIFY(result.isOk())) {
    Result::setResult(outResult, result);
    return {};
  }

  if (desc.data) {
    upload(handle, desc.data, desc.size, 0);
  }

  Result::setResult(outResult, Result());

  return {this, handle};
}

lvk::Holder<lvk::QueryPoolHandle> lvk::VulkanContext::createQueryPool(uint32_t numQueries, const char* debugName, Result* outResult) {
  LVK_PROFILER_FUNCTION();

  const VkQueryPoolCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .flags = 0,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = numQueries,
      .pipelineStatistics = 0,
  };

  VkQueryPool queryPool = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateQueryPool(vkDevice_, &createInfo, 0, &queryPool));

  if (!queryPool) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "Cannot create QueryPool"));
    return {};
  }

  if (debugName && *debugName) {
    VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_QUERY_POOL, (uint64_t)queryPool, debugName));
  }

  lvk::QueryPoolHandle handle = queriesPool_.create(std::move(queryPool));

  return {this, handle};
}

lvk::Holder<lvk::AccelStructHandle> lvk::VulkanContext::createAccelerationStructure(const AccelStructDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(has_KHR_acceleration_structure_)) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "VK_KHR_acceleration_structure is not enabled"));
    return {};
  }

  Result result;

  AccelStructHandle handle;

  switch (desc.type) {
  case AccelStructType_BLAS:
    handle = createBLAS(desc, &result);
    break;
  case AccelStructType_TLAS:
    handle = createTLAS(desc, &result);
    break;
  default:
    LVK_ASSERT_MSG(false, "Invalid acceleration structure type");
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "Invalid acceleration structure type"));
    return {};
  }

  if (!LVK_VERIFY(result.isOk() && handle.valid())) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "Cannot create AccelerationStructure"));
    return {};
  }

  Result::setResult(outResult, result);

  awaitingCreation_ = true;

  return {this, handle};
}

lvk::Holder<lvk::SamplerHandle> lvk::VulkanContext::createSampler(const SamplerStateDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION();

  Result result;

  const VkSamplerCreateInfo info = samplerStateDescToVkSamplerCreateInfo(desc, getVkPhysicalDeviceProperties().limits);

  SamplerHandle handle = createSampler(info, &result, Format_Invalid, desc.debugName);

  if (!LVK_VERIFY(result.isOk())) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "Cannot create Sampler"));
    return {};
  }

  Result::setResult(outResult, result);

  return {this, handle};
}

lvk::Holder<lvk::TextureHandle> lvk::VulkanContext::createTexture(const TextureDesc& requestedDesc,
                                                                  const char* debugName,
                                                                  Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  TextureDesc desc(requestedDesc);

  if (debugName && *debugName) {
    desc.debugName = debugName;
  }

  auto getClosestDepthStencilFormat = [this](lvk::Format desiredFormat) -> VkFormat {
    // Get a list of compatible depth formats for a given desired format.
    // The list will contain depth format that are ordered from most to least closest
    const std::vector<VkFormat> compatibleDepthStencilFormatList = getCompatibleDepthStencilFormats(desiredFormat);

    // check if any of the format in compatible list is supported
    for (VkFormat format : compatibleDepthStencilFormatList) {
      if (std::find(deviceDepthFormats_.cbegin(), deviceDepthFormats_.cend(), format) != deviceDepthFormats_.cend()) {
        return format;
      }
    }

    // no matching found, choose the first supported format
    return !deviceDepthFormats_.empty() ? deviceDepthFormats_[0] : VK_FORMAT_D24_UNORM_S8_UINT;
  };

  const VkFormat vkFormat = lvk::isDepthOrStencilFormat(desc.format) ? getClosestDepthStencilFormat(desc.format)
                                                                     : formatToVkFormat(desc.format);

  LVK_ASSERT_MSG(vkFormat != VK_FORMAT_UNDEFINED, "Invalid VkFormat value");

  const lvk::TextureType type = desc.type;
  if (!LVK_VERIFY(type == TextureType_2D || type == TextureType_Cube || type == TextureType_3D)) {
    LVK_ASSERT_MSG(false, "Only 2D, 3D and Cube textures are supported");
    Result::setResult(outResult, Result::Code::RuntimeError);
    return {};
  }

  if (desc.numMipLevels == 0) {
    LVK_ASSERT_MSG(false, "The number of mip levels specified must be greater than 0");
    desc.numMipLevels = 1;
  }

  if (desc.numSamples > 1 && desc.numMipLevels != 1) {
    LVK_ASSERT_MSG(false, "The number of mip levels for multisampled images should be 1");
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "The number of mip-levels for multisampled images should be 1");
    return {};
  }

  if (desc.numSamples > 1 && type == TextureType_3D) {
    LVK_ASSERT_MSG(false, "Multisampled 3D images are not supported");
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Multisampled 3D images are not supported");
    return {};
  }

  if (!LVK_VERIFY(desc.numMipLevels <= lvk::calcNumMipLevels(desc.dimensions.width, desc.dimensions.height))) {
    Result::setResult(outResult,
                      Result::Code::ArgumentOutOfRange,
                      "The number of specified mip-levels is greater than the maximum possible "
                      "number of mip-levels.");
    return {};
  }

  if (desc.usage == 0) {
    LVK_ASSERT_MSG(false, "Texture usage flags are not set");
    desc.usage = lvk::TextureUsageBits_Sampled;
  }

  /* Use staging device to transfer data into the image when the storage is private to the device */
  VkImageUsageFlags usageFlags = (desc.storage == StorageType_Device) ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0;

  if (desc.usage & lvk::TextureUsageBits_Sampled) {
    usageFlags |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (desc.usage & lvk::TextureUsageBits_Storage) {
    LVK_ASSERT_MSG(desc.numSamples <= 1, "Storage images cannot be multisampled");
    usageFlags |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (desc.usage & lvk::TextureUsageBits_Attachment) {
    usageFlags |= lvk::isDepthOrStencilFormat(desc.format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                                           : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (desc.storage == lvk::StorageType_Memoryless) {
      usageFlags |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
    }
  }
  if (desc.usage & lvk::TextureUsageBits_InputAttachment) {
    LVK_ASSERT_MSG(desc.usage & lvk::TextureUsageBits_Attachment, "Input attachments must be TextureUsageBits_Attachment");
    usageFlags |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
  }

  if (desc.storage != lvk::StorageType_Memoryless) {
    // For now, always set this flag so we can read it back
    usageFlags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }

  LVK_ASSERT_MSG(usageFlags != 0, "Invalid usage flags");

  const VkMemoryPropertyFlags memFlags = storageTypeToVkMemoryPropertyFlags(desc.storage);

  const bool hasDebugName = desc.debugName && *desc.debugName;

  char debugNameImage[256] = {0};
  char debugNameImageView[256] = {0};

  if (hasDebugName) {
    snprintf(debugNameImage, sizeof(debugNameImage) - 1, "Image: %s", desc.debugName);
    snprintf(debugNameImageView, sizeof(debugNameImageView) - 1, "Image View: %s", desc.debugName);
  }

  VkImageCreateFlags vkCreateFlags = 0;
  VkImageViewType vkImageViewType;
  VkImageType vkImageType;
  VkSampleCountFlagBits vkSamples = VK_SAMPLE_COUNT_1_BIT;
  uint32_t numLayers = desc.numLayers;
  switch (desc.type) {
  case TextureType_2D:
    vkImageViewType = numLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vkImageType = VK_IMAGE_TYPE_2D;
    vkSamples = lvk::getVulkanSampleCountFlags(desc.numSamples, getFramebufferMSAABitMask());
    break;
  case TextureType_3D:
    vkImageViewType = VK_IMAGE_VIEW_TYPE_3D;
    vkImageType = VK_IMAGE_TYPE_3D;
    break;
  case TextureType_Cube:
    vkImageViewType = numLayers > 1 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    vkImageType = VK_IMAGE_TYPE_2D;
    vkCreateFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    numLayers *= 6;
    break;
  default:
    LVK_ASSERT_MSG(false, "Code should NOT be reached");
    Result::setResult(outResult, Result::Code::RuntimeError, "Unsupported texture type");
    return {};
  }

  const VkExtent3D vkExtent{desc.dimensions.width, desc.dimensions.height, desc.dimensions.depth};
  const uint32_t numLevels = desc.numMipLevels;

  if (!LVK_VERIFY(validateImageLimits(vkImageType, vkSamples, vkExtent, getVkPhysicalDeviceProperties().limits, outResult))) {
    return {};
  }

  LVK_ASSERT_MSG(numLevels > 0, "The image must contain at least one mip-level");
  LVK_ASSERT_MSG(numLayers > 0, "The image must contain at least one layer");
  LVK_ASSERT_MSG(vkSamples > 0, "The image must contain at least one sample");
  LVK_ASSERT(vkExtent.width > 0);
  LVK_ASSERT(vkExtent.height > 0);
  LVK_ASSERT(vkExtent.depth > 0);

  lvk::VulkanImage image = {
      .vkUsageFlags_ = usageFlags,
      .vkExtent_ = vkExtent,
      .vkType_ = vkImageType,
      .vkImageFormat_ = vkFormat,
      .vkSamples_ = vkSamples,
      .numLevels_ = numLevels,
      .numLayers_ = numLayers,
      .isDepthFormat_ = VulkanImage::isDepthFormat(vkFormat),
      .isStencilFormat_ = VulkanImage::isStencilFormat(vkFormat),
  };

  if (hasDebugName) {
    // store debug name
    snprintf(image.debugName_, sizeof(image.debugName_) - 1, "%s", desc.debugName);
  }

  const uint32_t numPlanes = lvk::getNumImagePlanes(desc.format);
  const bool isDisjoint = numPlanes > 1;

  if (isDisjoint) {
    // some constraints for multiplanar image formats
    LVK_ASSERT(vkImageType == VK_IMAGE_TYPE_2D);
    LVK_ASSERT(vkSamples == VK_SAMPLE_COUNT_1_BIT);
    LVK_ASSERT(numLayers == 1);
    LVK_ASSERT(numLevels == 1);
    VkFormatProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
    };
    vkGetPhysicalDeviceFormatProperties2(vkPhysicalDevice_, vkFormat, &props);
    if ((props.formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DISJOINT_BIT) == 0) {
      LLOGW("VK_FORMAT_FEATURE_DISJOINT_BIT is not supported for VkFormat = %u\n", (uint32_t)vkFormat);
    }
    vkCreateFlags |= VK_IMAGE_CREATE_DISJOINT_BIT | VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    awaitingNewImmutableSamplers_ = true;
  }

  const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = vkCreateFlags,
      .imageType = vkImageType,
      .format = vkFormat,
      .extent = vkExtent,
      .mipLevels = numLevels,
      .arrayLayers = numLayers,
      .samples = vkSamples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  if (LVK_VULKAN_USE_VMA && numPlanes == 1) {
    VmaAllocationCreateInfo vmaAllocInfo = {
        .usage = memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? VMA_MEMORY_USAGE_CPU_TO_GPU : VMA_MEMORY_USAGE_AUTO,
    };

    VkResult result = vmaCreateImage((VmaAllocator)getVmaAllocator(), &ci, &vmaAllocInfo, &image.vkImage_, &image.vmaAllocation_, nullptr);

    if (!LVK_VERIFY(result == VK_SUCCESS)) {
      LLOGW("Failed: error result: %d, memflags: %d,  imageformat: %d\n", result, memFlags, image.vkImageFormat_);
      Result::setResult(outResult, Result::Code::RuntimeError, "vmaCreateImage() failed");
      return {};
    }

    // handle memory-mapped buffers
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      vmaMapMemory((VmaAllocator)getVmaAllocator(), image.vmaAllocation_, &image.mappedPtr_);
    }
  } else {
    // create image
    VK_ASSERT(vkCreateImage(vkDevice_, &ci, nullptr, &image.vkImage_));

    // back the image with some memory
    constexpr uint32_t kNumMaxImagePlanes = LVK_ARRAY_NUM_ELEMENTS(image.vkMemory_);

    VkMemoryRequirements2 memRequirements[kNumMaxImagePlanes] = {
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
        {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2},
    };

    const VkImagePlaneMemoryRequirementsInfo planes[kNumMaxImagePlanes] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO, .planeAspect = VK_IMAGE_ASPECT_PLANE_0_BIT},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO, .planeAspect = VK_IMAGE_ASPECT_PLANE_1_BIT},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO, .planeAspect = VK_IMAGE_ASPECT_PLANE_2_BIT},
    };

    const VkImage img = image.vkImage_;

    const VkImageMemoryRequirementsInfo2 imgRequirements[kNumMaxImagePlanes] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .pNext = numPlanes > 0 ? &planes[0] : nullptr, .image = img},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .pNext = numPlanes > 1 ? &planes[1] : nullptr, .image = img},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .pNext = numPlanes > 2 ? &planes[2] : nullptr, .image = img},
    };

    const VkDeviceSize maxMemoryAllocationSize = vkPhysicalDeviceVulkan11Properties_.maxMemoryAllocationSize;

    for (uint32_t p = 0; p != numPlanes; p++) {
      vkGetImageMemoryRequirements2(vkDevice_, &imgRequirements[p], &memRequirements[p]);
      LVK_ASSERT(memRequirements[p].memoryRequirements.size <= maxMemoryAllocationSize);
      VK_ASSERT(lvk::allocateMemory2(vkPhysicalDevice_, vkDevice_, &memRequirements[p], memFlags, &image.vkMemory_[p]));
    }

    const VkBindImagePlaneMemoryInfo bindImagePlaneMemoryInfo[kNumMaxImagePlanes] = {
        {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, VK_IMAGE_ASPECT_PLANE_0_BIT},
        {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, VK_IMAGE_ASPECT_PLANE_1_BIT},
        {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, VK_IMAGE_ASPECT_PLANE_2_BIT},
    };
    const VkBindImageMemoryInfo bindInfo[kNumMaxImagePlanes] = {
        lvk::getBindImageMemoryInfo(isDisjoint ? &bindImagePlaneMemoryInfo[0] : nullptr, img, image.vkMemory_[0]),
        lvk::getBindImageMemoryInfo(&bindImagePlaneMemoryInfo[1], img, image.vkMemory_[1]),
        lvk::getBindImageMemoryInfo(&bindImagePlaneMemoryInfo[2], img, image.vkMemory_[2]),
    };
    VK_ASSERT(vkBindImageMemory2(vkDevice_, numPlanes, bindInfo));

    // handle memory-mapped images
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT && numPlanes == 1) {
      VK_ASSERT(vkMapMemory(vkDevice_, image.vkMemory_[0], 0, VK_WHOLE_SIZE, 0, &image.mappedPtr_));
    }
  }

  VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_IMAGE, (uint64_t)image.vkImage_, debugNameImage));

  // Get physical device's properties for the image's format
  vkGetPhysicalDeviceFormatProperties(vkPhysicalDevice_, image.vkImageFormat_, &image.vkFormatProperties_);

  VkImageAspectFlags aspect = 0;
  if (image.isDepthFormat_ || image.isStencilFormat_) {
    if (image.isDepthFormat_) {
      aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    } else if (image.isStencilFormat_) {
      aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  }

  const VkComponentMapping components = {
      .r = VkComponentSwizzle(desc.components.r),
      .g = VkComponentSwizzle(desc.components.g),
      .b = VkComponentSwizzle(desc.components.b),
      .a = VkComponentSwizzle(desc.components.a),
  };

  const VkSamplerYcbcrConversionInfo* ycbcrInfo = isDisjoint ? getOrCreateYcbcrConversionInfo(desc.format) : nullptr;

  image.imageView_ = image.createImageView(
      vkDevice_, vkImageViewType, vkFormat, aspect, 0, VK_REMAINING_MIP_LEVELS, 0, numLayers, components, ycbcrInfo, debugNameImageView);

  if (image.vkUsageFlags_ & VK_IMAGE_USAGE_STORAGE_BIT) {
    if (!desc.components.identity()) {
      // use identity swizzle for storage images
      image.imageViewStorage_ = image.createImageView(
          vkDevice_, vkImageViewType, vkFormat, aspect, 0, VK_REMAINING_MIP_LEVELS, 0, numLayers, {}, ycbcrInfo, debugNameImageView);
      LVK_ASSERT(image.imageViewStorage_ != VK_NULL_HANDLE);
    }
  }

  if (!LVK_VERIFY(image.imageView_ != VK_NULL_HANDLE)) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Cannot create VkImageView");
    return {};
  }

  TextureHandle handle = texturesPool_.create(std::move(image));

  awaitingCreation_ = true;

  if (desc.data) {
    LVK_ASSERT(desc.type == TextureType_2D || desc.type == TextureType_Cube);
    LVK_ASSERT(desc.dataNumMipLevels <= desc.numMipLevels);
    const uint32_t numLayers = desc.type == TextureType_Cube ? 6 : 1;
    Result res = upload(handle, {.dimensions = desc.dimensions, .numLayers = numLayers, .numMipLevels = desc.dataNumMipLevels}, desc.data);
    if (!res.isOk()) {
      Result::setResult(outResult, res);
      return {};
    }
    if (desc.generateMipmaps) {
      this->generateMipmap(handle);
    }
  }

  Result::setResult(outResult, Result());

  return {this, handle};
}

lvk::Holder<lvk::TextureHandle> lvk::VulkanContext::createTextureView(lvk::TextureHandle texture,
                                                                      const TextureViewDesc& desc,
                                                                      const char* debugName,
                                                                      Result* outResult) {
  if (!texture) {
    LVK_ASSERT(texture.valid());
    return {};
  }

  // make a copy and make it non-owning
  VulkanImage image = *texturesPool_.get(texture);
  image.isOwningVkImage_ = false;

  // drop all existing image views - they belong to the base image
  memset(&image.imageViewStorage_, 0, sizeof(image.imageViewStorage_));
  memset(&image.imageViewForFramebuffer_, 0, sizeof(image.imageViewForFramebuffer_));
  memset(&image.imageViewForFramebufferMultiview_, 0, sizeof(image.imageViewForFramebufferMultiview_));

  VkImageAspectFlags aspect = 0;
  if (image.isDepthFormat_ || image.isStencilFormat_) {
    if (image.isDepthFormat_) {
      aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    } else if (image.isStencilFormat_) {
      aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  } else {
    aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  }

  VkImageViewType vkImageViewType = VK_IMAGE_VIEW_TYPE_MAX_ENUM;
  switch (desc.type) {
  case TextureType_2D:
    vkImageViewType = desc.numLayers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    break;
  case TextureType_3D:
    vkImageViewType = VK_IMAGE_VIEW_TYPE_3D;
    break;
  case TextureType_Cube:
    vkImageViewType = desc.numLayers > 1 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    break;
  default:
    LVK_ASSERT_MSG(false, "Code should NOT be reached");
    Result::setResult(outResult, Result::Code::RuntimeError, "Unsupported texture view type");
    return {};
  }

  const VkComponentMapping components = {
      .r = VkComponentSwizzle(desc.components.r),
      .g = VkComponentSwizzle(desc.components.g),
      .b = VkComponentSwizzle(desc.components.b),
      .a = VkComponentSwizzle(desc.components.a),
  };

  LVK_ASSERT_MSG(lvk::getNumImagePlanes(image.vkImageFormat_) == 1, "Unsupported multiplanar image");

  image.imageView_ = image.createImageView(vkDevice_,
                                           vkImageViewType,
                                           image.vkImageFormat_,
                                           aspect,
                                           desc.mipLevel,
                                           desc.numMipLevels,
                                           desc.layer,
                                           desc.numLayers,
                                           components,
                                           nullptr,
                                           debugName);

  if (!LVK_VERIFY(image.imageView_ != VK_NULL_HANDLE)) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Cannot create VkImageView");
    return {};
  }

  if (image.vkUsageFlags_ & VK_IMAGE_USAGE_STORAGE_BIT) {
    if (!desc.components.identity()) {
      // use identity swizzle for storage images
      image.imageViewStorage_ = image.createImageView(vkDevice_,
                                                      vkImageViewType,
                                                      image.vkImageFormat_,
                                                      aspect,
                                                      desc.mipLevel,
                                                      desc.numMipLevels,
                                                      desc.layer,
                                                      desc.numLayers,
                                                      {},
                                                      nullptr,
                                                      debugName);
      LVK_ASSERT(image.imageViewStorage_ != VK_NULL_HANDLE);
    }
  }

  TextureHandle handle = texturesPool_.create(std::move(image));

  awaitingCreation_ = true;

  return {this, handle};
}

lvk::AccelStructHandle lvk::VulkanContext::createBLAS(const AccelStructDesc& desc, Result* outResult) {
  VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
  VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
  getBuildInfoBLAS(desc, accelerationStructureGeometry, accelerationStructureBuildSizesInfo);

  char debugNameBuffer[256] = {0};
  if (desc.debugName) {
    snprintf(debugNameBuffer, sizeof(debugNameBuffer) - 1, "Buffer: %s", desc.debugName);
  }
  lvk::AccelerationStructure accelStruct = {
      .buildRangeInfo =
          {
              .primitiveCount = desc.buildRange.primitiveCount,
              .primitiveOffset = desc.buildRange.primitiveOffset,
              .firstVertex = desc.buildRange.firstVertex,
              .transformOffset = desc.buildRange.transformOffset,
          },
      .buffer = createBuffer(
          {
              .usage = lvk::BufferUsageBits_AccelStructStorage,
              .storage = lvk::StorageType_Device,
              .size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
              .debugName = debugNameBuffer,
          },
          nullptr,
          outResult),
  };
  const VkAccelerationStructureCreateInfoKHR ciAccelerationStructure = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = getVkBuffer(this, accelStruct.buffer),
      .size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
  };
  VK_ASSERT(vkCreateAccelerationStructureKHR(vkDevice_, &ciAccelerationStructure, nullptr, &accelStruct.vkHandle));

  lvk::Holder<lvk::BufferHandle> scratchBuffer = createBuffer(
      {
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = accelerationStructureBuildSizesInfo.buildScratchSize,
          .debugName = "Buffer: BLAS scratch",
      },
      nullptr,
      outResult);

  const VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = buildFlagsToVkBuildAccelerationStructureFlags(desc.buildFlags),
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .dstAccelerationStructure = accelStruct.vkHandle,
      .geometryCount = 1,
      .pGeometries = &accelerationStructureGeometry,
      .scratchData = {.deviceAddress = getAlignedAddress(gpuAddress(scratchBuffer),
                                                         accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment)},
  };

  const VkAccelerationStructureBuildRangeInfoKHR* accelerationBuildStructureRangeInfos[] = {&accelStruct.buildRangeInfo};

  lvk::ICommandBuffer& buffer = acquireCommandBuffer();
  vkCmdBuildAccelerationStructuresKHR(
      lvk::getVkCommandBuffer(buffer), 1, &accelerationBuildGeometryInfo, accelerationBuildStructureRangeInfos);
  wait(submit(buffer, {}));

  const VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
      .accelerationStructure = accelStruct.vkHandle,
  };
  accelStruct.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(vkDevice_, &accelerationDeviceAddressInfo);

  return accelStructuresPool_.create(std::move(accelStruct));
}

lvk::AccelStructHandle lvk::VulkanContext::createTLAS(const AccelStructDesc& desc, Result* outResult) {
  VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
  VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
  getBuildInfoTLAS(desc, accelerationStructureGeometry, accelerationStructureBuildSizesInfo);

  char debugNameBuffer[256] = {0};
  if (desc.debugName) {
    snprintf(debugNameBuffer, sizeof(debugNameBuffer) - 1, "Buffer: %s", desc.debugName);
  }
  lvk::AccelerationStructure accelStruct = {
      .isTLAS = true,
      .buildRangeInfo =
          {
              .primitiveCount = desc.buildRange.primitiveCount,
              .primitiveOffset = desc.buildRange.primitiveOffset,
              .firstVertex = desc.buildRange.firstVertex,
              .transformOffset = desc.buildRange.transformOffset,
          },
      .buffer = createBuffer(
          {
              .usage = lvk::BufferUsageBits_AccelStructStorage,
              .storage = lvk::StorageType_Device,
              .size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
              .debugName = debugNameBuffer,
          },
          nullptr,
          outResult),
  };

  const VkAccelerationStructureCreateInfoKHR ciAccelerationStructure = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = getVkBuffer(this, accelStruct.buffer),
      .size = accelerationStructureBuildSizesInfo.accelerationStructureSize,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
  };
  vkCreateAccelerationStructureKHR(vkDevice_, &ciAccelerationStructure, nullptr, &accelStruct.vkHandle);

  lvk::Holder<lvk::BufferHandle> scratchBuffer = createBuffer(
      {
          .usage = lvk::BufferUsageBits_Storage,
          .storage = lvk::StorageType_Device,
          .size = accelerationStructureBuildSizesInfo.buildScratchSize,
          .debugName = "Buffer: TLAS scratch",
      },
      nullptr,
      outResult);

  const VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = buildFlagsToVkBuildAccelerationStructureFlags(desc.buildFlags),
      .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
      .dstAccelerationStructure = accelStruct.vkHandle,
      .geometryCount = 1,
      .pGeometries = &accelerationStructureGeometry,
      .scratchData = {.deviceAddress = getAlignedAddress(gpuAddress(scratchBuffer),
                                                         accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment)},
  };
  if (desc.buildFlags & lvk::AccelStructBuildFlagBits_AllowUpdate) {
    // Store scratch buffer for future updates
    accelStruct.scratchBuffer = std::move(scratchBuffer);
  }

  const VkAccelerationStructureBuildRangeInfoKHR* accelerationBuildStructureRangeInfos[] = {&accelStruct.buildRangeInfo};

  lvk::ICommandBuffer& buffer = acquireCommandBuffer();
  vkCmdBuildAccelerationStructuresKHR(
      lvk::getVkCommandBuffer(buffer), 1, &accelerationBuildGeometryInfo, accelerationBuildStructureRangeInfos);
  wait(submit(buffer, {}));

  const VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
      .accelerationStructure = accelStruct.vkHandle,
  };
  accelStruct.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(vkDevice_, &accelerationDeviceAddressInfo);

  return accelStructuresPool_.create(std::move(accelStruct));
}

static_assert(1 << (sizeof(lvk::Format) * 8) <= LVK_ARRAY_NUM_ELEMENTS(lvk::VulkanContextImpl::ycbcrConversionData_),
              "There aren't enough elements in `ycbcrConversionData_` to be accessed by lvk::Format");

VkSampler lvk::VulkanContext::getOrCreateYcbcrSampler(lvk::Format format) {
  const VkSamplerYcbcrConversionInfo* info = getOrCreateYcbcrConversionInfo(format);

  if (!info) {
    return VK_NULL_HANDLE;
  }

  return *samplersPool_.get(pimpl_->ycbcrConversionData_[format].sampler);
}

const VkSamplerYcbcrConversionInfo* lvk::VulkanContext::getOrCreateYcbcrConversionInfo(lvk::Format format) {
  if (pimpl_->ycbcrConversionData_[format].info.sType) {
    return &pimpl_->ycbcrConversionData_[format].info;
  }

  if (!LVK_VERIFY(vkFeatures11_.samplerYcbcrConversion)) {
    LVK_ASSERT_MSG(false, "Ycbcr samplers are not supported");
    return nullptr;
  }

  const VkFormat vkFormat = lvk::formatToVkFormat(format);

  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(getVkPhysicalDevice(), vkFormat, &props);

  const bool cosited = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT) != 0;
  const bool midpoint = (props.optimalTilingFeatures & VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT) != 0;

  if (!LVK_VERIFY(cosited || midpoint)) {
    LVK_ASSERT_MSG(cosited || midpoint, "Unsupported Ycbcr feature");
    return nullptr;
  }

  const VkSamplerYcbcrConversionCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
      .format = vkFormat,
      .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
      .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
      .components =
          {
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .xChromaOffset = midpoint ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN,
      .yChromaOffset = midpoint ? VK_CHROMA_LOCATION_MIDPOINT : VK_CHROMA_LOCATION_COSITED_EVEN,
      .chromaFilter = VK_FILTER_LINEAR,
      .forceExplicitReconstruction = VK_FALSE,
  };

  VkSamplerYcbcrConversionInfo info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
      .pNext = nullptr,
  };
  vkCreateSamplerYcbcrConversion(vkDevice_, &ci, nullptr, &info.conversion);

  // check properties
  VkSamplerYcbcrConversionImageFormatProperties samplerYcbcrConversionImageFormatProps = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
  };
  VkImageFormatProperties2 imageFormatProps = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &samplerYcbcrConversionImageFormatProps,
  };
  const VkPhysicalDeviceImageFormatInfo2 imageFormatInfo = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = vkFormat,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .flags = VK_IMAGE_CREATE_DISJOINT_BIT,
  };
  vkGetPhysicalDeviceImageFormatProperties2(getVkPhysicalDevice(), &imageFormatInfo, &imageFormatProps);

  LVK_ASSERT(samplerYcbcrConversionImageFormatProps.combinedImageSamplerDescriptorCount <= 3);

  // update maximum
  pimpl_->maxCombinedImageSamplerDescriptorCount_ =
      std::max(pimpl_->maxCombinedImageSamplerDescriptorCount_, samplerYcbcrConversionImageFormatProps.combinedImageSamplerDescriptorCount);

  const VkSamplerCreateInfo cinfo = samplerStateDescToVkSamplerCreateInfo({}, getVkPhysicalDeviceProperties().limits);

  pimpl_->ycbcrConversionData_[format].info = info;
  pimpl_->ycbcrConversionData_[format].sampler = {this, this->createSampler(cinfo, nullptr, format, "YUV sampler")};
  pimpl_->numYcbcrSamplers_++;
  awaitingNewImmutableSamplers_ = true;

  return &pimpl_->ycbcrConversionData_[format].info;
}

VkPipeline lvk::VulkanContext::getVkPipeline(RenderPipelineHandle handle, uint32_t viewMask) {
  lvk::RenderPipelineState* rps = renderPipelinesPool_.get(handle);

  if (!rps) {
    return VK_NULL_HANDLE;
  }

  checkAndUpdateDescriptorSets();

  const DescriptorSet& dset = DSets_[lastUpdatedDSet_];

  if (rps->lastVkDescriptorSetLayout_ != dset.vkDSL || rps->viewMask_ != viewMask) {
    deferredTask(std::packaged_task<void()>(
        [device = getVkDevice(), pipeline = rps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
    deferredTask(std::packaged_task<void()>(
        [device = getVkDevice(), layout = rps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));
    rps->pipeline_ = VK_NULL_HANDLE;
    rps->lastVkDescriptorSetLayout_ = dset.vkDSL;
    rps->viewMask_ = viewMask;
  }

  if (rps->pipeline_ != VK_NULL_HANDLE) {
    return rps->pipeline_;
  }

  // build a new Vulkan pipeline

  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;

  const RenderPipelineDesc& desc = rps->desc_;

  const uint32_t numColorAttachments = rps->desc_.getNumColorAttachments();

  // Not all attachments are valid. We need to create color blend attachments only for active attachments
  VkPipelineColorBlendAttachmentState colorBlendAttachmentStates[LVK_MAX_COLOR_ATTACHMENTS] = {};
  VkFormat colorAttachmentFormats[LVK_MAX_COLOR_ATTACHMENTS] = {};

  for (uint32_t i = 0; i != numColorAttachments; i++) {
    const lvk::ColorAttachment& attachment = desc.color[i];
    LVK_ASSERT(attachment.format != Format_Invalid);
    colorAttachmentFormats[i] = formatToVkFormat(attachment.format);
    if (!attachment.blendEnabled) {
      colorBlendAttachmentStates[i] = VkPipelineColorBlendAttachmentState{
          .blendEnable = VK_FALSE,
          .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
          .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
          .colorBlendOp = VK_BLEND_OP_ADD,
          .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
          .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
          .alphaBlendOp = VK_BLEND_OP_ADD,
          .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
    } else {
      colorBlendAttachmentStates[i] = VkPipelineColorBlendAttachmentState{
          .blendEnable = VK_TRUE,
          .srcColorBlendFactor = blendFactorToVkBlendFactor(attachment.srcRGBBlendFactor),
          .dstColorBlendFactor = blendFactorToVkBlendFactor(attachment.dstRGBBlendFactor),
          .colorBlendOp = blendOpToVkBlendOp(attachment.rgbBlendOp),
          .srcAlphaBlendFactor = blendFactorToVkBlendFactor(attachment.srcAlphaBlendFactor),
          .dstAlphaBlendFactor = blendFactorToVkBlendFactor(attachment.dstAlphaBlendFactor),
          .alphaBlendOp = blendOpToVkBlendOp(attachment.alphaBlendOp),
          .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
    }
  }

  const lvk::ShaderModuleState* vertModule = shaderModulesPool_.get(desc.smVert);
  const lvk::ShaderModuleState* tescModule = shaderModulesPool_.get(desc.smTesc);
  const lvk::ShaderModuleState* teseModule = shaderModulesPool_.get(desc.smTese);
  const lvk::ShaderModuleState* geomModule = shaderModulesPool_.get(desc.smGeom);
  const lvk::ShaderModuleState* fragModule = shaderModulesPool_.get(desc.smFrag);
  const lvk::ShaderModuleState* taskModule = shaderModulesPool_.get(desc.smTask);
  const lvk::ShaderModuleState* meshModule = shaderModulesPool_.get(desc.smMesh);

  LVK_ASSERT(vertModule || meshModule);
  LVK_ASSERT(fragModule);

  if (tescModule || teseModule || desc.patchControlPoints) {
    LVK_ASSERT_MSG(tescModule && teseModule, "Both tessellation control and evaluation shaders should be provided");
    LVK_ASSERT(desc.patchControlPoints > 0 &&
               desc.patchControlPoints <= vkPhysicalDeviceProperties2_.properties.limits.maxTessellationPatchSize);
  }

  const VkPipelineVertexInputStateCreateInfo ciVertexInputState = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = rps->numBindings_,
      .pVertexBindingDescriptions = rps->numBindings_ ? rps->vkBindings_ : nullptr,
      .vertexAttributeDescriptionCount = rps->numAttributes_,
      .pVertexAttributeDescriptions = rps->numAttributes_ ? rps->vkAttributes_ : nullptr,
  };

  VkSpecializationMapEntry entries[SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX] = {};

  const VkSpecializationInfo si = lvk::getPipelineShaderStageSpecializationInfo(desc.specInfo, entries);

  // create pipeline layout
  {
#define UPDATE_PUSH_CONSTANT_SIZE(sm, bit)                                  \
  if (sm) {                                                                 \
    pushConstantsSize = std::max(pushConstantsSize, sm->pushConstantsSize); \
    rps->shaderStageFlags_ |= bit;                                          \
  }
    rps->shaderStageFlags_ = 0;
    uint32_t pushConstantsSize = 0;
    UPDATE_PUSH_CONSTANT_SIZE(vertModule, VK_SHADER_STAGE_VERTEX_BIT);
    UPDATE_PUSH_CONSTANT_SIZE(tescModule, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    UPDATE_PUSH_CONSTANT_SIZE(teseModule, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    UPDATE_PUSH_CONSTANT_SIZE(geomModule, VK_SHADER_STAGE_GEOMETRY_BIT);
    UPDATE_PUSH_CONSTANT_SIZE(fragModule, VK_SHADER_STAGE_FRAGMENT_BIT);
    UPDATE_PUSH_CONSTANT_SIZE(taskModule, VK_SHADER_STAGE_TASK_BIT_EXT);
    UPDATE_PUSH_CONSTANT_SIZE(meshModule, VK_SHADER_STAGE_MESH_BIT_EXT);
#undef UPDATE_PUSH_CONSTANT_SIZE

    // maxPushConstantsSize is guaranteed to be at least 128 bytes
    // https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#features-limits
    // Table 32. Required Limits
    const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;
    if (!LVK_VERIFY(pushConstantsSize <= limits.maxPushConstantsSize)) {
      LLOGW("Push constants size exceeded %u (max %u bytes)", pushConstantsSize, limits.maxPushConstantsSize);
    }

    // duplicate for MoltenVK
    const VkDescriptorSetLayout dsls[kDescriptorSet_InputAttachments + 1] = {
        dset.vkDSL,
        dset.vkDSL,
        dset.vkDSL,
        dset.vkDSL,
        dslInputAttachments_,
    };
    const VkPushConstantRange range = {
        .stageFlags = rps->shaderStageFlags_,
        .offset = 0,
        .size = (uint32_t)getAlignedSize(pushConstantsSize, 16),
    };
    const VkPipelineLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = (uint32_t)LVK_ARRAY_NUM_ELEMENTS(dsls),
        .pSetLayouts = dsls,
        .pushConstantRangeCount = pushConstantsSize ? 1u : 0u,
        .pPushConstantRanges = pushConstantsSize ? &range : nullptr,
    };
    VK_ASSERT(vkCreatePipelineLayout(vkDevice_, &ci, nullptr, &layout));
    char pipelineLayoutName[256] = {0};
    if (rps->desc_.debugName) {
      snprintf(pipelineLayoutName, sizeof(pipelineLayoutName) - 1, "Pipeline Layout: %s", rps->desc_.debugName);
    }
    VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)layout, pipelineLayoutName));
  }

  lvk::VulkanPipelineBuilder()
      // from Vulkan 1.0
      .dynamicState(VK_DYNAMIC_STATE_VIEWPORT)
      .dynamicState(VK_DYNAMIC_STATE_SCISSOR)
      .dynamicState(VK_DYNAMIC_STATE_DEPTH_BIAS)
      .dynamicState(VK_DYNAMIC_STATE_BLEND_CONSTANTS)
      // from Vulkan 1.3 or VK_EXT_extended_dynamic_state
      .dynamicState(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE)
      .dynamicState(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE)
      .dynamicState(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP)
      // from Vulkan 1.3 or VK_EXT_extended_dynamic_state2
      .dynamicState(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE)
      .primitiveTopology(topologyToVkPrimitiveTopology(desc.topology))
      .rasterizationSamples(getVulkanSampleCountFlags(desc.samplesCount, getFramebufferMSAABitMask()), desc.minSampleShading)
      .polygonMode(polygonModeToVkPolygonMode(desc.polygonMode))
      .stencilStateOps(VK_STENCIL_FACE_FRONT_BIT,
                       stencilOpToVkStencilOp(desc.frontFaceStencil.stencilFailureOp),
                       stencilOpToVkStencilOp(desc.frontFaceStencil.depthStencilPassOp),
                       stencilOpToVkStencilOp(desc.frontFaceStencil.depthFailureOp),
                       compareOpToVkCompareOp(desc.frontFaceStencil.stencilCompareOp))
      .stencilStateOps(VK_STENCIL_FACE_BACK_BIT,
                       stencilOpToVkStencilOp(desc.backFaceStencil.stencilFailureOp),
                       stencilOpToVkStencilOp(desc.backFaceStencil.depthStencilPassOp),
                       stencilOpToVkStencilOp(desc.backFaceStencil.depthFailureOp),
                       compareOpToVkCompareOp(desc.backFaceStencil.stencilCompareOp))
      .stencilMasks(VK_STENCIL_FACE_FRONT_BIT, 0xFF, desc.frontFaceStencil.writeMask, desc.frontFaceStencil.readMask)
      .stencilMasks(VK_STENCIL_FACE_BACK_BIT, 0xFF, desc.backFaceStencil.writeMask, desc.backFaceStencil.readMask)
      .shaderStage(taskModule
                       ? lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_TASK_BIT_EXT, taskModule->ci, desc.entryPointTask, &si)
                       : VkPipelineShaderStageCreateInfo{.module = VK_NULL_HANDLE})
      .shaderStage(meshModule
                       ? lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_MESH_BIT_EXT, meshModule->ci, desc.entryPointMesh, &si)
                       : lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertModule->ci, desc.entryPointVert, &si))
      .shaderStage(lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragModule->ci, desc.entryPointFrag, &si))
      .shaderStage(tescModule ? lvk::getPipelineShaderStageCreateInfo(
                                    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tescModule->ci, desc.entryPointTesc, &si)
                              : VkPipelineShaderStageCreateInfo{.module = VK_NULL_HANDLE})
      .shaderStage(teseModule ? lvk::getPipelineShaderStageCreateInfo(
                                    VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, teseModule->ci, desc.entryPointTese, &si)
                              : VkPipelineShaderStageCreateInfo{.module = VK_NULL_HANDLE})
      .shaderStage(geomModule
                       ? lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_GEOMETRY_BIT, geomModule->ci, desc.entryPointGeom, &si)
                       : VkPipelineShaderStageCreateInfo{.module = VK_NULL_HANDLE})
      .cullMode(cullModeToVkCullMode(desc.cullMode))
      .frontFace(windingModeToVkFrontFace(desc.frontFace))
      .vertexInputState(ciVertexInputState)
      .viewMask(viewMask)
      .colorAttachments(colorBlendAttachmentStates, colorAttachmentFormats, numColorAttachments)
      .depthAttachmentFormat(formatToVkFormat(desc.depthFormat))
      .stencilAttachmentFormat(formatToVkFormat(desc.stencilFormat))
      .patchControlPoints(desc.patchControlPoints)
      .build(vkDevice_, pipelineCache_, layout, &pipeline, desc.debugName);

  rps->pipeline_ = pipeline;
  rps->pipelineLayout_ = layout;

  return pipeline;
}

VkPipeline lvk::VulkanContext::getVkPipeline(RayTracingPipelineHandle handle) {
  lvk::RayTracingPipelineState* rtps = rayTracingPipelinesPool_.get(handle);

  if (!rtps) {
    return VK_NULL_HANDLE;
  }

  checkAndUpdateDescriptorSets();

  const DescriptorSet& dset = DSets_[lastUpdatedDSet_];

  if (rtps->lastVkDescriptorSetLayout_ != dset.vkDSL) {
    deferredTask(
        std::packaged_task<void()>([device = vkDevice_, pipeline = rtps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
    deferredTask(std::packaged_task<void()>(
        [device = vkDevice_, layout = rtps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));
    rtps->pipeline_ = VK_NULL_HANDLE;
    rtps->pipelineLayout_ = VK_NULL_HANDLE;
    rtps->lastVkDescriptorSetLayout_ = dset.vkDSL;
  }

  if (rtps->pipeline_) {
    return rtps->pipeline_;
  }

  // build a new Vulkan ray tracing pipeline
  const RayTracingPipelineDesc& desc = rtps->desc_;

  const lvk::ShaderModuleState* moduleRGen[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  const lvk::ShaderModuleState* moduleAHit[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  const lvk::ShaderModuleState* moduleCHit[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  const lvk::ShaderModuleState* moduleMiss[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  const lvk::ShaderModuleState* moduleIntr[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  const lvk::ShaderModuleState* moduleCall[LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE] = {};
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {
    if (desc.smRayGen[i]) {
      moduleRGen[i] = shaderModulesPool_.get(desc.smRayGen[i]);
    }
    if (desc.smAnyHit[i]) {
      moduleAHit[i] = shaderModulesPool_.get(desc.smAnyHit[i]);
    }
    if (desc.smClosestHit[i]) {
      moduleCHit[i] = shaderModulesPool_.get(desc.smClosestHit[i]);
    }
    if (desc.smMiss[i]) {
      moduleMiss[i] = shaderModulesPool_.get(desc.smMiss[i]);
    }
    if (desc.smIntersection[i]) {
      moduleIntr[i] = shaderModulesPool_.get(desc.smIntersection[i]);
    }
    if (desc.smCallable[i]) {
      moduleCall[i] = shaderModulesPool_.get(desc.smCallable[i]);
    }
  }

  LVK_ASSERT(moduleRGen);

  // create pipeline layout
  {
#define UPDATE_PUSH_CONSTANT_SIZE(sm, bit)                                       \
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {              \
    if (sm[i] && sm[i]->pushConstantsSize) {                                     \
      pushConstantsSize = std::max(pushConstantsSize, sm[i]->pushConstantsSize); \
      rtps->shaderStageFlags_ |= bit;                                            \
    }                                                                            \
  }
    rtps->shaderStageFlags_ = 0;
    uint32_t pushConstantsSize = 0;
    UPDATE_PUSH_CONSTANT_SIZE(moduleRGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    UPDATE_PUSH_CONSTANT_SIZE(moduleAHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
    UPDATE_PUSH_CONSTANT_SIZE(moduleCHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    UPDATE_PUSH_CONSTANT_SIZE(moduleMiss, VK_SHADER_STAGE_MISS_BIT_KHR);
    UPDATE_PUSH_CONSTANT_SIZE(moduleIntr, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
    UPDATE_PUSH_CONSTANT_SIZE(moduleCall, VK_SHADER_STAGE_CALLABLE_BIT_KHR);
#undef UPDATE_PUSH_CONSTANT_SIZE

    // maxPushConstantsSize is guaranteed to be at least 128 bytes
    // https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#features-limits
    // Table 32. Required Limits
    const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;
    if (!LVK_VERIFY(pushConstantsSize <= limits.maxPushConstantsSize)) {
      LLOGW("Push constants size exceeded %u (max %u bytes)", pushConstantsSize, limits.maxPushConstantsSize);
    }

    const VkDescriptorSetLayout dsls[] = {dset.vkDSL, dset.vkDSL, dset.vkDSL, dset.vkDSL};
    const VkPushConstantRange range = {
        .stageFlags = rtps->shaderStageFlags_,
        .size = (uint32_t)getAlignedSize(pushConstantsSize, 16),
    };

    const VkPipelineLayoutCreateInfo ciPipelineLayout = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = LVK_ARRAY_NUM_ELEMENTS(dsls),
        .pSetLayouts = dsls,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    };
    VK_ASSERT(vkCreatePipelineLayout(vkDevice_, &ciPipelineLayout, nullptr, &rtps->pipelineLayout_));
    char pipelineLayoutName[256] = {0};
    if (rtps->desc_.debugName) {
      snprintf(pipelineLayoutName, sizeof(pipelineLayoutName) - 1, "Pipeline Layout: %s", rtps->desc_.debugName);
    }
    VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)rtps->pipelineLayout_, pipelineLayoutName));
  }

  VkSpecializationMapEntry entries[SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX] = {};

  const VkSpecializationInfo siComp = lvk::getPipelineShaderStageSpecializationInfo(rtps->desc_.specInfo, entries);

  const uint32_t kMaxRayTracingShaderStages = 6 * LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE;
  VkPipelineShaderStageCreateInfo ciShaderStages[kMaxRayTracingShaderStages];
  uint32_t numShaderStages = 0;
#define ADD_STAGE(shaderModule, vkStageFlag)                                                                                        \
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {                                                                 \
    if (shaderModule[i])                                                                                                            \
      ciShaderStages[numShaderStages++] = lvk::getPipelineShaderStageCreateInfo(vkStageFlag, shaderModule[i]->ci, "main", &siComp); \
  }
  ADD_STAGE(moduleRGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  ADD_STAGE(moduleMiss, VK_SHADER_STAGE_MISS_BIT_KHR);
  ADD_STAGE(moduleCHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  ADD_STAGE(moduleAHit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
  ADD_STAGE(moduleIntr, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
  ADD_STAGE(moduleCall, VK_SHADER_STAGE_CALLABLE_BIT_KHR);
#undef ADD_STAGE

  const uint32_t kMaxShaderGroups = 4;
  VkRayTracingShaderGroupCreateInfoKHR shaderGroups[kMaxShaderGroups];
  uint32_t numShaderGroups = 0;
  uint32_t numShaders = 0;
  uint32_t idxMiss = 0;
  uint32_t idxHit = 0;
  uint32_t idxCallable = 0;
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {
    if (moduleRGen[i]) {
      // ray generation group
      shaderGroups[numShaderGroups++] = VkRayTracingShaderGroupCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
          .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
          .generalShader = numShaders++,
          .closestHitShader = VK_SHADER_UNUSED_KHR,
          .anyHitShader = VK_SHADER_UNUSED_KHR,
          .intersectionShader = VK_SHADER_UNUSED_KHR,
      };
    }
  }
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {
    if (moduleMiss[i]) {
      // miss group
      if (i == 0)
        idxMiss = numShaders;
      shaderGroups[numShaderGroups++] = VkRayTracingShaderGroupCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
          .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
          .generalShader = numShaders++,
          .closestHitShader = VK_SHADER_UNUSED_KHR,
          .anyHitShader = VK_SHADER_UNUSED_KHR,
          .intersectionShader = VK_SHADER_UNUSED_KHR,
      };
    }
  }
  // hit group
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {
    if (moduleAHit[i] || moduleCHit[i] || moduleIntr[i]) {
      if (i == 0)
        idxHit = numShaders;
      shaderGroups[numShaderGroups++] = VkRayTracingShaderGroupCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
          .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
          .generalShader = VK_SHADER_UNUSED_KHR,
          .closestHitShader = moduleCHit[i] ? numShaders++ : VK_SHADER_UNUSED_KHR,
          .anyHitShader = moduleAHit[i] ? numShaders++ : VK_SHADER_UNUSED_KHR,
          .intersectionShader = moduleIntr[i] ? numShaders++ : VK_SHADER_UNUSED_KHR,
      };
    }
  }
  // callable group
  for (int i = 0; i < LVK_MAX_RAY_TRACING_SHADER_GROUP_SIZE; ++i) {
    if (moduleCall[i]) {
      if (i == 0)
        idxCallable = numShaders;
      shaderGroups[numShaderGroups++] = VkRayTracingShaderGroupCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
          .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
          .generalShader = numShaders++,
          .closestHitShader = VK_SHADER_UNUSED_KHR,
          .anyHitShader = VK_SHADER_UNUSED_KHR,
          .intersectionShader = VK_SHADER_UNUSED_KHR,
      };
    }
  }

  const VkRayTracingPipelineCreateInfoKHR ciRayTracingPipeline = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount = numShaderStages,
      .pStages = ciShaderStages,
      .groupCount = numShaderGroups,
      .pGroups = shaderGroups,
      .maxPipelineRayRecursionDepth = rayTracingPipelineProperties_.maxRayRecursionDepth,
      .layout = rtps->pipelineLayout_,
  };
  VK_ASSERT(vkCreateRayTracingPipelinesKHR(vkDevice_, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ciRayTracingPipeline, nullptr, &rtps->pipeline_));

  // shader binding table
  const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& props = rayTracingPipelineProperties_;
  const uint32_t handleSize = props.shaderGroupHandleSize;
  const uint32_t handleSizeAligned = getAlignedSize(props.shaderGroupHandleSize, props.shaderGroupHandleAlignment);
  const uint32_t sbtSize = numShaderGroups * handleSizeAligned;

  LVK_ASSERT(sbtSize);

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  VK_ASSERT(vkGetRayTracingShaderGroupHandlesKHR(vkDevice_, rtps->pipeline_, 0, numShaderGroups, sbtSize, shaderHandleStorage.data()));

  const uint32_t sbtEntrySizeAligned = getAlignedSize(handleSizeAligned, props.shaderGroupBaseAlignment);
  const uint32_t sbtBufferSize = numShaderGroups * sbtEntrySizeAligned;

  // repack SBT respecting `shaderGroupBaseAlignment`
  std::vector<uint8_t> sbtStorage(sbtBufferSize);
  for (uint32_t i = 0; i != numShaderGroups; i++) {
    memcpy(sbtStorage.data() + i * sbtEntrySizeAligned, shaderHandleStorage.data() + i * handleSizeAligned, handleSize);
  }

  rtps->sbt = createBuffer(
      {
          .usage = lvk::BufferUsageBits_ShaderBindingTable,
          .storage = lvk::StorageType_Device,
          .size = sbtBufferSize + props.shaderGroupBaseAlignment,
          .debugName = "Buffer: SBT",
      },
      nullptr,
      nullptr);
  const uint64_t baseAddress = getAlignedAddress(gpuAddress(rtps->sbt), props.shaderGroupBaseAlignment);
  const uint64_t offset = baseAddress - gpuAddress(rtps->sbt);
  upload(rtps->sbt, sbtStorage.data(), sbtBufferSize, offset);
  // generate SBT entries
  rtps->sbtEntryRayGen = {
      .deviceAddress = baseAddress,
      .stride = handleSizeAligned,
      .size = handleSizeAligned,
  };
  rtps->sbtEntryMiss = {
      .deviceAddress = idxMiss ? baseAddress + idxMiss * sbtEntrySizeAligned : 0,
      .stride = handleSizeAligned,
      .size = handleSizeAligned,
  };
  rtps->sbtEntryHit = {
      .deviceAddress = idxHit ? baseAddress + idxHit * sbtEntrySizeAligned : 0,
      .stride = handleSizeAligned,
      .size = handleSizeAligned,
  };
  rtps->sbtEntryCallable = {
      .deviceAddress = idxCallable ? baseAddress + idxCallable * sbtEntrySizeAligned : 0,
      .stride = handleSizeAligned,
      .size = handleSizeAligned,
  };

  return rtps->pipeline_;
}

VkPipeline lvk::VulkanContext::getVkPipeline(ComputePipelineHandle handle) {
  lvk::ComputePipelineState* cps = computePipelinesPool_.get(handle);

  if (!cps) {
    return VK_NULL_HANDLE;
  }

  checkAndUpdateDescriptorSets();

  const DescriptorSet& dset = DSets_[lastUpdatedDSet_];

  if (cps->lastVkDescriptorSetLayout_ != dset.vkDSL) {
    deferredTask(
        std::packaged_task<void()>([device = vkDevice_, pipeline = cps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
    deferredTask(std::packaged_task<void()>(
        [device = vkDevice_, layout = cps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));
    cps->pipeline_ = VK_NULL_HANDLE;
    cps->pipelineLayout_ = VK_NULL_HANDLE;
    cps->lastVkDescriptorSetLayout_ = dset.vkDSL;
  }

  if (cps->pipeline_ == VK_NULL_HANDLE) {
    const lvk::ShaderModuleState* sm = shaderModulesPool_.get(cps->desc_.smComp);

    LVK_ASSERT(sm);

    VkSpecializationMapEntry entries[SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX] = {};

    const VkSpecializationInfo siComp = lvk::getPipelineShaderStageSpecializationInfo(cps->desc_.specInfo, entries);

    // create pipeline layout
    {
      // duplicate for MoltenVK
      const VkDescriptorSetLayout dsls[] = {dset.vkDSL, dset.vkDSL, dset.vkDSL, dset.vkDSL};
      const VkPushConstantRange range = {
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
          .offset = 0,
          .size = (uint32_t)getAlignedSize(sm->pushConstantsSize, 16),
      };
      const VkPipelineLayoutCreateInfo ci = {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
          .setLayoutCount = (uint32_t)LVK_ARRAY_NUM_ELEMENTS(dsls),
          .pSetLayouts = dsls,
          .pushConstantRangeCount = 1,
          .pPushConstantRanges = &range,
      };
      VK_ASSERT(vkCreatePipelineLayout(vkDevice_, &ci, nullptr, &cps->pipelineLayout_));
      char pipelineLayoutName[256] = {0};
      if (cps->desc_.debugName) {
        snprintf(pipelineLayoutName, sizeof(pipelineLayoutName) - 1, "Pipeline Layout: %s", cps->desc_.debugName);
      }
      VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)cps->pipelineLayout_, pipelineLayoutName));
    }

    const VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .flags = 0,
        .stage = lvk::getPipelineShaderStageCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, sm->ci, cps->desc_.entryPoint, &siComp),
        .layout = cps->pipelineLayout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    VK_ASSERT(vkCreateComputePipelines(vkDevice_, pipelineCache_, 1, &ci, nullptr, &cps->pipeline_));
    VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_PIPELINE, (uint64_t)cps->pipeline_, cps->desc_.debugName));
  }

  return cps->pipeline_;
}

lvk::Holder<lvk::ComputePipelineHandle> lvk::VulkanContext::createComputePipeline(const ComputePipelineDesc& desc, Result* outResult) {
  if (!LVK_VERIFY(desc.smComp.valid())) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Missing compute shader");
    return {};
  }

  lvk::ComputePipelineState cps{desc};

  if (desc.specInfo.data && desc.specInfo.dataSize) {
    // copy into a local storage
    cps.specConstantDataStorage_ = malloc(desc.specInfo.dataSize);
    memcpy(cps.specConstantDataStorage_, desc.specInfo.data, desc.specInfo.dataSize);
    cps.desc_.specInfo.data = cps.specConstantDataStorage_;
  }

  return {this, computePipelinesPool_.create(std::move(cps))};
}

lvk::Holder<lvk::RayTracingPipelineHandle> lvk::VulkanContext::createRayTracingPipeline(const RayTracingPipelineDesc& desc,
                                                                                        Result* outResult) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(has_KHR_ray_tracing_pipeline_)) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "VK_KHR_ray_tracing_pipeline is not enabled"));
    return {};
  }

  RayTracingPipelineState rtps{desc};

  if (desc.specInfo.data && desc.specInfo.dataSize) {
    // copy into a local storage
    rtps.specConstantDataStorage_ = malloc(desc.specInfo.dataSize);
    memcpy(rtps.specConstantDataStorage_, desc.specInfo.data, desc.specInfo.dataSize);
    rtps.desc_.specInfo.data = rtps.specConstantDataStorage_;
  }

  return {this, rayTracingPipelinesPool_.create(std::move(rtps))};
}

lvk::Holder<lvk::RenderPipelineHandle> lvk::VulkanContext::createRenderPipeline(const RenderPipelineDesc& desc, Result* outResult) {
  const bool hasColorAttachments = desc.getNumColorAttachments() > 0;
  const bool hasDepthAttachment = desc.depthFormat != Format_Invalid;
  const bool hasAnyAttachments = hasColorAttachments || hasDepthAttachment;
  if (!LVK_VERIFY(hasAnyAttachments)) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Need at least one attachment");
    return {};
  }

  if (desc.smMesh.valid()) {
    if (!LVK_VERIFY(!desc.vertexInput.getNumAttributes() && !desc.vertexInput.getNumInputBindings())) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Cannot have vertexInput with mesh shaders");
      return {};
    }
    if (!LVK_VERIFY(!desc.smVert.valid())) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Cannot have both vertex and mesh shaders");
      return {};
    }
    if (!LVK_VERIFY(!desc.smTesc.valid() && !desc.smTese.valid())) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Cannot have both tessellation and mesh shaders");
      return {};
    }
    if (!LVK_VERIFY(!desc.smGeom.valid())) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Cannot have both geometry and mesh shaders");
      return {};
    }
  } else {
    if (!LVK_VERIFY(desc.smVert.valid())) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Missing vertex shader");
      return {};
    }
  }

  if (!LVK_VERIFY(desc.smFrag.valid())) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Missing fragment shader");
    return {};
  }

  RenderPipelineState rps = {.desc_ = desc};

  // Iterate and cache vertex input bindings and attributes
  const lvk::VertexInput& vstate = rps.desc_.vertexInput;

  bool bufferAlreadyBound[VertexInput::LVK_VERTEX_BUFFER_MAX] = {};

  rps.numAttributes_ = vstate.getNumAttributes();

  for (uint32_t i = 0; i != rps.numAttributes_; i++) {
    const VertexInput::VertexAttribute& attr = vstate.attributes[i];

    rps.vkAttributes_[i] = {
        .location = attr.location, .binding = attr.binding, .format = vertexFormatToVkFormat(attr.format), .offset = (uint32_t)attr.offset};

    if (!bufferAlreadyBound[attr.binding]) {
      bufferAlreadyBound[attr.binding] = true;
      rps.vkBindings_[rps.numBindings_++] = {
          .binding = attr.binding,
          .stride = vstate.inputBindings[attr.binding].stride,
          .inputRate = vertexInputRateToVkVertexInputRate(vstate.inputBindings[attr.binding].inputRate),
      };
    }
  }

  if (desc.specInfo.data && desc.specInfo.dataSize) {
    // copy into a local storage
    rps.specConstantDataStorage_ = malloc(desc.specInfo.dataSize);
    memcpy(rps.specConstantDataStorage_, desc.specInfo.data, desc.specInfo.dataSize);
    rps.desc_.specInfo.data = rps.specConstantDataStorage_;
  }

  return {this, renderPipelinesPool_.create(std::move(rps))};
}

void lvk::VulkanContext::destroy(lvk::RayTracingPipelineHandle handle) {
  lvk::RayTracingPipelineState* rtps = rayTracingPipelinesPool_.get(handle);

  if (!rtps) {
    return;
  }

  free(rtps->specConstantDataStorage_);

  deferredTask(
      std::packaged_task<void()>([device = getVkDevice(), pipeline = rtps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
  deferredTask(std::packaged_task<void()>(
      [device = getVkDevice(), layout = rtps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));

  rayTracingPipelinesPool_.destroy(handle);
}

void lvk::VulkanContext::destroy(lvk::ComputePipelineHandle handle) {
  lvk::ComputePipelineState* cps = computePipelinesPool_.get(handle);

  if (!cps) {
    return;
  }

  free(cps->specConstantDataStorage_);

  deferredTask(
      std::packaged_task<void()>([device = getVkDevice(), pipeline = cps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
  deferredTask(std::packaged_task<void()>(
      [device = getVkDevice(), layout = cps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));

  computePipelinesPool_.destroy(handle);
}

void lvk::VulkanContext::destroy(lvk::RenderPipelineHandle handle) {
  lvk::RenderPipelineState* rps = renderPipelinesPool_.get(handle);

  if (!rps) {
    return;
  }

  free(rps->specConstantDataStorage_);

  deferredTask(
      std::packaged_task<void()>([device = getVkDevice(), pipeline = rps->pipeline_]() { vkDestroyPipeline(device, pipeline, nullptr); }));
  deferredTask(std::packaged_task<void()>(
      [device = getVkDevice(), layout = rps->pipelineLayout_]() { vkDestroyPipelineLayout(device, layout, nullptr); }));

  renderPipelinesPool_.destroy(handle);
}

void lvk::VulkanContext::destroy(lvk::ShaderModuleHandle handle) {
  const lvk::ShaderModuleState* state = shaderModulesPool_.get(handle);

  if (!state) {
    return;
  }

  free((void*)state->ci.pCode);

  shaderModulesPool_.destroy(handle);
}

void lvk::VulkanContext::destroy(SamplerHandle handle) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_DESTROY);

  VkSampler sampler = *samplersPool_.get(handle);

  samplersPool_.destroy(handle);

  deferredTask(std::packaged_task<void()>([device = vkDevice_, sampler = sampler]() { vkDestroySampler(device, sampler, nullptr); }));
}

void lvk::VulkanContext::destroy(BufferHandle handle) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_DESTROY);

  SCOPE_EXIT {
    buffersPool_.destroy(handle);
  };

  lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  if (!buf) {
    return;
  }

  if (LVK_VULKAN_USE_VMA) {
    if (buf->mappedPtr_) {
      vmaUnmapMemory((VmaAllocator)getVmaAllocator(), buf->vmaAllocation_);
    }
    deferredTask(std::packaged_task<void()>([vma = getVmaAllocator(), buffer = buf->vkBuffer_, allocation = buf->vmaAllocation_]() {
      vmaDestroyBuffer((VmaAllocator)vma, buffer, allocation);
    }));
  } else {
    if (buf->mappedPtr_) {
      vkUnmapMemory(vkDevice_, buf->vkMemory_);
    }
    deferredTask(std::packaged_task<void()>([device = vkDevice_, buffer = buf->vkBuffer_, memory = buf->vkMemory_]() {
      vkDestroyBuffer(device, buffer, nullptr);
      vkFreeMemory(device, memory, nullptr);
    }));
  }
}

void lvk::VulkanContext::destroy(lvk::TextureHandle handle) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_DESTROY);

  SCOPE_EXIT {
    texturesPool_.destroy(handle);
    awaitingCreation_ = true; // make the validation layers happy
  };

  lvk::VulkanImage* tex = texturesPool_.get(handle);

  if (!tex) {
    return;
  }

  deferredTask(std::packaged_task<void()>(
      [device = getVkDevice(), imageView = tex->imageView_]() { vkDestroyImageView(device, imageView, nullptr); }));
  if (tex->imageViewStorage_) {
    deferredTask(std::packaged_task<void()>(
        [device = getVkDevice(), imageView = tex->imageViewStorage_]() { vkDestroyImageView(device, imageView, nullptr); }));
  }

  for (size_t i = 0; i != LVK_MAX_MIP_LEVELS; i++) {
    for (size_t j = 0; j != LVK_ARRAY_NUM_ELEMENTS(tex->imageViewForFramebuffer_[0]); j++) {
      VkImageView v = tex->imageViewForFramebuffer_[i][j];
      if (v != VK_NULL_HANDLE) {
        deferredTask(
            std::packaged_task<void()>([device = getVkDevice(), imageView = v]() { vkDestroyImageView(device, imageView, nullptr); }));
      }
    }
    VkImageView v = tex->imageViewForFramebufferMultiview_[i];
    if (v != VK_NULL_HANDLE) {
      deferredTask(
          std::packaged_task<void()>([device = getVkDevice(), imageView = v]() { vkDestroyImageView(device, imageView, nullptr); }));
    }
  }

  if (!tex->isOwningVkImage_) {
    return;
  }

  if (LVK_VULKAN_USE_VMA && tex->vkMemory_[1] == VK_NULL_HANDLE) {
    if (tex->mappedPtr_) {
      vmaUnmapMemory((VmaAllocator)getVmaAllocator(), tex->vmaAllocation_);
    }
    deferredTask(std::packaged_task<void()>([vma = getVmaAllocator(), image = tex->vkImage_, allocation = tex->vmaAllocation_]() {
      vmaDestroyImage((VmaAllocator)vma, image, allocation);
    }));
  } else {
    if (tex->mappedPtr_) {
      vkUnmapMemory(vkDevice_, tex->vkMemory_[0]);
    }
    deferredTask(std::packaged_task<void()>([device = vkDevice_,
                                             image = tex->vkImage_,
                                             memory0 = tex->vkMemory_[0],
                                             memory1 = tex->vkMemory_[1],
                                             memory2 = tex->vkMemory_[2]]() {
      vkDestroyImage(device, image, nullptr);
      if (memory0 != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory0, nullptr);
      }
      if (memory1 != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory1, nullptr);
      }
      if (memory2 != VK_NULL_HANDLE) {
        vkFreeMemory(device, memory2, nullptr);
      }
    }));
  }
}

void lvk::VulkanContext::destroy(lvk::QueryPoolHandle handle) {
  VkQueryPool pool = *queriesPool_.get(handle);

  queriesPool_.destroy(handle);

  deferredTask(std::packaged_task<void()>([device = vkDevice_, pool = pool]() { vkDestroyQueryPool(device, pool, nullptr); }));
}

void lvk::VulkanContext::destroy(lvk::AccelStructHandle handle) {
  AccelerationStructure* accelStruct = accelStructuresPool_.get(handle);

  SCOPE_EXIT {
    accelStructuresPool_.destroy(handle);
  };

  deferredTask(std::packaged_task<void()>(
      [device = vkDevice_, as = accelStruct->vkHandle]() { vkDestroyAccelerationStructureKHR(device, as, nullptr); }));
}

void lvk::VulkanContext::destroy(Framebuffer& fb) {
  auto destroyFbTexture = [this](TextureHandle& handle) {
    {
      if (handle.empty())
        return;
      lvk::VulkanImage* tex = texturesPool_.get(handle);
      if (!tex || !tex->isOwningVkImage_)
        return;
      destroy(handle);
      handle = {};
    }
  };

  for (Framebuffer::AttachmentDesc& a : fb.color) {
    destroyFbTexture(a.texture);
    destroyFbTexture(a.resolveTexture);
  }
  destroyFbTexture(fb.depthStencil.texture);
  destroyFbTexture(fb.depthStencil.resolveTexture);
}

uint64_t lvk::VulkanContext::gpuAddress(AccelStructHandle handle) const {
  const lvk::AccelerationStructure* as = accelStructuresPool_.get(handle);

  LVK_ASSERT(as && as->deviceAddress);

  return as ? (uint64_t)as->deviceAddress : 0u;
}

lvk::Result lvk::VulkanContext::upload(lvk::BufferHandle handle, const void* data, size_t size, size_t offset) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(data)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  LVK_ASSERT_MSG(size, "Data size should be non-zero");

  lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  if (!LVK_VERIFY(buf)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  if (!LVK_VERIFY(offset + size <= buf->bufferSize_)) {
    return Result(Result::Code::ArgumentOutOfRange, "Out of range");
  }

  stagingDevice_->bufferSubData(*buf, offset, size, data);

  return Result();
}

lvk::Result lvk::VulkanContext::download(lvk::BufferHandle handle, void* data, size_t size, size_t offset) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(data)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  LVK_ASSERT_MSG(size, "Data size should be non-zero");

  lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  if (!LVK_VERIFY(buf)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  if (!LVK_VERIFY(offset + size <= buf->bufferSize_)) {
    return Result(Result::Code::ArgumentOutOfRange, "Out of range");
  }

  buf->getBufferSubData(*this, offset, size, data);

  return Result();
}

uint8_t* lvk::VulkanContext::getMappedPtr(BufferHandle handle) const {
  const lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  LVK_ASSERT(buf);

  return buf->isMapped() ? buf->getMappedPtr() : nullptr;
}

uint64_t lvk::VulkanContext::gpuAddress(BufferHandle handle, size_t offset) const {
  LVK_ASSERT_MSG((offset & 7) == 0, "Buffer offset must be 8 bytes aligned as per GLSL_EXT_buffer_reference spec.");

  const lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  LVK_ASSERT(buf && buf->vkDeviceAddress_);

  return buf ? (uint64_t)buf->vkDeviceAddress_ + offset : 0u;
}

void lvk::VulkanContext::flushMappedMemory(BufferHandle handle, size_t offset, size_t size) const {
  const lvk::VulkanBuffer* buf = buffersPool_.get(handle);

  LVK_ASSERT(buf);

  buf->flushMappedMemory(*this, offset, size);
}

lvk::Result lvk::VulkanContext::download(lvk::TextureHandle handle, const TextureRangeDesc& range, void* outData) {
  if (!outData) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  lvk::VulkanImage* texture = texturesPool_.get(handle);

  LVK_ASSERT(texture);

  if (!texture) {
    return Result(Result::Code::RuntimeError);
  }

  const Result result = validateRange(texture->vkExtent_, texture->numLevels_, range);

  if (!LVK_VERIFY(result.isOk())) {
    return result;
  }

  stagingDevice_->getImageData(*texture,
                               VkOffset3D{range.offset.x, range.offset.y, range.offset.z},
                               VkExtent3D{range.dimensions.width, range.dimensions.height, range.dimensions.depth},
                               VkImageSubresourceRange{
                                   .aspectMask = texture->getImageAspectFlags(),
                                   .baseMipLevel = range.mipLevel,
                                   .levelCount = range.numMipLevels,
                                   .baseArrayLayer = range.layer,
                                   .layerCount = range.numLayers,
                               },
                               texture->vkImageFormat_,
                               outData);

  return Result();
}

lvk::Result lvk::VulkanContext::upload(lvk::TextureHandle handle,
                                       const TextureRangeDesc& range,
                                       const void* data,
                                       uint32_t bufferRowLength) {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(data)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  lvk::VulkanImage* texture = texturesPool_.get(handle);

  if (!LVK_VERIFY(texture)) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  const Result result = validateRange(texture->vkExtent_, texture->numLevels_, range);

  if (!LVK_VERIFY(result.isOk())) {
    return Result(Result::Code::ArgumentOutOfRange);
  }

  const uint32_t numLayers = std::max(range.numLayers, 1u);

  VkFormat vkFormat = texture->vkImageFormat_;

  if (texture->vkType_ == VK_IMAGE_TYPE_3D) {
    stagingDevice_->imageData3D(*texture,
                                VkOffset3D{range.offset.x, range.offset.y, range.offset.z},
                                VkExtent3D{range.dimensions.width, range.dimensions.height, range.dimensions.depth},
                                vkFormat,
                                data);
  } else {
    const VkRect2D imageRegion = {
        .offset = {.x = range.offset.x, .y = range.offset.y},
        .extent = {.width = range.dimensions.width, .height = range.dimensions.height},
    };
    stagingDevice_->imageData2D(
        *texture, imageRegion, range.mipLevel, range.numMipLevels, range.layer, range.numLayers, vkFormat, data, bufferRowLength);
  }

  return Result();
}

lvk::Dimensions lvk::VulkanContext::getDimensions(TextureHandle handle) const {
  if (!handle) {
    return {};
  }

  const lvk::VulkanImage* tex = texturesPool_.get(handle);

  LVK_ASSERT(tex);

  if (!tex) {
    return {};
  }

  return {
      .width = tex->vkExtent_.width,
      .height = tex->vkExtent_.height,
      .depth = tex->vkExtent_.depth,
  };
}

float lvk::VulkanContext::getAspectRatio(TextureHandle handle) const {
  if (!handle) {
    return 1.0f;
  }

  const lvk::VulkanImage* tex = texturesPool_.get(handle);

  LVK_ASSERT(tex);

  if (!tex) {
    return 1.0f;
  }

  return static_cast<float>(tex->vkExtent_.width) / static_cast<float>(tex->vkExtent_.height);
}

void lvk::VulkanContext::generateMipmap(TextureHandle handle) const {
  if (handle.empty()) {
    return;
  }

  const lvk::VulkanImage* tex = texturesPool_.get(handle);

  if (tex->numLevels_ <= 1) {
    return;
  }

  LVK_ASSERT(tex->vkImageLayout_ != VK_IMAGE_LAYOUT_UNDEFINED);
  const lvk::VulkanImmediateCommands::CommandBufferWrapper& wrapper = immediate_->acquire();
  tex->generateMipmap(wrapper.cmdBuf_);
  immediate_->submit(wrapper);
}

lvk::Format lvk::VulkanContext::getFormat(TextureHandle handle) const {
  if (handle.empty()) {
    return Format_Invalid;
  }

  return vkFormatToFormat(texturesPool_.get(handle)->vkImageFormat_);
}

lvk::Holder<lvk::ShaderModuleHandle> lvk::VulkanContext::createShaderModule(const ShaderModuleDesc& desc, Result* outResult) {
  Result result;
  auto isSlang = [](const char* code) {
    if (!code)
      return false;
    return strstr(code, "[shader(\"") != nullptr;
  };
  ShaderModuleState sm = desc.dataSize ? createShaderModuleFromSPIRV(desc.data, desc.dataSize, desc.debugName, &result) // binary
                         : isSlang(desc.data) // text
                             ? createShaderModuleFromSlang(desc.stage, desc.data, desc.entryPointName, desc.debugName, &result)
                             : createShaderModuleFromGLSL(desc.stage, desc.data, desc.debugName, &result);

  if (!result.isOk()) {
    Result::setResult(outResult, result);
    return {};
  }
  Result::setResult(outResult, result);

  return {this, shaderModulesPool_.create(std::move(sm))};
}

lvk::ShaderModuleState lvk::VulkanContext::createShaderModuleFromSPIRV(const void* spirv,
                                                                       size_t numBytes,
                                                                       const char* debugName,
                                                                       Result* outResult) const {
  (void)debugName;

  if (!spirv || !numBytes) {
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "Expecting non-empty SPIR-V code"));
    return {};
  }

  SpvReflectShaderModule mdl;
  SpvReflectResult result = spvReflectCreateShaderModule(numBytes, spirv, &mdl);
  LVK_ASSERT(result == SPV_REFLECT_RESULT_SUCCESS);
  SCOPE_EXIT {
    spvReflectDestroyShaderModule(&mdl);
  };

  uint32_t pushConstantsSize = 0;

  for (uint32_t i = 0; i < mdl.push_constant_block_count; ++i) {
    const SpvReflectBlockVariable& block = mdl.push_constant_blocks[i];
    pushConstantsSize = std::max(pushConstantsSize, block.offset + block.size);
  }

  const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = numBytes,
      .pCode = (const uint32_t*)malloc(numBytes),
  };

  memcpy((void*)ci.pCode, spirv, numBytes);

  return {
      .ci = ci,
      .pushConstantsSize = pushConstantsSize,
  };
}

lvk::ShaderModuleState lvk::VulkanContext::createShaderModuleFromGLSL(ShaderStage stage,
                                                                      const char* source,
                                                                      const char* debugName,
                                                                      Result* outResult) const {
  const VkShaderStageFlagBits vkStage = shaderStageToVkShaderStage(stage);
  LVK_ASSERT(vkStage != VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM);
  LVK_ASSERT(source);

  std::string sourcePatched;

  if (!source || !*source) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Shader source is empty");
    return {};
  }

  auto addCode = [source, &sourcePatched](const char* substr, const char* code) -> void {
    if (strstr(source, substr)) {
      sourcePatched.append(code);
    }
  };

  if (strstr(source, "#version ") == nullptr) {
    if (vkStage == VK_SHADER_STAGE_TASK_BIT_EXT || vkStage == VK_SHADER_STAGE_MESH_BIT_EXT) {
      sourcePatched +=
          "#version 460\n"
          "#extension GL_EXT_buffer_reference : require\n"
          "#extension GL_EXT_buffer_reference_uvec2 : require\n"
          "#extension GL_EXT_debug_printf : enable\n"
          "#extension GL_EXT_nonuniform_qualifier : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
          "#extension GL_EXT_mesh_shader : require\n";
    }
    if (vkStage == VK_SHADER_STAGE_VERTEX_BIT || vkStage == VK_SHADER_STAGE_COMPUTE_BIT ||
        vkStage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT || vkStage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) {
      sourcePatched +=
          "#version 460\n"
          "#extension GL_EXT_buffer_reference : require\n"
          "#extension GL_EXT_buffer_reference_uvec2 : require\n"
          "#extension GL_EXT_debug_printf : enable\n"
          "#extension GL_EXT_nonuniform_qualifier : require\n"
          "#extension GL_EXT_samplerless_texture_functions : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    }
    if (vkStage == VK_SHADER_STAGE_FRAGMENT_BIT) {
      // Note how nonuniformEXT() should be used:
      // https://github.com/KhronosGroup/Vulkan-Samples/blob/main/shaders/descriptor_indexing/nonuniform-quads.frag#L33-L39
      sourcePatched +=
          "#version 460\n"
          "#extension GL_EXT_buffer_reference_uvec2 : require\n"
          "#extension GL_EXT_debug_printf : enable\n"
          "#extension GL_EXT_nonuniform_qualifier : require\n"
          "#extension GL_EXT_samplerless_texture_functions : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
      addCode("kTLAS[",
              "#extension GL_EXT_buffer_reference : require\n"
              "#extension GL_EXT_ray_query : require\n"
              "layout(set = 0, binding = 4) uniform accelerationStructureEXT kTLAS[];\n");
      sourcePatched +=
          "layout (set = 0, binding = 0) uniform texture2D   kTextures2D[];\n"
          "layout (set = 1, binding = 0) uniform texture3D   kTextures3D[];\n"
          "layout (set = 2, binding = 0) uniform textureCube kTexturesCube[];\n"
          "layout (set = 3, binding = 0) uniform texture2D   kTextures2DShadow[];\n"
          "layout (set = 0, binding = 1) uniform sampler       kSamplers[];\n"
          "layout (set = 3, binding = 1) uniform samplerShadow kSamplersShadow[];\n"
          "layout (set = 0, binding = 3) uniform sampler2D     kSamplersYUV[];\n";
      addCode("textureBindless2D(",
              "vec4 textureBindless2D(uint textureid, uint samplerid, vec2 uv) {\n"
              "  return texture(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv);\n"
              "}\n");
      addCode("textureBindless2DLod(",
              "vec4 textureBindless2DLod(uint textureid, uint samplerid, vec2 uv, float lod) {\n"
              "  return textureLod(nonuniformEXT(sampler2D(kTextures2D[textureid], kSamplers[samplerid])), uv, lod);\n"
              "}\n");
      addCode("textureBindless2DShadow(",
              "float textureBindless2DShadow(uint textureid, uint samplerid, vec3 uvw) {"
              "  return texture(nonuniformEXT(sampler2DShadow(kTextures2DShadow[textureid], kSamplersShadow[samplerid])), uvw);\n"
              "}\n");
      addCode("textureBindlessSize2D(",
              "ivec2 textureBindlessSize2D(uint textureid) {\n"
              "  return textureSize(nonuniformEXT(kTextures2D[textureid]), 0);\n"
              "}\n");
      addCode("textureBindlessCube(",
              "vec4 textureBindlessCube(uint textureid, uint samplerid, vec3 uvw) {\n"
              "  return texture(nonuniformEXT(samplerCube(kTexturesCube[textureid], kSamplers[samplerid])), uvw);\n"
              "}\n");
      addCode("textureBindlessCubeLod(",
              "vec4 textureBindlessCubeLod(uint textureid, uint samplerid, vec3 uvw, float lod) {\n"
              "  return textureLod(nonuniformEXT(samplerCube(kTexturesCube[textureid], kSamplers[samplerid])), uvw, lod);\n"
              "}\n");
      addCode("textureBindlessQueryLevels2D(",
              "int textureBindlessQueryLevels2D(uint textureid) {\n"
              "  return textureQueryLevels(nonuniformEXT(kTextures2D[textureid]));\n"
              "}\n");
      addCode("textureBindlessQueryLevelsCube(",
              "int textureBindlessQueryLevelsCube(uint textureid) {\n"
              "  return textureQueryLevels(nonuniformEXT(kTexturesCube[textureid]));\n"
              "}\n");
    }
    sourcePatched += source;
    source = sourcePatched.c_str();
  }

  const glslang_resource_t glslangResource = lvk::getGlslangResource(getVkPhysicalDeviceProperties().limits);

  std::vector<uint8_t> spirv;
  lvk::Result::setResult(outResult, lvk::compileShaderGlslang(stage, source, &spirv, &glslangResource));

  return createShaderModuleFromSPIRV(spirv.data(), spirv.size(), debugName, outResult);
}

lvk::ShaderModuleState lvk::VulkanContext::createShaderModuleFromSlang(ShaderStage stage,
                                                                       const char* source,
                                                                       const char* entryPointName,
                                                                       const char* debugName,
                                                                       Result* outResult) const {
  std::string sourcePatched;

  if (!source || !*source) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Shader source is empty");
    return {};
  }

  auto addCode = [source, &sourcePatched](const char* substr, const char* code) -> void {
    if (strstr(source, substr)) {
      sourcePatched.append(code);
    }
  };

  // overloaded operators to mimic GLSL matrix operations
  sourcePatched +=
      "float2x2 operator*(float2x2 a, float2x2 b) { return mul(b, a); }\n"
      "float2   operator*(float2x2 a, float2   b) { return mul(b, a); }\n"
      "float3x3 operator*(float3x3 a, float3x3 b) { return mul(b, a); }\n"
      "float3   operator*(float3x3 a, float3   b) { return mul(b, a); }\n"
      "float4x4 operator*(float4x4 a, float4x4 b) { return mul(b, a); }\n"
      "float4   operator*(float4x4 a, float4   b) { return mul(b, a); }\n";
  // bindless texture and sampler arrays
  sourcePatched +=
      "[[vk::binding(0, 0)]] Texture2D    kTextures2D[];\n"
      "[[vk::binding(0, 1)]] Texture3D    kTextures3D[];\n"
      "[[vk::binding(0, 2)]] TextureCube  kTexturesCube[];\n"
      "[[vk::binding(0, 3)]] Texture2D    kTextures2DShadow[];\n"
      "[[vk::binding(1, 0)]] SamplerState kSamplers[];\n"
      "[[vk::binding(1, 3)]] SamplerComparisonState kSamplersShadow[];\n"
      "[[vk::binding(3, 0)]] Sampler2D    kSamplersYUV[];\n";
  // cannot handle unbounded arrays https://github.com/shader-slang/slang/issues/8902
  addCode("kTLAS[", "[[vk::binding(4, 0)]] RaytracingAccelerationStructure kTLAS[];\n");
  addCode("textureBindless2D(",
          "float4 textureBindless2D(uint textureid, uint samplerid, float2 uv) {\n"
          "  return kTextures2D[NonUniformResourceIndex(textureid)].Sample(\n"
          "    kSamplers[NonUniformResourceIndex(samplerid)], uv);\n"
          "}\n");
  addCode("textureBindless2DLod(",
          "float4 textureBindless2DLod(uint textureid, uint samplerid, float2 uv, float lod) {\n"
          "  return kTextures2D[NonUniformResourceIndex(textureid)].SampleLevel(\n"
          "    kSamplers[NonUniformResourceIndex(samplerid)], uv, lod);\n"
          "}\n");
  addCode("textureBindlessCube(",
          "float4 textureBindlessCube(uint textureid, uint samplerid, float3 dir) {\n"
          "  return kTexturesCube[NonUniformResourceIndex(textureid)].Sample(\n"
          "    kSamplers[NonUniformResourceIndex(samplerid)], dir);\n"
          "}\n");
  addCode("textureBindlessSize2D(",
          "int2 textureBindlessSize2D(uint textureid) {\n"
          "  uint width, height;\n"
          "  kTextures2D[NonUniformResourceIndex(textureid)].GetDimensions(width, height);\n"
          "  return int2(width, height);\n"
          "}\n");
  addCode("textureBindless2DShadow(",
          "float textureBindless2DShadow(uint textureid, uint samplerid, float3 uvw) {\n"
          "  return kTextures2DShadow[NonUniformResourceIndex(textureid)].SampleCmpLevelZero(\n"
          "    kSamplersShadow[NonUniformResourceIndex(samplerid)], uvw.xy, uvw.z);\n"
          "}\n");

  sourcePatched += source;
  source = sourcePatched.c_str();

  std::vector<uint8_t> spirv;
  lvk::Result::setResult(outResult, lvk::compileShaderSlang(stage, source, entryPointName, &spirv));

  return createShaderModuleFromSPIRV(spirv.data(), spirv.size(), debugName, outResult);
}

lvk::Format lvk::VulkanContext::getSwapchainFormat() const {
  if (!hasSwapchain()) {
    return Format_Invalid;
  }

  return vkFormatToFormat(swapchain_->getSurfaceFormat().format);
}

lvk::ColorSpace lvk::VulkanContext::getSwapchainColorSpace() const {
  if (!hasSwapchain()) {
    return ColorSpace_SRGB_NONLINEAR;
  }

  return vkColorSpaceToColorSpace(swapchain_->getSurfaceFormat().colorSpace);
}

uint32_t lvk::VulkanContext::getNumSwapchainImages() const {
  return hasSwapchain() ? swapchain_->getNumSwapchainImages() : 0;
}

lvk::TextureHandle lvk::VulkanContext::getCurrentSwapchainTexture() {
  LVK_PROFILER_FUNCTION();

  if (!hasSwapchain()) {
    return {};
  }

  TextureHandle tex = swapchain_->getCurrentTexture();

  if (!LVK_VERIFY(tex.valid())) {
    LVK_ASSERT_MSG(false, "Swapchain has no valid texture");
    return {};
  }

  LVK_ASSERT_MSG(texturesPool_.get(tex)->vkImageFormat_ != VK_FORMAT_UNDEFINED, "Invalid image format");

  return tex;
}

uint32_t lvk::VulkanContext::getSwapchainCurrentImageIndex() const {
  if (hasSwapchain()) {
    // make sure we do not use a stale image
    (void)swapchain_->getCurrentTexture();
  }

  return hasSwapchain() ? swapchain_->getSwapchainCurrentImageIndex() : 0;
}

void lvk::VulkanContext::recreateSwapchain(int newWidth, int newHeight) {
  initSwapchain(newWidth, newHeight);
}

uint32_t lvk::VulkanContext::getFramebufferMSAABitMask() const {
  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;
  return limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
}

double lvk::VulkanContext::getTimestampPeriodToMs() const {
  return double(getVkPhysicalDeviceProperties().limits.timestampPeriod) * 1e-6;
}

bool lvk::VulkanContext::getQueryPoolResults(QueryPoolHandle pool,
                                             uint32_t firstQuery,
                                             uint32_t queryCount,
                                             size_t dataSize,
                                             void* outData,
                                             size_t stride) const {
  VkQueryPool vkPool = *queriesPool_.get(pool);

  VK_ASSERT(vkGetQueryPoolResults(
      vkDevice_, vkPool, firstQuery, queryCount, dataSize, outData, stride, VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_64_BIT));

  return true;
}

lvk::AccelStructSizes lvk::VulkanContext::getAccelStructSizes(const AccelStructDesc& desc, Result* outResult) const {
  LVK_PROFILER_FUNCTION();

  if (!LVK_VERIFY(has_KHR_acceleration_structure_)) {
    Result::setResult(outResult, Result(Result::Code::RuntimeError, "VK_KHR_acceleration_structure is not enabled"));
    return {};
  }

  Result result;
  VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
  VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
  switch (desc.type) {
  case AccelStructType_BLAS:
    getBuildInfoBLAS(desc, accelerationStructureGeometry, accelerationStructureBuildSizesInfo);
    break;
  case AccelStructType_TLAS:
    getBuildInfoTLAS(desc, accelerationStructureGeometry, accelerationStructureBuildSizesInfo);
    break;
  default:
    LVK_ASSERT_MSG(false, "Invalid acceleration structure type");
    Result::setResult(outResult, Result(Result::Code::ArgumentOutOfRange, "Invalid acceleration structure type"));
    return {};
  }

  Result::setResult(outResult, Result::Code::Ok);

  return lvk::AccelStructSizes{
      .accelerationStructureSize = accelerationStructureBuildSizesInfo.accelerationStructureSize,
      .updateScratchSize = accelerationStructureBuildSizesInfo.updateScratchSize,
      .buildScratchSize = accelerationStructureBuildSizesInfo.buildScratchSize,
  };
}

void lvk::VulkanContext::createInstance() {
  vkInstance_ = VK_NULL_HANDLE;

  // check if we have validation layers in the system
  {
    uint32_t numLayerProperties = 0;
    vkEnumerateInstanceLayerProperties(&numLayerProperties, nullptr);
    std::vector<VkLayerProperties> layerProperties(numLayerProperties);
    vkEnumerateInstanceLayerProperties(&numLayerProperties, layerProperties.data());

    [this, &layerProperties]() -> void {
      for (const VkLayerProperties& props : layerProperties) {
        for (const char* layer : kDefaultValidationLayers) {
          if (!strcmp(props.layerName, layer)) {
            khronosValidationVersion_ = props.specVersion;
            return;
          }
        }
      }
      config_.enableValidation = false; // no validation layers available
    }();
  }

  std::vector<VkExtensionProperties> allInstanceExtensions;
  {
    uint32_t count = 0;
    VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
    allInstanceExtensions.resize(count);
    VK_ASSERT(vkEnumerateInstanceExtensionProperties(nullptr, &count, allInstanceExtensions.data()));
  }
  // collect instance extensions from all validation layers
  if (config_.enableValidation) {
    for (const char* layer : kDefaultValidationLayers) {
      uint32_t count = 0;
      VK_ASSERT(vkEnumerateInstanceExtensionProperties(layer, &count, nullptr));
      if (count > 0) {
        const size_t sz = allInstanceExtensions.size();
        allInstanceExtensions.resize(sz + count);
        VK_ASSERT(vkEnumerateInstanceExtensionProperties(layer, &count, allInstanceExtensions.data() + sz));
      }
    }
  }

  enabledInstanceExtensionNames_ = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME,
#if defined(_WIN32)
      VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
      VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
#elif defined(__linux__)
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
      VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#else
      VK_KHR_XLIB_SURFACE_EXTENSION_NAME,
#endif
#elif defined(__APPLE__)
      VK_EXT_LAYER_SETTINGS_EXTENSION_NAME,
      VK_MVK_MACOS_SURFACE_EXTENSION_NAME,
#endif
#if defined(LVK_WITH_VULKAN_PORTABILITY)
      VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
#endif
  };

  // check if we have the VK_EXT_debug_utils extension
  const bool hasDebugUtils = hasExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, allInstanceExtensions);

  if (hasDebugUtils) {
    enabledInstanceExtensionNames_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  if (config_.enableValidation) {
    enabledInstanceExtensionNames_.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME); // enabled only for validation
  }

  if (config_.enableHeadlessSurface) {
    enabledInstanceExtensionNames_.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
  }

  if (hasExtension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, allInstanceExtensions)) {
    has_EXT_swapchain_colorspace_ = true;
    enabledInstanceExtensionNames_.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
  }

  if (hasExtension(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, allInstanceExtensions)) {
    // required by the instance extension VK_EXT_surface_maintenance1
    enabledInstanceExtensionNames_.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
  }

  for (const char* ext : config_.extensionsInstance) {
    if (ext) {
      enabledInstanceExtensionNames_.push_back(ext);
    }
  }

#if !defined(ANDROID)
  // GPU Assisted Validation doesn't work on Android.
  // It implicitly requires vertexPipelineStoresAndAtomics feature that's not supported even on high-end devices.
  const VkValidationFeatureEnableEXT validationFeaturesEnabled[] = {
      VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
      VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
  };
#endif // ANDROID

#if defined(__APPLE__)
  // Shader validation doesn't work in MoltenVK for SPIR-V 1.6 under Vulkan 1.3:
  // "Invalid SPIR-V binary version 1.6 for target environment SPIR-V 1.5 (under Vulkan 1.2 semantics)."
  const VkValidationFeatureDisableEXT validationFeaturesDisabled[] = {
      VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT,
      VK_VALIDATION_FEATURE_DISABLE_SHADER_VALIDATION_CACHE_EXT,
  };
#endif // __APPLE__

  const VkValidationFeaturesEXT features = {
      .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
      .pNext = nullptr,
#if !defined(ANDROID)
      .enabledValidationFeatureCount = config_.enableValidation ? (uint32_t)LVK_ARRAY_NUM_ELEMENTS(validationFeaturesEnabled) : 0u,
      .pEnabledValidationFeatures = config_.enableValidation ? validationFeaturesEnabled : nullptr,
#endif
#if defined(__APPLE__)
      .disabledValidationFeatureCount = config_.enableValidation ? (uint32_t)LVK_ARRAY_NUM_ELEMENTS(validationFeaturesDisabled) : 0u,
      .pDisabledValidationFeatures = config_.enableValidation ? validationFeaturesDisabled : nullptr,
#endif
  };

#if defined(VK_EXT_layer_settings) && VK_EXT_layer_settings
  // https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Configuration_Parameters.md
  const int useMetalArgumentBuffers = 1;
  const VkBool32 gpuav_descriptor_checks = VK_FALSE; // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8688
  const VkBool32 gpuav_indirect_draws_buffers = VK_FALSE; // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/8579
  const VkBool32 gpuav_post_process_descriptor_indexing = VK_FALSE; // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/9222
#define LAYER_SETTINGS_BOOL32(name, var)                                                                                        \
  VkLayerSettingEXT {                                                                                                           \
    .pLayerName = kDefaultValidationLayers[0], .pSettingName = name, .type = VK_LAYER_SETTING_TYPE_BOOL32_EXT, .valueCount = 1, \
    .pValues = var,                                                                                                             \
  }
  const VkLayerSettingEXT settings[] = {
      LAYER_SETTINGS_BOOL32("gpuav_descriptor_checks", &gpuav_descriptor_checks),
      LAYER_SETTINGS_BOOL32("gpuav_indirect_draws_buffers", &gpuav_indirect_draws_buffers),
      LAYER_SETTINGS_BOOL32("gpuav_post_process_descriptor_indexing", &gpuav_post_process_descriptor_indexing),
      {"MoltenVK", "MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS", VK_LAYER_SETTING_TYPE_INT32_EXT, 1, &useMetalArgumentBuffers},
  };
#undef LAYER_SETTINGS_BOOL32
  const VkLayerSettingsCreateInfoEXT layerSettingsCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT,
      .pNext = config_.enableValidation ? &features : nullptr,
      .settingCount = (uint32_t)LVK_ARRAY_NUM_ELEMENTS(settings),
      .pSettings = settings,
  };
#endif // defined(VK_EXT_layer_settings) && VK_EXT_layer_settings

  const VkApplicationInfo appInfo = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = nullptr,
      .pApplicationName = "LVK/Vulkan",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "LVK/Vulkan",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
#if defined(VK_API_VERSION_1_4)
      .apiVersion = config_.vulkanVersion == VulkanVersion_1_3 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_4,
#else
      .apiVersion = VK_API_VERSION_1_3,
#endif // VK_API_VERSION_1_4
  };

  VkInstanceCreateFlags flags = 0;
#if defined(LVK_WITH_VULKAN_PORTABILITY)
  flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
  const VkInstanceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if defined(VK_EXT_layer_settings) && VK_EXT_layer_settings
      .pNext = &layerSettingsCreateInfo,
#else
      .pNext = config_.enableValidation ? &features : nullptr,
#endif // defined(VK_EXT_layer_settings) && VK_EXT_layer_settings
      .flags = flags,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = config_.enableValidation ? (uint32_t)LVK_ARRAY_NUM_ELEMENTS(kDefaultValidationLayers) : 0u,
      .ppEnabledLayerNames = config_.enableValidation ? kDefaultValidationLayers : nullptr,
      .enabledExtensionCount = (uint32_t)enabledInstanceExtensionNames_.size(),
      .ppEnabledExtensionNames = enabledInstanceExtensionNames_.data(),
  };
  VK_ASSERT(vkCreateInstance(&ci, nullptr, &vkInstance_));

  volkLoadInstance(vkInstance_);

  // debug messenger
  if (hasDebugUtils) {
    const VkDebugUtilsMessengerCreateInfoEXT ci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = &vulkanDebugCallback,
        .pUserData = this,
    };
    VK_ASSERT(vkCreateDebugUtilsMessengerEXT(vkInstance_, &ci, nullptr, &vkDebugUtilsMessenger_));
  }

  LLOGL("%s layer version: %u.%u.%u\n",
        kDefaultValidationLayers[0],
        VK_VERSION_MAJOR(khronosValidationVersion_),
        VK_VERSION_MINOR(khronosValidationVersion_),
        VK_VERSION_PATCH(khronosValidationVersion_));

  // log available instance extensions
  LLOGL("\nVulkan instance extensions:\n");

  for (const VkExtensionProperties& extension : allInstanceExtensions) {
    LLOGL("  %s\n", extension.extensionName);
  }
}

void lvk::VulkanContext::createHeadlessSurface() {
  const VkHeadlessSurfaceCreateInfoEXT ci = {
      .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
      .pNext = nullptr,
      .flags = 0,
  };
  VK_ASSERT(vkCreateHeadlessSurfaceEXT(vkInstance_, &ci, nullptr, &vkSurface_));
}

void lvk::VulkanContext::createSurface(void* window, void* display) {
#if defined(VK_USE_PLATFORM_WIN32_KHR)
  const VkWin32SurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
      .hinstance = GetModuleHandle(nullptr),
      .hwnd = (HWND)window,
  };
  VK_ASSERT(vkCreateWin32SurfaceKHR(vkInstance_, &ci, nullptr, &vkSurface_));
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  const VkAndroidSurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .pNext = nullptr, .flags = 0, .window = (ANativeWindow*)window};
  VK_ASSERT(vkCreateAndroidSurfaceKHR(vkInstance_, &ci, nullptr, &vkSurface_));
#elif defined(VK_USE_PLATFORM_XLIB_KHR)
  const VkXlibSurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
      .flags = 0,
      .dpy = (Display*)display,
      .window = (Window)window,
  };
  VK_ASSERT(vkCreateXlibSurfaceKHR(vkInstance_, &ci, nullptr, &vkSurface_));
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  const VkWaylandSurfaceCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
      .flags = 0,
      .display = (wl_display*)display,
      .surface = (wl_surface*)window,
  };
  VK_ASSERT(vkCreateWaylandSurfaceKHR(vkInstance_, &ci, nullptr, &vkSurface_));
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
  const VkMacOSSurfaceCreateInfoMVK ci = {
      .sType = VK_STRUCTURE_TYPE_MACOS_SURFACE_CREATE_INFO_MVK,
      .flags = 0,
      .pView = window,
  };
  VK_ASSERT(vkCreateMacOSSurfaceMVK(vkInstance_, &ci, nullptr, &vkSurface_));
#else
#error Implement for other platforms
#endif
}

uint32_t lvk::VulkanContext::queryDevices(HWDeviceDesc* outDevices, uint32_t maxOutDevices) {
  // Physical devices
  uint32_t deviceCount = 0;
  VK_ASSERT(vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr));
  std::vector<VkPhysicalDevice> vkDevices(deviceCount);
  VK_ASSERT(vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, vkDevices.data()));

  auto convertVulkanDeviceTypeToLVK = [](VkPhysicalDeviceType vkDeviceType) -> HWDeviceType {
    switch (vkDeviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return HWDeviceType_Integrated;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return HWDeviceType_Discrete;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return HWDeviceType_External;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return HWDeviceType_Software;
    default:
      return HWDeviceType_Software;
    }
  };

  uint32_t numCompatibleDevices = 0;

  for (uint32_t i = 0; i < deviceCount; ++i) {
    VkPhysicalDevice physicalDevice = vkDevices[i];
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    const HWDeviceType deviceType = convertVulkanDeviceTypeToLVK(deviceProperties.deviceType);

    if (outDevices && numCompatibleDevices < maxOutDevices) {
      outDevices[numCompatibleDevices] = {.guid = (uintptr_t)vkDevices[i], .type = deviceType};
      strncpy(outDevices[numCompatibleDevices].name, deviceProperties.deviceName, strlen(deviceProperties.deviceName));
      numCompatibleDevices++;
    }
  }

  return numCompatibleDevices;
}

void lvk::VulkanContext::addNextPhysicalDeviceProperties(void* properties) {
  if (!properties)
    return;

  std::launder(reinterpret_cast<VkBaseOutStructure*>(properties))->pNext =
      std::launder(reinterpret_cast<VkBaseOutStructure*>(vkPhysicalDeviceProperties2_.pNext));

  vkPhysicalDeviceProperties2_.pNext = properties;
}

void lvk::VulkanContext::getBuildInfoBLAS(const AccelStructDesc& desc,
                                          VkAccelerationStructureGeometryKHR& outGeometry,
                                          VkAccelerationStructureBuildSizesInfoKHR& outSizesInfo) const {
  LVK_ASSERT(desc.type == AccelStructType_BLAS);
  LVK_ASSERT(desc.geometryType == AccelStructGeomType_Triangles);
  LVK_ASSERT(desc.numVertices);
  LVK_ASSERT(desc.indexBuffer.valid());
  LVK_ASSERT(desc.vertexBuffer.valid());
  LVK_ASSERT(desc.transformBuffer.valid());
  LVK_ASSERT(desc.buildRange.primitiveCount);

  LVK_ASSERT(buffersPool_.get(desc.indexBuffer)->vkUsageFlags_ & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  LVK_ASSERT(buffersPool_.get(desc.vertexBuffer)->vkUsageFlags_ & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  LVK_ASSERT(buffersPool_.get(desc.transformBuffer)->vkUsageFlags_ & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  VkGeometryFlagsKHR geometryFlags = 0;

  if (desc.geometryFlags & AccelStructGeometryFlagBits_Opaque) {
    geometryFlags |= VK_GEOMETRY_OPAQUE_BIT_KHR;
  }
  if (desc.geometryFlags & AccelStructGeometryFlagBits_NoDuplicateAnyHit) {
    geometryFlags |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  }

  outGeometry = VkAccelerationStructureGeometryKHR{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
      .geometry =
          {
              .triangles =
                  {
                      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                      .vertexFormat = vertexFormatToVkFormat(desc.vertexFormat),
                      .vertexData = {.deviceAddress = gpuAddress(desc.vertexBuffer)},
                      .vertexStride = desc.vertexStride ? desc.vertexStride : lvk::getVertexFormatSize(desc.vertexFormat),
                      .maxVertex = desc.numVertices - 1,
                      .indexType = VK_INDEX_TYPE_UINT32,
                      .indexData = {.deviceAddress = gpuAddress(desc.indexBuffer)},
                      .transformData = {.deviceAddress = gpuAddress(desc.transformBuffer)},
                  },
          },
      .flags = geometryFlags,
  };

  const VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
      .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      .geometryCount = 1,
      .pGeometries = &outGeometry,
  };

  outSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR(vkDevice_,
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &accelerationBuildGeometryInfo,
                                          &desc.buildRange.primitiveCount,
                                          &outSizesInfo);
  const uint32_t alignment = accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment;
  outSizesInfo.accelerationStructureSize += alignment;
  outSizesInfo.updateScratchSize += alignment;
  outSizesInfo.buildScratchSize += alignment;
}

void lvk::VulkanContext::getBuildInfoTLAS(const AccelStructDesc& desc,
                                          VkAccelerationStructureGeometryKHR& outGeometry,
                                          VkAccelerationStructureBuildSizesInfoKHR& outSizesInfo) const {
  LVK_ASSERT(desc.type == AccelStructType_TLAS);
  LVK_ASSERT(desc.geometryType == AccelStructGeomType_Instances);
  LVK_ASSERT(desc.numVertices == 0);
  LVK_ASSERT(desc.instancesBuffer.valid());
  LVK_ASSERT(desc.buildRange.primitiveCount);
  LVK_ASSERT(buffersPool_.get(desc.instancesBuffer)->vkUsageFlags_ & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

  VkGeometryFlagsKHR geometryFlags = 0;

  if (desc.geometryFlags & AccelStructGeometryFlagBits_Opaque) {
    geometryFlags |= VK_GEOMETRY_OPAQUE_BIT_KHR;
  }
  if (desc.geometryFlags & AccelStructGeometryFlagBits_NoDuplicateAnyHit) {
    geometryFlags |= VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  }

  outGeometry = VkAccelerationStructureGeometryKHR{
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
      .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
      .geometry =
          {
              .instances =
                  {
                      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                      .arrayOfPointers = VK_FALSE,
                      .data = {.deviceAddress = gpuAddress(desc.instancesBuffer)},
                  },
          },
      .flags = geometryFlags,
  };

  const VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
      .flags = buildFlagsToVkBuildAccelerationStructureFlags(desc.buildFlags),
      .geometryCount = 1,
      .pGeometries = &outGeometry,
  };

  outSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
  vkGetAccelerationStructureBuildSizesKHR(vkDevice_,
                                          VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &accelerationStructureBuildGeometryInfo,
                                          &desc.buildRange.primitiveCount,
                                          &outSizesInfo);
  const uint32_t alignment = accelerationStructureProperties_.minAccelerationStructureScratchOffsetAlignment;
  outSizesInfo.accelerationStructureSize += alignment;
  outSizesInfo.updateScratchSize += alignment;
  outSizesInfo.buildScratchSize += alignment;
}

lvk::Result lvk::VulkanContext::initContext(const HWDeviceDesc& desc) {
  if (desc.guid == 0UL) {
    LLOGW("Invalid hardwareGuid(%lu)", desc.guid);
    return Result(Result::Code::RuntimeError, "Vulkan is not supported");
  }

  vkPhysicalDevice_ = (VkPhysicalDevice)desc.guid;

  useStaging_ = !isHostVisibleSingleHeapMemory(vkPhysicalDevice_);

  std::vector<VkExtensionProperties> allDeviceExtensions;
  getDeviceExtensionProps(vkPhysicalDevice_, allDeviceExtensions);
  if (config_.enableValidation) {
    for (const char* layer : kDefaultValidationLayers) {
      getDeviceExtensionProps(vkPhysicalDevice_, allDeviceExtensions, layer);
    }
  }

  if (hasExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, allDeviceExtensions)) {
    addNextPhysicalDeviceProperties(&accelerationStructureProperties_);
  }
  if (hasExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, allDeviceExtensions)) {
    addNextPhysicalDeviceProperties(&rayTracingPipelineProperties_);
  }

#if defined(VK_API_VERSION_1_4)
  if (config_.vulkanVersion >= VulkanVersion_1_4) {
    addNextPhysicalDeviceProperties(&vkPhysicalDeviceVulkan14Properties_);
    vkFeatures13_.pNext = &vkFeatures14_;
  }
#else
  LVK_ASSERT_MSG(config_.vulkanVersion == VulkanVersion_1_3, "Only Vulkan 1.3 is supported on this platform");
#endif // VK_API_VERSION_1_4

  vkGetPhysicalDeviceFeatures2(vkPhysicalDevice_, &vkFeatures10_);
  vkGetPhysicalDeviceProperties2(vkPhysicalDevice_, &vkPhysicalDeviceProperties2_);

  const uint32_t apiVersion = vkPhysicalDeviceProperties2_.properties.apiVersion;

  LLOGL("Vulkan physical device: %s\n", vkPhysicalDeviceProperties2_.properties.deviceName);
  LLOGL("           API version: %i.%i.%i.%i\n",
        VK_API_VERSION_MAJOR(apiVersion),
        VK_API_VERSION_MINOR(apiVersion),
        VK_API_VERSION_PATCH(apiVersion),
        VK_API_VERSION_VARIANT(apiVersion));
  LLOGL("           Driver info: %s %s\n", vkPhysicalDeviceDriverProperties_.driverName, vkPhysicalDeviceDriverProperties_.driverInfo);

  LLOGL("Vulkan physical device extensions:\n");

  // log available physical device extensions
  for (const VkExtensionProperties& ext : allDeviceExtensions) {
    LLOGL("  %s\n", ext.extensionName);
  }

  deviceQueues_.graphicsQueueFamilyIndex = lvk::findQueueFamilyIndex(vkPhysicalDevice_, VK_QUEUE_GRAPHICS_BIT);
  deviceQueues_.computeQueueFamilyIndex = lvk::findQueueFamilyIndex(vkPhysicalDevice_, VK_QUEUE_COMPUTE_BIT);

  if (deviceQueues_.graphicsQueueFamilyIndex == DeviceQueues::INVALID) {
    LLOGW("VK_QUEUE_GRAPHICS_BIT is not supported");
    return Result(Result::Code::RuntimeError, "VK_QUEUE_GRAPHICS_BIT is not supported");
  }

  if (deviceQueues_.computeQueueFamilyIndex == DeviceQueues::INVALID) {
    LLOGW("VK_QUEUE_COMPUTE_BIT is not supported");
    return Result(Result::Code::RuntimeError, "VK_QUEUE_COMPUTE_BIT is not supported");
  }

  const float queuePriority = 1.0f;

  const VkDeviceQueueCreateInfo ciQueue[2] = {
      {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = deviceQueues_.graphicsQueueFamilyIndex,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      },
      {
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = deviceQueues_.computeQueueFamilyIndex,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      },
  };
  const uint32_t numQueues = ciQueue[0].queueFamilyIndex == ciQueue[1].queueFamilyIndex ? 1 : 2;

  enabledDeviceExtensionNames_ = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if defined(LVK_WITH_VULKAN_PORTABILITY)
      "VK_KHR_portability_subset",
#endif
  };

  for (const char* ext : config_.extensionsDevice) {
    if (ext) {
      enabledDeviceExtensionNames_.push_back(ext);
    }
  }

  VkPhysicalDeviceFeatures deviceFeatures10 = {
      .independentBlend = vkFeatures10_.features.independentBlend, // enable if supported
      .geometryShader = vkFeatures10_.features.geometryShader, // enable if supported
      .tessellationShader = vkFeatures10_.features.tessellationShader, // enable if supported
      .sampleRateShading = VK_TRUE,
      .multiDrawIndirect = VK_TRUE,
      .drawIndirectFirstInstance = VK_TRUE,
      .depthBiasClamp = VK_TRUE,
      .fillModeNonSolid = vkFeatures10_.features.fillModeNonSolid, // enable if supported
      .samplerAnisotropy = VK_TRUE,
      .textureCompressionBC = vkFeatures10_.features.textureCompressionBC, // enable if supported
      .vertexPipelineStoresAndAtomics = vkFeatures10_.features.vertexPipelineStoresAndAtomics, // enable if supported
      .fragmentStoresAndAtomics = VK_TRUE,
      .shaderImageGatherExtended = VK_TRUE,
      .shaderInt64 = vkFeatures10_.features.shaderInt64, // enable if supported
      .shaderInt16 = vkFeatures10_.features.shaderInt16, // enable if supported
  };
  VkPhysicalDeviceVulkan11Features deviceFeatures11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = config_.extensionsDeviceFeatures,
      .storageBuffer16BitAccess = VK_TRUE,
      .multiview = vkFeatures11_.multiview, // enable if supported
      .samplerYcbcrConversion = vkFeatures11_.samplerYcbcrConversion, // enable if supported
      .shaderDrawParameters = VK_TRUE,
  };
  VkPhysicalDeviceVulkan12Features deviceFeatures12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &deviceFeatures11,
      .samplerMirrorClampToEdge = VK_TRUE,
      .drawIndirectCount = vkFeatures12_.drawIndirectCount, // enable if supported
      .storageBuffer8BitAccess = vkFeatures12_.storageBuffer8BitAccess, // enable if supported
      .uniformAndStorageBuffer8BitAccess = vkFeatures12_.uniformAndStorageBuffer8BitAccess, // enable if supported
      .shaderFloat16 = vkFeatures12_.shaderFloat16, // enable if supported
      .shaderInt8 = vkFeatures12_.shaderInt8, // enable if supported
      .descriptorIndexing = VK_TRUE,
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
      .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingStorageImageUpdateAfterBind = VK_TRUE,
      .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
      .descriptorBindingPartiallyBound = VK_TRUE,
      .descriptorBindingVariableDescriptorCount = VK_TRUE,
      .runtimeDescriptorArray = VK_TRUE,
      .scalarBlockLayout = VK_TRUE,
      .uniformBufferStandardLayout = VK_TRUE,
      .hostQueryReset = vkFeatures12_.hostQueryReset, // enable if supported
      .timelineSemaphore = VK_TRUE,
      .bufferDeviceAddress = VK_TRUE,
      .vulkanMemoryModel = vkFeatures12_.vulkanMemoryModel, // enable if supported
      .vulkanMemoryModelDeviceScope = vkFeatures12_.vulkanMemoryModelDeviceScope, // enable if supported
  };
  VkPhysicalDeviceVulkan13Features deviceFeatures13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &deviceFeatures12,
      .shaderDemoteToHelperInvocation = vkFeatures13_.shaderDemoteToHelperInvocation, // enable if supported
      .subgroupSizeControl = VK_TRUE,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
      .maintenance4 = VK_TRUE,
  };
#if defined(VK_API_VERSION_1_4)
  VkPhysicalDeviceVulkan14Features deviceFeatures14 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
      .pNext = &deviceFeatures13,
      .indexTypeUint8 = vkFeatures14_.indexTypeUint8,
      .dynamicRenderingLocalRead = VK_TRUE,
      .maintenance5 = VK_TRUE,
  };
#endif // VK_API_VERSION_1_4

#if defined(VK_API_VERSION_1_4)
  void* createInfoNext = config_.vulkanVersion >= VulkanVersion_1_4 ? (void*)&deviceFeatures14 : (void*)&deviceFeatures13;
#else
  void* createInfoNext = &deviceFeatures13;
#endif
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .accelerationStructure = VK_TRUE,
      .accelerationStructureCaptureReplay = VK_FALSE,
      .accelerationStructureIndirectBuild = VK_FALSE,
      .accelerationStructureHostCommands = VK_FALSE,
      .descriptorBindingAccelerationStructureUpdateAfterBind = VK_TRUE,
  };
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
      .rayTracingPipeline = VK_TRUE,
      .rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE,
      .rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE,
      .rayTracingPipelineTraceRaysIndirect = VK_TRUE,
      .rayTraversalPrimitiveCulling = VK_FALSE,
  };
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
      .rayQuery = VK_TRUE,
  };
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT rayTracingInvocationReorderFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT,
      .rayTracingInvocationReorder = VK_TRUE,
  };
  VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT swapchainMaintenance1Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT,
      .swapchainMaintenance1 = VK_TRUE,
  };
  VkPhysicalDeviceIndexTypeUint8FeaturesEXT indexTypeUint8Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT,
      .indexTypeUint8 = VK_TRUE,
  };
  VkPhysicalDeviceFaultFeaturesEXT deviceFaultFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT,
      .deviceFault = VK_TRUE,
  };
  VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5Features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .maintenance5 = VK_TRUE,
  };
  VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR dynamicRenderingLocalReadFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR,
      .dynamicRenderingLocalRead = VK_TRUE,
  };
  VkPhysicalDeviceShaderTileImageFeaturesEXT shaderTileImageFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TILE_IMAGE_FEATURES_EXT,
      .shaderTileImageColorReadAccess = VK_TRUE,
      .shaderTileImageDepthReadAccess = VK_TRUE,
  };

  auto addExtension = [&allDeviceExtensions, this, &createInfoNext](const char* name, void* features = nullptr) mutable -> void {
    if (!hasExtension(name, allDeviceExtensions)) {
      LLOGW("Unsupported mandatory extension `%s`\n", name);
      return;
    }
    enabledDeviceExtensionNames_.push_back(name);
    if (features) {
      std::launder(reinterpret_cast<VkBaseOutStructure*>(features))->pNext =
          std::launder(reinterpret_cast<VkBaseOutStructure*>(createInfoNext));
      createInfoNext = features;
    }
  };

  auto addOptionalExtension = [&allDeviceExtensions, this, &createInfoNext](
                                  const char* name, bool& enabled, void* features = nullptr) mutable -> bool {
    if (!hasExtension(name, allDeviceExtensions))
      return false;
    enabled = true;
    enabledDeviceExtensionNames_.push_back(name);
    if (features) {
      std::launder(reinterpret_cast<VkBaseOutStructure*>(features))->pNext =
          std::launder(reinterpret_cast<VkBaseOutStructure*>(createInfoNext));
      createInfoNext = features;
    }
    return true;
  };
  auto addOptionalExtensions = [&allDeviceExtensions, this, &createInfoNext](
                                   const char* name1, const char* name2, bool& enabled, void* features = nullptr) mutable {
    if (!hasExtension(name1, allDeviceExtensions) || !hasExtension(name2, allDeviceExtensions))
      return;
    enabled = true;
    enabledDeviceExtensionNames_.push_back(name1);
    enabledDeviceExtensionNames_.push_back(name2);
    if (features) {
      std::launder(reinterpret_cast<VkBaseOutStructure*>(features))->pNext =
          std::launder(reinterpret_cast<VkBaseOutStructure*>(createInfoNext));
      createInfoNext = features;
    }
  };

  if (config_.vulkanVersion < VulkanVersion_1_4) {
    addExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, nullptr);
    addExtension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME, &maintenance5Features);
    addExtension(VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME, &dynamicRenderingLocalReadFeatures);
    if (!addOptionalExtension(VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME, has_8BitIndices_, &indexTypeUint8Features)) {
      addOptionalExtension(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, has_8BitIndices_, &indexTypeUint8Features);
    }
  }
#if defined(LVK_WITH_TRACY)
  addOptionalExtension(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, has_EXT_calibrated_timestamps_, nullptr);
#endif
  addOptionalExtensions(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                        has_KHR_acceleration_structure_,
                        &accelerationStructureFeatures);
  addOptionalExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, has_KHR_ray_query_, &rayQueryFeatures);
  addOptionalExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, has_KHR_ray_tracing_pipeline_, &rayTracingFeatures);
  addOptionalExtension(
      VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, has_EXT_ray_tracing_invocation_reorder, &rayTracingInvocationReorderFeatures);
  addOptionalExtension(VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME, has_EXT_swapchain_maintenance1_, &swapchainMaintenance1Features);
  addOptionalExtension(VK_EXT_HDR_METADATA_EXTENSION_NAME, has_EXT_hdr_metadata_);
  addOptionalExtension(VK_EXT_DEVICE_FAULT_EXTENSION_NAME, has_EXT_device_fault_, &deviceFaultFeatures);
  addOptionalExtension(VK_EXT_SHADER_TILE_IMAGE_EXTENSION_NAME, has_EXT_shader_tile_image, &shaderTileImageFeatures);

  // check extensions
  {
    std::string missingExtensions;
    for (const char* ext : enabledDeviceExtensionNames_) {
      if (!hasExtension(ext, allDeviceExtensions))
        missingExtensions += "\n   " + std::string(ext);
    }
    if (!missingExtensions.empty()) {
      MINILOG_LOG_PROC(minilog::FatalError, "Missing Vulkan device extensions: %s\n", missingExtensions.c_str());
      assert(false);
      return Result(Result::Code::RuntimeError);
    }
  }

  // check features
  {
    std::string missingFeatures;
#define CHECK_VULKAN_FEATURE(reqFeatures, availFeatures, feature, version)     \
  if ((reqFeatures.feature == VK_TRUE) && (availFeatures.feature == VK_FALSE)) \
    missingFeatures.append("\n   " version " ." #feature);
#define CHECK_FEATURE_1_0(feature) CHECK_VULKAN_FEATURE(deviceFeatures10, vkFeatures10_.features, feature, "1.0 ");
    CHECK_FEATURE_1_0(robustBufferAccess);
    CHECK_FEATURE_1_0(fullDrawIndexUint32);
    CHECK_FEATURE_1_0(imageCubeArray);
    CHECK_FEATURE_1_0(independentBlend);
    CHECK_FEATURE_1_0(geometryShader);
    CHECK_FEATURE_1_0(tessellationShader);
    CHECK_FEATURE_1_0(sampleRateShading);
    CHECK_FEATURE_1_0(dualSrcBlend);
    CHECK_FEATURE_1_0(logicOp);
    CHECK_FEATURE_1_0(multiDrawIndirect);
    CHECK_FEATURE_1_0(drawIndirectFirstInstance);
    CHECK_FEATURE_1_0(depthClamp);
    CHECK_FEATURE_1_0(depthBiasClamp);
    CHECK_FEATURE_1_0(fillModeNonSolid);
    CHECK_FEATURE_1_0(depthBounds);
    CHECK_FEATURE_1_0(wideLines);
    CHECK_FEATURE_1_0(largePoints);
    CHECK_FEATURE_1_0(alphaToOne);
    CHECK_FEATURE_1_0(multiViewport);
    CHECK_FEATURE_1_0(samplerAnisotropy);
    CHECK_FEATURE_1_0(textureCompressionETC2);
    CHECK_FEATURE_1_0(textureCompressionASTC_LDR);
    CHECK_FEATURE_1_0(textureCompressionBC);
    CHECK_FEATURE_1_0(occlusionQueryPrecise);
    CHECK_FEATURE_1_0(pipelineStatisticsQuery);
    CHECK_FEATURE_1_0(vertexPipelineStoresAndAtomics);
    CHECK_FEATURE_1_0(fragmentStoresAndAtomics);
    CHECK_FEATURE_1_0(shaderTessellationAndGeometryPointSize);
    CHECK_FEATURE_1_0(shaderImageGatherExtended);
    CHECK_FEATURE_1_0(shaderStorageImageExtendedFormats);
    CHECK_FEATURE_1_0(shaderStorageImageMultisample);
    CHECK_FEATURE_1_0(shaderStorageImageReadWithoutFormat);
    CHECK_FEATURE_1_0(shaderStorageImageWriteWithoutFormat);
    CHECK_FEATURE_1_0(shaderUniformBufferArrayDynamicIndexing);
    CHECK_FEATURE_1_0(shaderSampledImageArrayDynamicIndexing);
    CHECK_FEATURE_1_0(shaderStorageBufferArrayDynamicIndexing);
    CHECK_FEATURE_1_0(shaderStorageImageArrayDynamicIndexing);
    CHECK_FEATURE_1_0(shaderClipDistance);
    CHECK_FEATURE_1_0(shaderCullDistance);
    CHECK_FEATURE_1_0(shaderFloat64);
    CHECK_FEATURE_1_0(shaderInt64);
    CHECK_FEATURE_1_0(shaderInt16);
    CHECK_FEATURE_1_0(shaderResourceResidency);
    CHECK_FEATURE_1_0(shaderResourceMinLod);
    CHECK_FEATURE_1_0(sparseBinding);
    CHECK_FEATURE_1_0(sparseResidencyBuffer);
    CHECK_FEATURE_1_0(sparseResidencyImage2D);
    CHECK_FEATURE_1_0(sparseResidencyImage3D);
    CHECK_FEATURE_1_0(sparseResidency2Samples);
    CHECK_FEATURE_1_0(sparseResidency4Samples);
    CHECK_FEATURE_1_0(sparseResidency8Samples);
    CHECK_FEATURE_1_0(sparseResidency16Samples);
    CHECK_FEATURE_1_0(sparseResidencyAliased);
    CHECK_FEATURE_1_0(variableMultisampleRate);
    CHECK_FEATURE_1_0(inheritedQueries);
#undef CHECK_FEATURE_1_0
#define CHECK_FEATURE_1_1(feature) CHECK_VULKAN_FEATURE(deviceFeatures11, vkFeatures11_, feature, "1.1 ");
    CHECK_FEATURE_1_1(storageBuffer16BitAccess);
    CHECK_FEATURE_1_1(uniformAndStorageBuffer16BitAccess);
    CHECK_FEATURE_1_1(storagePushConstant16);
    CHECK_FEATURE_1_1(storageInputOutput16);
    CHECK_FEATURE_1_1(multiview);
    CHECK_FEATURE_1_1(multiviewGeometryShader);
    CHECK_FEATURE_1_1(multiviewTessellationShader);
    CHECK_FEATURE_1_1(variablePointersStorageBuffer);
    CHECK_FEATURE_1_1(variablePointers);
    CHECK_FEATURE_1_1(protectedMemory);
    CHECK_FEATURE_1_1(samplerYcbcrConversion);
    CHECK_FEATURE_1_1(shaderDrawParameters);
#undef CHECK_FEATURE_1_1
#define CHECK_FEATURE_1_2(feature) CHECK_VULKAN_FEATURE(deviceFeatures12, vkFeatures12_, feature, "1.2 ");
    CHECK_FEATURE_1_2(samplerMirrorClampToEdge);
    CHECK_FEATURE_1_2(drawIndirectCount);
    CHECK_FEATURE_1_2(storageBuffer8BitAccess);
    CHECK_FEATURE_1_2(uniformAndStorageBuffer8BitAccess);
    CHECK_FEATURE_1_2(storagePushConstant8);
    CHECK_FEATURE_1_2(shaderBufferInt64Atomics);
    CHECK_FEATURE_1_2(shaderSharedInt64Atomics);
    CHECK_FEATURE_1_2(shaderFloat16);
    CHECK_FEATURE_1_2(shaderInt8);
    CHECK_FEATURE_1_2(descriptorIndexing);
    CHECK_FEATURE_1_2(shaderInputAttachmentArrayDynamicIndexing);
    CHECK_FEATURE_1_2(shaderUniformTexelBufferArrayDynamicIndexing);
    CHECK_FEATURE_1_2(shaderStorageTexelBufferArrayDynamicIndexing);
    CHECK_FEATURE_1_2(shaderUniformBufferArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderSampledImageArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderStorageBufferArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderStorageImageArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderInputAttachmentArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderUniformTexelBufferArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(shaderStorageTexelBufferArrayNonUniformIndexing);
    CHECK_FEATURE_1_2(descriptorBindingUniformBufferUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingSampledImageUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingStorageImageUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingStorageBufferUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingUniformTexelBufferUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingStorageTexelBufferUpdateAfterBind);
    CHECK_FEATURE_1_2(descriptorBindingUpdateUnusedWhilePending);
    CHECK_FEATURE_1_2(descriptorBindingPartiallyBound);
    CHECK_FEATURE_1_2(descriptorBindingVariableDescriptorCount);
    CHECK_FEATURE_1_2(runtimeDescriptorArray);
    CHECK_FEATURE_1_2(samplerFilterMinmax);
    CHECK_FEATURE_1_2(scalarBlockLayout);
    CHECK_FEATURE_1_2(imagelessFramebuffer);
    CHECK_FEATURE_1_2(uniformBufferStandardLayout);
    CHECK_FEATURE_1_2(shaderSubgroupExtendedTypes);
    CHECK_FEATURE_1_2(separateDepthStencilLayouts);
    CHECK_FEATURE_1_2(hostQueryReset);
    CHECK_FEATURE_1_2(timelineSemaphore);
    CHECK_FEATURE_1_2(bufferDeviceAddress);
    CHECK_FEATURE_1_2(bufferDeviceAddressCaptureReplay);
    CHECK_FEATURE_1_2(bufferDeviceAddressMultiDevice);
    CHECK_FEATURE_1_2(vulkanMemoryModel);
    CHECK_FEATURE_1_2(vulkanMemoryModelDeviceScope);
    CHECK_FEATURE_1_2(vulkanMemoryModelAvailabilityVisibilityChains);
    CHECK_FEATURE_1_2(shaderOutputViewportIndex);
    CHECK_FEATURE_1_2(shaderOutputLayer);
    CHECK_FEATURE_1_2(subgroupBroadcastDynamicId);
#undef CHECK_FEATURE_1_2
#define CHECK_FEATURE_1_3(feature) CHECK_VULKAN_FEATURE(deviceFeatures13, vkFeatures13_, feature, "1.3 ");
    CHECK_FEATURE_1_3(robustImageAccess);
    CHECK_FEATURE_1_3(inlineUniformBlock);
    CHECK_FEATURE_1_3(descriptorBindingInlineUniformBlockUpdateAfterBind);
    CHECK_FEATURE_1_3(pipelineCreationCacheControl);
    CHECK_FEATURE_1_3(privateData);
    CHECK_FEATURE_1_3(shaderDemoteToHelperInvocation);
    CHECK_FEATURE_1_3(shaderTerminateInvocation);
    CHECK_FEATURE_1_3(subgroupSizeControl);
    CHECK_FEATURE_1_3(computeFullSubgroups);
    CHECK_FEATURE_1_3(synchronization2);
    CHECK_FEATURE_1_3(textureCompressionASTC_HDR);
    CHECK_FEATURE_1_3(shaderZeroInitializeWorkgroupMemory);
    CHECK_FEATURE_1_3(dynamicRendering);
    CHECK_FEATURE_1_3(shaderIntegerDotProduct);
    CHECK_FEATURE_1_3(maintenance4);
#undef CHECK_FEATURE_1_3
    if (!missingFeatures.empty()) {
      MINILOG_LOG_PROC(
#ifndef __APPLE__
          minilog::FatalError,
#else
          minilog::Warning,
#endif
          "Missing Vulkan features: %s\n",
          missingFeatures.c_str());
      // Do not exit here in case of MoltenVK, some 1.3 features are available via extensions.
#ifndef __APPLE__
      assert(false);
      return Result(Result::Code::RuntimeError);
#endif
    }
  }

  const VkDeviceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = createInfoNext,
      .queueCreateInfoCount = numQueues,
      .pQueueCreateInfos = ciQueue,
      .enabledExtensionCount = (uint32_t)enabledDeviceExtensionNames_.size(),
      .ppEnabledExtensionNames = enabledDeviceExtensionNames_.data(),
      .pEnabledFeatures = &deviceFeatures10,
  };
  VK_ASSERT_RETURN(vkCreateDevice(vkPhysicalDevice_, &ci, nullptr, &vkDevice_));

  volkLoadDevice(vkDevice_);

  vkGetDeviceQueue(vkDevice_, deviceQueues_.graphicsQueueFamilyIndex, 0, &deviceQueues_.graphicsQueue);
  vkGetDeviceQueue(vkDevice_, deviceQueues_.computeQueueFamilyIndex, 0, &deviceQueues_.computeQueue);

  VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_DEVICE, (uint64_t)vkDevice_, "Device: VulkanContext::vkDevice_"));

  immediate_ = std::make_unique<lvk::VulkanImmediateCommands>(
      vkDevice_, deviceQueues_.graphicsQueueFamilyIndex, has_EXT_device_fault_, "VulkanContext::immediate_");

  // create Vulkan pipeline cache
  {
    const VkPipelineCacheCreateInfo ci = {
        VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        nullptr,
        VkPipelineCacheCreateFlags(0),
        config_.pipelineCacheDataSize,
        config_.pipelineCacheData,
    };
    vkCreatePipelineCache(vkDevice_, &ci, nullptr, &pipelineCache_);
  }

  if (LVK_VULKAN_USE_VMA) {
    pimpl_->vma_ = lvk::createVmaAllocator(
        vkPhysicalDevice_, vkDevice_, vkInstance_, apiVersion > VK_API_VERSION_1_3 ? VK_API_VERSION_1_3 : apiVersion);
    LVK_ASSERT(pimpl_->vma_ != VK_NULL_HANDLE);
  }

  stagingDevice_ = std::make_unique<lvk::VulkanStagingDevice>(*this);

  // default texture
  {
    const uint32_t pixel = 0xFF000000;
    Result result;
    dummyTexture_ = this->createTexture(
                            {
                                .format = lvk::Format_RGBA_UN8,
                                .dimensions = {1, 1, 1},
                                .usage = TextureUsageBits_Sampled | TextureUsageBits_Storage,
                                .data = &pixel,
                            },
                            "Dummy 1x1 (black)",
                            &result)
                        .release();
    if (!LVK_VERIFY(result.isOk())) {
      return result;
    }
    LVK_ASSERT(texturesPool_.numObjects() == 1);
  }

  // default sampler
  LVK_ASSERT(samplersPool_.numObjects() == 0);
  createSampler(
      {
          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .magFilter = VK_FILTER_LINEAR,
          .minFilter = VK_FILTER_LINEAR,
          .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
          .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
          .mipLodBias = 0.0f,
          .anisotropyEnable = VK_FALSE,
          .maxAnisotropy = 0.0f,
          .compareEnable = VK_FALSE,
          .compareOp = VK_COMPARE_OP_ALWAYS,
          .minLod = 0.0f,
          .maxLod = static_cast<float>(LVK_MAX_MIP_LEVELS - 1),
          .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
          .unnormalizedCoordinates = VK_FALSE,
      },
      nullptr,
      Format_Invalid,
      "Sampler: default");

  querySurfaceCapabilities();

#if defined(LVK_WITH_TRACY_GPU)
  std::vector<VkTimeDomainEXT> timeDomains;

  if (has_EXT_calibrated_timestamps_) {
    uint32_t numTimeDomains = 0;
    VK_ASSERT(vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(vkPhysicalDevice_, &numTimeDomains, nullptr));
    timeDomains.resize(numTimeDomains);
    VK_ASSERT(vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(vkPhysicalDevice_, &numTimeDomains, timeDomains.data()));
  }

  const bool hasHostQuery = vkFeatures12_.hostQueryReset && [&timeDomains]() -> bool {
    for (VkTimeDomainEXT domain : timeDomains)
      if (domain == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT || domain == VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT)
        return true;
    return false;
  }();

  if (hasHostQuery) {
    pimpl_->tracyVkCtx_ = TracyVkContextHostCalibrated(
        vkPhysicalDevice_, vkDevice_, vkResetQueryPool, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
  } else {
    const VkCommandPoolCreateInfo ciCommandPool = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = deviceQueues_.graphicsQueueFamilyIndex,
    };
    VK_ASSERT(vkCreateCommandPool(vkDevice_, &ciCommandPool, nullptr, &pimpl_->tracyCommandPool_));
    VK_ASSERT(lvk::setDebugObjectName(
        vkDevice_, VK_OBJECT_TYPE_COMMAND_POOL, (uint64_t)pimpl_->tracyCommandPool_, "Command Pool: VulkanContextImpl::tracyCommandPool_"));
    const VkCommandBufferAllocateInfo aiCommandBuffer = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pimpl_->tracyCommandPool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_ASSERT(vkAllocateCommandBuffers(vkDevice_, &aiCommandBuffer, &pimpl_->tracyCommandBuffer_));
    if (has_EXT_calibrated_timestamps_) {
      pimpl_->tracyVkCtx_ = TracyVkContextCalibrated(vkPhysicalDevice_,
                                                     vkDevice_,
                                                     deviceQueues_.graphicsQueue,
                                                     pimpl_->tracyCommandBuffer_,
                                                     vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
                                                     vkGetCalibratedTimestampsEXT);
    } else {
      pimpl_->tracyVkCtx_ = TracyVkContext(vkPhysicalDevice_, vkDevice_, deviceQueues_.graphicsQueue, pimpl_->tracyCommandBuffer_);
    };
  }
  LVK_ASSERT(pimpl_->tracyVkCtx_);
#endif // LVK_WITH_TRACY_GPU

  // add one empty dset
  DSets_.push_back({});

  // create one separate DSL for input attachments
  {
    VkDescriptorSetLayoutBinding bindings[LVK_MAX_COLOR_ATTACHMENTS];
    for (uint32_t i = 0; i != LVK_ARRAY_NUM_ELEMENTS(bindings); i++) {
      bindings[i] = lvk::getDSLBinding(i, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    };
    const VkPhysicalDeviceLimits& limits = this->getVkPhysicalDeviceProperties().limits;
    const VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = std::min((uint32_t)LVK_MAX_COLOR_ATTACHMENTS, limits.maxPerStageDescriptorInputAttachments),
        .pBindings = bindings,
    };
    VK_ASSERT(vkCreateDescriptorSetLayout(vkDevice_, &dslci, nullptr, &dslInputAttachments_));
    VK_ASSERT(lvk::setDebugObjectName(vkDevice_,
                                      VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                      (uint64_t)dslInputAttachments_,
                                      "Descriptor Set Layout: VulkanContext::dslInputAttachments_"));
  }
  return Result();
}

lvk::Result lvk::VulkanContext::initSwapchain(uint32_t width, uint32_t height) {
  if (!vkDevice_ || !immediate_) {
    LLOGW("Call initContext() first");
    return Result(Result::Code::RuntimeError, "Call initContext() first");
  }

  if (swapchain_) {
    // destroy the old swapchain first
    // TODO: replace with VK_EXT_swapchain_maintenance1
    VK_ASSERT(vkDeviceWaitIdle(vkDevice_));
    swapchain_ = nullptr;
    vkDestroySemaphore(vkDevice_, timelineSemaphore_, nullptr);
  }

  if (!width || !height) {
    return Result(Result::Code::ArgumentOutOfRange, "Width and Height cannot be 0");
  }

  swapchain_ = std::make_unique<lvk::VulkanSwapchain>(*this, width, height);

  timelineSemaphore_ = lvk::createSemaphoreTimeline(vkDevice_, swapchain_->getNumSwapchainImages() - 1, "Semaphore: timelineSemaphore_");

  return swapchain_ ? Result() : Result(Result::Code::RuntimeError, "Failed to create swapchain");
}

lvk::Result lvk::VulkanContext::growDescriptorPool(VulkanContext::DescriptorSet& dset,
                                                   uint32_t maxTextures,
                                                   uint32_t maxSamplers,
                                                   uint32_t maxAccelStructs) {
  if (maxTextures == dset.maxTextures && maxSamplers == dset.maxSamplers && maxAccelStructs == dset.maxAccelStructs &&
      !awaitingNewImmutableSamplers_) {
    return Result();
  }

  dset.maxTextures = maxTextures;
  dset.maxSamplers = maxSamplers;
  dset.maxAccelStructs = maxAccelStructs;

#if LVK_VULKAN_PRINT_COMMANDS
  LLOGL("growDescriptorPool(%u, %u)\n", maxTextures, maxSamplers);
#endif // LVK_VULKAN_PRINT_COMMANDS

  if (!LVK_VERIFY(maxTextures <= vkPhysicalDeviceVulkan12Properties_.maxDescriptorSetUpdateAfterBindSampledImages)) {
    LLOGW("Max Textures exceeded: %u (max %u)",
          maxTextures,
          vkPhysicalDeviceVulkan12Properties_.maxDescriptorSetUpdateAfterBindSampledImages);
  }

  if (!LVK_VERIFY(maxSamplers <= vkPhysicalDeviceVulkan12Properties_.maxDescriptorSetUpdateAfterBindSamplers)) {
    LLOGW("Max Samplers exceeded %u (max %u)", maxSamplers, vkPhysicalDeviceVulkan12Properties_.maxDescriptorSetUpdateAfterBindSamplers);
  }

  if (dset.vkDSL != VK_NULL_HANDLE) {
    deferredTask(
        std::packaged_task<void()>([device = vkDevice_, dsl = dset.vkDSL]() { vkDestroyDescriptorSetLayout(device, dsl, nullptr); }));
  }
  if (dset.vkDPool != VK_NULL_HANDLE) {
    deferredTask(std::packaged_task<void()>([device = vkDevice_, dp = dset.vkDPool]() { vkDestroyDescriptorPool(device, dp, nullptr); }));
  }

  VkSampler firstYcbcrSampler = VK_NULL_HANDLE;

  // check if we have any YUV images
  for (const auto& obj : texturesPool_.objects_) {
    const VulkanImage* img = &obj.obj_;
    // multisampled images cannot be directly accessed from shaders
    const bool isTextureAvailable = (img->vkSamples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT;
    const bool isYUVImage = isTextureAvailable && img->isSampledImage() && lvk::getNumImagePlanes(img->vkImageFormat_) > 1;
    if (isYUVImage) {
      // find the first Ycbcr sampler and use it as a dummy
      // (https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html#VUID-VkDescriptorSetLayoutBinding-descriptorType-12200)
      firstYcbcrSampler = getOrCreateYcbcrSampler(vkFormatToFormat(img->vkImageFormat_));
      break;
    }
  }

  std::vector<VkSampler> immutableSamplers;
  const VkSampler* immutableSamplersData = nullptr;

  if (firstYcbcrSampler) {
    immutableSamplers.resize(maxTextures, firstYcbcrSampler);
    for (size_t i = 0; i != texturesPool_.objects_.size(); i++) {
      const auto& obj = texturesPool_.objects_[i];
      const VulkanImage* img = &obj.obj_;
      // multisampled images cannot be directly accessed from shaders
      const bool isTextureAvailable = (img->vkSamples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT;
      const bool isYUVImage = isTextureAvailable && img->isSampledImage() && lvk::getNumImagePlanes(img->vkImageFormat_) > 1;
      immutableSamplers[i] = isYUVImage ? getOrCreateYcbcrSampler(vkFormatToFormat(img->vkImageFormat_)) : firstYcbcrSampler;
    }
    immutableSamplersData = immutableSamplers.data();
  }

  // create default descriptor set layout which is going to be shared by graphics pipelines
  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                  VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
  if (has_KHR_ray_tracing_pipeline_) {
    stageFlags |= (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                   VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR);
  }
  const VkDescriptorSetLayoutBinding bindings[kBinding_NumBindings] = {
      lvk::getDSLBinding(kBinding_Textures, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures, stageFlags),
      lvk::getDSLBinding(kBinding_Samplers, VK_DESCRIPTOR_TYPE_SAMPLER, maxSamplers, stageFlags),
      lvk::getDSLBinding(kBinding_StorageImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, maxTextures, stageFlags),
      lvk::getDSLBinding(kBinding_YUVImages,
                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         (uint32_t)immutableSamplers.size() ? maxTextures : 0,
                         stageFlags,
                         immutableSamplersData),
      lvk::getDSLBinding(kBinding_AccelerationStructures, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, maxAccelStructs, stageFlags),
  };
  const uint32_t flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |
                         VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
  VkDescriptorBindingFlags bindingFlags[kBinding_NumBindings];
  for (int i = 0; i < kBinding_NumBindings; ++i) {
    bindingFlags[i] = flags;
  }
  const VkDescriptorSetLayoutBindingFlagsCreateInfo setLayoutBindingFlagsCI = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT,
      .bindingCount = uint32_t(has_KHR_acceleration_structure_ ? kBinding_NumBindings : kBinding_NumBindings - 1),
      .pBindingFlags = bindingFlags,
  };
  const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &setLayoutBindingFlagsCI,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT,
      .bindingCount = uint32_t(has_KHR_acceleration_structure_ ? kBinding_NumBindings : kBinding_NumBindings - 1),
      .pBindings = bindings,
  };
  VK_ASSERT(vkCreateDescriptorSetLayout(vkDevice_, &dslci, nullptr, &dset.vkDSL));
  VK_ASSERT(lvk::setDebugObjectName(
      vkDevice_, VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)dset.vkDSL, "Descriptor Set Layout: VulkanContext::vkDSL"));

  {
    // create default descriptor pool and allocate 1 descriptor set
    VkDescriptorPoolSize poolSizes[kBinding_NumBindings] = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, maxTextures},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, maxSamplers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, maxTextures},
    };
    uint32_t numPoolSizes = 3;
    if (!immutableSamplers.empty()) {
      poolSizes[numPoolSizes++] = VkDescriptorPoolSize{
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          pimpl_->maxCombinedImageSamplerDescriptorCount_ * maxTextures,
      };
    }
    if (has_KHR_acceleration_structure_) {
      poolSizes[numPoolSizes++] = VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, maxAccelStructs};
    }
    LVK_ASSERT(numPoolSizes <= kBinding_NumBindings);
    const VkDescriptorPoolCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = numPoolSizes,
        .pPoolSizes = poolSizes,
    };
    VK_ASSERT_RETURN(vkCreateDescriptorPool(vkDevice_, &ci, nullptr, &dset.vkDPool));
    const VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dset.vkDPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dset.vkDSL,
    };
    VK_ASSERT_RETURN(vkAllocateDescriptorSets(vkDevice_, &ai, &dset.vkDSet));
  }

  awaitingNewImmutableSamplers_ = false;

  return Result();
}

lvk::BufferHandle lvk::VulkanContext::createBuffer(VkDeviceSize bufferSize,
                                                   VkBufferUsageFlags usageFlags,
                                                   VkMemoryPropertyFlags memFlags,
                                                   lvk::Result* outResult,
                                                   const char* debugName) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  LVK_ASSERT(bufferSize > 0);

#define ENSURE_BUFFER_SIZE(flag, maxSize)                                                             \
  if (usageFlags & flag) {                                                                            \
    if (!LVK_VERIFY(bufferSize <= maxSize)) {                                                         \
      Result::setResult(outResult, Result(Result::Code::RuntimeError, "Buffer size exceeded" #flag)); \
      return {};                                                                                      \
    }                                                                                                 \
  }

  const VkPhysicalDeviceLimits& limits = getVkPhysicalDeviceProperties().limits;

  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, limits.maxUniformBufferRange);
  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, limits.maxStorageBufferRange);
  // any buffer
  ENSURE_BUFFER_SIZE(VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM, vkPhysicalDeviceVulkan11Properties_.maxMemoryAllocationSize);
#undef ENSURE_BUFFER_SIZE

  VulkanBuffer buf = {
      .bufferSize_ = bufferSize,
      .vkUsageFlags_ = usageFlags,
      .vkMemFlags_ = memFlags,
  };

  const VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = bufferSize,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };

  if (LVK_VULKAN_USE_VMA) {
    VmaAllocationCreateInfo vmaAllocInfo = {};

    // Initialize VmaAllocation Info
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      vmaAllocInfo = {
          .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
          .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
          .preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      };
    }

    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      // Check if coherent buffer is available.
      VK_ASSERT(vkCreateBuffer(vkDevice_, &ci, nullptr, &buf.vkBuffer_));
      VkMemoryRequirements requirements = {};
      vkGetBufferMemoryRequirements(vkDevice_, buf.vkBuffer_, &requirements);
      vkDestroyBuffer(vkDevice_, buf.vkBuffer_, nullptr);
      buf.vkBuffer_ = VK_NULL_HANDLE;

      if (requirements.memoryTypeBits & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        vmaAllocInfo.requiredFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        buf.isCoherentMemory_ = true;
      }
    }

    vmaAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    vmaCreateBufferWithAlignment((VmaAllocator)getVmaAllocator(), &ci, &vmaAllocInfo, 16, &buf.vkBuffer_, &buf.vmaAllocation_, nullptr);

    // handle memory-mapped buffers
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      vmaMapMemory((VmaAllocator)getVmaAllocator(), buf.vmaAllocation_, &buf.mappedPtr_);
    }
  } else {
    // create buffer
    VK_ASSERT(vkCreateBuffer(vkDevice_, &ci, nullptr, &buf.vkBuffer_));

    // back the buffer with some memory
    {
      const VkBufferMemoryRequirementsInfo2 ri = {
          .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
          .buffer = buf.vkBuffer_,
      };
      VkMemoryRequirements2 requirements = {
          .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      };
      vkGetBufferMemoryRequirements2(vkDevice_, &ri, &requirements);
      if (requirements.memoryRequirements.memoryTypeBits & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
        buf.isCoherentMemory_ = true;
      }

      VK_ASSERT(lvk::allocateMemory2(vkPhysicalDevice_, vkDevice_, &requirements, memFlags, &buf.vkMemory_));
      VK_ASSERT(vkBindBufferMemory(vkDevice_, buf.vkBuffer_, buf.vkMemory_, 0));
    }

    // handle memory-mapped buffers
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      VK_ASSERT(vkMapMemory(vkDevice_, buf.vkMemory_, 0, buf.bufferSize_, 0, &buf.mappedPtr_));
    }
  }

  LVK_ASSERT(buf.vkBuffer_ != VK_NULL_HANDLE);

  // set debug name
  VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_BUFFER, (uint64_t)buf.vkBuffer_, debugName));

  // handle shader access
  if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    const VkBufferDeviceAddressInfo ai = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buf.vkBuffer_,
    };
    buf.vkDeviceAddress_ = vkGetBufferDeviceAddress(vkDevice_, &ai);
    LVK_ASSERT(buf.vkDeviceAddress_);
  }

  return buffersPool_.create(std::move(buf));
}

void lvk::VulkanContext::bindDefaultDescriptorSets(VkCommandBuffer cmdBuf, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
  LVK_PROFILER_FUNCTION();
  const VkDescriptorSet dset = DSets_[lastUpdatedDSet_].vkDSet;
  const VkDescriptorSet dsets[4] = {dset, dset, dset, dset};
  vkCmdBindDescriptorSets(cmdBuf, bindPoint, layout, 0, (uint32_t)LVK_ARRAY_NUM_ELEMENTS(dsets), dsets, 0, nullptr);
}

void lvk::VulkanContext::checkAndUpdateDescriptorSets() {
  if (!awaitingCreation_) {
    // nothing to update here
    return;
  }

  // newly created resources can be used immediately - make sure they are put into descriptor sets
  LVK_PROFILER_FUNCTION();

  // update Vulkan descriptor set here

  lastUpdatedDSet_ = (lastUpdatedDSet_ + 1) % DSets_.size();

  if (const DescriptorSet& dset = DSets_[lastUpdatedDSet_]; dset.vkDSet) {
    // we can't reuse a dset that's either waiting to be submitted in a draw call
    // (which happens when textures are created mid-frame) or is still being processed
    if (dset.handle_.empty() || !immediate_->isReady(dset.handle_)) {
      // add a new empty dset to be populated right away
      lastUpdatedDSet_ = DSets_.size();
      DSets_.push_back({});
    }
  }

  DescriptorSet& dset = DSets_[lastUpdatedDSet_];

  dset.handle_ = {};

  // make sure the guard values are always there
  LVK_ASSERT(texturesPool_.numObjects() >= 1);
  LVK_ASSERT(samplersPool_.numObjects() >= 1);

  uint32_t newMaxTextures = std::max(dset.maxTextures, 16u);
  uint32_t newMaxSamplers = std::max(dset.maxSamplers, 16u);
  uint32_t newMaxAccelStructs = std::max(dset.maxAccelStructs, 1u);

  while (texturesPool_.objects_.size() > newMaxTextures) {
    newMaxTextures *= 2;
  }
  while (samplersPool_.objects_.size() > newMaxSamplers) {
    newMaxSamplers *= 2;
  }
  while (accelStructuresPool_.objects_.size() > newMaxAccelStructs) {
    newMaxAccelStructs *= 2;
  }
  growDescriptorPool(dset, newMaxTextures, newMaxSamplers, newMaxAccelStructs);

  // 1. Sampled and storage images
  std::vector<VkDescriptorImageInfo> infoSampledImages;
  std::vector<VkDescriptorImageInfo> infoStorageImages;
  std::vector<VkDescriptorImageInfo> infoYUVImages;

  infoSampledImages.reserve(texturesPool_.numObjects());
  infoStorageImages.reserve(texturesPool_.numObjects());

  const bool hasYcbcrSamplers = pimpl_->numYcbcrSamplers_ > 0;

  if (hasYcbcrSamplers) {
    infoYUVImages.reserve(texturesPool_.numObjects());
  }

  // use dummies to avoid sparse arrays
  VkImageView dummyImageView = texturesPool_.objects_[0].obj_.imageView_;
  VkSampler dummySampler = samplersPool_.objects_[0].obj_;

  for (const auto& obj : texturesPool_.objects_) {
    const VulkanImage& img = obj.obj_;
    const VkImageView view = obj.obj_.imageView_;
    const VkImageView storageView = obj.obj_.imageViewStorage_ ? obj.obj_.imageViewStorage_ : view;
    // multisampled images cannot be directly accessed from shaders
    const bool isTextureAvailable = (img.vkSamples_ & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT;
    const bool isYUVImage = isTextureAvailable && img.isSampledImage() && lvk::getNumImagePlanes(img.vkImageFormat_) > 1;
    const bool isSampledImage = isTextureAvailable && img.isSampledImage() && !isYUVImage;
    const bool isStorageImage = isTextureAvailable && img.isStorageImage();
    infoSampledImages.push_back(VkDescriptorImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = isSampledImage ? view : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    });
    LVK_ASSERT(infoSampledImages.back().imageView != VK_NULL_HANDLE);
    infoStorageImages.push_back(VkDescriptorImageInfo{
        .sampler = VK_NULL_HANDLE,
        .imageView = isStorageImage ? storageView : dummyImageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    });
    if (hasYcbcrSamplers) {
      // we don't need to update this if there're no YUV samplers
      infoYUVImages.push_back(VkDescriptorImageInfo{
          .sampler = dummySampler, // this will be replaced by immutable samplers from VkPipeline
          .imageView = isYUVImage ? view : dummyImageView,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      });
    }
  }

  // 2. Samplers
  std::vector<VkDescriptorImageInfo> infoSamplers;
  infoSamplers.reserve(samplersPool_.objects_.size());

  for (const auto& sampler : samplersPool_.objects_) {
    infoSamplers.push_back({
        .sampler = sampler.obj_ ? sampler.obj_ : samplersPool_.objects_[0].obj_,
        .imageView = VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
  }

  // 3. Acceleration structures
  std::vector<VkAccelerationStructureKHR> handlesAccelStructs;
  handlesAccelStructs.reserve(accelStructuresPool_.objects_.size());

  // use the first valid TLAS as a dummy
  const VkAccelerationStructureKHR dummyTLAS = [this]() -> VkAccelerationStructureKHR {
    for (const auto& as : accelStructuresPool_.objects_) {
      if (as.obj_.vkHandle && as.obj_.isTLAS)
        return as.obj_.vkHandle;
    }
    return VK_NULL_HANDLE;
  }();

  for (const auto& as : accelStructuresPool_.objects_) {
    handlesAccelStructs.push_back(as.obj_.isTLAS ? as.obj_.vkHandle : dummyTLAS);
  }

  VkWriteDescriptorSetAccelerationStructureKHR writeAccelStruct = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = (uint32_t)handlesAccelStructs.size(),
      .pAccelerationStructures = handlesAccelStructs.data(),
  };

  VkWriteDescriptorSet write[kBinding_NumBindings] = {};
  uint32_t numWrites = 0;

  if (!handlesAccelStructs.empty()) {
    write[numWrites++] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &writeAccelStruct,
        .dstSet = dset.vkDSet,
        .dstBinding = kBinding_AccelerationStructures,
        .dstArrayElement = 0,
        .descriptorCount = (uint32_t)handlesAccelStructs.size(),
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    };
  }

  if (!infoSampledImages.empty()) {
    write[numWrites++] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dset.vkDSet,
        .dstBinding = kBinding_Textures,
        .dstArrayElement = 0,
        .descriptorCount = (uint32_t)infoSampledImages.size(),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = infoSampledImages.data(),
    };
  }

  if (!infoSamplers.empty()) {
    write[numWrites++] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dset.vkDSet,
        .dstBinding = kBinding_Samplers,
        .dstArrayElement = 0,
        .descriptorCount = (uint32_t)infoSamplers.size(),
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = infoSamplers.data(),
    };
  }

  if (!infoStorageImages.empty()) {
    write[numWrites++] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dset.vkDSet,
        .dstBinding = kBinding_StorageImages,
        .dstArrayElement = 0,
        .descriptorCount = (uint32_t)infoStorageImages.size(),
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = infoStorageImages.data(),
    };
  }

  if (!infoYUVImages.empty()) {
    write[numWrites++] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = dset.vkDSet,
        .dstBinding = kBinding_YUVImages,
        .dstArrayElement = 0,
        .descriptorCount = (uint32_t)infoYUVImages.size(),
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = infoYUVImages.data(),
    };
  }

  // do not switch to the next descriptor set if there is nothing to update
  if (numWrites) {
#if LVK_VULKAN_PRINT_COMMANDS
    LLOGL("vkUpdateDescriptorSets()\n");
#endif // LVK_VULKAN_PRINT_COMMANDS
    LVK_PROFILER_ZONE("vkUpdateDescriptorSets()", LVK_PROFILER_COLOR_PRESENT);
    vkUpdateDescriptorSets(vkDevice_, numWrites, write, 0, nullptr);
    LVK_PROFILER_ZONE_END();
  }

  awaitingCreation_ = false;
}

lvk::SamplerHandle lvk::VulkanContext::createSampler(const VkSamplerCreateInfo& ci,
                                                     lvk::Result* outResult,
                                                     lvk::Format yuvFormat,
                                                     const char* debugName) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  VkSamplerCreateInfo cinfo = ci;

  if (yuvFormat != Format_Invalid) {
    cinfo.pNext = getOrCreateYcbcrConversionInfo(yuvFormat);
    // must be CLAMP_TO_EDGE
    // https://vulkan.lunarg.com/doc/view/1.3.268.0/windows/1.3-extensions/vkspec.html#VUID-VkSamplerCreateInfo-addressModeU-01646
    cinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cinfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    cinfo.anisotropyEnable = VK_FALSE;
    cinfo.unnormalizedCoordinates = VK_FALSE;
  }

  VkSampler sampler = VK_NULL_HANDLE;
  VK_ASSERT(vkCreateSampler(vkDevice_, &cinfo, nullptr, &sampler));
  VK_ASSERT(lvk::setDebugObjectName(vkDevice_, VK_OBJECT_TYPE_SAMPLER, (uint64_t)sampler, debugName));

  SamplerHandle handle = samplersPool_.create(VkSampler(sampler));

  awaitingCreation_ = true;

  return handle;
}

void lvk::VulkanContext::querySurfaceCapabilities() {
  // enumerate only the formats we are using
  const VkFormat depthFormats[] = {
      VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM};
  for (const VkFormat& depthFormat : depthFormats) {
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(vkPhysicalDevice_, depthFormat, &formatProps);

    if (formatProps.optimalTilingFeatures) {
      deviceDepthFormats_.push_back(depthFormat);
    }
  }

  if (vkSurface_ == VK_NULL_HANDLE) {
    return;
  }

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice_, vkSurface_, &deviceSurfaceCaps_);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_, vkSurface_, &formatCount, nullptr);

  if (formatCount) {
    deviceSurfaceFormats_.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_, vkSurface_, &formatCount, deviceSurfaceFormats_.data());
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_, vkSurface_, &presentModeCount, nullptr);

  if (presentModeCount) {
    devicePresentModes_.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_, vkSurface_, &presentModeCount, devicePresentModes_.data());
  }
}

std::vector<uint8_t> lvk::VulkanContext::getPipelineCacheData() const {
  size_t size = 0;
  vkGetPipelineCacheData(vkDevice_, pipelineCache_, &size, nullptr);

  std::vector<uint8_t> data(size);

  if (size) {
    vkGetPipelineCacheData(vkDevice_, pipelineCache_, &size, data.data());
  }

  return data;
}

void lvk::VulkanContext::deferredTask(std::packaged_task<void()>&& task, SubmitHandle handle) const {
  if (handle.empty()) {
    handle = immediate_->getNextSubmitHandle();
  }
  pimpl_->deferredTasks_.emplace_back(std::move(task), handle);
}

void* lvk::VulkanContext::getVmaAllocator() const {
  return pimpl_->vma_;
}

void lvk::VulkanContext::processDeferredTasks() const {
  std::vector<DeferredTask>::iterator it = pimpl_->deferredTasks_.begin();

  while (it != pimpl_->deferredTasks_.end() && immediate_->isReady(it->handle_, true)) {
    (it++)->task_();
  }

  pimpl_->deferredTasks_.erase(pimpl_->deferredTasks_.begin(), it);
}

void lvk::VulkanContext::waitDeferredTasks() {
  for (auto& task : pimpl_->deferredTasks_) {
    immediate_->wait(task.handle_);
    task.task_();
  }
  pimpl_->deferredTasks_.clear();
}

uint32_t lvk::VulkanContext::getMaxStorageBufferRange() const {
  return vkPhysicalDeviceProperties2_.properties.limits.maxStorageBufferRange;
}

bool lvk::VulkanContext::isExtensionEnabled(const char* ext) const {
  for (const char* name : enabledInstanceExtensionNames_) {
    if (strcmp(ext, name) == 0)
      return true;
  }
  for (const char* name : enabledDeviceExtensionNames_) {
    if (strcmp(ext, name) == 0)
      return true;
  }
  return false;
}
