/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "WebGPUClasses.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <glslang/Include/glslang_c_interface.h>

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
#elif LVK_WITH_SDL3
#include <SDL3/SDL.h>
#if defined(_WIN32)
#include <windows.h>
#endif // _WIN32
#endif // LVK_WITH_GLFW

#if defined(__APPLE__) && (LVK_WITH_GLFW || LVK_WITH_SDL3)
// lvk/LVK.mm
void* createCocoaWindowView(void* window, void** outLayer);
#endif

// Warns once per call site
#define LVK_WEBGPU_WARN_ONCE(...)   \
  do {                              \
    static bool isReported = false; \
    if (!isReported) {              \
      isReported = true;            \
      LLOGW(__VA_ARGS__);           \
    }                               \
  } while (0)

#define LVK_WEBGPU_NOT_IMPLEMENTED() LVK_WEBGPU_WARN_ONCE("%s() is not implemented in the WebGPU backend\n", __FUNCTION__)

namespace {

// exposes experimental features, such as the resource table
const char* kDawnToggles[] = {"allow_unsafe_apis"};

// nullptr gives the null string {NULL, WGPU_STRLEN}, which is not the empty string {NULL, 0}
WGPUStringView toStringView(const char* str) {
  return {.data = str, .length = WGPU_STRLEN};
}

uint64_t alignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// WebGPU has only 1x and 4x MSAA
uint32_t getSampleCountWGPU(uint32_t numSamples) {
  return numSamples > 1 ? 4u : 1u;
}

// integer formats are cleared with the integer members of ClearColorValue
WGPUColor clearColorToWGPU(const lvk::ClearColorValue& c, lvk::Format format) {
  using namespace lvk;
  switch (format) {
  case Format_R_UI8:
  case Format_R_UI16:
  case Format_R_UI32:
  case Format_RG_UI8:
  case Format_RG_UI16:
  case Format_RG_UI32:
  case Format_RGBA_UI8:
  case Format_RGBA_UI16:
  case Format_RGBA_UI32:
    return {double(c.uint32[0]), double(c.uint32[1]), double(c.uint32[2]), double(c.uint32[3])};
  case Format_R_I8:
  case Format_R_I16:
  case Format_R_I32:
  case Format_RG_I8:
  case Format_RG_I16:
  case Format_RG_I32:
  case Format_RGBA_I8:
  case Format_RGBA_I16:
  case Format_RGBA_I32:
    return {double(c.int32[0]), double(c.int32[1]), double(c.int32[2]), double(c.int32[3])};
  default:
    return {c.float32[0], c.float32[1], c.float32[2], c.float32[3]};
  }
}

WGPUTextureFormat formatToWGPU(lvk::Format format) {
  using namespace lvk;
  switch (format) {
  case Format_R_UN8:
    return WGPUTextureFormat_R8Unorm;
  case Format_R_UI8:
    return WGPUTextureFormat_R8Uint;
  case Format_R_I8:
    return WGPUTextureFormat_R8Sint;
  case Format_R_UI16:
    return WGPUTextureFormat_R16Uint;
  case Format_R_I16:
    return WGPUTextureFormat_R16Sint;
  case Format_R_UI32:
    return WGPUTextureFormat_R32Uint;
  case Format_R_I32:
    return WGPUTextureFormat_R32Sint;
  case Format_R_UN16:
    return WGPUTextureFormat_R16Unorm;
  case Format_R_F16:
    return WGPUTextureFormat_R16Float;
  case Format_R_F32:
    return WGPUTextureFormat_R32Float;
  case Format_RG_UN8:
    return WGPUTextureFormat_RG8Unorm;
  case Format_RG_UI8:
    return WGPUTextureFormat_RG8Uint;
  case Format_RG_I8:
    return WGPUTextureFormat_RG8Sint;
  case Format_RG_UI16:
    return WGPUTextureFormat_RG16Uint;
  case Format_RG_I16:
    return WGPUTextureFormat_RG16Sint;
  case Format_RG_UI32:
    return WGPUTextureFormat_RG32Uint;
  case Format_RG_I32:
    return WGPUTextureFormat_RG32Sint;
  case Format_RG_UN16:
    return WGPUTextureFormat_RG16Unorm;
  case Format_RG_F16:
    return WGPUTextureFormat_RG16Float;
  case Format_RG_F32:
    return WGPUTextureFormat_RG32Float;
  case Format_RGBA_UN8:
    return WGPUTextureFormat_RGBA8Unorm;
  case Format_RGBA_SN8:
    return WGPUTextureFormat_RGBA8Snorm;
  case Format_RGBA_UI8:
    return WGPUTextureFormat_RGBA8Uint;
  case Format_RGBA_I8:
    return WGPUTextureFormat_RGBA8Sint;
  case Format_RGBA_UI16:
    return WGPUTextureFormat_RGBA16Uint;
  case Format_RGBA_I16:
    return WGPUTextureFormat_RGBA16Sint;
  case Format_RGBA_UI32:
    return WGPUTextureFormat_RGBA32Uint;
  case Format_RGBA_I32:
    return WGPUTextureFormat_RGBA32Sint;
  case Format_RGBA_UN16:
    return WGPUTextureFormat_RGBA16Unorm;
  case Format_RGBA_F16:
    return WGPUTextureFormat_RGBA16Float;
  case Format_RGBA_F32:
    return WGPUTextureFormat_RGBA32Float;
  case Format_RGBA_SRGB8:
    return WGPUTextureFormat_RGBA8UnormSrgb;
  case Format_BGRA_UN8:
    return WGPUTextureFormat_BGRA8Unorm;
  case Format_BGRA_SRGB8:
    return WGPUTextureFormat_BGRA8UnormSrgb;
  case Format_A2B10G10R10_UN:
    return WGPUTextureFormat_RGB10A2Unorm;
  case Format_B10G11R11_UF:
    return WGPUTextureFormat_RG11B10Ufloat;
  case Format_E5B9G9R9_UF:
    return WGPUTextureFormat_RGB9E5Ufloat;
  case Format_ETC2_RGB8:
    return WGPUTextureFormat_ETC2RGB8Unorm;
  case Format_ETC2_SRGB8:
    return WGPUTextureFormat_ETC2RGB8UnormSrgb;
  case Format_EAC_RG11:
    return WGPUTextureFormat_EACRG11Unorm;
  // WebGPU has no BC1 format without alpha: the RGBA one decodes punch-through alpha blocks differently
  case Format_BC1_RGB:
  case Format_BC1_RGBA:
    return WGPUTextureFormat_BC1RGBAUnorm;
  case Format_BC1_SRGB:
  case Format_BC1_SRGBA:
    return WGPUTextureFormat_BC1RGBAUnormSrgb;
  case Format_BC2_RGBA:
    return WGPUTextureFormat_BC2RGBAUnorm;
  case Format_BC2_SRGBA:
    return WGPUTextureFormat_BC2RGBAUnormSrgb;
  case Format_BC3_RGBA:
    return WGPUTextureFormat_BC3RGBAUnorm;
  case Format_BC3_SRGBA:
    return WGPUTextureFormat_BC3RGBAUnormSrgb;
  case Format_BC4_R:
    return WGPUTextureFormat_BC4RUnorm;
  case Format_BC5_RG:
    return WGPUTextureFormat_BC5RGUnorm;
  case Format_BC5_RG_SN:
    return WGPUTextureFormat_BC5RGSnorm;
  case Format_BC7_RGBA:
    return WGPUTextureFormat_BC7RGBAUnorm;
  case Format_BC7_SRGBA:
    return WGPUTextureFormat_BC7RGBAUnormSrgb;
  case Format_ASTC_4x4_RGBA:
    return WGPUTextureFormat_ASTC4x4Unorm;
  case Format_ASTC_4x4_SRGBA:
    return WGPUTextureFormat_ASTC4x4UnormSrgb;
  case Format_ASTC_5x5_RGBA:
    return WGPUTextureFormat_ASTC5x5Unorm;
  case Format_ASTC_5x5_SRGBA:
    return WGPUTextureFormat_ASTC5x5UnormSrgb;
  case Format_ASTC_6x6_RGBA:
    return WGPUTextureFormat_ASTC6x6Unorm;
  case Format_ASTC_6x6_SRGBA:
    return WGPUTextureFormat_ASTC6x6UnormSrgb;
  case Format_Z_UN16:
    return WGPUTextureFormat_Depth16Unorm;
  case Format_Z_UN24:
    return WGPUTextureFormat_Depth24Plus;
  case Format_Z_F32:
    return WGPUTextureFormat_Depth32Float;
  case Format_Z_UN24_S_UI8:
    return WGPUTextureFormat_Depth24PlusStencil8;
  case Format_Z_F32_S_UI8:
    return WGPUTextureFormat_Depth32FloatStencil8;
  case Format_S_UI8:
    return WGPUTextureFormat_Stencil8;
  default:
    // A_UN8, BGRA_UI8, A2R10G10B10_UN, 16-bit packed formats, YUV
    return WGPUTextureFormat_Undefined;
  }
}

lvk::Format wgpuFormatToFormat(WGPUTextureFormat format) {
  switch (format) {
  case WGPUTextureFormat_BGRA8Unorm:
    return lvk::Format_BGRA_UN8;
  case WGPUTextureFormat_BGRA8UnormSrgb:
    return lvk::Format_BGRA_SRGB8;
  case WGPUTextureFormat_RGBA8Unorm:
    return lvk::Format_RGBA_UN8;
  case WGPUTextureFormat_RGBA8UnormSrgb:
    return lvk::Format_RGBA_SRGB8;
  case WGPUTextureFormat_RGBA16Float:
    return lvk::Format_RGBA_F16;
  case WGPUTextureFormat_RGB10A2Unorm:
    return lvk::Format_A2B10G10R10_UN;
  default:
    return lvk::Format_Invalid;
  }
}

bool hasDepth(lvk::Format format) {
  return format == lvk::Format_Z_UN16 || format == lvk::Format_Z_UN24 || format == lvk::Format_Z_F32 ||
         format == lvk::Format_Z_UN24_S_UI8 || format == lvk::Format_Z_F32_S_UI8;
}

bool hasStencil(lvk::Format format) {
  return format == lvk::Format_Z_UN24_S_UI8 || format == lvk::Format_Z_F32_S_UI8 || format == lvk::Format_S_UI8;
}

WGPUVertexFormat vertexFormatToWGPU(lvk::VertexFormat format) {
  using namespace lvk;
  switch (format) {
  case VertexFormat_Float1:
    return WGPUVertexFormat_Float32;
  case VertexFormat_Float2:
    return WGPUVertexFormat_Float32x2;
  case VertexFormat_Float3:
    return WGPUVertexFormat_Float32x3;
  case VertexFormat_Float4:
    return WGPUVertexFormat_Float32x4;
  case VertexFormat_Byte1:
    return WGPUVertexFormat_Sint8;
  case VertexFormat_Byte2:
    return WGPUVertexFormat_Sint8x2;
  case VertexFormat_Byte4:
    return WGPUVertexFormat_Sint8x4;
  case VertexFormat_UByte1:
    return WGPUVertexFormat_Uint8;
  case VertexFormat_UByte2:
    return WGPUVertexFormat_Uint8x2;
  case VertexFormat_UByte4:
    return WGPUVertexFormat_Uint8x4;
  case VertexFormat_Short1:
    return WGPUVertexFormat_Sint16;
  case VertexFormat_Short2:
    return WGPUVertexFormat_Sint16x2;
  case VertexFormat_Short4:
    return WGPUVertexFormat_Sint16x4;
  case VertexFormat_UShort1:
    return WGPUVertexFormat_Uint16;
  case VertexFormat_UShort2:
    return WGPUVertexFormat_Uint16x2;
  case VertexFormat_UShort4:
    return WGPUVertexFormat_Uint16x4;
  case VertexFormat_Byte2Norm:
    return WGPUVertexFormat_Snorm8x2;
  case VertexFormat_Byte4Norm:
    return WGPUVertexFormat_Snorm8x4;
  case VertexFormat_UByte2Norm:
    return WGPUVertexFormat_Unorm8x2;
  case VertexFormat_UByte4Norm:
    return WGPUVertexFormat_Unorm8x4;
  case VertexFormat_Short2Norm:
    return WGPUVertexFormat_Snorm16x2;
  case VertexFormat_Short4Norm:
    return WGPUVertexFormat_Snorm16x4;
  case VertexFormat_UShort2Norm:
    return WGPUVertexFormat_Unorm16x2;
  case VertexFormat_UShort4Norm:
    return WGPUVertexFormat_Unorm16x4;
  case VertexFormat_Int1:
    return WGPUVertexFormat_Sint32;
  case VertexFormat_Int2:
    return WGPUVertexFormat_Sint32x2;
  case VertexFormat_Int3:
    return WGPUVertexFormat_Sint32x3;
  case VertexFormat_Int4:
    return WGPUVertexFormat_Sint32x4;
  case VertexFormat_UInt1:
    return WGPUVertexFormat_Uint32;
  case VertexFormat_UInt2:
    return WGPUVertexFormat_Uint32x2;
  case VertexFormat_UInt3:
    return WGPUVertexFormat_Uint32x3;
  case VertexFormat_UInt4:
    return WGPUVertexFormat_Uint32x4;
  case VertexFormat_HalfFloat1:
    return WGPUVertexFormat_Float16;
  case VertexFormat_HalfFloat2:
    return WGPUVertexFormat_Float16x2;
  case VertexFormat_HalfFloat4:
    return WGPUVertexFormat_Float16x4;
  default:
    // 3-component 8/16-bit formats and signed 2_10_10_10 do not exist in WebGPU
    return WGPUVertexFormat_Force32;
  }
}

WGPUPrimitiveTopology topologyToWGPU(lvk::Topology topology) {
  switch (topology) {
  case lvk::Topology_Point:
    return WGPUPrimitiveTopology_PointList;
  case lvk::Topology_Line:
    return WGPUPrimitiveTopology_LineList;
  case lvk::Topology_LineStrip:
    return WGPUPrimitiveTopology_LineStrip;
  case lvk::Topology_Triangle:
    return WGPUPrimitiveTopology_TriangleList;
  case lvk::Topology_TriangleStrip:
    return WGPUPrimitiveTopology_TriangleStrip;
  case lvk::Topology_Patch:
    return WGPUPrimitiveTopology_Undefined;
  }
  return WGPUPrimitiveTopology_Undefined;
}

WGPUCullMode cullModeToWGPU(lvk::CullMode mode) {
  switch (mode) {
  case lvk::CullMode_None:
    return WGPUCullMode_None;
  case lvk::CullMode_Front:
    return WGPUCullMode_Front;
  case lvk::CullMode_Back:
    return WGPUCullMode_Back;
  }
  return WGPUCullMode_None;
}

WGPUBlendOperation blendOpToWGPU(lvk::BlendOp op) {
  switch (op) {
  case lvk::BlendOp_Add:
    return WGPUBlendOperation_Add;
  case lvk::BlendOp_Subtract:
    return WGPUBlendOperation_Subtract;
  case lvk::BlendOp_ReverseSubtract:
    return WGPUBlendOperation_ReverseSubtract;
  case lvk::BlendOp_Min:
    return WGPUBlendOperation_Min;
  case lvk::BlendOp_Max:
    return WGPUBlendOperation_Max;
  }
  return WGPUBlendOperation_Add;
}

WGPUBlendFactor blendFactorToWGPU(lvk::BlendFactor factor) {
  using namespace lvk;
  switch (factor) {
  case BlendFactor_Zero:
    return WGPUBlendFactor_Zero;
  case BlendFactor_One:
    return WGPUBlendFactor_One;
  case BlendFactor_SrcColor:
    return WGPUBlendFactor_Src;
  case BlendFactor_OneMinusSrcColor:
    return WGPUBlendFactor_OneMinusSrc;
  case BlendFactor_SrcAlpha:
    return WGPUBlendFactor_SrcAlpha;
  case BlendFactor_OneMinusSrcAlpha:
    return WGPUBlendFactor_OneMinusSrcAlpha;
  case BlendFactor_DstColor:
    return WGPUBlendFactor_Dst;
  case BlendFactor_OneMinusDstColor:
    return WGPUBlendFactor_OneMinusDst;
  case BlendFactor_DstAlpha:
    return WGPUBlendFactor_DstAlpha;
  case BlendFactor_OneMinusDstAlpha:
    return WGPUBlendFactor_OneMinusDstAlpha;
  case BlendFactor_SrcAlphaSaturated:
    return WGPUBlendFactor_SrcAlphaSaturated;
  // WebGPU has a single blend constant for color and alpha
  case BlendFactor_BlendColor:
  case BlendFactor_BlendAlpha:
    return WGPUBlendFactor_Constant;
  case BlendFactor_OneMinusBlendColor:
  case BlendFactor_OneMinusBlendAlpha:
    return WGPUBlendFactor_OneMinusConstant;
  case BlendFactor_Src1Color:
    return WGPUBlendFactor_Src1;
  case BlendFactor_OneMinusSrc1Color:
    return WGPUBlendFactor_OneMinusSrc1;
  case BlendFactor_Src1Alpha:
    return WGPUBlendFactor_Src1Alpha;
  case BlendFactor_OneMinusSrc1Alpha:
    return WGPUBlendFactor_OneMinusSrc1Alpha;
  }
  return WGPUBlendFactor_One;
}

WGPUPresentMode presentModeToWGPU(lvk::PresentMode mode) {
  switch (mode) {
  case lvk::PresentMode_FIFO:
    return WGPUPresentMode_Fifo;
  case lvk::PresentMode_FIFO_Relaxed:
    return WGPUPresentMode_FifoRelaxed;
  case lvk::PresentMode_Immediate:
    return WGPUPresentMode_Immediate;
  case lvk::PresentMode_Mailbox:
    return WGPUPresentMode_Mailbox;
  default:
    return WGPUPresentMode_Undefined;
  }
}

lvk::PresentMode wgpuPresentModeToPresentMode(WGPUPresentMode mode) {
  switch (mode) {
  case WGPUPresentMode_FifoRelaxed:
    return lvk::PresentMode_FIFO_Relaxed;
  case WGPUPresentMode_Immediate:
    return lvk::PresentMode_Immediate;
  case WGPUPresentMode_Mailbox:
    return lvk::PresentMode_Mailbox;
  default:
    return lvk::PresentMode_FIFO;
  }
}

WGPULoadOp loadOpToWGPU(lvk::LoadOp op) {
  // WebGPU has no "don't care": clearing is the cheapest defined option; `None` keeps the contents
  return op == lvk::LoadOp_Load || op == lvk::LoadOp_None ? WGPULoadOp_Load : WGPULoadOp_Clear;
}

WGPUStoreOp storeOpToWGPU(lvk::StoreOp op) {
  return op == lvk::StoreOp_Store || op == lvk::StoreOp_None ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
}

WGPUCompareFunction compareOpToWGPU(lvk::CompareOp op) {
  switch (op) {
  case lvk::CompareOp_Never:
    return WGPUCompareFunction_Never;
  case lvk::CompareOp_Less:
    return WGPUCompareFunction_Less;
  case lvk::CompareOp_Equal:
    return WGPUCompareFunction_Equal;
  case lvk::CompareOp_LessEqual:
    return WGPUCompareFunction_LessEqual;
  case lvk::CompareOp_Greater:
    return WGPUCompareFunction_Greater;
  case lvk::CompareOp_NotEqual:
    return WGPUCompareFunction_NotEqual;
  case lvk::CompareOp_GreaterEqual:
    return WGPUCompareFunction_GreaterEqual;
  case lvk::CompareOp_AlwaysPass:
    return WGPUCompareFunction_Always;
  }
  return WGPUCompareFunction_Always;
}

WGPUStencilOperation stencilOpToWGPU(lvk::StencilOp op) {
  switch (op) {
  case lvk::StencilOp_Keep:
    return WGPUStencilOperation_Keep;
  case lvk::StencilOp_Zero:
    return WGPUStencilOperation_Zero;
  case lvk::StencilOp_Replace:
    return WGPUStencilOperation_Replace;
  case lvk::StencilOp_IncrementClamp:
    return WGPUStencilOperation_IncrementClamp;
  case lvk::StencilOp_DecrementClamp:
    return WGPUStencilOperation_DecrementClamp;
  case lvk::StencilOp_Invert:
    return WGPUStencilOperation_Invert;
  case lvk::StencilOp_IncrementWrap:
    return WGPUStencilOperation_IncrementWrap;
  case lvk::StencilOp_DecrementWrap:
    return WGPUStencilOperation_DecrementWrap;
  }
  return WGPUStencilOperation_Keep;
}

WGPUStencilFaceState stencilFaceToWGPU(const lvk::StencilFaceState& face) {
  return {
      .compare = compareOpToWGPU(face.stencilCompareOp),
      .failOp = stencilOpToWGPU(face.stencilFailureOp),
      .depthFailOp = stencilOpToWGPU(face.depthFailureOp),
      .passOp = stencilOpToWGPU(face.depthStencilPassOp),
  };
}

WGPUAddressMode addressModeToWGPU(lvk::SamplerWrap wrap) {
  switch (wrap) {
  case lvk::SamplerWrap_Repeat:
    return WGPUAddressMode_Repeat;
  case lvk::SamplerWrap_Clamp:
    return WGPUAddressMode_ClampToEdge;
  case lvk::SamplerWrap_MirrorRepeat:
    return WGPUAddressMode_MirrorRepeat;
  case lvk::SamplerWrap_ClampToBorder:
  case lvk::SamplerWrap_MirrorClampToEdge:
    // WebGPU has neither border colors nor mirror-once
    LLOGW("SamplerWrap_ClampToBorder and SamplerWrap_MirrorClampToEdge are not supported by WebGPU: using ClampToEdge\n");
    return WGPUAddressMode_ClampToEdge;
  }
  return WGPUAddressMode_Repeat;
}

WGPUComponentSwizzle swizzleToWGPU(lvk::Swizzle swizzle, WGPUComponentSwizzle identity) {
  switch (swizzle) {
  case lvk::Swizzle_Default:
    return identity;
  case lvk::Swizzle_0:
    return WGPUComponentSwizzle_Zero;
  case lvk::Swizzle_1:
    return WGPUComponentSwizzle_One;
  case lvk::Swizzle_R:
    return WGPUComponentSwizzle_R;
  case lvk::Swizzle_G:
    return WGPUComponentSwizzle_G;
  case lvk::Swizzle_B:
    return WGPUComponentSwizzle_B;
  case lvk::Swizzle_A:
    return WGPUComponentSwizzle_A;
  }
  return identity;
}

// block size of compressed formats, 1 for everything else
uint32_t getBlockSize(lvk::Format format) {
  using namespace lvk;
  switch (format) {
  case Format_ETC2_RGB8:
  case Format_ETC2_SRGB8:
  case Format_EAC_RG11:
  case Format_BC1_RGB:
  case Format_BC1_SRGB:
  case Format_BC1_RGBA:
  case Format_BC1_SRGBA:
  case Format_BC2_RGBA:
  case Format_BC2_SRGBA:
  case Format_BC3_RGBA:
  case Format_BC3_SRGBA:
  case Format_BC4_R:
  case Format_BC5_RG:
  case Format_BC5_RG_SN:
  case Format_BC7_RGBA:
  case Format_BC7_SRGBA:
  case Format_ASTC_4x4_RGBA:
  case Format_ASTC_4x4_SRGBA:
    return 4;
  case Format_ASTC_5x5_RGBA:
  case Format_ASTC_5x5_SRGBA:
    return 5;
  case Format_ASTC_6x6_RGBA:
  case Format_ASTC_6x6_SRGBA:
    return 6;
  default:
    return 1;
  }
}

// color formats generateMipmap() can render and filter in every WebGPU implementation
bool isMipmapGenerationSupported(WGPUTextureFormat format) {
  switch (format) {
  case WGPUTextureFormat_R8Unorm:
  case WGPUTextureFormat_RG8Unorm:
  case WGPUTextureFormat_RGBA8Unorm:
  case WGPUTextureFormat_RGBA8UnormSrgb:
  case WGPUTextureFormat_BGRA8Unorm:
  case WGPUTextureFormat_BGRA8UnormSrgb:
  case WGPUTextureFormat_R16Float:
  case WGPUTextureFormat_RG16Float:
  case WGPUTextureFormat_RGBA16Float:
  case WGPUTextureFormat_RGB10A2Unorm:
    return true;
  default:
    return false;
  }
}

// formats with the storage texture capability in core WebGPU
bool isStorageSupported(WGPUTextureFormat format) {
  switch (format) {
  case WGPUTextureFormat_RGBA8Unorm:
  case WGPUTextureFormat_RGBA8Snorm:
  case WGPUTextureFormat_RGBA8Uint:
  case WGPUTextureFormat_RGBA8Sint:
  case WGPUTextureFormat_RGBA16Uint:
  case WGPUTextureFormat_RGBA16Sint:
  case WGPUTextureFormat_RGBA16Float:
  case WGPUTextureFormat_R32Uint:
  case WGPUTextureFormat_R32Sint:
  case WGPUTextureFormat_R32Float:
  case WGPUTextureFormat_RG32Uint:
  case WGPUTextureFormat_RG32Sint:
  case WGPUTextureFormat_RG32Float:
  case WGPUTextureFormat_RGBA32Uint:
  case WGPUTextureFormat_RGBA32Sint:
  case WGPUTextureFormat_RGBA32Float:
    return true;
  default:
    return false;
  }
}

WGPUTextureViewDimension getViewDimension(lvk::TextureType type, uint32_t numLayers) {
  switch (type) {
  case lvk::TextureType_2D:
    return numLayers > 1 ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
  case lvk::TextureType_3D:
    return WGPUTextureViewDimension_3D;
  case lvk::TextureType_Cube:
    return numLayers > 6 ? WGPUTextureViewDimension_CubeArray : WGPUTextureViewDimension_Cube;
  }
  return WGPUTextureViewDimension_2D;
}

bool isSameStencilFace(const lvk::StencilFaceState& a, const lvk::StencilFaceState& b) {
  return a.stencilFailureOp == b.stencilFailureOp && a.depthFailureOp == b.depthFailureOp && a.depthStencilPassOp == b.depthStencilPassOp &&
         a.stencilCompareOp == b.stencilCompareOp && a.readMask == b.readMask && a.writeMask == b.writeMask;
}

// the stencil reference is set on the render pass
bool isSameDynamicState(const lvk::WebGPUDynamicState& a, const lvk::WebGPUDynamicState& b) {
  return a.depthCompareOp == b.depthCompareOp && a.isDepthWriteEnabled == b.isDepthWriteEnabled &&
         a.isStencilEnabled == b.isStencilEnabled && isSameStencilFace(a.stencilFront, b.stencilFront) &&
         isSameStencilFace(a.stencilBack, b.stencilBack) && a.isDepthBiasEnabled == b.isDepthBiasEnabled &&
         a.depthBiasConstant == b.depthBiasConstant && a.depthBiasSlope == b.depthBiasSlope && a.depthBiasClamp == b.depthBiasClamp;
}

// only the constants the shader declares (WebGPU rejects the others); booleans are VkBool32
std::vector<lvk::WebGPUPipelineConstant> getPipelineConstants(const lvk::SpecializationConstantDesc& specInfo,
                                                              const lvk::WebGPUShaderModuleState* module,
                                                              const char* debugName) {
  std::vector<lvk::WebGPUPipelineConstant> constants;

  if (!module) {
    return constants;
  }

  const uint32_t numConstants = specInfo.getNumSpecializationConstants();

  for (uint32_t i = 0; i != numConstants; i++) {
    const lvk::SpecializationConstantEntry& entry = specInfo.entries[i];
    const lvk::WGSLOverride* wgslOverride = nullptr;
    for (const lvk::WGSLOverride& o : module->overrides) {
      if (o.constantId == entry.constantId) {
        wgslOverride = &o;
        break;
      }
    }
    if (!wgslOverride) {
      continue;
    }
    const bool isInData = specInfo.data && entry.offset + entry.size <= specInfo.dataSize;
    if (entry.size != sizeof(uint32_t) || !isInData) {
      LLOGW("Pipeline `%s`: specialization constant %u must be 4 bytes within SpecializationConstantDesc::data\n",
            debugName,
            entry.constantId);
      continue;
    }
    const uint8_t* ptr = static_cast<const uint8_t*>(specInfo.data) + entry.offset;
    uint32_t valueUint = 0;
    float valueFloat = 0.0f;
    memcpy(&valueUint, ptr, sizeof(valueUint));
    memcpy(&valueFloat, ptr, sizeof(valueFloat));
    constants.push_back({
        .constantId = entry.constantId,
        .value = wgslOverride->type == lvk::WGSLOverrideType_Bool   ? (valueUint ? 1.0 : 0.0)
                 : wgslOverride->type == lvk::WGSLOverrideType_Int  ? double(int32_t(valueUint))
                 : wgslOverride->type == lvk::WGSLOverrideType_Uint ? double(valueUint)
                                                                    : double(valueFloat),
    });
  }

  return constants;
}

// the keys are decimal constant ids, matching `@id(constantId) override`
struct ConstantEntries {
  char keys[lvk::SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX][12] = {};
  WGPUConstantEntry entries[lvk::SpecializationConstantDesc::LVK_SPECIALIZATION_CONSTANTS_MAX] = {};
  size_t count = 0;
};

void getConstantEntries(const std::vector<lvk::WebGPUPipelineConstant>& constants, ConstantEntries& out) {
  LVK_ASSERT(constants.size() <= LVK_ARRAY_NUM_ELEMENTS(out.entries));
  for (const lvk::WebGPUPipelineConstant& c : constants) {
    snprintf(out.keys[out.count], sizeof(out.keys[out.count]), "%u", c.constantId);
    out.entries[out.count] = {
        .key = {.data = out.keys[out.count], .length = strlen(out.keys[out.count])},
        .value = c.value,
    };
    out.count++;
  }
}

const char* kMipmapShaderWGSL = R"(
@group(0) @binding(0) var srcTexture : texture_2d<f32>;
@group(0) @binding(1) var srcSampler : sampler;

struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
}

@vertex fn vertexMain(@builtin(vertex_index) index : u32) -> VertexOutput {
  // a full-screen triangle
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  var out : VertexOutput;
  out.position = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  out.uv = uv;
  return out;
}

@fragment fn fragmentMain(in : VertexOutput) -> @location(0) vec4f {
  return textureSampleLevel(srcTexture, srcSampler, in.uv, 0.0);
}
)";

void onUncapturedError(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void* userdata1, void*) {
  LLOGW("WebGPU error (%u): %.*s\n", unsigned(type), int(message.length), message.data);
  const bool terminateOnValidationError = *static_cast<const bool*>(userdata1);
  if (terminateOnValidationError) {
    abort();
  }
}

void onDeviceLost(const WGPUDevice*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
  if (reason != WGPUDeviceLostReason_Destroyed && reason != WGPUDeviceLostReason_CallbackCancelled) {
    LLOGW("WebGPU device lost (%u): %.*s\n", unsigned(reason), int(message.length), message.data);
  }
}

WGPUAdapter requestAdapter(WGPUInstance instance, WGPUSurface surface, WGPUBackendType backendType) {
  // not const: WGPURequestAdapterOptions::nextInChain is a non-const pointer
  WGPUDawnTogglesDescriptor toggles = {
      .chain = {.sType = WGPUSType_DawnTogglesDescriptor},
      .enabledToggleCount = LVK_ARRAY_NUM_ELEMENTS(kDawnToggles),
      .enabledToggles = kDawnToggles,
  };
  const WGPURequestAdapterOptions options = {
      .nextInChain = &toggles.chain,
      .featureLevel = WGPUFeatureLevel_Core,
      .powerPreference = WGPUPowerPreference_HighPerformance,
      .backendType = backendType,
      .compatibleSurface = surface,
  };
  WGPUAdapter adapter = nullptr;
  const WGPURequestAdapterCallbackInfo callbackInfo = {
      .mode = WGPUCallbackMode_WaitAnyOnly,
      .callback =
          [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*) {
            if (status != WGPURequestAdapterStatus_Success) {
              LLOGL("WebGPU adapter request failed: %.*s\n", int(message.length), message.data);
            }
            *static_cast<WGPUAdapter*>(userdata1) = status == WGPURequestAdapterStatus_Success ? adapter : nullptr;
          },
      .userdata1 = &adapter,
  };
  WGPUFutureWaitInfo waitInfo = {.future = wgpuInstanceRequestAdapter(instance, &options, callbackInfo)};
  wgpuInstanceWaitAny(instance, 1, &waitInfo, UINT64_MAX);
  return adapter;
}

} // namespace

lvk::WebGPUCommandBuffer::WebGPUCommandBuffer(WebGPUContext* ctx) : ctx_(ctx) {
  encoder_ = wgpuDeviceCreateCommandEncoder(ctx_->device_, nullptr);
}

lvk::WebGPUCommandBuffer::~WebGPUCommandBuffer() {
  if (pass_) {
    wgpuRenderPassEncoderEnd(pass_);
    wgpuRenderPassEncoderRelease(pass_);
  }
  for (WGPUTextureView view : passViews_) {
    wgpuTextureViewRelease(view);
  }
  if (encoder_) {
    wgpuCommandEncoderRelease(encoder_);
  }
  // not submitted: the GPU never used them
  for (WebGPUPushConstantsChunk& chunk : pushConstantsChunks_) {
    ctx_->freePushConstantsChunks_.push_back(std::move(chunk));
  }
}

void lvk::WebGPUCommandBuffer::cmdBarrier(const Barrier&) {
  // WebGPU tracks resource usage and inserts barriers itself
}

void lvk::WebGPUCommandBuffer::cmdTransitionToGeneral(const ldr::Span<TextureHandle>&, ShaderStage) const {
  // WebGPU has no image layouts
}

void lvk::WebGPUCommandBuffer::cmdTransitionToShaderReadOnly(const ldr::Span<TextureHandle>&, ShaderStage) const {
  // WebGPU has no image layouts
}

void lvk::WebGPUCommandBuffer::cmdTransitionToRenderingLocalRead(const ldr::Span<TextureHandle>&) const {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdPushDebugGroupLabel(const char* label, uint32_t) const {
  LVK_ASSERT(label);
  if (pass_) {
    wgpuRenderPassEncoderPushDebugGroup(pass_, toStringView(label));
  } else {
    wgpuCommandEncoderPushDebugGroup(encoder_, toStringView(label));
  }
}

void lvk::WebGPUCommandBuffer::cmdInsertDebugEventLabel(const char* label, uint32_t) const {
  LVK_ASSERT(label);
  if (pass_) {
    wgpuRenderPassEncoderInsertDebugMarker(pass_, toStringView(label));
  } else {
    wgpuCommandEncoderInsertDebugMarker(encoder_, toStringView(label));
  }
}

void lvk::WebGPUCommandBuffer::cmdPopDebugGroupLabel() const {
  if (pass_) {
    wgpuRenderPassEncoderPopDebugGroup(pass_);
  } else {
    wgpuCommandEncoderPopDebugGroup(encoder_);
  }
}

void lvk::WebGPUCommandBuffer::cmdBindRayTracingPipeline(RayTracingPipelineHandle) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdBindComputePipeline(ComputePipelineHandle handle) {
  // WebGPU compute passes are created per dispatch
  currentComputePipeline_ = handle;
}

WGPUComputePassEncoder lvk::WebGPUCommandBuffer::beginComputePass(const WebGPUComputePipelineState& cps) {
  const WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder_, nullptr);
  wgpuComputePassEncoderSetPipeline(pass, cps.pipeline);
  if (cps.hasPushConstantsBuffer) {
    uint32_t offset = 0;
    const WebGPUPushConstantsChunk& chunk = getPushConstantsSlot(&offset);
    wgpuComputePassEncoderSetBindGroup(pass, 0, chunk.bindGroupReadWrite, 1, &offset);
  } else {
    const uint32_t offset = 0;
    wgpuComputePassEncoderSetBindGroup(pass, 0, ctx_->arenasBindGroupReadWrite_, 1, &offset);
  }
  if (ctx_->resourceTable_) {
    wgpuComputePassEncoderSetResourceTable(pass, ctx_->resourceTable_);
  }
  // the whole range the pipeline reserves: pushConstants_ is zero-initialized past what was set
  if (cps.immediateSize) {
    wgpuComputePassEncoderSetImmediates(pass, 0, pushConstants_, cps.immediateSize);
  }
  return pass;
}

void lvk::WebGPUCommandBuffer::cmdDispatch(const Dimensions& groupCount, const Dependencies&) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CMD_DISPATCH);
  LVK_ASSERT_MSG(!pass_, "cmdDispatch() cannot be called inside a render pass");
  const WebGPUComputePipelineState* cps = ctx_->computePipelinesPool_.get(currentComputePipeline_);
  if (!cps) {
    LVK_WEBGPU_WARN_ONCE("cmdDispatch(): no valid compute pipeline is bound, the dispatch is skipped\n");
  }
  if (!cps || pass_) {
    return;
  }
  hasDispatches_ = true;
  const WGPUComputePassEncoder pass = beginComputePass(*cps);
  wgpuComputePassEncoderDispatchWorkgroups(pass, groupCount.width, groupCount.height, groupCount.depth);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
}

void lvk::WebGPUCommandBuffer::cmdDispatchIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, const Dependencies&) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CMD_DISPATCH);
  LVK_ASSERT_MSG(!pass_, "cmdDispatchIndirect() cannot be called inside a render pass");
  const WebGPUComputePipelineState* cps = ctx_->computePipelinesPool_.get(currentComputePipeline_);
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(indirectBuffer);
  LVK_ASSERT(buf);
  if (!cps) {
    LVK_WEBGPU_WARN_ONCE("cmdDispatchIndirect(): no valid compute pipeline is bound, the dispatch is skipped\n");
  }
  if (!cps || !buf || pass_) {
    return;
  }
  // indirect arguments cannot be read from an arena bound as writable storage: copy them out first
  wgpuCommandEncoderCopyBufferToBuffer(
      encoder_, buf->buffer, buf->offset + indirectBufferOffset, ctx_->indirectScratchBuffer_, 0, 3 * sizeof(uint32_t));
  hasDispatches_ = true;
  const WGPUComputePassEncoder pass = beginComputePass(*cps);
  wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, ctx_->indirectScratchBuffer_, 0);
  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);
}

WGPUTextureView lvk::WebGPUCommandBuffer::getAttachmentView(const WebGPUTexture& tex, uint32_t layer, uint32_t level) {
  // attachments need views with one mip-level: 2D views with one layer, or 3D views (the layer is the depth slice then)
  const bool is3D = tex.type == TextureType_3D;
  const bool isOwnViewUsable = tex.numLevels == 1 && !level && (is3D || (tex.viewDimension == WGPUTextureViewDimension_2D && !layer));
  if (isOwnViewUsable) {
    return tex.view;
  }
  const WGPUTextureViewDescriptor viewDesc = {
      .dimension = is3D ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D,
      .baseMipLevel = tex.baseLevel + level,
      .mipLevelCount = 1,
      .baseArrayLayer = is3D ? 0 : tex.baseLayer + layer,
      .arrayLayerCount = 1,
      .aspect = WGPUTextureAspect_All,
      .usage = WGPUTextureUsage_RenderAttachment,
  };
  const WGPUTextureView view = wgpuTextureCreateView(tex.texture, &viewDesc);
  passViews_.push_back(view);
  return view;
}

void lvk::WebGPUCommandBuffer::cmdBeginRendering(const RenderPass& renderPass, const Framebuffer& fb, const Dependencies&) {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT_MSG(!pass_, "cmdBeginRendering() was called twice without cmdEndRendering()");

  const uint32_t numColorAttachments = fb.getNumColorAttachments();

  LVK_ASSERT_MSG(renderPass.getNumColorAttachments() >= numColorAttachments,
                 "The render pass has fewer color attachments than the framebuffer");
  const bool isLayered = renderPass.layerCount > 1 || renderPass.viewMask;
  if (isLayered) {
    LVK_WEBGPU_WARN_ONCE("cmdBeginRendering(): WebGPU has no layered or multiview rendering, only layer 0 / view 0 is rendered\n");
  }

  WGPURenderPassColorAttachment colorAttachments[LVK_MAX_COLOR_ATTACHMENTS] = {};

  for (uint32_t i = 0; i != numColorAttachments; i++) {
    const RenderPass::AttachmentDesc& desc = renderPass.color[i];
    const WebGPUTexture* tex = ctx_->texturesPool_.get(fb.color[i].texture);
    const WebGPUTexture* resolveTex = ctx_->texturesPool_.get(fb.color[i].resolveTexture);
    LVK_ASSERT_MSG(tex && tex->view, "Invalid color attachment %u", i);
    if (!tex) {
      continue;
    }
    // the layer of a 3D texture is its depth slice
    const bool is3D = tex->type == TextureType_3D;
    colorAttachments[i] = {
        .view = getAttachmentView(*tex, desc.layer, desc.level),
        .depthSlice = is3D ? desc.layer : WGPU_DEPTH_SLICE_UNDEFINED,
        .resolveTarget = resolveTex ? getAttachmentView(*resolveTex, desc.layer, desc.level) : nullptr,
        .loadOp = loadOpToWGPU(desc.loadOp),
        .storeOp = storeOpToWGPU(desc.storeOp),
        .clearValue = clearColorToWGPU(desc.clearColor, tex->format),
    };
  }

  const WebGPUTexture* depthTex = ctx_->texturesPool_.get(fb.depthStencil.texture);
  const bool isDepth = depthTex && hasDepth(depthTex->format);
  const bool isStencil = depthTex && hasStencil(depthTex->format);

  // WebGPU requires undefined load/store ops for the aspects a format does not have
  const WGPURenderPassDepthStencilAttachment depthStencilAttachment = {
      .view = depthTex ? getAttachmentView(*depthTex, renderPass.depth.layer, renderPass.depth.level) : nullptr,
      .depthLoadOp = isDepth ? loadOpToWGPU(renderPass.depth.loadOp) : WGPULoadOp_Undefined,
      .depthStoreOp = isDepth ? storeOpToWGPU(renderPass.depth.storeOp) : WGPUStoreOp_Undefined,
      .depthClearValue = renderPass.depth.clearDepth,
      .stencilLoadOp = isStencil ? loadOpToWGPU(renderPass.stencil.loadOp) : WGPULoadOp_Undefined,
      .stencilStoreOp = isStencil ? storeOpToWGPU(renderPass.stencil.storeOp) : WGPUStoreOp_Undefined,
      .stencilClearValue = renderPass.stencil.clearStencil,
  };

  const WGPURenderPassDescriptor desc = {
      .label = toStringView(fb.debugName),
      .colorAttachmentCount = numColorAttachments,
      .colorAttachments = colorAttachments,
      .depthStencilAttachment = depthTex ? &depthStencilAttachment : nullptr,
  };

  pass_ = wgpuCommandEncoderBeginRenderPass(encoder_, &desc);

  // group 0 holds the arenas; draws with push constants in a buffer rebind it
  bindRenderPassArenas();
  if (ctx_->resourceTable_) {
    wgpuRenderPassEncoderSetResourceTable(pass_, ctx_->resourceTable_);
  }

  if (pushConstantsSize_) {
    setRenderPassImmediates(0, pushConstants_, pushConstantsSize_);
  }

  for (uint32_t i = 0; i != LVK_ARRAY_NUM_ELEMENTS(vertexBuffers_); i++) {
    if (vertexBuffers_[i].buffer) {
      wgpuRenderPassEncoderSetVertexBuffer(pass_, i, vertexBuffers_[i].buffer, vertexBuffers_[i].offset, vertexBuffers_[i].size);
    }
  }
  if (indexBuffer_.buffer) {
    wgpuRenderPassEncoderSetIndexBuffer(pass_, indexBuffer_.buffer, indexFormat_, indexBuffer_.offset, indexBuffer_.size);
  }

  // the defaults of the Vulkan backend's cmdBeginRendering(); the depth bias is kept
  dynamicState_ = {
      .depthBiasConstant = dynamicState_.depthBiasConstant,
      .depthBiasSlope = dynamicState_.depthBiasSlope,
      .depthBiasClamp = dynamicState_.depthBiasClamp,
  };
  lastBoundPipeline_ = nullptr;
}

void lvk::WebGPUCommandBuffer::bindRenderPassArenas() {
  const uint32_t dynamicOffset = 0;
  wgpuRenderPassEncoderSetBindGroup(pass_, 0, ctx_->arenasBindGroupReadOnly_, 1, &dynamicOffset);
  boundArenasVersion_ = ctx_->arenasVersion_;
  isPushConstantsSlotBound_ = false;
}

void lvk::WebGPUCommandBuffer::cmdEndRendering() {
  LVK_ASSERT_MSG(pass_, "cmdEndRendering() was called without cmdBeginRendering()");
  if (!pass_) {
    return;
  }
  wgpuRenderPassEncoderEnd(pass_);
  wgpuRenderPassEncoderRelease(pass_);
  pass_ = nullptr;
  currentPipeline_ = {};
  lastBoundPipeline_ = nullptr;
  for (WGPUTextureView view : passViews_) {
    wgpuTextureViewRelease(view);
  }
  passViews_.clear();
}

void lvk::WebGPUCommandBuffer::cmdNextSubpass() {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdBindViewport(const Viewport& viewport) {
  // a new render pass has a full-framebuffer viewport and scissor
  if (!pass_) {
    return;
  }
  wgpuRenderPassEncoderSetViewport(pass_, viewport.x, viewport.y, viewport.width, viewport.height, viewport.minDepth, viewport.maxDepth);
}

void lvk::WebGPUCommandBuffer::cmdBindScissorRect(const ScissorRect& rect) {
  if (!pass_) {
    return;
  }
  wgpuRenderPassEncoderSetScissorRect(pass_, rect.x, rect.y, rect.width, rect.height);
}

void lvk::WebGPUCommandBuffer::cmdBindRenderPipeline(RenderPipelineHandle handle) {
  LVK_ASSERT(pass_);
  LVK_ASSERT(!handle.empty());
  // the next draw binds the variant for this state
  currentPipeline_ = handle;
}

bool lvk::WebGPUCommandBuffer::flushRenderPipeline() {
  LVK_ASSERT_MSG(pass_, "Draw commands must be called inside a render pass");
  LVK_ASSERT_MSG(!currentPipeline_.empty(), "No render pipeline is bound");
  if (!pass_ || currentPipeline_.empty()) {
    return false;
  }
  const WGPURenderPipeline pipeline = ctx_->getRenderPipeline(currentPipeline_, dynamicState_);
  if (!pipeline) {
    return false;
  }
  if (pipeline != lastBoundPipeline_) {
    wgpuRenderPassEncoderSetPipeline(pass_, pipeline);
    lastBoundPipeline_ = pipeline;
  }
  // a buffer created during this render pass can be in a new arena
  if (boundArenasVersion_ != ctx_->arenasVersion_) {
    bindRenderPassArenas();
  }
  const WebGPURenderPipelineState* rps = ctx_->renderPipelinesPool_.get(currentPipeline_);
  // every render pipeline has the same group 0 layout: the slot stays bound until the push constants change
  const bool needsPushConstantsSlot = rps->hasPushConstantsBuffer && !(hasPushConstantsSlot_ && isPushConstantsSlotBound_);
  if (needsPushConstantsSlot) {
    uint32_t offset = 0;
    const WebGPUPushConstantsChunk& chunk = getPushConstantsSlot(&offset);
    wgpuRenderPassEncoderSetBindGroup(pass_, 0, chunk.bindGroupReadOnly, 1, &offset);
    isPushConstantsSlotBound_ = true;
  }
  return true;
}

const lvk::WebGPUPushConstantsChunk& lvk::WebGPUCommandBuffer::getPushConstantsSlot(uint32_t* outOffset) {
  if (!hasPushConstantsSlot_) {
    const bool isChunkFull = pushConstantsChunks_.empty() || pushConstantsChunks_.back().numUsedSlots == kPushConstantsSlotsPerChunk;
    if (isChunkFull) {
      pushConstantsChunks_.push_back(ctx_->acquirePushConstantsChunk());
    }
    WebGPUPushConstantsChunk& chunk = pushConstantsChunks_.back();
    chunk.data.resize(size_t(chunk.numUsedSlots + 1) * kPushConstantsSlotSize);
    memcpy(chunk.data.data() + size_t(chunk.numUsedSlots) * kPushConstantsSlotSize, pushConstants_, sizeof(pushConstants_));
    chunk.numUsedSlots++;
    hasPushConstantsSlot_ = true;
    isPushConstantsSlotBound_ = false;
  }
  WebGPUPushConstantsChunk& chunk = pushConstantsChunks_.back();
  // an arena may have been added since the chunk was acquired
  ctx_->updatePushConstantsChunkBindGroups(chunk);
  *outOffset = (chunk.numUsedSlots - 1) * kPushConstantsSlotSize;
  return chunk;
}

void lvk::WebGPUCommandBuffer::setRenderPassImmediates(uint32_t offset, const void* data, uint32_t size) const {
  const uint32_t maxSize = ctx_->limits_.maxImmediateSize;
  if (offset >= maxSize) {
    return;
  }
  wgpuRenderPassEncoderSetImmediates(pass_, offset, data, offset + size > maxSize ? maxSize - offset : size);
}

void lvk::WebGPUCommandBuffer::cmdBindDepthState(const DepthState& state) {
  // the depth state is baked into pipelines: the next draw binds a variant
  dynamicState_.depthCompareOp = state.compareOp;
  dynamicState_.isDepthWriteEnabled = state.isDepthWriteEnabled;
  if (state.isDepthBoundsTestEnabled) {
    LVK_WEBGPU_WARN_ONCE("cmdBindDepthState(): WebGPU has no depth bounds test\n");
  }
}

void lvk::WebGPUCommandBuffer::cmdBindStencilState(const StencilState& state) {
  dynamicState_.isStencilEnabled = state.enable;
  dynamicState_.stencilFront = state.enable ? state.front : StencilFaceState{};
  dynamicState_.stencilBack = state.enable ? state.back : StencilFaceState{};
  if (!state.enable) {
    return;
  }
  const bool hasSameFaceParams = state.front.readMask == state.back.readMask && state.front.writeMask == state.back.writeMask &&
                                 state.front.reference == state.back.reference;
  if (!hasSameFaceParams) {
    LVK_WEBGPU_WARN_ONCE(
        "cmdBindStencilState(): WebGPU has one stencil reference and one pair of masks for both faces: using the front face\n");
  }
  if (pass_) {
    wgpuRenderPassEncoderSetStencilReference(pass_, state.front.reference);
  }
}

void lvk::WebGPUCommandBuffer::cmdBindVertexBuffer(uint32_t index, BufferHandle buffer, uint64_t bufferOffset, uint64_t bufferSize) {
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(buffer);
  LVK_ASSERT(buf && index < LVK_ARRAY_NUM_ELEMENTS(vertexBuffers_));
  if (!buf || index >= LVK_ARRAY_NUM_ELEMENTS(vertexBuffers_)) {
    return;
  }
  // a buffer can be a range of an arena: never bind past its end
  vertexBuffers_[index] = {
      .buffer = buf->buffer,
      .offset = buf->offset + bufferOffset,
      .size = bufferSize == LVK_WHOLE_SIZE ? buf->size - bufferOffset : bufferSize,
  };
  if (pass_) {
    wgpuRenderPassEncoderSetVertexBuffer(
        pass_, index, vertexBuffers_[index].buffer, vertexBuffers_[index].offset, vertexBuffers_[index].size);
  }
}

void lvk::WebGPUCommandBuffer::cmdBindIndexBuffer(BufferHandle indexBuffer,
                                                  IndexFormat indexFormat,
                                                  uint64_t bufferOffset,
                                                  uint64_t bufferSize) {
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(indexBuffer);
  LVK_ASSERT(buf);
  LVK_ASSERT_MSG(indexFormat != IndexFormat_UI8, "WebGPU does not support 8-bit indices");
  if (!buf || indexFormat == IndexFormat_UI8) {
    return;
  }
  indexBuffer_ = {
      .buffer = buf->buffer,
      .offset = buf->offset + bufferOffset,
      .size = bufferSize == LVK_WHOLE_SIZE ? buf->size - bufferOffset : bufferSize,
  };
  indexFormat_ = indexFormat == IndexFormat_UI16 ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
  if (pass_) {
    wgpuRenderPassEncoderSetIndexBuffer(pass_, indexBuffer_.buffer, indexFormat_, indexBuffer_.offset, indexBuffer_.size);
  }
}

void lvk::WebGPUCommandBuffer::cmdPushConstants(const void* data, size_t size, size_t offset) {
  LVK_ASSERT_MSG(size % 4 == 0 && offset % 4 == 0, "WebGPU immediates need 4-byte aligned sizes and offsets");
  LVK_ASSERT_MSG(offset + size <= sizeof(pushConstants_), "Push constants are too large");
  if (!data || offset + size > sizeof(pushConstants_)) {
    return;
  }

  memcpy(pushConstants_ + offset, data, size);
  pushConstantsSize_ = uint32_t(offset + size) > pushConstantsSize_ ? uint32_t(offset + size) : pushConstantsSize_;
  // the next draw or dispatch of a pipeline with push constants in a buffer copies them into a new slot
  hasPushConstantsSlot_ = false;

  // compute passes get the immediates in beginComputePass()
  if (pass_) {
    setRenderPassImmediates(uint32_t(offset), data, uint32_t(size));
  }
}

void lvk::WebGPUCommandBuffer::cmdCopyBuffer(BufferHandle srcBuffer,
                                             BufferHandle dstBuffer,
                                             size_t srcOffset,
                                             size_t dstOffset,
                                             size_t size) {
  LVK_ASSERT_MSG(!pass_, "cmdCopyBuffer() cannot be called inside a render pass");
  LVK_ASSERT_MSG(srcOffset % 4 == 0 && dstOffset % 4 == 0 && size % 4 == 0, "WebGPU copies need 4-byte aligned offsets and sizes");
  const WebGPUBuffer* src = ctx_->buffersPool_.get(srcBuffer);
  const WebGPUBuffer* dst = ctx_->buffersPool_.get(dstBuffer);
  LVK_ASSERT(src && dst);
  if (!src || !dst) {
    return;
  }
  if (src->buffer == dst->buffer) {
    // WebGPU cannot copy within one buffer: go through a scratch buffer
    const WGPUBuffer scratch = ctx_->getCopyScratchBuffer(size);
    wgpuCommandEncoderCopyBufferToBuffer(encoder_, src->buffer, src->offset + srcOffset, scratch, 0, size);
    wgpuCommandEncoderCopyBufferToBuffer(encoder_, scratch, 0, dst->buffer, dst->offset + dstOffset, size);
  } else {
    wgpuCommandEncoderCopyBufferToBuffer(encoder_, src->buffer, src->offset + srcOffset, dst->buffer, dst->offset + dstOffset, size);
  }
  markHostVisibleBufferWritten(dstBuffer, *dst);
}

void lvk::WebGPUCommandBuffer::markHostVisibleBufferWritten(BufferHandle handle, const WebGPUBuffer& buf) {
  if (buf.storage != StorageType_HostVisible || isHostVisibleBufferWritten(handle)) {
    return;
  }
  writtenHostVisibleBuffers_.push_back(handle);
}

bool lvk::WebGPUCommandBuffer::isHostVisibleBufferWritten(BufferHandle handle) const {
  for (BufferHandle written : writtenHostVisibleBuffers_) {
    if (written == handle) {
      return true;
    }
  }
  return false;
}

void lvk::WebGPUCommandBuffer::cmdFillBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, uint32_t data) {
  LVK_ASSERT_MSG(!pass_, "cmdFillBuffer() cannot be called inside a render pass");
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(buffer);
  LVK_ASSERT(buf);
  if (!buf) {
    return;
  }
  const uint64_t fillSize = size == LVK_WHOLE_SIZE ? buf->size - bufferOffset : size;
  markHostVisibleBufferWritten(buffer, *buf);
  if (!data) {
    wgpuCommandEncoderClearBuffer(encoder_, buf->buffer, buf->offset + bufferOffset, fillSize);
    return;
  }
  // WebGPU only clears to zero: other values are written
  const std::vector<uint32_t> values(size_t(fillSize / sizeof(uint32_t)), data);
  writeBuffer(buf->buffer, buf->offset + bufferOffset, values.data(), values.size() * sizeof(uint32_t));
}

void lvk::WebGPUCommandBuffer::writeBuffer(WGPUBuffer buffer, uint64_t offset, const void* data, uint64_t size) {
  // a staging copy: wgpuCommandEncoderWriteBuffer() is Dawn-only
  const WGPUBufferDescriptor desc = {
      .label = toStringView("Buffer: write staging"),
      .usage = WGPUBufferUsage_CopySrc,
      .size = alignUp(size, 4),
      .mappedAtCreation = true,
  };
  const WGPUBuffer staging = wgpuDeviceCreateBuffer(ctx_->device_, &desc);
  void* const mapped = staging ? wgpuBufferGetMappedRange(staging, 0, desc.size) : nullptr;
  if (!mapped) {
    LLOGW("writeBuffer(): failed to map a staging buffer\n");
    if (staging) {
      wgpuBufferRelease(staging);
    }
    return;
  }
  memcpy(mapped, data, size);
  wgpuBufferUnmap(staging);
  wgpuCommandEncoderCopyBufferToBuffer(encoder_, staging, 0, buffer, offset, size);
  // the command buffer keeps it alive until the copy is done
  wgpuBufferRelease(staging);
}

void lvk::WebGPUCommandBuffer::cmdUpdateBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, const void* data) {
  LVK_ASSERT_MSG(!pass_, "cmdUpdateBuffer() cannot be called inside a render pass");
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(buffer);
  LVK_ASSERT(buf);
  if (!buf) {
    return;
  }
  writeBuffer(buf->buffer, buf->offset + bufferOffset, data, size);
  markHostVisibleBufferWritten(buffer, *buf);
}

void lvk::WebGPUCommandBuffer::cmdDraw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t baseInstance) {
  if (!vertexCount || !flushRenderPipeline()) {
    return;
  }
  wgpuRenderPassEncoderDraw(pass_, vertexCount, instanceCount, firstVertex, baseInstance);
}

void lvk::WebGPUCommandBuffer::cmdDrawIndexed(uint32_t indexCount,
                                              uint32_t instanceCount,
                                              uint32_t firstIndex,
                                              int32_t vertexOffset,
                                              uint32_t baseInstance) {
  if (!indexCount || !flushRenderPipeline()) {
    return;
  }
  wgpuRenderPassEncoderDrawIndexed(pass_, indexCount, instanceCount, firstIndex, vertexOffset, baseInstance);
}

void lvk::WebGPUCommandBuffer::cmdDrawIndirect(BufferHandle indirectBuffer,
                                               size_t indirectBufferOffset,
                                               uint32_t drawCount,
                                               uint32_t stride) {
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(indirectBuffer);
  LVK_ASSERT(buf);
  if (!buf || !flushRenderPipeline()) {
    return;
  }
  // WebGPU has no multi-draw
  const uint32_t drawStride = stride ? stride : 4 * sizeof(uint32_t);
  for (uint32_t i = 0; i != drawCount; i++) {
    wgpuRenderPassEncoderDrawIndirect(pass_, buf->buffer, buf->offset + indirectBufferOffset + uint64_t(i) * drawStride);
  }
}

void lvk::WebGPUCommandBuffer::cmdDrawIndexedIndirect(BufferHandle indirectBuffer,
                                                      size_t indirectBufferOffset,
                                                      uint32_t drawCount,
                                                      uint32_t stride) {
  const WebGPUBuffer* buf = ctx_->buffersPool_.get(indirectBuffer);
  LVK_ASSERT(buf);
  if (!buf || !flushRenderPipeline()) {
    return;
  }
  // WebGPU has no multi-draw
  const uint32_t drawStride = stride ? stride : 5 * sizeof(uint32_t);
  for (uint32_t i = 0; i != drawCount; i++) {
    wgpuRenderPassEncoderDrawIndexedIndirect(pass_, buf->buffer, buf->offset + indirectBufferOffset + uint64_t(i) * drawStride);
  }
}

void lvk::WebGPUCommandBuffer::cmdDrawIndexedIndirectCount(BufferHandle, size_t, BufferHandle, size_t, uint32_t, uint32_t) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdDrawMeshTasks(const Dimensions&) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdDrawMeshTasksIndirect(BufferHandle, size_t, uint32_t, uint32_t) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdDrawMeshTasksIndirectCount(BufferHandle, size_t, BufferHandle, size_t, uint32_t, uint32_t) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdTraceRays(uint32_t, uint32_t, uint32_t, const Dependencies&) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdSetBlendColor(const float color[4]) {
  LVK_ASSERT(pass_);
  const WGPUColor blendColor = {color[0], color[1], color[2], color[3]};
  wgpuRenderPassEncoderSetBlendConstant(pass_, &blendColor);
}

void lvk::WebGPUCommandBuffer::cmdSetDepthBias(float constantFactor, float slopeFactor, float clamp) {
  // baked into render pipelines like the depth state
  dynamicState_.depthBiasConstant = constantFactor;
  dynamicState_.depthBiasSlope = slopeFactor;
  dynamicState_.depthBiasClamp = clamp;
}

void lvk::WebGPUCommandBuffer::cmdSetDepthBiasEnable(bool enable) {
  dynamicState_.isDepthBiasEnabled = enable;
}

void lvk::WebGPUCommandBuffer::cmdSetDepthBounds(float minDepthBounds, float maxDepthBounds) {
  // [0, 1] is the default range of the Vulkan backend
  if (minDepthBounds != 0.0f || maxDepthBounds != 1.0f) {
    LVK_WEBGPU_NOT_IMPLEMENTED();
  }
}

void lvk::WebGPUCommandBuffer::cmdSetPrimitiveRestartEnable(bool enable) {
  if (enable) {
    LVK_WEBGPU_NOT_IMPLEMENTED();
  }
}

void lvk::WebGPUCommandBuffer::cmdSetFragmentShadingRate(const Dimensions&, ShadingRateCombinerOp, ShadingRateCombinerOp) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdResetQueryPool(QueryPoolHandle, uint32_t, uint32_t) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdWriteTimestamp(QueryPoolHandle, uint32_t) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdClearColorImage(TextureHandle, const ClearColorValue&, const TextureLayers&) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdCopyImage(TextureHandle src,
                                            TextureHandle dst,
                                            const Dimensions& extent,
                                            const Offset3D& srcOffset,
                                            const Offset3D& dstOffset,
                                            const TextureLayers& srcLayers,
                                            const TextureLayers& dstLayers) {
  LVK_ASSERT_MSG(!pass_, "cmdCopyImage() cannot be called inside a render pass");
  const WebGPUTexture* srcTex = ctx_->texturesPool_.get(src);
  const WebGPUTexture* dstTex = ctx_->texturesPool_.get(dst);
  LVK_ASSERT(srcTex && dstTex);
  if (!srcTex || !dstTex) {
    return;
  }
  LVK_ASSERT_MSG(srcLayers.numLayers == dstLayers.numLayers, "cmdCopyImage(): layer counts must match");
  // 3D textures address slices with the Z offset, array textures address layers with it
  const bool isSrc3D = srcTex->type == TextureType_3D;
  const bool isDst3D = dstTex->type == TextureType_3D;
  const WGPUTexelCopyTextureInfo srcInfo = {
      .texture = srcTex->texture,
      .mipLevel = srcTex->baseLevel + srcLayers.mipLevel,
      .origin = {uint32_t(srcOffset.x), uint32_t(srcOffset.y), isSrc3D ? uint32_t(srcOffset.z) : srcTex->baseLayer + srcLayers.layer},
      .aspect = WGPUTextureAspect_All,
  };
  const WGPUTexelCopyTextureInfo dstInfo = {
      .texture = dstTex->texture,
      .mipLevel = dstTex->baseLevel + dstLayers.mipLevel,
      .origin = {uint32_t(dstOffset.x), uint32_t(dstOffset.y), isDst3D ? uint32_t(dstOffset.z) : dstTex->baseLayer + dstLayers.layer},
      .aspect = WGPUTextureAspect_All,
  };
  const WGPUExtent3D copySize = {extent.width, extent.height, isSrc3D ? extent.depth : srcLayers.numLayers};
  wgpuCommandEncoderCopyTextureToTexture(encoder_, &srcInfo, &dstInfo, &copySize);
}

void lvk::WebGPUCommandBuffer::cmdGenerateMipmap(TextureHandle handle) {
  LVK_ASSERT_MSG(!pass_, "cmdGenerateMipmap() cannot be called inside a render pass");
  const WebGPUTexture* tex = ctx_->texturesPool_.get(handle);
  LVK_ASSERT(tex);
  if (!tex || pass_) {
    return;
  }
  ctx_->generateMipmap(encoder_, *tex);
}

void lvk::WebGPUCommandBuffer::cmdUpdateTLAS(AccelStructHandle, BufferHandle) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdUpdateBLAS(const ldr::Span<AccelStructHandle>&) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdCopyTensor(TensorHandle, TensorHandle) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

void lvk::WebGPUCommandBuffer::cmdDispatchDataGraph(DataGraphPipelineHandle,
                                                    const ldr::Span<const TensorHandle>&,
                                                    const ldr::Span<const TensorHandle>&,
                                                    const Dependencies&) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
}

lvk::WebGPUContext::WebGPUContext(const ContextConfig& config, const WGPUChainedStruct* surfaceSource, uint32_t width, uint32_t height)
: config_(config) {
  LVK_PROFILER_FUNCTION();

  glslang_initialize_process();

  const WGPUInstanceFeatureName instanceFeatures[] = {WGPUInstanceFeatureName_TimedWaitAny};
  const WGPUInstanceDescriptor instanceDesc = {
      .requiredFeatureCount = LVK_ARRAY_NUM_ELEMENTS(instanceFeatures),
      .requiredFeatures = instanceFeatures,
  };
  instance_ = wgpuCreateInstance(&instanceDesc);
  if (!instance_) {
    LLOGW("wgpuCreateInstance() failed\n");
    return;
  }

  if (surfaceSource) {
    // not const: WGPUSurfaceDescriptor::nextInChain is a non-const pointer
    WGPUSurfaceDescriptor surfaceDesc = {.nextInChain = const_cast<WGPUChainedStruct*>(surfaceSource)};
    surface_ = wgpuInstanceCreateSurface(instance_, &surfaceDesc);
    if (!surface_) {
      LLOGW("wgpuInstanceCreateSurface() failed\n");
      return;
    }
  }

  // Dawn's D3D12 backend has the resource table only with DXC
  adapter_ = requestAdapter(instance_, surface_, WGPUBackendType_Vulkan);
  if (!adapter_) {
    adapter_ = requestAdapter(instance_, surface_, WGPUBackendType_Undefined);
  }
  if (!adapter_) {
    LLOGW("No WebGPU adapter found\n");
    return;
  }

  WGPUAdapterInfo info = {};
  wgpuAdapterGetInfo(adapter_, &info);
  LLOGL("WebGPU adapter: %.*s (%.*s)\n", int(info.device.length), info.device.data, int(info.description.length), info.description.data);
  wgpuAdapterInfoFreeMembers(info);

  // request everything the adapter can do
  WGPULimits adapterLimits = {};
  wgpuAdapterGetLimits(adapter_, &adapterLimits);

  const WGPUFeatureName optionalFeatures[] = {
      WGPUFeatureName_ChromiumExperimentalSamplingResourceTable,
      WGPUFeatureName_TextureCompressionBC,
      WGPUFeatureName_TextureCompressionETC2,
      WGPUFeatureName_TextureCompressionASTC,
      WGPUFeatureName_TextureComponentSwizzle,
      WGPUFeatureName_Float32Filterable,
      WGPUFeatureName_Float32Blendable,
      WGPUFeatureName_Depth32FloatStencil8,
      WGPUFeatureName_DepthClipControl,
      WGPUFeatureName_DualSourceBlending,
      WGPUFeatureName_IndirectFirstInstance,
      WGPUFeatureName_RG11B10UfloatRenderable,
      WGPUFeatureName_BGRA8UnormStorage,
      WGPUFeatureName_ShaderF16,
      WGPUFeatureName_ClipDistances,
  };
  WGPUFeatureName requiredFeatures[LVK_ARRAY_NUM_ELEMENTS(optionalFeatures)] = {};
  uint32_t numRequiredFeatures = 0;
  for (WGPUFeatureName feature : optionalFeatures) {
    if (wgpuAdapterHasFeature(adapter_, feature)) {
      requiredFeatures[numRequiredFeatures++] = feature;
    }
  }

  WGPUDawnTogglesDescriptor deviceToggles = {
      .chain = {.sType = WGPUSType_DawnTogglesDescriptor},
      .enabledToggleCount = LVK_ARRAY_NUM_ELEMENTS(kDawnToggles),
      .enabledToggles = kDawnToggles,
  };
  const WGPUDeviceDescriptor deviceDesc = {
      .nextInChain = &deviceToggles.chain,
      .label = toStringView("LVK WebGPU device"),
      .requiredFeatureCount = numRequiredFeatures,
      .requiredFeatures = requiredFeatures,
      .requiredLimits = &adapterLimits,
      .deviceLostCallbackInfo = {.mode = WGPUCallbackMode_AllowSpontaneous, .callback = onDeviceLost},
      .uncapturedErrorCallbackInfo = {.callback = onUncapturedError, .userdata1 = &config_.terminateOnValidationError},
  };
  const WGPURequestDeviceCallbackInfo callbackInfo = {
      .mode = WGPUCallbackMode_WaitAnyOnly,
      .callback =
          [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void*) {
            if (status != WGPURequestDeviceStatus_Success) {
              LLOGW("WebGPU device request failed: %.*s\n", int(message.length), message.data);
            }
            *static_cast<WGPUDevice*>(userdata1) = status == WGPURequestDeviceStatus_Success ? device : nullptr;
          },
      .userdata1 = &device_,
  };
  WGPUFutureWaitInfo waitInfo = {.future = wgpuAdapterRequestDevice(adapter_, &deviceDesc, callbackInfo)};
  wgpuInstanceWaitAny(instance_, 1, &waitInfo, UINT64_MAX);
  if (!device_) {
    return;
  }

  wgpuDeviceGetLimits(device_, &limits_);
  queue_ = wgpuDeviceGetQueue(device_);

  hasTextureComponentSwizzle_ = wgpuDeviceHasFeature(device_, WGPUFeatureName_TextureComponentSwizzle);

  if (wgpuDeviceHasFeature(device_, WGPUFeatureName_ChromiumExperimentalSamplingResourceTable)) {
    const WGPUResourceTableDescriptor tableDesc = {
        .label = toStringView("Resource table: bindless textures and samplers"),
        .size = kResourceTableSize,
    };
    resourceTable_ = wgpuDeviceCreateResourceTable(device_, &tableDesc);
  } else {
    LLOGW("WebGPU: chromium_experimental_sampling_resource_table is not supported, bindless textures will not work\n");
  }

  // the arenas: read-only for render pipelines (vertex shaders cannot write storage buffers), read-write for compute
  {
    // the arenas, then the push constants buffer (see WebGPUPushConstantsChunk)
    WGPUBindGroupLayoutEntry readOnlyEntries[kNumArenas + 1] = {};
    WGPUBindGroupLayoutEntry readWriteEntries[kNumArenas + 1] = {};
    readOnlyEntries[kNumArenas] = {
        .binding = kNumArenas,
        .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
        .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .hasDynamicOffset = true, .minBindingSize = kPushConstantsSlotSize},
    };
    readWriteEntries[kNumArenas] = {
        .binding = kNumArenas,
        .visibility = WGPUShaderStage_Compute,
        .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage, .hasDynamicOffset = true, .minBindingSize = kPushConstantsSlotSize},
    };
    for (uint32_t i = 0; i != kNumArenas; i++) {
      readOnlyEntries[i] = {
          .binding = i,
          .visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment,
          .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
      };
      readWriteEntries[i] = {
          .binding = i,
          .visibility = WGPUShaderStage_Compute,
          .buffer = {.type = WGPUBufferBindingType_Storage},
      };
    }
    const WGPUBindGroupLayoutDescriptor readOnlyDesc = {
        .label = toStringView("Bind group layout: arenas (read-only)"),
        .entryCount = kNumArenas + 1,
        .entries = readOnlyEntries,
    };
    const WGPUBindGroupLayoutDescriptor readWriteDesc = {
        .label = toStringView("Bind group layout: arenas (read-write)"),
        .entryCount = kNumArenas + 1,
        .entries = readWriteEntries,
    };
    arenasLayoutReadOnly_ = wgpuDeviceCreateBindGroupLayout(device_, &readOnlyDesc);
    arenasLayoutReadWrite_ = wgpuDeviceCreateBindGroupLayout(device_, &readWriteDesc);

    const WGPUBufferDescriptor dummyDesc = {
        .label = toStringView("Buffer: dummy arena"),
        .usage = WGPUBufferUsage_Storage,
        .size = kArenaAlignment,
    };
    for (WGPUBuffer& buffer : dummyArenaBuffers_) {
      buffer = wgpuDeviceCreateBuffer(device_, &dummyDesc);
    }
    const WGPUBufferDescriptor scratchDesc = {
        .label = toStringView("Buffer: cmdDispatchIndirect() arguments"),
        .usage = WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst,
        .size = 4 * sizeof(uint32_t),
    };
    indirectScratchBuffer_ = wgpuDeviceCreateBuffer(device_, &scratchDesc);
    const WGPUBufferDescriptor dummyPushConstantsDesc = {
        .label = toStringView("Buffer: dummy push constants"),
        .usage = WGPUBufferUsage_Storage,
        .size = kPushConstantsSlotSize,
    };
    dummyPushConstantsBuffer_ = wgpuDeviceCreateBuffer(device_, &dummyPushConstantsDesc);

    updateArenaBindGroups();
  }

  // index 0 is "no texture", like in the Vulkan backend
  const uint32_t pixel = 0xFF000000;
  const TextureHandle dummyTexture = createTexture(
                                         {
                                             .format = Format_RGBA_UN8,
                                             .dimensions = {1, 1, 1},
                                             .usage = TextureUsageBits_Sampled,
                                             .data = &pixel,
                                         },
                                         "Dummy 1x1 (black)",
                                         nullptr)
                                         .release();
  const SamplerHandle dummySampler = createSampler({.mipMap = SamplerMip_Linear, .debugName = "Sampler: default"}, nullptr).release();
  LVK_ASSERT(dummyTexture.index() == 0 && dummySampler.index() == 0);

  // without a surface, a size asks for a headless swapchain
  if (surface_ || (width && height)) {
    if (!initSwapchain(width, height).isOk()) {
      LLOGW("Failed to configure the WebGPU swapchain\n");
    }
  }
}

lvk::WebGPUContext::~WebGPUContext() {
  LVK_PROFILER_FUNCTION();

  if (device_) {
    waitIdle();
  }

  commandBuffers_.clear();

  for (WebGPUPushConstantsChunk& chunk : freePushConstantsChunks_) {
    releasePushConstantsChunk(chunk);
  }
  if (dummyPushConstantsBuffer_) {
    wgpuBufferDestroy(dummyPushConstantsBuffer_);
    wgpuBufferRelease(dummyPushConstantsBuffer_);
  }

  releaseSwapchainTexture();
  texturesPool_.destroy(swapchainTexture_);
  swapchainTexture_ = {};
  if (headlessSwapchain_) {
    wgpuTextureRelease(headlessSwapchain_);
  }
  if (retainedFrame_) {
    wgpuTextureRelease(retainedFrame_);
  }

  // whatever the application did not destroy
  for (WebGPUShaderModuleState& sm : shaderModulesPool_.objects_) {
    if (sm.module) {
      wgpuShaderModuleRelease(sm.module);
    }
  }
  for (WebGPURenderPipelineState& rps : renderPipelinesPool_.objects_) {
    for (WebGPURenderPipelineVariant& variant : rps.variants) {
      if (variant.pipeline) {
        wgpuRenderPipelineRelease(variant.pipeline);
      }
    }
    if (rps.layout) {
      wgpuPipelineLayoutRelease(rps.layout);
    }
    if (rps.vert) {
      wgpuShaderModuleRelease(rps.vert);
    }
    if (rps.frag) {
      wgpuShaderModuleRelease(rps.frag);
    }
  }
  for (WebGPUComputePipelineState& cps : computePipelinesPool_.objects_) {
    if (cps.pipeline) {
      wgpuComputePipelineRelease(cps.pipeline);
    }
    if (cps.layout) {
      wgpuPipelineLayoutRelease(cps.layout);
    }
  }
  // buffers in arenas are released with their arenas
  for (WebGPUBuffer& buf : buffersPool_.objects_) {
    if (buf.buffer && !buf.arena) {
      wgpuBufferDestroy(buf.buffer);
      wgpuBufferRelease(buf.buffer);
    }
  }
  for (WebGPUArena& arena : arenas_) {
    if (arena.buffer) {
      wgpuBufferDestroy(arena.buffer);
      wgpuBufferRelease(arena.buffer);
    }
  }
  for (WGPUBindGroup bindGroup : {arenasBindGroupReadOnly_, arenasBindGroupReadWrite_}) {
    if (bindGroup) {
      wgpuBindGroupRelease(bindGroup);
    }
  }
  for (WGPUBindGroupLayout layout : {arenasLayoutReadOnly_, arenasLayoutReadWrite_}) {
    if (layout) {
      wgpuBindGroupLayoutRelease(layout);
    }
  }
  for (WGPUBuffer buffer : dummyArenaBuffers_) {
    if (buffer) {
      wgpuBufferDestroy(buffer);
      wgpuBufferRelease(buffer);
    }
  }
  if (indirectScratchBuffer_) {
    wgpuBufferDestroy(indirectScratchBuffer_);
    wgpuBufferRelease(indirectScratchBuffer_);
  }
  if (copyScratchBuffer_) {
    wgpuBufferDestroy(copyScratchBuffer_);
    wgpuBufferRelease(copyScratchBuffer_);
  }
  for (WebGPUTexture& tex : texturesPool_.objects_) {
    releaseTexture(tex);
  }
  for (WebGPUSampler& smp : samplersPool_.objects_) {
    if (smp.sampler) {
      wgpuSamplerRelease(smp.sampler);
    }
  }
  for (MipmapPipeline& mp : mipmapPipelines_) {
    wgpuRenderPipelineRelease(mp.pipeline);
  }
  if (mipmapShader_) {
    wgpuShaderModuleRelease(mipmapShader_);
  }
  if (mipmapSampler_) {
    wgpuSamplerRelease(mipmapSampler_);
  }
  if (resourceTable_) {
    wgpuResourceTableDestroy(resourceTable_);
    wgpuResourceTableRelease(resourceTable_);
  }

  destroySlangGlobalSessionWGSL(slangGlobalSession_);
  glslang_finalize_process();

  if (surface_) {
    if (device_) {
      wgpuSurfaceUnconfigure(surface_);
    }
    wgpuSurfaceRelease(surface_);
  }
  if (queue_) {
    wgpuQueueRelease(queue_);
  }
  if (device_) {
    wgpuDeviceDestroy(device_);
    wgpuDeviceRelease(device_);
  }
  if (adapter_) {
    wgpuAdapterRelease(adapter_);
  }
  if (instance_) {
    // let Dawn deliver the device lost callback before the instance goes away
    wgpuInstanceProcessEvents(instance_);
    wgpuInstanceRelease(instance_);
  }
}

lvk::ICommandBuffer& lvk::WebGPUContext::acquireCommandBuffer(bool dedicatedCompute) {
  LVK_PROFILER_FUNCTION();

  LVK_ASSERT_MSG(!dedicatedCompute, "WebGPU has a single queue");

  commandBuffers_.push_back(std::make_unique<WebGPUCommandBuffer>(this));

  return *commandBuffers_.back();
}

lvk::SubmitHandle lvk::WebGPUContext::submit(ICommandBuffer& commandBuffer, TextureHandle present, const ldr::Span<TextureHandle>&) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_SUBMIT);

  WebGPUCommandBuffer* cmdBuffer = static_cast<WebGPUCommandBuffer*>(&commandBuffer);

  LVK_ASSERT_MSG(!cmdBuffer->pass_, "submit() was called inside a render pass");

  const uint64_t submitValue = ++lastSubmitValue_;

  // queue writes are ordered before the next submit
  flushHostVisibleBuffers();
  // and after the previous submits: the chunks can be reused right away
  for (WebGPUPushConstantsChunk& chunk : cmdBuffer->pushConstantsChunks_) {
    wgpuQueueWriteBuffer(queue_, chunk.buffer, 0, chunk.data.data(), size_t(chunk.numUsedSlots) * kPushConstantsSlotSize);
    freePushConstantsChunks_.push_back(std::move(chunk));
  }
  cmdBuffer->pushConstantsChunks_.clear();
  readBackHostVisibleBuffers(*cmdBuffer, submitValue);

  const bool isPresenting = present && present == swapchainTexture_ && isSwapchainTextureAcquired_;
  const bool isFrameRetained = isPresenting && isPresentedFrameRetained_ && surface_;
  if (isFrameRetained) {
    retainPresentedFrame(cmdBuffer->encoder_);
  }

  const WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(cmdBuffer->encoder_, nullptr);
  wgpuQueueSubmit(queue_, 1, &cmd);
  wgpuCommandBufferRelease(cmd);

  if (!pendingReadbacks_.empty() && pendingReadbacks_.back().submitValue == submitValue) {
    PendingReadback& readback = pendingReadbacks_.back();
    const WGPUBufferMapCallbackInfo mapInfo = {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback =
            [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
              if (status != WGPUMapAsyncStatus_Success) {
                LLOGW("Mapped buffers emulation: wgpuBufferMapAsync() failed: %.*s\n", int(message.length), message.data);
              }
              WebGPUContext* ctx = static_cast<WebGPUContext*>(userdata1);
              const uint64_t value = uint64_t(reinterpret_cast<uintptr_t>(userdata2));
              for (PendingReadback& rb : ctx->pendingReadbacks_) {
                if (rb.submitValue == value) {
                  rb.status = status == WGPUMapAsyncStatus_Success ? PendingReadback::Status_Mapped : PendingReadback::Status_Failed;
                }
              }
            },
        .userdata1 = this,
        .userdata2 = reinterpret_cast<void*>(uintptr_t(submitValue)),
    };
    readback.future = wgpuBufferMapAsync(readback.staging, WGPUMapMode_Read, 0, readback.stagingSize, mapInfo);
  }

  const WGPUQueueWorkDoneCallbackInfo workDoneInfo = {
      .mode = WGPUCallbackMode_AllowProcessEvents,
      .callback =
          [](WGPUQueueWorkDoneStatus status, WGPUStringView, void* userdata1, void* userdata2) {
            if (status != WGPUQueueWorkDoneStatus_Success) {
              return;
            }
            WebGPUContext* ctx = static_cast<WebGPUContext*>(userdata1);
            const uint64_t value = uint64_t(reinterpret_cast<uintptr_t>(userdata2));
            ctx->completedSubmitValue_ = value > ctx->completedSubmitValue_ ? value : ctx->completedSubmitValue_;
          },
      .userdata1 = this,
      .userdata2 = reinterpret_cast<void*>(uintptr_t(submitValue)),
  };
  wgpuQueueOnSubmittedWorkDone(queue_, workDoneInfo);

  if (isPresenting) {
    LVK_PROFILER_ZONE("WebGPU present", LVK_PROFILER_COLOR_PRESENT);
    if (surface_) {
      wgpuSurfacePresent(surface_);
    }
    LVK_PROFILER_ZONE_END();
    releaseSwapchainTexture();
  }

  wgpuInstanceProcessEvents(instance_);

  for (size_t i = 0; i != commandBuffers_.size(); i++) {
    if (commandBuffers_[i].get() == cmdBuffer) {
      commandBuffers_.erase(commandBuffers_.begin() + i);
      break;
    }
  }

  processDeferredSlots();
  processReadbacks(0);

  SubmitHandle handle;
  handle.value_ = submitValue;
  return handle;
}

void lvk::WebGPUContext::wait(SubmitHandle handle) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_WAIT);

  if (!handle.empty() && isReady(handle)) {
    return;
  }

  // WebGPU can only wait for all the work submitted so far
  waitIdle();
}

bool lvk::WebGPUContext::isReady(SubmitHandle handle) const {
  if (handle.empty()) {
    return true;
  }
  wgpuInstanceProcessEvents(instance_);
  if (completedSubmitValue_ < handle.value_) {
    return false;
  }
  // the CPU copies of host-visible buffers get what this work wrote
  const_cast<WebGPUContext*>(this)->processReadbacks(handle.value_);
  return true;
}

void lvk::WebGPUContext::waitIdle() {
  const WGPUQueueWorkDoneCallbackInfo workDoneInfo = {
      .mode = WGPUCallbackMode_WaitAnyOnly,
      .callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void*, void*) {},
  };
  WGPUFutureWaitInfo waitInfo = {.future = wgpuQueueOnSubmittedWorkDone(queue_, workDoneInfo)};
  wgpuInstanceWaitAny(instance_, 1, &waitInfo, UINT64_MAX);
  // deliver the callbacks of the individual submits
  wgpuInstanceProcessEvents(instance_);
  processDeferredSlots();
  processReadbacks(lastSubmitValue_);
}

WGPUBuffer lvk::WebGPUContext::getCopyScratchBuffer(uint64_t size) {
  if (size <= copyScratchBufferSize_) {
    return copyScratchBuffer_;
  }
  uint64_t newSize = 1024 * 1024;
  while (newSize < size) {
    newSize *= 2;
  }
  // command encoders keep a reference to the old buffer they use
  if (copyScratchBuffer_) {
    wgpuBufferRelease(copyScratchBuffer_);
  }
  const WGPUBufferDescriptor desc = {
      .label = toStringView("Buffer: copy scratch"),
      .usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
      .size = newSize,
  };
  copyScratchBuffer_ = wgpuDeviceCreateBuffer(device_, &desc);
  copyScratchBufferSize_ = copyScratchBuffer_ ? newSize : 0;
  return copyScratchBuffer_;
}

void lvk::WebGPUContext::flushHostVisibleBuffers() {
  LVK_PROFILER_FUNCTION();

  // nearby changed words are written together
  enum { kMaxGapWords = 64 };

  for (BufferHandle handle : hostVisibleBuffers_) {
    WebGPUBuffer* buf = buffersPool_.get(handle);
    uint32_t* mapped = buf->mappedData.data();
    uint32_t* known = buf->gpuData.data();
    const size_t numWords = buf->mappedData.size();
    size_t i = 0;
    while (i != numWords) {
      if (mapped[i] == known[i]) {
        i++;
        continue;
      }
      size_t end = i + 1;
      for (size_t j = end; j != numWords && j - end < kMaxGapWords; j++) {
        if (mapped[j] != known[j]) {
          end = j + 1;
        }
      }
      const size_t numBytes = (end - i) * sizeof(uint32_t);
      wgpuQueueWriteBuffer(queue_, buf->buffer, buf->offset + i * sizeof(uint32_t), mapped + i, numBytes);
      memcpy(known + i, mapped + i, numBytes);
      i = end;
    }
  }
}

void lvk::WebGPUContext::readBackHostVisibleBuffers(WebGPUCommandBuffer& cmdBuffer, uint64_t submitValue) {
  LVK_PROFILER_FUNCTION();

  PendingReadback readback = {.submitValue = submitValue};

  for (BufferHandle handle : hostVisibleBuffers_) {
    const WebGPUBuffer* buf = buffersPool_.get(handle);
    // render pipelines get read-only arenas
    const bool isWrittenByDispatch = cmdBuffer.hasDispatches_ && (buf->usage & BufferUsageBits_Storage);
    const bool isWritten = isWrittenByDispatch || cmdBuffer.isHostVisibleBufferWritten(handle);
    if (!isWritten) {
      continue;
    }
    readback.entries.push_back({
        .buffer = handle,
        .stagingOffset = readback.stagingSize,
        .snapshot = buf->gpuData,
    });
    readback.stagingSize += buf->size;
  }

  if (readback.entries.empty()) {
    return;
  }

  const WGPUBufferDescriptor stagingDesc = {
      .label = toStringView("Buffer: mapped buffers readback"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
      .size = readback.stagingSize,
  };
  readback.staging = wgpuDeviceCreateBuffer(device_, &stagingDesc);
  if (!readback.staging) {
    LLOGW("Mapped buffers emulation: cannot create a readback buffer of %llu bytes\n", (unsigned long long)readback.stagingSize);
    return;
  }

  for (const PendingReadback::Entry& entry : readback.entries) {
    const WebGPUBuffer* buf = buffersPool_.get(entry.buffer);
    wgpuCommandEncoderCopyBufferToBuffer(cmdBuffer.encoder_, buf->buffer, buf->offset, readback.staging, entry.stagingOffset, buf->size);
  }

  pendingReadbacks_.push_back(std::move(readback));
}

void lvk::WebGPUContext::processReadbacks(uint64_t waitSubmitValue) {
  // in submit order: mergeReadback() corrects the snapshots of the later submits
  while (!pendingReadbacks_.empty()) {
    if (pendingReadbacks_.front().status == PendingReadback::Status_Pending) {
      if (pendingReadbacks_.front().submitValue > waitSubmitValue) {
        break;
      }
      WGPUFutureWaitInfo waitInfo = {.future = pendingReadbacks_.front().future};
      wgpuInstanceWaitAny(instance_, 1, &waitInfo, UINT64_MAX);
    }
    const PendingReadback& readback = pendingReadbacks_.front();
    if (readback.status == PendingReadback::Status_Mapped) {
      mergeOldestReadback();
      wgpuBufferUnmap(readback.staging);
    }
    wgpuBufferDestroy(readback.staging);
    wgpuBufferRelease(readback.staging);
    pendingReadbacks_.erase(pendingReadbacks_.begin());
  }
}

void lvk::WebGPUContext::mergeOldestReadback() {
  const PendingReadback& readback = pendingReadbacks_.front();
  const uint32_t* data = static_cast<const uint32_t*>(wgpuBufferGetConstMappedRange(readback.staging, 0, readback.stagingSize));
  if (!data) {
    return;
  }
  for (const PendingReadback::Entry& entry : readback.entries) {
    WebGPUBuffer* buf = buffersPool_.get(entry.buffer);
    if (!buf) {
      // destroyed while the work was in flight
      continue;
    }
    const uint32_t* gpu = data + entry.stagingOffset / sizeof(uint32_t);
    const uint32_t* snapshot = entry.snapshot.data();
    const size_t numWords = entry.snapshot.size();
    // only the words this work changed and no later submit wrote
    for (size_t i = 0; i != numWords; i++) {
      const bool isChangedByGPU = gpu[i] != snapshot[i];
      const bool isOverwrittenSince = buf->gpuData[i] != snapshot[i];
      if (isChangedByGPU && !isOverwrittenSince) {
        buf->mappedData[i] = gpu[i];
        buf->gpuData[i] = gpu[i];
      }
    }
    // the snapshots of the later submits predate these changes
    for (size_t r = 1; r != pendingReadbacks_.size(); r++) {
      for (PendingReadback::Entry& later : pendingReadbacks_[r].entries) {
        if (later.buffer != entry.buffer) {
          continue;
        }
        for (size_t i = 0; i != numWords; i++) {
          const bool isChangedByGPU = gpu[i] != snapshot[i];
          const bool isWrittenBeforeLater = later.snapshot[i] != snapshot[i];
          if (isChangedByGPU && !isWrittenBeforeLater) {
            later.snapshot[i] = gpu[i];
          }
        }
      }
    }
  }
}

void lvk::WebGPUContext::processDeferredSlots() {
  for (size_t i = 0; i != deferredSlots_.size();) {
    const DeferredSlot& slot = deferredSlots_[i];
    if (slot.submitValue > completedSubmitValue_) {
      i++;
      continue;
    }
    texturesPool_.destroy(slot.texture);
    samplersPool_.destroy(slot.sampler);
    if (slot.arena) {
      freeToArena(slot.arena, slot.arenaOffset, slot.arenaSize);
    }
    deferredSlots_[i] = deferredSlots_.back();
    deferredSlots_.pop_back();
  }
}

void lvk::WebGPUContext::updateResourceTable(uint32_t slot, const WGPUBindingResource& resource) const {
  if (!resourceTable_) {
    return;
  }
  if (wgpuResourceTableUpdate(resourceTable_, slot, &resource) != WGPUStatus_Success) {
    LLOGW("wgpuResourceTableUpdate() failed for slot %u: the slot is still in use by the GPU\n", slot);
  }
}

lvk::TextureHandle lvk::WebGPUContext::createTextureInResourceTable(WebGPUTexture&& tex) {
  const TextureHandle handle = texturesPool_.create(std::move(tex));
  if (handle.index() >= kSamplersSlotOffset) {
    LLOGW("Too many textures for the resource table: texture %u is not accessible from shaders\n", handle.index());
  } else if (const WGPUTextureView sampledView = texturesPool_.get(handle)->sampledView) {
    updateResourceTable(handle.index(), {.textureView = sampledView});
  }
  return handle;
}

void lvk::WebGPUContext::submitEncoder(WGPUCommandEncoder encoder) const {
  const WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
  wgpuQueueSubmit(queue_, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(encoder);
}

void lvk::WebGPUContext::updateArenaBindGroups() {
  if (arenasBindGroupReadOnly_) {
    wgpuBindGroupRelease(arenasBindGroupReadOnly_);
  }
  if (arenasBindGroupReadWrite_) {
    wgpuBindGroupRelease(arenasBindGroupReadWrite_);
  }

  arenasBindGroupReadOnly_ = createArenasBindGroup(false, dummyPushConstantsBuffer_, "Bind group: arenas (read-only)");
  arenasBindGroupReadWrite_ = createArenasBindGroup(true, dummyPushConstantsBuffer_, "Bind group: arenas (read-write)");
  // the push constant chunks have their own copies
  arenasVersion_++;
}

WGPUBindGroup lvk::WebGPUContext::createArenasBindGroup(bool isCompute, WGPUBuffer pushConstants, const char* debugName) const {
  // arena `i + 1` is at binding `i`; arenas which do not exist yet are a small dummy buffer
  WGPUBindGroupEntry entries[kNumArenas + 1] = {};
  for (uint32_t i = 0; i != kNumArenas; i++) {
    entries[i] = {
        .binding = i,
        .buffer = arenas_[i].buffer ? arenas_[i].buffer : dummyArenaBuffers_[i],
        .offset = 0,
        .size = WGPU_WHOLE_SIZE,
    };
  }
  // one slot, selected with the dynamic offset
  entries[kNumArenas] = {
      .binding = kNumArenas,
      .buffer = pushConstants,
      .offset = 0,
      .size = kPushConstantsSlotSize,
  };
  const WGPUBindGroupDescriptor desc = {
      .label = toStringView(debugName),
      .layout = isCompute ? arenasLayoutReadWrite_ : arenasLayoutReadOnly_,
      .entryCount = kNumArenas + 1,
      .entries = entries,
  };
  return wgpuDeviceCreateBindGroup(device_, &desc);
}

lvk::WebGPUPushConstantsChunk lvk::WebGPUContext::acquirePushConstantsChunk() {
  if (!freePushConstantsChunks_.empty()) {
    WebGPUPushConstantsChunk chunk = std::move(freePushConstantsChunks_.back());
    freePushConstantsChunks_.pop_back();
    chunk.numUsedSlots = 0;
    updatePushConstantsChunkBindGroups(chunk);
    return chunk;
  }
  const WGPUBufferDescriptor desc = {
      .label = toStringView("Buffer: push constants"),
      .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
      .size = uint64_t(kPushConstantsSlotSize) * kPushConstantsSlotsPerChunk,
  };
  WebGPUPushConstantsChunk chunk = {
      .buffer = wgpuDeviceCreateBuffer(device_, &desc),
  };
  chunk.data.reserve(size_t(desc.size));
  updatePushConstantsChunkBindGroups(chunk);
  return chunk;
}

void lvk::WebGPUContext::updatePushConstantsChunkBindGroups(WebGPUPushConstantsChunk& chunk) const {
  if (chunk.bindGroupReadOnly && chunk.arenasVersion == arenasVersion_) {
    return;
  }
  // command encoders keep a reference to the bind groups they use
  if (chunk.bindGroupReadOnly) {
    wgpuBindGroupRelease(chunk.bindGroupReadOnly);
    wgpuBindGroupRelease(chunk.bindGroupReadWrite);
  }
  chunk.bindGroupReadOnly = createArenasBindGroup(false, chunk.buffer, "Bind group: arenas + push constants (read-only)");
  chunk.bindGroupReadWrite = createArenasBindGroup(true, chunk.buffer, "Bind group: arenas + push constants (read-write)");
  chunk.arenasVersion = arenasVersion_;
}

void lvk::WebGPUContext::releasePushConstantsChunk(WebGPUPushConstantsChunk& chunk) const {
  if (chunk.bindGroupReadOnly) {
    wgpuBindGroupRelease(chunk.bindGroupReadOnly);
    wgpuBindGroupRelease(chunk.bindGroupReadWrite);
  }
  if (chunk.buffer) {
    wgpuBufferRelease(chunk.buffer);
  }
  chunk = {};
}

bool lvk::WebGPUContext::allocateFromArena(uint64_t size, uint32_t* outArena, uint64_t* outOffset, const char* debugName) {
  // a reasonable default; bigger buffers get an arena of their own size
  const uint64_t kDefaultArenaSize = 256ull * 1024ull * 1024ull;

  const uint64_t alignedSize = alignUp(size, kArenaAlignment);
  // the arena has to fit into one storage buffer binding, and offsets into the low 32 bits of a GPU address
  const uint64_t maxBindingSize = limits_.maxStorageBufferBindingSize < limits_.maxBufferSize ? limits_.maxStorageBufferBindingSize
                                                                                              : limits_.maxBufferSize;
  const uint64_t maxArenaSize = (maxBindingSize < 0xFFFFFFFFull ? maxBindingSize : 0xFFFFFFFFull) & ~uint64_t(kArenaAlignment - 1);

  if (alignedSize > maxArenaSize) {
    LLOGW("Buffer `%s` (%llu bytes) is larger than the largest WebGPU storage buffer binding (%llu bytes)\n",
          debugName,
          (unsigned long long)size,
          (unsigned long long)maxArenaSize);
    return false;
  }

  for (uint32_t i = 0; i != kNumArenas; i++) {
    WebGPUArena& arena = arenas_[i];

    if (!arena.buffer) {
      const uint64_t arenaSize = alignedSize > kDefaultArenaSize ? alignedSize : kDefaultArenaSize;
      const uint64_t clampedSize = arenaSize < maxArenaSize ? arenaSize : maxArenaSize;
      char label[64] = {};
      snprintf(label, sizeof(label), "Buffer: arena %u", i + 1);
      // every kind of buffer with a GPU address can live in an arena
      const WGPUBufferDescriptor bufferDesc = {
          .label = toStringView(label),
          .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex | WGPUBufferUsage_Index | WGPUBufferUsage_Indirect |
                   WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
          .size = clampedSize,
      };
      arena = {
          .buffer = wgpuDeviceCreateBuffer(device_, &bufferDesc),
          .size = clampedSize,
          .freeRanges = {{.offset = 0, .size = clampedSize}},
      };
      if (!arena.buffer) {
        arena = {};
        return false;
      }
      LLOGL("WebGPU: created arena %u (%llu MB)\n", i + 1, (unsigned long long)(clampedSize >> 20));
      updateArenaBindGroups();
    }

    // first fit
    for (size_t r = 0; r != arena.freeRanges.size(); r++) {
      WebGPUArena::Range& range = arena.freeRanges[r];
      if (range.size < alignedSize) {
        continue;
      }
      *outArena = i + 1;
      *outOffset = range.offset;
      range.offset += alignedSize;
      range.size -= alignedSize;
      if (!range.size) {
        arena.freeRanges.erase(arena.freeRanges.begin() + r);
      }
      return true;
    }
  }

  LLOGW("Buffer `%s` (%llu bytes): all %u WebGPU arenas are full\n", debugName, (unsigned long long)size, uint32_t(kNumArenas));
  return false;
}

void lvk::WebGPUContext::freeToArena(uint32_t arena, uint64_t offset, uint64_t size) {
  LVK_ASSERT(arena && arena <= kNumArenas);
  std::vector<WebGPUArena::Range>& ranges = arenas_[arena - 1].freeRanges;
  const uint64_t alignedSize = alignUp(size, kArenaAlignment);

  // keep the free list sorted by offset and merge adjacent ranges
  size_t pos = 0;
  while (pos != ranges.size() && ranges[pos].offset < offset) {
    pos++;
  }
  ranges.insert(ranges.begin() + pos, {.offset = offset, .size = alignedSize});
  if (pos + 1 != ranges.size() && ranges[pos].offset + ranges[pos].size == ranges[pos + 1].offset) {
    ranges[pos].size += ranges[pos + 1].size;
    ranges.erase(ranges.begin() + pos + 1);
  }
  if (pos && ranges[pos - 1].offset + ranges[pos - 1].size == ranges[pos].offset) {
    ranges[pos - 1].size += ranges[pos].size;
    ranges.erase(ranges.begin() + pos);
  }
}

WGPUPipelineLayout lvk::WebGPUContext::createPipelineLayout(bool isCompute, uint32_t immediateSize, const char* debugName) const {
  // bindless textures and samplers come from the resource table
  WGPUPipelineLayoutResourceTable layoutResourceTable = {
      .chain = {.sType = WGPUSType_PipelineLayoutResourceTable},
      .usesResourceTable = resourceTable_ != nullptr,
  };
  const WGPUBindGroupLayout arenasLayout = isCompute ? arenasLayoutReadWrite_ : arenasLayoutReadOnly_;
  const WGPUPipelineLayoutDescriptor layoutDesc = {
      .nextInChain = resourceTable_ ? &layoutResourceTable.chain : nullptr,
      .label = toStringView(debugName),
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &arenasLayout,
      .immediateSize = immediateSize,
  };
  return wgpuDeviceCreatePipelineLayout(device_, &layoutDesc);
}

lvk::Holder<lvk::BufferHandle> lvk::WebGPUContext::createBuffer(const BufferDesc& desc, const char* debugName, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const char* name = debugName && *debugName ? debugName : desc.debugName;

  if (!desc.size || !desc.usage) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Buffer size or usage is zero");
    return {};
  }

  // WebGPU writes and copies multiples of 4 bytes
  const uint64_t size = alignUp(desc.size, 4);

  // buffers with a GPU address live in the arenas
  const bool hasGpuAddress = desc.usage & (BufferUsageBits_Uniform | BufferUsageBits_Storage | BufferUsageBits_Indirect);

  const bool isHostVisible = desc.storage == StorageType_HostVisible;

  // the initial data padded to 4 bytes, also the CPU copy of host-visible buffers (see WebGPUBuffer)
  std::vector<uint32_t> data(isHostVisible || desc.data ? size_t(size / sizeof(uint32_t)) : 0, 0);
  if (desc.data) {
    memcpy(data.data(), desc.data, desc.size);
  }

  uint32_t arena = 0;
  uint64_t offset = 0;
  WGPUBuffer buffer = nullptr;

  if (hasGpuAddress) {
    if (!allocateFromArena(size, &arena, &offset, name)) {
      Result::setResult(outResult, Result::Code::RuntimeError, "Cannot allocate the buffer from the WebGPU arenas");
      return {};
    }
    buffer = arenas_[arena - 1].buffer;
    // a reused arena range has old contents: match the CPU copy
    if (!data.empty()) {
      wgpuQueueWriteBuffer(queue_, buffer, offset, data.data(), size);
    }
  } else {
    // vertex and index buffers without a GPU address are standalone
    const WGPUBufferUsage usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                                  (desc.usage & BufferUsageBits_Index ? WGPUBufferUsage_Index : WGPUBufferUsage_None) |
                                  (desc.usage & BufferUsageBits_Vertex ? WGPUBufferUsage_Vertex : WGPUBufferUsage_None);
    const WGPUBufferDescriptor bufferDesc = {
        .label = toStringView(name),
        .usage = usage,
        .size = size,
        .mappedAtCreation = !data.empty(),
    };
    buffer = wgpuDeviceCreateBuffer(device_, &bufferDesc);
    if (!buffer) {
      Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateBuffer() failed");
      return {};
    }
    if (!data.empty()) {
      if (void* dst = wgpuBufferGetMappedRange(buffer, 0, size)) {
        memcpy(dst, data.data(), size);
      }
      wgpuBufferUnmap(buffer);
    }
  }

  Result::setResult(outResult, Result());

  const BufferHandle handle = buffersPool_.create({
      .buffer = buffer,
      .offset = offset,
      .size = size,
      .arena = arena,
      .usage = desc.usage,
      .storage = desc.storage,
      .mappedData = isHostVisible ? data : std::vector<uint32_t>{},
      .gpuData = isHostVisible ? std::move(data) : std::vector<uint32_t>{},
  });

  if (isHostVisible) {
    hostVisibleBuffers_.push_back(handle);
  }

  return {this, handle};
}

lvk::Holder<lvk::SamplerHandle> lvk::WebGPUContext::createSampler(const SamplerStateDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  // WebGPU allows anisotropic filtering only with linear filters everywhere
  const bool isAnisotropic = desc.maxAnisotropic > 1 && desc.minFilter == SamplerFilter_Linear && desc.magFilter == SamplerFilter_Linear &&
                             desc.mipMap == SamplerMip_Linear;

  const WGPUSamplerDescriptor samplerDesc = {
      .label = toStringView(desc.debugName),
      .addressModeU = addressModeToWGPU(desc.wrapU),
      .addressModeV = addressModeToWGPU(desc.wrapV),
      .addressModeW = addressModeToWGPU(desc.wrapW),
      .magFilter = desc.magFilter == SamplerFilter_Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
      .minFilter = desc.minFilter == SamplerFilter_Linear ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest,
      .mipmapFilter = desc.mipMap == SamplerMip_Linear ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest,
      .lodMinClamp = float(desc.mipLodMin),
      // the same as the Vulkan backend: no mip-mapping means the base level only
      .lodMaxClamp = desc.mipMap == SamplerMip_Disabled ? float(desc.mipLodMin) : float(desc.mipLodMax),
      .compare = desc.depthCompareEnabled ? compareOpToWGPU(desc.depthCompareOp) : WGPUCompareFunction_Undefined,
      .maxAnisotropy = isAnisotropic ? uint16_t(desc.maxAnisotropic < 16 ? desc.maxAnisotropic : 16) : uint16_t(1),
  };

  const WGPUSampler sampler = wgpuDeviceCreateSampler(device_, &samplerDesc);
  if (!sampler) {
    Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateSampler() failed");
    return {};
  }

  const SamplerHandle handle = samplersPool_.create({.sampler = sampler});

  if (kSamplersSlotOffset + handle.index() >= kResourceTableSize) {
    LLOGW("Too many samplers for the resource table: sampler %u is not accessible from shaders\n", handle.index());
  } else {
    updateResourceTable(kSamplersSlotOffset + handle.index(), {.sampler = sampler});
  }

  Result::setResult(outResult, Result());

  return {this, handle};
}

lvk::Result lvk::WebGPUContext::createTextureViews(WebGPUTexture& tex, const TextureViewDesc& desc, const char* debugName) const {
  const bool isDepthFormat = hasDepth(tex.format);
  const bool isStencilFormat = hasStencil(tex.format);
  const uint32_t arrayLayerCount = tex.type == TextureType_3D ? 1 : tex.numLayers;

  if (desc.aspect >= TextureAspect_Plane0) {
    return Result(Result::Code::ArgumentOutOfRange, "WebGPU has no multi-planar textures");
  }

  // the resource table takes one aspect per view: depth for depth-stencil formats, unless the stencil is asked for
  const WGPUTextureAspect sampledAspect = desc.aspect == TextureAspect_Stencil ? WGPUTextureAspect_StencilOnly
                                          : desc.aspect == TextureAspect_Depth ? WGPUTextureAspect_DepthOnly
                                          : isDepthFormat && isStencilFormat   ? WGPUTextureAspect_DepthOnly
                                                                               : WGPUTextureAspect_All;

  const WGPUTextureViewDescriptor viewDesc = {
      .label = toStringView(debugName),
      .dimension = tex.viewDimension,
      .baseMipLevel = tex.baseLevel,
      .mipLevelCount = tex.numLevels,
      .baseArrayLayer = tex.baseLayer,
      .arrayLayerCount = arrayLayerCount,
      .aspect = WGPUTextureAspect_All,
  };
  tex.view = wgpuTextureCreateView(tex.texture, &viewDesc);
  if (!tex.view) {
    return Result(Result::Code::RuntimeError, "wgpuTextureCreateView() failed");
  }

  if (!(tex.usage & TextureUsageBits_Sampled)) {
    return Result();
  }

  const bool hasSwizzle = !desc.components.identity();
  if (hasSwizzle && !hasTextureComponentSwizzle_) {
    LLOGW("Texture `%s`: component swizzle is not supported by this WebGPU device\n", debugName);
  }
  WGPUTextureComponentSwizzleDescriptor swizzle = {
      .chain = {.sType = WGPUSType_TextureComponentSwizzleDescriptor},
      .swizzle =
          {
              .r = swizzleToWGPU(desc.components.r, WGPUComponentSwizzle_R),
              .g = swizzleToWGPU(desc.components.g, WGPUComponentSwizzle_G),
              .b = swizzleToWGPU(desc.components.b, WGPUComponentSwizzle_B),
              .a = swizzleToWGPU(desc.components.a, WGPUComponentSwizzle_A),
          },
  };
  const WGPUTextureViewDescriptor sampledViewDesc = {
      .nextInChain = hasSwizzle && hasTextureComponentSwizzle_ ? &swizzle.chain : nullptr,
      .label = toStringView(debugName),
      .dimension = tex.viewDimension,
      .baseMipLevel = tex.baseLevel,
      .mipLevelCount = tex.numLevels,
      .baseArrayLayer = tex.baseLayer,
      .arrayLayerCount = arrayLayerCount,
      .aspect = sampledAspect,
      .usage = WGPUTextureUsage_TextureBinding,
  };
  tex.sampledView = wgpuTextureCreateView(tex.texture, &sampledViewDesc);
  if (!tex.sampledView) {
    return Result(Result::Code::RuntimeError, "wgpuTextureCreateView() failed");
  }

  return Result();
}

void lvk::WebGPUContext::releaseTexture(WebGPUTexture& tex) const {
  if (tex.sampledView) {
    wgpuTextureViewRelease(tex.sampledView);
  }
  if (tex.view) {
    wgpuTextureViewRelease(tex.view);
  }
  if (tex.texture && !tex.isSwapchainTexture) {
    // WebGPU frees the memory once the submitted work using the texture is done; an unsubmitted command buffer using a
    // destroyed texture fails to submit, so then the last reference frees it
    if (tex.isOwningTexture && commandBuffers_.empty()) {
      wgpuTextureDestroy(tex.texture);
    }
    wgpuTextureRelease(tex.texture);
  }
  tex = {};
}

lvk::Holder<lvk::TextureHandle> lvk::WebGPUContext::createTexture(const TextureDesc& desc, const char* debugName, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const char* name = debugName && *debugName ? debugName : desc.debugName;
  const WGPUTextureFormat format = formatToWGPU(desc.format);

  if (format == WGPUTextureFormat_Undefined) {
    LLOGW("Texture `%s`: format %u is not supported by WebGPU\n", name, unsigned(desc.format));
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Texture format is not supported by WebGPU");
    return {};
  }
  if (!desc.dimensions.width || !desc.dimensions.height || !desc.dimensions.depth || !desc.numLayers || !desc.numMipLevels) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Texture dimensions, layers and mip-levels cannot be zero");
    return {};
  }
  if (desc.numSamples > 1 && (desc.numMipLevels > 1 || desc.type != TextureType_2D)) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Multisampled textures must be 2D with one mip-level");
    return {};
  }

  const bool isDepthStencil = hasDepth(desc.format) || hasStencil(desc.format);
  const bool isMultisampled = desc.numSamples > 1;
  const uint32_t numSamples = getSampleCountWGPU(desc.numSamples);
  if (numSamples != desc.numSamples) {
    LLOGW("Texture `%s`: WebGPU supports only 1 and 4 samples, %u samples become %u\n", name, desc.numSamples, numSamples);
  }
  // generateMipmap() renders each level from the previous one
  const bool isMipmapRenderable = desc.numMipLevels > 1 && desc.type != TextureType_3D && isMipmapGenerationSupported(format);
  const bool isStorage = (desc.usage & TextureUsageBits_Storage) && isStorageSupported(format);

  if ((desc.usage & TextureUsageBits_Storage) && !isStorage) {
    LLOGW("Texture `%s`: the format cannot be a WebGPU storage texture, TextureUsageBits_Storage is ignored\n", name);
  }
  if (desc.usage & (TextureUsageBits_InputAttachment | TextureUsageBits_FragmentDensityMap | TextureUsageBits_ShadingRateAttachment)) {
    LLOGW("Texture `%s`: input attachments, fragment density maps and shading rate attachments are not supported by WebGPU\n", name);
  }
  if (desc.storage == StorageType_HostVisible) {
    LLOGW("Texture `%s`: host-visible textures are not supported by WebGPU, use upload()\n", name);
  }

  // upload(), download() and cmdCopyImage() need single-sampled color textures to be copyable
  const bool needsTextureBinding = (desc.usage & TextureUsageBits_Sampled) || isMipmapRenderable;
  const bool needsRenderAttachment = (desc.usage & TextureUsageBits_Attachment) || isMipmapRenderable;
  const bool isCopyable = !isDepthStencil && !isMultisampled;
  const WGPUTextureUsage usage = (needsTextureBinding ? WGPUTextureUsage_TextureBinding : WGPUTextureUsage_None) |
                                 (isStorage ? WGPUTextureUsage_StorageBinding : WGPUTextureUsage_None) |
                                 (needsRenderAttachment ? WGPUTextureUsage_RenderAttachment : WGPUTextureUsage_None) |
                                 (isCopyable ? WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst : WGPUTextureUsage_None);

  // array layers: 6 per cube map
  const uint32_t numLayers = desc.type == TextureType_Cube ? 6 * desc.numLayers : desc.type == TextureType_3D ? 1 : desc.numLayers;

  const WGPUTextureDescriptor textureDesc = {
      .label = toStringView(name),
      .usage = usage,
      .dimension = desc.type == TextureType_3D ? WGPUTextureDimension_3D : WGPUTextureDimension_2D,
      .size = {desc.dimensions.width, desc.dimensions.height, desc.type == TextureType_3D ? desc.dimensions.depth : numLayers},
      .format = format,
      .mipLevelCount = desc.numMipLevels,
      .sampleCount = numSamples,
  };
  const WGPUTexture texture = wgpuDeviceCreateTexture(device_, &textureDesc);
  if (!texture) {
    Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateTexture() failed");
    return {};
  }

  WebGPUTexture tex = {
      .texture = texture,
      .viewDimension = getViewDimension(desc.type, numLayers),
      .format = desc.format,
      .type = desc.type,
      .extent = desc.dimensions,
      .numLevels = desc.numMipLevels,
      .numLayers = numLayers,
      .numSamples = numSamples,
      .usage = desc.usage,
  };

  const Result viewsResult = createTextureViews(tex, {.type = desc.type, .components = desc.components}, name);
  if (!viewsResult.isOk()) {
    releaseTexture(tex);
    Result::setResult(outResult, viewsResult);
    return {};
  }

  const TextureHandle handle = createTextureInResourceTable(std::move(tex));

  Holder<TextureHandle> holder = {this, handle};

  if (desc.data) {
    LVK_ASSERT(desc.type == TextureType_2D || desc.type == TextureType_Cube);
    LVK_ASSERT(desc.dataNumMipLevels <= desc.numMipLevels);
    const Result result = upload(handle,
                                 {
                                     .dimensions = desc.dimensions,
                                     .numLayers = desc.type == TextureType_Cube ? 6u : 1u,
                                     .numMipLevels = desc.dataNumMipLevels,
                                 },
                                 desc.data,
                                 0);
    if (!result.isOk()) {
      Result::setResult(outResult, result);
      return {};
    }
    if (desc.generateMipmaps) {
      const WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
      generateMipmap(encoder, *texturesPool_.get(handle));
      submitEncoder(encoder);
    }
  }

  Result::setResult(outResult, Result());

  return holder;
}

lvk::Holder<lvk::TextureHandle> lvk::WebGPUContext::createTextureView(TextureHandle texture,
                                                                      const TextureViewDesc& desc,
                                                                      const char* debugName,
                                                                      Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const WebGPUTexture* src = texturesPool_.get(texture);

  if (!src || !src->texture || src->isSwapchainTexture) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Invalid texture");
    return {};
  }
  if (desc.mipLevel + desc.numMipLevels > src->numLevels || desc.layer + desc.numLayers > src->numLayers) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "The view is out of the texture's range");
    return {};
  }

  // destroying the source texture still frees its memory, like in Vulkan
  wgpuTextureAddRef(src->texture);

  const uint32_t baseLevel = src->baseLevel + desc.mipLevel;

  WebGPUTexture tex = {
      .texture = src->texture,
      .viewDimension = getViewDimension(desc.type, desc.numLayers),
      .format = src->format,
      .type = desc.type,
      .extent = {.width = src->extent.width >> desc.mipLevel ? src->extent.width >> desc.mipLevel : 1,
                 .height = src->extent.height >> desc.mipLevel ? src->extent.height >> desc.mipLevel : 1,
                 .depth = src->extent.depth >> desc.mipLevel ? src->extent.depth >> desc.mipLevel : 1},
      .baseLevel = baseLevel,
      .baseLayer = src->baseLayer + desc.layer,
      .numLevels = desc.numMipLevels,
      .numLayers = desc.numLayers,
      .numSamples = src->numSamples,
      .usage = src->usage,
      .isOwningTexture = false,
  };

  const Result viewsResult = createTextureViews(tex, desc, debugName);
  if (!viewsResult.isOk()) {
    releaseTexture(tex);
    Result::setResult(outResult, viewsResult);
    return {};
  }

  const TextureHandle handle = createTextureInResourceTable(std::move(tex));

  Result::setResult(outResult, Result());

  return {this, handle};
}

lvk::Holder<lvk::ComputePipelineHandle> lvk::WebGPUContext::createComputePipeline(const ComputePipelineDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const WebGPUShaderModuleState* comp = shaderModulesPool_.get(desc.smComp);

  if (!comp || comp->stage != Stage_Comp) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Missing compute shader");
    return {};
  }
  // rounded up like a WGSL struct size, as for render pipelines
  const uint32_t immediateSize = comp->pushConstants == PushConstantsLocation_Immediates ? uint32_t(alignUp(comp->pushConstantsSize, 16))
                                                                                         : 0;
  const WGPUPipelineLayout layout = createPipelineLayout(true, immediateSize, desc.debugName);

  ConstantEntries constants;
  getConstantEntries(getPipelineConstants(desc.specInfo, comp, desc.debugName), constants);

  const WGPUComputePipelineDescriptor pipelineDesc = {
      .label = toStringView(desc.debugName),
      .layout = layout,
      .compute =
          {
              .module = comp->module,
              .entryPoint = toStringView(nullptr),
              .constantCount = constants.count,
              .constants = constants.entries,
          },
  };
  const WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device_, &pipelineDesc);
  if (!pipeline) {
    wgpuPipelineLayoutRelease(layout);
    Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateComputePipeline() failed");
    return {};
  }

  Result::setResult(outResult, Result());

  return {this,
          computePipelinesPool_.create({
              .pipeline = pipeline,
              .layout = layout,
              .immediateSize = immediateSize,
              .hasPushConstantsBuffer = comp->pushConstants == PushConstantsLocation_Buffer,
          })};
}

lvk::Holder<lvk::RenderPipelineHandle> lvk::WebGPUContext::createRenderPipeline(const RenderPipelineDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  const bool hasUnsupportedStages = desc.smTesc.valid() || desc.smTese.valid() || desc.smGeom.valid() || desc.smTask.valid() ||
                                    desc.smMesh.valid();
  if (hasUnsupportedStages) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "WebGPU supports only vertex and fragment shaders");
    return {};
  }

  const WebGPUShaderModuleState* vert = shaderModulesPool_.get(desc.smVert);
  const WebGPUShaderModuleState* frag = shaderModulesPool_.get(desc.smFrag);

  if (!vert) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "Missing vertex shader");
    return {};
  }

  if (desc.polygonMode != PolygonMode_Fill) {
    LLOGW("Pipeline `%s`: WebGPU supports only solid rendering, PolygonMode_Line is drawn as PolygonMode_Fill\n", desc.debugName);
  }
  if (getSampleCountWGPU(desc.samplesCount) != desc.samplesCount) {
    LLOGW("Pipeline `%s`: WebGPU supports only 1 and 4 samples, %u samples become %u\n",
          desc.debugName,
          desc.samplesCount,
          getSampleCountWGPU(desc.samplesCount));
  }
  if (desc.topology == Topology_Patch) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "WebGPU does not support patches");
    return {};
  }

  // each stage has its push constants either in immediates or in a buffer, and both get the same data
  const uint32_t vertImmediateSize = vert->pushConstants == PushConstantsLocation_Immediates ? vert->pushConstantsSize : 0;
  const uint32_t fragImmediateSize = frag && frag->pushConstants == PushConstantsLocation_Immediates ? frag->pushConstantsSize : 0;
  // rounded up like a WGSL struct size
  const uint32_t immediateSize = uint32_t(alignUp(vertImmediateSize > fragImmediateSize ? vertImmediateSize : fragImmediateSize, 16));
  const bool hasPushConstantsBuffer = vert->pushConstants == PushConstantsLocation_Buffer ||
                                      (frag && frag->pushConstants == PushConstantsLocation_Buffer);

  const WGPUPipelineLayout layout = createPipelineLayout(false, immediateSize, desc.debugName);

  // referenced: the application can destroy the shader modules, like with the Vulkan backend
  wgpuShaderModuleAddRef(vert->module);
  if (frag) {
    wgpuShaderModuleAddRef(frag->module);
  }

  WebGPURenderPipelineState rps = {
      .desc = desc,
      .vert = vert->module,
      .frag = frag ? frag->module : nullptr,
      .layout = layout,
      .immediateSize = immediateSize,
      .hasPushConstantsBuffer = hasPushConstantsBuffer,
      .vertConstants = getPipelineConstants(desc.specInfo, vert, desc.debugName),
      .fragConstants = getPipelineConstants(desc.specInfo, frag, desc.debugName),
      .debugName = desc.debugName ? desc.debugName : "",
  };

  // the default variant is created now, so its validation errors are reported at creation
  const WGPURenderPipeline pipeline = createRenderPipelineVariant(rps, {});
  if (!pipeline) {
    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(rps.vert);
    if (rps.frag) {
      wgpuShaderModuleRelease(rps.frag);
    }
    Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateRenderPipeline() failed");
    return {};
  }
  rps.variants.push_back({.state = {}, .pipeline = pipeline});

  Result::setResult(outResult, Result());

  return {this, renderPipelinesPool_.create(std::move(rps))};
}

WGPURenderPipeline lvk::WebGPUContext::getRenderPipeline(RenderPipelineHandle handle, const WebGPUDynamicState& state) {
  WebGPURenderPipelineState* rps = renderPipelinesPool_.get(handle);
  LVK_ASSERT(rps);
  if (!rps) {
    return nullptr;
  }

  // only the state the pipeline can use is a part of the key
  const Format depthStencilFormat = rps->desc.depthFormat != Format_Invalid ? rps->desc.depthFormat : rps->desc.stencilFormat;
  const bool isDepth = hasDepth(depthStencilFormat);
  const bool isStencil = hasStencil(depthStencilFormat) && state.isStencilEnabled;
  const bool isTriangles = rps->desc.topology == Topology_Triangle || rps->desc.topology == Topology_TriangleStrip;
  const bool isDepthBias = isDepth && isTriangles && state.isDepthBiasEnabled;
  const WebGPUDynamicState key = {
      .depthCompareOp = isDepth ? state.depthCompareOp : CompareOp_AlwaysPass,
      .isDepthWriteEnabled = isDepth && state.isDepthWriteEnabled,
      .isStencilEnabled = isStencil,
      .stencilFront = isStencil ? state.stencilFront : StencilFaceState{},
      .stencilBack = isStencil ? state.stencilBack : StencilFaceState{},
      .isDepthBiasEnabled = isDepthBias,
      .depthBiasConstant = isDepthBias ? state.depthBiasConstant : 0.0f,
      .depthBiasSlope = isDepthBias ? state.depthBiasSlope : 0.0f,
      .depthBiasClamp = isDepthBias ? state.depthBiasClamp : 0.0f,
  };

  for (const WebGPURenderPipelineVariant& variant : rps->variants) {
    if (isSameDynamicState(variant.state, key)) {
      return variant.pipeline;
    }
  }

  // a failed variant stays nullptr, so it is not retried on every draw
  const WGPURenderPipeline pipeline = createRenderPipelineVariant(*rps, key);
  rps->variants.push_back({.state = key, .pipeline = pipeline});
  return pipeline;
}

WGPURenderPipeline lvk::WebGPUContext::createRenderPipelineVariant(const WebGPURenderPipelineState& rps,
                                                                   const WebGPUDynamicState& state) const {
  const RenderPipelineDesc& desc = rps.desc;

  // vertex input: attributes are grouped by their buffer bindings
  const uint32_t numAttributes = desc.vertexInput.getNumAttributes();
  const uint32_t numBindings = desc.vertexInput.getNumInputBindings();

  WGPUVertexAttribute attributes[VertexInput::LVK_VERTEX_ATTRIBUTES_MAX] = {};
  WGPUVertexBufferLayout bufferLayouts[VertexInput::LVK_VERTEX_BUFFER_MAX] = {};

  uint32_t numAttributesAdded = 0;
  for (uint32_t b = 0; b != numBindings; b++) {
    const uint32_t firstAttribute = numAttributesAdded;
    for (uint32_t a = 0; a != numAttributes; a++) {
      const VertexInput::VertexAttribute& attr = desc.vertexInput.attributes[a];
      if (attr.binding != b) {
        continue;
      }
      const WGPUVertexFormat format = vertexFormatToWGPU(attr.format);
      if (format == WGPUVertexFormat_Force32) {
        LLOGW("createRenderPipeline(): vertex format %u is not supported by WebGPU\n", unsigned(attr.format));
        return nullptr;
      }
      attributes[numAttributesAdded++] = {
          .format = format,
          .offset = attr.offset,
          .shaderLocation = attr.location,
      };
    }
    bufferLayouts[b] = {
        .stepMode = desc.vertexInput.inputBindings[b].inputRate == VertexInputRate_Instance ? WGPUVertexStepMode_Instance
                                                                                            : WGPUVertexStepMode_Vertex,
        .arrayStride = desc.vertexInput.inputBindings[b].stride,
        .attributeCount = numAttributesAdded - firstAttribute,
        .attributes = &attributes[firstAttribute],
    };
  }

  const uint32_t numColorAttachments = desc.getNumColorAttachments();

  WGPUBlendState blendStates[LVK_MAX_COLOR_ATTACHMENTS] = {};
  WGPUColorTargetState colorTargets[LVK_MAX_COLOR_ATTACHMENTS] = {};

  for (uint32_t i = 0; i != numColorAttachments; i++) {
    const ColorAttachment& attachment = desc.color[i];
    blendStates[i] = {
        .color = {.operation = blendOpToWGPU(attachment.rgbBlendOp),
                  .srcFactor = blendFactorToWGPU(attachment.srcRGBBlendFactor),
                  .dstFactor = blendFactorToWGPU(attachment.dstRGBBlendFactor)},
        .alpha = {.operation = blendOpToWGPU(attachment.alphaBlendOp),
                  .srcFactor = blendFactorToWGPU(attachment.srcAlphaBlendFactor),
                  .dstFactor = blendFactorToWGPU(attachment.dstAlphaBlendFactor)},
    };
    colorTargets[i] = {
        .format = formatToWGPU(attachment.format),
        .blend = attachment.blendEnabled ? &blendStates[i] : nullptr,
        // LVK's ColorComponentBits have the same values as WGPUColorWriteMask
        .writeMask = WGPUColorWriteMask(attachment.colorWriteMask),
    };
  }

  // the dynamic depth-stencil state of LVK, baked into this variant
  const Format depthStencilFormat = desc.depthFormat != Format_Invalid ? desc.depthFormat : desc.stencilFormat;
  const bool isDepth = hasDepth(depthStencilFormat);
  const WGPUDepthStencilState depthStencil = {
      .format = formatToWGPU(depthStencilFormat),
      .depthWriteEnabled = !isDepth                    ? WGPUOptionalBool_Undefined
                           : state.isDepthWriteEnabled ? WGPUOptionalBool_True
                                                       : WGPUOptionalBool_False,
      .depthCompare = isDepth ? compareOpToWGPU(state.depthCompareOp) : WGPUCompareFunction_Undefined,
      .stencilFront = stencilFaceToWGPU(state.stencilFront),
      .stencilBack = stencilFaceToWGPU(state.stencilBack),
      .stencilReadMask = state.isStencilEnabled ? state.stencilFront.readMask : 0xFFFFFFFF,
      .stencilWriteMask = state.isStencilEnabled ? state.stencilFront.writeMask : 0xFFFFFFFF,
      .depthBias = int32_t(state.depthBiasConstant),
      .depthBiasSlopeScale = state.depthBiasSlope,
      .depthBiasClamp = state.depthBiasClamp,
  };

  ConstantEntries vertConstants;
  ConstantEntries fragConstants;
  getConstantEntries(rps.vertConstants, vertConstants);
  getConstantEntries(rps.fragConstants, fragConstants);

  const WGPUFragmentState fragmentState = {
      .module = rps.frag,
      .entryPoint = toStringView(nullptr),
      .constantCount = fragConstants.count,
      .constants = fragConstants.entries,
      .targetCount = numColorAttachments,
      .targets = colorTargets,
  };

  const WGPURenderPipelineDescriptor pipelineDesc = {
      .label = toStringView(rps.debugName.c_str()),
      .layout = rps.layout,
      .vertex =
          {
              .module = rps.vert,
              .entryPoint = toStringView(nullptr),
              .constantCount = vertConstants.count,
              .constants = vertConstants.entries,
              .bufferCount = numBindings,
              .buffers = bufferLayouts,
          },
      .primitive =
          {
              .topology = topologyToWGPU(desc.topology),
              // LVK's clip space and winding match WebGPU (the Vulkan backend flips Y)
              .frontFace = desc.frontFace == WindingMode_CCW ? WGPUFrontFace_CCW : WGPUFrontFace_CW,
              .cullMode = cullModeToWGPU(desc.cullMode),
          },
      .depthStencil = depthStencilFormat != Format_Invalid ? &depthStencil : nullptr,
      .multisample =
          {
              .count = getSampleCountWGPU(desc.samplesCount),
              .mask = 0xFFFFFFFF,
              .alphaToCoverageEnabled = desc.alphaToCoverage,
          },
      .fragment = rps.frag ? &fragmentState : nullptr,
  };

  return wgpuDeviceCreateRenderPipeline(device_, &pipelineDesc);
}

void lvk::WebGPUContext::generateMipmap(WGPUCommandEncoder encoder, const WebGPUTexture& tex) {
  LVK_PROFILER_FUNCTION();

  if (tex.numLevels <= 1) {
    return;
  }

  const WGPUTextureFormat format = formatToWGPU(tex.format);

  if (!isMipmapGenerationSupported(format) || tex.type == TextureType_3D) {
    LLOGW("generateMipmap(): format %u is not supported by the WebGPU backend\n", unsigned(tex.format));
    return;
  }

  if (!mipmapShader_) {
    WGPUShaderSourceWGSL source = {
        .chain = {.sType = WGPUSType_ShaderSourceWGSL},
        .code = toStringView(kMipmapShaderWGSL),
    };
    const WGPUShaderModuleDescriptor moduleDesc = {
        .nextInChain = &source.chain,
        .label = toStringView("Shader Module: generateMipmap()"),
    };
    mipmapShader_ = wgpuDeviceCreateShaderModule(device_, &moduleDesc);
    const WGPUSamplerDescriptor samplerDesc = {
        .label = toStringView("Sampler: generateMipmap()"),
        .addressModeU = WGPUAddressMode_ClampToEdge,
        .addressModeV = WGPUAddressMode_ClampToEdge,
        .addressModeW = WGPUAddressMode_ClampToEdge,
        .magFilter = WGPUFilterMode_Linear,
        .minFilter = WGPUFilterMode_Linear,
        .mipmapFilter = WGPUMipmapFilterMode_Nearest,
        .lodMaxClamp = 32.0f,
        .maxAnisotropy = 1,
    };
    mipmapSampler_ = wgpuDeviceCreateSampler(device_, &samplerDesc);
  }

  WGPURenderPipeline pipeline = nullptr;
  for (const MipmapPipeline& mp : mipmapPipelines_) {
    if (mp.format == format) {
      pipeline = mp.pipeline;
      break;
    }
  }
  if (!pipeline) {
    const WGPUColorTargetState target = {
        .format = format,
        .writeMask = WGPUColorWriteMask_All,
    };
    const WGPUFragmentState fragmentState = {
        .module = mipmapShader_,
        .entryPoint = toStringView("fragmentMain"),
        .targetCount = 1,
        .targets = &target,
    };
    const WGPURenderPipelineDescriptor pipelineDesc = {
        .label = toStringView("Pipeline: generateMipmap()"),
        .vertex = {.module = mipmapShader_, .entryPoint = toStringView("vertexMain")},
        .primitive = {.topology = WGPUPrimitiveTopology_TriangleList, .frontFace = WGPUFrontFace_CCW, .cullMode = WGPUCullMode_None},
        .multisample = {.count = 1, .mask = 0xFFFFFFFF},
        .fragment = &fragmentState,
    };
    pipeline = wgpuDeviceCreateRenderPipeline(device_, &pipelineDesc);
    mipmapPipelines_.push_back({.format = format, .pipeline = pipeline});
  }

  const WGPUBindGroupLayout bindGroupLayout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);

  for (uint32_t layer = 0; layer != tex.numLayers; layer++) {
    for (uint32_t level = 1; level != tex.numLevels; level++) {
      const WGPUTextureViewDescriptor srcDesc = {
          .dimension = WGPUTextureViewDimension_2D,
          .baseMipLevel = tex.baseLevel + level - 1,
          .mipLevelCount = 1,
          .baseArrayLayer = tex.baseLayer + layer,
          .arrayLayerCount = 1,
          .aspect = WGPUTextureAspect_All,
          .usage = WGPUTextureUsage_TextureBinding,
      };
      const WGPUTextureViewDescriptor dstDesc = {
          .dimension = WGPUTextureViewDimension_2D,
          .baseMipLevel = tex.baseLevel + level,
          .mipLevelCount = 1,
          .baseArrayLayer = tex.baseLayer + layer,
          .arrayLayerCount = 1,
          .aspect = WGPUTextureAspect_All,
          .usage = WGPUTextureUsage_RenderAttachment,
      };
      const WGPUTextureView srcView = wgpuTextureCreateView(tex.texture, &srcDesc);
      const WGPUTextureView dstView = wgpuTextureCreateView(tex.texture, &dstDesc);

      const WGPUBindGroupEntry entries[] = {
          {.binding = 0, .textureView = srcView},
          {.binding = 1, .sampler = mipmapSampler_},
      };
      const WGPUBindGroupDescriptor bindGroupDesc = {
          .layout = bindGroupLayout,
          .entryCount = LVK_ARRAY_NUM_ELEMENTS(entries),
          .entries = entries,
      };
      const WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device_, &bindGroupDesc);

      const WGPURenderPassColorAttachment colorAttachment = {
          .view = dstView,
          .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
          .loadOp = WGPULoadOp_Clear,
          .storeOp = WGPUStoreOp_Store,
      };
      const WGPURenderPassDescriptor passDesc = {
          .label = toStringView("Render pass: generateMipmap()"),
          .colorAttachmentCount = 1,
          .colorAttachments = &colorAttachment,
      };
      const WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);

      wgpuRenderPassEncoderRelease(pass);
      wgpuBindGroupRelease(bindGroup);
      wgpuTextureViewRelease(srcView);
      wgpuTextureViewRelease(dstView);
    }
  }

  wgpuBindGroupLayoutRelease(bindGroupLayout);
}

lvk::Holder<lvk::RayTracingPipelineHandle> lvk::WebGPUContext::createRayTracingPipeline(const RayTracingPipelineDesc&, Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no ray tracing");
  return {};
}

lvk::Holder<lvk::ShaderModuleHandle> lvk::WebGPUContext::createShaderModule(const ShaderModuleDesc& desc, Result* outResult) {
  LVK_PROFILER_FUNCTION_COLOR(LVK_PROFILER_COLOR_CREATE);

  WGSLShader shader;
  Result result = compileShaderWGSL(slangGlobalSession_, desc, limits_.maxImmediateSize < 64 ? limits_.maxImmediateSize : 64, &shader);

  if (!result.isOk()) {
    LLOGW("Shader module `%s` failed: %s\n", desc.debugName, result.message);
    Result::setResult(outResult, result);
    return {};
  }

  WGPUShaderSourceWGSL source = {
      .chain = {.sType = WGPUSType_ShaderSourceWGSL},
      .code = {.data = shader.code.c_str(), .length = shader.code.size()},
  };
  const WGPUShaderModuleDescriptor moduleDesc = {
      .nextInChain = &source.chain,
      .label = toStringView(desc.debugName),
  };
  const WGPUShaderModule module = wgpuDeviceCreateShaderModule(device_, &moduleDesc);
  if (!module) {
    Result::setResult(outResult, Result::Code::RuntimeError, "wgpuDeviceCreateShaderModule() failed");
    return {};
  }

  Result::setResult(outResult, Result());

  return {this,
          shaderModulesPool_.create({
              .module = module,
              .stage = desc.stage,
              .pushConstants = shader.pushConstants,
              .pushConstantsSize = shader.pushConstantsSize,
              .overrides = std::move(shader.overrides),
          })};
}

lvk::Holder<lvk::QueryPoolHandle> lvk::WebGPUContext::createQueryPool(uint32_t, const char*, Result* outResult) {
  LVK_WEBGPU_NOT_IMPLEMENTED();
  Result::setResult(outResult, Result::Code::RuntimeError, "Not implemented in the WebGPU backend");
  return {};
}

lvk::Holder<lvk::AccelStructHandle> lvk::WebGPUContext::createAccelerationStructure(const AccelStructDesc&, Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no ray tracing");
  return {};
}

lvk::Holder<lvk::TensorHandle> lvk::WebGPUContext::createTensor(const TensorDesc&, const char*, Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no tensors");
  return {};
}

lvk::Holder<lvk::TensorHandle> lvk::WebGPUContext::createTensorBufferAliasing(BufferHandle,
                                                                              uint64_t,
                                                                              const TensorDesc&,
                                                                              const char*,
                                                                              Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no tensors");
  return {};
}

lvk::Holder<lvk::TensorHandle> lvk::WebGPUContext::createTensorTextureAliasing(TextureHandle,
                                                                               const TensorDesc&,
                                                                               const char*,
                                                                               Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no tensors");
  return {};
}

lvk::Holder<lvk::DataGraphPipelineHandle> lvk::WebGPUContext::createDataGraphPipeline(const DataGraphPipelineDesc&, Result* outResult) {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no data graph pipelines");
  return {};
}

void lvk::WebGPUContext::destroy(ComputePipelineHandle handle) {
  WebGPUComputePipelineState* cps = computePipelinesPool_.get(handle);
  if (!cps) {
    return;
  }
  wgpuComputePipelineRelease(cps->pipeline);
  wgpuPipelineLayoutRelease(cps->layout);
  computePipelinesPool_.destroy(handle);
}

void lvk::WebGPUContext::destroy(RenderPipelineHandle handle) {
  WebGPURenderPipelineState* rps = renderPipelinesPool_.get(handle);
  if (!rps) {
    return;
  }
  for (WebGPURenderPipelineVariant& variant : rps->variants) {
    if (variant.pipeline) {
      wgpuRenderPipelineRelease(variant.pipeline);
    }
  }
  wgpuPipelineLayoutRelease(rps->layout);
  wgpuShaderModuleRelease(rps->vert);
  if (rps->frag) {
    wgpuShaderModuleRelease(rps->frag);
  }
  renderPipelinesPool_.destroy(handle);
}

void lvk::WebGPUContext::destroy(RayTracingPipelineHandle) {}

void lvk::WebGPUContext::destroy(ShaderModuleHandle handle) {
  WebGPUShaderModuleState* sm = shaderModulesPool_.get(handle);
  if (!sm) {
    return;
  }
  wgpuShaderModuleRelease(sm->module);
  shaderModulesPool_.destroy(handle);
}

void lvk::WebGPUContext::destroy(SamplerHandle handle) {
  WebGPUSampler* smp = samplersPool_.get(handle);
  // a sampler waiting in deferredSlots_ has no WGPUSampler anymore
  if (!smp || !smp->sampler) {
    return;
  }
  if (resourceTable_ && kSamplersSlotOffset + handle.index() < kResourceTableSize) {
    wgpuResourceTableRemove(resourceTable_, kSamplersSlotOffset + handle.index());
  }
  wgpuSamplerRelease(smp->sampler);
  *smp = {};
  // the resource table slot can be reused only after the GPU is done with the work submitted so far
  deferredSlots_.push_back({.sampler = handle, .submitValue = getNextSubmitValue()});
}

void lvk::WebGPUContext::destroy(BufferHandle handle) {
  WebGPUBuffer* buf = buffersPool_.get(handle);
  if (!buf) {
    return;
  }
  for (size_t i = 0; i != hostVisibleBuffers_.size(); i++) {
    if (hostVisibleBuffers_[i] == handle) {
      hostVisibleBuffers_[i] = hostVisibleBuffers_.back();
      hostVisibleBuffers_.pop_back();
      break;
    }
  }
  // readbacks in flight and unsubmitted command buffers drop the stale handle
  for (PendingReadback& readback : pendingReadbacks_) {
    for (PendingReadback::Entry& entry : readback.entries) {
      if (entry.buffer == handle) {
        entry.buffer = {};
      }
    }
  }
  for (const std::unique_ptr<WebGPUCommandBuffer>& cmdBuffer : commandBuffers_) {
    std::vector<BufferHandle>& written = cmdBuffer->writtenHostVisibleBuffers_;
    for (size_t i = 0; i != written.size(); i++) {
      if (written[i] == handle) {
        written[i] = written.back();
        written.pop_back();
        break;
      }
    }
  }
  if (buf->arena) {
    // the range is reused only after the GPU is done with the work submitted so far
    deferredSlots_.push_back({
        .arena = buf->arena,
        .arenaOffset = buf->offset,
        .arenaSize = buf->size,
        .submitValue = getNextSubmitValue(),
    });
  } else {
    // like textures in releaseTexture()
    if (commandBuffers_.empty()) {
      wgpuBufferDestroy(buf->buffer);
    }
    wgpuBufferRelease(buf->buffer);
  }
  buffersPool_.destroy(handle);
}

void lvk::WebGPUContext::destroy(TextureHandle handle) {
  if (handle == swapchainTexture_) {
    // the swapchain texture is owned by the context
    return;
  }
  WebGPUTexture* tex = texturesPool_.get(handle);
  // a texture waiting in deferredSlots_ has no WGPUTexture anymore
  if (!tex || !tex->texture) {
    return;
  }
  if (resourceTable_ && tex->sampledView && handle.index() < kSamplersSlotOffset) {
    wgpuResourceTableRemove(resourceTable_, handle.index());
  }
  releaseTexture(*tex);
  // the resource table slot can be reused only after the GPU is done with the work submitted so far
  deferredSlots_.push_back({.texture = handle, .submitValue = getNextSubmitValue()});
}

void lvk::WebGPUContext::destroy(QueryPoolHandle) {}

void lvk::WebGPUContext::destroy(AccelStructHandle) {}

void lvk::WebGPUContext::destroy(TensorHandle) {}

void lvk::WebGPUContext::destroy(DataGraphPipelineHandle) {}

void lvk::WebGPUContext::destroy(Framebuffer& fb) {
  for (Framebuffer::AttachmentDesc& attachment : fb.color) {
    destroy(attachment.texture);
    destroy(attachment.resolveTexture);
  }
  destroy(fb.depthStencil.texture);
  destroy(fb.depthStencil.resolveTexture);
  fb = {};
}

uint64_t lvk::WebGPUContext::gpuAddress(AccelStructHandle) const {
  return 0;
}

lvk::AccelStructSizes lvk::WebGPUContext::getAccelStructSizes(const AccelStructDesc&, Result* outResult) const {
  Result::setResult(outResult, Result::Code::RuntimeError, "WebGPU has no ray tracing");
  return {};
}

lvk::Result lvk::WebGPUContext::upload(BufferHandle handle, const void* data, size_t size, size_t offset) {
  LVK_PROFILER_FUNCTION();

  WebGPUBuffer* buf = buffersPool_.get(handle);

  if (!buf || !data) {
    return Result(Result::Code::ArgumentOutOfRange, "Invalid buffer or data");
  }
  if (offset + size > buf->size) {
    return Result(Result::Code::ArgumentOutOfRange, "Out of range");
  }
  if (!buf->mappedData.empty()) {
    // the next submit() writes it to the GPU
    memcpy(reinterpret_cast<uint8_t*>(buf->mappedData.data()) + offset, data, size);
    return Result();
  }
  if (offset % 4 || size % 4) {
    return Result(Result::Code::ArgumentOutOfRange, "WebGPU buffer uploads need 4-byte aligned offsets and sizes");
  }

  wgpuQueueWriteBuffer(queue_, buf->buffer, buf->offset + offset, data, size);

  return Result();
}

lvk::Result lvk::WebGPUContext::download(BufferHandle handle, void* data, size_t size, size_t offset) {
  LVK_PROFILER_FUNCTION();

  const WebGPUBuffer* buf = buffersPool_.get(handle);

  if (!buf || !data) {
    return Result(Result::Code::ArgumentOutOfRange, "Invalid buffer or data");
  }
  if (offset + size > buf->size) {
    return Result(Result::Code::ArgumentOutOfRange, "Out of range");
  }
  if (!size) {
    return Result();
  }
  if (!buf->mappedData.empty()) {
    // what the GPU wrote arrives in wait() or isReady()
    memcpy(data, reinterpret_cast<const uint8_t*>(buf->mappedData.data()) + offset, size);
    return Result();
  }

  // WebGPU copies 4-byte aligned ranges
  const uint64_t begin = (buf->offset + offset) & ~uint64_t(3);
  const uint64_t end = (buf->offset + offset + size + 3) & ~uint64_t(3);

  const WGPUBufferDescriptor stagingDesc = {
      .label = toStringView("Buffer: download staging"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
      .size = end - begin,
  };
  const WGPUBuffer staging = wgpuDeviceCreateBuffer(device_, &stagingDesc);

  const WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(encoder, buf->buffer, begin, staging, 0, end - begin);

  const uint8_t* mapped = submitAndMapStagingBuffer(encoder, staging, end - begin);
  if (mapped) {
    memcpy(data, mapped + (buf->offset + offset - begin), size);
  }
  releaseStagingBuffer(staging, mapped);

  return mapped ? Result() : Result(Result::Code::RuntimeError, "Cannot map the staging buffer");
}

uint8_t* lvk::WebGPUContext::getMappedPtr(BufferHandle handle) const {
  const WebGPUBuffer* buf = buffersPool_.get(handle);

  LVK_ASSERT(buf);

  // the CPU copy of a host-visible buffer (see WebGPUBuffer)
  return buf && !buf->mappedData.empty() ? reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(buf->mappedData.data())) : nullptr;
}

uint64_t lvk::WebGPUContext::gpuAddress(BufferHandle handle, size_t offset) const {
  const WebGPUBuffer* buf = buffersPool_.get(handle);
  LVK_ASSERT_MSG(buf && buf->arena, "Only buffers with Uniform, Storage or Indirect usage have a GPU address");
  if (!buf || !buf->arena) {
    return 0;
  }
  // shaders see it as `vec2u(byte offset, arena)`
  return (uint64_t(buf->arena) << 32) | (buf->offset + offset);
}

void lvk::WebGPUContext::flushMappedMemory(BufferHandle, size_t, size_t) const {
  // submit() writes every change of the CPU copies
}

void lvk::WebGPUContext::invalidateMappedMemory(BufferHandle, size_t, size_t) const {
  // wait() and isReady() copy what the GPU wrote into the CPU copies
}

uint32_t lvk::WebGPUContext::getMaxStorageBufferRange() const {
  return uint32_t(limits_.maxStorageBufferBindingSize < 0xFFFFFFFFull ? limits_.maxStorageBufferBindingSize : 0xFFFFFFFFull);
}

lvk::Result lvk::WebGPUContext::upload(TensorHandle, const void*, size_t, size_t) {
  return Result(Result::Code::RuntimeError, "WebGPU has no tensors");
}

lvk::Result lvk::WebGPUContext::download(TensorHandle, void*, size_t, size_t) {
  return Result(Result::Code::RuntimeError, "WebGPU has no tensors");
}

uint8_t* lvk::WebGPUContext::getMappedPtr(TensorHandle) const {
  return nullptr;
}

lvk::TensorDesc lvk::WebGPUContext::getTensorDesc(TensorHandle) const {
  return {};
}

lvk::Result lvk::WebGPUContext::upload(TextureHandle handle, const TextureRangeDesc& range, const void* data, uint32_t bufferRowLength) {
  LVK_PROFILER_FUNCTION();

  const WebGPUTexture* tex = texturesPool_.get(handle);

  if (!tex || !tex->texture || !data) {
    return Result(Result::Code::ArgumentOutOfRange, "Invalid texture or data");
  }
  // see `isCopyable` in createTexture()
  const bool isCopyable = !hasDepth(tex->format) && !hasStencil(tex->format) && tex->numSamples == 1;
  if (!isCopyable) {
    return Result(Result::Code::ArgumentOutOfRange, "WebGPU cannot upload depth, stencil or multisampled textures");
  }

  const bool is3D = tex->type == TextureType_3D;
  const bool isInRange = range.mipLevel + range.numMipLevels <= tex->numLevels &&
                         (is3D || range.layer + range.numLayers <= tex->numLayers) && range.dimensions.width && range.dimensions.height &&
                         range.dimensions.depth && range.numLayers && range.numMipLevels;
  if (!isInRange) {
    return Result(Result::Code::ArgumentOutOfRange, "The range is out of the texture's mip-levels or layers");
  }
  if (bufferRowLength && (range.numMipLevels > 1 || range.numLayers > 1)) {
    return Result(Result::Code::ArgumentOutOfRange, "bufferRowLength works only with one mip-level and layer");
  }

  const uint32_t blockSize = getBlockSize(tex->format);
  const uint32_t rowLength = bufferRowLength ? bufferRowLength : range.dimensions.width;
  const uint8_t* src = static_cast<const uint8_t*>(data);

  // `data` is laid out mip-major, layer-minor, like in the Vulkan backend
  for (uint32_t mip = 0; mip != range.numMipLevels; mip++) {
    const uint32_t width = range.dimensions.width >> mip ? range.dimensions.width >> mip : 1;
    const uint32_t height = range.dimensions.height >> mip ? range.dimensions.height >> mip : 1;
    const uint32_t depth = is3D ? (range.dimensions.depth >> mip ? range.dimensions.depth >> mip : 1) : 1;
    const uint32_t levelRowLength = rowLength >> mip ? rowLength >> mip : 1;
    // one row of blocks
    const uint32_t bytesPerRow = getTextureBytesPerLayer(levelRowLength, 1, tex->format, 0);
    const uint32_t bytesPerImage = getTextureBytesPerLayer(rowLength, range.dimensions.height, tex->format, mip);
    const uint32_t rowsPerImage = bytesPerImage / bytesPerRow;
    // WebGPU copies whole blocks: small mip-levels of compressed textures use their physical size
    const WGPUExtent3D extent = {uint32_t(alignUp(width, blockSize)), uint32_t(alignUp(height, blockSize)), depth};
    const WGPUTexelCopyBufferLayout layout = {.offset = 0, .bytesPerRow = bytesPerRow, .rowsPerImage = rowsPerImage};
    const uint32_t numLayers = is3D ? 1 : range.numLayers;
    for (uint32_t layer = 0; layer != numLayers; layer++) {
      const WGPUTexelCopyTextureInfo dst = {
          .texture = tex->texture,
          .mipLevel = tex->baseLevel + range.mipLevel + mip,
          .origin = {uint32_t(range.offset.x) >> mip,
                     uint32_t(range.offset.y) >> mip,
                     is3D ? uint32_t(range.offset.z) >> mip : tex->baseLayer + range.layer + layer},
          .aspect = WGPUTextureAspect_All,
      };
      const size_t size = size_t(bytesPerImage) * depth;
      wgpuQueueWriteTexture(queue_, &dst, src, size, &layout, &extent);
      src += size;
    }
  }

  return Result();
}

lvk::Result lvk::WebGPUContext::download(TextureHandle handle, const TextureRangeDesc& range, void* outData) {
  LVK_PROFILER_FUNCTION();

  const WebGPUTexture* tex = texturesPool_.get(handle);

  if (!tex || !tex->texture || !outData) {
    return Result(Result::Code::ArgumentOutOfRange, "Invalid texture or output data");
  }

  // the formats before Format_ETC2_RGB8 are uncompressed color formats
  const bool isSupported = range.numLayers == 1 && range.numMipLevels == 1 && range.dimensions.depth == 1 && tex->numSamples == 1 &&
                           tex->format < Format_ETC2_RGB8;
  if (!isSupported) {
    return Result(Result::Code::ArgumentOutOfRange, "WebGPU downloads support one 2D layer and mip-level of uncompressed color formats");
  }

  const uint32_t width = range.dimensions.width;
  const uint32_t height = range.dimensions.height;
  const uint32_t rowSize = getTextureBytesPerLayer(width, 1, tex->format, 0);
  // WebGPU copies need 256-byte aligned rows
  const uint32_t bytesPerRow = uint32_t(alignUp(rowSize, 256));

  const WGPUBufferDescriptor stagingDesc = {
      .label = toStringView("Buffer: download staging"),
      .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
      .size = uint64_t(bytesPerRow) * height,
  };
  const WGPUBuffer staging = wgpuDeviceCreateBuffer(device_, &stagingDesc);

  // the range of a texture view is relative to its base level and layer, like in upload()
  const WGPUTexelCopyTextureInfo src = {
      .texture = tex->texture,
      .mipLevel = tex->baseLevel + range.mipLevel,
      .origin = {uint32_t(range.offset.x),
                 uint32_t(range.offset.y),
                 tex->type == TextureType_3D ? uint32_t(range.offset.z) : tex->baseLayer + range.layer},
      .aspect = WGPUTextureAspect_All,
  };
  const WGPUTexelCopyBufferInfo dst = {
      .layout = {.offset = 0, .bytesPerRow = bytesPerRow, .rowsPerImage = height},
      .buffer = staging,
  };
  const WGPUExtent3D extent = {width, height, 1};

  const WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
  wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

  const uint8_t* mapped = submitAndMapStagingBuffer(encoder, staging, stagingDesc.size);
  if (mapped) {
    for (uint32_t y = 0; y != height; y++) {
      memcpy(static_cast<uint8_t*>(outData) + size_t(y) * rowSize, mapped + size_t(y) * bytesPerRow, rowSize);
    }
  }
  releaseStagingBuffer(staging, mapped);

  return mapped ? Result() : Result(Result::Code::RuntimeError, "Cannot map the staging buffer");
}

const uint8_t* lvk::WebGPUContext::submitAndMapStagingBuffer(WGPUCommandEncoder encoder, WGPUBuffer staging, uint64_t size) {
  submitEncoder(encoder);
  bool isMapped = false;
  const WGPUBufferMapCallbackInfo mapInfo = {
      .mode = WGPUCallbackMode_WaitAnyOnly,
      .callback =
          [](WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*) {
            if (status != WGPUMapAsyncStatus_Success) {
              LLOGW("wgpuBufferMapAsync() failed: %.*s\n", int(message.length), message.data);
            }
            *static_cast<bool*>(userdata1) = status == WGPUMapAsyncStatus_Success;
          },
      .userdata1 = &isMapped,
  };
  WGPUFutureWaitInfo waitInfo = {.future = wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, mapInfo)};
  wgpuInstanceWaitAny(instance_, 1, &waitInfo, UINT64_MAX);
  return isMapped ? static_cast<const uint8_t*>(wgpuBufferGetConstMappedRange(staging, 0, size)) : nullptr;
}

void lvk::WebGPUContext::releaseStagingBuffer(WGPUBuffer staging, bool isMapped) const {
  if (isMapped) {
    wgpuBufferUnmap(staging);
  }
  wgpuBufferDestroy(staging);
  wgpuBufferRelease(staging);
}

lvk::Dimensions lvk::WebGPUContext::getDimensions(TextureHandle handle) const {
  const WebGPUTexture* tex = texturesPool_.get(handle);
  if (!tex) {
    return {};
  }
  return tex->extent;
}

float lvk::WebGPUContext::getAspectRatio(TextureHandle handle) const {
  const Dimensions dim = getDimensions(handle);
  return dim.height ? float(dim.width) / float(dim.height) : 1.0f;
}

lvk::Format lvk::WebGPUContext::getFormat(TextureHandle handle) const {
  const WebGPUTexture* tex = texturesPool_.get(handle);
  return tex ? tex->format : Format_Invalid;
}

lvk::Result lvk::WebGPUContext::initSwapchain(uint32_t width, uint32_t height) {
  LVK_PROFILER_FUNCTION();

  if (!device_) {
    return Result(Result::Code::RuntimeError, "There is no WebGPU device");
  }
  if (!width || !height) {
    return Result(Result::Code::ArgumentOutOfRange, "Swapchain size is zero");
  }

  releaseSwapchainTexture();

  if (swapchainTexture_.empty()) {
    swapchainTexture_ = texturesPool_.create({.isSwapchainTexture = true});
  } else {
    // drop the borrowed texture of the last frame (see releaseSwapchainTexture())
    *texturesPool_.get(swapchainTexture_) = {.isSwapchainTexture = true};
  }

  if (!surface_) {
    return initHeadlessSwapchain(width, height);
  }

  WGPUSurfaceCapabilities caps = {};
  if (wgpuSurfaceGetCapabilities(surface_, adapter_, &caps) != WGPUStatus_Success) {
    return Result(Result::Code::RuntimeError, "wgpuSurfaceGetCapabilities() failed");
  }
  if (!caps.formatCount) {
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    return Result(Result::Code::RuntimeError, "The WebGPU surface has no formats");
  }

  auto hasFormat = [&caps](WGPUTextureFormat format) -> bool {
    for (size_t i = 0; i != caps.formatCount; i++) {
      if (caps.formats[i] == format) {
        return true;
      }
    }
    return false;
  };
  auto hasPresentMode = [&caps](WGPUPresentMode mode) -> bool {
    for (size_t i = 0; i != caps.presentModeCount; i++) {
      if (caps.presentModes[i] == mode) {
        return true;
      }
    }
    return false;
  };

  // the same formats as the Vulkan backend picks for ColorSpace_SRGB_NONLINEAR
  swapchainFormat_ = hasFormat(WGPUTextureFormat_BGRA8Unorm)   ? WGPUTextureFormat_BGRA8Unorm
                     : hasFormat(WGPUTextureFormat_RGBA8Unorm) ? WGPUTextureFormat_RGBA8Unorm
                                                               : caps.formats[0];
  if (config_.swapchainRequestedColorSpace != ColorSpace_SRGB_NONLINEAR) {
    LLOGW("The WebGPU backend supports only ColorSpace_SRGB_NONLINEAR swapchains\n");
  }

  presentMode_ = WGPUPresentMode_Fifo;
  for (PresentMode mode : config_.presentModes) {
    const WGPUPresentMode wgpuMode = presentModeToWGPU(mode);
    if (hasPresentMode(wgpuMode)) {
      presentMode_ = wgpuMode;
      break;
    }
  }

  // for screenshots
  const WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment | (caps.usages & WGPUTextureUsage_CopySrc);

  const WGPUSurfaceConfiguration surfaceConfig = {
      .device = device_,
      .format = swapchainFormat_,
      .usage = usage,
      .width = width,
      .height = height,
      .alphaMode = WGPUCompositeAlphaMode_Auto,
      .presentMode = presentMode_,
  };
  wgpuSurfaceConfigure(surface_, &surfaceConfig);
  wgpuSurfaceCapabilitiesFreeMembers(caps);

  swapchainWidth_ = width;
  swapchainHeight_ = height;

  return Result();
}

lvk::Result lvk::WebGPUContext::initHeadlessSwapchain(uint32_t width, uint32_t height) {
  if (headlessSwapchain_) {
    wgpuTextureRelease(headlessSwapchain_);
  }
  // the format surfaces usually have
  swapchainFormat_ = WGPUTextureFormat_BGRA8Unorm;
  const WGPUTextureDescriptor desc = {
      .label = toStringView("Texture: headless swapchain"),
      .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding,
      .dimension = WGPUTextureDimension_2D,
      .size = {width, height, 1},
      .format = swapchainFormat_,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  headlessSwapchain_ = wgpuDeviceCreateTexture(device_, &desc);
  if (!headlessSwapchain_) {
    return Result(Result::Code::RuntimeError, "Cannot create the headless swapchain texture");
  }
  swapchainWidth_ = width;
  swapchainHeight_ = height;
  return Result();
}

void lvk::WebGPUContext::retainPresentedFrame(WGPUCommandEncoder encoder) {
  const WebGPUTexture* tex = texturesPool_.get(swapchainTexture_);
  if (!(wgpuTextureGetUsage(tex->texture) & WGPUTextureUsage_CopySrc)) {
    LLOGW("The WebGPU surface does not support copies: presented frames cannot be retained\n");
    isPresentedFrameRetained_ = false;
    return;
  }
  const bool isSameSize = retainedFrame_ && wgpuTextureGetWidth(retainedFrame_) == tex->extent.width &&
                          wgpuTextureGetHeight(retainedFrame_) == tex->extent.height &&
                          wgpuTextureGetFormat(retainedFrame_) == swapchainFormat_;
  if (!isSameSize) {
    if (retainedFrame_) {
      wgpuTextureRelease(retainedFrame_);
    }
    const WGPUTextureDescriptor desc = {
        .label = toStringView("Texture: retained presented frame"),
        .usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst,
        .dimension = WGPUTextureDimension_2D,
        .size = {tex->extent.width, tex->extent.height, 1},
        .format = swapchainFormat_,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    retainedFrame_ = wgpuDeviceCreateTexture(device_, &desc);
    if (!retainedFrame_) {
      isPresentedFrameRetained_ = false;
      return;
    }
  }
  const WGPUTexelCopyTextureInfo src = {.texture = tex->texture, .aspect = WGPUTextureAspect_All};
  const WGPUTexelCopyTextureInfo dst = {.texture = retainedFrame_, .aspect = WGPUTextureAspect_All};
  const WGPUExtent3D extent = {tex->extent.width, tex->extent.height, 1};
  wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
}

lvk::TextureHandle lvk::WebGPUContext::getCurrentSwapchainTexture() {
  LVK_PROFILER_FUNCTION();

  if (swapchainTexture_.empty()) {
    return {};
  }

  if (isSwapchainTextureAcquired_) {
    return swapchainTexture_;
  }

  // setCurrentPresentMode() was called while a frame was being rendered
  if (isSwapchainReconfigurePending_) {
    isSwapchainReconfigurePending_ = false;
    if (!initSwapchain(swapchainWidth_, swapchainHeight_).isOk()) {
      return {};
    }
  }

  // the slot owns a reference to its texture
  const WGPUTexture texture = surface_ ? acquireSurfaceTexture() : headlessSwapchain_;
  if (!texture) {
    return {};
  }
  if (!surface_) {
    wgpuTextureAddRef(texture);
  }

  *texturesPool_.get(swapchainTexture_) = {
      .texture = texture,
      .view = wgpuTextureCreateView(texture, nullptr),
      .format = wgpuFormatToFormat(swapchainFormat_),
      .extent = {.width = wgpuTextureGetWidth(texture), .height = wgpuTextureGetHeight(texture)},
      .isSwapchainTexture = true,
  };
  isSwapchainTextureAcquired_ = true;

  return swapchainTexture_;
}

WGPUTexture lvk::WebGPUContext::acquireSurfaceTexture() {
  WGPUSurfaceTexture surfaceTexture = {};
  wgpuSurfaceGetCurrentTexture(surface_, &surfaceTexture);

  const bool isOutdated = surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Outdated ||
                          surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_Lost;
  if (isOutdated) {
    if (surfaceTexture.texture) {
      wgpuTextureRelease(surfaceTexture.texture);
    }
    // the window was resized, but the application has not called initSwapchain() yet
    if (!initSwapchain(swapchainWidth_, swapchainHeight_).isOk()) {
      return nullptr;
    }
    surfaceTexture = {};
    wgpuSurfaceGetCurrentTexture(surface_, &surfaceTexture);
  }

  const bool isAcquired = surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
                          surfaceTexture.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal;
  if (!isAcquired) {
    LLOGW("wgpuSurfaceGetCurrentTexture() failed with status %u\n", unsigned(surfaceTexture.status));
    if (surfaceTexture.texture) {
      wgpuTextureRelease(surfaceTexture.texture);
    }
    return nullptr;
  }

  return surfaceTexture.texture;
}

void lvk::WebGPUContext::releaseSwapchainTexture() {
  if (!isSwapchainTextureAcquired_) {
    return;
  }
  WebGPUTexture* tex = texturesPool_.get(swapchainTexture_);
  if (tex->view) {
    wgpuTextureViewRelease(tex->view);
  }
  if (tex->texture) {
    wgpuTextureRelease(tex->texture);
  }
  // until the next acquire, the slot shows the last frame (headless or retained) without owning it, for download()
  const WGPUTexture lastFrame = !surface_ ? headlessSwapchain_ : isPresentedFrameRetained_ ? retainedFrame_ : nullptr;
  *tex = {
      .texture = lastFrame,
      .format = lastFrame ? wgpuFormatToFormat(swapchainFormat_) : Format_Invalid,
      .extent = lastFrame ? Dimensions{.width = wgpuTextureGetWidth(lastFrame), .height = wgpuTextureGetHeight(lastFrame)} : Dimensions{},
      .isSwapchainTexture = true,
  };
  isSwapchainTextureAcquired_ = false;
}

lvk::Format lvk::WebGPUContext::getSwapchainFormat() const {
  return wgpuFormatToFormat(swapchainFormat_);
}

lvk::ColorSpace lvk::WebGPUContext::getSwapchainColorSpace() const {
  return ColorSpace_SRGB_NONLINEAR;
}

uint32_t lvk::WebGPUContext::getSwapchainCurrentImageIndex() const {
  // WebGPU does not expose swapchain images: there is one handle for the current texture
  return 0;
}

uint32_t lvk::WebGPUContext::getNumSwapchainImages() const {
  return swapchainTexture_.empty() ? 0 : 1;
}

lvk::SurfaceTransform lvk::WebGPUContext::getSwapchainSurfaceTransform() const {
  return SurfaceTransform_Identity;
}

bool lvk::WebGPUContext::setCurrentPresentMode(PresentMode mode) {
  PresentMode modes[kMaxPresentModes] = {};
  const uint32_t numModes = getPresentModes(modes, LVK_ARRAY_NUM_ELEMENTS(modes));
  bool isSupported = false;
  for (uint32_t i = 0; i != numModes && !isSupported; i++) {
    isSupported = modes[i] == mode;
  }
  if (!isSupported) {
    return false;
  }
  // the first present mode of the config is tried first in initSwapchain()
  const PresentMode presentModes[kMaxPresentModes] = {mode};
  memcpy(config_.presentModes, presentModes, sizeof(presentModes));
  if (presentModeToWGPU(mode) == presentMode_) {
    return true;
  }
  // reconfiguring would drop the current frame: the next frame gets the mode, like in the Vulkan backend
  if (isSwapchainTextureAcquired_) {
    isSwapchainReconfigurePending_ = true;
    return true;
  }
  return initSwapchain(swapchainWidth_, swapchainHeight_).isOk();
}

uint32_t lvk::WebGPUContext::getPresentModes(PresentMode* outModes, uint32_t maxModes) const {
  WGPUSurfaceCapabilities caps = {};
  if (!surface_ || wgpuSurfaceGetCapabilities(surface_, adapter_, &caps) != WGPUStatus_Success) {
    return 0;
  }
  uint32_t numModes = 0;
  for (size_t i = 0; i != caps.presentModeCount && numModes != maxModes; i++) {
    outModes[numModes++] = wgpuPresentModeToPresentMode(caps.presentModes[i]);
  }
  wgpuSurfaceCapabilitiesFreeMembers(caps);
  return numModes;
}

lvk::PresentMode lvk::WebGPUContext::getCurrentPresentMode() const {
  return wgpuPresentModeToPresentMode(presentMode_);
}

uint32_t lvk::WebGPUContext::getFramebufferMSAABitMask() const {
  // WebGPU guarantees 1 and 4 samples
  return 1 | 4;
}

lvk::Dimensions lvk::WebGPUContext::getShadingRateAttachmentMinTexelSize() const {
  return {.width = 0, .height = 0, .depth = 0};
}

lvk::Dimensions lvk::WebGPUContext::getShadingRateAttachmentMaxTexelSize() const {
  return {.width = 0, .height = 0, .depth = 0};
}

ldr::Span<const lvk::Dimensions> lvk::WebGPUContext::getSupportedFragmentShadingRates() const {
  return {};
}

lvk::Dimensions lvk::WebGPUContext::getFragmentDensityMapMinTexelSize() const {
  return {.width = 0, .height = 0, .depth = 0};
}

bool lvk::WebGPUContext::isExtensionEnabled(const char*) const {
  return false;
}

bool lvk::WebGPUContext::supportsAsyncCompute() const {
  return false;
}

bool lvk::WebGPUContext::supportsDepthBounds() const {
  return false;
}

bool lvk::WebGPUContext::supportsMeshShader() const {
  return false;
}

bool lvk::WebGPUContext::supportsProvokingVertex() const {
  return false;
}

bool lvk::WebGPUContext::supportsRayQuery() const {
  return false;
}

bool lvk::WebGPUContext::supportsRayTracingPipeline() const {
  return false;
}

bool lvk::WebGPUContext::supportsOpacityMicromap() const {
  return false;
}

bool lvk::WebGPUContext::supportsShaderInterlock() const {
  return false;
}

bool lvk::WebGPUContext::supportsCooperativeMatrix() const {
  return false;
}

bool lvk::WebGPUContext::supportsTextureFormat(Format format, TextureUsageFlags usageFlags) const {
  const WGPUTextureFormat wgpuFormat = formatToWGPU(format);
  if (wgpuFormat == WGPUTextureFormat_Undefined) {
    return false;
  }
  // compressed formats need their device features
  const bool isBC = wgpuFormat >= WGPUTextureFormat_BC1RGBAUnorm && wgpuFormat <= WGPUTextureFormat_BC7RGBAUnormSrgb;
  const bool isETC2 = wgpuFormat >= WGPUTextureFormat_ETC2RGB8Unorm && wgpuFormat <= WGPUTextureFormat_EACRG11Snorm;
  const bool isASTC = wgpuFormat >= WGPUTextureFormat_ASTC4x4Unorm && wgpuFormat <= WGPUTextureFormat_ASTC12x12UnormSrgb;
  const bool hasFeature = (!isBC || wgpuDeviceHasFeature(device_, WGPUFeatureName_TextureCompressionBC)) &&
                          (!isETC2 || wgpuDeviceHasFeature(device_, WGPUFeatureName_TextureCompressionETC2)) &&
                          (!isASTC || wgpuDeviceHasFeature(device_, WGPUFeatureName_TextureCompressionASTC));
  const bool hasFormatFeature =
      (wgpuFormat != WGPUTextureFormat_Depth32FloatStencil8 || wgpuDeviceHasFeature(device_, WGPUFeatureName_Depth32FloatStencil8)) &&
      (wgpuFormat != WGPUTextureFormat_RG11B10Ufloat || !(usageFlags & TextureUsageBits_Attachment) ||
       wgpuDeviceHasFeature(device_, WGPUFeatureName_RG11B10UfloatRenderable));
  const bool isRenderable = !isBC && !isETC2 && !isASTC && wgpuFormat != WGPUTextureFormat_RGBA8Snorm &&
                            wgpuFormat != WGPUTextureFormat_RGB9E5Ufloat;
  const bool isAttachmentUsable = !(usageFlags & TextureUsageBits_Attachment) || isRenderable;
  const bool isStorageUsable = !(usageFlags & TextureUsageBits_Storage) || isStorageSupported(wgpuFormat);
  return hasFeature && hasFormatFeature && isAttachmentUsable && isStorageUsable;
}

bool lvk::WebGPUContext::supportsTensorsARM() const {
  return false;
}

bool lvk::WebGPUContext::supportsDataGraphARM() const {
  return false;
}

lvk::TensorProperties lvk::WebGPUContext::getTensorProperties() const {
  return {};
}

lvk::TensorUsageFlags lvk::WebGPUContext::getTensorFormatSupport(Format, TensorTiling) const {
  return 0;
}

uint32_t lvk::WebGPUContext::getMultiviewMaxViewCount() const {
  return 1;
}

uint32_t lvk::WebGPUContext::getMultiviewMaxMeshViewCount() const {
  return 1;
}

size_t lvk::WebGPUContext::getPipelineCacheData(void*, size_t) const {
  return 0;
}

double lvk::WebGPUContext::getTimestampPeriodToMs() const {
  return 1.0;
}

bool lvk::WebGPUContext::getQueryPoolResults(QueryPoolHandle, uint32_t, uint32_t, size_t, void*, size_t) const {
  LVK_WEBGPU_NOT_IMPLEMENTED();
  return false;
}

// the WebGPU functions of LVK.h

#if LVK_WITH_GLFW || LVK_WITH_SDL3
std::unique_ptr<lvk::IContext> lvk::createWebGPUContextWithSwapchain(LVKwindow* window,
                                                                     uint32_t width,
                                                                     uint32_t height,
                                                                     const lvk::ContextConfig& cfg) {
  LVK_PROFILER_FUNCTION();

  // the surface source structs must outlive the WebGPUContext constructor
#if defined(_WIN32)
#if LVK_WITH_GLFW
  void* hwnd = window ? (void*)glfwGetWin32Window(window) : nullptr;
#else
  void* hwnd = window ? SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr) : nullptr;
#endif // LVK_WITH_GLFW
  const WGPUSurfaceSourceWindowsHWND source = {
      .chain = {.sType = WGPUSType_SurfaceSourceWindowsHWND},
      .hinstance = GetModuleHandleW(nullptr),
      .hwnd = hwnd,
  };
  const bool hasSurface = hwnd != nullptr;
#elif defined(__linux__)
#if defined(LVK_WITH_WAYLAND)
#if LVK_WITH_GLFW
  void* display = window ? (void*)glfwGetWaylandDisplay() : nullptr;
  void* surface = window ? (void*)glfwGetWaylandWindow(window) : nullptr;
#else
  void* display = window ? SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, nullptr)
                         : nullptr;
  void* surface = window ? SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, nullptr)
                         : nullptr;
#endif // LVK_WITH_GLFW
  const WGPUSurfaceSourceWaylandSurface source = {
      .chain = {.sType = WGPUSType_SurfaceSourceWaylandSurface},
      .display = display,
      .surface = surface,
  };
  const bool hasSurface = display && surface;
#else
#if LVK_WITH_GLFW
  void* display = window ? (void*)glfwGetX11Display() : nullptr;
  const uint64_t x11Window = window ? uint64_t(glfwGetX11Window(window)) : 0;
#else
  void* display = window ? SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, nullptr) : nullptr;
  const uint64_t x11Window = window ? uint64_t(SDL_GetNumberProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0))
                                    : 0;
#endif // LVK_WITH_GLFW
  const WGPUSurfaceSourceXlibWindow source = {
      .chain = {.sType = WGPUSType_SurfaceSourceXlibWindow},
      .display = display,
      .window = x11Window,
  };
  const bool hasSurface = display && x11Window;
#endif // LVK_WITH_WAYLAND
#elif defined(__APPLE__)
#if LVK_WITH_GLFW
  void* nativeWindow = window ? (void*)glfwGetCocoaWindow(window) : nullptr;
#else
  void* nativeWindow = window ? SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr)
                              : nullptr;
#endif // LVK_WITH_GLFW
  void* layer = nullptr;
  if (nativeWindow) {
    createCocoaWindowView(nativeWindow, &layer);
  }
  const WGPUSurfaceSourceMetalLayer source = {
      .chain = {.sType = WGPUSType_SurfaceSourceMetalLayer},
      .layer = layer,
  };
  const bool hasSurface = layer != nullptr;
#else
#error Unsupported OS
#endif

  // SDL3's headless mode has an "offscreen" window without native handles
  if (window && !hasSurface) {
    LLOGL("The window has no native handles: using a headless swapchain\n");
  }

  std::unique_ptr<WebGPUContext> ctx = std::make_unique<WebGPUContext>(cfg, hasSurface ? &source.chain : nullptr, width, height);

  if (!ctx->isValid()) {
    LVK_ASSERT_MSG(false, "Cannot create a WebGPU device");
    return nullptr;
  }

  return ctx;
}
#endif // LVK_WITH_GLFW || LVK_WITH_SDL3

uint32_t lvk::getPresentModesWebGPU(const IContext* ctx, PresentMode* outModes, uint32_t maxModes) {
  return ctx ? static_cast<const WebGPUContext*>(ctx)->getPresentModes(outModes, maxModes) : 0;
}

void lvk::setPresentedFramesRetainedWebGPU(IContext* ctx, bool retain) {
  if (ctx) {
    static_cast<WebGPUContext*>(ctx)->setPresentedFramesRetained(retain);
  }
}
