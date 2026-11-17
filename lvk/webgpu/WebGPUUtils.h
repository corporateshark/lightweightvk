/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include <lvk/LVK.h>

namespace slang {
struct IGlobalSession;
} // namespace slang

namespace lvk {

// LVK's shader resources in WebGPU:
// - bindless textures and samplers: the resource table, samplers at kSamplersSlotOffset + index
// - buffer device addresses: `vec2u(offset, arena)`; arena `i` is @group(0) @binding(i - 1), writable only in compute
// - push constants: `var<immediate>` if they fit and have no arrays, otherwise `var<storage, read>` at @binding(kNumArenas)
enum : uint32_t {
  kSamplersSlotOffset = 32768,
  kResourceTableSize = 65536,
  kNumArenas = 4,
};

enum PushConstantsLocation : uint8_t {
  PushConstantsLocation_None = 0,
  PushConstantsLocation_Immediates,
  PushConstantsLocation_Buffer, // var<storage, read>
};

// the type of a specialization constant
enum WGSLOverrideType : uint8_t {
  WGSLOverrideType_Bool = 0,
  WGSLOverrideType_Int,
  WGSLOverrideType_Uint,
  WGSLOverrideType_Float,
};

// a specialization constant: `@id(constantId) override`
struct WGSLOverride {
  uint32_t constantId = 0;
  WGSLOverrideType type = WGSLOverrideType_Bool;
};

struct WGSLShader {
  std::string code;
  PushConstantsLocation pushConstants = PushConstantsLocation_None;
  uint32_t pushConstantsSize = 0;
  std::vector<WGSLOverride> overrides; // the specialization constants with a SpecId
};

// SPIR-V, or GLSL/Slang with LVK's preamble -> SPIR-V -> WGSL; `slangGlobalSession` is created on first use
Result compileShaderWGSL(slang::IGlobalSession*& slangGlobalSession,
                         const ShaderModuleDesc& desc,
                         uint32_t maxImmediateSize,
                         WGSLShader* outShader);

void destroySlangGlobalSessionWGSL(slang::IGlobalSession* slangGlobalSession);

} // namespace lvk
