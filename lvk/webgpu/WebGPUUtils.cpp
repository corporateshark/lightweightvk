/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "WebGPUUtils.h"

#include <cstring>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include <ldrutils/lutils/ScopeExit.h>
#include <spirv_wgsl.hpp>

#if defined(LVK_WITH_SLANG)
#include <slang-com-ptr.h>
#include <slang.h>
#endif // LVK_WITH_SLANG

namespace {

// the GLSL/Slang preambles of the Vulkan backend: keep them in sync

std::string addShaderPreambleGLSL(lvk::ShaderStage stage, const char* source) {
  std::string sourcePatched;

  auto addCode = [source, &sourcePatched](const char* substr, const char* code) -> void {
    if (strstr(source, substr)) {
      sourcePatched.append(code);
    }
  };

  if (strstr(source, "#version ") == nullptr) {
    if (stage == lvk::Stage_Vert || stage == lvk::Stage_Comp) {
      sourcePatched +=
          "#version 460\n"
          "#extension GL_EXT_buffer_reference : require\n"
          "#extension GL_EXT_buffer_reference_uvec2 : require\n"
          "#extension GL_EXT_debug_printf : enable\n"
          "#extension GL_EXT_nonuniform_qualifier : require\n"
          "#extension GL_EXT_samplerless_texture_functions : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    }
    if (stage == lvk::Stage_Frag) {
      // Note how nonuniformEXT() should be used:
      // https://github.com/KhronosGroup/Vulkan-Samples/blob/main/shaders/descriptor_indexing/glsl/nonuniform-quads.frag#L38
      sourcePatched +=
          "#version 460\n"
          "#extension GL_EXT_buffer_reference_uvec2 : require\n"
          "#extension GL_EXT_debug_printf : enable\n"
          "#extension GL_EXT_nonuniform_qualifier : require\n"
          "#extension GL_EXT_samplerless_texture_functions : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n"
          "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable\n"
          "#extension GL_EXT_shader_atomic_int64 : enable\n";
      sourcePatched +=
          "layout (set = 0, binding = 0) uniform texture2D   kTextures2D[];\n"
          "layout (set = 0, binding = 0) uniform texture3D   kTextures3D[];\n"
          "layout (set = 0, binding = 0) uniform textureCube kTexturesCube[];\n"
          "layout (set = 0, binding = 0) uniform texture2D   kTextures2DShadow[];\n"
          "layout (set = 0, binding = 0) uniform utexture2D  kTextures2DUint[];\n" // integer format cannot be filtered
          "layout (set = 0, binding = 0) uniform itexture2D  kTextures2DInt[];\n" // integer format cannot be filtered
          "layout (set = 0, binding = 1) uniform sampler       kSamplers[];\n"
          "layout (set = 0, binding = 1) uniform samplerShadow kSamplersShadow[];\n"
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
      addCode("texelFetchBindlessUint2D(",
              "uvec4 texelFetchBindlessUint2D(uint textureid, ivec2 coord, int lod) {\n"
              "  return texelFetch(nonuniformEXT(kTextures2DUint[textureid]), coord, lod);\n"
              "}\n");
      addCode("texelFetchBindlessInt2D(",
              "ivec4 texelFetchBindlessInt2D(uint textureid, ivec2 coord, int lod) {\n"
              "  return texelFetch(nonuniformEXT(kTextures2DInt[textureid]), coord, lod);\n"
              "}\n");
      addCode("textureBindlessCube(",
              "vec4 textureBindlessCube(uint textureid, uint samplerid, vec3 uvw) {\n"
              "  return texture(nonuniformEXT(samplerCube(kTexturesCube[textureid], kSamplers[samplerid])), uvw);\n"
              "}\n");
      addCode("textureBindlessCubeLod(",
              "vec4 textureBindlessCubeLod(uint textureid, uint samplerid, vec3 uvw, float lod) {\n"
              "  return textureLod(nonuniformEXT(samplerCube(kTexturesCube[textureid], kSamplers[samplerid])), uvw, lod);\n"
              "}\n");
      addCode("textureBindless3D(",
              "vec4 textureBindless3D(uint textureid, uint samplerid, vec3 uvw) {\n"
              "  return texture(nonuniformEXT(sampler3D(kTextures3D[textureid], kSamplers[samplerid])), uvw);\n"
              "}\n");
      addCode("textureBindless3DLod(",
              "vec4 textureBindless3DLod(uint textureid, uint samplerid, vec3 uvw, float lod) {\n"
              "  return textureLod(nonuniformEXT(sampler3D(kTextures3D[textureid], kSamplers[samplerid])), uvw, lod);\n"
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
  }
  sourcePatched += source;

  return sourcePatched;
}

std::string addShaderPreambleSlang(const char* source) {
  std::string sourcePatched;

  auto addCode = [source, &sourcePatched](const char* substr, const char* code) -> void {
    if (strstr(source, substr)) {
      sourcePatched.append(code);
    }
  };

  // Slang `v2026.11+` hard-codes a fast path for builtin matrix operators which skips `operator*` overload resolution
  // (https://github.com/shader-slang/slang/pull/11493 and https://github.com/shader-slang/slang/issues/11877).
  // Importing `glsl` puts the module into "GLSL operator scope", so the overloads below are used again.
  sourcePatched += "import glsl;\n";
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
      "[[vk::binding(0, 0)]] Texture3D    kTextures3D[];\n"
      "[[vk::binding(0, 0)]] TextureCube  kTexturesCube[];\n"
      "[[vk::binding(0, 0)]] Texture2D    kTextures2DShadow[];\n"
      "[[vk::binding(0, 0)]] Texture2D<uint4> kTextures2DUint[];\n"
      "[[vk::binding(0, 0)]] Texture2D<int4>  kTextures2DInt[];\n"
      "[[vk::binding(1, 0)]] SamplerState kSamplers[];\n"
      "[[vk::binding(1, 0)]] SamplerComparisonState kSamplersShadow[];\n"
      "[[vk::binding(3, 0)]] Sampler2D    kSamplersYUV[];\n";
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
  addCode("texelFetchBindlessUint2D(",
          "uint4 texelFetchBindlessUint2D(uint textureid, int2 coord, int lod) {\n"
          "  return kTextures2DUint[NonUniformResourceIndex(textureid)].Load(int3(coord, lod));\n"
          "}\n");
  addCode("texelFetchBindlessInt2D(",
          "int4 texelFetchBindlessInt2D(uint textureid, int2 coord, int lod) {\n"
          "  return kTextures2DInt[NonUniformResourceIndex(textureid)].Load(int3(coord, lod));\n"
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
  addCode("textureBindless3D(",
          "float4 textureBindless3D(uint textureid, uint samplerid, float3 uvw) {\n"
          "  return kTextures3D[NonUniformResourceIndex(textureid)].Sample(\n"
          "    kSamplers[NonUniformResourceIndex(samplerid)], uvw);\n"
          "}\n");
  addCode("textureBindless3DLod(",
          "float4 textureBindless3DLod(uint textureid, uint samplerid, float3 uvw, float lod) {\n"
          "  return kTextures3D[NonUniformResourceIndex(textureid)].SampleLevel(\n"
          "    kSamplers[NonUniformResourceIndex(samplerid)], uvw, lod);\n"
          "}\n");

  sourcePatched += source;

  return sourcePatched;
}

glslang_stage_t getGlslangStage(lvk::ShaderStage stage) {
  switch (stage) {
  case lvk::Stage_Frag:
    return GLSLANG_STAGE_FRAGMENT;
  case lvk::Stage_Comp:
    return GLSLANG_STAGE_COMPUTE;
  default:
    return GLSLANG_STAGE_VERTEX;
  }
}

// the settings of the Vulkan backend; debug names keep the WGSL readable
// glslang_initialize_process() is called by WebGPUContext
lvk::Result compileGLSL(lvk::ShaderStage stage, const char* code, std::vector<uint32_t>* outSPIRV) {
  const glslang_input_t input = {
      .language = GLSLANG_SOURCE_GLSL,
      .stage = getGlslangStage(stage),
      .client = GLSLANG_CLIENT_VULKAN,
      .client_version = GLSLANG_TARGET_VULKAN_1_3,
      .target_language = GLSLANG_TARGET_SPV,
      .target_language_version = GLSLANG_TARGET_SPV_1_6,
      .code = code,
      .default_version = 100,
      .default_profile = GLSLANG_NO_PROFILE,
      .force_default_version_and_profile = false,
      .forward_compatible = false,
      .messages = GLSLANG_MSG_DEFAULT_BIT,
      .resource = glslang_default_resource(),
  };

  glslang_shader_t* shader = glslang_shader_create(&input);
  SCOPE_EXIT {
    glslang_shader_delete(shader);
  };

  if (!glslang_shader_preprocess(shader, &input)) {
    LLOGW("Shader preprocessing failed:\n%s\n%s\n", glslang_shader_get_info_log(shader), glslang_shader_get_info_debug_log(shader));
    return lvk::Result(lvk::Result::Code::RuntimeError, "glslang_shader_preprocess() failed");
  }

  if (!glslang_shader_parse(shader, &input)) {
    LLOGW("Shader parsing failed:\n%s\n%s\n", glslang_shader_get_info_log(shader), glslang_shader_get_info_debug_log(shader));
    return lvk::Result(lvk::Result::Code::RuntimeError, "glslang_shader_parse() failed");
  }

  glslang_program_t* program = glslang_program_create();
  SCOPE_EXIT {
    glslang_program_delete(program);
  };
  glslang_program_add_shader(program, shader);

  if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
    LLOGW("Shader linking failed:\n%s\n%s\n", glslang_program_get_info_log(program), glslang_program_get_info_debug_log(program));
    return lvk::Result(lvk::Result::Code::RuntimeError, "glslang_program_link() failed");
  }

  glslang_spv_options_t options = {
      .generate_debug_info = true,
      .strip_debug_info = false,
      .disable_optimizer = true,
      .optimize_size = false,
      .disassemble = false,
      .validate = true,
      .emit_nonsemantic_shader_debug_info = false,
      .emit_nonsemantic_shader_debug_source = false,
  };

  glslang_program_SPIRV_generate_with_options(program, input.stage, &options);

  if (glslang_program_SPIRV_get_messages(program)) {
    LLOGW("%s\n", glslang_program_SPIRV_get_messages(program));
  }

  const uint32_t* spirv = glslang_program_SPIRV_get_ptr(program);
  *outSPIRV = std::vector<uint32_t>(spirv, spirv + glslang_program_SPIRV_get_size(program));

  return lvk::Result();
}

#if defined(LVK_WITH_SLANG)
// the settings of the Vulkan backend
lvk::Result compileSlang(slang::IGlobalSession*& slangGlobalSession,
                         const char* code,
                         const char* entryPointName,
                         std::vector<uint32_t>* outSPIRV) {
  if (!slangGlobalSession) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    // `enableGLSL` allows the preamble's `import glsl;`
    const SlangGlobalSessionDesc globalSessionDesc = {
        .enableGLSL = true,
    };
    if (SLANG_FAILED(slang::createGlobalSession(&globalSessionDesc, globalSession.writeRef()))) {
      return lvk::Result(lvk::Result::Code::RuntimeError, "slang::createGlobalSession() failed");
    }
    slangGlobalSession = globalSession.detach();
  }

  const slang::CompilerOptionEntry compilerOptions[] = {
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "SPV_GOOGLE_user_type"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvDerivativeControl"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvImageQuery"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvImageGatherExtended"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvSparseResidency"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvMinLod"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvFragmentFullyCoveredEXT"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvRayTracingPositionFetchKHR"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvRayQueryKHR"}},
      {.name = slang::CompilerOptionName::Capability,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "spvInt64Atomics"}},
  };

  const slang::TargetDesc targetDesc = {
      .format = SLANG_SPIRV,
      .profile = slangGlobalSession->findProfile("spirv_1_6"),
      .flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY,
      .forceGLSLScalarBufferLayout = true,
      .compilerOptionEntries = &compilerOptions[0],
      .compilerOptionEntryCount = LVK_ARRAY_NUM_ELEMENTS(compilerOptions),
  };

  // descriptor aliasing (multiple bindless arrays sharing a binding) is intended
  const slang::CompilerOptionEntry sessionOptions[] = {
      {.name = slang::CompilerOptionName::DisableWarnings,
       .value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = "39001"}},
  };

  const slang::SessionDesc sessionDesc = {
      .targets = &targetDesc,
      .targetCount = 1,
      .compilerOptionEntries = sessionOptions,
      .compilerOptionEntryCount = LVK_ARRAY_NUM_ELEMENTS(sessionOptions),
  };

  Slang::ComPtr<slang::ISession> session;
  if (SLANG_FAILED(slangGlobalSession->createSession(sessionDesc, session.writeRef()))) {
    return lvk::Result(lvk::Result::Code::RuntimeError, "slang::createSession() failed");
  }

  Slang::ComPtr<slang::IBlob> diagnostics;
  slang::IModule* slangModule = session->loadModuleFromSourceString("", "", code, diagnostics.writeRef());
  if (diagnostics) {
    LLOGW("%s", (const char*)diagnostics->getBufferPointer());
  }
  if (!slangModule) {
    return lvk::Result(lvk::Result::Code::RuntimeError, "slang::loadModuleFromSourceString() failed");
  }

  Slang::ComPtr<slang::IEntryPoint> entryPoint;
  if (SLANG_FAILED(slangModule->findEntryPointByName(entryPointName, entryPoint.writeRef()))) {
    LLOGW("Entry point %s() not found\n", entryPointName);
    return lvk::Result(lvk::Result::Code::RuntimeError, "Entry point not found");
  }

  slang::IComponentType* componentTypes[] = {slangModule, entryPoint};
  Slang::ComPtr<slang::IComponentType> composedProgram;
  {
    Slang::ComPtr<slang::IBlob> diagnosticBlob;
    const SlangResult result = session->createCompositeComponentType(
        componentTypes, LVK_ARRAY_NUM_ELEMENTS(componentTypes), composedProgram.writeRef(), diagnosticBlob.writeRef());
    if (diagnosticBlob) {
      LLOGW("%s\n", (const char*)diagnosticBlob->getBufferPointer());
    }
    if (SLANG_FAILED(result)) {
      return lvk::Result(lvk::Result::Code::RuntimeError, "slang::createCompositeComponentType() failed");
    }
  }

  Slang::ComPtr<slang::IBlob> spirvCode;
  {
    Slang::ComPtr<slang::IBlob> diagnosticBlob;
    const SlangResult result = composedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticBlob.writeRef());
    if (diagnosticBlob) {
      LLOGW("%s\n", (const char*)diagnosticBlob->getBufferPointer());
    }
    if (SLANG_FAILED(result)) {
      return lvk::Result(lvk::Result::Code::RuntimeError, "slang::getEntryPointCode() failed");
    }
  }

  const uint32_t* spirv = reinterpret_cast<const uint32_t*>(spirvCode->getBufferPointer());
  *outSPIRV = std::vector<uint32_t>(spirv, spirv + spirvCode->getBufferSize() / sizeof(uint32_t));

  return lvk::Result();
}
#endif // LVK_WITH_SLANG

// WGSL immediates cannot have arrays, also not inside structs
bool hasArrays(const spirv_cross::Compiler& compiler, uint32_t typeId) {
  const spirv_cross::SPIRType& type = compiler.get_type(typeId);
  if (!type.array.empty()) {
    return true;
  }
  // a buffer device address, not the struct it points to
  const bool isPhysicalPointer = type.pointer && type.storage == spv::StorageClassPhysicalStorageBuffer;
  if (isPhysicalPointer) {
    return false;
  }
  for (const spirv_cross::TypeID member : type.member_types) {
    if (hasArrays(compiler, member)) {
      return true;
    }
  }
  return false;
}

// WGSL overrides are 32-bit scalars; compile() fails on specialization constants of other types
std::vector<lvk::WGSLOverride> getOverrides(const spirv_cross::Compiler& compiler) {
  std::vector<lvk::WGSLOverride> overrides;
  for (const spirv_cross::SpecializationConstant& sc : compiler.get_specialization_constants()) {
    const spirv_cross::SPIRType::BaseType baseType = compiler.get_type(compiler.get_constant(sc.id).constant_type).basetype;
    overrides.push_back({
        .constantId = sc.constant_id,
        .type = baseType == spirv_cross::SPIRType::Boolean ? lvk::WGSLOverrideType_Bool
                : baseType == spirv_cross::SPIRType::Int   ? lvk::WGSLOverrideType_Int
                : baseType == spirv_cross::SPIRType::UInt  ? lvk::WGSLOverrideType_Uint
                                                           : lvk::WGSLOverrideType_Float,
    });
  }
  return overrides;
}

lvk::Result translateSPIRV(const std::vector<uint32_t>& words, uint32_t maxImmediateSize, lvk::WGSLShader* outShader) {
  try {
    spirv_cross::CompilerWGSL compiler(words);

    // at most one push constant block
    const spirv_cross::SmallVector<spirv_cross::Resource> pcs = compiler.get_shader_resources().push_constant_buffers;
    const uint32_t pushConstantsSize = pcs.empty() ? 0
                                                   : uint32_t(compiler.get_declared_struct_size(compiler.get_type(pcs[0].base_type_id)));
    const bool pushConstantsHaveArrays = !pcs.empty() && hasArrays(compiler, pcs[0].base_type_id);
    // a WGSL struct's size is rounded up to its alignment, at most 16 bytes
    const uint32_t wgslPushConstantsSize = (pushConstantsSize + 15) / 16 * 16;
    const bool asImmediates = pushConstantsSize && !pushConstantsHaveArrays && wgslPushConstantsSize <= maxImmediateSize;
    // vertex shaders cannot write storage buffers
    const bool isCompute = compiler.get_execution_model() == spv::ExecutionModelGLCompute;

    // the sampling resource table has no storage textures
    const spirv_cross::ShaderResources resources = compiler.get_shader_resources(compiler.get_active_interface_variables());
    if (!resources.storage_images.empty()) {
      LLOGW("SPIR-V -> WGSL translation failed: the WebGPU backend does not support storage images (`%s`)\n",
            resources.storage_images[0].name.c_str());
      return lvk::Result(lvk::Result::Code::RuntimeError, "Storage images are not supported by the WebGPU backend");
    }

    const spirv_cross::CompilerWGSL::Options opts = {
        .push_constant_group = 0,
        .push_constant_binding = lvk::kNumArenas,
        .use_resource_table = true,
        .resource_table_group = 0,
        .resource_table_binding_offsets = {{1, lvk::kSamplersSlotOffset}},
        .bda_arena_count = lvk::kNumArenas,
        .bda_arena_group = 0,
        .bda_arena_binding = 0,
        .bda_arenas_writable = isCompute,
        .push_constants_as_immediates = asImmediates,
        .immediate_size_limit = maxImmediateSize,
        .push_constants_as_storage = true,
    };
    compiler.set_wgsl_options(opts);
    std::string code = compiler.compile();
    for (const spirv_cross::CompilerWGSL::Warning& warning : compiler.get_warnings()) {
      LLOGW("WGSL: %s\n", warning.message.c_str());
    }
    *outShader = {
        .code = std::move(code),
        .pushConstants = !pushConstantsSize ? lvk::PushConstantsLocation_None
                         : asImmediates     ? lvk::PushConstantsLocation_Immediates
                                            : lvk::PushConstantsLocation_Buffer,
        .pushConstantsSize = pushConstantsSize,
        .overrides = getOverrides(compiler),
    };
  } catch (const std::exception& e) {
    LLOGW("SPIR-V -> WGSL translation failed: %s\n", e.what());
    return lvk::Result(lvk::Result::Code::RuntimeError, "SPIR-V -> WGSL translation failed");
  }
  return lvk::Result();
}

} // namespace

lvk::Result lvk::compileShaderWGSL(slang::IGlobalSession*& slangGlobalSession,
                                   const ShaderModuleDesc& desc,
                                   uint32_t maxImmediateSize,
                                   WGSLShader* outShader) {
  LVK_PROFILER_FUNCTION();

  if (!desc.data || (!desc.dataSize && !*desc.data) || desc.dataSize % sizeof(uint32_t)) {
    return Result(Result::Code::ArgumentOutOfRange, "Expecting non-empty shader code");
  }

  std::vector<uint32_t> spirv;

  if (desc.dataSize) {
    // copied to guarantee the alignment of SPIR-V words
    spirv.resize(desc.dataSize / sizeof(uint32_t));
    memcpy(spirv.data(), desc.data, desc.dataSize);
    return translateSPIRV(spirv, maxImmediateSize, outShader);
  }

  if (desc.stage != Stage_Vert && desc.stage != Stage_Frag && desc.stage != Stage_Comp) {
    return Result(Result::Code::ArgumentOutOfRange, "WebGPU supports only vertex, fragment, and compute shaders");
  }

  Result result;

  if (strstr(desc.data, "[shader(\"")) {
#if defined(LVK_WITH_SLANG)
    // the same default entry points as the Vulkan backend
    const char* entryPoint = desc.entryPointName        ? desc.entryPointName
                             : desc.stage == Stage_Vert ? "vertexMain"
                             : desc.stage == Stage_Frag ? "fragmentMain"
                                                        : "computeMain";
    result = compileSlang(slangGlobalSession, addShaderPreambleSlang(desc.data).c_str(), entryPoint, &spirv);
#else
    result = Result(Result::Code::RuntimeError, "Slang shaders require LVK_WITH_SLANG=ON");
#endif // LVK_WITH_SLANG
  } else {
    result = compileGLSL(desc.stage, addShaderPreambleGLSL(desc.stage, desc.data).c_str(), &spirv);
  }

  return result.isOk() ? translateSPIRV(spirv, maxImmediateSize, outShader) : result;
}

void lvk::destroySlangGlobalSessionWGSL(slang::IGlobalSession* slangGlobalSession) {
#if defined(LVK_WITH_SLANG)
  if (slangGlobalSession) {
    slangGlobalSession->release();
  }
#endif // LVK_WITH_SLANG
}
