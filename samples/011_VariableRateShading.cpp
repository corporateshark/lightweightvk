/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <filesystem>

// Two ways of varying the shading rate across the screen:
//   VK_EXT_fragment_density_map  - a texture of shading densities, resolved before rasterization (mobile/XR)
//   VK_KHR_fragment_shading_rate - pipeline, per-primitive and attachment rates, combined at draw time
// LVK can enable only one of them at a time (`ContextConfig::enableFragmentShadingRate`), so the UI offers the modes of whichever is
// enabled. The image is shaded into a quarter-resolution texture and magnified 4x, which makes a coarse rate easy to see.

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

// Slang
const char* codeSlang = R"(
struct PushConstants {
  uint texture0;
  uint sampler0;
  uint visualize;
};

[[vk::push_constant]] PushConstants pc;

// a YCbCr sampler must be indexed by a constant expression
[[vk::constant_id(0)]] const uint backgroundTexture = 0;

struct VSOutput {
  float4 sv_Position : SV_Position;
  float2 uv : TEXCOORD0;
};

// fullscreen triangle
VSOutput fullscreenTriangle(uint vertexID) {
  VSOutput out;
  out.uv = float2((vertexID << 1) & 2, vertexID & 2);
  out.sv_Position = float4(out.uv * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID) {
  return fullscreenTriangle(vertexID);
}

struct VSOutputPrimitiveRate {
  float4 sv_Position : SV_Position;
  float2 uv : TEXCOORD0;
  uint shadingRate : SV_ShadingRate; // SV_ShadingRate on a vertex output becomes gl_PrimitiveShadingRateEXT
};

[shader("vertex")]
VSOutputPrimitiveRate vertexMainPrimitiveRate(uint vertexID : SV_VertexID) {
  const uint quad = vertexID / 6;
  const uint v = vertexID % 6;
  const float cx = (v == 1 || v == 3 || v == 4) ? 1.0 : 0.0;
  const float cy = (v == 2 || v == 4 || v == 5) ? 1.0 : 0.0;
  VSOutputPrimitiveRate out;
  out.uv = float2(0.25 * (float(quad) + cx), cy);
  out.sv_Position = float4(2.0 * out.uv.x - 1.0, 2.0 * cy - 1.0, 0.0, 1.0);
  // 1x1, 2x1, 1x2, 2x2 - Vertical2Pixels is bit 0, Horizontal2Pixels is bit 2
  const uint rates[4] = {0, 4, 1, 5};
  out.shadingRate = rates[quad];
  return out;
}

[shader("fragment")]
float4 fragmentMain(float2 uv : TEXCOORD0) : SV_Target0 {
  return kSamplersYUV[backgroundTexture].Sample(uv); // background texture is a YCbCr sampler
}

// SV_ShadingRate on a fragment input becomes gl_ShadingRateEXT
[shader("fragment")]
float4 fragmentMainVisualize(float2 uv : TEXCOORD0, uint shadingRate : SV_ShadingRate) : SV_Target0 {
  if (pc.visualize != 0) {
    // the rate packs log2(width) in bits 2..3 and log2(height) in bits 0..1
    const float w = float(1 << ((shadingRate >> 2) & 3));
    const float h = float(1 << (shadingRate & 3));
    return float4((w - 1.0) / 3.0, (h - 1.0) / 3.0, 0.2, 1.0);
  }
  return kSamplersYUV[backgroundTexture].Sample(uv); // background texture is a YCbCr sampler
}

// magnifies the offscreen texture; a NEAREST sampler keeps every shaded fragment a crisp block
[shader("fragment")]
float4 fragmentMainUpscale(float2 uv : TEXCOORD0) : SV_Target0 {
  return textureBindless2D(pc.texture0, pc.sampler0, uv);
}
)";

// GLSL
const char* codeVS = R"(
layout (location=0) out vec2 uv;

void main() {
  // fullscreen triangle
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
)";

// 4 quads, each requesting its own rate - a pipeline rate is uniform per draw and cannot express this.
// Only takes effect when the primitive combiner is REPLACE.
const char* codeVSPrimitiveRate = R"(
layout (location=0) out vec2 uv;

void main() {
  const uint quad = uint(gl_VertexIndex) / 6u;
  const uint v = uint(gl_VertexIndex) % 6u;
  const float cx = (v == 1u || v == 3u || v == 4u) ? 1.0 : 0.0;
  const float cy = (v == 2u || v == 4u || v == 5u) ? 1.0 : 0.0;
  uv = vec2(0.25 * (float(quad) + cx), cy);
  gl_Position = vec4(2.0 * uv.x - 1.0, 2.0 * cy - 1.0, 0.0, 1.0);
  // 1x1, 2x1, 1x2, 2x2 - the anisotropic rates smear along one axis only
  const int rates[4] = int[4](0,
                              gl_ShadingRateFlag2HorizontalPixelsEXT,
                              gl_ShadingRateFlag2VerticalPixelsEXT,
                              gl_ShadingRateFlag2HorizontalPixelsEXT | gl_ShadingRateFlag2VerticalPixelsEXT);
  gl_PrimitiveShadingRateEXT = rates[quad];
}
)";

const char* codeFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;
layout (constant_id = 0) const uint backgroundTexture = 0; // a YCbCr sampler

void main() {
  out_FragColor = texture(kSamplersYUV[backgroundTexture], uv);
}
)";

// same as codeFS, but can color each fragment by the rate it was shaded at
const char* codeFSVisualize = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform PerFrame {
  uint texture0;
  uint sampler0;
  uint visualize;
} pc;

layout (constant_id = 0) const uint backgroundTexture = 0; // a YCbCr sampler

void main() {
  if (pc.visualize != 0) {
    // gl_ShadingRateEXT packs log2(width) in bits 2..3 and log2(height) in bits 0..1
    const float w = float(1 << ((gl_ShadingRateEXT >> 2) & 3));
    const float h = float(1 << (gl_ShadingRateEXT & 3));
    out_FragColor = vec4((w - 1.0) / 3.0, (h - 1.0) / 3.0, 0.2, 1.0);
    return;
  }
  out_FragColor = texture(kSamplersYUV[backgroundTexture], uv);
}
)";

// magnifies the offscreen texture; a nearest sampler keeps every shaded fragment a crisp block
const char* codeFSUpscale = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform PerFrame {
  uint texture0;
  uint sampler0;
  uint visualize;
} pc;

void main() {
  out_FragColor = textureBindless2D(pc.texture0, pc.sampler0, uv);
}
)";

float radialDistance(uint32_t x, uint32_t y, uint32_t width, uint32_t height) {
  const float nx = ((float)x + 0.5f) / (float)width * 2.0f - 1.0f;
  const float ny = ((float)y + 0.5f) / (float)height * 2.0f - 1.0f;
  return std::min(1.0f, sqrtf(nx * nx + ny * ny));
}

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = -90,
      .height = -90,
      .resizable = true,
      // prefer VK_KHR_fragment_shading_rate; LVK falls back to VK_EXT_fragment_density_map when it is unavailable
      .contextConfig = {.enableFragmentShadingRate = true},
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  {
    const bool hasFSR = ctx->isExtensionEnabled("VK_KHR_fragment_shading_rate");
    const bool hasFDM = ctx->isExtensionEnabled("VK_EXT_fragment_density_map");

    if (hasFSR) {
      LLOGL("VK_KHR_fragment_shading_rate enabled\n");
    } else if (hasFDM) {
      LLOGL("VK_EXT_fragment_density_map enabled - using a full-resolution fragment density map\n");
    } else {
      LLOGW("Neither VK_KHR_fragment_shading_rate nor VK_EXT_fragment_density_map is supported - rendering at full rate\n");
    }

    // one texel of the shading rate attachment covers this many pixels; must be within the min/max the device reports
    const lvk::Dimensions fsrTexelSize = hasFSR ? ctx->getShadingRateAttachmentMinTexelSize() : lvk::Dimensions{};

#if defined(LVK_DEMO_WITH_SLANG)
    const lvk::ShaderModuleDesc descVert = {codeSlang, "vertexMain", lvk::Stage_Vert, "Shader Module: main (vert)"};
    const lvk::ShaderModuleDesc descVertRate = {
        codeSlang, "vertexMainPrimitiveRate", lvk::Stage_Vert, "Shader Module: primitive rate (vert)"};
    const lvk::ShaderModuleDesc descFrag = {
        codeSlang, hasFSR ? "fragmentMainVisualize" : "fragmentMain", lvk::Stage_Frag, "Shader Module: main (frag)"};
    const lvk::ShaderModuleDesc descFragUpscale = {codeSlang, "fragmentMainUpscale", lvk::Stage_Frag, "Shader Module: upscale (frag)"};
#else
    const lvk::ShaderModuleDesc descVert = {codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"};
    const lvk::ShaderModuleDesc descVertRate = {codeVSPrimitiveRate, lvk::Stage_Vert, "Shader Module: primitive rate (vert)"};
    const lvk::ShaderModuleDesc descFrag = {hasFSR ? codeFSVisualize : codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"};
    const lvk::ShaderModuleDesc descFragUpscale = {codeFSUpscale, lvk::Stage_Frag, "Shader Module: upscale (frag)"};
#endif // defined(LVK_DEMO_WITH_SLANG)

    const lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule(descVert);
    const lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule(descFrag);
    const lvk::Holder<lvk::ShaderModuleHandle> fragUpscale_ = ctx->createShaderModule(descFragUpscale);
    const lvk::Holder<lvk::ShaderModuleHandle> vertRate_ = hasFSR ? ctx->createShaderModule(descVertRate)
                                                                  : lvk::Holder<lvk::ShaderModuleHandle>{};

    // the 1920x1080 NV12 frame from the YUV demo - a photo shows a coarse rate as blocky edges
    const std::string filePath = (std::filesystem::path(app.folderContentRoot_) / "src" / "igl-samples/output_frame_900.nv12.yuv").string();
    const std::vector<uint8_t> pixels = app.loadFile(filePath.c_str());
    LVK_ASSERT_MSG(!pixels.empty(), "Cannot load the texture. Run `deploy_content.py`/`deploy_content_android.py` first.");
    if (pixels.empty()) {
      printf("Cannot load the texture. Run `deploy_content.py`/`deploy_content_android.py` first.\n");
      std::terminate();
    }
    const lvk::Holder<lvk::TextureHandle> texture_ = ctx->createTexture({
        .format = lvk::Format_YUV_NV12,
        .dimensions = {1920, 1080},
        .usage = lvk::TextureUsageBits_Sampled,
        .data = pixels.data(),
        .debugName = "Background (NV12)",
    });
    // nearest filtering turns the magnified offscreen texture into crisp blocks
    const lvk::Holder<lvk::SamplerHandle> sampler_ = ctx->createSampler({
        .minFilter = VK_FILTER_NEAREST,
        .magFilter = VK_FILTER_NEAREST,
        .debugName = "Sampler: nearest",
    });

    const uint32_t backgroundTextureId = texture_.index();
    const lvk::SpecializationConstantDesc specBackground = {
        .entries = {{.constantId = 0, .size = sizeof(uint32_t)}}, .data = &backgroundTextureId, .dataSize = sizeof(backgroundTextureId)};
    const auto createPipeline = [&](lvk::ShaderModuleHandle vert, lvk::ShaderModuleHandle frag, bool withBackground) {
      return ctx->createRenderPipeline({
          .smVert = vert,
          .smFrag = frag,
          .specInfo = withBackground ? specBackground : lvk::SpecializationConstantDesc{},
          .color = {{.format = ctx->getSwapchainFormat()}},
      });
    };
    const lvk::Holder<lvk::RenderPipelineHandle> pipeline_ = createPipeline(vert_, frag_, true);
    const lvk::Holder<lvk::RenderPipelineHandle> pipelineUpscale_ = createPipeline(vert_, fragUpscale_, false);
    const lvk::Holder<lvk::RenderPipelineHandle> pipelinePrimitiveRate_ = hasFSR ? createPipeline(vertRate_, frag_, true)
                                                                                 : lvk::Holder<lvk::RenderPipelineHandle>{};
    LVK_ASSERT(pipeline_.valid());

    // recreated whenever the swapchain size changes
    lvk::Holder<lvk::TextureHandle> offscreen_;
    lvk::Holder<lvk::TextureHandle> densityMap_;
    lvk::Holder<lvk::TextureHandle> shadingRateMap_;
    lvk::Dimensions offscreenSize = {0, 0};

    // maxFragmentDensityTexelSize is at least (1,1) and the map has only a lower size bound, so a full-resolution map is always valid.
    // A real renderer would query the limit and use a much smaller map.
    const auto createDensityMap = [&](const lvk::Dimensions& size) {
      // RG_UN8: R = shading density along X, G = along Y, both in (0..1], 1.0 = full rate
      std::vector<uint8_t> texels(size.width * size.height * 2);
      for (uint32_t y = 0; y != size.height; y++) {
        for (uint32_t x = 0; x != size.width; x++) {
          // full density in the center falling off towards the edges; the distance is in [0, 1] so no clamping is needed
          const uint8_t value = (uint8_t)(glm::mix(1.0f, 0.25f, radialDistance(x, y, size.width, size.height)) * 255.0f + 0.5f);
          texels[(y * size.width + x) * 2 + 0] = value;
          texels[(y * size.width + x) * 2 + 1] = value;
        }
      }
      densityMap_ = ctx->createTexture({
          .format = lvk::Format_RG_UN8,
          .dimensions = size,
          .usage = lvk::TextureUsageBits_FragmentDensityMap,
          .data = texels.data(),
          .debugName = "Fragment Density Map",
      });
    };

    // one texel of the map covers `fsrTexelSize` pixels
    const auto createShadingRateMap = [&](const lvk::Dimensions& size) {
      const lvk::Dimensions mapSize = {(size.width + fsrTexelSize.width - 1) / fsrTexelSize.width,
                                       (size.height + fsrTexelSize.height - 1) / fsrTexelSize.height};
      std::vector<uint8_t> texels(mapSize.width * mapSize.height);
      for (uint32_t y = 0; y != mapSize.height; y++) {
        for (uint32_t x = 0; x != mapSize.width; x++) {
          // 1x1 in the center, then 2x2, then 4x4 towards the edges
          const float dist = radialDistance(x, y, mapSize.width, mapSize.height);
          const uint32_t log2Size = dist < 0.33f ? 0u : (dist < 0.66f ? 1u : 2u);
          // the spec mandates the encoding: log2(width) in bits 2..3, log2(height) in bits 0..1
          texels[y * mapSize.width + x] = (uint8_t)((log2Size << 2) | log2Size);
        }
      }
      shadingRateMap_ = ctx->createTexture({
          .format = lvk::Format_R_UI8,
          .dimensions = mapSize,
          .usage = lvk::TextureUsageBits_ShadingRateAttachment,
          .data = texels.data(),
          .debugName = "Shading Rate Attachment",
      });
    };

    enum Mode { Mode_Off = 0, Mode_Pipeline, Mode_Primitive, Mode_Attachment, Mode_DensityMap };

    int mode = hasFSR ? Mode_Attachment : (hasFDM ? Mode_DensityMap : Mode_Off);

    struct {
      uint32_t texture0 = 0;
      uint32_t sampler0 = 0;
      uint32_t visualize = 0;
    } pc = {
        .sampler0 = sampler_.index(),
    };

    app.run([&](ldr::Span<const RenderView> views, float deltaSeconds) {
      const lvk::TextureHandle swapchain = ctx->getCurrentSwapchainTexture();
      const lvk::Dimensions dim = ctx->getDimensions(swapchain);

      // shading at quarter resolution scales every shaded fragment up 4x4
      const lvk::Dimensions quarter = {std::max(1u, dim.width / 4), std::max(1u, dim.height / 4)};

      if (quarter != offscreenSize) {
        offscreen_ = ctx->createTexture({
            .format = ctx->getSwapchainFormat(),
            .dimensions = quarter,
            .usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
            .debugName = "Offscreen (quarter resolution)",
        });
        if (hasFDM) {
          createDensityMap(quarter);
        }
        if (hasFSR) {
          createShadingRateMap(quarter);
        }
        offscreenSize = quarter;
      }

      const lvk::Framebuffer framebuffer = {
          .color = {{.texture = offscreen_}},
          .fragmentDensityMap = mode == Mode_DensityMap ? densityMap_ : lvk::TextureHandle{},
          .shadingRateAttachment = mode == Mode_Attachment ? shadingRateMap_ : lvk::TextureHandle{},
          .shadingRateAttachmentTexelSize = mode == Mode_Attachment ? fsrTexelSize : lvk::Dimensions{},
      };

      lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

      // 1. shade the background at quarter resolution
      buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}}}, framebuffer);
      buffer.cmdBindRenderPipeline(mode == Mode_Primitive ? pipelinePrimitiveRate_ : pipeline_);
      if (hasFSR) {
        // the pipeline rate is the first combiner input and the primitive rate the second: KEEP the first, REPLACE the second
        buffer.cmdSetFragmentShadingRate(
            mode == Mode_Pipeline ? lvk::Dimensions{2, 2} : lvk::Dimensions{1, 1},
            mode == Mode_Primitive ? VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR : VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
            mode == Mode_Attachment ? VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR : VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR);
        buffer.cmdPushConstants(pc); // only `visualize` is read here - the background is a specialization constant
      }
      buffer.cmdPushDebugGroupLabel("Shading rate", 0xff0000ff);
      buffer.cmdDraw(mode == Mode_Primitive ? 6 * 4 : 3); // the primitive rate mode needs one quad per rate
      buffer.cmdPopDebugGroupLabel();
      buffer.cmdEndRendering();

      // 2. magnify it into the swapchain and draw the UI on top (both at full shading rate)
      const lvk::Framebuffer fbSwapchain = {.color = {{.texture = swapchain}}};
      buffer.cmdBeginRendering(
          {.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}}}, fbSwapchain, {.sampledImages = {offscreen_}});
      buffer.cmdBindRenderPipeline(pipelineUpscale_);
      pc.texture0 = offscreen_.index();
      buffer.cmdPushConstants(pc);
      buffer.cmdDraw(3);

      app.imgui_->beginFrame(fbSwapchain);
      ImGui::Begin("Shading rate", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::RadioButton("Off (1x1)", &mode, Mode_Off);
      ImGui::BeginDisabled(!hasFSR);
      ImGui::RadioButton("Pipeline rate (2x2)", &mode, Mode_Pipeline);
      ImGui::RadioButton("Primitive rate (per-primitive)", &mode, Mode_Primitive);
      ImGui::RadioButton("Rate attachment (foveated)", &mode, Mode_Attachment);
      ImGui::EndDisabled();
      ImGui::BeginDisabled(!hasFDM);
      ImGui::RadioButton("Fragment density map (foveated)", &mode, Mode_DensityMap);
      ImGui::EndDisabled();
      if (hasFSR) {
        bool visualize = pc.visualize != 0;
        ImGui::Separator();
        ImGui::Checkbox("Visualize gl_ShadingRateEXT", &visualize);
        pc.visualize = visualize ? 1u : 0u;
      }
      ImGui::End();
      app.drawFPS();
      app.imgui_->endFrame(buffer);

      buffer.cmdEndRendering();
      ctx->submit(buffer, swapchain);
    });
  }

  VULKAN_APP_EXIT();
}
