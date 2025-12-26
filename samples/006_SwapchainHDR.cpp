/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

const char* codeSlang = R"(
struct PushConstants {
  float3 col[3];
};

[[vk::push_constant]] PushConstants pc;

struct VSOutput {
  float4 sv_Position : SV_Position;
  float3 color : COLOR0;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID) {
  const float2 pos[3] = {
    float2(-0.6, -0.4),
    float2( 0.6, -0.4),
    float2( 0.0,  0.6)
  };

  VSOutput out;
  out.sv_Position = float4(pos[vertexID], 0.0, 1.0);
  out.color = pc.col[vertexID];

  return out;
}

[shader("fragment")]
float4 fragmentMain(float3 color : COLOR0) : SV_Target0 {
  return float4(color, 1.0);
}
)";

const char* codeVS = R"(
#version 460
#extension GL_EXT_scalar_block_layout : require
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
	vec2(-0.6, -0.4),
	vec2( 0.6, -0.4),
	vec2( 0.0,  0.6)
);
layout(scalar, push_constant) uniform constants {
   vec3 col[3];
};

void main() {
	gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
	color = col[gl_VertexIndex];
}
)";

const char* codeFS = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;

void main() {
	out_FragColor = vec4(color, 1.0);
};
)";

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = -80,
      .height = -80,
      .contextConfig =
          {
              .swapchainRequestedColorSpace = lvk::ColorSpace_HDR10,
              //.swapchainRequestedColorSpace = lvk::ColorSpace_SRGB_EXTENDED_LINEAR,
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  LLOGL("Swapchain format     : %u\n", ctx->getSwapchainFormat());
  LLOGL("Swapchain color space: %u\n", ctx->getSwapchainColorSpace());

  {
#if defined(LVK_DEMO_WITH_SLANG)
    lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
#else
    lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

    lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_ = ctx->createRenderPipeline(
        {
            .smVert = vert_,
            .smFrag = frag_,
            .color = {{.format = ctx->getSwapchainFormat()}},
        },
        nullptr);

    LVK_ASSERT(renderPipelineState_Triangle_.valid());

    struct {
      vec3 rgb0;
      vec3 rgb1;
      vec3 rgb2;
    } pc = {
        .rgb0 = {1, 0, 0},
        .rgb1 = {0, 1, 0},
        .rgb2 = {0, 0, 1},
    };

    app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
      lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();
      const lvk::Framebuffer fb = {
          .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
      };
      buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}}}, fb);
      buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
      buffer.cmdPushConstants(pc);
      buffer.cmdDraw(3);
      app.imgui_->beginFrame(fb);
      ImGui::Begin("Colors", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      ImGui::SliderFloat3("Color #0", glm::value_ptr(pc.rgb0), 0.0f, 1.0f);
      ImGui::SliderFloat3("Color #1", glm::value_ptr(pc.rgb1), 0.0f, 1.0f);
      ImGui::SliderFloat3("Color #2", glm::value_ptr(pc.rgb2), 0.0f, 1.0f);
      ImGui::End();
      app.imgui_->endFrame(buffer);
      buffer.cmdEndRendering();
      ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
    });
  }

  VULKAN_APP_EXIT();
}
