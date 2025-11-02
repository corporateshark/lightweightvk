/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

// Slang
const char* codeSlang = R"(
static const float2 pos[3] = float2[3](
  float2(-0.6, -0.4),
  float2( 0.6, -0.4),
  float2( 0.0,  0.6)
);
static const float3 col[3] = float3[3](
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0)
);

struct VertexStageOutput {
  float4 sv_Position  : SV_Position;
  float3 color        : COLOR0;
};

[shader("vertex")]
VertexStageOutput vertexMain(uint vertexID : SV_VertexID) {
  return {
    float4(pos[vertexID], 0.0, 1.0),
    col[vertexID],
  };
}

[shader("fragment")]
float4 fragmentMain(float3 color : COLOR0) : SV_Target {
  return float4(color, 1.0);
}
)";

// GLSL
const char* codeVS = R"(
#version 460
layout (location=0) out vec3 color;
const vec2 pos[3] = vec2[3](
  vec2(-0.6, -0.4),
  vec2( 0.6, -0.4),
  vec2( 0.0,  0.6)
);
const vec3 col[3] = vec3[3](
  vec3(1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, 0.0, 1.0)
);
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
      .width = 800,
      .height = 600,
      .resizable = true,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

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

    app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
      lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

      // this will clear the framebuffer
      buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}}},
                               {.color = {{.texture = ctx->getCurrentSwapchainTexture()}}});
      buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
      buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
      buffer.cmdDraw(3);
      buffer.cmdPopDebugGroupLabel();
      buffer.cmdEndRendering();
      ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
    });
  }

  VULKAN_APP_EXIT();
}
