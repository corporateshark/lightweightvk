/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <lvk/vulkan/VulkanClasses.h>

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

const char* codeTriangleSlang = R"(
struct PushConstants {
  float time;
};

[[vk::push_constant]] PushConstants pc;

static const float2 pos[3] = {
  float2(-0.6, -0.6),
  float2( 0.6, -0.6),
  float2( 0.0,  0.6)
};

struct FragmentOutput {
  float4 color0 : SV_Target0;
  float4 color1 : SV_Target1;
  float4 color2 : SV_Target2;
  float4 color3 : SV_Target3;
  float4 color4 : SV_Target4;
  float4 color5 : SV_Target5;
};

[shader("vertex")]
float4 vertexMain(uint vertexID : SV_VertexID) : SV_Position {
  return float4(pos[vertexID] * (1.5 + sin(pc.time)) * 0.5, 0.0, 1.0);
}

[shader("fragment")]
FragmentOutput fragmentMain() {
  return {
    float4(1.0, 0.0, 0.0, 1.0),
    float4(0.0, 1.0, 0.0, 1.0),
    float4(0.0, 0.0, 1.0, 1.0),
    float4(1.0, 0.0, 1.0, 1.0),
    float4(1.0, 1.0, 0.0, 1.0),
    float4(0.0, 1.0, 1.0, 1.0),
  };
}
)";

const char* codeTriangleVS = R"(
#version 460
const vec2 pos[3] = vec2[3](
  vec2(-0.6, -0.6),
  vec2( 0.6, -0.6),
  vec2( 0.0,  0.6)
);
layout(push_constant) uniform constants {
  float time;
} pc;

void main() {
  gl_Position = vec4(pos[gl_VertexIndex] * (1.5 + sin(pc.time)) * 0.5, 0.0, 1.0);
}
)";

const char* codeTriangleFS = R"(
#version 460
layout (location=0) out vec4 out_FragColor0;
layout (location=1) out vec4 out_FragColor1;
layout (location=2) out vec4 out_FragColor2;
layout (location=3) out vec4 out_FragColor3;
layout (location=4) out vec4 out_FragColor4;
layout (location=5) out vec4 out_FragColor5;

void main() {
  out_FragColor0 = vec4(1.0, 0.0, 0.0, 1.0);
  out_FragColor1 = vec4(0.0, 1.0, 0.0, 1.0);
  out_FragColor2 = vec4(0.0, 0.0, 1.0, 1.0);
  out_FragColor3 = vec4(1.0, 0.0, 1.0, 1.0);
  out_FragColor4 = vec4(1.0, 1.0, 0.0, 1.0);
  out_FragColor5 = vec4(0.0, 1.0, 1.0, 1.0);
};
)";

const char* codeSlang = R"(
struct VertexStageOutput {
  float4 sv_Position             : SV_Position;
  float3 dir                     : COLOR0;
  nointerpolation uint textureId : TEXTUREID;
};

struct PushConstants {
  float4x4 mvp;
  uint texture0;
};

[[vk::push_constant]] PushConstants pc;

static const float3 vertices[8] = {
  float3(-1.0, -1.0,  1.0), float3( 1.0, -1.0,  1.0),
  float3( 1.0,  1.0,  1.0), float3(-1.0,  1.0,  1.0),
  float3(-1.0, -1.0, -1.0), float3( 1.0, -1.0, -1.0),
  float3( 1.0,  1.0, -1.0), float3(-1.0,  1.0, -1.0)
};

[shader("vertex")]
VertexStageOutput vertexMain(uint vertexID : SV_VertexID) {
  VertexStageOutput out;
  float3 v = vertices[vertexID];
  out.sv_Position = pc.mvp * float4(v, 1.0);
  out.dir = v;
  out.textureId = pc.texture0;
  return out;
}

struct PSInput {
  float3 dir                     : COLOR0;
  nointerpolation uint textureId : TEXTUREID;
};

[shader("fragment")]
float4 fragmentMain(PSInput input) {
  return textureBindlessCube(input.textureId, 0, normalize(input.dir));
}
)";

const char* codeVS = R"(
layout (location=0) out vec3 dir;
layout (location=1) out flat uint textureId;

const vec3 vertices[8] = vec3[8](
  vec3(-1.0,-1.0, 1.0), vec3( 1.0,-1.0, 1.0), vec3( 1.0, 1.0, 1.0), vec3(-1.0, 1.0, 1.0),
  vec3(-1.0,-1.0,-1.0), vec3( 1.0,-1.0,-1.0), vec3( 1.0, 1.0,-1.0), vec3(-1.0, 1.0,-1.0)
);

layout(push_constant) uniform constants {
  mat4 mvp;
  uint texture0;
} pc;

void main() {
  vec3 v = vertices[gl_VertexIndex];
  gl_Position = pc.mvp * vec4(v, 1.0);
  dir = v;
  textureId = pc.texture0;
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 dir;
layout (location=1) in flat uint textureId;
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = textureBindlessCube(textureId, 0, normalize(dir));
};
)";

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 0,
      .height = 0,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  {
    const VkPhysicalDeviceProperties& props = static_cast<lvk::VulkanContext*>(ctx)->getVkPhysicalDeviceProperties();

    if (props.limits.maxColorAttachments < 6) {
      LVK_ASSERT_MSG(false, "This demo needs at least 6 color attachments to be supported");
      std::terminate();
    }

    const uint16_t indexData[36] = {0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
                                    4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3};

    lvk::Holder<lvk::BufferHandle> ib0_ = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Index,
        .storage = lvk::StorageType_Device,
        .size = sizeof(indexData),
        .data = indexData,
        .debugName = "Buffer: index",
    });

    lvk::Holder<lvk::TextureHandle> texture_ = ctx->createTexture({
        .type = lvk::TextureType_Cube,
        .format = lvk::Format_BGRA_UN8,
        .dimensions = {512, 512},
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment,
        .debugName = "CubeMap",
    });

#if defined(LVK_DEMO_WITH_SLANG)
    lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
    lvk::Holder<lvk::ShaderModuleHandle> vertTriangle_ =
        ctx->createShaderModule({codeTriangleSlang, lvk::Stage_Vert, "Shader Module: triangle (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragTriangle_ =
        ctx->createShaderModule({codeTriangleSlang, lvk::Stage_Frag, "Shader Module: triangle (frag)"});
#else
    lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
    lvk::Holder<lvk::ShaderModuleHandle> vertTriangle_ =
        ctx->createShaderModule({codeTriangleVS, lvk::Stage_Vert, "Shader Module: triangle (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragTriangle_ =
        ctx->createShaderModule({codeTriangleFS, lvk::Stage_Frag, "Shader Module: triangle (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

    lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_ = ctx->createRenderPipeline({
        .smVert = vert_,
        .smFrag = frag_,
        .color = {{.format = ctx->getSwapchainFormat()}},
        .cullMode = lvk::CullMode_Back,
        .frontFace = lvk::WindingMode_CW,
        .debugName = "Pipeline: mesh",
    });
    lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_ = ctx->createRenderPipeline({
        .smVert = vertTriangle_,
        .smFrag = fragTriangle_,
        .color = {{.format = ctx->getFormat(texture_)},
                  {.format = ctx->getFormat(texture_)},
                  {.format = ctx->getFormat(texture_)},
                  {.format = ctx->getFormat(texture_)},
                  {.format = ctx->getFormat(texture_)},
                  {.format = ctx->getFormat(texture_)}},
        .debugName = "Pipeline: triangle",
    });

    // Main loop
    app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
      LVK_PROFILER_FUNCTION();

      const float fov = float(45.0f * (M_PI / 180.0f));
      const mat4 proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 500.0f);
      const mat4 view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 5.0f));
      const mat4 model = glm::rotate(mat4(1.0f), (float)glfwGetTime(), glm::normalize(vec3(1.0f, 1.0f, 1.0f)));

      // Command buffers (1-N per thread): create, submit and forget
      lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

      buffer.cmdPushDebugGroupLabel("Render to Cube Map", 0xff0000ff);
      buffer.cmdBeginRendering(
          {.color =
               {
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 0, .clearColor = {0.3f, 0.1f, 0.1f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 1, .clearColor = {0.1f, 0.3f, 0.1f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 2, .clearColor = {0.1f, 0.1f, 0.3f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 3, .clearColor = {0.3f, 0.1f, 0.3f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 4, .clearColor = {0.3f, 0.3f, 0.1f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .layer = 5, .clearColor = {0.1f, 0.3f, 0.3f, 1.0f}},
               }},
          {.color = {
               {.texture = texture_},
               {.texture = texture_},
               {.texture = texture_},
               {.texture = texture_},
               {.texture = texture_},
               {.texture = texture_},
           }});
      buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
      buffer.cmdPushConstants(float(10.0 * glfwGetTime()));
      buffer.cmdDraw(3);
      buffer.cmdEndRendering();
      buffer.cmdPopDebugGroupLabel();

      buffer.cmdBeginRendering({.color = {{
                                    .loadOp = lvk::LoadOp_Clear,
                                    .storeOp = lvk::StoreOp_Store,
                                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f},
                                }}},
                               {.color = {{.texture = ctx->getCurrentSwapchainTexture()}}},
                               {.textures = {lvk::TextureHandle(texture_)}});
      {
        buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
        buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
        buffer.cmdBindScissorRect({0, 0, (uint32_t)width, (uint32_t)height});
        buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
        buffer.cmdBindDepthState({});
        buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
        struct {
          mat4 mvp;
          uint32_t texture;
        } bindings = {
            .mvp = proj * view * model,
            .texture = texture_.index(),
        };
        buffer.cmdPushConstants(bindings);
        buffer.cmdDrawIndexed(3 * 6 * 2);
        buffer.cmdPopDebugGroupLabel();
      }
      buffer.cmdEndRendering();

      ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
    });
  }

  VULKAN_APP_EXIT();
}
