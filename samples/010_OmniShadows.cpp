/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include "lmath/GeometryShapes.h"

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

const char* codeShadowSlang = R"(
struct Vertex {
  float x, y, z;
  float u, v;
  float r, g, b;
};

struct PerFrame {
  float4x4 proj[6];
  float4x4 view[6];
};

struct PerObject {
  float4x4 model[];
};

struct PerLight {
  float4 lightPos;
  float shadowNear;
  float shadowFar;
  uint shadowMap;
};

struct VertexBuffer {
  Vertex vertices[];
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  VertexBuffer* vb;
  PerLight* perLight;
};

[[vk::push_constant]] PushConstants pc;

struct VertexStageOutput {
  float4 sv_Position : SV_Position;
  float4 worldPos;
};

[shader("vertex")]
VertexStageOutput vertexMain(uint vertexID   : SV_VertexID,
                             uint instanceID : SV_InstanceID,
                             uint viewIndex  : SV_ViewID)
{
  float4x4 proj = pc.perFrame->proj[viewIndex];
  float4x4 view = pc.perFrame->view[viewIndex];
  float4x4 model = pc.perObject->model[instanceID];

  Vertex v = pc.vb->vertices[vertexID];

  VertexStageOutput out;

  out.worldPos = model * float4(v.x, v.y, v.z, 1.0);
  out.sv_Position = proj * view * out.worldPos;

  return out;
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput input) : SV_Target
{
  // get distance between fragment and light source
  float lightDistance = length(input.worldPos.xyz - pc.perLight.lightPos.xyz);
    
  // remap to [0...1]
  lightDistance = lightDistance / pc.perLight->shadowFar;

  return float4(lightDistance);
}
)";

const char* codeSlang = R"(
struct Vertex {
  float x, y, z;
  float u, v;
  float nx, ny, nz;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
};

struct PerObject {
  float4x4 model[];
};

struct PerLight {
  float4 lightPos;
  float shadowNear;
  float shadowFar;
  uint shadowMap;
};

struct VertexBuffer {
  Vertex vertices[];
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  VertexBuffer* vb;
  PerLight* perLight;
};

[[vk::push_constant]] PushConstants pc;

struct VertexStageOutput {
  float4 sv_Position : SV_Position;
  float4 worldPos;
  float3 color;
  float3 normal;
};

[shader("vertex")]
VertexStageOutput vertexMain(uint vertexID   : SV_VertexID,
                             uint instanceID : SV_InstanceID)
{
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model[instanceID];

  Vertex v = pc.vb->vertices[vertexID];

  VertexStageOutput out;

  out.worldPos = model * float4(v.x, v.y, v.z, 1.0);
  out.sv_Position = proj * view * out.worldPos;
  out.color = out.worldPos.xyz * 0.03 + float3(0.6);
  out.normal = normalize(float3(v.nx, v.ny, v.nz)); // object space normal as we have an identity model matrix  

  return out;
}

float shadowFactor(float3 fragToLight) {
  // our Y axis is inverted
  fragToLight.y = -fragToLight.y;
  
  // sample from the depth cube map and re-transform back to original value
  float closestDepth = pc.perLight->shadowFar * textureBindlessCube(pc.perLight->shadowMap, 0, fragToLight).r;

  // get current linear depth as the length between the fragment and light position
  float currentDepth = length(fragToLight);

  // now test for shadows
  float bias = 0.1;

  return currentDepth - bias > closestDepth ? 0.0 : 1.0;
}

float shadowFactorPCF3x3x3(float3 fragToLight) {
  float factor = shadowFactor(fragToLight);
  float k = length(fragToLight) * 0.0015;

  for (int x = -1; x != 2; x++)
    for (int y = -1; y != 2; y++)
      for (int z = -1; z != 2; z++)
        factor += shadowFactor(fragToLight + k * float3(x, y, z));

  return factor / 28.0;
}

float attenuation(float distToLight, float radius) {
  float I = distToLight / radius;
  return max(1.0 - I * I, 0.0);
}

[shader("fragment")]
float4 fragmentMain(VertexStageOutput input) : SV_Target {
  float3 fragToLight = input.worldPos.xyz - pc.perLight->lightPos.xyz;
  float NdotL = max(dot(normalize(input.normal), normalize(-fragToLight)), 0.0);
  
  float3 finalColor = input.color * NdotL * shadowFactorPCF3x3x3(fragToLight) * attenuation(length(fragToLight), 50.0);
  
  // add ambient so shadows are not completely black
  return float4(max(finalColor, input.color * 0.3), 1.0);
}
)";

const char* codeShadowVS = R"(
#extension GL_EXT_multiview : enable

layout (location=0) out vec4 v_WorldPos;

struct Vertex {
  float x, y, z;
  float u, v;
  float nx, ny, nz;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj[6];
  mat4 view[6];
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  VertexBuffer vb;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj[gl_ViewIndex];
  mat4 view = pc.perFrame.view[gl_ViewIndex];
  mat4 model = pc.perObject.model[gl_InstanceIndex];
  Vertex v = pc.vb.vertices[gl_VertexIndex];
  v_WorldPos = model * vec4(v.x, v.y, v.z, 1.0);
  gl_Position = proj * view * v_WorldPos;
}
)";

const char* codeShadowFS = R"(
layout (location=0) in vec4 v_WorldPos;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerLight {
  vec4 lightPos;
  float shadowNear;
  float shadowFar;
  uint shadowMap;
};

layout(push_constant) uniform constants {
  vec2 perFrame;
  vec2 perObject;
  vec2 vb;  
  PerLight perLight;
} pc;

void main() {
  // get distance between fragment and light source
  float lightDistance = length(v_WorldPos.xyz - pc.perLight.lightPos.xyz);
    
  // remap to [0...1]
  lightDistance = lightDistance / pc.perLight.shadowFar;
    
  out_FragColor = vec4(lightDistance);
}
)";

const char* codeVS = R"(
layout (location=0) out vec3 v_Color;
layout (location=1) out vec3 v_Normal;
layout (location=2) out vec4 v_WorldPos;

struct Vertex {
  float x, y, z;
  float u, v;
  float nx, ny, nz;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;  
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model[];
};

layout(std430, buffer_reference) readonly buffer PerLight {
  vec4 lightPos;
  float shadowNear;
  float shadowFar;
  uint shadowMap;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  VertexBuffer vb;
  PerLight perLight;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model[gl_InstanceIndex];
  Vertex v = pc.vb.vertices[gl_VertexIndex];
  v_WorldPos = model * vec4(v.x, v.y, v.z, 1.0);
  gl_Position = proj * view * v_WorldPos;
 
  v_Color = v_WorldPos.xyz * 0.03 + vec3(0.6);
  v_Normal = normalize(vec3(v.nx, v.ny, v.nz)); // object space normal as we have an identity model matrix  
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 v_Color;
layout (location=1) in vec3 v_Normal;
layout (location=2) in vec4 v_WorldPos;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerLight {
  vec4 lightPos;
  float shadowNear;
  float shadowFar;
  uint shadowMap;
};

layout(push_constant) uniform constants {
	PerFrame perFrame;
   vec2 perObject;
   vec2 vb;      
   PerLight perLight;
} pc;

float shadowFactor(vec3 fragToLight) {
  // our Y axis is inverted
  fragToLight.y = -fragToLight.y;
  
  // sample from the depth cube map and re-transform back to original value
  float closestDepth = pc.perLight.shadowFar * textureBindlessCube(pc.perLight.shadowMap, 0, fragToLight).r;

  // get current linear depth as the length between the fragment and light position
  float currentDepth = length(fragToLight);

  // now test for shadows
  float bias = 0.1;

  return currentDepth - bias > closestDepth ? 0.0 : 1.0;
}

float shadowFactorPCF3x3x3(vec3 fragToLight) {
  float factor = shadowFactor(fragToLight);
  float k = length(fragToLight) * 0.0015;

  for (int x = -1; x != 2; x++)
    for (int y = -1; y != 2; y++)
      for (int z = -1; z != 2; z++)
        factor += shadowFactor(fragToLight + k * vec3(x, y, z));

  return factor / 28.0;
}

float attenuation(float distToLight, float radius) {
  float I = distToLight / radius;
  return max(1.0 - I * I, 0.0);
}

void main() {
  vec3 fragToLight = v_WorldPos.xyz - pc.perLight.lightPos.xyz;
  float NdotL = max(dot(normalize(v_Normal), normalize(-fragToLight)), 0.0);
  
  vec3 finalColor = v_Color * NdotL * shadowFactorPCF3x3x3(fragToLight) * attenuation(length(fragToLight), 50.0);
  
  // add ambient so shadows are not completely black
  out_FragColor = vec4(max(finalColor, v_Color * 0.3), 1.0);
};
)";

struct PerFrame {
  mat4 proj;
  mat4 view;
};

struct PerFrameShadow {
  mat4 proj[6];
  mat4 view[6];
};

struct PerLight {
  vec4 lightPos;
  float shadowNear;
  float shadowFar;
  uint32_t shadowMap;
};

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = -90,
      .height = -90,
      .resizable = true,
      .initialCameraPos = vec3(-12, 10, 10),
      .initialCameraTarget = vec3(0, 0, 0),
      .initialCameraUpVector = vec3(0, 0, 1),
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  std::vector<GeometryShapes::Vertex> vertexData;

  // pillars
  const int sizeX = 4;
  const int sizeY = 4;
  const float sx = 0.5f;
  const float sy = 0.5f;
  const float h = 3.0f;
  const float spacing = 4.0f;
  for (int x = 0; x != sizeX; x++) {
    for (int y = 0; y != sizeY; y++) {
      GeometryShapes::addAxisAlignedBox(
          vertexData, vec3(spacing * (x - (sizeX - 1) / 2.0f), spacing * (y - (sizeY - 1) / 2.0f), 0.0f), vec3(sx, sy, h));
    }
  }
  // walls
  GeometryShapes::addAxisAlignedBox(vertexData, vec3(-sizeX * spacing, 0, 0), vec3(sx / 2, 2 * sizeY * spacing * sx, 2 * h));
  GeometryShapes::addAxisAlignedBox(vertexData, vec3(+sizeX * spacing, 0, 0), vec3(sx / 2, 2 * sizeY * spacing * sx, 2 * h));
  GeometryShapes::addAxisAlignedBox(vertexData, vec3(0, -sizeY * spacing, 0), vec3(2 * sizeX * spacing * sx, sy / 2, h));
  GeometryShapes::addAxisAlignedBox(vertexData, vec3(0, +sizeY * spacing, 0), vec3(2 * sizeX * spacing * sx, sy / 2, h));
  GeometryShapes::addAxisAlignedBox(vertexData, vec3(0, 0, -h), vec3(2 * sizeX * spacing * sx, 2 * sizeY * spacing * sx, (sx + sy) / 4));

  lvk::Holder<lvk::BufferHandle> vb0_ = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(GeometryShapes::Vertex) * vertexData.size(),
      .data = vertexData.data(),
      .debugName = "Buffer: vertices",
  });

  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });
  lvk::Holder<lvk::BufferHandle> bufPerFrameShadow = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrameShadow),
      .debugName = "Buffer: per frame (shadow)",
  });
  lvk::Holder<lvk::BufferHandle> bufPerLight = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerLight),
      .debugName = "Buffer: per light",
  });

  std::vector<mat4> modelMatrices(1);

  lvk::Holder<lvk::BufferHandle> bufPerObject = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(mat4) * modelMatrices.size(),
      .debugName = "Buffer: model matrices",
  });

#if defined(LVK_DEMO_WITH_SLANG)
  lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> vertShadow_ =
      ctx->createShaderModule({codeShadowSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> fragShadow_ =
      ctx->createShaderModule({codeShadowSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
#else
  lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
  lvk::Holder<lvk::ShaderModuleHandle> vertShadow_ =
      ctx->createShaderModule({codeShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> fragShadow_ =
      ctx->createShaderModule({codeShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_ = ctx->createRenderPipeline({
      .smVert = vert_,
      .smFrag = frag_,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_Back,
      .debugName = "Pipeline: mesh",
  });

  const uint32_t shadowMapSize = 1024;

  lvk::Holder<lvk::TextureHandle> shadowMapColor = ctx->createTexture({
      .type = lvk::TextureType_Cube,
      .format = lvk::Format_R_F16,
      .dimensions = {shadowMapSize, shadowMapSize},
      .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment,
      .debugName = "Texture: shadow map (color)",
  });
  lvk::Holder<lvk::TextureHandle> shadowMap = ctx->createTexture({
      .type = lvk::TextureType_Cube,
      .format = app.getDepthFormat(),
      .dimensions = {shadowMapSize, shadowMapSize},
      .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment,
      .debugName = "Texture: shadow map",
  });

  lvk::Holder<lvk::TextureHandle> layers[6];

  for (uint32_t l = 0; l != LVK_ARRAY_NUM_ELEMENTS(layers); l++) {
    layers[l] = ctx->createTextureView(shadowMap,
                                       {.layer = l,
                                        .swizzle = {
                                            .r = lvk::Swizzle_R,
                                            .g = lvk::Swizzle_R,
                                            .b = lvk::Swizzle_R,
                                            .a = lvk::Swizzle_1,
                                        }});
  }

  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Shadow_ = ctx->createRenderPipeline({
      .smVert = vertShadow_,
      .smFrag = fragShadow_,
      .color = {{ctx->getFormat(shadowMapColor)}},
      .depthFormat = ctx->getFormat(shadowMap),
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: shadow",
  });

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const float fov = glm::radians(45.0f);
    const float fovShadow = glm::radians(90.0f);
    const vec3 lightPos = vec3(4.5f * cosf(glfwGetTime()), 4.5f * sinf(glfwGetTime()), 5.0f);
    const PerLight perLight = {
        .lightPos = vec4(lightPos, 1.0f),
        .shadowNear = 0.1f,
        .shadowFar = 100.0f,
        .shadowMap = shadowMapColor.index(),
    };
    const PerFrame perFrame = {
        .proj = glm::perspective(fov, aspectRatio, 0.1f, 100.0f),
        .view = app.camera_.getViewMatrix(),
    };
    const PerFrameShadow perFrameShadow = {
        .proj = {glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar),
                 glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar),
                 glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar),
                 glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar),
                 glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar),
                 glm::perspective(fovShadow, 1.0f, perLight.shadowNear, perLight.shadowFar)},
        .view = {glm::lookAt(lightPos, lightPos + vec3(+1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0)),
                 glm::lookAt(lightPos, lightPos + vec3(-1.0, 0.0, 0.0), vec3(0.0, -1.0, 0.0)),
                 glm::lookAt(lightPos, lightPos + vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, -1.0)),
                 glm::lookAt(lightPos, lightPos + vec3(0.0, +1.0, 0.0), vec3(0.0, 0.0, +1.0)),
                 glm::lookAt(lightPos, lightPos + vec3(0.0, 0.0, +1.0), vec3(0.0, -1.0, 0.0)),
                 glm::lookAt(lightPos, lightPos + vec3(0.0, 0.0, -1.0), vec3(0.0, -1.0, 0.0))},
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    auto drawMesh = [&](lvk::BufferHandle bufPerFrame) {
      const struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t vb;
        uint64_t perLight;
      } bindings = {
          .perFrame = ctx->gpuAddress(bufPerFrame),
          .perObject = ctx->gpuAddress(bufPerObject),
          .vb = ctx->gpuAddress(vb0_),
          .perLight = ctx->gpuAddress(bufPerLight),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDraw(vertexData.size());
    };

    modelMatrices[0] = mat4(1.0f);

    buffer.cmdUpdateBuffer(bufPerFrame, perFrame);
    buffer.cmdUpdateBuffer(bufPerFrameShadow, perFrameShadow);
    buffer.cmdUpdateBuffer(bufPerLight, perLight);
    buffer.cmdUpdateBuffer(bufPerObject, 0, sizeof(mat4) * modelMatrices.size(), modelMatrices.data());

    // 1. Render shadow map
    buffer.cmdBeginRendering(
        lvk::RenderPass{
            .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {1000.0f, 1000.0f, 1000.0f, 1000.0f}}},
            .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0},
            .layerCount = 6,
            .viewMask = 0b111111,
        },
        {
            .color = {{shadowMapColor}},
            .depthStencil = shadowMap,
        });
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Shadow_);
      buffer.cmdBindViewport({0.0f, 0.0f, (float)shadowMapSize, (float)shadowMapSize, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, shadowMapSize, shadowMapSize});
      buffer.cmdPushDebugGroupLabel("Render Shadow", 0xff0000ff);
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
      drawMesh(bufPerFrameShadow);
      buffer.cmdPopDebugGroupLabel();
    }
    buffer.cmdEndRendering();
    // 2. Render scene
    const lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
        .depthStencil = {app.getDepthTexture()},
    };
    buffer.cmdBeginRendering(
        lvk::RenderPass{
            .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}},
            .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0},
        },
        framebuffer,
        {
            .textures = {lvk::TextureHandle(shadowMapColor)},
        });
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
      buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, (uint32_t)width, (uint32_t)height});
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
      drawMesh(bufPerFrame);
      buffer.cmdPopDebugGroupLabel();
    }
    app.imgui_->beginFrame(framebuffer);
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
    ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    for (uint32_t l = 0; l != LVK_ARRAY_NUM_ELEMENTS(layers); l++) {
      ImGui::Image(layers[l].index(), ImVec2(256, 256));
    }
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buffer);
    buffer.cmdEndRendering();

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
