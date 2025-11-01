/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

#include <filesystem>

#include <ldrutils/lutils/ScopeExit.h>
#include <stb/stb_image.h>

constexpr uint32_t kNumObjects = 16;

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

const char* codeSlang = R"(
struct Vertex {
  float x, y, z;
  float r, g, b;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
  uint texture0;
  uint texture1;
  uint sampler0;
};

struct PerObject {
  float4x4 model[];
};

struct VertexBuffer {
  Vertex vertices[];
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  VertexBuffer* vb;
};

[[vk::push_constant]] PushConstants pc;

struct VertexStageOutput {
  float4 position : SV_Position;
  float3 color;
  float3 normal;
};

[shader("vertex")]
void vertexMain(uint vertexID   : SV_VertexID,
                uint instanceID : SV_InstanceID,
                out VertexStageOutput output)
{
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model[instanceID];

  Vertex v = pc.vb->vertices[vertexID];

  output.position = mul(float4(v.x, v.y, v.z, 1.0), mul(model, mul(view, proj)));
  output.color = float3(v.r, v.g, v.b);
  output.normal = normalize(float3(v.x, v.y, v.z)); // object space normal
}

// bindless texture and sampler arrays
[[vk::binding(0, 0)]] Texture2D    kTextures2D[];
[[vk::binding(1, 0)]] SamplerState kSamplers[];

float4 textureBindless2D(uint textureid, uint samplerid, float2 uv) {
  return kTextures2D[NonUniformResourceIndex(textureid)].Sample(
    kSamplers[NonUniformResourceIndex(samplerid)], uv);
}

float4 triplanar(uint tex, float3 worldPos, float3 normal) {
  // generate weights, show texture on both sides of the object (positive and negative)
  float3 weights = abs(normal);
  // make the transition sharper
  weights = pow(weights, float3(8.0, 8.0, 8.0));
  // make sure the sum of all components is 1
  weights = weights / (weights.x + weights.y + weights.z);

  // sample the texture for 3 different projections
  float4 cXY = textureBindless2D(tex, pc.perFrame->sampler0, worldPos.xy);
  float4 cZY = textureBindless2D(tex, pc.perFrame->sampler0, worldPos.zy);
  float4 cXZ = textureBindless2D(tex, pc.perFrame->sampler0, worldPos.xz);

  // combine the projected colors
  return cXY * weights.z + cZY * weights.x + cXZ * weights.y;
}

[shader("fragment")]
float4 fragmentMain(
  VertexStageOutput input,
  float4 position : SV_Position) : SV_Target
{
  // triplanar mapping in object-space; for our icosahedron, object-space position and normal vectors are the same
  float4 t0 = triplanar(pc.perFrame->texture0, input.normal, input.normal);
  float4 t1 = triplanar(pc.perFrame->texture1, input.normal, input.normal);

  return float4(input.color * (t0.rgb + t1.rgb), 1.0);
}
)";

const char* codeVS = R"(
layout (location=0) out vec3 color;
layout (location=1) out vec3 normal;

struct Vertex {
  float x, y, z;
  float r, g, b;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture0;
  uint texture1;
  uint sampler0;
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
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model[gl_InstanceIndex];
  Vertex v = pc.vb.vertices[gl_VertexIndex];
  gl_Position = proj * view * model * vec4(v.x, v.y, v.z, 1.0);
  color = vec3(v.r, v.g, v.b);
  normal = normalize(vec3(v.x, v.y, v.z)); // object space normal
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 color;
layout (location=1) in vec3 normal;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture0;
  uint texture1;
  uint sampler0;
};

layout(push_constant) uniform constants {
	PerFrame perFrame;
} pc;

vec4 triplanar(uint tex, vec3 worldPos, vec3 normal) {
  // generate weights, show texture on both sides of the object (positive and negative)
  vec3 weights = abs(normal);
  // make the transition sharper
  weights = pow(weights, vec3(8.0));
  // make sure the sum of all components is 1
  weights = weights / (weights.x + weights.y + weights.z);

  // sample the texture for 3 different projections
  vec4 cXY = textureBindless2D(tex, pc.perFrame.sampler0, worldPos.xy);
  vec4 cZY = textureBindless2D(tex, pc.perFrame.sampler0, worldPos.zy);
  vec4 cXZ = textureBindless2D(tex, pc.perFrame.sampler0, worldPos.xz);

  // combine the projected colors
  return cXY * weights.z + cZY * weights.x + cXZ * weights.y;
}

void main() {
  // triplanar mapping in object-space; for our icosahedron, object-space position and normal vectors are the same
  vec4 t0 = triplanar(pc.perFrame.texture0, normal, normal);
  vec4 t1 = triplanar(pc.perFrame.texture1, normal, normal);
  out_FragColor = vec4(color * (t0.rgb + t1.rgb), 1.0);
};
)";

struct VertexPosUvw {
  vec3 pos;
  vec3 color;
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  uint32_t texture0;
  uint32_t texture1;
  uint32_t sampler;
};

lvk::Holder<lvk::TextureHandle> texture0_;
lvk::Holder<lvk::TextureHandle> texture1_;

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 1280,
      .height = 1024,
      .resizable = true,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  // clang-format off
  // icosahedron
  const float t = (1.0f + sqrtf(5.0f)) / 2.0f;
  const VertexPosUvw vertexData[] = {
    {{-1,  t, 0}, {0, 1, 0}},
    {{ 1,  t, 0}, {1, 1, 0}},
    {{-1, -t, 0}, {0, 1, 0}},
    {{ 1, -t, 0}, {1, 1, 0}},

    {{0, -1,  t}, {0, 0, 1}},
    {{0,  1,  t}, {0, 1, 1}},
    {{0, -1, -t}, {0, 0, 1}},
    {{0,  1, -t}, {0, 1, 1}},

    {{ t, 0, -1}, {1, 0, 0}},
    {{ t, 0,  1}, {1, 0, 1}},
    {{-t, 0, -1}, {1, 0, 0}},
    {{-t, 0,  1}, {1, 0, 1}},
  };
  // clang-format on

  lvk::Holder<lvk::BufferHandle> vb0_ = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = sizeof(vertexData),
      .data = vertexData,
      .debugName = "Buffer: vertices",
  });

  const uint16_t indexData[] = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4,  11, 10, 2,  10, 7, 6, 7, 1, 8,
                                3, 9,  4, 3, 4, 2, 3, 2, 6, 3, 6, 8,  3, 8,  9,  4, 9, 5, 2, 4,  11, 6,  2,  10, 8,  6, 7, 9, 8, 1};

  lvk::Holder<lvk::BufferHandle> ib0_ = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Index,
      .storage = lvk::StorageType_Device,
      .size = sizeof(indexData),
      .data = indexData,
      .debugName = "Buffer: indices",
  });

  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });
  lvk::Holder<lvk::BufferHandle> bufModelMatrices = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = kNumObjects * sizeof(mat4),
      .debugName = "Buffer: model matrices",
  });

  lvk::Holder<lvk::SamplerHandle> sampler_ = ctx->createSampler({.debugName = "Sampler: linear"}, nullptr);

  // texture 0
  {
    const uint32_t texWidth = 256;
    const uint32_t texHeight = 256;
    std::vector<uint32_t> pixels(texWidth * texHeight);
    for (uint32_t y = 0; y != texHeight; y++) {
      for (uint32_t x = 0; x != texWidth; x++) {
        // create a XOR pattern
        pixels[y * texWidth + x] = 0xFF000000 + ((x ^ y) << 16) + ((x ^ y) << 8) + (x ^ y);
      }
    }
    texture0_ = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_BGRA_UN8,
        .dimensions = {texWidth, texHeight},
        .usage = lvk::TextureUsageBits_Sampled,
        .data = pixels.data(),
        .debugName = "XOR pattern",
    });
  }

  // texture 1
  {
    using namespace std::filesystem;
    path dir = app.folderContentRoot_;
    int32_t texWidth = 0;
    int32_t texHeight = 0;
    int32_t channels = 0;
    uint8_t* pixels = stbi_load(
        (dir / path("src/bistro/BuildingTextures/wood_polished_01_diff.png")).string().c_str(), &texWidth, &texHeight, &channels, 4);
    SCOPE_EXIT {
      stbi_image_free(pixels);
    };    
    if (!pixels) {
      LVK_ASSERT_MSG(false, "Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
      LLOGW("Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
      std::terminate();
    }
    texture1_ = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_RGBA_UN8,
        .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
        .usage = lvk::TextureUsageBits_Sampled,
        .data = pixels,
        .debugName = "wood_polished_01_diff.png",
    });
  }

  vec3 axis_[kNumObjects]; // uninitialized

  // generate random rotation axes
  for (vec3& v : axis_) {
    v = glm::sphericalRand(1.0f);
  }

#if defined(LVK_DEMO_WITH_SLANG)
  lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
#else
  lvk::Holder<lvk::ShaderModuleHandle> vert_ = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_ = ctx->createRenderPipeline({
      .smVert = vert_,
      .smFrag = frag_,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_Back,
      .frontFaceWinding = lvk::WindingMode_CW,
      .debugName = "Pipeline: mesh",
  });

#if !defined(ANDROID)
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_T && action == GLFW_PRESS) {
      texture1_.reset();
    }
  });
#endif // !ANDROID

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const float fov = float(45.0f * (M_PI / 180.0f));
    const PerFrame perFrame = {
        .proj = glm::perspectiveLH(fov, aspectRatio, 0.1f, 100.0f),
        // place the "camera" behind the objects, the distance depends on the total number of objects
        .view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, sqrtf(kNumObjects / 16) * 14.0f * t)),
        .texture0 = texture0_.index(),
        .texture1 = texture1_.index(),
        .sampler = sampler_.index(),
    };

    mat4 modelMatrices[kNumObjects]; // uninitialized

    // rotate objects around random axes
    for (uint32_t i = 0; i != kNumObjects; i++) {
      const float direction = powf(-1, (float)(i + 1));
      const uint32_t cubesInLine = (uint32_t)sqrt(kNumObjects);
      const vec3 offset =
          vec3(-1.5f * sqrt(kNumObjects) + 4.0f * (i % cubesInLine), -1.5f * sqrt(kNumObjects) + 4.0f * (i / cubesInLine), 0);
      modelMatrices[i] = glm::rotate(glm::translate(mat4(1.0f), offset), float(direction * glfwGetTime()), axis_[i]);
    }

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    buffer.cmdUpdateBuffer(bufPerFrame, perFrame);
    buffer.cmdUpdateBuffer(bufModelMatrices, modelMatrices);

    lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
        .depthStencil = {app.getDepthTexture()},
    };
    buffer.cmdBeginRendering(
        lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}},
                        .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0}},
        framebuffer);
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
      buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, (uint32_t)width, (uint32_t)height});
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
      const struct {
        uint64_t perFrame;
        uint64_t perObject;
        uint64_t vb;
      } bindings = {
          .perFrame = ctx->gpuAddress(bufPerFrame),
          .perObject = ctx->gpuAddress(bufModelMatrices),
          .vb = ctx->gpuAddress(vb0_),
      };
      buffer.cmdPushConstants(bindings);
      buffer.cmdDrawIndexed(LVK_ARRAY_NUM_ELEMENTS(indexData), kNumObjects);
      buffer.cmdPopDebugGroupLabel();
    }
    app.imgui_->beginFrame(framebuffer);
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_Once);
    ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Image(texture1_.index(), ImVec2(512, 512));
    if (texture1_.valid())
      ImGui::Text("Press T to unload texture");
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buffer);
    buffer.cmdEndRendering();

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  // destroy all the Vulkan stuff before closing the window
  texture0_.reset();
  texture1_.reset();

  VULKAN_APP_EXIT();
}
