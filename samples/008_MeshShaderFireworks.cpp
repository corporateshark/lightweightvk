/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

#include "VulkanApp.h"

// we are going to use raw Vulkan here to initialize VK_EXT_mesh_shader
#include <lvk/vulkan/VulkanUtils.h>

const char* codeSlang = R"(
struct Vertex {
  float3 position;
  float3 color;
  float flare;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
  uint texture;
};

struct PushConstants {
  PerFrame* perFrame;
  Vertex* vb;
};

[[vk::push_constant]] PushConstants pc;

struct VertexOutput {
  float3 color : COLOR0;
  float2 uv    : TEXCOORD0;
};

static const float2 offs[4] = {
  float2(-1.0, -1.0),
  float2(+1.0, -1.0),
  float2(-1.0, +1.0),
  float2(+1.0, +1.0)
};

struct MeshOutput {
  float4 position : SV_Position;
  float3 color : COLOR0;
  float2 uv    : TEXCOORD0;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void meshMain(
  uint3 groupID : SV_GroupID,
  out vertices MeshOutput verts[4],
  out indices uint3 triangles[2]
) {
  SetMeshOutputCounts(4, 2);
  
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  Vertex v = pc.vb[groupID.x];
  float4 center = view * float4(v.position, 1.0);
  
  float2 size  = v.flare > 0.5 ? float2(0.08, 0.4) : float2(0.2, 0.2);
  float3 color = v.flare > 0.5 ? 0.5 * v.color : v.color;
  
  for (uint i = 0; i < 4; i++) {
    float4 offset = float4(size * offs[i], 0, 0);
    verts[i].position = proj * (center + offset);
    verts[i].color = color;
    verts[i].uv = (offs[i] + 1.0) * 0.5; // convert from [-1, 1] to [0, 1]
  }
  
  triangles[0] = uint3(0, 1, 2);
  triangles[1] = uint3(2, 1, 3);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input : VertexOutput) : SV_Target
{
  float alpha = textureBindless2D(pc.perFrame->texture, 0, input.uv).r;
  return float4(input.color, alpha);
}
)";

const char* codeMesh = R"(
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = 4, max_primitives = 2) out;

struct Vertex {
  float x, y, z;
  float r, g, b, flare;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  VertexBuffer vb;
} pc;

layout (location=0) out vec3 colors[4];
layout (location=1) out vec2 uvs[4];

const vec2 offs[4] = vec2[4](
  vec2(-1.0, -1.0),
  vec2(+1.0, -1.0),
  vec2(-1.0, +1.0),
  vec2(+1.0, +1.0)
);

void main() {
  SetMeshOutputsEXT(4, 2);

  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  Vertex v = pc.vb.vertices[gl_WorkGroupID.x];
  vec4 center = view * vec4(v.x, v.y, v.z, 1.0);

  vec2 size  = v.flare > 0.5 ? vec2(0.08, 0.4) : vec2(0.2, 0.2);
  vec3 color = v.flare > 0.5 ? 0.5 * vec3(v.r, v.g, v.b) : vec3(v.r, v.g, v.b);

  for (uint i = 0; i != 4; i++) {
    vec4 offset = vec4(size * offs[i], 0, 0);
    gl_MeshVerticesEXT[i].gl_Position = proj * (center + offset);
    colors[i] = color;    
    uvs[i] = (offs[i] + 1.0) * 0.5; // convert from [-1, 1] to [0, 1]
  }

  // two triangles forming a quad
  gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = false;
  gl_MeshPrimitivesEXT[1].gl_CullPrimitiveEXT = false;
  gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
  gl_PrimitiveTriangleIndicesEXT[1] = uvec3(2, 1, 3);
}
)";

const char* codeFS = R"(
layout (location=0) in vec3 color;
layout (location=1) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  uint texture;
};

layout(push_constant) uniform constants {
	PerFrame perFrame;
} pc;

void main() {
  float alpha = textureBindless2D(pc.perFrame.texture, 0, uv).r;
  out_FragColor = vec4(color, alpha);
};
)";

float random(float x) {
  return glm::linearRand(0.0f, x);
}

const int kMaxParticles = 50000;
const int kStackSize = 50000;

vec3 g_Gravity = {0, -0.001, 0};
bool g_Pause = false;

enum ParticleStateMessage {
  PSM_None = 0,
  PSM_Kill = 1,
  PSM_Emission = 2,
};

struct Particle {
  vec3 pos = vec3(0.0f);
  vec3 velocity = vec3(0.0f);
  vec3 baseColor = vec3(0.0f);
  vec3 currentColor = vec3(0.0f);

  // lifetime tracking
  int TTL = 0;
  int initialLT = 1;

  // state flags
  bool alive = false;
  bool flare = false;
  bool spawnExplosion = false;

  // visual properties
  bool fadingOut = false;
  bool emission = false;

  Particle() = default;

  Particle(vec3 pos, vec3 vel, vec3 color, int ttl, bool fadingOut = false)
  : pos(pos)
  , velocity(vel)
  , baseColor(color)
  , currentColor(color)
  , TTL(ttl)
  , initialLT(ttl)
  , alive(true)
  , fadingOut(fadingOut) {}

  ParticleStateMessage update() {
    pos += velocity;
    velocity += g_Gravity;
    TTL--;

    if (fadingOut) {
      const float t = static_cast<float>(TTL) / initialLT;
      currentColor = baseColor * t;
    }

    if (TTL < 0)
      return PSM_Kill;

    return emission ? PSM_Emission : PSM_None;
  }
};

struct ParticleSystem {
  Particle particles[kMaxParticles];
  Particle particlesStack[kStackSize];
  int totalParticles = 0;
  int queuedParticles = 0;

  void nextFrame() {
    int processedParticles = 0;

    for (int i = 0; i < kMaxParticles; i++) {
      if (particles[i].alive) {
        processedParticles++;

        switch (particles[i].update()) {
        case PSM_None:
          break;
        case PSM_Kill:
          if (particles[i].spawnExplosion)
            addExplosion(particles[i].pos);
          particles[i].alive = false;
          totalParticles--;
          break;
        case PSM_Emission:
          addParticle(Particle(particles[i].pos,
                               particles[i].velocity * (random(10) / 10.0f),
                               particles[i].currentColor * 0.9f,
                               particles[i].TTL >> 2,
                               true));
          break;
        }
      } else if (queuedParticles > 0) {
        particles[i] = particlesStack[--queuedParticles];
        totalParticles++;
      } else if (processedParticles >= totalParticles) {
        return;
      }
    }
  }

  void addParticle(const Particle& particle) {
    if (queuedParticles < kStackSize)
      particlesStack[queuedParticles++] = particle;
  }
  void addExplosion(vec3 pos) {
    const vec3 FlarePal[3] = {
        {0.2, 0.30, 0.8},
        {0.7, 0.25, 0.3},
        {0.1, 0.80, 0.2},
    };

    const int paletteIndex = static_cast<int>(random(3));

    for (int i = 0; i < 300; i++) {
      float radius = random(1) / 10;
      float angle = random(M_PI * 2.0f);
      vec3 velocity(radius * cosf(angle), radius * sinf(angle), (random(100) - 50) / 5000.0f);
      vec3 color = FlarePal[paletteIndex] + vec3(random(1) / 5, random(1) / 5, random(1) / 5);

      Particle particle(pos, velocity, color, 90 + random(20), true);
      particle.emission = true;

      addParticle(particle);
    }
  }
};

ParticleSystem g_Points;

struct Vertex {
  vec3 pos;
  vec4 color;
};

struct PerFrame {
  mat4 proj;
  mat4 view;
  uint32_t texture;
};

void generateParticleTexture(uint8_t* image, int size) {
  const float center = 0.5f * (size - 1);
  const float maxDist = center;

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      const float dx = x - center;
      const float dy = y - center;
      const float dist = sqrtf(dx * dx + dy * dy);

      const float normalizedDist = dist < maxDist ? dist / maxDist : 1.0f;

      // steep falloff with a soft center
      const float falloff = 1.0f - normalizedDist;

      // use cubic falloff and scale to match the max brightness 255 at the center
      const float value = falloff * falloff * falloff * 255.0f;

      image[y * size + x] = static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, value)));
    }
  }
}

VULKAN_APP_MAIN {
  VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
      .taskShader = VK_TRUE,
      .meshShader = VK_TRUE,
  };
  const VulkanAppConfig cfg{
      .width = -90,
      .height = -90,
      .resizable = true,
      .contextConfig =
          {
              .extensionsDevice = {"VK_EXT_mesh_shader"},
              .extensionsDeviceFeatures = &meshShaderFeatures,
          },
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  lvk::Holder<lvk::BufferHandle> vb0_[3];
  for (auto& vb : vb0_) {
    vb = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Storage,
        .storage = lvk::StorageType_Device,
        .size = sizeof(Vertex) * kMaxParticles,
        .debugName = "Buffer: vertices",
    });
  }

  lvk::Holder<lvk::BufferHandle> bufPerFrame = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(PerFrame),
      .debugName = "Buffer: per frame",
  });

  lvk::Holder<lvk::SamplerHandle> sampler_ = ctx->createSampler({.debugName = "Sampler: linear"}, nullptr);

  uint8_t particleTextureData[64 * 64];
  generateParticleTexture(particleTextureData, 64);

  lvk::Holder<lvk::TextureHandle> texture_ = ctx->createTexture({
      .type = lvk::TextureType_2D,
      .format = lvk::Format_R_UN8,
      .dimensions = {64, 64},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = particleTextureData,
      .debugName = "Particle",
  });

#if defined(LVK_DEMO_WITH_SLANG)
  lvk::Holder<lvk::ShaderModuleHandle> mesh_ = ctx->createShaderModule({codeSlang, lvk::Stage_Mesh, "Shader Module: main (mesh)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
#else
  lvk::Holder<lvk::ShaderModuleHandle> mesh_ = ctx->createShaderModule({codeMesh, lvk::Stage_Mesh, "Shader Module: main (mesh)"});
  lvk::Holder<lvk::ShaderModuleHandle> frag_ = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_ = ctx->createRenderPipeline({
      .smMesh = mesh_,
      .smFrag = frag_,
      .color = {{
          .format = ctx->getSwapchainFormat(),
          .blendEnabled = true,
          .rgbBlendOp = lvk::BlendOp_Add,
          .alphaBlendOp = lvk::BlendOp_Add,
          .srcRGBBlendFactor = lvk::BlendFactor_SrcAlpha,
          .srcAlphaBlendFactor = lvk::BlendFactor_SrcAlpha,
          .dstRGBBlendFactor = lvk::BlendFactor_One,
          .dstAlphaBlendFactor = lvk::BlendFactor_One,
      }},
      .cullMode = lvk::CullMode_None,
      .debugName = "Pipeline: mesh",
  });

#if !defined(ANDROID)
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
      g_Gravity.x += 0.001f;
    }
    if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
      g_Gravity.x -= 0.001f;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
      g_Pause = !g_Pause;
    }
  });
#endif // !ANDROID

  std::vector<Vertex> vertices;
  vertices.reserve(kMaxParticles);

  const float kTimeQuantum = 0.02f;
  double accTime = 0;
  uint32_t bufferIndex = 0;

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    if (!g_Pause)
      accTime += deltaSeconds;

    if (accTime >= kTimeQuantum) {
      accTime -= kTimeQuantum;
      g_Points.nextFrame();
      if (random(50) <= 1) {
        // shoot a new firework
        const vec3 position((random(100) - 50) / 10, -5, 0);
        const vec3 velocity((random(100) - 50) / 500.0f, 0.25f + (random(200)) / 500.0f, (random(100) - 50) / 500.0f);
        const vec3 color(0.5f, 0.8f, 0.9f);
        Particle flare(position, velocity, color, 20);
        flare.flare = true;
        flare.spawnExplosion = true;
        g_Points.addParticle(flare);
      }
      vertices.clear();
      for (int i = 0; i != kMaxParticles; i++) {
        if (g_Points.particles[i].alive) {
          const Particle& p = g_Points.particles[i];
          vertices.push_back(Vertex{
              .pos = p.pos,
              .color = vec4(p.currentColor, p.flare ? 1.0f : 0.0f),
          });
        }
      }
      if (!vertices.empty()) {
        bufferIndex = (bufferIndex + 1) % LVK_ARRAY_NUM_ELEMENTS(vb0_);
        ctx->upload(vb0_[bufferIndex], vertices.data(), sizeof(Vertex) * vertices.size());
      }
    }

    const PerFrame perFrame = {
        .proj = glm::perspective(glm::radians(90.0f), aspectRatio, 0.1f, 100.0f),
        .view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, -8.0f)),
        .texture = texture_.index(),
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    buffer.cmdUpdateBuffer(bufPerFrame, perFrame);

    const lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
    };
    buffer.cmdBeginRendering(
        lvk::RenderPass{
            .color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        },
        framebuffer);
    {
      buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
      buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, (uint32_t)width, (uint32_t)height});
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_AlwaysPass, .isDepthWriteEnabled = false});
      const struct {
        uint64_t perFrame;
        uint64_t vb;
      } bindings = {
          .perFrame = ctx->gpuAddress(bufPerFrame),
          .vb = ctx->gpuAddress(vb0_[bufferIndex]),
      };
      buffer.cmdPushConstants(bindings);
      if (!vertices.empty()) {
        buffer.cmdDrawMeshTasks({(uint32_t)vertices.size(), 1, 1});
      }
      buffer.cmdPopDebugGroupLabel();
    }
    app.imgui_->beginFrame(framebuffer);
    ImGui::SetNextWindowPos({0, 0});
    ImGui::Begin("Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
    ImGui::Text("Particles: %i", g_Points.totalParticles);
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buffer);
    buffer.cmdEndRendering();

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
