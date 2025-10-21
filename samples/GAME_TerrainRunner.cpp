/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"

// Simple procedural terrain runner: a deformed grid animated over time.
// Camera is controlled by VulkanApp's first-person controller (WASD + mouse).

// Vertex shader: reads xz grid positions from a buffer, computes height in VS.
static const char* kVS = R"(
layout (location=0) in vec2 inPosXZ;
layout (location=0) out vec3 vColor;
layout (location=1) out vec3 vWorldPos;
layout (location=2) out vec3 vNormal;

layout(push_constant) uniform PC {
  mat4 mvp;
  vec4 params;     // (time, amplitude, frequency, speed)
  vec4 sunWater;   // (sun.x, sun.y, sun.z, waterLevel)
  vec4 biome;      // (beachH, grassH, rockH, snowH)
  vec4 camPos;     // (cam.x, cam.y, cam.z, fogDist)
} pc;

float heightFunc(vec2 p, float t, float amp, float freq) {
  // Lightweight FBM-like pattern using sines; cheap and portable.
  float h = 0.0;
  float a = 1.0;
  float f = freq;
  for (int i = 0; i < 4; ++i) {
    h += a * (sin(f*(p.x + 0.31*t)) * 0.5 + cos(f*(p.y - 0.37*t)) * 0.5);
    a *= 0.5; f *= 2.0;
  }
  return amp * h;
}

vec3 computeNormal(vec2 p, float t, float amp, float freq) {
  // Finite differences for normal; small epsilon in x,z.
  const float e = 0.05;
  float h  = heightFunc(p, t, amp, freq);
  float hx = heightFunc(p + vec2(e, 0.0), t, amp, freq);
  float hz = heightFunc(p + vec2(0.0, e), t, amp, freq);
  vec3 dx = vec3(e, hx - h, 0.0);
  vec3 dz = vec3(0.0, hz - h, e);
  return normalize(cross(dz, dx));
}

void main() {
  const float time = pc.params.x;
  const float amp  = pc.params.y;
  const float freq = pc.params.z;
  const float speed= pc.params.w;

  vec2 p = inPosXZ;
  float y = heightFunc(p, time * speed, amp, freq);
  vec3 n = computeNormal(p, time * speed, amp, freq);

  vec4 wp = vec4(p.x, y, p.y, 1.0);
  gl_Position = pc.mvp * wp;
  // simple coloring by normal.y and height
  vColor = mix(vec3(0.05, 0.25, 0.05), vec3(0.6, 0.55, 0.4), clamp(0.5 + 0.5*n.y + 0.2*y, 0.0, 1.0));
  vWorldPos = wp.xyz;
  vNormal = n;
}
)";

// Fragment shader: output color
static const char* kFS = R"(
layout (location=0) in vec3 vColor;
layout (location=1) in vec3 vWorldPos;
layout (location=2) in vec3 vNormal;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform PC {
  mat4 mvp;
  vec4 params;     // (time, amplitude, frequency, speed)
  vec4 sunWater;   // (sun.x, sun.y, sun.z, waterLevel)
  vec4 biome;      // (beachH, grassH, rockH, snowH)
  vec4 camPos;     // (cam.x, cam.y, cam.z, fogDist)
} pc;

vec3 shadeEarthlike(vec3 wp, vec3 n) {
  float h = wp.y; // height above water plane
  float water = pc.sunWater.w;
  float beachH = pc.biome.x;
  float grassH = pc.biome.y;
  float rockH  = pc.biome.z;
  float snowH  = pc.biome.w;

  // Base biome colors
  vec3 colWaterDeep = vec3(0.02, 0.10, 0.30);
  vec3 colWaterShal = vec3(0.05, 0.25, 0.50);
  vec3 colBeach     = vec3(0.76, 0.70, 0.50);
  vec3 colGrass     = vec3(0.20, 0.45, 0.18);
  vec3 colRock      = vec3(0.35, 0.33, 0.32);
  vec3 colSnow      = vec3(0.92, 0.94, 0.96);

  // Water or land
  if (h < water) {
    float t = clamp((water - h) * 0.15, 0.0, 1.0);
    return mix(colWaterShal, colWaterDeep, t);
  }

  // Heights relative to water
  float rel = h - water;
  // Slope factor biases towards rock on steep slopes
  float slope = 1.0 - clamp(n.y, 0.0, 1.0);

  vec3 col = colGrass;
  if (rel < beachH) {
    float t = clamp(rel / max(beachH, 1e-3), 0.0, 1.0);
    col = mix(colBeach, colGrass, t);
  } else if (rel < grassH) {
    col = colGrass;
  } else if (rel < rockH) {
    float t = (rel - grassH) / max(rockH - grassH, 1e-3);
    col = mix(colGrass, colRock, clamp(t + 0.5 * slope, 0.0, 1.0));
  } else if (rel < snowH) {
    float t = (rel - rockH) / max(snowH - rockH, 1e-3);
    col = mix(colRock, colSnow, clamp(t + 0.25 * slope, 0.0, 1.0));
  } else {
    col = colSnow;
  }

  return col;
}

void main(){
  vec3 N = normalize(vNormal);
  vec3 L = normalize(pc.sunWater.xyz);
  vec3 V = normalize(pc.camPos.xyz - vWorldPos);
  vec3 H = normalize(L + V);

  vec3 albedo = shadeEarthlike(vWorldPos, N);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), 32.0);
  float ambient = 0.25;

  vec3 lit = albedo * (ambient + diff) + 0.08 * spec;

  // Simple distance fog towards horizon
  float dist = length(pc.camPos.xyz - vWorldPos);
  float fogT = clamp(dist / max(pc.camPos.w, 1e-3), 0.0, 1.0);
  vec3 sky = vec3(0.62, 0.82, 1.0);
  vec3 color = mix(lit, sky, fogT * 0.6);
  out_FragColor = vec4(color, 1.0);
}
)";

struct VertexXZ { float x, z; };

struct PushConstants {
  mat4 mvp;
  vec4 params;   // (time, amp, freq, speed)
  vec4 sunWater; // (sun.x, sun.y, sun.z, waterLevel)
  vec4 biome;    // (beachH, grassH, rockH, snowH)
  vec4 camPos;   // (cam.x, cam.y, cam.z, fogDist)
};

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 1280,
      .height = 720,
      .resizable = true,
      .initialCameraPos = vec3(0.0f, 6.0f, -12.0f),
      .initialCameraTarget = vec3(0.0f, 0.0f, 0.0f),
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  // Terrain grid parameters
  const uint32_t N = 128;          // grid resolution (N x N vertices)
  const float    extent = 50.0f;   // world half-size in XZ
  const float    step = (2.0f * extent) / float(N - 1);

  // Generate grid vertices (XZ only; Y computed in shader)
  std::vector<VertexXZ> vertices;
  vertices.reserve(N * N);
  for (uint32_t j = 0; j < N; ++j) {
    for (uint32_t i = 0; i < N; ++i) {
      const float x = -extent + float(i) * step;
      const float z = -extent + float(j) * step;
      vertices.push_back({x, z});
    }
  }

  // Generate indices (two triangles per quad)
  std::vector<uint32_t> indices;
  indices.reserve((N - 1) * (N - 1) * 6);
  for (uint32_t j = 0; j < N - 1; ++j) {
    for (uint32_t i = 0; i < N - 1; ++i) {
      uint32_t v0 = j * N + i;
      uint32_t v1 = v0 + 1;
      uint32_t v2 = v0 + N;
      uint32_t v3 = v2 + 1;
      // CCW triangles
      indices.push_back(v0); indices.push_back(v2); indices.push_back(v1);
      indices.push_back(v1); indices.push_back(v2); indices.push_back(v3);
    }
  }

  // Buffers
  lvk::Holder<lvk::BufferHandle> vb = ctx->createBuffer({
      .usage = (lvk::BufferUsageBits)(lvk::BufferUsageBits_Vertex),
      .storage = lvk::StorageType_Device,
      .size = (uint32_t)(vertices.size() * sizeof(VertexXZ)),
      .data = vertices.data(),
      .debugName = "VB: Terrain XZ",
  });

  lvk::Holder<lvk::BufferHandle> ib = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Index,
      .storage = lvk::StorageType_Device,
      .size = (uint32_t)(indices.size() * sizeof(uint32_t)),
      .data = indices.data(),
      .debugName = "IB: Terrain",
  });

  // Pipeline
  lvk::Holder<lvk::ShaderModuleHandle> smVS = ctx->createShaderModule({kVS, lvk::Stage_Vert, "VS: Terrain"});
  lvk::Holder<lvk::ShaderModuleHandle> smFS = ctx->createShaderModule({kFS, lvk::Stage_Frag, "FS: Terrain"});
  lvk::RenderPipelineDesc rpd = {
      .topology = lvk::Topology_Triangle,
      .smVert = smVS,
      .smFrag = smFS,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .depthFormat = app.getDepthFormat(),
      .cullMode = lvk::CullMode_Back,
      .frontFaceWinding = lvk::WindingMode_CW,
      .debugName = "Pipeline: Terrain"
  };
  // Vertex input: binding 0, location 0, Float2 (x,z)
  rpd.vertexInput.attributes[0] = {.location = 0, .binding = 0, .format = lvk::VertexFormat::Float2, .offset = 0};
  rpd.vertexInput.inputBindings[0] = {.stride = (uint32_t)sizeof(VertexXZ)};
  lvk::Holder<lvk::RenderPipelineHandle> pipe = ctx->createRenderPipeline(rpd);

  // UI parameters
  float amplitude = 0.8f;
  float frequency = 0.10f;
  float speed = 1.0f;
  // Earth-like params
  float waterLevel = 0.0f;
  float beachH = 0.6f;
  float grassH = 2.0f;
  float rockH  = 4.0f;
  float snowH  = 6.0f;
  float fogDist = 120.0f;
  vec3 sunDir = normalize(vec3(0.6f, 0.8f, 0.2f));

  app.run([&](uint32_t width, uint32_t height, float aspect, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    const float fov = float(50.0f * (M_PI / 180.0f));
    const mat4 mvp = glm::perspectiveLH_ZO(fov, aspect, 0.1f, 500.0f) * app.camera_.getViewMatrix();
    PushConstants pc{
        .mvp = mvp,
        .params = vec4((float)glfwGetTime(), amplitude, frequency, speed),
        .sunWater = vec4(sunDir, waterLevel),
        .biome = vec4(beachH, grassH, rockH, snowH),
        .camPos = vec4(app.positioner_.getPosition(), fogDist),
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    lvk::Framebuffer fb = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
        .depthStencil = {app.getDepthTexture()},
    };
    buffer.cmdBeginRendering(
        lvk::RenderPass{.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.62f, 0.82f, 1.0f, 1.0f}}},
                        .depth = {.loadOp = lvk::LoadOp_Clear, .clearDepth = 1.0f}},
        fb);
    {
      buffer.cmdBindRenderPipeline(pipe);
      buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
      buffer.cmdBindScissorRect({0, 0, (uint32_t)width, (uint32_t)height});
      buffer.cmdBindDepthState({.compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true});
      buffer.cmdBindVertexBuffer(0, vb, 0);
      buffer.cmdBindIndexBuffer(ib, lvk::IndexFormat_UI32);
      buffer.cmdPushConstants(pc);
      buffer.cmdPushDebugGroupLabel("Draw Terrain", 0xff00ff00);
      buffer.cmdDrawIndexed((uint32_t)indices.size());
      buffer.cmdPopDebugGroupLabel();
    }

    // UI
    app.imgui_->beginFrame(fb);
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::Begin("Terrain Runner", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::SliderFloat("Amplitude", &amplitude, 0.0f, 3.0f);
    ImGui::SliderFloat("Frequency", &frequency, 0.02f, 0.50f);
    ImGui::SliderFloat("Speed", &speed, 0.0f, 3.0f);
    ImGui::Separator();
    ImGui::SliderFloat("Water Level", &waterLevel, -2.0f, 2.0f);
    ImGui::SliderFloat("Beach H", &beachH, 0.0f, 2.0f);
    ImGui::SliderFloat("Grass H", &grassH, 0.5f, 5.0f);
    ImGui::SliderFloat("Rock H", &rockH, 1.0f, 8.0f);
    ImGui::SliderFloat("Snow H", &snowH, 2.0f, 12.0f);
    ImGui::SliderFloat("Fog Dist", &fogDist, 20.0f, 400.0f);
    ImGui::Text("Sun Dir");
    ImGui::SliderFloat("sun.x", &sunDir.x, -1.0f, 1.0f);
    ImGui::SliderFloat("sun.y", &sunDir.y, -1.0f, 1.0f);
    ImGui::SliderFloat("sun.z", &sunDir.z, -1.0f, 1.0f);
    sunDir = glm::normalize(sunDir);
    ImGui::Text("Grid: %ux%u, extent=%.1f", N, N, extent);
    ImGui::Text("Controls: WASD + Mouse, Space reset");
    app.drawFPS();
    ImGui::End();
    app.imgui_->endFrame(buffer);

    buffer.cmdEndRendering();
    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
