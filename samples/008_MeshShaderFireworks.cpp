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

#include <lmath/Random.h>

const char* codeSlang = R"(
struct Vertex {
  float3 position;
  float3 color;
  float flare;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
};

struct PushConstants {
  PerFrame* perFrame;
  Vertex* vb;
  uint texture;
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
  float alpha = textureBindless2D(pc.texture, 0, input.uv).r;
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
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  VertexBuffer vb;
  uint texture;
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
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  uvec2 vb;
  uint texture;
} pc;

void main() {
  float alpha = textureBindless2D(pc.texture, 0, uv).r;
  out_FragColor = vec4(color, alpha);
};
)";

LRandom g_rng;

float random(float x) {
  return g_rng.randomInRange(0.0f, x);
}

float randomRange(float lo, float hi) {
  return g_rng.randomInRange(lo, hi);
}

const int kMaxParticles = 50000;
const int kStackSize = 50000;

vec3 g_Gravity = {0, -0.001, 0};
float g_AirResistance = 0.98f;
bool g_Pause = false;

// Color palettes for fireworks
struct ColorPalette {
  vec3 primary;
  vec3 secondary;
  vec3 sparkle;
};

const ColorPalette g_Palettes[] = {
    {{1.0f, 0.3f, 0.1f}, {1.0f, 0.6f, 0.2f}, {1.0f, 1.0f, 0.4f}}, // Classic Red-Orange
    {{0.2f, 0.4f, 1.0f}, {0.4f, 0.7f, 1.0f}, {0.4f, 0.9f, 1.0f}}, // Electric Blue
    {{0.1f, 0.9f, 0.3f}, {0.3f, 1.0f, 0.5f}, {0.3f, 1.0f, 0.8f}}, // Emerald Green
    {{0.7f, 0.2f, 0.9f}, {0.9f, 0.4f, 1.0f}, {1.0f, 0.4f, 1.0f}}, // Royal Purple
    {{1.0f, 0.8f, 0.2f}, {1.0f, 0.9f, 0.5f}, {1.0f, 1.0f, 0.5f}}, // Golden
    {{1.0f, 0.2f, 0.6f}, {1.0f, 0.5f, 0.8f}, {1.0f, 0.5f, 1.0f}}, // Magenta-Pink
    {{0.1f, 0.9f, 0.9f}, {0.3f, 1.0f, 1.0f}, {0.4f, 1.0f, 1.0f}}, // Cyan-Teal
    {{0.7f, 0.2f, 0.1f}, {0.9f, 0.5f, 0.1f}, {1.0f, 0.5f, 0.3f}}, // Red
};

enum ExplosionType {
  ExplosionType_Sphere,
  ExplosionType_Ring,
  ExplosionType_DoubleRing,
  ExplosionType_Willow,
  ExplosionType_Chrysanthemum,
  ExplosionType_Crackle,
  ExplosionType_Palm,
  ExplosionType_Crossette,
  NumExplosionTypes,
};

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

  float brightness = 1.0f;
  int explosionType = ExplosionType_Sphere;
  int paletteIndex = 0;
  int secondaryExplosionTimer = -1;

  ParticleStateMessage update() {
    pos += velocity;
    velocity *= g_AirResistance;
    velocity += g_Gravity;
    TTL--;

    if (fadingOut) {
      const float t = static_cast<float>(TTL) / initialLT;
      currentColor = baseColor * t * t * brightness; // quadratic fade
    }

    if (secondaryExplosionTimer > 0) {
      secondaryExplosionTimer--;
      if (secondaryExplosionTimer == 0) {
        spawnExplosion = true;
        return PSM_Kill;
      }
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
  vec3 viewerPos = vec3(0.0f);
  bool useViewerFacingExplosions = false;

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
            addExplosion(particles[i].pos, particles[i].explosionType, particles[i].paletteIndex);
          particles[i].alive = false;
          totalParticles--;
          break;
        case PSM_Emission:
          if (random(1.0f) < 0.7f) {
            const vec3 trailColor = particles[i].currentColor * 0.6f;
            const int trailTTL = particles[i].TTL >> 3;
            addParticle({
                .pos = particles[i].pos,
                .velocity = particles[i].velocity * randomRange(0.02f, 0.1f),
                .baseColor = trailColor,
                .currentColor = trailColor,
                .TTL = trailTTL,
                .initialLT = trailTTL,
                .alive = true,
                .fadingOut = true,
                .brightness = 0.5f,
            });
          }
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

  void addSphereExplosion(vec3 pos, const ColorPalette& pal, int count, float speed, int lifetime) {
    for (int i = 0; i < count; i++) {
      const float theta = random(float(M_PI) * 2.0f);
      const float phi = acosf(randomRange(-1.0f, 1.0f));
      const float r = speed * powf(random(1.0f), 0.33f);

      const vec3 velocity(r * sinf(phi) * cosf(theta), r * sinf(phi) * sinf(theta), r * cosf(phi));

      vec3 color = glm::mix(pal.primary, pal.secondary, random(1.0f));
      color += vec3(randomRange(-0.1f, 0.1f));
      color = glm::clamp(color, vec3(0.0f), vec3(1.0f));

      const int ttl = lifetime + int(random(30));
      const bool core = random(1.0f) < 0.15f;
      const vec3 finalColor = core ? pal.sparkle : color;
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = finalColor,
          .currentColor = finalColor,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .fadingOut = true,
          .emission = random(1.0f) < 0.3f,
          .brightness = core ? 1.5f : 1.0f,
      });
    }
  }

  void addRingExplosion(vec3 pos, const ColorPalette& pal, int count, float speed, int lifetime, float tiltX = 0, float tiltZ = 0) {
    // optionally build an orthonormal basis perpendicular to the viewer direction
    vec3 right(1, 0, 0);
    vec3 up(0, 1, 0);
    vec3 viewDir(0, 0, 1);
    if (useViewerFacingExplosions) {
      const vec3 toViewer = viewerPos - pos;
      const float dist = glm::length(toViewer);
      viewDir = dist > 0.001f ? toViewer / dist : vec3(0, 0, 1);
      const vec3 ref = fabsf(viewDir.y) < 0.99f ? vec3(0, 1, 0) : vec3(1, 0, 0);
      right = glm::normalize(glm::cross(ref, viewDir));
      up = glm::cross(viewDir, right);
    }

    for (int i = 0; i < count; i++) {
      const float angle = (float(i) / count) * float(M_PI) * 2.0f + random(0.2f);
      const float r = speed * randomRange(0.9f, 1.1f);

      vec3 velocity = right * (r * cosf(angle)) + up * (r * sinf(angle)) + viewDir * random(0.02f);

      // apply tilt on top of the basis
      const float cx = cosf(tiltX);
      const float sx = sinf(tiltX);
      const float cz = cosf(tiltZ);
      const float sz = sinf(tiltZ);
      const vec3 v1(velocity.x * cx - velocity.y * sx, velocity.x * sx + velocity.y * cx, velocity.z);
      velocity = vec3(v1.x, v1.y, v1.z * cz - v1.y * sz);

      const vec3 color = glm::mix(pal.primary, pal.secondary, random(1.0f));

      const int ttl = lifetime + int(random(20));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = color,
          .currentColor = color,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .fadingOut = true,
          .emission = true,
      });
    }
  }

  void addWillowExplosion(vec3 pos, const ColorPalette& pal, int count) {
    for (int i = 0; i < count; i++) {
      const float theta = random(float(M_PI) * 2.0f);
      const float phi = randomRange(0.3f, float(M_PI) * 0.7f);
      const float r = randomRange(0.06f, 0.12f);

      const vec3 velocity(r * sinf(phi) * cosf(theta), r * cosf(phi) + 0.05f, r * sinf(phi) * sinf(theta));
      const vec3 color = glm::mix(pal.primary, pal.sparkle, random(0.5f));

      const int ttl = 150 + int(random(50));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = color,
          .currentColor = color,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .fadingOut = true,
          .emission = true,
          .brightness = randomRange(0.8f, 1.2f),
      });
    }
  }

  void addChrysanthemumExplosion(vec3 pos, const ColorPalette& pal, int count) {
    for (int i = 0; i < count; i++) {
      const float theta = random(float(M_PI) * 2.0f);
      const float phi = acosf(randomRange(-1.0f, 1.0f));
      const float r = randomRange(0.08f, 0.14f);

      const vec3 velocity(r * sinf(phi) * cosf(theta), r * sinf(phi) * sinf(theta), r * cosf(phi));
      const vec3 color = (random(1.0f) < 0.3f) ? pal.sparkle : pal.primary;

      const int ttl = 100 + int(random(40));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = color,
          .currentColor = color,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .fadingOut = true,
          .emission = true,
      });
    }

    for (int i = 0; i < 30; i++) {
      const float theta = random(float(M_PI) * 2.0f);
      const float phi = acosf(randomRange(-1.0f, 1.0f));
      const float r = randomRange(0.03f, 0.05f);

      const vec3 velocity(r * sinf(phi) * cosf(theta), r * sinf(phi) * sinf(theta), r * cosf(phi));

      const int sparkleTTL = 30 + int(random(20));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = pal.sparkle,
          .currentColor = pal.sparkle,
          .TTL = sparkleTTL,
          .initialLT = sparkleTTL,
          .alive = true,
          .fadingOut = true,
          .brightness = 2.0f,
      });
    }
  }

  void addCrackleExplosion(vec3 pos, const ColorPalette& pal, int count) {
    addSphereExplosion(pos, pal, count / 2, 0.08f, 60);

    for (int i = 0; i < 25; i++) {
      const float theta = random(float(M_PI) * 2.0f);
      const float phi = acosf(randomRange(-1.0f, 1.0f));
      const float r = randomRange(0.06f, 0.1f);

      const vec3 velocity(r * sinf(phi) * cosf(theta), r * sinf(phi) * sinf(theta), r * cosf(phi));

      const int ttl = 50 + int(random(30));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = pal.sparkle,
          .currentColor = pal.sparkle,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .spawnExplosion = true,
          .brightness = 1.5f,
          .explosionType = ExplosionType_Sphere,
          .paletteIndex = static_cast<int>(random(LVK_ARRAY_NUM_ELEMENTS(g_Palettes))),
      });
    }
  }

  void addPalmExplosion(vec3 pos, const ColorPalette& pal, int count) {
    for (int i = 0; i < count; i++) {
      const float angle = random(float(M_PI) * 2.0f);
      const float spread = randomRange(0.02f, 0.06f);
      const float upward = randomRange(0.1f, 0.15f);

      const vec3 velocity(spread * cosf(angle), upward, spread * sinf(angle));
      const vec3 color = glm::mix(pal.primary, pal.secondary, random(1.0f));

      const int ttl = 120 + int(random(40));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = color,
          .currentColor = color,
          .TTL = ttl,
          .initialLT = ttl,
          .alive = true,
          .fadingOut = true,
          .emission = true,
      });
    }

    for (int i = 0; i < 50; i++) {
      const float angle = random(float(M_PI) * 2.0f);
      const float spread = randomRange(0.03f, 0.08f);

      const vec3 velocity(spread * cosf(angle), 0.08f + random(0.04f), spread * sinf(angle));

      const int tipTTL = 80 + int(random(30));
      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = pal.sparkle,
          .currentColor = pal.sparkle,
          .TTL = tipTTL,
          .initialLT = tipTTL,
          .alive = true,
          .fadingOut = true,
          .brightness = 1.3f,
      });
    }
  }

  void addCrossetteExplosion(vec3 pos, const ColorPalette& pal, int count) {
    for (int i = 0; i < count; i++) {
      const float theta = (float(i) / count) * float(M_PI) * 2.0f;
      const float r = randomRange(0.08f, 0.12f);

      const vec3 velocity(r * cosf(theta), random(0.04f), r * sinf(theta));

      addParticle({
          .pos = pos,
          .velocity = velocity,
          .baseColor = pal.primary,
          .currentColor = pal.primary,
          .TTL = 80,
          .initialLT = 80,
          .alive = true,
          .spawnExplosion = true,
          .fadingOut = true,
          .emission = true,
          .explosionType = ExplosionType_Sphere,
          .paletteIndex = static_cast<int>(random(LVK_ARRAY_NUM_ELEMENTS(g_Palettes))),
          .secondaryExplosionTimer = 30 + int(random(20)),
      });
    }
  }

  void addExplosion(vec3 pos, int type = -1, int paletteIdx = -1) {
    if (type < 0)
      type = static_cast<int>(random(NumExplosionTypes));
    if (paletteIdx < 0)
      paletteIdx = static_cast<int>(random(LVK_ARRAY_NUM_ELEMENTS(g_Palettes)));

    const ColorPalette& pal = g_Palettes[paletteIdx % LVK_ARRAY_NUM_ELEMENTS(g_Palettes)];

    switch (type) {
    case ExplosionType_Sphere:
      addSphereExplosion(pos, pal, 350, 0.1f, 80);
      break;
    case ExplosionType_Ring:
      addRingExplosion(pos, pal, 120, 0.12f, 100, random(1.0f), random(1.0f));
      break;
    case ExplosionType_DoubleRing:
      addRingExplosion(pos, pal, 100, 0.1f, 90, 0.5f, 0.0f);
      addRingExplosion(pos, pal, 100, 0.1f, 90, -0.5f, 0.8f);
      addSphereExplosion(pos, {pal.sparkle, pal.sparkle, pal.sparkle}, 50, 0.04f, 40);
      break;
    case ExplosionType_Willow:
      addWillowExplosion(pos, pal, 400);
      break;
    case ExplosionType_Chrysanthemum:
      addChrysanthemumExplosion(pos, pal, 500);
      break;
    case ExplosionType_Crackle:
      addCrackleExplosion(pos, pal, 300);
      break;
    case ExplosionType_Palm:
      addPalmExplosion(pos, pal, 250);
      break;
    case ExplosionType_Crossette:
      addCrossetteExplosion(pos, pal, 12);
      addSphereExplosion(pos, pal, 100, 0.05f, 40);
      break;
    }

    // flash at explosion center
    for (int i = 0; i < 20; i++) {
      const int flashTTL = 5 + int(random(5));
      addParticle({
          .pos = pos,
          .velocity = vec3(randomRange(-0.01f, 0.01f), randomRange(-0.01f, 0.01f), randomRange(-0.01f, 0.01f)),
          .baseColor = vec3(1.0f),
          .currentColor = vec3(1.0f),
          .TTL = flashTTL,
          .initialLT = flashTTL,
          .alive = true,
          .fadingOut = true,
          .brightness = 3.0f,
      });
    }
  }

  void launchFirework(vec3 position) {
    const int explosionType = static_cast<int>(random(NumExplosionTypes));
    const int paletteIdx = static_cast<int>(random(LVK_ARRAY_NUM_ELEMENTS(g_Palettes)));
    const ColorPalette& pal = g_Palettes[paletteIdx];

    const vec3 velocity(randomRange(-0.03f, 0.03f), randomRange(0.22f, 0.32f), randomRange(-0.03f, 0.03f));

    const int flareTTL = 30 + int(random(20));
    addParticle({
        .pos = position,
        .velocity = velocity,
        .baseColor = pal.sparkle,
        .currentColor = pal.sparkle,
        .TTL = flareTTL,
        .initialLT = flareTTL,
        .alive = true,
        .flare = true,
        .spawnExplosion = true,
        .brightness = 1.5f,
        .explosionType = explosionType,
        .paletteIndex = paletteIdx,
    });

    const vec3 trailColor = pal.primary * 0.5f;
    for (int i = 0; i < 5; i++) {
      const int trailTTL = 10 + int(random(10));
      addParticle({
          .pos = position,
          .velocity = velocity * randomRange(0.1f, 0.3f),
          .baseColor = trailColor,
          .currentColor = trailColor,
          .TTL = trailTTL,
          .initialLT = trailTTL,
          .alive = true,
          .flare = true,
          .fadingOut = true,
          .brightness = 0.7f,
      });
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
  const VulkanAppConfig cfg{
#if defined(LVK_DEMO_WITH_OPENXR)
      .enableOpenXR = true,
#else
      .width = -90,
      .height = -90,
      .resizable = true,
#endif // LVK_DEMO_WITH_OPENXR
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
      .size = sizeof(PerFrame) * 2, // unified for both XR and non-XR variants
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
#if defined(LVK_DEMO_WITH_OPENXR)
          .format = ctx->getFormat(app.xrSwapchainTexture(0, 0)),
#else
          .format = ctx->getSwapchainFormat(),
#endif
          .blendEnabled = true,
          .rgbBlendOp = VK_BLEND_OP_ADD,
          .alphaBlendOp = VK_BLEND_OP_ADD,
          .srcRGBBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
          .srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
          .dstRGBBlendFactor = VK_BLEND_FACTOR_ONE,
          .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      }},
      .cullMode = VK_CULL_MODE_NONE,
      .debugName = "Pipeline: mesh",
  });

#if !defined(ANDROID) && !defined(LVK_DEMO_WITH_OPENXR)
#if LVK_WITH_GLFW
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
#elif LVK_WITH_SDL3
  app.addKeyCallback([](SDL_Window* window, SDL_KeyboardEvent* event) {
    if (event->key == SDLK_1 && event->down) {
      g_Gravity.x += 0.001f;
    }
    if (event->key == SDLK_2 && event->down) {
      g_Gravity.x -= 0.001f;
    }
    if (event->key == SDLK_SPACE && event->down) {
      g_Pause = !g_Pause;
    }
  });
#endif
#endif // !ANDROID && !LVK_DEMO_WITH_OPENXR

  std::vector<Vertex> vertices;
  vertices.reserve(kMaxParticles);

  const float kTimeQuantum = 0.02f;
  double accTime = 0;
  uint32_t bufferIndex = 0;
  float launchTimer = 0.0f;

#if defined(LVK_DEMO_WITH_OPENXR)
  g_Points.useViewerFacingExplosions = true;
#endif

  app.run([&](lvk::Span<const RenderView> views, float deltaSeconds) {
    LVK_PROFILER_FUNCTION();

    // simulation
    if (!g_Pause) {
      const float dt = app.cfg_.screenshotFrameNumber ? kTimeQuantum : deltaSeconds;
      accTime += dt;
      launchTimer += dt;
    }

    while (accTime >= kTimeQuantum) {
      accTime -= kTimeQuantum;
#if defined(LVK_DEMO_WITH_OPENXR)
      const mat4 headPose = glm::inverse(views[0].view);
      g_Points.viewerPos = vec3(headPose[3]);
#endif
      g_Points.nextFrame();
      if (launchTimer >= randomRange(0.7f, 1.8f)) {
        launchTimer = 0.0f;
#if defined(LVK_DEMO_WITH_OPENXR)
        // launch 20m ahead of the viewer along the XZ forward direction, with lateral spread
        const mat4 hp = glm::inverse(views[0].view);
        const vec3 viewerPos = vec3(hp[3]);
        const vec3 fwd3 = vec3(hp * vec4(0.0f, 0.0f, -1.0f, 0.0f));
        const vec2 fwd = glm::normalize(vec2(fwd3.x, fwd3.z));
        const vec2 perp = vec2(-fwd.y, fwd.x);
        const vec2 launchXZ = vec2(viewerPos.x, viewerPos.z) + fwd * 20.0f + perp * randomRange(-5.0f, 5.0f);
        const vec3 position(launchXZ.x, viewerPos.y - 5.0f, launchXZ.y);
#else
        const vec3 position(randomRange(-5.0f, 5.0f), -6.0f, randomRange(-2.0f, 2.0f));
#endif
        g_Points.launchFirework(position);
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

    // upload PerFrame for all views
    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    for (uint32_t i = 0; i != views.size(); i++) {
#if defined(LVK_DEMO_WITH_OPENXR)
      const PerFrame perFrame = {.proj = views[i].proj, .view = views[i].view};
#else
      const PerFrame perFrame = {
          .proj = glm::perspective(glm::radians(90.0f), views[i].aspectRatio, 0.1f, 100.0f),
          .view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, -8.0f)),
      };
#endif
      buffer.cmdUpdateBuffer(bufPerFrame, perFrame, i * sizeof(PerFrame));
    }

    // render all views
    for (uint32_t i = 0; i != views.size(); i++) {
      buffer.cmdBeginRendering(
          lvk::RenderPass{
              .color = {{.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}}},
          },
          lvk::Framebuffer{
              .color = {{.texture = views[i].colorTexture}},
          });

      buffer.cmdBindRenderPipeline(renderPipelineState_Mesh_);
      buffer.cmdBindViewport(views[i].viewport);
      buffer.cmdBindScissorRect(views[i].scissorRect);
      buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
      const struct {
        uint64_t perFrame;
        uint64_t vb;
        uint32_t texture;
      } bindings = {
          .perFrame = ctx->gpuAddress(bufPerFrame, i * sizeof(PerFrame)),
          .vb = ctx->gpuAddress(vb0_[bufferIndex]),
          .texture = texture_.index(),
      };
      buffer.cmdPushConstants(bindings);
      if (!vertices.empty()) {
        buffer.cmdDrawMeshTasks({(uint32_t)vertices.size(), 1, 1});
      }
      buffer.cmdPopDebugGroupLabel();

      buffer.cmdEndRendering();
    }

#if !defined(LVK_DEMO_WITH_OPENXR)
    // ImGui overlay (non-XR only)
    const lvk::Framebuffer framebuffer = {.color = {{.texture = views[0].colorTexture}}};
    buffer.cmdBeginRendering(
        lvk::RenderPass{
            .color = {{.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE}},
        },
        framebuffer);
    app.imgui_->beginFrame(framebuffer);
    ImGui::SetNextWindowPos({0, 0});
    ImGui::Begin("Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
    ImGui::Text("Particles: %i", g_Points.totalParticles);
    ImGui::End();
    app.drawFPS();
    app.imgui_->endFrame(buffer);
    buffer.cmdEndRendering();
    ctx->submit(buffer, views[0].colorTexture);
#else
    ctx->submit(buffer);
#endif
  });

  VULKAN_APP_EXIT();
}
