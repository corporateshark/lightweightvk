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

// disable for better perf & benchmarking (ImGui does not use input attachments)
#define ENABLE_IMGUI_DEBUG_OVERLAY 1

// Bilingual: GLSL (default) and Slang. Define the macro LVK_DEMO_WITH_SLANG to switch to Slang.

const char* codeSlangDeferred = R"(
struct PerFrame {
  float4x4 mvp;
  float4x4 model;
  float4 cameraPos;
  uint texture0;
  uint texture1;
  uint texture2;
};

struct PushConstants {
  PerFrame* perFrame;
};

[[vk::push_constant]] PushConstants pc;

static const float3 positions[24] = {
  float3(-1.0, -1.0,  1.0), float3( 1.0, -1.0,  1.0), float3( 1.0,  1.0,  1.0), float3(-1.0,  1.0,  1.0), // +Z
  float3( 1.0, -1.0, -1.0), float3(-1.0, -1.0, -1.0), float3(-1.0,  1.0, -1.0), float3( 1.0,  1.0, -1.0), // -Z
  float3( 1.0, -1.0,  1.0), float3( 1.0, -1.0, -1.0), float3( 1.0,  1.0, -1.0), float3( 1.0,  1.0,  1.0), // +X
  float3(-1.0, -1.0, -1.0), float3(-1.0, -1.0,  1.0), float3(-1.0,  1.0,  1.0), float3(-1.0,  1.0, -1.0), // -X
  float3(-1.0,  1.0,  1.0), float3( 1.0,  1.0,  1.0), float3( 1.0,  1.0, -1.0), float3(-1.0,  1.0, -1.0), // +Y
  float3(-1.0, -1.0, -1.0), float3( 1.0, -1.0, -1.0), float3( 1.0, -1.0,  1.0), float3(-1.0, -1.0,  1.0)  // -Y
};

static const float3 normals[24] = {
  float3( 0.0,  0.0,  1.0), float3( 0.0,  0.0,  1.0), float3( 0.0,  0.0,  1.0), float3( 0.0,  0.0,  1.0), // +Z
  float3( 0.0,  0.0, -1.0), float3( 0.0,  0.0, -1.0), float3( 0.0,  0.0, -1.0), float3( 0.0,  0.0, -1.0), // -Z
  float3( 1.0,  0.0,  0.0), float3( 1.0,  0.0,  0.0), float3( 1.0,  0.0,  0.0), float3( 1.0,  0.0,  0.0), // +X
  float3(-1.0,  0.0,  0.0), float3(-1.0,  0.0,  0.0), float3(-1.0,  0.0,  0.0), float3(-1.0,  0.0,  0.0), // -X
  float3( 0.0,  1.0,  0.0), float3( 0.0,  1.0,  0.0), float3( 0.0,  1.0,  0.0), float3( 0.0,  1.0,  0.0), // +Y
  float3( 0.0, -1.0,  0.0), float3( 0.0, -1.0,  0.0), float3( 0.0, -1.0,  0.0), float3( 0.0, -1.0,  0.0)  // -Y
};

static const float2 uvs[24] = {
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0), // +Z
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0), // -Z
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0), // +X
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0), // -X
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0), // +Y
  float2(0.0, 1.0), float2(1.0, 1.0), float2(1.0, 0.0), float2(0.0, 0.0)  // -Y
};

float3x3 toFloat3x3(float4x4 m) {
  return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

struct DeferredVSOutput {
  float4 pos      : SV_Position;
  float2 uv       : TEXCOORD0;
  float3 normal   : NORMAL;
  float3 worldPos : TEXCOORD1;
  float3 eyeDir   : TEXCOORD2;
  uint textureId  : TEXCOORD3;
  uint normalId   : TEXCOORD4;
  uint heightId   : TEXCOORD5;
};

[shader("vertex")]
DeferredVSOutput vertexMain(uint vertexId : SV_VertexID) {
  DeferredVSOutput out;

  float3 pos = positions[vertexId];
  float3 worldPos = (pc.perFrame->model * float4(pos, 1.0)).xyz;

  out.pos       = pc.perFrame->mvp * float4(pos, 1.0);
  out.uv        = uvs[vertexId];
  out.normal    = toFloat3x3(pc.perFrame->model) * normals[vertexId];
  out.worldPos  = worldPos;
  out.eyeDir    = pc.perFrame->cameraPos.xyz - worldPos;
  out.textureId = pc.perFrame->texture0;
  out.normalId  = pc.perFrame->texture1;
  out.heightId  = pc.perFrame->texture2;

  return out;
}

struct DeferredFSOutput {
  float4 fragColor : SV_Target0; // unused
  float4 albedo    : SV_Target1;
  float4 normal    : SV_Target2;
  float4 worldPos  : SV_Target3;
};

static const float kHeightScale   = 0.03;
static const int   kParallaxSteps = 8;

// derivative-based orthonormal TBN (rows = T, B, N): http://www.thetenthplanet.de/archives/1180
float3x3 cotangentFrame(float3 N, float3 p, float2 uv) {
  float3 dp1 = ddx(p);
  float3 dp2 = ddy(p);
  float2 duv1 = ddx(uv);
  float2 duv2 = ddy(uv);
  float3 dp2perp = cross(dp2, N);
  float3 dp1perp = cross(N, dp1);
  float3 T = normalize(dp2perp * duv1.x + dp1perp * duv2.x);
  float3 B = normalize(dp2perp * duv1.y + dp1perp * duv2.y);
  return float3x3(T, B, N);
}

// Steep parallax + binary-search refinement; viewTS points surface -> eye.
// Linear search brackets the intersection in <=8 fetches; 3 binary halvings refine it;
// a final secant lerp gives precision equivalent to an 8 * 2^3 = 64-step linear search.
float2 parallaxUV(float2 uv, float3 viewTS, uint heightId) {
  const float layerStep = 1.0 / float(kParallaxSteps);
  const float2 deltaUV = (viewTS.xy / max(viewTS.z, 0.01)) * (kHeightScale / float(kParallaxSteps));

  float2 cur = uv;
  float  curLayer = 0.0;
  float  curDepth = 1.0 - textureBindless2D(heightId, 0, cur).r;

  // track the previous (above-surface) sample inside the loop so no extra fetch is needed after
  float2 prevCur = cur;
  float  prevLayer = curLayer;
  float  prevDepth = curDepth;

  [loop] for (int i = 0; i < kParallaxSteps && curLayer < curDepth; ++i) {
    prevCur = cur; prevLayer = curLayer; prevDepth = curDepth;
    cur -= deltaUV;
    curLayer += layerStep;
    curDepth = 1.0 - textureBindless2D(heightId, 0, cur).r;
  }

  [unroll] for (int k = 0; k < 3; ++k) {
    const float2 midCur = 0.5 * (prevCur + cur);
    const float  midLayer = 0.5 * (prevLayer + curLayer);
    const float  midDepth = 1.0 - textureBindless2D(heightId, 0, midCur).r;
    const bool advance = midLayer < midDepth; // mid is above the surface — push prev forward
    prevCur   = advance ? midCur   : prevCur;
    prevLayer = advance ? midLayer : prevLayer;
    prevDepth = advance ? midDepth : prevDepth;
    cur       = advance ? cur      : midCur;
    curLayer  = advance ? curLayer : midLayer;
    curDepth  = advance ? curDepth : midDepth;
  }

  const float after  = curDepth - curLayer;
  const float before = prevDepth - prevLayer;
  const float weight = after / (after - before);
  return lerp(cur, prevCur, weight);
}

[shader("fragment")]
DeferredFSOutput fragmentMain(DeferredVSOutput input) {
  DeferredFSOutput out;

  float3 N = normalize(input.normal);
  float3 V = normalize(input.eyeDir);
  float3x3 TBN = cotangentFrame(N, input.worldPos, input.uv);

  // rows of TBN are T, B, N so mul(TBN, V) = (T.V, B.V, N.V) — world -> tangent
  float3 viewTS = mul(TBN, V);
  float2 uv = parallaxUV(input.uv, viewTS, input.heightId);

  float3 nm = textureBindless2D(input.normalId, 0, uv).xyz * 2.0 - 1.0;
  // mul(nm, TBN) = nm.x*T + nm.y*B + nm.z*N — tangent -> world
  float3 n  = normalize(mul(nm, TBN));

  out.fragColor = float4(0, 0, 0, 1);
  out.albedo   = 10.0 * textureBindless2D(input.textureId, 0, uv);
  out.normal   = float4(n * 0.5 + 0.5, 1.0);
  out.worldPos = float4(input.worldPos, 1.0);

  return out;
}
)";

const char* codeSlangCompose = R"(
struct ComposeVSOutput {
  float4 pos : SV_Position;
  float2 uv  : TEXCOORD0;
};

[shader("vertex")]
ComposeVSOutput vertexMain(uint vertexId : SV_VertexID) {
  ComposeVSOutput out;

  out.uv  = float2((vertexId << 1) & 2, vertexId & 2);
  out.pos = float4(out.uv * float2(2, -2) + float2(-1, 1), 0.0, 1.0);

  return out;
}

[[vk::input_attachment_index(0)]] [[vk::binding(0, 1)]] SubpassInput inputAlbedo;
[[vk::input_attachment_index(1)]] [[vk::binding(1, 1)]] SubpassInput inputNormal;
[[vk::input_attachment_index(2)]] [[vk::binding(2, 1)]] SubpassInput inputWorldPos;

[shader("fragment")]
float4 fragmentMain(ComposeVSOutput input) : SV_Target0 {
  // sample G-buffer via input attachments (reads at current fragment position)
  float4 albedo   = inputAlbedo.SubpassLoad();
  float3 normal   = inputNormal.SubpassLoad().xyz * 2.0 - 1.0;
  float3 worldPos = inputWorldPos.SubpassLoad().xyz;

  float3 lightDir = normalize(float3(1, 1, -1) - worldPos);

  float NdotL = clamp(dot(normal, lightDir), 0.3, 1.0);

  return float4(NdotL * albedo.rgb, 1.0);
}
)";

const char* codeDeferredVS = R"(
layout (location=0) out vec2 out_UV;
layout (location=1) out vec3 out_Normal;
layout (location=2) out vec3 out_WorldPos;
layout (location=3) out flat uint out_TextureId;
layout (location=4) out flat uint out_NormalId;
layout (location=5) out flat uint out_HeightId;
layout (location=6) out vec3 out_EyeDir;

const vec3 positions[24] = vec3[24](
  vec3(-1.0, -1.0,  1.0), vec3( 1.0, -1.0,  1.0), vec3( 1.0,  1.0,  1.0), vec3(-1.0,  1.0,  1.0), // +Z
  vec3( 1.0, -1.0, -1.0), vec3(-1.0, -1.0, -1.0), vec3(-1.0,  1.0, -1.0), vec3( 1.0,  1.0, -1.0), // -Z
  vec3( 1.0, -1.0,  1.0), vec3( 1.0, -1.0, -1.0), vec3( 1.0,  1.0, -1.0), vec3( 1.0,  1.0,  1.0), // +X
  vec3(-1.0, -1.0, -1.0), vec3(-1.0, -1.0,  1.0), vec3(-1.0,  1.0,  1.0), vec3(-1.0,  1.0, -1.0), // -X
  vec3(-1.0,  1.0,  1.0), vec3( 1.0,  1.0,  1.0), vec3( 1.0,  1.0, -1.0), vec3(-1.0,  1.0, -1.0), // +Y
  vec3(-1.0, -1.0, -1.0), vec3( 1.0, -1.0, -1.0), vec3( 1.0, -1.0,  1.0), vec3(-1.0, -1.0,  1.0)  // -Y
);

const vec3 normals[24] = vec3[24](
  vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0), // +Z
  vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0), // -Z
  vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0), // +X
  vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), // -X
  vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0), // +Y
  vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0)  // -Y
);

const vec2 uvs[24] = vec2[24](
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), // +Z
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), // -Z
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), // +X
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), // -X
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0), // +Y
  vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0), vec2(0.0, 0.0)  // -Y
);

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 mvp;
  mat4 model;
  vec4 cameraPos;
  uint texture0;
  uint texture1;
  uint texture2;
};

layout(push_constant) uniform constants {
  PerFrame pc;
};

void main() {
  vec3 pos = positions[gl_VertexIndex];
  vec3 worldPos = (pc.model * vec4(pos, 1.0)).xyz;

  gl_Position = pc.mvp * vec4(pos, 1.0);

  out_UV = uvs[gl_VertexIndex];
  out_Normal = mat3(pc.model) * normals[gl_VertexIndex];
  out_WorldPos = worldPos;
  out_EyeDir = pc.cameraPos.xyz - worldPos;
  out_TextureId = pc.texture0;
  out_NormalId = pc.texture1;
  out_HeightId = pc.texture2;
}
)";

const char* codeDeferredFS = R"(
layout (location=0) in vec2 in_UV;
layout (location=1) in vec3 in_Normal;
layout (location=2) in vec3 in_WorldPos;
layout (location=3) in flat uint in_TextureId;
layout (location=4) in flat uint in_NormalId;
layout (location=5) in flat uint in_HeightId;
layout (location=6) in vec3 in_EyeDir;

layout (location=0) out vec4 out_FragColor; // unused
layout (location=1) out vec4 out_Albedo;
layout (location=2) out vec4 out_Normal;
layout (location=3) out vec4 out_WorldPos;

const float kHeightScale   = 0.03;
const int   kParallaxSteps = 8;

// derivative-based orthonormal TBN: http://www.thetenthplanet.de/archives/1180
mat3 cotangentFrame(vec3 N, vec3 p, vec2 uv) {
  vec3 dp1 = dFdx(p);
  vec3 dp2 = dFdy(p);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);
  vec3 dp2perp = cross(dp2, N);
  vec3 dp1perp = cross(N, dp1);
  vec3 T = normalize(dp2perp * duv1.x + dp1perp * duv2.x);
  vec3 B = normalize(dp2perp * duv1.y + dp1perp * duv2.y);
  return mat3(T, B, N);
}

// Steep parallax + binary-search refinement; viewTS points surface -> eye.
// Linear search brackets the intersection in <=8 fetches; 3 binary halvings refine it;
// a final secant lerp gives precision equivalent to an 8 * 2^3 = 64-step linear search.
vec2 parallaxUV(vec2 uv, vec3 viewTS, uint heightId) {
  const float layerStep = 1.0 / float(kParallaxSteps);
  const vec2 deltaUV = (viewTS.xy / max(viewTS.z, 0.01)) * (kHeightScale / float(kParallaxSteps));

  vec2  cur = uv;
  float curLayer = 0.0;
  float curDepth = 1.0 - textureBindless2D(heightId, 0, cur).r;

  // track the previous (above-surface) sample inside the loop so no extra fetch is needed after
  vec2  prevCur = cur;
  float prevLayer = curLayer;
  float prevDepth = curDepth;

  for (int i = 0; i < kParallaxSteps && curLayer < curDepth; ++i) {
    prevCur = cur; prevLayer = curLayer; prevDepth = curDepth;
    cur -= deltaUV;
    curLayer += layerStep;
    curDepth = 1.0 - textureBindless2D(heightId, 0, cur).r;
  }

  for (int k = 0; k < 3; ++k) {
    const vec2  midCur = 0.5 * (prevCur + cur);
    const float midLayer = 0.5 * (prevLayer + curLayer);
    const float midDepth = 1.0 - textureBindless2D(heightId, 0, midCur).r;
    const bool advance = midLayer < midDepth; // mid is above the surface — push prev forward
    prevCur   = advance ? midCur   : prevCur;
    prevLayer = advance ? midLayer : prevLayer;
    prevDepth = advance ? midDepth : prevDepth;
    cur       = advance ? cur      : midCur;
    curLayer  = advance ? curLayer : midLayer;
    curDepth  = advance ? curDepth : midDepth;
  }

  const float after  = curDepth - curLayer;
  const float before = prevDepth - prevLayer;
  const float weight = after / (after - before);
  return mix(cur, prevCur, weight);
}

void main() {
  vec3 N = normalize(in_Normal);
  vec3 V = normalize(in_EyeDir);
  mat3 TBN = cotangentFrame(N, in_WorldPos, in_UV);

  vec3 viewTS = V * TBN; // world -> tangent (= transpose(TBN) * V for orthonormal TBN)
  vec2 uv = parallaxUV(in_UV, viewTS, in_HeightId);

  vec3 nm = textureBindless2D(in_NormalId, 0, uv).xyz * 2.0 - 1.0;
  vec3 n  = normalize(TBN * nm);

  out_Albedo   = 10.0 * textureBindless2D(in_TextureId, 0, uv);
  out_Normal   = vec4(n * 0.5 + 0.5, 1.0);
  out_WorldPos = vec4(in_WorldPos, 1.0);
}
)";

const char* codeComposeVS = R"(
layout (location=0) out vec2 uv;

void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* codeComposeFS = R"(
#if defined(has_EXT_shader_tile_image)
  #extension GL_EXT_shader_tile_image : require
  layout (location=1) tileImageEXT highp attachmentEXT inputAlbedo;
  layout (location=2) tileImageEXT highp attachmentEXT inputNormal;
  layout (location=3) tileImageEXT highp attachmentEXT inputWorldPos;
#else
  layout (input_attachment_index=0, set=1, binding=0) uniform subpassInput inputAlbedo;
  layout (input_attachment_index=1, set=1, binding=1) uniform subpassInput inputNormal;
  layout (input_attachment_index=2, set=1, binding=2) uniform subpassInput inputWorldPos;
#endif // defined(has_EXT_shader_tile_image)

layout (location=0) in vec2 in_UV;

layout (location=0) out vec4 out_FragColor;

void main() {
  // sample G-buffer via input attachments (reads at current fragment position)
#if defined(has_EXT_shader_tile_image)
  vec4 albedo   = colorAttachmentReadEXT(inputAlbedo);
  vec3 normal   = colorAttachmentReadEXT(inputNormal).xyz * 2.0 - 1.0;
  vec3 worldPos = colorAttachmentReadEXT(inputWorldPos).xyz;
#else
  vec4 albedo   = subpassLoad(inputAlbedo);
  vec3 normal   = subpassLoad(inputNormal).xyz * 2.0 - 1.0; // from [0,1] to [-1,1]
  vec3 worldPos = subpassLoad(inputWorldPos).xyz;
#endif // defined(has_EXT_shader_tile_image)

  vec3 lightDir = normalize(vec3(1, 1, -1) - worldPos);

  float NdotL = clamp(dot(normal, lightDir), 0.3, 1.0);

  out_FragColor = vec4(NdotL * albedo.rgb, 1.0);
}
)";

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 0,
      .height = 0,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  {
    const uint16_t indexData[36] = {0,  1,  2,  2,  3,  0,  4,  5,  6,  6,  7,  4,  8,  9,  10, 10, 11, 8,
                                    12, 13, 14, 14, 15, 12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20};

    lvk::Holder<lvk::BufferHandle> ib0_ = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Index,
        .storage = lvk::StorageType_Device,
        .size = sizeof(indexData),
        .data = indexData,
        .debugName = "Buffer: index",
    });

    const lvk::Dimensions dim = ctx->getDimensions(ctx->getCurrentSwapchainTexture());

    lvk::Holder<lvk::TextureHandle> texAlbedo = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_BGRA_UN8,
        .dimensions = dim,
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_InputAttachment,
        .debugName = "Albedo",
    });

    lvk::Holder<lvk::TextureHandle> texNormal = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_A2B10G10R10_UN,
        .dimensions = dim,
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_InputAttachment,
        .debugName = "Normals",
    });

    lvk::Holder<lvk::TextureHandle> texWorldPos = ctx->createTexture({
        .type = lvk::TextureType_2D,
        .format = lvk::Format_BGRA_UN8,
        .dimensions = dim,
        .usage = lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_InputAttachment,
        .debugName = "WorldPositions",
    });

    lvk::Holder<lvk::TextureHandle> texMatAlbedo;
    lvk::Holder<lvk::TextureHandle> texMatNormal;
    lvk::Holder<lvk::TextureHandle> texMatHeight;

    {
      using namespace std::filesystem;
      const path dir = path(app.folderThirdParty_) / path("ktx-software/tests/srcimages/Iron_Bars/");
      struct MatSlot {
        const char* fileName;
        lvk::Holder<lvk::TextureHandle>* out;
      };
      const MatSlot slots[] = {
          {"Iron_Bars_001_basecolor.jpg", &texMatAlbedo},
          {"Iron_Bars_001_normal.jpg", &texMatNormal},
          {"Iron_Bars_001_height.png", &texMatHeight},
      };
      for (const MatSlot& s : slots) {
        int32_t texWidth = 0;
        int32_t texHeight = 0;
        int32_t channels = 0;
        uint8_t* pixels = stbi_load((dir / path(s.fileName)).string().c_str(), &texWidth, &texHeight, &channels, 4);
        SCOPE_EXIT {
          stbi_image_free(pixels);
        };
        if (!pixels) {
          LVK_ASSERT_MSG(false, "Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
          LLOGW("Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
          std::terminate();
        }
        *s.out = ctx->createTexture({
            .type = lvk::TextureType_2D,
            .format = lvk::Format_RGBA_UN8,
            .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
            .usage = lvk::TextureUsageBits_Sampled,
            .data = pixels,
            .generateMipmaps = true,
            .debugName = s.fileName,
        });
      }
    }

#if defined(LVK_DEMO_WITH_SLANG)
    const bool has_EXT_shader_tile_image = false; // not implemented in Slang
    lvk::Holder<lvk::ShaderModuleHandle> vertDeferred =
        ctx->createShaderModule({codeSlangDeferred, lvk::Stage_Vert, "Shader Module: deferred (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragDeferred =
        ctx->createShaderModule({codeSlangDeferred, lvk::Stage_Frag, "Shader Module: deferred (frag)"});
    lvk::Holder<lvk::ShaderModuleHandle> vertCompose =
        ctx->createShaderModule({codeSlangCompose, lvk::Stage_Vert, "Shader Module: compose (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragCompose =
        ctx->createShaderModule({codeSlangCompose, lvk::Stage_Frag, "Shader Module: compose (frag)"});
#else
    const bool has_EXT_shader_tile_image = ctx->isExtensionEnabled("VK_EXT_shader_tile_image");
    lvk::Holder<lvk::ShaderModuleHandle> vertDeferred =
        ctx->createShaderModule({codeDeferredVS, lvk::Stage_Vert, "Shader Module: deferred (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragDeferred =
        ctx->createShaderModule({codeDeferredFS, lvk::Stage_Frag, "Shader Module: deferred (frag)"});
    lvk::Holder<lvk::ShaderModuleHandle> vertCompose =
        ctx->createShaderModule({codeComposeVS, lvk::Stage_Vert, "Shader Module: compose (vert)"});
    lvk::Holder<lvk::ShaderModuleHandle> fragCompose = ctx->createShaderModule(
        {(has_EXT_shader_tile_image ? std::string("#define has_EXT_shader_tile_image 1\n") + codeComposeFS : codeComposeFS).c_str(),
         lvk::Stage_Frag,
         "Shader Module: compose (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

    lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Deferred = ctx->createRenderPipeline({
        .smVert = vertDeferred,
        .smFrag = fragDeferred,
        .color =
            {
                {.format = ctx->getSwapchainFormat()},
                {.format = ctx->getFormat(texAlbedo)},
                {.format = ctx->getFormat(texNormal)},
                {.format = ctx->getFormat(texWorldPos)},
            },
        .cullMode = lvk::CullMode_Back,
        .frontFace = lvk::WindingMode_CW,
        .debugName = "Pipeline: deferred",
    });
    lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Compose = ctx->createRenderPipeline({
        .smVert = vertCompose,
        .smFrag = fragCompose,
        .color =
            {
                {.format = ctx->getSwapchainFormat()},
                {.format = ctx->getFormat(texAlbedo)},
                {.format = ctx->getFormat(texNormal)},
                {.format = ctx->getFormat(texWorldPos)},
            },
        .debugName = "Pipeline: compose",
    });

    struct PerFrame {
      mat4 mvp;
      mat4 model;
      vec4 cameraPos;
      uint32_t texture;
      uint32_t textureNormal;
      uint32_t textureHeight;
    };

    lvk::Holder<lvk::BufferHandle> perFrameBuffer = ctx->createBuffer({
        .usage = lvk::BufferUsageBits_Uniform,
        .storage = lvk::StorageType_Device,
        .size = sizeof(PerFrame),
        .debugName = "Buffer: perFrame",
    });

    // main loop
    app.run([&](lvk::Span<const RenderView> views, float deltaSeconds) {
      LVK_PROFILER_FUNCTION();

      const float fov = float(45.0f * (M_PI / 180.0f));
      const mat4 proj = glm::perspectiveLH(fov, views[0].aspectRatio, 0.1f, 500.0f);
      const mat4 view = glm::translate(mat4(1.0f), vec3(0.0f, 0.0f, 5.0f));
      const mat4 model = glm::rotate(mat4(1.0f), (float)app.getSimulatedTime(), glm::normalize(vec3(1.0f, 1.0f, 1.0f)));
      const vec4 cameraPos = glm::inverse(view) * vec4(0.0f, 0.0f, 0.0f, 1.0f);
      const PerFrame bindingsDeferred = {
          .mvp = proj * view * model,
          .model = model,
          .cameraPos = cameraPos,
          .texture = texMatAlbedo.index(),
          .textureNormal = texMatNormal.index(),
          .textureHeight = texMatHeight.index(),
      };
      const lvk::Framebuffer framebuffer = {
          .color = {{.texture = ctx->getCurrentSwapchainTexture()},
                    {.texture = texAlbedo},
                    {.texture = texNormal},
                    {.texture = texWorldPos}},
      };

      lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

      buffer.cmdUpdateBuffer(perFrameBuffer, 0, sizeof(bindingsDeferred), &bindingsDeferred);
      buffer.cmdBeginRendering(
          {.color =
               {
                   {.loadOp = lvk::LoadOp_DontCare, .storeOp = lvk::StoreOp_Store},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}},
                   {.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}},
               }},
          framebuffer,
          {.inputAttachments = {texAlbedo, texNormal, texWorldPos}});
      buffer.cmdPushDebugGroupLabel("Render deferred", 0xff0000ff);
      buffer.cmdBindRenderPipeline(renderPipelineState_Deferred);
      buffer.cmdPushConstants(ctx->gpuAddress(perFrameBuffer));
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
      buffer.cmdDrawIndexed(36);
      buffer.cmdPopDebugGroupLabel();

      if (!has_EXT_shader_tile_image) {
        buffer.cmdNextSubpass();
      }

      buffer.cmdPushDebugGroupLabel("Compose", 0xff0000ff);
      buffer.cmdBindRenderPipeline(renderPipelineState_Compose);
      buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI16);
      buffer.cmdDraw(3);
      buffer.cmdPopDebugGroupLabel();

#if ENABLE_IMGUI_DEBUG_OVERLAY
      // ImGui textures do not use input attachments
      buffer.cmdEndRendering();
      const lvk::Framebuffer framebufferGUI = {
          .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
      };
      buffer.cmdBeginRendering(
          {.color = {{.loadOp = lvk::LoadOp_Load, .storeOp = lvk::StoreOp_Store}}},
          framebufferGUI,
          {.sampledImages = {texAlbedo, texNormal, texWorldPos}});
      app.imgui_->beginFrame(framebufferGUI);
      const ImGuiViewport* v = ImGui::GetMainViewport();
      const float size = 0.175f * v->WorkSize.x;
      ImGui::SetNextWindowPos({0, 15}, ImGuiCond_Always);
      ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoMove);
      ImGui::Text("Albedo:");
      ImGui::Image(texAlbedo.index(), ImVec2(size, size / views[0].aspectRatio));
      ImGui::Text("Normals:");
      ImGui::Image(texNormal.index(), ImVec2(size, size / views[0].aspectRatio));
      ImGui::Text("World positions:");
      ImGui::Image(texWorldPos.index(), ImVec2(size, size / views[0].aspectRatio));
      ImGui::End();
#else
      app.imgui_->beginFrame(framebuffer);
#endif // ENABLE_IMGUI_DEBUG_OVERLAY
      app.drawFPS();
      app.imgui_->endFrame(buffer);
      buffer.cmdEndRendering();
      ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
    });
  }

  VULKAN_APP_EXIT();
}
