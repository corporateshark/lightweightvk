/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const char* codeComputeTestSlang = R"(
struct PushConstants {
  uint tex;
  uint width;
  uint height;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(2, 0)]] RWTexture2D<float4> kTextures2DInOut[];

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 globalID : SV_DispatchThreadID) {
  int2 pos = int2(globalID.xy);
  if (pos.x < pc.width && pos.y < pc.height) {
    float4 pixel = kTextures2DInOut[pc.tex][pos];
    float luminance = dot(pixel, float4(0.299, 0.587, 0.114, 0.0)); // https://www.w3.org/TR/AERT/#color-contrast
    kTextures2DInOut[pc.tex][pos] = float4(luminance.xxx, 1.0);
  }
}
)";

const char* codeFullscreenSlang = R"(
struct PushConstants {
  uint tex;
};

struct VSOutput {
  float4 sv_Position : SV_Position;
  float2 uv          : TEXCOORD0;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID) {
  VSOutput out;
  
  // generate a triangle covering the entire screen
  out.uv = float2((vertexID << 1) & 2, vertexID & 2);
  out.sv_Position = float4(out.uv * float2(2, -2) + float2(-1, 1), 0.0, 1.0);

  return out;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0 {
  return textureBindless2D(pc.tex, 0, input.uv);
}
)";

const char* codeSlang = R"(
struct Material {
  float4 ambient;
  float4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};

struct PerFrame {
  float4x4 proj;
  float4x4 view;
  float4x4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

struct PerObject {
  float4x4 model;
  float4x4 normal;
};

struct Materials {
  Material mtl[];
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
  Materials* materials;
};

struct PerVertex {
  float3 normal : NORMAL;
  float2 uv : TEXCOORD0;
  float4 shadowCoords : TEXCOORD1;
};

struct VSOutput {
  PerVertex vtx;
  Material mtl : MATERIAL;
  float4 position : SV_Position;
};

[[vk::push_constant]] PushConstants pc;

[[vk::constant_id(0)]] const bool bDrawNormals = false;

// https://www.shadertoy.com/view/llfcRl
float2 unpackSnorm2x8(uint d) {
  return float2(uint2(d, d >> 8) & 255u) / 127.5 - 1.0;
}

float3 unpackOctahedral16(uint data) {
  float2 v = unpackSnorm2x8(data);
  // https://x.com/Stubbesaurus/status/937994790553227264
  float3 n = float3(v, 1.0 - abs(v.x) - abs(v.y));
  float t = max(-n.z, 0.0);
  n.x += (n.x > 0.0) ? -t : t;
  n.y += (n.y > 0.0) ? -t : t;
  return normalize(n);
}

[shader("vertex")]
VSOutput vertexMain(
  float3 pos : POSITION,
  float2 uv : TEXCOORD0,
  uint normal : NORMAL,
  uint mtlIndex : MATERIAL_INDEX
) {
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;
  float4x4 light = pc.perFrame->light;

  VSOutput out;
  
  out.mtl = pc.materials->mtl[mtlIndex];
  out.position = proj * view * model * float4(pos, 1.0);
  
  // compute the normal in world-space
  out.vtx.normal = normalize(float3x3(pc.perObject->normal) * unpackOctahedral16(normal));
  out.vtx.uv = uv;
  out.vtx.shadowCoords = light * model * float4(pos, 1.0);
  
  return out;
}

float PCF3(float3 uvw) {
  float size = 1.0 / textureBindlessSize2D(pc.perFrame->texShadow).x;
  float shadow = 0.0;
  for (int v = -1; v <= +1; v++)
    for (int u = -1; u <= +1; u++)
      shadow += textureBindless2DShadow(pc.perFrame->texShadow, pc.perFrame->samplerShadow0, uvw + size * float3(u, v, 0));
  return shadow / 9;
}

float shadow(float4 s) {
  s = s / s.w;
  if (s.z > -1.0 && s.z < 1.0) {
    float depthBias = -0.00005;
    float shadowSample = PCF3(float3(s.x, 1.0 - s.y, s.z + depthBias));
    return lerp(0.3, 1.0, shadowSample);
  }
  return 1.0;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0 {
  float4 alpha = textureBindless2D(input.mtl.texAlpha, pc.perFrame->sampler0, input.vtx.uv);
  if (input.mtl.texAlpha > 0 && alpha.r < 0.5)
    discard;
  
  float4 Ka = input.mtl.ambient * textureBindless2D(input.mtl.texAmbient, pc.perFrame->sampler0, input.vtx.uv);
  float4 Kd = input.mtl.diffuse * textureBindless2D(input.mtl.texDiffuse, pc.perFrame->sampler0, input.vtx.uv);
  
  if (Kd.a < 0.5)
    discard;
  
  float3 n = normalize(input.vtx.normal);
  float NdotL1 = clamp(dot(n, normalize(float3(-1, 1, +1))), 0.0, 1.0);
  float NdotL2 = clamp(dot(n, normalize(float3(-1, 1, -1))), 0.0, 1.0);
  float NdotL = 0.5 * (NdotL1 + NdotL2);
  
  // IBL diffuse
  const float4 f0 = float4(0.04, 0.04, 0.04, 0.04);
  float4 diffuse = textureBindlessCube(pc.perFrame->texSkyboxIrradiance, pc.perFrame->sampler0, n) * Kd * (float4(1.0, 1.0, 1.0, 1.0) - f0);
  
  return bDrawNormals ?
    float4(0.5 * (n + float3(1.0, 1.0, 1.0)), 1.0) :
    Ka + diffuse * shadow(input.vtx.shadowCoords);
}
)";

const char* codeWireframeSlang = R"(
struct PerFrame {
  float4x4 proj;
  float4x4 view;
};

struct PerObject {
  float4x4 model;
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
float4 vertexMain(float3 pos : POSITION) : SV_Position
{
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;
  
  return proj * view * model * float4(pos, 1.0);
}

[shader("fragment")]
float4 fragmentMain() : SV_Target0 {
  return float4(1.0);
}
)";

const char* codeShadowSlang = R"(
struct PerFrame {
  float4x4 proj;
  float4x4 view;
  float4x4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

struct PerObject {
  float4x4 model;
};

struct PushConstants {
  PerFrame* perFrame;
  PerObject* perObject;
};

[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
float4 vertexMain(float3 pos : POSITION) : SV_Position {
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;
  float4x4 model = pc.perObject->model;

  return proj * view * model * float4(pos, 1.0);
}

[shader("fragment")]
void fragmentMain() {
}
)";

const char* codeSkyboxSlang = R"(
struct PerFrame {
  float4x4 proj;
  float4x4 view;
  float4x4 light;
  uint texSkyboxRadiance;
  uint texSkyboxIrradiance;
  uint texShadow;
  uint sampler0;
  uint samplerShadow0;
};

struct PushConstants {
  PerFrame* perFrame;
};

struct VSOutput {
  float4 sv_Position : SV_Position;
  float3 dir         : TEXCOORD0;
};

[[vk::push_constant]] PushConstants pc;

static const float3 positions[8] = {
  float3(-1.0, -1.0,  1.0), float3( 1.0, -1.0,  1.0), float3( 1.0,  1.0,  1.0), float3(-1.0,  1.0,  1.0),
  float3(-1.0, -1.0, -1.0), float3( 1.0, -1.0, -1.0), float3( 1.0,  1.0, -1.0), float3(-1.0,  1.0, -1.0)
};

static const int indices[36] = {
  0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID) {
  float4x4 proj = pc.perFrame->proj;
  float4x4 view = pc.perFrame->view;

  view[3] = float4(0, 0, 0, 1); // discard translation

  float3 pos = positions[indices[vertexID]];

  VSOutput out;

  out.sv_Position = (proj * view * float4(pos, 1.0)).xyww;
  out.dir = pos; // skybox
  
  return out;
}

[shader("fragment")]
float4 fragmentMain(float3 dir : TEXCOORD0) : SV_Target0 {
  return textureBindlessCube(pc.perFrame->texSkyboxRadiance, pc.perFrame->sampler0, dir);
}
)";
