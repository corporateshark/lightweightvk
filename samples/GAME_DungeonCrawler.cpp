/*
 * LightweightVK
 *
 * Top-Down Dungeon Crawler (minimal slice)
 * - Tilemap: walls + floor
 * - Player moves with WASD/Arrows (grid step)
 * - Orthographic rendering with per-tile draw and push constants
 */

#include "VulkanApp.h"

// Simple quad via gl_VertexIndex
static const char* kVS = R"(
layout (location=0) out vec4 vColor;

layout(push_constant) uniform PC {
  mat4 mvp;    // projection * model
  vec4 color;  // RGBA
} pc;

void main(){
  // 2 triangles for a unit quad centered at origin
  const vec2 POS[6] = vec2[6](
    vec2(-0.5, -0.5), vec2( 0.5, -0.5), vec2(-0.5,  0.5),
    vec2( 0.5, -0.5), vec2( 0.5,  0.5), vec2(-0.5,  0.5)
  );
  vec2 p = POS[gl_VertexIndex];
  gl_Position = pc.mvp * vec4(p, 0.0, 1.0);
  vColor = pc.color;
}
)";

static const char* kFS = R"(
layout (location=0) in vec4 vColor;
layout (location=0) out vec4 out_FragColor;
void main(){ out_FragColor = vColor; }
)";

enum Tile : uint8_t { Tile_Floor = 0, Tile_Wall = 1 };

struct PushConstants {
  mat4 mvp;
  vec4 color;
};

struct IVec2 { int x, y; };

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 1280,
      .height = 720,
      .resizable = true,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

  // Map settings
  const int W = 32;
  const int H = 18;
  std::vector<uint8_t> map(W * H, Tile_Floor);

  // Build simple dungeon: border walls + some internal rooms/corridors
  auto at = [&](int x, int y) -> uint8_t& { return map[y * W + x]; };
  for (int x = 0; x < W; ++x) { at(x, 0) = at(x, H - 1) = Tile_Wall; }
  for (int y = 0; y < H; ++y) { at(0, y) = at(W - 1, y) = Tile_Wall; }
  for (int y = 3; y < H - 3; ++y) at(W / 2, y) = Tile_Wall;
  for (int x = 4; x < W - 4; ++x) if (x % 6 == 0) at(x, H / 2) = Tile_Wall;
  // A small room
  for (int y = 5; y <= 8; ++y) for (int x = 5; x <= 10; ++x) at(x, y) = (y == 5 || y == 8 || x == 5 || x == 10) ? Tile_Wall : Tile_Floor;
  at(7, 5) = Tile_Floor; // doorway

  // Player state
  IVec2 player{2, 2};
  auto canMove = [&](int nx, int ny) { return nx >= 0 && ny >= 0 && nx < W && ny < H && at(nx, ny) != Tile_Wall; };

#if !defined(ANDROID)
  // Key polling state (edge-triggered movement)
  bool prevLeft=false, prevRight=false, prevUp=false, prevDown=false, prevA=false, prevD=false, prevW=false, prevS=false;
#endif

  // Pipeline
  lvk::Holder<lvk::ShaderModuleHandle> smVS = ctx->createShaderModule({kVS, lvk::Stage_Vert, "VS: DungeonQuad"});
  lvk::Holder<lvk::ShaderModuleHandle> smFS = ctx->createShaderModule({kFS, lvk::Stage_Frag, "FS: Color"});
  lvk::Holder<lvk::RenderPipelineHandle> pipe = ctx->createRenderPipeline({
      .smVert = smVS,
      .smFrag = smFS,
      .color = {{.format = ctx->getSwapchainFormat()}},
      .debugName = "Pipeline: Dungeon2D"
  });

  // Colors
  const vec4 colFloor = vec4(0.12f, 0.12f, 0.14f, 1.0f);
  const vec4 colWall  = vec4(0.30f, 0.30f, 0.34f, 1.0f);
  const vec4 colPlayer= vec4(0.90f, 0.80f, 0.20f, 1.0f);

  app.run([&](uint32_t width, uint32_t height, float aspect, float /*deltaSeconds*/) {
    // Ortho: map coordinates in [0..W]x[0..H]; keep aspect by scaling view
    const float worldW = (float)W;
    const float worldH = (float)H;
    // Fit world into viewport preserving aspect
    float targetAspect = worldW / worldH;
    float left, right, bottom, top;
    if (aspect > targetAspect) {
      float viewW = worldH * aspect;
      float pad = (viewW - worldW) * 0.5f;
      left = -pad; right = worldW + pad; bottom = 0.0f; top = worldH;
    } else {
      float viewH = worldW / aspect;
      float pad = (viewH - worldH) * 0.5f;
      left = 0.0f; right = worldW; bottom = -pad; top = worldH + pad;
    }
    const mat4 proj = glm::orthoLH_ZO(left, right, bottom, top, -1.0f, 1.0f);

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();
    buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearColor = {0.05f, 0.06f, 0.09f, 1.0f}}}},
                             {.color = {{.texture = ctx->getCurrentSwapchainTexture()}}});
    buffer.cmdBindRenderPipeline(pipe);
    buffer.cmdBindViewport({0.0f, 0.0f, (float)width, (float)height, 0.0f, +1.0f});
    buffer.cmdBindScissorRect({0, 0, width, height});

    auto drawQuad = [&](float cx, float cy, vec4 color, float sx = 1.0f, float sy = 1.0f) {
      mat4 model = glm::translate(mat4(1.0f), vec3(cx, cy, 0.0f));
      model = glm::scale(model, vec3(sx, sy, 1.0f));
      PushConstants pc{.mvp = proj * model, .color = color};
      buffer.cmdPushConstants(pc);
      buffer.cmdDraw(6);
    };

    // Handle input (desktop)
#if !defined(ANDROID)
    auto keyPressed = [&](int k){ return app.window_ && glfwGetKey(app.window_, k) == GLFW_PRESS; };
    bool leftKey  = keyPressed(GLFW_KEY_LEFT);  bool a = keyPressed(GLFW_KEY_A);
    bool rightKey = keyPressed(GLFW_KEY_RIGHT); bool d = keyPressed(GLFW_KEY_D);
    bool upKey    = keyPressed(GLFW_KEY_UP);    bool w = keyPressed(GLFW_KEY_W);
    bool downKey  = keyPressed(GLFW_KEY_DOWN);  bool s = keyPressed(GLFW_KEY_S);
    auto onEdge = [&](bool now, bool& prev){ bool e = now && !prev; prev = now; return e; };
    IVec2 p = player;
    if (onEdge(leftKey, prevLeft)  || onEdge(a, prevA))   p.x -= 1;
    if (onEdge(rightKey,prevRight) || onEdge(d, prevD))   p.x += 1;
    if (onEdge(upKey,   prevUp)    || onEdge(w, prevW))   p.y += 1;
    if (onEdge(downKey, prevDown)  || onEdge(s, prevS))   p.y -= 1;
    if (canMove(p.x, p.y)) player = p;
#endif

    // Draw tiles
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const uint8_t t = at(x, y);
        const vec4 c = (t == Tile_Wall) ? colWall : colFloor;
        drawQuad(x + 0.5f, y + 0.5f, c);
      }
    }

    // Draw player (slightly smaller quad)
    drawQuad((float)player.x + 0.5f, (float)player.y + 0.5f, colPlayer, 0.7f, 0.7f);

    // UI
    app.imgui_->beginFrame({.color = {{.texture = ctx->getCurrentSwapchainTexture()}}});
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_Once);
    ImGui::Begin("Dungeon Crawler", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Grid: %dx%d", W, H);
    ImGui::Text("Controls: WASD/Arrows to move");
    ImGui::Text("Player: (%d, %d)", player.x, player.y);
    app.drawFPS();
    ImGui::End();
    app.imgui_->endFrame(buffer);

    buffer.cmdEndRendering();
    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  VULKAN_APP_EXIT();
}
