/*
 * LightweightVK
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "VulkanApp.h"
#include <ldrutils/lutils/ScopeExit.h>

#include <filesystem>

const char* codeSlang = R"(
struct VertexOutput {
  float4 sv_Position : SV_Position;
  float2 uv          : TEXCOORD0;
};

static const float2 pos[4] = {
  float2(-1.0, -1.0),
  float2(-1.0, +1.0),
  float2(+1.0, -1.0),
  float2(+1.0, +1.0)
};

[[vk::constant_id(0)]] const uint textureId = 0;

[shader("vertex")]
VertexOutput vertexMain(uint vertexID : SV_VertexID) {
  VertexOutput out;
  out.sv_Position = float4(pos[vertexID], 0.0, 1.0);
  out.uv = pos[vertexID] * 0.5 + 0.5;
  out.uv.y = 1.0 - out.uv.y;
  return out;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target0 {
  return kSamplersYUV[textureId].Sample(input.uv);
}
)";

const char* codeVS = R"(
#version 460
layout (location=0) out vec2 uv;
const vec2 pos[4] = vec2[4](
  vec2(-1.0, -1.0),
  vec2(-1.0, +1.0),
  vec2(+1.0, -1.0),
  vec2(+1.0, +1.0)
);
void main() {
  gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
  uv = pos[gl_VertexIndex] * 0.5 + 0.5;
    uv.y = 1.0-uv.y;
}
)";

const char* codeFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout (constant_id = 0) const uint textureId = 0;

void main() {
  out_FragColor = texture(kSamplersYUV[textureId], uv);
}
)";

size_t currentDemo_ = 0;

// demonstrate different YUV formats
struct YUVFormatDemo {
  const char* name;
  lvk::Format format;
  lvk::Holder<lvk::TextureHandle> texture;
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState;
};

struct Resources {
  lvk::Holder<lvk::ShaderModuleHandle> vert;
  lvk::Holder<lvk::ShaderModuleHandle> frag;
  std::vector<YUVFormatDemo> demos;
};

Resources res_;

void createDemo(lvk::IContext* ctx, const char* contentFolder, const char* name, lvk::Format format, const char* fileName) {
  using namespace std::filesystem;
  path dir(contentFolder);
  int32_t texWidth = 1920;
  int32_t texHeight = 1080;
  FILE* file = fopen((dir / "src" / path(fileName)).string().c_str(), "rb");
  SCOPE_EXIT {
    if (file) {
      fclose(file);
    }
  };
  fseek(file, 0, SEEK_END);
  const uint32_t length = ftell(file);
  fseek(file, 0, SEEK_SET);

  LVK_ASSERT_MSG(file && length, "Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
  if (!file || !length) {
    printf("Cannot load textures. Run `deploy_content.py`/`deploy_content_android.py` before running this app.");
    std::terminate();
  }

  LVK_ASSERT(length == texWidth * texHeight * 3 / 2);

  std::vector<uint8_t> pixels(length);
  fread(pixels.data(), 1, length, file);

  lvk::Holder<lvk::TextureHandle> texture = ctx->createTexture({
      .type = lvk::TextureType_2D,
      .format = format,
      .dimensions = {(uint32_t)texWidth, (uint32_t)texHeight},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = pixels.data(),
      .debugName = name,
  });

  const uint32_t textureId = texture.index();

  res_.demos.push_back({
      .name = name,
      .format = format,
      .texture = std::move(texture),
      .renderPipelineState = ctx->createRenderPipeline({
          .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
          .smVert = res_.vert,
          .smFrag = res_.frag,
          .specInfo = {.entries = {{.constantId = 0, .size = sizeof(uint32_t)}}, .data = &textureId, .dataSize = sizeof(textureId)},
          .color = {{.format = ctx->getSwapchainFormat()}},
          .debugName = name,
      }),
  });
}

VULKAN_APP_MAIN {
  const VulkanAppConfig cfg{
      .width = 0,
      .height = 0,
  };
  VULKAN_APP_DECLARE(app, cfg);

  lvk::IContext* ctx = app.ctx_.get();

#if defined(LVK_DEMO_WITH_SLANG)
  res_.vert = ctx->createShaderModule({codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)"});
  res_.frag = ctx->createShaderModule({codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)"});
#else
  res_.vert = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader Module: main (vert)"});
  res_.frag = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader Module: main (frag)"});
#endif // defined(LVK_DEMO_WITH_SLANG)

  createDemo(ctx, app.folderContentRoot_.c_str(), "YUV NV12", lvk::Format_YUV_NV12, "igl-samples/output_frame_900.nv12.yuv");
  createDemo(ctx, app.folderContentRoot_.c_str(), "YUV 420p", lvk::Format_YUV_420p, "igl-samples/output_frame_900.420p.yuv");

#if !defined(ANDROID)
  app.addMouseButtonCallback([](auto* window, int button, int action, int mods) {
    if (action == GLFW_PRESS && !res_.demos.empty()) {
      currentDemo_ = (currentDemo_ + 1) % res_.demos.size();
    }
  });
  app.addKeyCallback([](GLFWwindow* window, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_T && action == GLFW_PRESS) {
      currentDemo_ = 0;
      if (!res_.demos.empty())
        res_.demos.pop_back();
    } else if (action == GLFW_PRESS && !res_.demos.empty()) {
      currentDemo_ = (currentDemo_ + 1) % res_.demos.size();
    }
  });
#endif // !ANDROID

  app.run([&](uint32_t width, uint32_t height, float aspectRatio, float deltaSeconds) {
    const lvk::Framebuffer framebuffer = {
        .color = {{.texture = ctx->getCurrentSwapchainTexture()}},
    };

    lvk::ICommandBuffer& buffer = ctx->acquireCommandBuffer();

    buffer.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_DontCare}}}, framebuffer);

    if (!res_.demos.empty()) {
      const YUVFormatDemo& demo = res_.demos[currentDemo_];
      buffer.cmdBindRenderPipeline(demo.renderPipelineState);
      buffer.cmdDraw(4);
      {
        app.imgui_->beginFrame(framebuffer);
        const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
                                       ImGuiWindowFlags_NoMove;
        ImGui::SetNextWindowPos({15.0f, 15.0f});
        ImGui::SetNextWindowBgAlpha(0.30f);
        ImGui::Begin("##FormatYUV", nullptr, flags);
        ImGui::Text("%s", demo.name);
        ImGui::Text("Press any key to change");
        ImGui::End();
        app.drawFPS();
        app.imgui_->endFrame(buffer);
      }
    }

    buffer.cmdEndRendering();

    ctx->submit(buffer, ctx->getCurrentSwapchainTexture());
  });

  // destroy all the Vulkan stuff before closing the window
  res_ = {};

  VULKAN_APP_EXIT();
}
