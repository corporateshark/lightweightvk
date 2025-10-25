/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include "VulkanApp.h"

#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

const char* codeSlang = R"(
static const float2 pos[3] = float2[3](
  float2(-0.6, -0.4),
  float2( 0.6, -0.4),
  float2( 0.0,  0.6)
);
static const float3 col[3] = float3[3](
  float3(1.0, 0.0, 0.0),
  float3(0.0, 1.0, 0.0),
  float3(0.0, 0.0, 1.0)
);

struct OutVertex {
  float3 color;
};

struct Fragment {
  float4 color;
};

struct VertexStageOutput {
  OutVertex    vertex       : OutVertex;
  float4       sv_position  : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(uint vertexID : SV_VertexID) {
  VertexStageOutput output;

  output.vertex.color = col[vertexID];
  output.sv_position = float4(pos[vertexID], 0.0, 1.0);

  return output;
}

[shader("fragment")]
float4 fragmentMain(OutVertex vertex : OutVertex) : SV_Target {
  return float4(vertex.color, 1.0);
}
)";

int width_ = 800;
int height_ = 600;
FramesPerSecondCounter fps_;

lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
std::unique_ptr<lvk::IContext> ctx_;
lvk::Holder<lvk::ShaderModuleHandle> vert_;
lvk::Holder<lvk::ShaderModuleHandle> frag_;

lvk::Holder<lvk::ShaderModuleHandle> slangCreateShaderModule(const char* code,
                                                             lvk::ShaderStage stage,
                                                             const char* debugName,
                                                             const bool dumpSPIRV = false) {
  const std::vector<uint8_t> spirv = compileSlangToSPIRV(code, stage);

  if (dumpSPIRV) {
    std::ofstream fout("dump." + std::to_string(stage), std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(spirv.data()), spirv.size());
  }

  return ctx_->createShaderModule({spirv.data(), spirv.size(), stage, debugName});
}

void init() {
  vert_ = slangCreateShaderModule(codeSlang, lvk::Stage_Vert, "Shader Module: main (vert)");
  frag_ = slangCreateShaderModule(codeSlang, lvk::Stage_Frag, "Shader Module: main (frag)");

  renderPipelineState_Triangle_ = ctx_->createRenderPipeline(
      {
          .smVert = vert_,
          .smFrag = frag_,
          .color = {{.format = ctx_->getSwapchainFormat()}},
      },
      nullptr);

  LVK_ASSERT(renderPipelineState_Triangle_.valid());
}

void destroy() {
  vert_ = nullptr;
  frag_ = nullptr;
  renderPipelineState_Triangle_ = nullptr;
  ctx_ = nullptr;
}

void resize() {
  if (!width_ || !height_) {
    return;
  }
  ctx_->recreateSwapchain(width_, height_);
}

void render() {
  if (!width_ || !height_) {
    return;
  }

  lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

  // This will clear the framebuffer
  buffer.cmdBeginRendering(
      {.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}}}},
      {.color = {{.texture = ctx_->getCurrentSwapchainTexture()}}});
  buffer.cmdBindRenderPipeline(renderPipelineState_Triangle_);
  buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
  buffer.cmdDraw(3);
  buffer.cmdPopDebugGroupLabel();
  buffer.cmdEndRendering();
  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan Hello Triangle", width_, height_, true);

  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {});
  if (!ctx_) {
    return 1;
  }
  init();

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int width, int height) {
    width_ = width;
    height_ = height;
    resize();
  });

  double prevTime = glfwGetTime();

  // main loop
  while (!glfwWindowShouldClose(window)) {
    const double newTime = glfwGetTime();
    fps_.tick(newTime - prevTime);
    prevTime = newTime;
    render();
    glfwPollEvents();
  }

  // destroy all the Vulkan stuff before closing the window
  destroy();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
