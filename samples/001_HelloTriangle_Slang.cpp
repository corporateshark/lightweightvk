/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <shared/UtilsFPS.h>

#include <lvk/LVK.h>
#if defined(ANDROID)
#include <android_native_app_glue.h>
#include <jni.h>
#include <time.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <slang.h>
#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <core/slang-basic.h>

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

std::vector<uint8_t> compileSlangToSPIRV(const char* code, lvk::ShaderStage stage) {
  using namespace Slang;

  ComPtr<slang::IGlobalSession> slangGlobalSession;
  if (SLANG_FAILED(slang::createGlobalSession(slangGlobalSession.writeRef()))) {
    return {};
  }

  const slang::TargetDesc targetDesc = {
      .format = SLANG_SPIRV,
      .profile = slangGlobalSession->findProfile("spirv_1_6"),
      .flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY,
  };

  const slang::SessionDesc sessionDesc = {
      .targets = &targetDesc,
      .targetCount = 1,
  };

  ComPtr<slang::ISession> session;
  if (SLANG_FAILED(slangGlobalSession->createSession(sessionDesc, session.writeRef()))) {
    return {};
  }

  slang::IModule* slangModule = nullptr;
  {
    ComPtr<slang::IBlob> diagnosticBlob;
    slangModule = session->loadModuleFromSourceString("", "", code, diagnosticBlob.writeRef());
    if (diagnosticBlob) {
      LLOGW("%s", (const char*)diagnosticBlob->getBufferPointer());
    }
    if (!slangModule) {
      return {};
    }
  }

  ComPtr<slang::IEntryPoint> entryPointVert;
  ComPtr<slang::IEntryPoint> entryPointFrag;
  slangModule->findEntryPointByName("vertexMain", entryPointVert.writeRef());
  slangModule->findEntryPointByName("fragmentMain", entryPointFrag.writeRef());

  Slang::List<slang::IComponentType*> componentTypes;
  componentTypes.add(slangModule);
  int entryPointCount = 0;
  int vertexEntryPointIndex = entryPointCount++;
  componentTypes.add(entryPointVert);
  int fragmentEntryPointIndex = entryPointCount++;
  componentTypes.add(entryPointFrag);

  ComPtr<slang::IComponentType> composedProgram;
  {
    ComPtr<slang::IBlob> diagnosticBlob;
    SlangResult result = session->createCompositeComponentType(
        componentTypes.getBuffer(), componentTypes.getCount(), composedProgram.writeRef(), diagnosticBlob.writeRef());
    if (diagnosticBlob) {
      LLOGW("%s", (const char*)diagnosticBlob->getBufferPointer());
    }
    if (SLANG_FAILED(result)) {
      return {};
    }
  }

  ComPtr<slang::IBlob> spirvCode;
  {
    ComPtr<slang::IBlob> diagnosticBlob;
    const int entryPoint = stage == lvk::Stage_Vert ? vertexEntryPointIndex : fragmentEntryPointIndex;
    SlangResult result = composedProgram->getEntryPointCode(entryPoint, 0, spirvCode.writeRef(), diagnosticBlob.writeRef());
    if (diagnosticBlob) {
      LLOGW("%s", (const char*)diagnosticBlob->getBufferPointer());
    }
    if (SLANG_FAILED(result)) {
      return {};
    }
  }

  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(spirvCode->getBufferPointer());

  return std::vector<uint8_t>(ptr, ptr + spirvCode->getBufferSize());
}

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

#if !defined(ANDROID)
int main(int argc, char* argv[]) {
#if defined(LVK_WITH_MINILOG)
  minilog::initialize(nullptr, {.threadNames = false});
#endif // LVK_WITH_MINILOG

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
#else
extern "C" {
void handle_cmd(android_app* app, int32_t cmd) {
  switch (cmd) {
  case APP_CMD_INIT_WINDOW:
    if (app->window != nullptr) {
      width_ = ANativeWindow_getWidth(app->window);
      height_ = ANativeWindow_getHeight(app->window);
      ctx_ = lvk::createVulkanContextWithSwapchain(app->window, width_, height_, {});
      init();
    }
    break;
  case APP_CMD_TERM_WINDOW:
    destroy();
    break;
  }
}

void resize_callback(ANativeActivity* activity, ANativeWindow* window) {
  int w = ANativeWindow_getWidth(window);
  int h = ANativeWindow_getHeight(window);
  if (width_ != w || height_ != h) {
    width_ = w;
    height_ = h;
    if (ctx_) {
      resize();
    }
  }
}

void android_main(android_app* app) {
  minilog::initialize(nullptr, {.threadNames = false});
  app->onAppCmd = handle_cmd;
  app->activity->callbacks->onNativeWindowResized = resize_callback;

  fps_.printFPS_ = false;

  timespec prevTime = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &prevTime);

  int events = 0;
  android_poll_source* source = nullptr;
  do {
    timespec newTime = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &newTime);
    if (fps_.tick(((double)newTime.tv_sec + 1.0e-9 * newTime.tv_nsec) - 
                  ((double)prevTime.tv_sec + 1.0e-9 * prevTime.tv_nsec))) {
      LLOGL("FPS: %.1f\n", fps_.getFPS());
    }
    prevTime = newTime;
    if (ctx_) {
      render();
    }
    if (ALooper_pollOnce(0, nullptr, &events, (void**)&source) >= 0) {
      if (source) {
        source->process(app, source);
      }
    }
  } while (!app->destroyRequested);
}
} // extern "C"
#endif
