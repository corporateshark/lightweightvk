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

// we are going to use raw Vulkan here to initialize VK_EXT_mesh_shader
#include <lvk/vulkan/VulkanUtils.h>

const char* codeTask = R"(
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
  EmitMeshTasksEXT(1, 1, 1);
}
)";

const char* codeMesh = R"(
layout(triangles, max_vertices = 3, max_primitives = 1) out;

layout (location=0) out vec3 color[3];

const vec2 pos[3] = vec2[3](
  vec2(-0.6, -0.4),
  vec2( 0.6, -0.4),
  vec2( 0.0,  0.6)
);
const vec3 col[3] = vec3[3](
  vec3(1.0, 0.0, 0.0),
  vec3(0.0, 1.0, 0.0),
  vec3(0.0, 0.0, 1.0)
);
void main() {
  SetMeshOutputsEXT(3, 1);

  for (uint i = 0; i != 3; i++) {
    gl_MeshVerticesEXT[i].gl_Position = vec4(pos[i], 0.0, 1.0);
    color[i] = col[i];
  }

  gl_MeshPrimitivesEXT[0].gl_CullPrimitiveEXT = false;
  gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
)";

const char* codeFrag = R"(
#version 460
layout (location=0) in vec3 color;
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(color, 1.0);
};
)";

int width_ = 800;
int height_ = 600;
FramesPerSecondCounter fps_;

std::unique_ptr<lvk::IContext> ctx_;

struct {
  lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Triangle_;
  lvk::Holder<lvk::ShaderModuleHandle> task_;
  lvk::Holder<lvk::ShaderModuleHandle> mesh_;
  lvk::Holder<lvk::ShaderModuleHandle> frag_;
} res;

void init() {
  res.task_ = ctx_->createShaderModule({codeTask, lvk::Stage_Task, "Shader Module: main (task)"});
  res.mesh_ = ctx_->createShaderModule({codeMesh, lvk::Stage_Mesh, "Shader Module: main (mesh)"});
  res.frag_ = ctx_->createShaderModule({codeFrag, lvk::Stage_Frag, "Shader Module: main (frag)"});

  res.renderPipelineState_Triangle_ = ctx_->createRenderPipeline(
      {
          .smTask = res.task_,
          .smMesh = res.mesh_,
          .smFrag = res.frag_,
          .color = {{.format = ctx_->getSwapchainFormat()}},
      },
      nullptr);

  LVK_ASSERT(res.renderPipelineState_Triangle_.valid());
}

void destroy() {
  res = {};
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
  buffer.cmdBindRenderPipeline(res.renderPipelineState_Triangle_);
  buffer.cmdPushDebugGroupLabel("Render Triangle", 0xff0000ff);
  buffer.cmdDrawMeshTasks({1, 1, 1});

  buffer.cmdPopDebugGroupLabel();
  buffer.cmdEndRendering();
  ctx_->submit(buffer, ctx_->getCurrentSwapchainTexture());
}

int main(int argc, char* argv[]) {
  minilog::initialize(nullptr, {.threadNames = false});

  GLFWwindow* window = lvk::initWindow("Vulkan Hello Mesh Shaders", width_, height_, true);

  ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_, {});
  if (!ctx_) {
    return 255;
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
