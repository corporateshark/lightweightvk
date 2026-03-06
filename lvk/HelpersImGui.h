/*
* LightweightVK
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#pragma once

#if !defined(IMGUI_DEFINE_MATH_OPERATORS)
#define IMGUI_DEFINE_MATH_OPERATORS
#endif // !defined(IMGUI_DEFINE_MATH_OPERATORS)

#include <lvk/LVK.h>
#include <imgui/imgui.h>

namespace lvk {

class ImGuiRenderer {
 public:
  explicit ImGuiRenderer(lvk::IContext& device, lvk::LVKwindow* window, const char* defaultFontTTF = nullptr, float fontSizePixels = 24.0f);
  ~ImGuiRenderer();
  
  void updateFont(const char* defaultFontTTF, float fontSizePixels);

  void beginFrame(const lvk::Framebuffer& desc);
  void endFrame(lvk::ICommandBuffer& cmdBuffer);
  
 private:
  lvk::Holder<lvk::RenderPipelineHandle> createNewPipelineState(const lvk::Framebuffer& desc);

 private:
  lvk::IContext& ctx_;
  lvk::Holder<lvk::ShaderModuleHandle> vert_;
  lvk::Holder<lvk::ShaderModuleHandle> frag_;
  lvk::Holder<lvk::RenderPipelineHandle> pipeline_;
  lvk::Format pipelineColorFormat_ = lvk::Format_Invalid;
  lvk::Holder<lvk::SamplerHandle> samplerClamp_;
  struct ImGuiRendererImpl* pimpl_ = nullptr;

  uint32_t frameIndex_ = 0;

  struct DrawableData {
    lvk::Holder<BufferHandle> vb_;
    lvk::Holder<BufferHandle> ib_;
    uint32_t numAllocatedIndices_ = 0;
    uint32_t numAllocatedVerteices_ = 0;
  };

  DrawableData drawables_[3] = {};
};

} // namespace lvk
