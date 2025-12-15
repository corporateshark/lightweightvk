/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <lvk/LVK.h>

namespace lvk {

class VgfModel final {
 public:
  struct TensorInfo {
    Format format = Format_Invalid;
    uint32_t rank = 0;
    int64_t dimensions[LVK_MAX_TENSOR_RANK] = {};
    bool matches(ldr::Span<const int64_t> dims) const;
    TensorDesc toTensorDesc(VkTensorUsageFlagsARM usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
                            VkTensorTilingARM tiling = VK_TENSOR_TILING_OPTIMAL_ARM) const;
  };

  VgfModel() = default;
  ~VgfModel();
  VgfModel(const VgfModel&) = delete;
  VgfModel& operator=(const VgfModel&) = delete;
  VgfModel(VgfModel&& other) noexcept;
  VgfModel& operator=(VgfModel&& other) noexcept;

  bool load(const void* data, size_t size);
  bool isValid() const;

  const char* getEntryPoint() const;
  const void* getSpirv() const;
  size_t getSpirvSize() const;
  uint32_t getNumInputs() const;
  uint32_t getNumOutputs() const;
  const TensorInfo& getInput(uint32_t index) const;
  const TensorInfo& getOutput(uint32_t index) const;
  int32_t findInput(ldr::Span<const int64_t> dims) const;
  int32_t findOutput(ldr::Span<const int64_t> dims) const;
  ldr::Span<const DataGraphConstant> getConstants() const;

  [[nodiscard]] Holder<DataGraphPipelineHandle> createDataGraphPipeline(IContext& ctx,
                                                                        const char* debugName = nullptr,
                                                                        Result* outResult = nullptr) const;
  [[nodiscard]] Holder<DataGraphPipelineHandle> createDataGraphPipeline(IContext& ctx,
                                                                        ldr::Span<const TensorHandle> inputs,
                                                                        ldr::Span<const TensorHandle> outputs,
                                                                        const char* debugName = nullptr,
                                                                        Result* outResult = nullptr) const;

 private:
  struct VgfModelImpl* pimpl_ = nullptr;
};

} // namespace lvk
