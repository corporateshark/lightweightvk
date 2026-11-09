/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "HelpersVgf.h"

#include <cstring>
#include <string>
#include <vector>

#include <lvk/vulkan/VulkanUtils.h>
#include <vgf/decoder.hpp>

namespace lvk {

struct VgfModelImpl {
  std::vector<uint8_t> file;
  std::vector<uint32_t> spirv;
  std::string entryPoint;
  std::vector<VgfModel::TensorInfo> inputs;
  std::vector<VgfModel::TensorInfo> outputs;
  std::vector<DataGraphConstant> constants;
};

namespace {

bool isTensorFormatSupported(VkFormat format) {
  switch (format) {
  case VK_FORMAT_R8_UNORM:
  case VK_FORMAT_R8_UINT:
  case VK_FORMAT_R8_SINT:
  case VK_FORMAT_R16_UNORM:
  case VK_FORMAT_R16_SFLOAT:
  case VK_FORMAT_R16_UINT:
  case VK_FORMAT_R16_SINT:
  case VK_FORMAT_R32_UINT:
  case VK_FORMAT_R32_SINT:
  case VK_FORMAT_R32_SFLOAT:
    return true;
  default:
    return false;
  }
}

bool describeResource(const mlsdk::vgflib::ModelResourceTableDecoder& resources, uint32_t mrtIndex, VgfModel::TensorInfo& info) {
  if (mrtIndex >= resources.size()) {
    LLOGW("VGF: resource %u is outside the model resource table (%zu)\n", mrtIndex, resources.size());
    return false;
  }
  const mlsdk::vgflib::DataView<int64_t> shape = resources.getTensorShape(mrtIndex);
  if (!shape.size() || shape.size() > LVK_MAX_TENSOR_RANK) {
    LLOGW("VGF: resource %u has rank %zu, 1 to %u is supported\n", mrtIndex, shape.size(), (uint32_t)LVK_MAX_TENSOR_RANK);
    return false;
  }
  for (size_t i = 0; i != shape.size(); i++) {
    if (shape[i] <= 0) {
      LLOGW("VGF: resource %u has a non-positive dimension %lld\n", mrtIndex, (long long)shape[i]);
      return false;
    }
  }
  const VkFormat format = (VkFormat)resources.getVkFormat(mrtIndex);
  if (!isTensorFormatSupported(format)) {
    LLOGW("VGF: resource %u has an unsupported format %u\n", mrtIndex, (uint32_t)format);
    return false;
  }
  info = {.format = lvk::vkFormatToFormat(format), .rank = (uint32_t)shape.size()};
  for (uint32_t i = 0; i != info.rank; i++) {
    info.dimensions[i] = shape[i];
  }
  return true;
}

std::string shapeToString(const VgfModel::TensorInfo& info) {
  std::string shape;
  for (uint32_t i = 0; i != info.rank; i++) {
    shape += (shape.empty() ? "" : ", ") + std::to_string(info.dimensions[i]);
  }
  return shape;
}

int32_t findTensor(const std::vector<VgfModel::TensorInfo>& list, ldr::Span<const int64_t> dims) {
  for (size_t i = 0; i != list.size(); i++) {
    if (list[i].matches(dims)) {
      return (int32_t)i;
    }
  }
  return -1;
}

} // namespace

bool VgfModel::TensorInfo::matches(ldr::Span<const int64_t> dims) const {
  if (dims.size() != rank) {
    return false;
  }
  for (uint32_t i = 0; i != rank; i++) {
    if (dims[i] >= 0 && dims[i] != dimensions[i]) {
      return false;
    }
  }
  return true;
}

TensorDesc VgfModel::TensorInfo::toTensorDesc(TensorUsageFlags usage, TensorTiling tiling) const {
  TensorDesc desc = {
      .format = format,
      .rank = rank,
      .tiling = tiling,
      .usage = usage,
  };
  for (uint32_t i = 0; i != rank; i++) {
    desc.dimensions[i] = dimensions[i];
  }
  return desc;
}

VgfModel::~VgfModel() {
  delete pimpl_;
}

VgfModel::VgfModel(VgfModel&& other) noexcept : pimpl_(other.pimpl_) {
  other.pimpl_ = nullptr;
}

VgfModel& VgfModel::operator=(VgfModel&& other) noexcept {
  std::swap(pimpl_, other.pimpl_);
  return *this;
}

bool VgfModel::load(const void* data, size_t size) {
  using namespace mlsdk::vgflib;

  delete pimpl_;
  pimpl_ = nullptr;

  if (!data || size < HeaderSize()) {
    LLOGW("VGF: invalid data\n");
    return false;
  }

  VgfModelImpl* impl = new VgfModelImpl();
  impl->file.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
  const uint8_t* bytes = impl->file.data();

  auto fail = [impl](const char* message) -> bool {
    LLOGW("VGF: %s\n", message);
    delete impl;
    return false;
  };

  const std::unique_ptr<HeaderDecoder> header = CreateHeaderDecoder(bytes, HeaderSize(), size);
  if (!header || !header->IsValid() || !header->CheckVersion()) {
    return fail("not a valid VGF file");
  }
  LLOGL("VGF: version %u.%u.%u\n", header->GetMajor(), header->GetMinor(), header->GetPatch());

  const std::unique_ptr<ModuleTableDecoder> modules =
      CreateModuleTableDecoder(bytes + header->GetModuleTableOffset(), header->GetModuleTableSize());
  const std::unique_ptr<ModelSequenceTableDecoder> sequence =
      CreateModelSequenceTableDecoder(bytes + header->GetModelSequenceTableOffset(), header->GetModelSequenceTableSize());
  const std::unique_ptr<ModelResourceTableDecoder> resources =
      CreateModelResourceTableDecoder(bytes + header->GetModelResourceTableOffset(), header->GetModelResourceTableSize());
  const std::unique_ptr<ConstantDecoder> constants =
      CreateConstantDecoder(bytes + header->GetConstantsOffset(), header->GetConstantsSize());
  if (!modules || !sequence || !resources || !constants) {
    return fail("cannot decode the file sections");
  }

  if (sequence->modelSequenceTableSize() != 1 || sequence->getSegmentType(0) != ModuleType::GRAPH) {
    return fail("expected exactly one graph segment");
  }
  const uint32_t segment = 0;
  const uint32_t moduleIdx = sequence->getSegmentModuleIndex(segment);
  if (moduleIdx >= modules->size()) {
    return fail("the graph segment references a module outside the module table");
  }
  if (!modules->isSPIRV(moduleIdx) || !modules->hasSPIRVCode(moduleIdx)) {
    return fail("the graph module has no SPIR-V code");
  }
  const DataView<uint32_t> code = modules->getSPIRVModuleCode(moduleIdx);
  impl->spirv.assign(code.begin(), code.end());
  impl->entryPoint = std::string(modules->getModuleEntryPoint(moduleIdx));
  if (impl->entryPoint.empty()) {
    impl->entryPoint = "main";
  }
  LLOGL("VGF: segment `%.*s`, module `%.*s`, entry point `%s`, %zu SPIR-V words\n",
        (int)sequence->getSegmentName(segment).size(),
        sequence->getSegmentName(segment).data(),
        (int)modules->getModuleName(moduleIdx).size(),
        modules->getModuleName(moduleIdx).data(),
        impl->entryPoint.c_str(),
        impl->spirv.size());

  auto collectTensors = [&](BindingSlotArrayHandle slots, std::vector<TensorInfo>& list, const char* kind) -> bool {
    for (uint32_t i = 0; i != sequence->getBindingsSize(slots); i++) {
      TensorInfo info = {};
      if (!describeResource(*resources, sequence->getBindingSlotMrtIndex(slots, i), info)) {
        return false;
      }
      LLOGL("VGF: %s %u: format %u, shape [%s]\n", kind, i, (uint32_t)info.format, shapeToString(info).c_str());
      list.push_back(info);
    }
    return true;
  };
  if (!collectTensors(sequence->getSegmentInputBindingSlotsHandle(segment), impl->inputs, "input") ||
      !collectTensors(sequence->getSegmentOutputBindingSlotsHandle(segment), impl->outputs, "output")) {
    return fail("cannot describe the graph inputs and outputs");
  }

  size_t constantsBytes = 0;
  for (const uint32_t constIdx : sequence->getSegmentConstantIndexes(segment)) {
    if (constIdx >= constants->size()) {
      LLOGW("VGF: constant %u is outside the constant section (%zu)\n", constIdx, constants->size());
      return fail("a constant is outside the constant section");
    }
    if (constants->isSparseConstant(constIdx)) {
      return fail("sparse constants are not supported");
    }
    TensorInfo info = {};
    if (!describeResource(*resources, constants->getConstantMrtIndex(constIdx), info)) {
      return fail("cannot describe a constant");
    }
    const DataView<uint8_t> constantData = constants->getConstant(constIdx);
    if (constantData.size() < getTensorDataSize(info.toTensorDesc())) {
      LLOGW("VGF: constant %u holds %zu bytes, its shape needs %llu\n",
            constIdx,
            constantData.size(),
            (unsigned long long)getTensorDataSize(info.toTensorDesc()));
      return fail("a constant is shorter than its shape");
    }
    DataGraphConstant c = {
        .id = constIdx,
        .format = info.format,
        .rank = info.rank,
        .data = constantData.data(),
    };
    memcpy(c.dimensions, info.dimensions, sizeof(c.dimensions));
    impl->constants.push_back(c);
    constantsBytes += constantData.size();
  }
  LLOGL("VGF: %zu constants (%zu bytes)\n", impl->constants.size(), constantsBytes);

  pimpl_ = impl;
  return true;
}

bool VgfModel::isValid() const {
  return pimpl_;
}

const char* VgfModel::getEntryPoint() const {
  return pimpl_ ? pimpl_->entryPoint.c_str() : "";
}

const void* VgfModel::getSpirv() const {
  return pimpl_ ? pimpl_->spirv.data() : nullptr;
}

size_t VgfModel::getSpirvSize() const {
  return pimpl_ ? pimpl_->spirv.size() * sizeof(uint32_t) : 0;
}

uint32_t VgfModel::getNumInputs() const {
  return pimpl_ ? (uint32_t)pimpl_->inputs.size() : 0;
}

uint32_t VgfModel::getNumOutputs() const {
  return pimpl_ ? (uint32_t)pimpl_->outputs.size() : 0;
}

const VgfModel::TensorInfo& VgfModel::getInput(uint32_t index) const {
  static const TensorInfo kInvalid = {};
  return pimpl_ && index < pimpl_->inputs.size() ? pimpl_->inputs[index] : kInvalid;
}

const VgfModel::TensorInfo& VgfModel::getOutput(uint32_t index) const {
  static const TensorInfo kInvalid = {};
  return pimpl_ && index < pimpl_->outputs.size() ? pimpl_->outputs[index] : kInvalid;
}

int32_t VgfModel::findInput(ldr::Span<const int64_t> dims) const {
  return pimpl_ ? findTensor(pimpl_->inputs, dims) : -1;
}

int32_t VgfModel::findOutput(ldr::Span<const int64_t> dims) const {
  return pimpl_ ? findTensor(pimpl_->outputs, dims) : -1;
}

ldr::Span<const DataGraphConstant> VgfModel::getConstants() const {
  return pimpl_ ? ldr::Span<const DataGraphConstant>(pimpl_->constants.data(), pimpl_->constants.size())
                : ldr::Span<const DataGraphConstant>();
}

namespace {

Holder<DataGraphPipelineHandle> createPipeline(IContext& ctx,
                                               const VgfModelImpl& model,
                                               const std::vector<TensorDesc>& inputs,
                                               const std::vector<TensorDesc>& outputs,
                                               const char* debugName,
                                               Result* outResult) {
  const char* name = debugName ? debugName : "VGF data graph";
  const Holder<ShaderModuleHandle> sm =
      ctx.createShaderModule({model.spirv.data(), model.spirv.size() * sizeof(uint32_t), Stage_DataGraph, name});
  if (sm.empty()) {
    Result::setResult(outResult, Result::Code::RuntimeError, "Cannot create the graph module");
    return {};
  }
  return ctx.createDataGraphPipeline(
      {
          .smGraph = sm,
          .entryPoint = model.entryPoint.c_str(),
          .inputs = {inputs.data(), inputs.size()},
          .outputs = {outputs.data(), outputs.size()},
          .constants = {model.constants.data(), model.constants.size()},
          .debugName = name,
      },
      outResult);
}

bool describeBoundTensors(IContext& ctx,
                          const std::vector<VgfModel::TensorInfo>& infos,
                          ldr::Span<const TensorHandle> handles,
                          const char* what,
                          std::vector<TensorDesc>& outDescs,
                          Result* outResult) {
  if (handles.size() != infos.size()) {
    Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "The number of tensors does not match the model");
    LLOGW("VGF: the model has %u %ss, %u given\n", (uint32_t)infos.size(), what, (uint32_t)handles.size());
    return false;
  }
  outDescs.reserve(infos.size());
  for (size_t i = 0; i != infos.size(); i++) {
    TensorDesc desc = ctx.getTensorDesc(handles[i]);
    if (desc.format != infos[i].format || !infos[i].matches({desc.dimensions, desc.rank})) {
      Result::setResult(outResult, Result::Code::ArgumentOutOfRange, "A tensor does not match the model");
      LLOGW("VGF: %s %u does not match the model\n", what, (uint32_t)i);
      return false;
    }
    desc.data = nullptr;
    outDescs.push_back(desc);
  }
  return true;
}

} // namespace

Holder<DataGraphPipelineHandle> VgfModel::createDataGraphPipeline(IContext& ctx, const char* debugName, Result* outResult) const {
  if (!pimpl_) {
    Result::setResult(outResult, Result::Code::RuntimeError, "The VGF model is not loaded");
    return {};
  }
  std::vector<TensorDesc> inputs;
  std::vector<TensorDesc> outputs;
  inputs.reserve(pimpl_->inputs.size());
  outputs.reserve(pimpl_->outputs.size());
  for (const TensorInfo& info : pimpl_->inputs) {
    inputs.push_back(info.toTensorDesc());
  }
  for (const TensorInfo& info : pimpl_->outputs) {
    outputs.push_back(info.toTensorDesc());
  }
  return createPipeline(ctx, *pimpl_, inputs, outputs, debugName, outResult);
}

Holder<DataGraphPipelineHandle> VgfModel::createDataGraphPipeline(IContext& ctx,
                                                                  ldr::Span<const TensorHandle> inputs,
                                                                  ldr::Span<const TensorHandle> outputs,
                                                                  const char* debugName,
                                                                  Result* outResult) const {
  if (!pimpl_) {
    Result::setResult(outResult, Result::Code::RuntimeError, "The VGF model is not loaded");
    return {};
  }
  std::vector<TensorDesc> inputDescs;
  std::vector<TensorDesc> outputDescs;
  if (!describeBoundTensors(ctx, pimpl_->inputs, inputs, "input", inputDescs, outResult) ||
      !describeBoundTensors(ctx, pimpl_->outputs, outputs, "output", outputDescs, outResult)) {
    return {};
  }
  return createPipeline(ctx, *pimpl_, inputDescs, outputDescs, debugName, outResult);
}

} // namespace lvk
