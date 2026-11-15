/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Rebuilds a VGF file around another graph module, replacing the shapes of the resources named on the command line.
 * Together with `make_nss_model.py` this shape-specializes a model, such as Arm NSS, for another resolution: the script
 * works out every shape and rewrites the SPIR-V module, this tool writes the model resource table and the model
 * sequence around it. Constant data is copied verbatim.
 *
 * usage: vgf_respecialize <input.vgf> <module.spv> <output.vgf> [<resource>:<dim>,<dim>,... ...]
 *        vgf_respecialize --dump <input.vgf>
 */

#include <vgf/decoder.hpp>
#include <vgf/encoder.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlsdk::vgflib;

namespace {

std::vector<char> readFile(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    std::fprintf(stderr, "cannot read the size of %s\n", path);
    std::exit(1);
  }
  file.seekg(0);
  std::vector<char> bytes(static_cast<size_t>(size));
  file.read(bytes.data(), size);
  return bytes;
}

} // namespace

int main(int argc, char** argv) {
  const bool dump = argc == 3 && !std::strcmp(argv[1], "--dump");

  if (!dump && argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <input.vgf> <module.spv> <output.vgf> [<resource>:<dim>,<dim>,... ...]\n"
                 "       %s --dump <input.vgf>\n",
                 argv[0],
                 argv[0]);
    return 1;
  }

  if (dump) {
    const std::vector<char> vgf = readFile(argv[2]);
    const char* const bytes = vgf.data();
    const std::unique_ptr<HeaderDecoder> header = CreateHeaderDecoder(bytes, HeaderSize(), vgf.size());
    if (!header || !header->IsValid() || !header->CheckVersion()) {
      std::fprintf(stderr, "not a valid VGF file\n");
      return 1;
    }
    const std::unique_ptr<ModelResourceTableDecoder> resources =
        CreateModelResourceTableDecoder(bytes + header->GetModelResourceTableOffset(), header->GetModelResourceTableSize());
    const std::unique_ptr<ModelSequenceTableDecoder> sequence =
        CreateModelSequenceTableDecoder(bytes + header->GetModelSequenceTableOffset(), header->GetModelSequenceTableSize());
    const std::unique_ptr<ConstantDecoder> constants =
        CreateConstantDecoder(bytes + header->GetConstantsOffset(), header->GetConstantsSize());
    if (!resources || !sequence || !constants) {
      std::fprintf(stderr, "cannot decode the sections\n");
      return 1;
    }
    static const char* kCategory[] = {"INPUT", "OUTPUT", "INTERMEDIATE", "CONSTANT"};
    for (uint32_t i = 0; i != resources->size(); i++) {
      const DataView<int64_t> shape = resources->getTensorShape(i);
      std::printf("resource %u %s format %d shape [", i, kCategory[(uint32_t)resources->getCategory(i)], (int)resources->getVkFormat(i));
      for (size_t d = 0; d != shape.size(); d++) {
        std::printf("%s%lld", d ? ", " : "", (long long)shape[d]);
      }
      std::printf("]\n");
    }
    auto slots = [&](BindingSlotArrayHandle handle, const char* kind) {
      for (uint32_t i = 0; i != sequence->getBindingsSize(handle); i++) {
        std::printf("%s %u: binding %u -> resource %u\n",
                    kind,
                    i,
                    sequence->getBindingSlotBinding(handle, i),
                    sequence->getBindingSlotMrtIndex(handle, i));
      }
    };
    slots(sequence->getSegmentInputBindingSlotsHandle(0), "input");
    slots(sequence->getSegmentOutputBindingSlotsHandle(0), "output");
    std::printf("constants: %u\n", (uint32_t)constants->size());
    return 0;
  }

  std::map<uint32_t, std::vector<int64_t>> newShapes;
  for (int arg = 4; arg != argc; arg++) {
    const char* colon = std::strchr(argv[arg], ':');
    if (!colon) {
      std::fprintf(stderr, "`%s` is not <resource>:<dim>,<dim>,...\n", argv[arg]);
      return 1;
    }
    std::vector<int64_t> shape;
    for (const char* p = colon + 1; *p;) {
      shape.push_back(std::atoll(p));
      const char* comma = std::strchr(p, ',');
      p = comma ? comma + 1 : p + std::strlen(p);
    }
    for (const int64_t d : shape) {
      if (d <= 0) {
        std::fprintf(stderr, "`%s` has a non-positive dimension\n", argv[arg]);
        return 1;
      }
    }
    newShapes[(uint32_t)std::atoll(argv[arg])] = shape;
  }

  const std::vector<char> vgf = readFile(argv[1]);
  const std::vector<char> spirvBytes = readFile(argv[2]);
  std::vector<uint32_t> spirv(spirvBytes.size() / sizeof(uint32_t));
  std::memcpy(spirv.data(), spirvBytes.data(), spirv.size() * sizeof(uint32_t));

  const char* const bytes = vgf.data();
  const std::unique_ptr<HeaderDecoder> header = CreateHeaderDecoder(bytes, HeaderSize(), vgf.size());
  if (!header || !header->IsValid() || !header->CheckVersion()) {
    std::fprintf(stderr, "not a valid VGF file\n");
    return 1;
  }
  const std::unique_ptr<ModuleTableDecoder> modules =
      CreateModuleTableDecoder(bytes + header->GetModuleTableOffset(), header->GetModuleTableSize());
  const std::unique_ptr<ModelSequenceTableDecoder> sequence =
      CreateModelSequenceTableDecoder(bytes + header->GetModelSequenceTableOffset(), header->GetModelSequenceTableSize());
  const std::unique_ptr<ModelResourceTableDecoder> resources =
      CreateModelResourceTableDecoder(bytes + header->GetModelResourceTableOffset(), header->GetModelResourceTableSize());
  const std::unique_ptr<ConstantDecoder> constants =
      CreateConstantDecoder(bytes + header->GetConstantsOffset(), header->GetConstantsSize());
  if (!modules || !sequence || !resources || !constants) {
    std::fprintf(stderr, "cannot decode the sections\n");
    return 1;
  }

  const std::unique_ptr<Encoder> encoder = CreateEncoder(header->GetEncoderVulkanHeadersVersion());

  std::vector<ResourceRef> resourceRefs;
  resourceRefs.reserve(resources->size());
  for (uint32_t i = 0; i != resources->size(); i++) {
    const DataView<int64_t> shapeView = resources->getTensorShape(i);
    const DataView<int64_t> strideView = resources->getTensorStride(i);
    std::vector<int64_t> shape(shapeView.begin(), shapeView.end());
    const std::vector<int64_t> strides(strideView.begin(), strideView.end());
    const ResourceCategory category = resources->getCategory(i);
    const FormatType format = resources->getVkFormat(i);
    const std::optional<DescriptorType> descriptorType = resources->getDescriptorType(i);
    const std::optional<AliasGroupId> aliasGroup = resources->getAliasGroupId(i);

    const std::map<uint32_t, std::vector<int64_t>>::const_iterator replacement = newShapes.find(i);
    if (replacement != newShapes.end()) {
      if (replacement->second.size() != shape.size()) {
        std::fprintf(stderr, "resource %u has rank %zu, given %zu dimensions\n", i, shape.size(), replacement->second.size());
        return 1;
      }
      shape = replacement->second;
    }

    switch (category) {
    case ResourceCategory::INPUT:
      resourceRefs.push_back(encoder->AddInputResource(descriptorType.value_or(0), format, shape, strides, aliasGroup));
      break;
    case ResourceCategory::OUTPUT:
      resourceRefs.push_back(encoder->AddOutputResource(descriptorType.value_or(0), format, shape, strides, aliasGroup));
      break;
    case ResourceCategory::INTERMEDIATE:
      resourceRefs.push_back(encoder->AddIntermediateResource(descriptorType.value_or(0), format, shape, strides, aliasGroup));
      break;
    case ResourceCategory::CONSTANT:
      resourceRefs.push_back(encoder->AddConstantResource(format, shape, strides));
      break;
    }
  }

  std::vector<ConstantRef> constantRefs;
  constantRefs.reserve(constants->size());
  for (uint32_t i = 0; i != constants->size(); i++) {
    const DataView<uint8_t> data = constants->getConstant(i);
    const int64_t sparsity = constants->isSparseConstant(i) ? constants->getConstantSparsityDimension(i) : CONSTANT_NOT_SPARSE_DIMENSION;
    constantRefs.push_back(encoder->AddConstant(resourceRefs[constants->getConstantMrtIndex(i)], data.begin(), data.size(), sparsity));
  }

  std::map<std::pair<uint32_t, uint32_t>, BindingSlotRef> slots;
  auto bindingSlots = [&](BindingSlotArrayHandle handle) {
    std::vector<BindingSlotRef> refs;
    for (uint32_t i = 0; i != sequence->getBindingsSize(handle); i++) {
      const uint32_t binding = sequence->getBindingSlotBinding(handle, i);
      const uint32_t mrtIndex = sequence->getBindingSlotMrtIndex(handle, i);
      const std::pair<uint32_t, uint32_t> key{binding, mrtIndex};
      const std::map<std::pair<uint32_t, uint32_t>, BindingSlotRef>::const_iterator found = slots.find(key);
      if (found == slots.end()) {
        const BindingSlotRef ref = encoder->AddBindingSlot(binding, resourceRefs[mrtIndex]);
        slots.emplace(key, ref);
        refs.push_back(ref);
      } else {
        refs.push_back(found->second);
      }
    }
    return refs;
  };

  for (uint32_t segment = 0; segment != sequence->modelSequenceTableSize(); segment++) {
    const uint32_t moduleIdx = sequence->getSegmentModuleIndex(segment);
    const std::string moduleName(modules->getModuleName(moduleIdx));
    const std::string entryPoint(modules->getModuleEntryPoint(moduleIdx));
    const bool isGraph = sequence->getSegmentType(segment) == ModuleType::GRAPH;
    const ModuleRef module =
        encoder->AddModule(sequence->getSegmentType(segment), moduleName, entryPoint, isGraph ? spirv : std::vector<uint32_t>());

    std::vector<DescriptorSetInfoRef> descriptors;
    for (uint32_t descIdx = 0; descIdx != sequence->getSegmentDescriptorSetInfosSize(segment); descIdx++) {
      descriptors.push_back(encoder->AddDescriptorSetInfo(bindingSlots(sequence->getDescriptorBindingSlotsHandle(segment, descIdx)),
                                                          sequence->getSegmentDescriptorSetIndex(segment, descIdx)));
    }

    std::vector<PushConstRangeRef> pushConstants;
    const PushConstantRangeHandle pushHandle = sequence->getSegmentPushConstRange(segment);
    for (uint32_t range = 0; range != sequence->getPushConstRangesSize(pushHandle); range++) {
      pushConstants.push_back(encoder->AddPushConstRange(sequence->getPushConstRangeStageFlags(pushHandle, range),
                                                         sequence->getPushConstRangeOffset(pushHandle, range),
                                                         sequence->getPushConstRangeSize(pushHandle, range)));
    }

    std::vector<ConstantRef> segmentConstants;
    for (const uint32_t constantIdx : sequence->getSegmentConstantIndexes(segment)) {
      segmentConstants.push_back(constantRefs[constantIdx]);
    }

    const DataView<uint32_t> dispatch = sequence->getSegmentDispatchShape(segment);
    std::array<uint32_t, 3> dispatchShape = {};
    for (uint32_t i = 0; i != dispatch.size() && i != 3; i++) {
      dispatchShape[i] = dispatch[i];
    }

    encoder->AddSegmentInfo(module,
                            std::string(sequence->getSegmentName(segment)),
                            descriptors,
                            bindingSlots(sequence->getSegmentInputBindingSlotsHandle(segment)),
                            bindingSlots(sequence->getSegmentOutputBindingSlotsHandle(segment)),
                            segmentConstants,
                            dispatchShape,
                            pushConstants);
  }

  auto names = [&](NameArrayHandle handle) {
    std::vector<std::string> list;
    for (uint32_t i = 0; i != sequence->getNamesSize(handle); i++) {
      list.emplace_back(sequence->getName(handle, i));
    }
    return list;
  };
  encoder->AddModelSequenceInputsOutputs(bindingSlots(sequence->getModelSequenceInputBindingSlotsHandle()),
                                         names(sequence->getModelSequenceInputNamesHandle()),
                                         bindingSlots(sequence->getModelSequenceOutputBindingSlotsHandle()),
                                         names(sequence->getModelSequenceOutputNamesHandle()));
  encoder->Finish();

  std::ofstream output(argv[3], std::ios::binary);
  if (!encoder->WriteTo(output)) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  std::printf("%s: %u resources, %u constants, module of %zu SPIR-V words\n",
              argv[3],
              (uint32_t)resources->size(),
              (uint32_t)constants->size(),
              spirv.size());
  return 0;
}
