/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ldrutils/lutils/Pool.h>
#include <webgpu/webgpu.h>

#if defined(__EMSCRIPTEN__)
// emdawnwebgpu has no resource table: the parts of Dawn's API LVK uses, implemented in WebGPUEmscripten.js
const WGPUFeatureName WGPUFeatureName_ChromiumExperimentalSamplingResourceTable = WGPUFeatureName(0x0005003A);
typedef struct WGPUResourceTableImpl* WGPUResourceTable;
typedef struct WGPUResourceTableDescriptor {
  WGPUChainedStruct* nextInChain;
  WGPUStringView label;
  uint32_t size;
} WGPUResourceTableDescriptor;
typedef struct WGPUBindingResource {
  WGPUSampler sampler;
  WGPUTextureView textureView;
} WGPUBindingResource;
#endif // __EMSCRIPTEN__

#include <lvk/LVK.h>
#include <lvk/webgpu/WebGPUUtils.h>

namespace lvk {

class WebGPUContext;

struct WebGPUShaderModuleState final {
  WGPUShaderModule module = nullptr;
  ShaderStage stage = Stage_Vert;
  PushConstantsLocation pushConstants = PushConstantsLocation_None;
  uint32_t pushConstantsSize = 0;
  std::vector<WGSLOverride> overrides;
};

// a specialization constant, set as the override "constantId"
struct WebGPUPipelineConstant final {
  uint32_t constantId = 0;
  double value = 0.0;
};

// LVK's dynamic state which WebGPU bakes into render pipelines
struct WebGPUDynamicState final {
  CompareOp depthCompareOp = CompareOp_AlwaysPass;
  bool isDepthWriteEnabled = false;
  bool isStencilEnabled = false;
  StencilFaceState stencilFront = {};
  StencilFaceState stencilBack = {};
  bool isDepthBiasEnabled = false;
  float depthBiasConstant = 0.0f;
  float depthBiasSlope = 0.0f;
  float depthBiasClamp = 0.0f;
};

struct WebGPURenderPipelineVariant final {
  WebGPUDynamicState state = {};
  WGPURenderPipeline pipeline = nullptr;
};

struct WebGPURenderPipelineState final {
  RenderPipelineDesc desc = {}; // desc.debugName and desc.specInfo.data are not retained
  WGPUShaderModule vert = nullptr;
  WGPUShaderModule frag = nullptr;
  WGPUPipelineLayout layout = nullptr;
  uint32_t immediateSize = 0;
  bool hasPushConstantsBuffer = false; // see WebGPUPushConstantsChunk
  std::vector<WebGPUPipelineConstant> vertConstants;
  std::vector<WebGPUPipelineConstant> fragConstants;
  std::vector<WebGPURenderPipelineVariant> variants; // created lazily
  std::string debugName;
};

// Buffers with a GPU address are ranges of kNumArenas big storage buffers. A GPU address is (arena << 32) | offset,
// `vec2u(offset, arena)` in shaders; arena 0 is null.
enum : uint32_t {
  kArenaAlignment = 256, // also satisfies vertex, index and indirect buffer offsets
};

struct WebGPUArena final {
  struct Range {
    uint64_t offset = 0;
    uint64_t size = 0;
  };
  WGPUBuffer buffer = nullptr;
  uint64_t size = 0;
  std::vector<Range> freeRanges; // sorted by offset
};

struct WebGPUBuffer final {
  WGPUBuffer buffer = nullptr; // the arena's buffer if `arena` is not 0
  uint64_t offset = 0; // within the arena
  uint64_t size = 0;
  uint32_t arena = 0; // 1-based; 0: standalone, without a GPU address
  BufferUsageFlags usage = 0;
  StorageType storage = StorageType_Device;
  // WebGPU has no persistently mapped buffers: getMappedPtr() returns this CPU copy. submit() writes its changes to the GPU,
  // the GPU changes are merged back after the work (see WebGPUContext::PendingReadback).
  std::vector<uint32_t> mappedData;
  std::vector<uint32_t> gpuData; // what the GPU buffer holds as far as the CPU knows
};

// Push constants which cannot be immediates are a `var<storage, read>` at @group(0) @binding(kNumArenas). Each draw or
// dispatch copies them into a 256-byte slot of a chunk, bound with a dynamic offset; submit() writes the chunks.
enum : uint32_t {
  kPushConstantsSlotSize = 256, // the largest push constant block, also the largest minStorageBufferOffsetAlignment
  kPushConstantsSlotsPerChunk = 256,
};

struct WebGPUPushConstantsChunk final {
  WGPUBuffer buffer = nullptr;
  WGPUBindGroup bindGroupReadOnly = nullptr; // the arenas and this buffer, for render pipelines
  WGPUBindGroup bindGroupReadWrite = nullptr; // for compute pipelines
  uint32_t arenasVersion = 0; // the bind groups are recreated when an arena is added
  uint32_t numUsedSlots = 0;
  std::vector<uint8_t> data; // the used slots, written to `buffer` by submit()
};

struct WebGPUComputePipelineState final {
  WGPUComputePipeline pipeline = nullptr;
  WGPUPipelineLayout layout = nullptr;
  uint32_t immediateSize = 0;
  bool hasPushConstantsBuffer = false; // see WebGPUPushConstantsChunk
};

struct WebGPUTexture final {
  WGPUTexture texture = nullptr;
  WGPUTextureView view = nullptr; // all aspects, levels and layers
  WGPUTextureView sampledView = nullptr; // TextureBinding usage and one aspect, for the resource table
  WGPUTextureViewDimension viewDimension = WGPUTextureViewDimension_2D;
  Format format = Format_Invalid;
  TextureType type = TextureType_2D;
  Dimensions extent = {};
  uint32_t baseLevel = 0; // non-zero for texture views
  uint32_t baseLayer = 0;
  uint32_t numLevels = 1;
  uint32_t numLayers = 1; // array layers, 6 per cube map
  uint32_t numSamples = 1;
  TextureUsageFlags usage = 0;
  bool isOwningTexture = true; // false for createTextureView() views
  // the swapchain slot: its texture is never destroyed
  bool isSwapchainTexture = false;
};

struct WebGPUSampler final {
  WGPUSampler sampler = nullptr;
};

class WebGPUCommandBuffer final : public ICommandBuffer {
 public:
  explicit WebGPUCommandBuffer(WebGPUContext* ctx);
  ~WebGPUCommandBuffer() override;

  void cmdBarrier(const Barrier& barrier) override;
  void cmdTransitionToGeneral(const ldr::Span<TextureHandle>& textures, ShaderStage extraDstStage) const override;
  void cmdTransitionToShaderReadOnly(const ldr::Span<TextureHandle>& textures, ShaderStage extraDstStage) const override;
  void cmdTransitionToRenderingLocalRead(const ldr::Span<TextureHandle>& textures) const override;

  void cmdPushDebugGroupLabel(const char* label, uint32_t colorRGBA) const override;
  void cmdInsertDebugEventLabel(const char* label, uint32_t colorRGBA) const override;
  void cmdPopDebugGroupLabel() const override;

  void cmdBindRayTracingPipeline(RayTracingPipelineHandle handle) override;

  void cmdBindComputePipeline(ComputePipelineHandle handle) override;
  void cmdDispatch(const Dimensions& groupCount, const Dependencies& deps) override;
  void cmdDispatchIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, const Dependencies& deps) override;

  void cmdBeginRendering(const RenderPass& renderPass, const Framebuffer& desc, const Dependencies& deps) override;
  void cmdEndRendering() override;
  void cmdNextSubpass() override;

  void cmdBindViewport(const Viewport& viewport) override;
  void cmdBindScissorRect(const ScissorRect& rect) override;

  void cmdBindRenderPipeline(RenderPipelineHandle handle) override;
  void cmdBindDepthState(const DepthState& state) override;
  void cmdBindStencilState(const StencilState& state) override;

  void cmdBindVertexBuffer(uint32_t index, BufferHandle buffer, uint64_t bufferOffset, uint64_t bufferSize) override;
  void cmdBindIndexBuffer(BufferHandle indexBuffer, IndexFormat indexFormat, uint64_t bufferOffset, uint64_t bufferSize) override;
  void cmdPushConstants(const void* data, size_t size, size_t offset) override;

  void cmdCopyBuffer(BufferHandle srcBuffer, BufferHandle dstBuffer, size_t srcOffset, size_t dstOffset, size_t size) override;
  void cmdFillBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, uint32_t data) override;
  void cmdUpdateBuffer(BufferHandle buffer, size_t bufferOffset, size_t size, const void* data) override;

  void cmdDraw(uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t baseInstance) override;
  void cmdDrawIndexed(uint32_t indexCount,
                      uint32_t instanceCount,
                      uint32_t firstIndex,
                      int32_t vertexOffset,
                      uint32_t baseInstance) override;
  void cmdDrawIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, uint32_t drawCount, uint32_t stride) override;
  void cmdDrawIndexedIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, uint32_t drawCount, uint32_t stride) override;
  void cmdDrawIndexedIndirectCount(BufferHandle indirectBuffer,
                                   size_t indirectBufferOffset,
                                   BufferHandle countBuffer,
                                   size_t countBufferOffset,
                                   uint32_t maxDrawCount,
                                   uint32_t stride) override;
  void cmdDrawMeshTasks(const Dimensions& threadgroupCount) override;
  void cmdDrawMeshTasksIndirect(BufferHandle indirectBuffer, size_t indirectBufferOffset, uint32_t drawCount, uint32_t stride) override;
  void cmdDrawMeshTasksIndirectCount(BufferHandle indirectBuffer,
                                     size_t indirectBufferOffset,
                                     BufferHandle countBuffer,
                                     size_t countBufferOffset,
                                     uint32_t maxDrawCount,
                                     uint32_t stride) override;
  void cmdTraceRays(uint32_t width, uint32_t height, uint32_t depth, const Dependencies& deps) override;

  void cmdSetBlendColor(const float color[4]) override;
  void cmdSetDepthBias(float constantFactor, float slopeFactor, float clamp) override;
  void cmdSetDepthBiasEnable(bool enable) override;
  void cmdSetDepthBounds(float minDepthBounds, float maxDepthBounds) override;
  void cmdSetPrimitiveRestartEnable(bool enable) override;
  void cmdSetFragmentShadingRate(const Dimensions& fragmentSize,
                                 ShadingRateCombinerOp primitiveOp,
                                 ShadingRateCombinerOp attachmentOp) override;

  void cmdResetQueryPool(QueryPoolHandle pool, uint32_t firstQuery, uint32_t queryCount) override;
  void cmdWriteTimestamp(QueryPoolHandle pool, uint32_t query) override;

  void cmdClearColorImage(TextureHandle tex, const ClearColorValue& value, const TextureLayers& layers) override;
  void cmdCopyImage(TextureHandle src,
                    TextureHandle dst,
                    const Dimensions& extent,
                    const Offset3D& srcOffset,
                    const Offset3D& dstOffset,
                    const TextureLayers& srcLayers,
                    const TextureLayers& dstLayers) override;
  void cmdGenerateMipmap(TextureHandle handle) override;
  void cmdUpdateTLAS(AccelStructHandle handle, BufferHandle instancesBuffer) override;
  void cmdUpdateBLAS(const ldr::Span<AccelStructHandle>& handles) override;

  void cmdCopyTensor(TensorHandle src, TensorHandle dst) override;
  void cmdDispatchDataGraph(DataGraphPipelineHandle handle,
                            const ldr::Span<const TensorHandle>& inputs,
                            const ldr::Span<const TensorHandle>& outputs,
                            const Dependencies& deps) override;

 private:
  friend class WebGPUContext;

  // binds the variant of the current render pipeline for the current dynamic state; returns false if there is none
  bool flushRenderPipeline();
  // group 0 with the current arenas, without push constant slot
  void bindRenderPassArenas();
  // a compute pass with the pipeline, the arenas, the resource table and the immediates
  WGPUComputePassEncoder beginComputePass(const WebGPUComputePipelineState& cps);
  // a single-layer, single-level view for this render pass if needed
  WGPUTextureView getAttachmentView(const WebGPUTexture& tex, uint32_t layer, uint32_t level);
  // copies changed push constants into a chunk slot; returns the chunk and offset
  const WebGPUPushConstantsChunk& getPushConstantsSlot(uint32_t* outOffset);
  // the immediates WebGPU can take: the push constant blocks of buffer pipelines can be bigger
  void setRenderPassImmediates(uint32_t offset, const void* data, uint32_t size) const;
  void markHostVisibleBufferWritten(BufferHandle handle, const WebGPUBuffer& buf);
  // a buffer write ordered with the other commands of this encoder
  void writeBuffer(WGPUBuffer buffer, uint64_t offset, const void* data, uint64_t size);
  bool isHostVisibleBufferWritten(BufferHandle handle) const;

  WebGPUContext* ctx_ = nullptr;
  WGPUCommandEncoder encoder_ = nullptr;
  WGPURenderPassEncoder pass_ = nullptr;
  RenderPipelineHandle currentPipeline_ = {};
  ComputePipelineHandle currentComputePipeline_ = {};
  WGPURenderPipeline lastBoundPipeline_ = nullptr;
  WebGPUDynamicState dynamicState_ = {};
  std::vector<WGPUTextureView> passViews_; // released at cmdEndRendering()
  // host-visible buffers the GPU may change: they are read back after this command buffer
  bool hasDispatches_ = false; // compute shaders can write any host-visible storage buffer
  std::vector<BufferHandle> writtenHostVisibleBuffers_; // destinations of copies, fills and updates
  uint32_t boundArenasVersion_ = 0; // of the arenas bind group bound in the current render pass
  // push constants set outside a render pass are applied when the next one begins
  uint8_t pushConstants_[kPushConstantsSlotSize] = {};
  uint32_t pushConstantsSize_ = 0;
  std::vector<WebGPUPushConstantsChunk> pushConstantsChunks_;
  bool hasPushConstantsSlot_ = false; // the last slot holds the current push constants
  bool isPushConstantsSlotBound_ = false; // in the current render pass
  // vertex and index buffers bound outside render passes, like in Vulkan
  struct BufferBinding {
    WGPUBuffer buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0;
  };
  BufferBinding vertexBuffers_[VertexInput::LVK_VERTEX_BUFFER_MAX] = {};
  BufferBinding indexBuffer_ = {};
  WGPUIndexFormat indexFormat_ = WGPUIndexFormat_Undefined;
};

class WebGPUContext final : public IContext {
 public:
  // `surfaceSource` is a chained WGPUSurfaceSource* struct, or nullptr for a context without a swapchain
  WebGPUContext(const ContextConfig& config, const WGPUChainedStruct* surfaceSource, uint32_t width, uint32_t height);
  ~WebGPUContext() override;

  [[nodiscard]] bool isValid() const {
    return device_ != nullptr;
  }

  ICommandBuffer& acquireCommandBuffer(bool dedicatedCompute) override;

  SubmitHandle submit(ICommandBuffer& commandBuffer, TextureHandle present, const ldr::Span<TextureHandle>& release) override;
  void wait(SubmitHandle handle) override;
  [[nodiscard]] bool isReady(SubmitHandle handle) const override;

  [[nodiscard]] Holder<BufferHandle> createBuffer(const BufferDesc& desc, const char* debugName, Result* outResult) override;
  [[nodiscard]] Holder<SamplerHandle> createSampler(const SamplerStateDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<TextureHandle> createTexture(const TextureDesc& desc, const char* debugName, Result* outResult) override;
  [[nodiscard]] Holder<TextureHandle> createTextureView(TextureHandle texture,
                                                        const TextureViewDesc& desc,
                                                        const char* debugName,
                                                        Result* outResult) override;
  [[nodiscard]] Holder<ComputePipelineHandle> createComputePipeline(const ComputePipelineDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<RenderPipelineHandle> createRenderPipeline(const RenderPipelineDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<RayTracingPipelineHandle> createRayTracingPipeline(const RayTracingPipelineDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<ShaderModuleHandle> createShaderModule(const ShaderModuleDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<QueryPoolHandle> createQueryPool(uint32_t numQueries, const char* debugName, Result* outResult) override;
  [[nodiscard]] Holder<AccelStructHandle> createAccelerationStructure(const AccelStructDesc& desc, Result* outResult) override;
  [[nodiscard]] Holder<TensorHandle> createTensor(const TensorDesc& desc, const char* debugName, Result* outResult) override;
  [[nodiscard]] Holder<TensorHandle> createTensorBufferAliasing(BufferHandle buffer,
                                                                uint64_t bufferOffset,
                                                                const TensorDesc& desc,
                                                                const char* debugName,
                                                                Result* outResult) override;
  [[nodiscard]] Holder<TensorHandle> createTensorTextureAliasing(TextureHandle texture,
                                                                 const TensorDesc& desc,
                                                                 const char* debugName,
                                                                 Result* outResult) override;
  [[nodiscard]] Holder<DataGraphPipelineHandle> createDataGraphPipeline(const DataGraphPipelineDesc& desc, Result* outResult) override;

  void destroy(ComputePipelineHandle handle) override;
  void destroy(RenderPipelineHandle handle) override;
  void destroy(RayTracingPipelineHandle handle) override;
  void destroy(ShaderModuleHandle handle) override;
  void destroy(SamplerHandle handle) override;
  void destroy(BufferHandle handle) override;
  void destroy(TextureHandle handle) override;
  void destroy(QueryPoolHandle handle) override;
  void destroy(AccelStructHandle handle) override;
  void destroy(TensorHandle handle) override;
  void destroy(DataGraphPipelineHandle handle) override;
  void destroy(Framebuffer& fb) override;

  [[nodiscard]] uint64_t gpuAddress(AccelStructHandle handle) const override;
  [[nodiscard]] AccelStructSizes getAccelStructSizes(const AccelStructDesc& desc, Result* outResult) const override;

  Result upload(BufferHandle handle, const void* data, size_t size, size_t offset) override;
  Result download(BufferHandle handle, void* data, size_t size, size_t offset) override;
  [[nodiscard]] uint8_t* getMappedPtr(BufferHandle handle) const override;
  [[nodiscard]] uint64_t gpuAddress(BufferHandle handle, size_t offset) const override;
  void flushMappedMemory(BufferHandle handle, size_t offset, size_t size) const override;
  void invalidateMappedMemory(BufferHandle handle, size_t offset, size_t size) const override;
  [[nodiscard]] uint32_t getMaxStorageBufferRange() const override;

  Result upload(TensorHandle handle, const void* data, size_t size, size_t offset) override;
  Result download(TensorHandle handle, void* data, size_t size, size_t offset) override;
  [[nodiscard]] uint8_t* getMappedPtr(TensorHandle handle) const override;
  [[nodiscard]] TensorDesc getTensorDesc(TensorHandle handle) const override;

  Result upload(TextureHandle handle, const TextureRangeDesc& range, const void* data, uint32_t bufferRowLength) override;
  Result download(TextureHandle handle, const TextureRangeDesc& range, void* outData) override;
  [[nodiscard]] Dimensions getDimensions(TextureHandle handle) const override;
  [[nodiscard]] float getAspectRatio(TextureHandle handle) const override;
  [[nodiscard]] Format getFormat(TextureHandle handle) const override;

  Result initSwapchain(uint32_t width, uint32_t height) override;
  TextureHandle getCurrentSwapchainTexture() override;
  Format getSwapchainFormat() const override;
  ColorSpace getSwapchainColorSpace() const override;
  uint32_t getSwapchainCurrentImageIndex() const override;
  uint32_t getNumSwapchainImages() const override;
  [[nodiscard]] SurfaceTransform getSwapchainSurfaceTransform() const override;
  [[nodiscard]] bool setCurrentPresentMode(PresentMode mode) override;
  [[nodiscard]] PresentMode getCurrentPresentMode() const override;

  uint32_t getFramebufferMSAABitMask() const override;

  [[nodiscard]] Dimensions getShadingRateAttachmentMinTexelSize() const override;
  [[nodiscard]] Dimensions getShadingRateAttachmentMaxTexelSize() const override;
  [[nodiscard]] ldr::Span<const Dimensions> getSupportedFragmentShadingRates() const override;
  [[nodiscard]] Dimensions getFragmentDensityMapMinTexelSize() const override;

  bool isExtensionEnabled(const char* ext) const override;
  bool supportsAsyncCompute() const override;
  bool supportsDepthBounds() const override;
  bool supportsMeshShader() const override;
  bool supportsProvokingVertex() const override;
  bool supportsRayQuery() const override;
  bool supportsRayTracingPipeline() const override;
  bool supportsOpacityMicromap() const override;
  bool supportsShaderInterlock() const override;
  bool supportsCooperativeMatrix() const override;
  bool supportsTextureFormat(Format format, TextureUsageFlags usageFlags) const override;
  bool supportsTensorsARM() const override;
  bool supportsDataGraphARM() const override;
  [[nodiscard]] TensorProperties getTensorProperties() const override;
  [[nodiscard]] TensorUsageFlags getTensorFormatSupport(Format format, TensorTiling tiling) const override;

  [[nodiscard]] uint32_t getMultiviewMaxViewCount() const override;
  [[nodiscard]] uint32_t getMultiviewMaxMeshViewCount() const override;

  [[nodiscard]] size_t getPipelineCacheData(void* outData, size_t dataSize) const override;

  double getTimestampPeriodToMs() const override;
  bool getQueryPoolResults(QueryPoolHandle pool, uint32_t firstQuery, uint32_t queryCount, size_t dataSize, void* outData, size_t stride)
      const override;

  // see setPresentedFramesRetainedWebGPU()
  void setPresentedFramesRetained(bool retain) {
    isPresentedFrameRetained_ = retain;
  }
  // see getPresentModesWebGPU()
  uint32_t getPresentModes(PresentMode* outModes, uint32_t maxModes) const;

 private:
  friend class WebGPUCommandBuffer;

  void releaseSwapchainTexture();
  // gets the current surface texture, reconfiguring the surface if it is outdated; nullptr on failure
  WGPUTexture acquireSurfaceTexture();
  Result initHeadlessSwapchain(uint32_t width, uint32_t height);
  // copies the swapchain texture into retainedFrame_ before it is presented
  void retainPresentedFrame(WGPUCommandEncoder encoder);
  // blocks until all the work submitted so far is done
  void waitIdle();
  // frees the slots and arena ranges of destroyed resources once the work submitted before is done
  void processDeferredSlots();
  // the submit value of the last command buffer acquired so far
  uint64_t getNextSubmitValue() const {
    return lastSubmitValue_ + commandBuffers_.size();
  }
  void releaseTexture(WebGPUTexture& tex) const;
  Result createTextureViews(WebGPUTexture& tex, const TextureViewDesc& desc, const char* debugName) const;
  void updateResourceTable(uint32_t slot, const WGPUBindingResource& resource) const;
  // adds the texture to the pool and its sampled view to the resource table
  TextureHandle createTextureInResourceTable(WebGPUTexture&& tex);
  WGPURenderPipeline getRenderPipeline(RenderPipelineHandle handle, const WebGPUDynamicState& state);
  WGPURenderPipeline createRenderPipelineVariant(const WebGPURenderPipelineState& rps, const WebGPUDynamicState& state) const;
  // group 0: the arenas (read-only for render pipelines, read-write for compute pipelines)
  WGPUPipelineLayout createPipelineLayout(bool isCompute, uint32_t immediateSize, const char* debugName) const;
  bool allocateFromArena(uint64_t size, uint32_t* outArena, uint64_t* outOffset, const char* debugName);
  void freeToArena(uint32_t arena, uint64_t offset, uint64_t size);
  void updateArenaBindGroups();
  void submitEncoder(WGPUCommandEncoder encoder) const;
  // submits the copy into a MapRead `staging` buffer and blocks until it is mapped; nullptr if mapping failed
  const uint8_t* submitAndMapStagingBuffer(WGPUCommandEncoder encoder, WGPUBuffer staging, uint64_t size);
  void releaseStagingBuffer(WGPUBuffer staging, bool isMapped) const;
  WGPUBuffer getCopyScratchBuffer(uint64_t size);
  // group 0: the arenas and one slot of `pushConstants` (bound with a dynamic offset)
  WGPUBindGroup createArenasBindGroup(bool isCompute, WGPUBuffer pushConstants, const char* debugName) const;
  WebGPUPushConstantsChunk acquirePushConstantsChunk();
  void updatePushConstantsChunkBindGroups(WebGPUPushConstantsChunk& chunk) const;
  void releasePushConstantsChunk(WebGPUPushConstantsChunk& chunk) const;
  // writes the CPU changes of host-visible buffers to the GPU
  void flushHostVisibleBuffers();
  // copies the host-visible buffers `cmdBuffer` may change into a staging buffer, mapped after `submitValue` is done
  void readBackHostVisibleBuffers(WebGPUCommandBuffer& cmdBuffer, uint64_t submitValue);
  // merges the finished readbacks into the CPU copies; blocks for those of the submits up to `waitSubmitValue`
  void processReadbacks(uint64_t waitSubmitValue);
  void mergeOldestReadback();
  // render passes downsampling each level from the previous one
  void generateMipmap(WGPUCommandEncoder encoder, const WebGPUTexture& tex);

  ContextConfig config_;
  WGPUInstance instance_ = nullptr;
  WGPUAdapter adapter_ = nullptr;
  WGPUDevice device_ = nullptr;
  WGPUQueue queue_ = nullptr;
  WGPUSurface surface_ = nullptr;
  WGPULimits limits_ = {};
  bool hasTextureComponentSwizzle_ = false;

  // buffer device addresses
  WebGPUArena arenas_[kNumArenas];
  // placeholders for missing arenas, one per binding (WebGPU forbids aliasing writable storage)
  WGPUBuffer dummyArenaBuffers_[kNumArenas] = {};
  WGPUBindGroupLayout arenasLayoutReadOnly_ = nullptr;
  WGPUBindGroupLayout arenasLayoutReadWrite_ = nullptr;
  WGPUBindGroup arenasBindGroupReadOnly_ = nullptr;
  WGPUBindGroup arenasBindGroupReadWrite_ = nullptr;
  WGPUBuffer indirectScratchBuffer_ = nullptr; // cmdDispatchIndirect() cannot read from a writable arena
  WGPUBuffer copyScratchBuffer_ = nullptr; // cmdCopyBuffer() within one arena
  uint64_t copyScratchBufferSize_ = 0;
  WGPUBuffer dummyPushConstantsBuffer_ = nullptr; // for pipelines without a push constants buffer
  uint32_t arenasVersion_ = 0; // incremented by updateArenaBindGroups()
  std::vector<WebGPUPushConstantsChunk> freePushConstantsChunks_;

  // bindless textures and samplers (chromium_experimental_sampling_resource_table)
  WGPUResourceTable resourceTable_ = nullptr;
  struct DeferredSlot {
    TextureHandle texture = {};
    SamplerHandle sampler = {};
    uint32_t arena = 0; // a range of an arena, freed by destroy(BufferHandle)
    uint64_t arenaOffset = 0;
    uint64_t arenaSize = 0;
    uint64_t submitValue = 0;
  };
  std::vector<DeferredSlot> deferredSlots_;

  // mapped buffers emulation
  std::vector<BufferHandle> hostVisibleBuffers_;
  // host-visible buffers after one submit: only the words which differ from the snapshot are merged into the CPU copy,
  // keeping CPU writes made while the work was in flight
  struct PendingReadback {
    struct Entry {
      BufferHandle buffer = {};
      uint64_t stagingOffset = 0;
      std::vector<uint32_t> snapshot;
    };
    uint64_t submitValue = 0;
    WGPUBuffer staging = nullptr;
    uint64_t stagingSize = 0;
    WGPUFuture future = {};
    enum Status { Status_Pending, Status_Mapped, Status_Failed } status = Status_Pending;
    std::vector<Entry> entries;
  };
  std::vector<PendingReadback> pendingReadbacks_; // in submit order

  // generateMipmap(): one pipeline per texture format
  struct MipmapPipeline {
    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    WGPURenderPipeline pipeline = nullptr;
  };
  WGPUShaderModule mipmapShader_ = nullptr;
  WGPUSampler mipmapSampler_ = nullptr;
  std::vector<MipmapPipeline> mipmapPipelines_;

  WGPUTextureFormat swapchainFormat_ = WGPUTextureFormat_Undefined;
  WGPUPresentMode presentMode_ = WGPUPresentMode_Fifo;
  uint32_t swapchainWidth_ = 0;
  uint32_t swapchainHeight_ = 0;
  TextureHandle swapchainTexture_ = {}; // the slot is reused every frame
  bool isSwapchainTextureAcquired_ = false;
  bool isSwapchainReconfigurePending_ = false; // a new present mode, applied before the next frame is acquired
  // without a surface, a context created with a size renders into this texture instead
  WGPUTexture headlessSwapchain_ = nullptr;
  // a copy of the last presented frame, which the swapchain slot shows until the next frame is acquired
  bool isPresentedFrameRetained_ = false;
  WGPUTexture retainedFrame_ = nullptr;

  slang::IGlobalSession* slangGlobalSession_ = nullptr;

  std::vector<std::unique_ptr<WebGPUCommandBuffer>> commandBuffers_; // acquired, not submitted yet
  uint64_t lastSubmitValue_ = 0;
  uint64_t completedSubmitValue_ = 0; // updated from queue callbacks in wgpuInstanceProcessEvents()

  ldr::Pool<ShaderModule, WebGPUShaderModuleState> shaderModulesPool_;
  ldr::Pool<RenderPipeline, WebGPURenderPipelineState> renderPipelinesPool_;
  ldr::Pool<Buffer, WebGPUBuffer> buffersPool_;
  ldr::Pool<Texture, WebGPUTexture> texturesPool_;
  ldr::Pool<Sampler, WebGPUSampler> samplersPool_;
  ldr::Pool<ComputePipeline, WebGPUComputePipelineState> computePipelinesPool_;
};

} // namespace lvk
