// Dawn's resource table API for emdawnwebgpu, on top of Chrome's JS API (`--enable-unsafe-webgpu`; see Dawn's
// src/dawn/node/interop/DawnExtensions.idl). WebGPUClasses.cpp maps Dawn's C API onto these.
addToLibrary({
  $LVKResourceTables__deps: ['$WebGPU'],
  // unknown to emdawnwebgpu's feature names
  $LVKResourceTables__postset: "WebGPU.FeatureName[0x0005003A] = 'chromium-experimental-sampling-resource-table';",
  $LVKResourceTables: {
    // a WGPUResourceTable is an index into this array; 0 is null
    tables: [null],
  },

  lvkWebGPUCreateResourceTable__deps: ['$LVKResourceTables'],
  lvkWebGPUCreateResourceTable__sig: 'ppi',
  lvkWebGPUCreateResourceTable: (devicePtr, size) => {
    const device = WebGPU.getJsObject(devicePtr);
    if (!device.createResourceTable) {
      return 0;
    }
    LVKResourceTables.tables.push(device.createResourceTable({ 'size': size }));
    return LVKResourceTables.tables.length - 1;
  },

  // a sampler or a texture view; invalid updates are validation errors
  lvkWebGPUResourceTableUpdate__deps: ['$LVKResourceTables'],
  lvkWebGPUResourceTableUpdate__sig: 'ipipp',
  lvkWebGPUResourceTableUpdate: (table, slot, samplerPtr, textureViewPtr) => {
    try {
      LVKResourceTables.tables[table].update(slot, WebGPU.getJsObject(samplerPtr || textureViewPtr));
      return 1; // WGPUStatus_Success
    } catch (e) {
      return 2; // WGPUStatus_Error
    }
  },

  lvkWebGPUResourceTableRemove__deps: ['$LVKResourceTables'],
  lvkWebGPUResourceTableRemove__sig: 'vpi',
  lvkWebGPUResourceTableRemove: (table, slot) => {
    LVKResourceTables.tables[table].remove(slot);
  },

  lvkWebGPUResourceTableDestroy__deps: ['$LVKResourceTables'],
  lvkWebGPUResourceTableDestroy__sig: 'vp',
  lvkWebGPUResourceTableDestroy: (table) => {
    LVKResourceTables.tables[table].destroy();
  },

  lvkWebGPUResourceTableRelease__deps: ['$LVKResourceTables'],
  lvkWebGPUResourceTableRelease__sig: 'vp',
  lvkWebGPUResourceTableRelease: (table) => {
    LVKResourceTables.tables[table] = undefined;
  },

  lvkWebGPUComputePassEncoderSetResourceTable__deps: ['$LVKResourceTables'],
  lvkWebGPUComputePassEncoderSetResourceTable__sig: 'vpp',
  lvkWebGPUComputePassEncoderSetResourceTable: (passPtr, table) => {
    WebGPU.getJsObject(passPtr).setResourceTable(LVKResourceTables.tables[table]);
  },

  // emdawnwebgpu builds GPURenderPassDescriptor without the table: the next beginRenderPass() of the encoder adds it
  lvkWebGPUCommandEncoderSetNextRenderPassResourceTable__deps: ['$LVKResourceTables'],
  lvkWebGPUCommandEncoderSetNextRenderPassResourceTable__sig: 'vpp',
  lvkWebGPUCommandEncoderSetNextRenderPassResourceTable: (encoderPtr, table) => {
    const resourceTable = LVKResourceTables.tables[table];
    WebGPU.getJsObject(encoderPtr).beginRenderPass = function (desc) {
      delete this.beginRenderPass;
      return this.beginRenderPass({ ...desc, 'resourceTable': resourceTable });
    };
  },

  // the same for GPUPipelineLayoutDescriptor.usesResourceTable
  lvkWebGPUDeviceSetNextPipelineLayoutUsesResourceTable__deps: ['$WebGPU'],
  lvkWebGPUDeviceSetNextPipelineLayoutUsesResourceTable__sig: 'vp',
  lvkWebGPUDeviceSetNextPipelineLayoutUsesResourceTable: (devicePtr) => {
    WebGPU.getJsObject(devicePtr).createPipelineLayout = function (desc) {
      delete this.createPipelineLayout;
      return this.createPipelineLayout({ ...desc, 'usesResourceTable': true });
    };
  },
});
