# WebGPU backend

`lvk/webgpu/` (`LVKWebGPU`) runs LVK on [Dawn](https://dawn.googlesource.com/dawn) natively and on the browser's WebGPU
through Emscripten. It does not use the Vulkan backend.

Shaders are translated at run time: GLSL/Slang + LVK preamble → SPIR-V (glslang/Slang) → WGSL
([spirv-cross-wgsl](https://github.com/corporateshark/spirv-cross-wgsl), a fork of
[Ludicon/spirv-cross-wgsl](https://github.com/Ludicon/spirv-cross-wgsl)).

- `lvk/LVK.h`: `createWebGPUContextWithSwapchain()` and two helpers, declared when `LVK_WITH_WEBGPU` is defined.
- `WebGPUClasses.h`/`.cpp`: `WebGPUContext` and `WebGPUCommandBuffer` on the `webgpu.h` C API.
- `WebGPUUtils.h`/`.cpp`: `compileShaderWGSL()`, with a copy of the Vulkan backend's vertex/fragment/compute preambles (keep in sync).
- `WebGPUEmscripten.js`: the resource table in the browser.

## How LVK maps to WebGPU

- **Bindless**: textures and samplers live in Dawn's resource table (`chromium_experimental_sampling_resource_table`):
  textures at their handle index, samplers at 32768 + index. Slots are reused once the work submitted before the destroy
  completes.
- **Buffer device addresses**: Uniform, Storage and Indirect buffers are allocated from up to 4 storage-buffer arenas.
  `gpuAddress()` is `(arena << 32) | offset`, `vec2u(offset, arena)` in shaders. Arenas are read-only in render
  pipelines and read-write in compute pipelines.
- **Push constants**: `var<immediate>` if they fit in 64 bytes without arrays, otherwise a `var<storage, read>` with
  std430 layout, copied per draw/dispatch into a slot bound with a dynamic offset.
- **Host-visible buffers**: `getMappedPtr()` returns a CPU copy. `submit()` uploads its changes; `wait()`/`isReady()` merge
  back what the GPU wrote. `flushMappedMemory()`/`invalidateMappedMemory()` do nothing.
- **Dynamic state**: depth/stencil state and depth bias are baked into cached pipeline variants. WebGPU has one stencil
  reference and mask pair: the front face's is used.
- **Specialization constants**: `@id(constantId) override`; 4-byte `bool`, `int`, `uint` or `float` only.
- **Compute**: one compute pass per dispatch. Indirect dispatches and copies within an arena go through scratch buffers.
- **Swapchain**: without a window, a context created with a size renders offscreen. `setPresentedFramesRetainedWebGPU()`
  keeps a copy of presented frames for `download()` (`VulkanApp` enables it for `--screenshot-frame`).

LVK's clip space already matches WebGPU, so shaders and `WindingMode` are passed through.

Unsupported features degrade with a one-time warning: `PolygonMode_Line` fills, MSAA other than 1x/4x becomes 4x,
layered and multiview passes render layer/view 0, and pipelines with storage images fail. Everything else missing
(timestamp queries, depth bounds, primitive restart, mesh shaders, ray tracing, tensors, ...) logs "not implemented".

A filtering sampler reads depth textures as 0: sample them with a nearest sampler (see `010_OmniShadows`).

## Samples

`ADD_WEBGPU_DEMO()` in `samples/CMakeLists.txt` adds `<sample>_WebGPU` and `<sample>_WebGPU_Slang`, linked to
`LVKVulkanAppWebGPU` (`VulkanApp` built with `LVK_DEMO_WITH_WEBGPU=1`).

| Sample | Changes for WebGPU |
|---|---|
| `001_HelloTriangle`, `002_RenderToCubeMap`, `DEMO_001_SolarSystem`, `Tiny_MeshLarge` | none |
| `003_RenderToCubeMapSinglePass` | skips the Vulkan `maxColorAttachments` check |
| `009_TriplanarMapping` | present modes from `getPresentModesWebGPU()` |
| `010_OmniShadows` | 6 passes instead of multiview, a nearest shadow sampler |

The other samples need YCbCr, shading rates, mesh shaders, HDR swapchains, subpass inputs or ray tracing.

## Native build

```bash
cmake .. -DLVK_WITH_WEBGPU=ON
cmake --build . --target 001_HelloTriangle_WebGPU
```

`deploy_deps.py` clones Dawn at a pinned tag with only the submodules LVK needs (`submodules` in `bootstrap-deps.json`).
Dawn is built as a separate Release shared library (`webgpu_dawn`) with the D3D12, Metal and Vulkan backends, copied
next to the executables.

The backend prefers Dawn's Vulkan adapter: the D3D12 resource table needs DXC, which is not built. Dawn's
`maxDynamicUniformBuffersPerPipelineLayout artificially reduced` warning with llvmpipe is harmless.

## Web build

```bash
emcmake cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release   # Emscripten 7+
cmake --build . --target Tiny_MeshLarge_WebGPU          # samples/Tiny_MeshLarge_WebGPU.html
```

Emscripten builds only the WebGPU backend and its samples, with SDL3 and GLSL. `webgpu.h` comes from
[emdawnwebgpu](https://dawn.googlesource.com/dawn/+/HEAD/src/emdawnwebgpu/) of the same Dawn release.

- wasm64, wasm exceptions (spirv-cross), JSPI (blocking waits; `isReady()` and present yield to the browser) and threads.
  Serve the pages with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
- Pages preload their content. `Tiny_MeshLarge` needs a native run first for the mesh cache and `.ktx` files;
  `samples/web_preload_bistro.py` lists them (about 1 GB) when CMake configures.
- Bindless needs Chrome with `--enable-unsafe-webgpu`; `WebGPUEmscripten.js` provides the resource table API on top of it.
- No debug labels: emdawnwebgpu breaks on `WGPUStringView` by value with wasm64.

## Translator changes

The `lvk-webgpu` branch of the fork adds to the WGSL backend:

| Option | Effect |
|---|---|
| (always) | `BaseVertex`/`BaseInstance` and `gl_ViewIndex` read as 0 (WGSL indices already include the bases) |
| `use_resource_table` | bindless arrays become `getResource<T>(offset + index)` |
| `bda_arena_count`, ... | buffer device addresses become `vec2u` into `array<u32>` arenas |
| `push_constants_as_immediates` | push constants become `var<immediate>` |
| `push_constants_as_storage` | otherwise `var<storage, read>`, keeping std430 |

It also follows `OpCopyObject` in the depth-compare analysis and fixes `LocalSizeId` with SPIR-V 1.6.

Unsupported: atomics, 16/64-bit types, `uint64_t` pointer casts, `kSamplersYUV`, subpass inputs, storage images.
