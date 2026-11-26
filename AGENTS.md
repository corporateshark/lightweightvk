# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source Control Rules

- **NEVER** commit, amend, or submit diffs unless explicitly asked to do so.

## Project Overview

LightweightVK is a bindless-only fork of IGL designed to run on Vulkan 1.3+ with optional mesh shaders and ray tracing support. It serves as a modern, minimalistic graphics API wrapper focused on rapid prototyping of Vulkan-based renderers.

## Build System and Commands

### Initial Setup
Before building, run the bootstrapping scripts to download dependencies:
```bash
python3 deploy_deps.py
python3 deploy_content.py
```
- `deploy_deps.py`: Clones/downloads third-party libraries (Vulkan headers, GLFW, GLM, ImGui, Tracy, Slang, etc.) into `third-party/deps/`. Driven by `third-party/bootstrap-deps.json`. Some libraries (Slang, screenshot tests) use Python predicates to allow CI jobs to skip them via environment variables (e.g., `LVK_WITH_SLANG=OFF`).
- `deploy_content.py`: Downloads sample assets (Bistro scene, solar system textures, glTF models, HDR skyboxes, etc.) into `third-party/content/`. Driven by `third-party/bootstrap-content.json`.

Both scripts invoke `third-party/bootstrap.py` which reads the corresponding JSON manifest and fetches git repos, archives, or individual files.

- `third-party/content/src/neural/dfaoit/train_dfaoit.py`: trains a neural OIT network of `ARM_002_NeuralOIT` and writes `dfaoit-<hidden1>x<hidden2>.bin` next to it, the integer weights the sample builds its TOSA graph from at start-up (`--hidden1/--hidden2` pick the topology; the defaults train the deployed `16x16`, the only one the sample loads). The network ships quantized to 8 bits, after a few epochs of quantization-aware fine-tuning, as the TOSA integer graph (int8 CONV2D into int32, per-channel RESCALE, a TABLE for the sigmoid) an NPU runs natively; the script evaluates that graph bit for bit and prints its validation error next to the float one's. Training data is synthetic and needs no renderer or scene file: `dfaoit_scenes.py` next to it builds scenes of spheres, boxes, quads and tori and fires tiles of orthographic rays through them; `--cache` keeps the sampled pixels for the next run; `third-party/content/src/neural/tosa_int8.py` has the TOSA integer arithmetic (RESCALE and its `scale32` multipliers). Their docstrings have the details.
- `third-party/content/src/neural/nlg/train_nlg.py`: the Neural Light Grid generator of `ARM_003_NeuralGI`: `generate` turns the support bakes of the GDS renderer (`--gi-nlg-bake`, per probe the visibility of 8192 support points and the lighting at 256 of them) into weighting-function sets, `train` fits the generator (PCA-512 of a function into a 6-layer 512-wide LeakyReLU MLP with skip connections, trained end to end against the function through the 3-8-8-1 network it writes) on those sets with held-out scenes for validation, `quantize` turns it, PCA included, into the int8 TOSA graph (calibrated per channel, optionally fine-tuned with the rounding in the loop, evaluated bit for bit; the graph keeps a fingerprint of the float weights it came from), and `export` writes `nlg.bin` next to it (the point pattern, the float layers, the output scale and the int8 layers the sample builds its TOSA graph from). The sample computes the functions itself at run time with the same algorithm; the generator is the only trained part. Needs PyTorch; its docstring has the details.

### Platform-Specific Build Commands

**Windows:**
```bash
cd build
cmake .. -G "Visual Studio 18 2026"
```

**Linux:**
```bash
sudo apt-get install clang xorg-dev libxinerama-dev libxcursor-dev libgles2-mesa-dev libegl1-mesa-dev libglfw3-dev libglew-dev libstdc++-12-dev extra-cmake-modules libxkbcommon-x11-dev wayland-protocols
cd build
cmake .. -G "Unix Makefiles"
```

For Wayland: `cmake .. -G "Unix Makefiles" -DLVK_WITH_WAYLAND=ON`

**macOS:**
Requires VulkanSDK 1.4.357+
```bash
cd build
cmake .. -G "Xcode"
```

**Android:**
Requires Android Studio, ANDROID_NDK, JAVA_HOME, and adb in PATH
```bash
cd build
cmake .. -DLVK_WITH_SAMPLES_ANDROID=ON
cd android/001_HelloTriangle  # or any other sample
./gradlew assembleDebug
```

For Android devices: `python3 deploy_content_android.py` (or `--obb <package>...` to push the content as the OBB of each given sample instead of the shared `$EXTERNAL_STORAGE/LVK`, or configure with `-DLVK_ANDROID_OBB_CONTENT=ON` and build the `LVKAndroidContent` target, which installs the same archive as the OBB of every generated demo)

### Building
```bash
cmake --build build --parallel
```

### Running and Testing
No unit test framework. Verify changes by building and running samples headless:
```bash
./build/samples/001_HelloTriangle --headless --screenshot-frame 1 --screenshot-file out.png
```

### Screenshot Tests
CI runs screenshot comparison tests (enabled by default via `LVK_DEPLOY_SCREENSHOT_TESTS`). Reference images, frame numbers, and the comparison script are in a [separate repository](https://github.com/corporateshark/lightweightvk_screenshot_tests) — see its [README](https://github.com/corporateshark/lightweightvk_screenshot_tests/blob/master/README.md) for details.

### CI (GitHub Actions)
Workflow file: `.github/workflows/c-cpp.yml`. Runs on every push and PR to any branch. Four jobs:

1. **Android (Ubuntu)** — generates Android projects with Ninja + NDK r29, assembles APKs for a subset of samples
2. **Build matrix** — Debug builds, no screenshot tests: Windows - MSVC 2026 (GLFW and SDL3), Ubuntu - Clang (GLFW, SDL3 and Wayland), Ubuntu - GCC (SDL3)
3. **macOS - Clang (Xcode)** — Debug build, Tracy disabled, no screenshot tests
4. **Ubuntu - Clang (screenshot tests)** — Debug build with `LVK_DEPLOY_SCREENSHOT_TESTS=ON`, runs samples headless at 1280×720, captures screenshots, then compares against reference images using `compare_screenshots.py` (threshold 1.0). Logs and screenshots are uploaded as artifacts (3-day retention)

All jobs cache `third-party/deps` keyed on `bootstrap-deps.json` hash; every job except Android uses Vulkan SDK 1.4.350.0.

To check CI status: `gh run list` or `gh run view <run-id>`.

### CMake Configuration Options
- `LVK_DEPLOY_DEPS`: Deploy dependencies via CMake (default: ON)
- `LVK_WITH_GLFW`: Enable GLFW (default: ON)
- `LVK_WITH_SDL3`: Enable SDL3 (default: OFF)
- `LVK_WITH_SAMPLES`: Enable sample demo apps (default: ON)
- `LVK_WITH_SAMPLES_ANDROID`: Generate Android projects for demo apps (default: OFF)
- `LVK_WITH_TRACY`: Enable Tracy profiler (default: ON)
- `LVK_WITH_TRACY_GPU`: Enable Tracy GPU profiler (default: OFF)
- `LVK_WITH_WAYLAND`: Enable Wayland on Linux (default: OFF)
- `LVK_WITH_IMPLOT`: Enable ImPlot (default: ON)
- `LVK_WITH_OPENXR`: Enable OpenXR (default: OFF)
- `LVK_WITH_ANDROID_VALIDATION`: Enable validation layers on Android (default: ON)
- `LVK_WITH_MINILOG`: Enable Minilog (default: ON)
- `LVK_WITH_SLANG`: Enable Slang compiler (default: OFF)
- `LVK_WITH_SPIRV_OPT`: Run SPIRV optimization on shaders (default: OFF)
- `LVK_WITH_RAW_VULKAN`: Enable raw Vulkan interop (default: OFF)
- `LVK_DEPLOY_SCREENSHOT_TESTS`: Deploy screenshot tests (default: ON)
- `LVK_IMGUI_EXTERNAL`: Use external ImGui library (default: OFF)
- `LVK_IMPLOT_EXTERNAL`: Use external ImPlot library (default: OFF)
- `LVK_GLSLANG_EXTERNAL`: Use external glslang and SPIRV-Tools (default: OFF)
- `LVK_SPIRV_REFLECT_EXTERNAL`: Use external SPIRV-Reflect library (default: OFF)
- `LVK_ENABLE_LONG_PATHS_WIN32`: Enable long paths on Windows (default: ON)
- `LVK_ANDROID_ABI`: Enabled ABI on Android (cache string, default: `arm64-v8a`)
- `LVK_ANDROID_OBB_CONTENT`: Pack the sample content into an OBB for every generated Android demo, through the `LVKAndroidContent` target (default: OFF). The archive is the same for all of them, only the OBB name differs. Left off, the samples expect `python deploy_content_android.py` to have pushed it to `$EXTERNAL_STORAGE/LVK` by hand
- `LVK_WITH_TENSORS`: Enable the machine learning helpers: the Arm VGF library and `lvk/HelpersVgf.h` (`lvk::VgfModel`, loads `.vgf` models into data graph pipelines) (default: ON). Shaders never access tensors directly (no `tensorARM`): `createTensorBufferAliasing()` puts a linear tensor on a buffer (shaders use its device address, `buffer_reference`; `getTensorBufferSize()` is how large the buffer has to be, as drivers round tensor memory up to its alignment), `createTensorTextureAliasing()` puts an optimal tensor on a `TextureUsageBits_TensorAliasing` texture (shaders sample or store it). Where linear tensors cannot feed a data graph, a buffer-aliasing tensor gets an internal optimally tiled tensor that data graph pipelines are created for, and `cmdDispatchDataGraph()` copies inputs into it before the dispatch and outputs out of it after
- `LVK_WITH_ML_EMULATION_LAYER`: Download and build the Arm ML Emulation Layer for Vulkan (`VK_LAYER_ML_Graph_Emulation`, `VK_LAYER_ML_Tensor_Emulation`) as an ExternalProject to emulate `VK_ARM_tensors`/`VK_ARM_data_graph` on GPUs without native support (default: OFF). It shares glslang, SPIRV-Headers, SPIRV-Tools and Vulkan-Headers with LVK; the layer manifests are installed into `build/ml-emulation-layer/bin` (Windows) or `build/ml-emulation-layer/share/vulkan/explicit_layer.d`. With `LVK_WITH_SAMPLES_ANDROID` the layer is also cross-built for every `LVK_ANDROID_ABI` (needs `ninja` and the NDK's `llvm-strip`) into `build/ml-emulation-layer-android/jniLibs`, which the generated projects list in `jniLibs.srcDirs`. Samples request the layers via `ContextConfig::layers`; they are always used, even on GPUs supporting `VK_ARM_tensors` natively; `VulkanApp` adds the built layer to `VK_ADD_LAYER_PATH`, and `VulkanContext` skips requested layers that are not installed; `--no-ml-emulation` drops them

## Coding Style

### Formatting
- Enforced by `.clang-format`: 2-space indent, 140 column limit, no tabs, sorted includes, left-aligned pointers
- Apply via `clang-format -i <file>`
- CMake files: `.cmake-format` (2-space indent, canonical command case)

### Naming
- Types/structs: `PascalCase` (e.g., `Result`, `Viewport`)
- Enums: `EnumName_Value`
- Functions: `lowerCamelCase` (e.g., `getVertexFormatSize()`)
- Macros: `LVK_*`

### C++ Conventions
- Use C++20 designated initializers whenever possible (e.g., `lvk::RenderPass{.color = {...}}`)
- Use `if (ptr)` instead of `if (ptr != nullptr)` for pointer checks
- Use `if (value)` instead of `if (value != 0)` for integer checks
- Use `if (handle)` instead of `if (handle != VK_NULL_HANDLE)` or `if (handle != XR_NULL_HANDLE)` for Vulkan/OpenXR handle checks
- No STL containers in public API; the only exception is `std::vector` which is allowed in `.cpp` files and samples
- Use `()` after function names in code comments and commit messages (e.g., `// call doSomething() first`)
- Instead of deep nested if-blocks, prefer early exit
- `const char* debugName` (or `debugName_`) is always the last member field of a struct; add new fields before it
- **Use `const` on local variables whenever possible**
- **NEVER use `auto` except for lambda types** — always use explicit types
- **NEVER use `goto`**
- `debugName` is the second-to-last function parameter, `outResult` the last
- Name functions object-first: `createTensorBufferAliasing()`, not `createBufferAliasingTensor()`
- No new STL headers
- Fill structs with one `const` designated initializer, using ternaries for conditional fields; don't patch fields afterwards
- Name compound conditions: `const bool isMisaligned = ...`
- No vendor suffixes in local names: `tensor`, not `tensorARM`

## Commit Conventions
- Start with capital letter, no trailing period
- Use past tense (e.g., "Added", "Fixed", "Updated", "Replaced", "Removed")
- Optional scope prefix: `Samples:`, `Android:`, `CMake:`, `GitHub:`, `ImGui:`, `HelpersImGui:`, etc.
- Changes touching only `CMakeLists.txt` or `cmake/` files must use the `CMake:` prefix
- When a scope prefix is used, the first letter after `:` should be lowercase (e.g., `GitHub: added ...`)
- Use backticks around code identifiers: functions with `()`, types, extensions, macros
- Reference GitHub issues when applicable (e.g., `(#64)`, `(fixed #63)`)

## Architecture Overview

### Core Components
- **LVK Library** (`lvk/`): Main graphics API abstraction layer
  - `LVK.h/cpp`: Core API definitions and implementations
  - `vulkan/`: Vulkan-specific backend implementation
  - `HelpersImGui.h/cpp`: ImGui integration helpers
  - `HelpersVgf.h/cpp`: `lvk::VgfModel` (built with `LVK_WITH_TENSORS`) parses VGF models from memory, e.g. `VulkanApp::loadFile()`, and creates their `VK_ARM_data_graph` pipelines

### Key Design Principles
1. **Bindless-only**: Utilizes Vulkan 1.3+ dynamic rendering, descriptor indexing, and buffer device address
2. **Minimal API surface**: No STL containers in public API
3. **Ray tracing integration**: Fully integrated with bindless design
4. **Cross-platform**: Windows, Linux, macOS (via KosmicKrisp), Android

### Sample Applications
Located in `samples/`, 20 demos covering:
- Basics: `001_HelloTriangle`, `002_RenderToCubeMap`, `003_RenderToCubeMapSinglePass`, `004_YUV`
- Advanced rendering: `005_MeshShaders`, `006_SwapchainHDR`, `007_DynamicRenderingLocalRead`, `008_MeshShaderFireworks`, `009_TriplanarMapping`, `010_OmniShadows`, `011_VariableRateShading`
- Ray tracing: `RTX_001_Hello`, `RTX_002_AO`, `RTX_003_Pipeline`, `RTX_004_Textures`
- Complex demos: `DEMO_001_SolarSystem`, `Tiny_MeshLarge`
- Multiview rendering examples: `010_OmniShadows`, `DEMO_001_SolarSystem`
- Neural graphics (`VK_ARM_tensors` + `VK_ARM_data_graph`; shaders write and read tensors through `buffer_reference` to the buffers they alias and through the textures they alias): `ARM_001_NeuralSuperSampling` runs Arm Neural Super Sampling (at half the size of the window, which opens at the size of the monitor or `--width`/`--height` with the shorthands `--1080p` and `--4k`, upscaled 2x to it and presented 1:1, with the same network, whose TOSA graph the sample reshapes at start-up) on the Bistro scene (mesh and textures through `samples/Bistro.h`); its compute entry points are compiled into the sample and the algorithm comes from the `neural-graphics-sdk` dependency (a fork of the Arm Neural Graphics SDK that binds NSS resources bindlessly, pulled in at run time by a small `#include` resolver); the graph writes its temporal feedback straight into a texture through `createTensorTextureAliasing()` where int8 tensors can alias textures, and otherwise into a buffer-aliasing tensor that a compute pass copies into the texture (the Arm ML Emulation Layer does not support aliasing optimal tensors); the network comes from the `neural/nss` entry of `bootstrap-content.json` (downloaded into `third-party/content/src/neural/nss/`); the same network also runs as int8 `VK_KHR_cooperative_matrix` tiles with no `VK_ARM_*` extension, bit for bit what the data graph computes: `buildCooperativeLayers()` turns the reshaped graph into one compute dispatch per CONV2D (with the RESCALE and TABLE after it and the 2x nearest RESIZE or the CONCAT before it folded in), each subgroup staging the input rows of its tiles of 16 pixels by 4 rows itself (`--nss-cm` starts with it; both paths are in the UI, each grayed out without the extensions it runs on)
- Order-independent transparency: `ARM_002_NeuralOIT` draws the whole Bistro transparent (mesh and textures through `samples/Bistro.h`) with three techniques (weighted blended at start-up, the cheapest): an A-buffer reference (the K nearest fragments per pixel, a lock-free sorted insertion of 64-bit depth+RGBA8 keys, evicted fragments tail-blended), weighted blended OIT (McGuire and Bavoil, JCGT 2013; one geometry pass, additive weighted accumulation plus a multiplicative revealage target, resolved in compute), and DFAOIT (Tsopouridis et al., CGF 2024), whose network is dispatched as a `VK_ARM_data_graph` pipeline the sample builds at start-up (`TosaGraphBuilder` of `samples/Bistro.h` writes the SPIR-V of its TOSA graph out of the weights); the same integer network also runs as int8 `VK_KHR_cooperative_matrix` tiles (`--technique dfaoit-cm`; both DFAOITs are in the UI, each grayed out without the extensions it runs on, and an unavailable one falls back on the other) in one compute pass from the features to the resolved colour, bit for bit what the data graph computes, every invocation a pixel and a row of the A matrices, a subgroup multiplying only its own rows so that none waits for another (no tile is skipped: deciding it across subgroups does not hold on every driver, and the matrices are cheap) (`getCooperativeMatrixShape()` of `samples/Bistro.h` picks the tile shape). `--compare` reports the mean squared error against the A-buffer, `--diff` shows it, A "GPU time" window always shows the GPU time per stage with a sparkline of its history, and `--timings` also prints it to the log. It renders at the size of the monitor (`--width`/`--height` override it), and the graph, which needs a fixed shape, is built for tensors as wide as the window with room for 3/4 of its pixels (1170x408 on Android); the network comes from `third-party/content/src/neural/dfaoit/train_dfaoit.py`, and the deployed one, `16x16`, is checked in next to it as `dfaoit-16x16.bin` (its `README.md` says how it was trained and how to train another). It keeps the k nearest fragments of a pixel exactly (the paper's DFAOIT_k, k = 4, read back out of the network as `inputs - 10`) and predicts only the tail behind them, which is what beats the authors' own weights on Bistro by an order of magnitude. A pixel with no more than k fragments has no tail to ask for, so the pack pass leaves it out of the input tensor and compacts the rest into the rows of the tensor (a workgroup-aggregated counter hands out the rows, the resolve pass reads each pixel's row back out of `slotsBuffer`, and a pixel that does not fit falls back to compositing the mean of its tail analytically)
- Global illumination: `ARM_003_NeuralGI` lights Bistro (or `--scene interior|sponza`) with the indirect diffuse light of a Neural Light Grid (Iwanicki et al., Advances in Real-Time Rendering 2024): `--probe-levels` lattices of probes (default 2), the first `--probe-spacing` apart and every next one half as far and only near geometry, where every probe gets a weighting function over 8192 support points around it (trilinear where the probe sees, diffused through the visibility graph elsewhere, points inside geometry excluded, probes inside geometry started from the largest connected lit cluster), a generator network turns the function's first 512 PCA coefficients into the 113 coefficients of the probe's own 3-8-8-1 network, and the probe's lighting is integrated over its support (256 k-means points lit in four passes over the lattice whenever the sun moves, summed with their shares, then left alone; the lighting follows the bake order over the probes baked so far and gets as many probes per frame as fit into 4 ms of GPU time (aiming at 3/4 of it), measured with timestamps read back a few frames later, and until a pixel's probes are lit it gets the constant sky ambient instead of their light); a pixel blends the 8 probes of its cell on the finest level that weights it with their networks (coefficients packed to f16) 0.2 spacings off the surface. The whole bake runs in the sample over batches of probes (33 ray queries per support point; probes whose cube of one spacing touches no triangle skip the visibility rays and the generator and copy a reference probe, and once the textures are in, the geometry without transparent texels is opaque in the BLASes); only the generator's weights are a file: `third-party/content/src/neural/nlg/nlg.bin`, which `deploy_content.py` downloads as a version 2 file: the support point pattern and the float layers, which the generator runs as in compute. A version 3 file, as `export` writes it, adds the generator with its PCA as int8 layers, which the sample then builds into a TOSA graph (int8 1x1 CONV2D into int32 with per-channel RESCALE, LeakyReLU as a TABLE, the skip connections summed in an int32 stream, int32 output scaled by a compute pass; `buildGeneratorGraph()` through `TosaGraphBuilder`) dispatched as a `VK_ARM_data_graph` pipeline over 256 probes at a time. Without the data graph the float layers also run as f16 `VK_KHR_cooperative_matrix` tiles accumulated in f32, per layer over the probes of a batch, partial sums of 2x2 tiles over slices of the inputs and a pass that sums them and finishes the layer (about 7x faster than the compute shader on an RTX 3070 Ti, its networks within the f16 precision they are packed to anyway); the UI has all three generators, each grayed out without what it runs on, switching bakes the probes again, and `--generator compute|data-graph|cm` picks one at start-up (by default the data graph, else the compute shader: the cooperative matrices only run when chosen, as a driver may expose them without matrix hardware, as on an AMD RDNA2 iGPU, where they take seconds per batch). `train_nlg.py` next to it trains the generator on the support bakes of the GDS renderer (148 glTF scenes), quantizes it (`export` writes a graph only next to the float weights it was quantized from) and exports them. The baked probes (networks, support shares, flags) go to `nlg_<scene>.bake` in the content folder, keyed by the lattice, the mesh, the generator (and which of the three runs it) and the sun, and the next run loads them and only lights them (`--rebake` ignores the file). Textures are off at first (the UI has a GI and a Textures checkbox, G and T toggle them, `--textures` starts with them) and stream in the first time they are enabled, as in `RTX_002_AO`: until then the scene is baked untextured (MTL colors, no alpha testing, fully opaque BLASes, `nlg_<scene>_untextured.bake`), and once they are in it is split into opaque and alpha-tested BLASes and baked again with them, the untextured probes (their networks and lighting; a probe's lighting update count lives in its SH, not in the flags the bake writes) lighting the scene until the textured ones replace them, and the lighting averaging into the untextured lighting (which already has the bounces the first passes of a restart build up) rather than starting over. Shading is PBR with the `map_Ks` (specular color, gloss in alpha) and `map_bump` maps the MTL names, loaded through `loadMaterialTextures(app, dir, true)` into `GPUMaterial::texSpecular` and `GPUMaterial::texNormal`: metallic-roughness after the BRDFs of the GDS renderer (glTF-Sample-Renderer): GGX sun, the probes' irradiance around the normal for the diffuse part and around the reflection for the specular part (split-sum with multiple-scattering Fresnel). `materialModel()` picks a model per material by its name: metals (metallic solved from the specular map), cloth (`fabric`, `curtain`: a Charlie sheen lobe of roughness 0.5 whose albedo is an analytic fit), foliage (diffuse transmission, the back side lit by the probes' irradiance around -N and by a shadow ray through the leaf, a quarter of the dielectric specular), and dielectrics for the rest. The direct sunlight is one shadow ray per pixel. Exposure is automatic (log-average luminance of a 64x36 grid mapped to 0.18, adapting with a time constant of 0.3 s; a UI slider shifts it in EV; `--no-auto-exposure` and the UI switch to a manual slider). A "GPU time" window always shows the GPU time per stage with a sparkline of its history (ImPlot, as in `Tiny_MeshLarge`), and `--timings` also prints it to the log; the bake and the lighting run over frames behind a progress bar. It renders at the size of the monitor (`--width`/`--height` override it, as in `RTX_002_AO`): only the G-buffer and the screen-space passes depend on the resolution, the probes, their networks and the generator graph (256 probes per dispatch) do not

### Android Project Generation
Android sample projects are generated by CMake (`ADD_DEMO()` macro in `samples/CMakeLists.txt`) into `build/android/<app>/`. The system uses two layers:
- **`android/lvklib/`**: Shared Android library module containing `LvkActivity` base class (fullscreen/immersive mode, storage permissions), styles, and data extraction rules. Not copied — referenced in-place from generated projects via a relative path in `settings.gradle`.
- **`android/app/`**: Minimal per-app templates (`.in` files) that only substitute `@APP_NAME@` and `@ANDROID_ABI_FILTERS@`. `MainActivity.java.in` just extends `LvkActivity` and loads the native library.

When modifying shared Android behavior (permissions, fullscreen logic, themes), edit `android/lvklib/`. When modifying per-app scaffolding (package name, native library wiring), edit the `.in` templates in `android/app/`.

### Common Development Patterns
- All samples use `VulkanApp` base class (`samples/VulkanApp.h`)
- Samples that render Bistro/Sponza share `samples/Bistro.h`: the mesh cache (`loadAndCache()`, `loadFromCache()`, `vertexData_`, `indexData_`, `cachedMaterials_`) and the asynchronous BC7 material texture loader (`loadMaterialTextures()`, `processLoadedMaterialTextures()`, `cancelLoadingMaterialTextures()`, `materials_`); the neural demos also share `GPUStageTimer` (with its "GPU time" window, `drawWindow()`), `createTensorWithBuffer()`, `getCooperativeMatrixShape()`, `getClipRotation()` and the GLSL snippets `BISTRO_GLSL` and `FULLSCREEN_TRIANGLE_GLSL`, `ARM_002_NeuralOIT` and `ARM_003_NeuralGI` its `TosaGraphBuilder`, `ARM_001_NeuralSuperSampling` and `ARM_003_NeuralGI` its BLAS/TLAS builder (`createSceneAccelStructs()`), and `ARM_001_NeuralSuperSampling` and `Tiny_MeshLarge` its skybox loader (`loadSkyboxTextures()`)
- Platform abstraction through preprocessor macros
- Resource management via LVK handles and holders
- Tracy profiler integration when enabled

### Dependencies
- Vulkan 1.3+ (required)
- GLFW (desktop platforms)
- SDL3 (desktop platforms, alternative to GLFW via `LVK_WITH_SDL3`)
- GLM (math library)
- ImGui (UI)
- Tracy (optional profiling)
- Various third-party libraries managed via the `third-party/bootstrap.py` script and listed in `third-party/bootstrap-deps.json`
- Content files managed via the `third-party/bootstrap.py` script and listed in `third-party/bootstrap-content.json`

### Vulkan Interop
`lvk/vulkan/VulkanUtils.h` provides helpers (`getVkDevice()`, `getVkCommandBuffer()`, `getVkImage()`, ...) to access
the underlying Vulkan objects, enabling mixing of LVK and raw Vulkan API calls. Building with `LVK_WITH_RAW_VULKAN=ON` additionally
pulls the Vulkan headers into `lvk/LVK.h` and exposes conversion operators such as `ICommandBuffer::operator VkCommandBuffer()`.
