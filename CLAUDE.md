# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightweightVK is a bindless-only fork of IGL designed to run on Vulkan 1.3+ with optional mesh shaders and ray tracing support. It serves as a modern, minimalistic graphics API wrapper focused on rapid prototyping of Vulkan-based renderers.

## Build System and Commands

### Initial Setup
Before building, run the bootstrapping scripts to download dependencies:
```bash
python3 deploy_deps.py
python3 deploy_content.py
```
- `deploy_deps.py`: Clones/downloads third-party libraries (Vulkan headers, GLFW, GLM, ImGui, Tracy, Slang, etc.) into `third-party/deps/`. Driven by `third-party/bootstrap-deps.json`. Some libraries (Slang, screenshot tests) use predicates to allow CI jobs to skip them via environment variables (e.g., `LVK_WITH_SLANG=OFF`).
- `deploy_content.py`: Downloads sample assets (Bistro scene, solar system textures, glTF models, HDR skyboxes, etc.) into `third-party/content/`. Driven by `third-party/bootstrap-content.json`.

Both scripts invoke `third-party/bootstrap.py` which reads the corresponding JSON manifest and fetches git repos, archives, or individual files.

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
Requires VulkanSDK 1.4.341+
```bash
cd build
cmake .. -G "Xcode"
```

**Android:**
Requires Android Studio, ANDROID_NDK, JAVA_HOME, and adb in PATH
```bash
cd build
cmake .. -DLVK_WITH_SAMPLES_ANDROID=ON
cd android/Tiny  # or any other sample
./gradlew assembleDebug
```

For Android devices: `python3 deploy_content_android.py`

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
2. **Windows - MSVC 2022** — Debug build with Tracy enabled, no screenshot tests
3. **Ubuntu - Clang** — two configs: default (X11) and Wayland, Debug builds, no screenshot tests
4. **Ubuntu - Clang (screenshot tests)** — Debug build with `LVK_DEPLOY_SCREENSHOT_TESTS=ON`, runs samples headless at 1280×720, captures screenshots, then compares against reference images using `compare_screenshots.py` (threshold 1.0). Logs and screenshots are uploaded as artifacts (3-day retention)
5. **macOS - Clang (Xcode)** — Debug build, Tracy disabled, no screenshot tests

All jobs use Vulkan SDK 1.4.341.0 and cache `third-party/deps` keyed on `bootstrap-deps.json` hash.

To check CI status: `gh run list` or `gh run view <run-id>`.

### CMake Configuration Options
- `LVK_DEPLOY_DEPS`: Deploy dependencies via CMake (default: ON)
- `LVK_WITH_GLFW`: Enable GLFW (default: ON)
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
- `LVK_DEPLOY_SCREENSHOT_TESTS`: Deploy screenshot tests (default: ON)

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
- Use `const` on local variables whenever possible
- Use `if (ptr)` instead of `if (ptr != nullptr)` for pointer checks
- No STL containers in public API; the only exception is `std::vector` which is allowed in `.cpp` files and samples
- Use `()` after function names in code comments and commit messages (e.g., `// call doSomething() first`)
- **NEVER use `auto` except for lambda types** — always use explicit types

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
  - `Pool.h`: Resource management utilities

### Key Design Principles
1. **Bindless-only**: Utilizes Vulkan 1.3+ dynamic rendering, descriptor indexing, and buffer device address
2. **Minimal API surface**: No STL containers in public API
3. **Ray tracing integration**: Fully integrated with bindless design
4. **Cross-platform**: Windows, Linux, macOS (via MoltenVK or KosmicKrisp), Android

### Sample Applications
Located in `samples/`, 17 demos covering:
- Basics: `001_HelloTriangle`, `002_RenderToCubeMap`, `003_RenderToCubeMapSinglePass`, `004_YUV`
- Advanced rendering: `005_MeshShaders`, `006_SwapchainHDR`, `007_DynamicRenderingLocalRead`, `008_MeshShaderFireworks`, `009_TriplanarMapping`, `010_OmniShadows`
- Ray tracing: `RTX_001_Hello`, `RTX_002_AO`, `RTX_003_Pipeline`, `RTX_004_Textures`
- Complex demos: `DEMO_001_SolarSystem`, `DEMO_002_Bistro`, `Tiny_MeshLarge`

### Android Project Generation
Android sample projects are generated by CMake (`ADD_DEMO()` macro in `samples/CMakeLists.txt`) into `build/android/<app>/`. The system uses two layers:
- **`android/lvklib/`**: Shared Android library module containing `LvkActivity` base class (fullscreen/immersive mode, storage permissions), styles, and data extraction rules. Not copied — referenced in-place from generated projects via a relative path in `settings.gradle`.
- **`android/app/`**: Minimal per-app templates (`.in` files) that only substitute `@APP_NAME@` and `@ANDROID_ABI_FILTERS@`. `MainActivity.java.in` just extends `LvkActivity` and loads the native library.

When modifying shared Android behavior (permissions, fullscreen logic, themes), edit `android/lvklib/`. When modifying per-app scaffolding (package name, native library wiring), edit the `.in` templates in `android/app/`.

### Common Development Patterns
- All samples use `VulkanApp` base class (`samples/VulkanApp.h`)
- Platform abstraction through preprocessor macros
- Resource management via LVK handles and holders
- Tracy profiler integration when enabled

### Dependencies
- Vulkan 1.3+ (required)
- GLFW (desktop platforms)
- GLM (math library)
- ImGui (UI)
- Tracy (optional profiling)
- Various third-party libraries managed via bootstrap scripts

### Vulkan Interop
LVK provides helper functions in `lvk/vulkan/VulkanUtils.h` to access underlying Vulkan objects, enabling mixing of LVK and raw Vulkan API calls.
