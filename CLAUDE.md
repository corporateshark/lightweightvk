# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightweightVK is a bindless-only fork of IGL designed to run on Vulkan 1.3+ with optional mesh shaders and ray tracing support. It serves as a modern, minimalistic graphics API wrapper focused on rapid prototyping of Vulkan-based renderers.

## Build System and Commands

### Initial Setup
Before building, run the deployment scripts to download dependencies:
```bash
python3 deploy_content.py
python3 deploy_deps.py
```

### Platform-Specific Build Commands

**Windows:**
```bash
cd build
cmake .. -G "Visual Studio 17 2022"
```

**Linux:**
```bash
sudo apt-get install clang xorg-dev libxinerama-dev libxcursor-dev libgles2-mesa-dev libegl1-mesa-dev libglfw3-dev libglew-dev libstdc++-12-dev extra-cmake-modules libxkbcommon-x11-dev wayland-protocols
cd build
cmake .. -G "Unix Makefiles"
```

For Wayland: `cmake .. -G "Unix Makefiles" -DLVK_WITH_WAYLAND=ON`

**macOS:**
Requires VulkanSDK 1.4.321+
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

### CMake Configuration Options
- `LVK_WITH_SAMPLES`: Enable sample demo apps (default: ON)
- `LVK_WITH_TRACY`: Enable Tracy profiler (default: ON)
- `LVK_WITH_TRACY_GPU`: Enable Tracy GPU profiler (default: OFF)
- `LVK_WITH_WAYLAND`: Enable Wayland on Linux (default: OFF)
- `LVK_WITH_SLANG`: Enable Slang compiler (default: OFF)

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
4. **Cross-platform**: Windows, Linux, macOS (via MoltenVK), Android

### Sample Applications
Located in `samples/`, includes:
- Basic rendering (`001_HelloTriangle.cpp`)
- Cube mapping (`002_RenderToCubeMap.cpp`)
- Mesh shaders (`005_MeshShaders.cpp`)
- Ray tracing demos (`RTX_*.cpp`)
- Complex demos (`DEMO_001_SolarSystem.cpp`)

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