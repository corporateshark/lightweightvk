# Repository Guidelines

## Project Structure & Modules
- `lvk/`: Core LightweightVK library (C++). Vulkan interop and backend in `lvk/vulkan/`.
- `samples/`: Desktop demo apps (headless-friendly flags available).
- `android/`: Templates for Android sample APKs (Gradle).
- `cmake/`, `CMakeLists.txt`: Build system.
- `third-party/`: External deps (populated by scripts).
- `.github/workflows/`: CI builds and headless run checks.

## Build, Test, Develop
- Bootstrap deps: `python3 deploy_content.py && python3 deploy_deps.py`.
- Configure (Unix Makefiles): `cmake -S . -B build -G "Unix Makefiles"`.
- macOS (Xcode): `cmake -S . -B build -G "Xcode"`.
- Windows (VS2022): `cmake -S . -B build -G "Visual Studio 17 2022"`.
- Build: `cmake --build build --parallel`.
- Android (generate Gradle projects): `cmake -S . -B build -DLVK_WITH_SAMPLES_ANDROID=ON`.
- Assemble APK (example): `cd build/android/009_TriplanarMapping && ./gradlew assembleDebug`.
- Quick run (headless): `./build/samples/001_HelloTriangle --headless --screenshot-frame 1 --screenshot-file out.png`.

## Coding Style & Naming
- C/C++: enforced by `.clang-format` (2‑space indent, width 140, no tabs, sorted includes, left‑aligned pointers). Apply via `clang-format -i`.
- CMake: `.cmake-format` (2‑space indent, canonical command case). Run `cmake-format -i` if available.
- Naming: Types/structs `PascalCase` (e.g., `Result`, `Viewport`); enums `EnumName_Value`; functions `lowerCamelCase` (e.g., `getVertexFormatSize`); macros `LVK_*`.

## Testing Guidelines
- No unit test framework; CI validates builds and runs selected samples headless.
- Local sanity test: build, then run at least `001_HelloTriangle` and `005_MeshShaders` with `--headless` and capture a screenshot/log.
- Keep samples runnable on Linux/macOS/Windows; note Mac limitations (MoltenVK 1.3+).

## Commit & Pull Requests
- Commits: imperative, concise, start with capital; optional scope prefix (e.g., `macOS:`, `Samples:`, `Android:`). Examples: `Update README`, `Fix shader stage flags`.
- PRs: clear description, motivation, and platform notes; link issues if applicable; include build/run commands used and (for samples) screenshots/logs. Ensure CI passes across matrices.

## Security & Configuration Tips
- Requirements: Vulkan SDK 1.4.309+ (CI), MoltenVK on macOS; Android needs `ANDROID_NDK`, `JAVA_HOME`, and `adb` in `PATH`.
- GPUs must support Vulkan 1.3 (see README for details).

