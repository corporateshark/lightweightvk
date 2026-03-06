# Repository Guidelines

See `CLAUDE.md` for build commands, coding style, naming conventions, commit conventions, and CMake options.

## Project Structure & Modules
- `lvk/`: Core LightweightVK library (C++). Vulkan interop and backend in `lvk/vulkan/`.
- `samples/`: Desktop demo apps (headless-friendly flags available).
- `android/`: Templates for Android sample APKs (Gradle).
- `cmake/`, `CMakeLists.txt`: Build system.
- `third-party/`: External deps (populated by scripts).
- `.github/workflows/`: CI builds and headless run checks.

## Testing Guidelines
- No unit test framework; GitHub Actions CI validates builds and runs selected samples headless (.github/workflows/c-cpp.yml).
- Local sanity test: build, then run at least `001_HelloTriangle` and `005_MeshShaders` with `--headless` and capture a screenshot/log.
- Keep samples runnable on Linux/macOS/Windows; note Mac limitations (MoltenVK 1.4+, VulkanSDK 1.4.341+).

## Pull Requests
- Clear description, motivation, and platform notes; link issues if applicable.
- Include build/run commands used and (for samples) screenshots/logs.
- Ensure CI passes across matrices.

## Security & Configuration Tips
- Requirements: Vulkan SDK 1.4.341+ (CI), MoltenVK on macOS; Android needs `ANDROID_NDK`, `JAVA_HOME`, and `adb` in `PATH`.
- GPUs must support Vulkan 1.3 (see README for details).
