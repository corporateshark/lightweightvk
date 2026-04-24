# Contributing Guidelines

## Before opening an issue

- Search existing issues (including closed ones) for similar reports.
- If you hit a crash or assert, attach a debugger callstack. Enable the most recent Vulkan Validation Layers and include their output.
- Provide a minimal repro: ideally a small patch against one of the `samples/` apps that reproduces the problem.
- Include: OS, GPU and driver version, Vulkan SDK version, build configuration, and the commit hash you are on.
- Attach screenshots or short videos for visual issues.

## Opening a pull request

- Create a dedicated branch per PR. Additional commits pushed to that branch will appear in the PR.
- When adding a feature, describe the usage context and why you need it.
- When fixing a compile warning or build failure, include the compiler log, version, and platform.
- Attach screenshots or videos for visual changes.

### Coding style

Follow the conventions documented in [`CLAUDE.md`](../CLAUDE.md) and enforced by [`.clang-format`](../.clang-format):

- 2-space indent, 140-column limit, no tabs, left-aligned pointers.
- Types: `PascalCase`. Functions: `lowerCamelCase()`. Enums: `EnumName_Value`. Macros: `LVK_*`.
- Use C++20 designated initializers, `const` on locals where possible, explicit types (no `auto` except for lambdas), early exits over nested `if`.
- No STL containers in the public API (`std::vector` is allowed in `.cpp` files and samples).

Run `clang-format -i <file>` before submitting.

### Commits

- Start with a capital letter, no trailing period, past tense (`Added`, `Fixed`, `Updated`).
- Optional scope prefix: `Samples:`, `Android:`, `CMake:`, `GitHub:`, `ImGui:`, etc.
- Changes touching only `CMakeLists.txt` or `cmake/` must use `CMake:`. After a prefix, the next word is lowercase.
- Wrap code identifiers in backticks; include `()` on function names.
- Reference GitHub issues when relevant (e.g. `(#64)`, `(fixed #63)`).

## License

LightweightVK is distributed under the [MIT License](../LICENSE.md). By submitting a PR you confirm the code is yours to contribute
and agree to license it under the same terms. Do not modify existing copyright headers.
