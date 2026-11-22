/*
 * LightweightVK
 *
 * Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 Helper functions to load and cache Bistro/Sponza meshes:

   bool loadAndCache(VulkanApp& app, const char* cacheFileName, const char* modelFileName)
   bool loadFromCache(VulkanApp& app, const char* cacheFileName)

 The result is stored in the global variables:

   std::vector<VertexData> vertexData_;
   std::vector<uint32_t> indexData_;
   std::vector<CachedMaterial> cachedMaterials_;

 and their material textures, loaded asynchronously and transcoded to BC7 on the first run (the transcoded ones are
 cached next to the content root):

   void loadMaterialTextures(VulkanApp& app, const char* pathPrefix, bool loadMaterialMaps = false)
   bool processLoadedMaterialTextures(lvk::ICommandBuffer& buffer, lvk::BufferHandle materialsBuffer)
   uint32_t numRemainingMaterialTextures()
   void drawMaterialTexturesLoadingProgress()
   void cancelLoadingMaterialTextures()

 `loadMaterialTextures()` returns immediately; call `processLoadedMaterialTextures()` once per frame to upload whatever
 is ready. The result is stored in the global variables:

   std::vector<GPUMaterial> materials_;
   std::vector<uint8_t> materialsAlphaTested_; // any texel with alpha below 250

 With `loadMaterialMaps`, the `map_Ks` (specular) and `map_bump` (normal) textures are loaded too, into `texSpecular` and
 `texNormal` (otherwise they stay 0, meaning "no texture").

 The samples rendering them share a few more helpers:

   BISTRO_GLSL                 // GLSL: `GPUMaterial` as `Material`, and `unpackOctahedral16()` for `VertexData::normal`
   FULLSCREEN_TRIANGLE_GLSL    // GLSL: `fullscreenTriangle()` turned by the swapchain pre-rotation `getClipRotation()`
   vec2 getClipRotation(lvk::IContext* ctx)
   lvk::Holder<lvk::TensorHandle> createTensorWithBuffer(...)
   CooperativeMatrixShape getCooperativeMatrixShape(...)
                               // a VK_KHR_cooperative_matrix tile shape, for the networks where VK_ARM_data_graph is missing
   class TosaGraphBuilder      // SPIR-V of a TOSA integer graph: CONV2D, RESCALE, TABLE, ADD
   struct GPUStageTimer        // GPU time per stage, measured every 2nd frame and read back a few frames later; its
                               // history in a "GPU time" window with a sparkline per stage (drawWindow())
   SceneAccelStructs createSceneAccelStructs(...)
   void loadSkyboxTextures(const VulkanApp& app, lvk::Holder<lvk::TextureHandle>& outReference,
                           lvk::Holder<lvk::TextureHandle>& outIrradiance)
*/

#pragma once

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cmath>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <fast_obj.h>
#include <meshoptimizer.h>
#include <shared/UtilsCubemap.h>
#include <taskflow/taskflow.hpp>

#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>

#include <ldrutils/lutils/ScopeExit.h>
#include <lvk/LVK.h>
#include <lvk/vulkan/VulkanUtils.h>

#include "VulkanApp.h"

#if defined(LVK_WITH_IMPLOT)
#include <implot.h>
#endif // LVK_WITH_IMPLOT

// must come after "VulkanApp.h": `gl_format.h` defines its own `VkFormat` unless `vulkan_core.h` was included first (`LVK_WITH_OPENXR`)
#include <ktx-software/lib/src/gl_format.h>
#include <ktx.h>

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

constexpr uint32_t kMeshCacheVersion = 0xC0DE000B;

#define MAX_MATERIAL_NAME 128

struct VertexData {
  vec3 position;
  uint32_t uv; // hvec2
  uint16_t normal; // Octahedral 16-bit https://www.shadertoy.com/view/llfcRl
  uint16_t mtlIndex;
};

static_assert(sizeof(VertexData) == 5 * sizeof(uint32_t));

std::vector<VertexData> vertexData_;
std::vector<uint32_t> indexData_;

struct CachedMaterial {
  char name[MAX_MATERIAL_NAME] = {};
  vec3 ambient = vec3(0.0f);
  vec3 diffuse = vec3(0.0f);
  char ambient_texname[MAX_MATERIAL_NAME] = {};
  char diffuse_texname[MAX_MATERIAL_NAME] = {};
  char alpha_texname[MAX_MATERIAL_NAME] = {};
  char specular_texname[MAX_MATERIAL_NAME] = {};
  char normal_texname[MAX_MATERIAL_NAME] = {};
};

std::vector<CachedMaterial> cachedMaterials_;

vec2 msign(vec2 v) {
  return vec2(v.x >= 0.0 ? 1.0f : -1.0f, v.y >= 0.0 ? 1.0f : -1.0f);
}

// https://www.shadertoy.com/view/llfcRl
uint16_t packSnorm2x8(vec2 v) {
  glm::uvec2 d = glm::uvec2(round(127.5f + v * 127.5f));
  return d.x | (d.y << 8u);
}

// https://www.shadertoy.com/view/llfcRl
uint16_t packOctahedral16(vec3 n) {
  n /= (abs(n.x) + abs(n.y) + abs(n.z));
  return ::packSnorm2x8((n.z >= 0.0) ? vec2(n.x, n.y) : (vec2(1.0) - abs(vec2(n.y, n.x))) * msign(vec2(n)));
}

std::string normalizeTextureName(const char* n) {
  if (!n)
    return std::string();
  LVK_ASSERT(strlen(n) < MAX_MATERIAL_NAME);
  std::string name(n);
#if defined(__linux__) || defined(__APPLE__) || defined(ANDROID)
  std::replace(name.begin(), name.end(), '\\', '/');
#endif
  return name;
}

namespace {
struct BistroMemFile {
  std::vector<uint8_t> data;
  size_t offset = 0;
};
void* bistroMemFileOpen(const char* path, void* userData) {
  VulkanApp* app = static_cast<VulkanApp*>(userData);
  BistroMemFile* file = new BistroMemFile();
  file->data = app->loadFile(path);
  if (file->data.empty()) {
    delete file;
    return nullptr;
  }
  return file;
}
void bistroMemFileClose(void* filePtr, void* /*userData*/) {
  delete static_cast<BistroMemFile*>(filePtr);
}
size_t bistroMemFileRead(void* filePtr, void* dst, size_t bytes, void* /*userData*/) {
  BistroMemFile* file = static_cast<BistroMemFile*>(filePtr);
  const size_t remaining = file->data.size() - file->offset;
  const size_t toRead = (bytes < remaining) ? bytes : remaining;
  memcpy(dst, file->data.data() + file->offset, toRead);
  file->offset += toRead;
  return toRead;
}
unsigned long bistroMemFileSize(void* filePtr, void* /*userData*/) {
  BistroMemFile* file = static_cast<BistroMemFile*>(filePtr);
  return (unsigned long)file->data.size();
}
} // namespace

bool loadAndCache(VulkanApp& app, const char* cacheFileName, const char* modelFileName) {
  LVK_PROFILER_FUNCTION();

  // load 3D model and cache it
  LLOGL("Loading `%s`... It can take a while in debug builds...\n", modelFileName);

  const std::string modelPath = (app.folderContentRoot_ + modelFileName);
  const fastObjCallbacks callbacks = {
      .file_open = bistroMemFileOpen,
      .file_close = bistroMemFileClose,
      .file_read = bistroMemFileRead,
      .file_size = bistroMemFileSize,
  };
  fastObjMesh* mesh = fast_obj_read_with_callbacks(modelPath.c_str(), &callbacks, &app);
  SCOPE_EXIT {
    if (mesh)
      fast_obj_destroy(mesh);
  };

  if (!LVK_VERIFY(mesh)) {
    LLOGW("Failed to load '%s'", modelFileName);
    LVK_ASSERT_MSG(false, "Did you read the tutorial at the top of this file?");
    return false;
  }

  LLOGL("Loaded.\n");

  uint32_t vertexCount = 0;

  for (uint32_t i = 0; i < mesh->face_count; ++i)
    vertexCount += mesh->face_vertices[i];

  vertexData_.reserve(vertexCount);

  uint32_t vertexIndex = 0;

  for (uint32_t face = 0; face < mesh->face_count; face++) {
    for (uint32_t v = 0; v < mesh->face_vertices[face]; v++) {
      LVK_ASSERT(v < 3);
      const fastObjIndex gi = mesh->indices[vertexIndex++];

      const float* p = &mesh->positions[gi.p * 3];
      const float* n = &mesh->normals[gi.n * 3];
      const float* t = &mesh->texcoords[gi.t * 2];

      vertexData_.push_back({
          .position = vec3(p[0], p[1], p[2]),
          .uv = glm::packHalf2x16(vec2(t[0], t[1])),
          .normal = packOctahedral16(vec3(n[0], n[1], n[2])),
          .mtlIndex = (uint16_t)mesh->face_materials[face],
      });
    }
  }

  // repack the mesh as described in https://github.com/zeux/meshoptimizer
  {
    // 1. Generate an index buffer
    const size_t indexCount = vertexData_.size();
    std::vector<uint32_t> remap(indexCount);
    const size_t remappedVertexCount =
        meshopt_generateVertexRemap(remap.data(), nullptr, indexCount, vertexData_.data(), indexCount, sizeof(VertexData));
    // 2. Remap vertices
    std::vector<VertexData> remappedVertices;
    indexData_.resize(indexCount);
    remappedVertices.resize(remappedVertexCount);
    meshopt_remapIndexBuffer(indexData_.data(), nullptr, indexCount, &remap[0]);
    meshopt_remapVertexBuffer(remappedVertices.data(), vertexData_.data(), indexCount, sizeof(VertexData), remap.data());
    vertexData_ = remappedVertices;
    // 3. Optimize for the GPU vertex cache reuse and overdraw
    meshopt_optimizeVertexCache(indexData_.data(), indexData_.data(), indexCount, remappedVertexCount);
    meshopt_optimizeOverdraw(
        indexData_.data(), indexData_.data(), indexCount, &vertexData_[0].position.x, remappedVertexCount, sizeof(VertexData), 1.05f);
    meshopt_optimizeVertexFetch(
        vertexData_.data(), indexData_.data(), indexCount, vertexData_.data(), remappedVertexCount, sizeof(VertexData));
  }

  // loop over materials
  for (uint32_t mtlIdx = 0; mtlIdx != mesh->material_count; mtlIdx++) {
    const fastObjMaterial& m = mesh->materials[mtlIdx];
    CachedMaterial mtl;
    mtl.ambient = vec3(m.Ka[0], m.Ka[1], m.Ka[2]);
    mtl.diffuse = vec3(m.Kd[0], m.Kd[1], m.Kd[2]);
    LVK_ASSERT(strlen(m.name) < MAX_MATERIAL_NAME);
    strcat(mtl.name, m.name);
    strcat(mtl.ambient_texname, normalizeTextureName(mesh->textures[m.map_Ka].name).c_str());
    strcat(mtl.diffuse_texname, normalizeTextureName(mesh->textures[m.map_Kd].name).c_str());
    strcat(mtl.alpha_texname, normalizeTextureName(mesh->textures[m.map_d].name).c_str());
    strcat(mtl.specular_texname, normalizeTextureName(mesh->textures[m.map_Ks].name).c_str());
    strcat(mtl.normal_texname, normalizeTextureName(mesh->textures[m.map_bump].name).c_str());
    cachedMaterials_.push_back(mtl);
  }

  LLOGL("Caching mesh...\n");

  std::filesystem::create_directories(std::filesystem::path(cacheFileName).parent_path());
  FILE* cacheFile = fopen(cacheFileName, "wb");
  if (cacheFile) {
    const uint32_t numMaterials = (uint32_t)cachedMaterials_.size();
    const uint32_t numVertices = (uint32_t)vertexData_.size();
    const uint32_t numIndices = (uint32_t)indexData_.size();
    fwrite(&kMeshCacheVersion, sizeof(kMeshCacheVersion), 1, cacheFile);
    fwrite(&numMaterials, sizeof(numMaterials), 1, cacheFile);
    fwrite(&numVertices, sizeof(numVertices), 1, cacheFile);
    fwrite(&numIndices, sizeof(numIndices), 1, cacheFile);
    fwrite(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile);
    fwrite(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile);
    fwrite(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile);
    fclose(cacheFile);
  }
  return true;
}

bool loadFromCache(VulkanApp& app, const char* cacheFileName) {
  const std::vector<uint8_t> data = app.loadFile(cacheFileName);
  if (data.empty())
    return false;

  size_t offset = 0;

  auto readBytes = [&data, &offset](void* dst, size_t bytes) -> bool {
    if (offset + bytes > data.size())
      return false;
    memcpy(dst, data.data() + offset, bytes);
    offset += bytes;
    return true;
  };

  uint32_t versionProbe = 0;
  if (!readBytes(&versionProbe, sizeof(versionProbe)))
    return false;
  if (versionProbe != kMeshCacheVersion) {
    LLOGL("Cache file has wrong version id\n");
    return false;
  }
  uint32_t numMaterials = 0;
  uint32_t numVertices = 0;
  uint32_t numIndices = 0;
  if (!readBytes(&numMaterials, sizeof(numMaterials)))
    return false;
  if (!readBytes(&numVertices, sizeof(numVertices)))
    return false;
  if (!readBytes(&numIndices, sizeof(numIndices)))
    return false;
  cachedMaterials_.resize(numMaterials);
  vertexData_.resize(numVertices);
  indexData_.resize(numIndices);
  if (!readBytes(cachedMaterials_.data(), sizeof(CachedMaterial) * numMaterials))
    return false;
  if (!readBytes(vertexData_.data(), sizeof(VertexData) * numVertices))
    return false;
  if (!readBytes(indexData_.data(), sizeof(uint32_t) * numIndices))
    return false;
#if defined(__linux__) || defined(__APPLE__) || defined(ANDROID)
  for (CachedMaterial& mtl : cachedMaterials_) {
    std::replace(std::begin(mtl.ambient_texname), std::end(mtl.ambient_texname), '\\', '/');
    std::replace(std::begin(mtl.diffuse_texname), std::end(mtl.diffuse_texname), '\\', '/');
    std::replace(std::begin(mtl.alpha_texname), std::end(mtl.alpha_texname), '\\', '/');
    std::replace(std::begin(mtl.specular_texname), std::end(mtl.specular_texname), '\\', '/');
    std::replace(std::begin(mtl.normal_texname), std::end(mtl.normal_texname), '\\', '/');
  }
#endif // __linux__ || __APPLE__ || ANDROID
  return true;
}

#if defined(ANDROID) || defined(__APPLE__)
constexpr bool kEnableTextureCompression = false;
#else
constexpr bool kEnableTextureCompression = true;
#endif // ANDROID || __APPLE__

struct GPUMaterial {
  vec4 ambient = vec4(0.0f);
  vec4 diffuse = vec4(0.0f);
  uint32_t texAmbient = 0;
  uint32_t texDiffuse = 0;
  uint32_t texAlpha = 0;
  uint32_t texSpecular = 0;
  uint32_t texNormal = 0;
  uint32_t padding[3] = {};
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<GPUMaterial> materials_;
std::vector<uint8_t> materialsAlphaTested_;

struct LoadedImage {
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t channels = 0;
  uint8_t* pixels = nullptr;
  std::string debugName;
  std::string compressedFileName;
  bool hasTransparentTexels = false;
};

struct LoadedMaterial {
  size_t idx = 0;
  LoadedImage ambient;
  LoadedImage diffuse;
  LoadedImage alpha;
  LoadedImage specular;
  LoadedImage normal;
};

VulkanApp* texturesApp_ = nullptr;
std::string texturesPathPrefix_;
bool loadMaterialMaps_ = false;
lvk::Holder<lvk::TextureHandle> textureDummyWhite_;

std::mutex imagesCacheMutex_;
std::unordered_map<std::string, LoadedImage> imagesCache_;
std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache_;
std::vector<LoadedMaterial> loadedMaterials_;
std::mutex loadedMaterialsMutex_;
std::atomic<bool> loaderShouldExit_ = false;
std::atomic<uint32_t> remainingMaterialsToLoad_ = 0;
std::unique_ptr<tf::Executor> loaderPool_;

std::string convertFileName(std::string fileName) {
  const std::string& contentRoot = texturesApp_->folderContentRoot_;

  if (fileName.find(contentRoot) == 0) {
    fileName = fileName.substr(contentRoot.length());
  }

  std::replace(fileName.begin(), fileName.end(), ':', '_');
  std::replace(fileName.begin(), fileName.end(), '.', '_');
  std::replace(fileName.begin(), fileName.end(), '/', '_');
  std::replace(fileName.begin(), fileName.end(), '\\', '_');

  return contentRoot + fileName + ".ktx";
}

void generateCompressedTexture(LoadedImage img) {
  LVK_PROFILER_FUNCTION();

  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  printf("...compressing texture to %s\n", img.compressedFileName.c_str());

  const uint32_t mipmapLevelCount = lvk::calcNumMipLevels(img.w, img.h);

  ktxTextureCreateInfo createInfoKTX2 = {
      .glInternalformat = GL_RGBA8,
      .vkFormat = VK_FORMAT_R8G8B8A8_UNORM,
      .baseWidth = img.w,
      .baseHeight = img.h,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipmapLevelCount,
      .numLayers = 1u,
      .numFaces = 1u,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture2* textureKTX2 = nullptr;
  (void)LVK_VERIFY(ktxTexture2_Create(&createInfoKTX2, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKTX2) == KTX_SUCCESS);
  if (!textureKTX2) {
    return;
  }

  SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(textureKTX2));
  };

  uint32_t w = img.w;
  uint32_t h = img.h;

  for (uint32_t i = 0; i != mipmapLevelCount; ++i) {
    size_t offset = 0;
    ktxTexture_GetImageOffset(ktxTexture(textureKTX2), i, 0, 0, &offset);

    stbir_resize_uint8_linear((const unsigned char*)img.pixels,
                              (int)img.w,
                              (int)img.h,
                              0,
                              ktxTexture_GetData(ktxTexture(textureKTX2)) + offset,
                              w,
                              h,
                              0,
                              STBIR_RGBA);

    h = h > 1 ? h >> 1 : 1;
    w = w > 1 ? w >> 1 : 1;
  }

  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  ktxBasisParams params = {
      .structSize = sizeof(params),
      .threadCount = 8,
      .compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL,
      .qualityLevel = 255,
  };
  (void)LVK_VERIFY(ktxTexture2_CompressBasisEx(textureKTX2, &params) == KTX_SUCCESS);
  (void)LVK_VERIFY(ktxTexture2_TranscodeBasis(textureKTX2, KTX_TTF_BC7_RGBA, 0) == KTX_SUCCESS);

  ktxTextureCreateInfo createInfoKTX1 = {
      .glInternalformat = GL_COMPRESSED_RGBA_BPTC_UNORM,
      .vkFormat = VK_FORMAT_BC7_UNORM_BLOCK,
      .baseWidth = img.w,
      .baseHeight = img.h,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipmapLevelCount,
      .numLayers = 1u,
      .numFaces = 1u,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture1* textureKTX1 = nullptr;
  (void)LVK_VERIFY(ktxTexture1_Create(&createInfoKTX1, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKTX1) == KTX_SUCCESS);
  if (!textureKTX1) {
    return;
  }

  for (uint32_t i = 0; i != mipmapLevelCount; ++i) {
    size_t offset1 = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(textureKTX1), i, 0, 0, &offset1) == KTX_SUCCESS);
    size_t offset2 = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(textureKTX2), i, 0, 0, &offset2) == KTX_SUCCESS);
    memcpy(ktxTexture_GetData(ktxTexture(textureKTX1)) + offset1,
           ktxTexture_GetData(ktxTexture(textureKTX2)) + offset2,
           ktxTexture_GetImageSize(ktxTexture(textureKTX1), i));
  }

  ktxTexture_WriteToNamedFile(ktxTexture(textureKTX1), img.compressedFileName.c_str());
}

LoadedImage loadImage(const char* fileName, int channels) {
  LVK_PROFILER_FUNCTION();

  if (!fileName || !*fileName) {
    return LoadedImage();
  }

  char debugStr[512] = {0};

  snprintf(debugStr, sizeof(debugStr) - 1, "%s (%i)", fileName, channels);

  const std::string debugName(debugStr);

  {
    std::lock_guard lock(imagesCacheMutex_);

    const std::unordered_map<std::string, LoadedImage>::const_iterator it = imagesCache_.find(debugName);

    if (it != imagesCache_.end()) {
      LVK_ASSERT(channels == it->second.channels);
      return it->second;
    }
  }

  int w = 0;
  int h = 0;
  int fileChannels = 0;
  const std::vector<uint8_t> fileData = texturesApp_->loadFile(fileName);
  uint8_t* pixels = fileData.empty() ? nullptr
                                     : stbi_load_from_memory(fileData.data(), (int)fileData.size(), &w, &h, &fileChannels, channels);

  bool hasTransparentTexels = false;

  // every image with alpha is scanned: the cache returns it for whichever material map asks next
  const bool hasAlpha = channels == 1 || fileChannels == 2 || fileChannels == 4;
  if (pixels && hasAlpha) {
    const size_t numPixels = size_t(w) * size_t(h);
    for (size_t p = 0; p != numPixels && !hasTransparentTexels; p++) {
      hasTransparentTexels = pixels[p * channels + channels - 1] < 250;
    }
  }

  const LoadedImage img = {
      .w = (uint32_t)w,
      .h = (uint32_t)h,
      .channels = (uint32_t)channels,
      .pixels = pixels,
      .debugName = debugName,
      .compressedFileName = convertFileName(fileName),
      .hasTransparentTexels = hasTransparentTexels,
  };

  if (img.pixels && kEnableTextureCompression && (channels != 1) && !std::filesystem::exists(img.compressedFileName.c_str())) {
    generateCompressedTexture(img);
  }

  std::lock_guard lock(imagesCacheMutex_);

  imagesCache_[debugName] = img;

  return img;
}

void loadMaterial(size_t i) {
  LVK_PROFILER_FUNCTION();

  SCOPE_EXIT {
    remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
  };

  // nothing more is loaded once the loader is asked to exit
  auto load = [](const char* fileName, int channels, bool enabled) -> LoadedImage {
    const bool skip = !enabled || !*fileName || loaderShouldExit_.load(std::memory_order_acquire);
    return skip ? LoadedImage() : loadImage((texturesPathPrefix_ + fileName).c_str(), channels);
  };
  const CachedMaterial& cached = cachedMaterials_[i];
  const LoadedMaterial mtl = {
      .idx = i,
      .ambient = load(cached.ambient_texname, 4, true),
      .diffuse = load(cached.diffuse_texname, 4, true),
      .alpha = load(cached.alpha_texname, 1, true),
      .specular = load(cached.specular_texname, 4, loadMaterialMaps_),
      .normal = load(cached.normal_texname, 4, loadMaterialMaps_),
  };
  if (loaderShouldExit_.load(std::memory_order_acquire)) {
    return;
  }

  materialsAlphaTested_[i] = mtl.diffuse.hasTransparentTexels || mtl.alpha.hasTransparentTexels ? 1 : 0;

  if (!mtl.ambient.pixels && !mtl.diffuse.pixels && !mtl.specular.pixels && !mtl.normal.pixels) {
    materials_[i].texDiffuse = 0;
  } else {
    std::lock_guard guard(loadedMaterialsMutex_);
    loadedMaterials_.push_back(mtl);
    remainingMaterialsToLoad_.fetch_add(1u, std::memory_order_release);
  }
}

lvk::Format formatFromChannels(uint32_t channels) {
  if (channels == 1) {
    return lvk::Format_R_UN8;
  }

  if (channels == 4) {
    return kEnableTextureCompression ? lvk::Format_BC7_RGBA : lvk::Format_RGBA_UN8;
  }

  return lvk::Format_Invalid;
}

lvk::TextureHandle createTextureFromLoadedImage(const LoadedImage& img) {
  if (!img.pixels) {
    return {};
  }

  const std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>>::const_iterator it = texturesCache_.find(img.debugName);

  if (it != texturesCache_.end()) {
    return it->second;
  }

  const bool hasCompressedTexture = kEnableTextureCompression && img.channels == 4 &&
                                    std::filesystem::exists(img.compressedFileName.c_str());

  const void* initialData = img.pixels;
  uint32_t initialDataNumMipLevels = 1u;

  ktxTexture* texture = nullptr;

  if (hasCompressedTexture) {
    if (!LVK_VERIFY(ktxTexture_CreateFromNamedFile(img.compressedFileName.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) ==
                    KTX_SUCCESS)) {
      printf("Failed to load %s\n", img.compressedFileName.c_str());
      return {};
    }
    initialData = texture->pData;
    initialDataNumMipLevels = lvk::calcNumMipLevels(img.w, img.h);
  }
  SCOPE_EXIT {
    if (texture)
      ktxTexture_Destroy(ktxTexture(texture));
  };

#if defined(__APPLE__) || defined(ANDROID)
  const bool generateMipmaps = true;
#else
  const bool generateMipmaps = !hasCompressedTexture;
#endif // __APPLE__ || ANDROID

  lvk::Holder<lvk::TextureHandle> tex = texturesApp_->ctx_->createTexture({
      .type = lvk::TextureType_2D,
      .format = formatFromChannels(img.channels),
      .dimensions = {img.w, img.h},
      .usage = lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(img.w, img.h),
      .components = (img.channels == 1) ? lvk::ComponentMapping{lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R}
                                        : lvk::ComponentMapping{},
      .data = initialData,
      .dataNumMipLevels = initialDataNumMipLevels,
      .generateMipmaps = generateMipmaps,
      .debugName = img.debugName.c_str(),
  });

  const lvk::TextureHandle handle = tex;

  texturesCache_[img.debugName] = std::move(tex);

  return handle;
}

void loadMaterialTextures(VulkanApp& app, const char* pathPrefix, bool loadMaterialMaps = false) {
  texturesApp_ = &app;
  texturesPathPrefix_ = app.folderContentRoot_ + pathPrefix;
  loadMaterialMaps_ = loadMaterialMaps;

  const uint32_t pixel = 0xFFFFFFFF;
  textureDummyWhite_ = app.ctx_->createTexture({
      .format = lvk::Format_RGBA_UN8,
      .dimensions = {1, 1},
      .usage = lvk::TextureUsageBits_Sampled,
      .data = &pixel,
      .debugName = "Texture: 1x1 white",
  });

  materials_.clear();
  materials_.reserve(cachedMaterials_.size());
  materialsAlphaTested_.assign(cachedMaterials_.size(), 0);

  for (const CachedMaterial& mtl : cachedMaterials_) {
    materials_.push_back(GPUMaterial{
        .ambient = vec4(mtl.ambient, 1.0f),
        .diffuse = vec4(mtl.diffuse, 1.0f),
        .texAmbient = textureDummyWhite_.index(),
        .texDiffuse = textureDummyWhite_.index(),
        .texAlpha = 0,
    });
  }

  stbi_set_flip_vertically_on_load(1);

  loaderShouldExit_ = false;
  remainingMaterialsToLoad_ = (uint32_t)cachedMaterials_.size();
  loaderPool_ = std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));

  for (size_t i = 0; i != cachedMaterials_.size(); i++) {
    loaderPool_->silent_async([i]() { loadMaterial(i); });
  }
}

bool processLoadedMaterialTextures(lvk::ICommandBuffer& buffer, lvk::BufferHandle materialsBuffer) {
  LoadedMaterial mtl;

  {
    std::lock_guard guard(loadedMaterialsMutex_);
    if (loadedMaterials_.empty()) {
      return false;
    }
    mtl = loadedMaterials_.back();
    loadedMaterials_.pop_back();
    remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
  }

  const lvk::TextureHandle ambient = createTextureFromLoadedImage(mtl.ambient);
  const lvk::TextureHandle diffuse = createTextureFromLoadedImage(mtl.diffuse);
  const lvk::TextureHandle alpha = createTextureFromLoadedImage(mtl.alpha);
  const lvk::TextureHandle specular = createTextureFromLoadedImage(mtl.specular);
  const lvk::TextureHandle normal = createTextureFromLoadedImage(mtl.normal);

  materials_[mtl.idx].texAmbient = ambient.index();
  materials_[mtl.idx].texDiffuse = diffuse.index();
  materials_[mtl.idx].texAlpha = alpha.index();
  materials_[mtl.idx].texSpecular = specular.index();
  materials_[mtl.idx].texNormal = normal.index();

  buffer.cmdUpdateBuffer(materialsBuffer, 0, sizeof(GPUMaterial) * materials_.size(), materials_.data());

  return true;
}

uint32_t numRemainingMaterialTextures() {
  return remainingMaterialsToLoad_.load(std::memory_order_acquire);
}

// a progress bar at the top of the screen; it goes into the background draw list so it stays beneath all ImGui windows
void drawMaterialTexturesLoadingProgress() {
  const uint32_t num = numRemainingMaterialTextures();
  if (!num || cachedMaterials_.empty()) {
    return;
  }
  const float progress = std::clamp(1.0f - float(num) / float(cachedMaterials_.size()), 0.0f, 1.0f);
  const ImVec2 size(ImGui::GetIO().DisplaySize.x, 32.0f);
  ImDrawList* drawList = ImGui::GetBackgroundDrawList();
  drawList->AddRectFilled(ImVec2(0, 0), size, ImGui::GetColorU32(ImGuiCol_FrameBg));
  drawList->AddRectFilled(ImVec2(0, 0), ImVec2(size.x * progress, size.y), ImGui::GetColorU32(ImGuiCol_PlotHistogram));
  char text[64] = {};
  snprintf(text, sizeof(text), "Loading textures... %.0f%%", progress * 100.0f);
  const ImVec2 textSize = ImGui::CalcTextSize(text);
  drawList->AddText(ImVec2(0.5f * (size.x - textSize.x), 0.5f * (size.y - textSize.y)), ImGui::GetColorU32(ImGuiCol_Text), text);
}

void cancelLoadingMaterialTextures() {
  loaderShouldExit_ = true;
  if (loaderPool_) {
    loaderPool_->wait_for_all();
    loaderPool_ = nullptr;
  }
  loadedMaterials_.clear();
  texturesCache_.clear();
  textureDummyWhite_ = nullptr;
  for (const std::pair<const std::string, LoadedImage>& img : imagesCache_) {
    if (img.second.pixels) {
      stbi_image_free(img.second.pixels);
    }
  }
  imagesCache_.clear();
  texturesApp_ = nullptr;
}

// GLSL: `GPUMaterial` and the octahedral normal of `VertexData`
#define BISTRO_GLSL                                                 \
  "struct Material {\n"                                             \
  "  vec4 ambient;\n"                                               \
  "  vec4 diffuse;\n"                                               \
  "  uint texAmbient;\n"                                            \
  "  uint texDiffuse;\n"                                            \
  "  uint texAlpha;\n"                                              \
  "  uint texSpecular;\n"                                           \
  "  uint texNormal;\n"                                             \
  "  uint padding[3];\n"                                            \
  "};\n"                                                            \
  "vec3 unpackOctahedral16(uint data) {\n"                          \
  "  vec2 v = vec2(uvec2(data, data >> 8) & 255u) / 127.5 - 1.0;\n" \
  "  vec3 n = vec3(v, 1.0 - abs(v.x) - abs(v.y));\n"                \
  "  float t = max(-n.z, 0.0);\n"                                   \
  "  n.x += (n.x > 0.0) ? -t : t;\n"                                \
  "  n.y += (n.y > 0.0) ? -t : t;\n"                                \
  "  return normalize(n);\n"                                        \
  "}\n"

// GLSL: a fullscreen triangle turned by the swapchain pre-rotation, `rotation` is `getClipRotation()`
#define FULLSCREEN_TRIANGLE_GLSL                                                                         \
  "vec4 fullscreenTriangle(vec2 rotation, out vec2 uv) {\n"                                              \
  "  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);\n"                                        \
  "  const vec2 p = uv * vec2(2, -2) + vec2(-1, 1);\n"                                                   \
  "  return vec4(rotation.x * p.x - rotation.y * p.y, rotation.y * p.x + rotation.x * p.y, 0.0, 1.0);\n" \
  "}\n"

// the swapchain pre-rotation as (cos, sin)
vec2 getClipRotation(lvk::IContext* ctx) {
  float c = 1.0f;
  float s = 0.0f;
  lvk::getSurfaceTransformRotation(ctx->getSwapchainSurfaceTransform(), c, s);
  return vec2(c, s);
}

// a tensor aliasing a new storage buffer of its size, which shaders reach through the buffer's device address
lvk::Holder<lvk::TensorHandle> createTensorWithBuffer(lvk::IContext* ctx,
                                                      const lvk::TensorDesc& desc,
                                                      lvk::Holder<lvk::BufferHandle>& outBuffer,
                                                      const char* debugName,
                                                      lvk::Result* outResult) {
  outBuffer = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_Storage,
      .storage = lvk::StorageType_Device,
      .size = lvk::getTensorDataSize(desc),
      .debugName = debugName,
  });
  return ctx->createTensorBufferAliasing(outBuffer, 0, desc, debugName, outResult);
}

// an M x N x K shape of subgroup-scope cooperative matrices computing A * B + C with these component types; all 0 without one
struct CooperativeMatrixShape {
  uint32_t m = 0;
  uint32_t n = 0;
  uint32_t k = 0;
  uint32_t minSubgroupSize = 0; // of the device, never smaller than m
  uint32_t maxSharedMemorySize = 0; // of a compute workgroup on the device
};

// the networks fall back on VK_KHR_cooperative_matrix where VK_ARM_data_graph is missing; square tiles are preferred, and M is
// never larger than the smallest subgroup, so that a subgroup's own invocations can fill the rows of a tile
CooperativeMatrixShape getCooperativeMatrixShape(lvk::IContext* ctx,
                                                 VkComponentTypeKHR typeA,
                                                 VkComponentTypeKHR typeB,
                                                 VkComponentTypeKHR typeC) {
  if (!ctx->supportsCooperativeMatrix()) {
    return {};
  }
  const VkPhysicalDevice physicalDevice = lvk::getVkPhysicalDevice(ctx);
  VkPhysicalDeviceVulkan13Properties properties13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceProperties2 properties2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &properties13};
  vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);
  uint32_t count = 0;
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, nullptr);
  std::vector<VkCooperativeMatrixPropertiesKHR> properties(count, {.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &count, properties.data());
  CooperativeMatrixShape shape;
  for (const VkCooperativeMatrixPropertiesKHR& p : properties) {
    const bool isMatch = p.AType == typeA && p.BType == typeB && p.CType == typeC && p.ResultType == typeC &&
                         p.scope == VK_SCOPE_SUBGROUP_KHR && !p.saturatingAccumulation && p.MSize <= properties13.minSubgroupSize;
    const bool isBetter = !shape.m || (p.MSize == p.NSize && shape.m != shape.n);
    if (isMatch && isBetter) {
      shape = {.m = p.MSize,
               .n = p.NSize,
               .k = p.KSize,
               .minSubgroupSize = properties13.minSubgroupSize,
               .maxSharedMemorySize = properties2.properties.limits.maxComputeSharedMemorySize};
    }
  }
  return shape;
}

// SPIR-V of a TOSA integer graph (SPV_ARM_graph, TOSA.001000.1) over NHWC tensors [1, height, width, channels], written one
// operator at a time from the int8 graph input: an int8 1x1 CONV2D into int32 (the weights and biases of the i-th one are the
// graph constants 2i and 2i + 1), a per-channel RESCALE between int8 and int32, a TABLE of 256 int8 entries, an int32 ADD.
// Every call returns the id of its result, and `build()` writes the module whose output is the given value
class TosaGraphBuilder final {
 public:
  TosaGraphBuilder(uint32_t width, uint32_t height, uint32_t numInputs) : width_(width), height_(height) {
    tosa_ = nextId_++;
    int8_ = declare(SpvOpTypeInt, 0, {8, 0});
    int32_ = declare(SpvOpTypeInt, 0, {32, 0});
    input_ = newValue(int8_, numInputs);
  }

  uint32_t input() const {
    return input_;
  }
  uint32_t conv2d(uint32_t value, uint32_t numOutputs, int32_t inputZeroPoint) {
    const uint32_t numInputs = values_[value].y;
    const uint32_t zero = constUint(0);
    const uint32_t one = constUint(1);
    const uint32_t stride = constTensor(int32_, {one, one});
    const uint32_t weights = declare(SpvOpGraphConstantARM, typeTensor(int8_, {numOutputs, 1, 1, numInputs}), {constantId_++});
    const uint32_t biases = declare(SpvOpGraphConstantARM, typeTensor(int32_, {numOutputs}), {constantId_++});
    const uint32_t accumulatorInt32 = one;
    return op(int32_,
              numOutputs,
              TOSA_CONV2D,
              {constTensor(int32_, {zero, zero, zero, zero}),
               stride,
               stride,
               accumulatorInt32,
               constBool(false),
               value,
               weights,
               biases,
               zeroPoint(int8_, inputZeroPoint),
               zeroPoint(int8_, 0)});
  }
  uint32_t rescale(uint32_t value, bool isInt32Output, const int32_t* multipliers, const int8_t* shifts, int32_t outputZeroPoint) {
    const uint32_t channels = values_[value].y;
    const uint32_t element = isInt32Output ? int32_ : int8_;
    std::vector<uint32_t> multiplierIds;
    std::vector<uint32_t> shiftIds;
    for (uint32_t i = 0; i != channels; i++) {
      multiplierIds.push_back(constUint((uint32_t)multipliers[i]));
      shiftIds.push_back(constInt8(shifts[i]));
    }
    const uint32_t roundingSingle = constUint(1);
    return op(element,
              channels,
              TOSA_RESCALE,
              {constBool(true),
               roundingSingle,
               constBool(true),
               constBool(false),
               constBool(false),
               value,
               constTensor(int32_, multiplierIds),
               constTensor(int8_, shiftIds),
               zeroPoint(values_[value].x, 0),
               zeroPoint(element, outputZeroPoint)});
  }
  uint32_t table(uint32_t value, const int8_t* entries) {
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i != 256; i++) {
      ids.push_back(constInt8(entries[i]));
    }
    return op(int8_, values_[value].y, TOSA_TABLE, {value, constTensor(int8_, ids)});
  }
  uint32_t add(uint32_t a, uint32_t b) {
    return op(int32_, values_[a].y, TOSA_ADD, {a, b});
  }

  std::vector<uint32_t> build(uint32_t output) {
    const uint32_t varInput = nextId_++;
    const uint32_t varOutput = nextId_++;
    const uint32_t zero = constUint(0);
    const uint32_t typeInput = typeValue(input_);
    const uint32_t typeOutput = typeValue(output);
    const uint32_t pointerInput = declare(SpvOpTypePointer, 0, {SpvStorageClassUniformConstant, typeInput});
    const uint32_t pointerOutput = declare(SpvOpTypePointer, 0, {SpvStorageClassUniformConstant, typeOutput});
    const uint32_t typeGraph = declare(SpvOpTypeGraphARM, 0, {1, typeInput, typeOutput});
    const uint32_t graph = nextId_++;

    std::vector<uint32_t> spirv = {SpvMagicNumber, SpvVersion1_6, 0, 0, 0};
    for (const uint32_t capability :
         {SpvCapabilityShader, SpvCapabilityInt8, SpvCapabilityTensorsARM, SpvCapabilityGraphARM, SpvCapabilityVulkanMemoryModel}) {
      emit(spirv, SpvOpCapability, {capability});
    }
    emit(spirv, SpvOpExtension, literalString("SPV_ARM_tensors"));
    emit(spirv, SpvOpExtension, literalString("SPV_ARM_graph"));
    std::vector<uint32_t> import = {tosa_};
    for (const uint32_t w : literalString("TOSA.001000.1")) {
      import.push_back(w);
    }
    emit(spirv, SpvOpExtInstImport, import);
    emit(spirv, SpvOpMemoryModel, {SpvAddressingModelLogical, SpvMemoryModelVulkan});
    emit(spirv, SpvOpDecorate, {varInput, SpvDecorationDescriptorSet, 0});
    emit(spirv, SpvOpDecorate, {varInput, SpvDecorationBinding, 0});
    emit(spirv, SpvOpDecorate, {varOutput, SpvDecorationDescriptorSet, 0});
    emit(spirv, SpvOpDecorate, {varOutput, SpvDecorationBinding, 1});
    spirv.insert(spirv.end(), declarations_.begin(), declarations_.end());
    emit(spirv, SpvOpVariable, {pointerInput, varInput, SpvStorageClassUniformConstant});
    emit(spirv, SpvOpVariable, {pointerOutput, varOutput, SpvStorageClassUniformConstant});
    std::vector<uint32_t> entryPoint = {graph};
    for (const uint32_t w : literalString("main")) {
      entryPoint.push_back(w);
    }
    entryPoint.push_back(varInput);
    entryPoint.push_back(varOutput);
    emit(spirv, SpvOpGraphEntryPointARM, entryPoint);
    emit(spirv, SpvOpGraphARM, {typeGraph, graph});
    emit(spirv, SpvOpGraphInputARM, {typeInput, input_, zero});
    spirv.insert(spirv.end(), body_.begin(), body_.end());
    emit(spirv, SpvOpGraphSetOutputARM, {output, zero});
    emit(spirv, SpvOpGraphEndARM, {});
    spirv[3] = nextId_;
    return spirv;
  }

 private:
  // the SPIR-V (spirv.core.grammar.json) and TOSA.001000.1 numbers the graph uses
  enum : uint32_t {
    SpvMagicNumber = 0x07230203,
    SpvVersion1_6 = 0x00010600,
    SpvOpExtension = 10,
    SpvOpExtInstImport = 11,
    SpvOpExtInst = 12,
    SpvOpMemoryModel = 14,
    SpvOpCapability = 17,
    SpvOpTypeBool = 20,
    SpvOpTypeInt = 21,
    SpvOpTypeArray = 28,
    SpvOpTypePointer = 32,
    SpvOpConstantTrue = 41,
    SpvOpConstantFalse = 42,
    SpvOpConstant = 43,
    SpvOpConstantComposite = 44,
    SpvOpVariable = 59,
    SpvOpDecorate = 71,
    SpvOpTypeTensorARM = 4163,
    SpvOpGraphConstantARM = 4181,
    SpvOpGraphEntryPointARM = 4182,
    SpvOpGraphARM = 4183,
    SpvOpGraphInputARM = 4184,
    SpvOpGraphSetOutputARM = 4185,
    SpvOpGraphEndARM = 4186,
    SpvOpTypeGraphARM = 4190,
    SpvCapabilityShader = 1,
    SpvCapabilityInt8 = 39,
    SpvCapabilityTensorsARM = 4174,
    SpvCapabilityGraphARM = 4191,
    SpvCapabilityVulkanMemoryModel = 5345,
    SpvAddressingModelLogical = 0,
    SpvMemoryModelVulkan = 3,
    SpvStorageClassUniformConstant = 0,
    SpvDecorationBinding = 33,
    SpvDecorationDescriptorSet = 34,
    TOSA_CONV2D = 2,
    TOSA_ADD = 14,
    TOSA_TABLE = 30,
    TOSA_RESCALE = 65,
  };

  static void emit(std::vector<uint32_t>& out, uint32_t opcode, const std::vector<uint32_t>& operands) {
    out.push_back(uint32_t(operands.size() + 1) << 16 | opcode);
    out.insert(out.end(), operands.begin(), operands.end());
  }
  static std::vector<uint32_t> literalString(const char* str) {
    std::vector<uint32_t> words(strlen(str) / 4 + 1, 0u);
    memcpy(words.data(), str, strlen(str));
    return words;
  }
  // types and constants are declared once: a type must be unique, and the graph repeats its zero points and shapes
  uint32_t declare(uint32_t opcode, uint32_t resultType, const std::vector<uint32_t>& operands) {
    std::vector<uint32_t> key = {opcode, resultType};
    key.insert(key.end(), operands.begin(), operands.end());
    const std::string keyBytes((const char*)key.data(), key.size() * sizeof(uint32_t));
    const std::unordered_map<std::string, uint32_t>::const_iterator it = declared_.find(keyBytes);
    if (it != declared_.end()) {
      return it->second;
    }
    const uint32_t id = nextId_++;
    std::vector<uint32_t> words;
    if (resultType) {
      words.push_back(resultType);
    }
    words.push_back(id);
    words.insert(words.end(), operands.begin(), operands.end());
    emit(declarations_, opcode, words);
    declared_[keyBytes] = id;
    return id;
  }
  uint32_t constUint(uint32_t v) {
    return declare(SpvOpConstant, int32_, {v});
  }
  uint32_t constInt8(int32_t v) {
    return declare(SpvOpConstant, int8_, {uint32_t(uint8_t(v))});
  }
  uint32_t constBool(bool v) {
    return declare(v ? SpvOpConstantTrue : SpvOpConstantFalse, declare(SpvOpTypeBool, 0, {}), {});
  }
  uint32_t typeTensor(uint32_t element, const std::vector<uint32_t>& dims) {
    const uint32_t rank = constUint((uint32_t)dims.size());
    std::vector<uint32_t> values;
    for (const uint32_t d : dims) {
      values.push_back(constUint(d));
    }
    const uint32_t shape = declare(SpvOpConstantComposite, declare(SpvOpTypeArray, 0, {int32_, rank}), values);
    return declare(SpvOpTypeTensorARM, 0, {element, rank, shape});
  }
  uint32_t typeValue(uint32_t value) {
    return typeTensor(values_[value].x, {1, height_, width_, values_[value].y});
  }
  uint32_t constTensor(uint32_t element, const std::vector<uint32_t>& values) {
    return declare(SpvOpConstantComposite, typeTensor(element, {(uint32_t)values.size()}), values);
  }
  uint32_t zeroPoint(uint32_t element, int32_t v) {
    return constTensor(element, {element == int32_ ? constUint((uint32_t)v) : constInt8(v)});
  }
  uint32_t newValue(uint32_t element, uint32_t channels) {
    const uint32_t id = nextId_++;
    values_.resize(nextId_);
    values_[id] = glm::uvec2(element, channels);
    return id;
  }
  uint32_t op(uint32_t element, uint32_t channels, uint32_t instruction, const std::vector<uint32_t>& operands) {
    const uint32_t type = typeTensor(element, {1, height_, width_, channels});
    const uint32_t id = newValue(element, channels);
    std::vector<uint32_t> words = {type, id, tosa_, instruction};
    words.insert(words.end(), operands.begin(), operands.end());
    emit(body_, SpvOpExtInst, words);
    return id;
  }

  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t nextId_ = 1;
  uint32_t tosa_ = 0;
  uint32_t int8_ = 0;
  uint32_t int32_ = 0;
  uint32_t input_ = 0;
  uint32_t constantId_ = 0;
  std::vector<glm::uvec2> values_; // (element type, channels) of every value id
  std::vector<uint32_t> declarations_;
  std::vector<uint32_t> body_;
  std::unordered_map<std::string, uint32_t> declared_;
};

// GPU time per stage of a frame: stage `i` runs from timestamp `i - 1` to timestamp `i`, the frame from the first to the last;
// every kFramesPerMeasurement-th frame writes its timestamps into the next of kFrameSlots sets, read back kFrameSlots - 1
// measurements later, when the GPU is done with them, so reading never waits; `isMeasuring` says whether the times got a new frame,
// and they are smoothed over the measured frames; the last kHistory of them are kept for drawWindow()
struct GPUStageTimer {
  enum {
    kMaxTimestamps = 8,
    kFramesPerMeasurement = 2,
    kFrameSlots = 4,
    kHistory = 128,
  };
  lvk::Holder<lvk::QueryPoolHandle> queryPool;
  uint32_t numTimestamps = 0;
  bool isWriting = false;
  bool isMeasuring = false;
  uint32_t slot = 0;
  bool slotWritten[kFrameSlots] = {};
  double stageTime[kMaxTimestamps] = {}; // ms
  double frameTime = 0.0; // ms
  float history[kMaxTimestamps][kHistory] = {}; // ms, the newest last; entry 0 is the whole frame
  uint32_t numHistory = 0;
  uint32_t windowFramesDrawn = 0;

  void create(lvk::IContext* ctx, uint32_t num) {
    LVK_ASSERT(num >= 2 && num <= kMaxTimestamps);
    numTimestamps = num;
    queryPool = ctx->createQueryPool(kFrameSlots * num, "Query pool: timestamps");
  }
  // writes timestamp 0
  void beginFrame(lvk::ICommandBuffer& buf, uint64_t frameIndex) {
    isWriting = frameIndex % kFramesPerMeasurement == 0;
    if (isWriting) {
      slot = (frameIndex / kFramesPerMeasurement) % kFrameSlots;
      slotWritten[slot] = true;
      buf.cmdResetQueryPool(queryPool, slot * numTimestamps, numTimestamps);
      buf.cmdWriteTimestamp(queryPool, slot * numTimestamps);
    }
  }
  void timestamp(lvk::ICommandBuffer& buf, uint32_t index) const {
    if (isWriting) {
      buf.cmdWriteTimestamp(queryPool, slot * numTimestamps + index);
    }
  }
  // after the frame was submitted: a measured frame reads back the oldest set, which the next measured frame writes again
  void endFrame(lvk::IContext* ctx) {
    const uint32_t oldest = (slot + 1) % kFrameSlots;
    isMeasuring = isWriting && slotWritten[oldest];
    if (!isMeasuring) {
      return;
    }
    slotWritten[oldest] = false;
    uint64_t timestamps[kMaxTimestamps] = {};
    ctx->getQueryPoolResults(
        queryPool, oldest * numTimestamps, numTimestamps, sizeof(timestamps[0]) * numTimestamps, timestamps, sizeof(timestamps[0]));
    const double toMs = ctx->getTimestampPeriodToMs();
    auto smooth = [](double& time, double ms) { time = time > 0.0 ? glm::mix(time, ms, 0.25) : ms; };
    for (uint32_t i = 1; i != numTimestamps; i++) {
      smooth(stageTime[i], double(timestamps[i] - timestamps[i - 1]) * toMs);
    }
    smooth(frameTime, double(timestamps[numTimestamps - 1] - timestamps[0]) * toMs);
    for (uint32_t i = 0; i != numTimestamps; i++) {
      std::copy(history[i] + 1, history[i] + kHistory, history[i]);
      history[i][kHistory - 1] = float(i ? stageTime[i] : frameTime);
    }
    numHistory = std::min<uint32_t>(numHistory + 1, kHistory);
  }

  // a row of the "GPU time" window: timestamp 0 is the whole frame
  struct Row {
    const char* name;
    uint32_t timestamp;
    ImVec4 color;
  };
  // the time of every row with a sparkline of its history (with ImPlot), 10 pixels off the bottom left corner, then free to move
  void drawWindow(ldr::Span<const Row> rows) {
    // an auto-resized window gets its size only on its second frame
    if (windowFramesDrawn < 2) {
      windowFramesDrawn++;
      const ImGuiViewport* v = ImGui::GetMainViewport();
      ImGui::SetNextWindowPos(ImVec2(v->WorkPos.x + 10.0f, v->WorkPos.y + v->WorkSize.y - 10.0f), ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    }
    ImGui::SetNextWindowBgAlpha(0.8f);
    ImGui::Begin("GPU time",
                 nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav);
    if (ImGui::BeginTable("##gpuTime", 3)) {
      ImGui::TableSetupColumn("Stage");
      ImGui::TableSetupColumn("ms");
      ImGui::TableSetupColumn("History", ImGuiTableColumnFlags_WidthFixed, 12.0f * ImGui::GetFontSize());
      ImGui::TableHeadersRow();
      for (const Row& row : rows) {
        const float* values = history[row.timestamp] + kHistory - numHistory;
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(row.color, "%s", row.name);
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(row.color, "%6.2f", numHistory ? values[numHistory - 1] : 0.0f);
        ImGui::TableSetColumnIndex(2);
#if defined(LVK_WITH_IMPLOT)
        const float sparklineHeight = ImGui::GetTextLineHeight() * 1.5f;
        if (numHistory < 2) {
          // the window keeps its size once the sparklines appear
          ImGui::Dummy(ImVec2(0.0f, sparklineHeight));
          continue;
        }
        const float maxTime = *std::max_element(values, values + numHistory);
        ImGui::PushID(row.name);
        ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
        ImPlot::SetNextAxesLimits(0, kHistory - 1, 0.0, std::max(1.2f * maxTime, 0.01f), ImGuiCond_Always);
        if (ImPlot::BeginPlot("##sparkline", ImVec2(-1, sparklineHeight), ImPlotFlags_CanvasOnly)) {
          ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
          // right-aligned: the newest measurement is always at the right edge
          const double xstart = double(kHistory - numHistory);
          ImPlot::PlotShaded(
              "##sparkline", values, numHistory, 0.0, 1.0, xstart, ImPlotSpec(ImPlotProp_LineColor, row.color, ImPlotProp_FillAlpha, 0.4f));
          ImPlot::PlotLine("##sparkline", values, numHistory, 1.0, xstart, ImPlotSpec(ImPlotProp_LineColor, row.color));
          ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
        ImGui::PopID();
#endif // LVK_WITH_IMPLOT
      }
      ImGui::EndTable();
    }
    ImGui::End();
  }
};

// ray tracing: BLASes over the triangles of `indexBuffer`, each small enough for `getMaxStorageBufferRange()`, in one TLAS scaling
// them by `scale`; the first `numOpaque` triangles are flagged opaque (ray queries skip their alpha test), and the instance custom
// index of a BLAS is its first triangle
struct SceneAccelStructs {
  std::vector<lvk::Holder<lvk::AccelStructHandle>> BLAS;
  lvk::Holder<lvk::AccelStructHandle> TLAS;
};

SceneAccelStructs createSceneAccelStructs(lvk::IContext* ctx,
                                          lvk::BufferHandle vertexBuffer,
                                          uint32_t numVertices,
                                          lvk::BufferHandle indexBuffer,
                                          uint32_t numTriangles,
                                          uint32_t numOpaque,
                                          float scale) {
  SceneAccelStructs out;
  const uint32_t maxStorageBufferSize = ctx->getMaxStorageBufferRange();
  const glm::mat3x4 transform(glm::scale(mat4(1.0f), vec3(scale)));
  const uint32_t ranges[2][2] = {{0, numOpaque}, {numOpaque, numTriangles}};
  std::vector<lvk::AccelStructInstance> instances;
  for (uint32_t r = 0; r != 2; r++) {
    const uint32_t begin = ranges[r][0];
    const uint32_t end = ranges[r][1];
    if (begin == end) {
      continue;
    }
    lvk::AccelStructDesc blasDesc{
        .type = lvk::AccelStructType_BLAS,
        .geometryType = lvk::AccelStructGeomType_Triangles,
        .geometryFlags = r == 0 ? uint8_t(lvk::AccelStructGeometryFlagBits_Opaque) : uint8_t(0),
        .vertexFormat = lvk::VertexFormat_Float3,
        .vertexBuffer = vertexBuffer,
        .vertexStride = sizeof(VertexData),
        .numVertices = numVertices,
        .indexFormat = lvk::IndexFormat_UI32,
        .indexBuffer = indexBuffer,
        .buildRange = {.primitiveCount = end - begin},
        .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
        .debugName = r == 0 ? "BLAS: opaque" : "BLAS",
    };
    const lvk::AccelStructSizes sizes = ctx->getAccelStructSizes(blasDesc);
    const uint32_t numBLAS = 1 + std::max(uint32_t(sizes.buildScratchSize / maxStorageBufferSize),
                                          uint32_t(sizes.accelerationStructureSize / maxStorageBufferSize));
    const uint32_t primitiveCount = (end - begin + numBLAS - 1) / numBLAS;
    for (uint32_t first = begin; first < end; first += primitiveCount) {
      blasDesc.buildRange.primitiveOffset = first * 3 * sizeof(uint32_t);
      blasDesc.buildRange.primitiveCount = std::min(primitiveCount, end - first);
      out.BLAS.emplace_back(ctx->createAccelerationStructure(blasDesc));
      instances.push_back(lvk::AccelStructInstance{
          .transform = (const lvk::mat3x4&)transform,
          .instanceCustomIndex = first,
          .mask = 0xff,
          .instanceShaderBindingTableRecordOffset = 0,
          .flags = lvk::AccelStructInstanceFlagBits_TriangleFacingCullDisable,
          .accelerationStructureReference = ctx->gpuAddress(out.BLAS.back()),
      });
    }
  }
  const lvk::Holder<lvk::BufferHandle> instancesBuffer = ctx->createBuffer({
      .usage = lvk::BufferUsageBits_AccelStructBuildInputReadOnly,
      .storage = lvk::StorageType_HostVisible,
      .size = sizeof(lvk::AccelStructInstance) * instances.size(),
      .data = instances.data(),
      .debugName = "Buffer: TLAS instances",
  });
  out.TLAS = ctx->createAccelerationStructure({
      .type = lvk::AccelStructType_TLAS,
      .geometryType = lvk::AccelStructGeomType_Instances,
      .instancesBuffer = instancesBuffer,
      .buildRange = {.primitiveCount = uint32_t(instances.size())},
      .buildFlags = lvk::AccelStructBuildFlagBits_PreferFastTrace,
      .debugName = "TLAS",
  });
  return out;
}

// skybox: the reference and irradiance cubemaps of an equirectangular HDR image, converted into KTX files in the content root on
// the first run
lvk::Holder<lvk::TextureHandle> createCubemapTexture(lvk::IContext* ctx, const std::vector<uint8_t>& dataKTX, const char* debugName) {
  LVK_PROFILER_FUNCTION();

  ktxTexture1* texture = nullptr;
  if (dataKTX.empty() ||
      ktxTexture1_CreateFromMemory(dataKTX.data(), dataKTX.size(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture) != KTX_SUCCESS) {
    LLOGW("Cannot load the cubemap `%s`\n", debugName);
    return {};
  }
  SCOPE_EXIT {
    ktxTexture_Destroy(ktxTexture(texture));
  };

  if (!LVK_VERIFY(texture->glInternalformat == GL_RGBA32F)) {
    return {};
  }

  const uint32_t width = texture->baseWidth;
  const uint32_t height = texture->baseHeight;

  return ctx->createTexture({
      .type = lvk::TextureType_Cube,
      .format = lvk::Format_RGBA_F32,
      .dimensions = {width, height},
      .usage = lvk::TextureUsageBits_Sampled,
      .numMipLevels = lvk::calcNumMipLevels(width, height),
      .data = texture->pData,
      // if compression is enabled, upload all mip-levels
      .dataNumMipLevels = kEnableTextureCompression ? lvk::calcNumMipLevels(width, height) : 1u,
      .generateMipmaps = !kEnableTextureCompression,
      .debugName = debugName,
  });
}

ktxTexture1* bitmapToCube(Bitmap& bmp) {
  LVK_ASSERT(bmp.comp_ == 3); // RGB
  LVK_ASSERT(bmp.type_ == eBitmapType_Cube);
  LVK_ASSERT(bmp.fmt_ == eBitmapFormat_Float);

  const int w = bmp.w_;
  const int h = bmp.h_;

  const uint32_t mipLevels = lvk::calcNumMipLevels(w, h);

  ktxTextureCreateInfo createInfo = {
      .glInternalformat = GL_RGBA32F,
      .vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
      .baseWidth = static_cast<uint32_t>(w),
      .baseHeight = static_cast<uint32_t>(h),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipLevels,
      .numLayers = 1u,
      .numFaces = 6u,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture1* texture = nullptr;
  (void)LVK_VERIFY(ktxTexture1_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture) == KTX_SUCCESS);

  const int numFacePixels = w * h;

  for (size_t face = 0; face != 6; face++) {
    const vec3* src = reinterpret_cast<vec3*>(bmp.data_.data()) + face * numFacePixels;
    size_t offset = 0;
    (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(texture), 0, 0, face, &offset) == KTX_SUCCESS);
    float* dst = (float*)(texture->pData + offset);
    for (int y = 0; y != h; y++) {
      for (int x = 0; x != w; x++) {
        const vec4 rgba = vec4(src[x + y * w], 1.0f);
        memcpy(dst, &rgba, sizeof(rgba));
        dst += 4;
      }
    }
  }

  return texture;
}

void saveCubemapWithMipmaps(const std::string& outFilename, ktxTexture1* cubemap) {
  LVK_PROFILER_FUNCTION();

  LLOGL("Generating mipmaps");

  LVK_ASSERT(cubemap);

  uint32_t prevWidth = cubemap->baseWidth;
  uint32_t prevHeight = cubemap->baseHeight;

  for (uint32_t face = 0; face != 6; face++) {
    LLOGL(".");
    for (uint32_t miplevel = 1; miplevel < cubemap->numLevels; miplevel++) {
      LLOGL(":");
      const uint32_t width = prevWidth > 1 ? prevWidth >> 1 : 1;
      const uint32_t height = prevHeight > 1 ? prevHeight >> 1 : 1;

      size_t prevOffset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel - 1, 0, face, &prevOffset) == KTX_SUCCESS);
      size_t offset = 0;
      (void)LVK_VERIFY(ktxTexture_GetImageOffset(ktxTexture(cubemap), miplevel, 0, face, &offset) == KTX_SUCCESS);

      stbir_resize_float_linear(reinterpret_cast<const float*>(cubemap->pData + prevOffset),
                                prevWidth,
                                prevHeight,
                                0,
                                reinterpret_cast<float*>(cubemap->pData + offset),
                                width,
                                height,
                                0,
                                STBIR_RGBA);

      prevWidth = width;
      prevHeight = height;
    }
    prevWidth = cubemap->baseWidth;
    prevHeight = cubemap->baseHeight;
  }

  LLOGL("\n");
  ktxTexture_WriteToNamedFile(ktxTexture(cubemap), outFilename.c_str());
}

void processCubemap(const std::string& inFilename, const std::string& outFilenameEnv, const std::string& outFilenameIrr) {
  LVK_PROFILER_FUNCTION();

  int sourceWidth, sourceHeight;
  float* pxs = stbi_loadf(inFilename.c_str(), &sourceWidth, &sourceHeight, nullptr, 3);
  SCOPE_EXIT {
    if (pxs) {
      stbi_image_free(pxs);
    }
  };

  if (!LVK_VERIFY(pxs)) {
    LVK_ASSERT_MSG(false, "Cannot load the HDR skybox: did you run `deploy_content.py`?");
    return;
  }

  // Environment map
  {
    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(sourceWidth, sourceHeight, 3, eBitmapFormat_Float, pxs));
    ktxTexture1* cube = bitmapToCube(bmp);
    saveCubemapWithMipmaps(outFilenameEnv, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }

  // Irradiance map
  {
    enum {
      dstW = 256,
      dstH = 128,
    };

    std::vector<vec3> out(dstW * dstH);
    convolveDiffuse((vec3*)pxs, sourceWidth, sourceHeight, dstW, dstH, out.data(), 1024);

    Bitmap bmp = convertEquirectangularMapToCubeMapFaces(Bitmap(dstW, dstH, 3, eBitmapFormat_Float, out.data()));
    ktxTexture1* cube = bitmapToCube(bmp);
    saveCubemapWithMipmaps(outFilenameIrr, cube);
    ktxTexture_Destroy(ktxTexture(cube));
  }
}

void loadSkyboxTextures(const VulkanApp& app,
                        lvk::Holder<lvk::TextureHandle>& outReference,
                        lvk::Holder<lvk::TextureHandle>& outIrradiance) {
  LVK_PROFILER_FUNCTION();

  const std::string skyboxFileName{"immenstadter_horn_2k"};
  const std::string skyboxSubdir{"src/skybox_hdr/"};

  const std::string fileNameRefKTX = app.folderContentRoot_ + skyboxFileName + "_ReferenceMap.ktx";
  const std::string fileNameIrrKTX = app.folderContentRoot_ + skyboxFileName + "_IrradianceMap.ktx";

  std::vector<uint8_t> dataRefKTX = app.loadFile(fileNameRefKTX.c_str());
  std::vector<uint8_t> dataIrrKTX = app.loadFile(fileNameIrrKTX.c_str());

  if (dataRefKTX.empty() || dataIrrKTX.empty()) {
    const std::string inFilename = app.folderContentRoot_ + skyboxSubdir + skyboxFileName + ".hdr";
    LLOGL("Cubemap in KTX format not found. Extracting from HDR file `%s`...\n", inFilename.c_str());

    processCubemap(inFilename, fileNameRefKTX, fileNameIrrKTX);

    dataRefKTX = app.loadFile(fileNameRefKTX.c_str());
    dataIrrKTX = app.loadFile(fileNameIrrKTX.c_str());
  }

  outReference = createCubemapTexture(app.ctx_.get(), dataRefKTX, fileNameRefKTX.c_str());
  outIrradiance = createCubemapTexture(app.ctx_.get(), dataIrrKTX, fileNameIrrKTX.c_str());
}
