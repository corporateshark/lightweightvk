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

   void loadMaterialTextures(VulkanApp& app, const char* pathPrefix)
   bool processLoadedMaterialTextures(lvk::ICommandBuffer& buffer, lvk::BufferHandle materialsBuffer)
   uint32_t numRemainingMaterialTextures()
   void drawMaterialTexturesLoadingProgress()
   void cancelLoadingMaterialTextures()

 `loadMaterialTextures()` returns immediately; call `processLoadedMaterialTextures()` once per frame to upload whatever
 is ready. The result is stored in the global variable:

   std::vector<GPUMaterial> materials_;
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
#include <taskflow/taskflow.hpp>

#include <stb/stb_image.h>
#include <stb/stb_image_resize2.h>

#include <ldrutils/lutils/ScopeExit.h>
#include <lvk/LVK.h>

#include "VulkanApp.h"

// must come after "VulkanApp.h": `gl_format.h` defines its own `VkFormat` unless `vulkan_core.h` was included first (`LVK_WITH_OPENXR`)
#include <ktx-software/lib/src/gl_format.h>
#include <ktx.h>

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

constexpr uint32_t kMeshCacheVersion = 0xC0DE000A;

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
  uint32_t padding = 0;
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

std::vector<GPUMaterial> materials_;

struct LoadedImage {
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t channels = 0;
  uint8_t* pixels = nullptr;
  std::string debugName;
  std::string compressedFileName;
};

struct LoadedMaterial {
  size_t idx = 0;
  LoadedImage ambient;
  LoadedImage diffuse;
  LoadedImage alpha;
};

VulkanApp* texturesApp_ = nullptr;
std::string texturesPathPrefix_;
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
  const std::vector<uint8_t> fileData = texturesApp_->loadFile(fileName);
  uint8_t* pixels = fileData.empty() ? nullptr : stbi_load_from_memory(fileData.data(), (int)fileData.size(), &w, &h, nullptr, channels);

  const LoadedImage img = {
      .w = (uint32_t)w,
      .h = (uint32_t)h,
      .channels = (uint32_t)channels,
      .pixels = pixels,
      .debugName = debugName,
      .compressedFileName = convertFileName(fileName),
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

#define LOAD_TEX(result, tex, channels)                                                                          \
  const LoadedImage result = std::string(cachedMaterials_[i].tex).empty()                                        \
                                 ? LoadedImage()                                                                 \
                                 : loadImage((texturesPathPrefix_ + cachedMaterials_[i].tex).c_str(), channels); \
  if (loaderShouldExit_.load(std::memory_order_acquire)) {                                                       \
    return;                                                                                                      \
  }

  LOAD_TEX(ambient, ambient_texname, 4);
  LOAD_TEX(diffuse, diffuse_texname, 4);
  LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

  const LoadedMaterial mtl{i, ambient, diffuse, alpha};

  if (!mtl.ambient.pixels && !mtl.diffuse.pixels) {
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

void loadMaterialTextures(VulkanApp& app, const char* pathPrefix) {
  texturesApp_ = &app;
  texturesPathPrefix_ = app.folderContentRoot_ + pathPrefix;

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

  materials_[mtl.idx].texAmbient = ambient.index();
  materials_[mtl.idx].texDiffuse = diffuse.index();
  materials_[mtl.idx].texAlpha = alpha.index();

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
