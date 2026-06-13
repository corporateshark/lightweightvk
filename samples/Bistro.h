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
*/

#pragma once

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif // _USE_MATH_DEFINES
#include <cmath>

#include <filesystem>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <fast_obj.h>
#include <meshoptimizer.h>
#include <taskflow/taskflow.hpp>

#include <ldrutils/lutils/ScopeExit.h>
#include <lvk/LVK.h>

#include "VulkanApp.h"

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
  return true;
}
