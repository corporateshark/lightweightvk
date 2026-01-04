# LightweightVK Samples

## 001_HelloTriangle

![lvk_android](../.github/samples/001_HelloTriangle.jpg)

## 002_RenderToCubeMap

![lvk_android](../.github/samples/002_RenderToCubeMap.jpg)

## 003_RenderToCubeMapSinglePass

![lvk_android](../.github/samples/003_RenderToCubeMapSinglePass.jpg)

## 004_YUV

![lvk_android](../.github/samples/004_YUV.jpg)

## 005_MeshShaders

![lvk_android](../.github/samples/005_MeshShaders.jpg)

## 007_DynamicRenderingLocalRead

![lvk_android](../.github/samples/007_DynamicRenderingLocalRead.jpg)

## 008_MeshShaderFireworks

![lvk_android](../.github/samples/008_MeshShaderFireworks.jpg)

## 009_TriplanarMapping

![lvk_android](../.github/samples/009_TriplanarMapping.jpg)

## 010_OmniShadows

![lvk_android](../.github/samples/010_OmniShadows.jpg)

## DEMO_001_SolarSystem

![lvk_android](../.github/screenshot02.jpg)

## RTX_001_Hello

![lvk_android](../.github/samples/006_RayTracingHello.jpg)

## RTX_002_AO

![lvk_android](../.github/samples/007_RayTracingAO.jpg)

## RTX_003_Pipeline

![lvk_android](../.github/samples/RTX_003_Pipeline.jpg)

## RTX_004_Textures

![lvk_android](../.github/samples/RTX_004_Textures.jpg)

## Tiny_MeshLarge

![lvk_android](../.github/samples/Tiny_MeshLarge.jpg)

### Performance measurements on Android

![lvk_android](../.github/samples/tiny_mesh_large_android.jpg)

|Device|GPU|Resolution|MSAA|Texture quality|Frame time|
|---|---|---|---|---|---|
|Xiaomi 13T Pro|Immortalis-G715|2712x1220|8x|High (2048x2048)|26ms|
|Xiaomi 13T Pro|Immortalis-G715|2712x1220|None|High (2048x2048)|16ms|
|Xiaomi 13T Pro|Immortalis-G715|2712x1220|None|Low (256x256)|**14ms**|
|Google Pixel 7 Pro|Mali-G710|3120x1440|8x|High (2048x2048)|85ms|
|Google Pixel 7 Pro|Mali-G710|3120x1440|None|High (2048x2048)|62ms|
|Google Pixel 7 Pro|Mali-G710|3120x1440|None|Low (256x256)|57ms|
|Google Pixel 7 Pro|Mali-G710|2712x1220|8x|High (2048x2048)|80ms|
|Google Pixel 7 Pro|Mali-G710|2712x1220|None|Low (256x256)|**54ms**|
