/* rendercore.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

namespace lh2core
{

//  +-----------------------------------------------------------------------------+
//  |  DeviceVars                                                                 |
//  |  Copy of device-side variables, to detect changes.                    LH2'19|
//  +-----------------------------------------------------------------------------+
struct DeviceVars
{
	// impossible values to trigger an update in the first frame
	float clampValue = -1.0f;
	float geometryEpsilon = 1e34f;
	float filterClampDirect = 2.5f;
	float filterClampIndirect = 15.0f;
	uint filterEnabled = 1;
	uint TAAEnabled = 1;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore
{
public:
	static RenderCore *instance;

	// methods
	void Init();
	void Render( const ViewPyramid &view, const Convergence converge, const float brightness, const float contrast );
	void Setting( const char *name, const float value );
	void SetTarget( GLTexture *target, const uint spp );
	void Shutdown();
	void KeyDown( const uint key ) {}
	void KeyUp( const uint key ) {}
	// passing data. Note: RenderCore always copies what it needs; the passed data thus remains the
	// property of the caller, and can be safely deleted or modified as soon as these calls return.
	void SetTextures( const CoreTexDesc *tex, const int textureCount );
	void SetMaterials( CoreMaterial *mat, const CoreMaterialEx *matEx, const int materialCount ); // textures must be in sync when calling this
	void SetLights( const CoreLightTri *areaLights, const int areaLightCount,
		const CorePointLight *pointLights, const int pointLightCount,
		const CoreSpotLight *spotLights, const int spotLightCount,
		const CoreDirectionalLight *directionalLights, const int directionalLightCount );
	void SetSkyData( const float3 *pixels, const uint width, const uint height );
	// geometry and instances:
	// a scene is setup by first passing a number of meshes (geometry), then a number of instances.
	// note that stored meshes can be used zero, one or multiple times in the scene.
	// also note that, when using alpha flags, materials must be in sync.
	void SetGeometry( const int meshIdx, const float4 *vertexData, const int vertexCount, const int triangleCount, const CoreTri *triangles, const uint *alphaFlags = 0 );
	void SetInstance( const int instanceIdx, const int modelIdx, const mat4 &transform );
	void UpdateToplevel();
	void SetProbePos( const int2 pos );

	// public data members
	vk::DispatchLoaderDynamic dynamicDispatcher; // Dynamic dispatcher for extension functions such as NV_RT

private:
	// internal methods
	void InitRenderer();
	void CreateInstance();
	void SetupValidationLayers( vk::InstanceCreateInfo &createInfo );

	void CreateDebugReportCallback();
	void CreateDevice();
	void CreateCommandBuffers();
	void CreateOffscreenBuffers();
	void ResizeBuffers();

	void CreateRayTracingPipeline();
	void CreateShadePipeline();
	void CreateFinalizePipeline();
	void CreateDescriptorSets();
	void RecordCommandBuffers();
	void CreateBuffers();
	void InitializeDescriptorSets();

	vk::Instance m_VkInstance = nullptr;
	vk::DebugUtilsMessengerEXT m_VkDebugMessenger = nullptr; // Debug validation messenger
	vk::CommandBuffer m_BlitCommandBuffer;
	std::vector<GeometryInstance> m_Instances;
	std::vector<bool> m_MeshChanged = std::vector<bool>( 256 );
	std::vector<vk::DescriptorBufferInfo> m_TriangleBufferInfos;
	std::vector<lh2core::CoreMesh *> m_Meshes{};
	std::vector<CoreTexDesc> m_TexDescs{};
	VulkanDevice m_Device;

	// Frame data
	VulkanGLTextureInterop *m_InteropTexture = nullptr;
	uint32_t m_SamplesPP;
	uint32_t m_CurrentFrame = 0;
	uint32_t m_ScrWidth = 0, m_ScrHeight = 0;
	int m_Initialized = false;
	int m_First = true;
	int m_InstanceMeshMappingDirty = true;
	int m_SamplesTaken = 0;
	uint4 m_LightCounts;
	bool m_FirstConvergingFrame = false;

	// Uniform data
	UniformCamera *m_UniformCamera{};
	UniformFinalizeParams *m_UniformFinalizeParams{};
	TopLevelAS *m_TopLevelAS = nullptr;

	// Ray trace pipeline
	VulkanDescriptorSet *rtDescriptorSet = nullptr;
	VulkanRayTraceNVPipeline *rtPipeline = nullptr;

	// Shade pipeline
	VulkanDescriptorSet *shadeDescriptorSet = nullptr;
	VulkanComputePipeline *shadePipeline = nullptr;

	// Finalize pipeline
	VulkanDescriptorSet *finalizeDescriptorSet = nullptr;
	VulkanComputePipeline *finalizePipeline = nullptr;

	// Storage buffers
	VulkanCoreBuffer<mat4> *m_InvTransformsBuffer = nullptr;
	VulkanCoreBuffer<Counters> *m_CounterTransferBuffer = nullptr;
	VulkanCoreBuffer<uint> *m_ARGB32Buffer = nullptr;
	VulkanCoreBuffer<float4> *m_ARGB128Buffer = nullptr;
	VulkanCoreBuffer<uint> *m_NRM32Buffer = nullptr;
	VulkanCoreBuffer<uint32_t> *m_InstanceMeshMappingBuffer = nullptr;
	VulkanCoreBuffer<Counters> *m_Counters = nullptr;
	VulkanCoreBuffer<CoreMaterial> *m_Materials = nullptr;
	VulkanCoreBuffer<CoreLightTri> *m_AreaLightBuffer = nullptr;
	VulkanCoreBuffer<CorePointLight> *m_PointLightBuffer = nullptr;
	VulkanCoreBuffer<CoreSpotLight> *m_SpotLightBuffer = nullptr;
	VulkanCoreBuffer<CoreDirectionalLight> *m_DirectionalLightBuffer = nullptr;
	VulkanCoreBuffer<uint> *m_BlueNoiseBuffer = nullptr;
	VulkanCoreBuffer<float4> *m_CombinedStateBuffer[2] = { nullptr, nullptr };
	VulkanCoreBuffer<float4> *m_AccumulationBuffer = nullptr;
	VulkanCoreBuffer<PotentialContribution> *m_PotentialContributionBuffer = nullptr;

	VulkanImage *m_SkyboxImage = nullptr;
	VulkanImage *m_OffscreenImage = nullptr; // Off-screen render image
	int2 m_ProbePos = make_int2( 0 );		 // triangle picking; primary ray for this pixel copies its triid to coreStats.
public:
	CoreStats coreStats; // rendering statistics
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
	void *pUserData )
{
	const char *severity = 0, *type = 0;
	switch (messageSeverity)
	{
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT):
		return VK_FALSE;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT):
		severity = "INFO";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT):
		severity = "WARNING";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT):
		severity = "ERROR";
		break;
	default:
		break;
	}

	switch (messageType)
	{
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT):
		type = "GENERAL";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT):
		type = "VALIDATION";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT):
		type = "PERFORMANCE";
		break;
	default:
		break;
	}

	char buffer[4096];
	snprintf( buffer, sizeof( buffer ), "Vulkan Validation Layer: [Severity: %s] [Type: %s] : %s\n", severity, type, pCallbackData->pMessage );
	printf( "%s", buffer );

	return VK_FALSE;
}

} // namespace lh2core

// EOF