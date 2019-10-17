/* rendercore.cpp - Copyright 2019 Utrecht University

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

#include "core_settings.h"

namespace lh2core {

constexpr std::array<const char *, 1> VALIDATION_LAYERS = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char *> DEVICE_EXTENSIONS = { VK_NV_RAY_TRACING_EXTENSION_NAME };

RenderCore *RenderCore::instance = nullptr;

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetProbePos                                                    |
//  |  Set the pixel for which the triid will be captured.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetProbePos( int2 pos )
{
	m_ProbePos = pos; // triangle id for this pixel will be stored in coreStats
}

void RenderCore::InitRenderer()
{
	// Create dynamic Vulkan function loader
	dynamicDispatcher.init( m_VkInstance, m_Device );
#ifndef NDEBUG
	CreateDebugReportCallback();
#endif
	CreateCommandBuffers();								  // Initialize blit buffers
	m_TopLevelAS = new TopLevelAS( m_Device, FastTrace ); // Create a top level AS, Vulkan doesn't like unbound buffers
	CreateDescriptorSets();								  // Create bindings for shaders
	CreateBuffers();									  // Create uniforms like our camera
	CreateOffscreenBuffers();							  // Create image buffer
	CreateRayTracingPipeline();							  // Create ray intersection pipeline
	CreateShadePipeline();								  // Create compute pipeline; wavefront shading
	CreateFinalizePipeline();							  // Create compute pipeline; plot accumulation buffer to image
	m_TopLevelAS->Build();								  // Build top level AS

	// Set initial sky box, Vulkan does not like having unbound buffers
	float3 dummy = make_float3( 0.0f );
	SetSkyData( &dummy, 1, 1 );

	m_Initialized = true;
}

void RenderCore::CreateInstance()
{
	std::vector<const char *> extensions;

#ifndef NDEBUG
	extensions.push_back( VK_EXT_DEBUG_REPORT_EXTENSION_NAME );
	extensions.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
#endif

	extensions.push_back( VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME );
	extensions.push_back( VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME );

	vk::ApplicationInfo appInfo = vk::ApplicationInfo( "Lighthouse 2", 1, "No Engine", VK_MAKE_VERSION( 1, 0, 0 ), VK_API_VERSION_1_1 );
	vk::InstanceCreateInfo createInfo{};
	createInfo.setPApplicationInfo( &appInfo );

	// Configure required extensions;
	createInfo.setEnabledExtensionCount( static_cast<uint32_t>(extensions.size()) );
	createInfo.setPpEnabledExtensionNames( extensions.data() );

#ifndef NDEBUG
	SetupValidationLayers( createInfo ); // Configure Vulkan validation layers
#endif

	m_VkInstance = vk::createInstance( createInfo );
	if (!m_VkInstance) FATALERROR( "Could not initialize Vulkan." );
	printf( "Successfully created Vulkan instance.\n" );
}

void RenderCore::SetupValidationLayers( vk::InstanceCreateInfo &createInfo )
{
	// Get supported layers
	const auto availableLayers = vk::enumerateInstanceLayerProperties();

	const auto hasLayer = [&availableLayers]( const char *layerName ) -> bool {
		for (const auto &layer : availableLayers)
		{
			if (strcmp( layerName, layer.layerName ) == 0)
				return true;
		}

		return false;
	};

	createInfo.setEnabledLayerCount( 0 );

	// Check if requested validation layers are present
#ifndef NDEBUG

	bool layersFound = true;
	for (auto layer : VALIDATION_LAYERS)
	{
		if (!hasLayer( layer ))
		{
			createInfo.setEnabledLayerCount( 0 ), layersFound = false;
			printf( "Could not enable validation layer: \"%s\"\n", layer );
			break;
		}
	}

	if (layersFound)
	{
		// All layers available
		createInfo.setEnabledLayerCount( static_cast<uint32_t>(VALIDATION_LAYERS.size()) );
		createInfo.setPpEnabledLayerNames( VALIDATION_LAYERS.data() );
	}
#endif
}

void RenderCore::CreateDebugReportCallback()
{
#ifndef NDEBUG
	vk::DebugUtilsMessengerCreateInfoEXT dbgMessengerCreateInfo{};
	dbgMessengerCreateInfo.setMessageSeverity( vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning );
	dbgMessengerCreateInfo.setMessageType( vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation );
	dbgMessengerCreateInfo.setPfnUserCallback( debugCallback );
	dbgMessengerCreateInfo.setPUserData( nullptr );

	m_VkDebugMessenger = m_VkInstance.createDebugUtilsMessengerEXT( dbgMessengerCreateInfo, nullptr, dynamicDispatcher );
	if (!m_VkDebugMessenger)
		printf( "Could not setup Vulkan debug utils messenger.\n" );
#endif
}

void RenderCore::CreateDevice()
{
	// Start with application defined required extensions
	std::vector<const char *> dev_extensions = DEVICE_EXTENSIONS;
	// In Lighthouse2 we render render core images using OpenGL, get required extensions for an interop
	for (const auto ext : VulkanGLTextureInterop::GetRequiredExtensions())
		dev_extensions.push_back( ext );

	// Retrieve a physical device that supports our requested extensions
	std::optional<vk::PhysicalDevice> physicalDevice = VulkanDevice::PickDeviceWithExtensions( m_VkInstance, dev_extensions );

	// Sanity check
	if (!physicalDevice.has_value())
		FATALERROR( "No supported Vulkan devices available." );

	// Create device
	m_Device = VulkanDevice( physicalDevice.value(), dev_extensions );
	coreStats.deviceName = physicalDevice->getProperties().deviceName;
	const auto properties = m_Device.GetPhysicalDevice().getProperties();
	char deviceName[256];
	memcpy( deviceName, properties.deviceName, 256 * sizeof( char ) );
	coreStats.deviceName = deviceName;
}

void RenderCore::CreateCommandBuffers()
{
	if (m_BlitCommandBuffer) m_Device.FreeCommandBuffer( m_BlitCommandBuffer );
	m_BlitCommandBuffer = m_Device.CreateCommandBuffer( vk::CommandBufferLevel::ePrimary );
}

void RenderCore::CreateOffscreenBuffers()
{
	vk::MemoryPropertyFlags memFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
	vk::Format format = vk::Format::eR32G32B32A32Sfloat;

	delete m_OffscreenImage;
	const vk::ImageUsageFlags usageFlags = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
	m_OffscreenImage = new VulkanImage( m_Device, vk::ImageType::e2D, format,
		{ m_ScrWidth, m_ScrHeight, 1 }, vk::ImageTiling::eOptimal,
		usageFlags, memFlags );

	m_OffscreenImage->CreateImageView( vk::ImageViewType::e2D, format,
		{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } );
	m_OffscreenImage->TransitionToLayout( vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eColorAttachmentWrite, nullptr );

	finalizeDescriptorSet->Bind( fOUTPUT, { m_OffscreenImage->GetDescriptorImageInfo() } );
}

void RenderCore::ResizeBuffers()
{
	const auto newPixelCount = uint32_t( (m_ScrWidth * m_ScrHeight) * 1.3f ); // Make buffer bigger than needed to prevent reallocating often
	const auto oldPixelCount = m_AccumulationBuffer->GetSize() / sizeof( float4 );

	if (oldPixelCount >= newPixelCount) return; // No need to resize buffers

	delete m_CombinedStateBuffer[0];
	delete m_CombinedStateBuffer[1];
	delete m_AccumulationBuffer;
	delete m_PotentialContributionBuffer;

	const auto limits = m_Device.GetPhysicalDevice().getProperties().limits;

	// Create 2 path trace state buffers, these buffers are ping-ponged every path iteration
	m_CombinedStateBuffer[0] = new VulkanCoreBuffer<float4>( m_Device, NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * 4, limits.minUniformBufferOffsetAlignment ), vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );
	m_CombinedStateBuffer[1] = new VulkanCoreBuffer<float4>( m_Device, NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * 4, limits.minUniformBufferOffsetAlignment ), vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );

	// Accumulation buffer for rendered image
	m_AccumulationBuffer = new VulkanCoreBuffer<float4>( m_Device, newPixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	// Shadow ray buffer
	m_PotentialContributionBuffer = new VulkanCoreBuffer<PotentialContribution>( m_Device, MAXPATHLENGTH * newPixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );

	const vk::DeviceSize singleSize = NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * sizeof( float4 ), 4 * limits.minUniformBufferOffsetAlignment );

	// Buffers got recreated so we need to update our descriptor sets
	rtDescriptorSet->Bind( rtPATH_STATES, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 0, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 0, singleSize ) } );
	rtDescriptorSet->Bind( rtPATH_ORIGINS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( singleSize, singleSize ) } );
	rtDescriptorSet->Bind( rtPATH_DIRECTIONS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ) } );
	rtDescriptorSet->Bind( rtACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
	rtDescriptorSet->Bind( rtPOTENTIAL_CONTRIBUTIONS, { m_PotentialContributionBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cPOTENTIAL_CONTRIBUTIONS, { m_PotentialContributionBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cPATH_STATES, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 0, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 0, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_ORIGINS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( singleSize, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_DIRECTIONS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_THROUGHPUTS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 3 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 3 * singleSize, singleSize ) } );
	finalizeDescriptorSet->Bind( fACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
}

void RenderCore::CreateRayTracingPipeline()
{
	rtPipeline = new VulkanRayTraceNVPipeline( m_Device );
	// Setup ray generation shader
	auto rgenShader = VulkanShader( m_Device, "rt_shaders.rgen" );
	auto chitShader = VulkanShader( m_Device, "rt_shaders.rchit" );
	auto missShader = VulkanShader( m_Device, "rt_shaders.rmiss" );
	auto shadowShader = VulkanShader( m_Device, "rt_shadow.rmiss" );

	// Setup pipeline
	rtPipeline->AddRayGenShaderStage( rgenShader );																		   // Ray generation
	rtPipeline->AddHitGroup( { nullptr, &chitShader } );																	   // Hit shader
	rtPipeline->AddMissShaderStage( missShader );																		   // General miss shader
	rtPipeline->AddMissShaderStage( shadowShader );																		   // Shadow ray miss shader
	rtPipeline->AddEmptyHitGroup();																						   // Any hit-stage requires a hit group, add an empty one for the shadow ray miss shader
	rtPipeline->AddPushConstant( vk::PushConstantRange( vk::ShaderStageFlagBits::eRaygenNV, 0, 3 * sizeof( uint32_t ) ) ); // Add a push constant for defining ray stage
	rtPipeline->SetMaxRecursionDepth( 5u );																				   // Not used in our wavefront implementation, but define it just in case
	rtPipeline->AddDescriptorSet( rtDescriptorSet );																	   // Bind descriptor set
	rtPipeline->Finalize();																								   // Create pipeline object
}

void RenderCore::CreateShadePipeline()
{
	auto computeShader = VulkanShader( m_Device, "rt_shade.comp" );															 // Shade compute shader
	shadePipeline = new VulkanComputePipeline( m_Device, computeShader );													 // Initialize pipeline
	shadePipeline->AddDescriptorSet( shadeDescriptorSet );																	 // Bind descriptor set
	shadePipeline->AddPushConstant( vk::PushConstantRange( vk::ShaderStageFlagBits::eCompute, 0, sizeof( uint32_t ) * 2 ) ); // Add push constant data
	shadePipeline->Finalize();																								 // Create pipeline object
}

void RenderCore::CreateFinalizePipeline()
{
	auto computeShader = VulkanShader( m_Device, "rt_finalize.comp" );		 // Finalize compute shader
	finalizePipeline = new VulkanComputePipeline( m_Device, computeShader ); // Initialize pipeline
	finalizePipeline->AddDescriptorSet( finalizeDescriptorSet );			 // Bind descriptor set
	finalizePipeline->Finalize();											 // Create pipeline object
}

void RenderCore::CreateDescriptorSets()
{
	// Create initial descriptor set objects
	rtDescriptorSet = new VulkanDescriptorSet( m_Device );
	shadeDescriptorSet = new VulkanDescriptorSet( m_Device );
	finalizeDescriptorSet = new VulkanDescriptorSet( m_Device );

	// Describe ray trace descriptor set
	rtDescriptorSet->AddBinding( rtACCELERATION_STRUCTURE, 1, vk::DescriptorType::eAccelerationStructureNV, vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV );
	rtDescriptorSet->AddBinding( rtCAMERA, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtPATH_STATES, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtPATH_ORIGINS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtPATH_DIRECTIONS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtPOTENTIAL_CONTRIBUTIONS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->AddBinding( rtBLUENOISE, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV );
	rtDescriptorSet->Finalize();

	// Describe shade descriptor set
	const auto limits = m_Device.GetPhysicalDevice().getProperties().limits;
	assert( limits.maxDescriptorSetStorageBuffers > MAX_TRIANGLE_BUFFERS ); // Make sure we don't have too many triangle buffers
	shadeDescriptorSet->AddBinding( cCOUNTERS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cCAMERA, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPATH_STATES, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPATH_ORIGINS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPATH_DIRECTIONS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPATH_THROUGHPUTS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPOTENTIAL_CONTRIBUTIONS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cMATERIALS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cSKYBOX, 1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cTRIANGLES, MAX_TRIANGLE_BUFFERS, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cTRIANGLE_BUFFER_INDICES, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cINVERSE_TRANSFORMS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cTEXTURE_ARGB32, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cTEXTURE_ARGB128, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cTEXTURE_NRM32, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cAREALIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cPOINTLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cSPOTLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cDIRECTIONALLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->AddBinding( cBLUENOISE, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	shadeDescriptorSet->Finalize();

	// Describe finalize descriptor set
	finalizeDescriptorSet->AddBinding( fACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute );
	finalizeDescriptorSet->AddBinding( fUNIFORM_CONSTANTS, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute );
	finalizeDescriptorSet->AddBinding( fOUTPUT, 1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute );
	finalizeDescriptorSet->Finalize();
}

void RenderCore::RecordCommandBuffers()
{
	vk::CommandBufferBeginInfo beginInfo{};
	vk::ImageSubresourceRange subresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 );
	vk::ImageCopy copyRegion{};
	copyRegion.setSrcSubresource( { vk::ImageAspectFlagBits::eColor, 0, 0, 1 } );
	copyRegion.setSrcOffset( { 0, 0, 0 } );
	copyRegion.setDstSubresource( { vk::ImageAspectFlagBits::eColor, 0, 0, 1 } );
	copyRegion.setDstOffset( { 0, 0, 0 } );
	copyRegion.setExtent( { m_ScrWidth, m_ScrHeight, 1 } );

	// Start recording
	m_BlitCommandBuffer.begin( beginInfo );
	m_InteropTexture->RecordTransitionToVulkan( m_BlitCommandBuffer ); // Make sure interop texture is ready to be used by Vulkan

	// Create a barrier to ensure our off-screen image has finished being written to
	vk::ImageMemoryBarrier imageMemoryBarrier{};
	imageMemoryBarrier.setPNext( nullptr );
	imageMemoryBarrier.setSrcAccessMask( vk::AccessFlagBits::eShaderWrite );
	imageMemoryBarrier.setDstAccessMask( vk::AccessFlagBits::eTransferRead );
	imageMemoryBarrier.setOldLayout( vk::ImageLayout::eGeneral );
	imageMemoryBarrier.setNewLayout( vk::ImageLayout::eTransferSrcOptimal );
	imageMemoryBarrier.setSrcQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	imageMemoryBarrier.setDstQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	imageMemoryBarrier.setImage( m_OffscreenImage->GetImage() );
	imageMemoryBarrier.setSubresourceRange( subresourceRange );
	m_BlitCommandBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
		vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier );

	vk::ImageSubresourceLayers subResourceLayers;
	subResourceLayers.setAspectMask( vk::ImageAspectFlagBits::eColor );
	subResourceLayers.setBaseArrayLayer( 0 );
	subResourceLayers.setLayerCount( 1 );
	subResourceLayers.setMipLevel( 0 );

	m_BlitCommandBuffer.copyImage( m_OffscreenImage->GetImage(), vk::ImageLayout::eTransferSrcOptimal, // Copy actual image
		m_InteropTexture->GetImage(), vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion );
	m_InteropTexture->RecordTransitionToGL( m_BlitCommandBuffer ); // Make image usable by GL again
	m_BlitCommandBuffer.end();
}

void RenderCore::CreateBuffers()
{
	const auto pixelCount = static_cast<vk::DeviceSize>(m_ScrWidth * m_ScrHeight);
	m_CounterTransferBuffer = new VulkanCoreBuffer<Counters>( m_Device, 1, vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eHostVisible,
		vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst );
	m_InvTransformsBuffer = new VulkanCoreBuffer<mat4>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		ON_DEVICE | ON_HOST );

	m_UniformCamera = new UniformCamera( m_Device );
	m_UniformFinalizeParams = new UniformFinalizeParams( m_Device );
	m_Counters = new VulkanCoreBuffer<Counters>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
		ON_DEVICE | ON_HOST );

	// Bind uniforms
	rtDescriptorSet->Bind( rtCAMERA, { m_UniformCamera->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cCAMERA, { m_UniformCamera->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cCOUNTERS, { m_Counters->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( fUNIFORM_CONSTANTS, { m_UniformFinalizeParams->GetDescriptorBufferInfo() } );

	m_Materials = new VulkanCoreBuffer<CoreMaterial>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_InstanceMeshMappingBuffer = new VulkanCoreBuffer<uint32_t>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, ON_HOST | ON_DEVICE );

	// Texture buffers
	m_ARGB32Buffer = new VulkanCoreBuffer<uint>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_ARGB128Buffer = new VulkanCoreBuffer<float4>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_NRM32Buffer = new VulkanCoreBuffer<uint>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );

	// Wavefront buffers
	m_AccumulationBuffer = new VulkanCoreBuffer<float4>( m_Device, pixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_PotentialContributionBuffer = new VulkanCoreBuffer<PotentialContribution>( m_Device, MAXPATHLENGTH * pixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );

	const auto limits = m_Device.GetPhysicalDevice().getProperties().limits;
	m_CombinedStateBuffer[0] = new VulkanCoreBuffer<float4>( m_Device, NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * 4, limits.minUniformBufferOffsetAlignment ), vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );
	m_CombinedStateBuffer[1] = new VulkanCoreBuffer<float4>( m_Device, NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * 4, limits.minUniformBufferOffsetAlignment ), vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer );

	// Light buffers
	m_AreaLightBuffer = new VulkanCoreBuffer<CoreLightTri>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_PointLightBuffer = new VulkanCoreBuffer<CorePointLight>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_SpotLightBuffer = new VulkanCoreBuffer<CoreSpotLight>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_DirectionalLightBuffer = new VulkanCoreBuffer<CoreDirectionalLight>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );

	// Blue Noise
	const uchar *data8 = (const uchar *)sob256_64;			// tables are 8 bit per entry
	uint *data32 = new uint[65536 * 5];						// we want a full uint per entry
	for (int i = 0; i < 65536; i++) data32[i] = data8[i]; // convert
	data8 = (uchar *)scr256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 65536] = data8[i];
	data8 = (uchar *)rnk256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 3 * 65536] = data8[i];
	m_BlueNoiseBuffer = new VulkanCoreBuffer<uint>( m_Device, 65536 * 5, vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_BlueNoiseBuffer->CopyToDevice( data32, m_BlueNoiseBuffer->GetSize() );
	delete[] data32;
}

void RenderCore::InitializeDescriptorSets()
{
	rtDescriptorSet->Bind( rtACCELERATION_STRUCTURE, { m_TopLevelAS->GetDescriptorBufferInfo() } );
	rtDescriptorSet->Bind( rtCAMERA, { m_UniformCamera->GetDescriptorBufferInfo() } );
	const auto limits = m_Device.GetPhysicalDevice().getProperties().limits;
	const vk::DeviceSize singleSize = NEXTMULTIPLEOF( m_ScrWidth * m_ScrHeight * sizeof( float4 ), limits.minUniformBufferOffsetAlignment );
	rtDescriptorSet->Bind( rtPATH_STATES, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 0, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 0, singleSize ) } );
	rtDescriptorSet->Bind( rtPATH_ORIGINS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( singleSize, singleSize ) } );
	rtDescriptorSet->Bind( rtPATH_DIRECTIONS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ) } );
	rtDescriptorSet->Bind( rtPOTENTIAL_CONTRIBUTIONS, { m_PotentialContributionBuffer->GetDescriptorBufferInfo() } );
	rtDescriptorSet->Bind( rtACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
	rtDescriptorSet->Bind( rtBLUENOISE, { m_BlueNoiseBuffer->GetDescriptorBufferInfo() } );

	// Update descriptor set contents
	rtDescriptorSet->UpdateSetContents();

	if (m_Meshes.empty())
	{
		m_Meshes.push_back( new CoreMesh( m_Device ) ); // Make sure at least 1 mesh exists
		m_MeshChanged.push_back( false );
	}
	if (m_TriangleBufferInfos.size() != m_Meshes.size()) // Recreate triangle buffer info for all
	{
		m_TriangleBufferInfos.resize( m_Meshes.size() );
		for (uint i = 0; i < m_Meshes.size(); i++) m_TriangleBufferInfos[i] = m_Meshes[i]->triangles->GetDescriptorBufferInfo();
	}
	else
	{
		for (uint i = 0; i < m_Meshes.size(); i++) // Update only those triangle buffer infos that have changed
			if (m_MeshChanged[i]) m_TriangleBufferInfos[i] = m_Meshes[i]->triangles->GetDescriptorBufferInfo();
	}

	shadeDescriptorSet->Bind( cCOUNTERS, { m_Counters->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cCAMERA, { m_UniformCamera->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cPATH_STATES, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 0, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 0, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_ORIGINS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( singleSize, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_DIRECTIONS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 2 * singleSize, singleSize ) } );
	shadeDescriptorSet->Bind( cPATH_THROUGHPUTS, { m_CombinedStateBuffer[0]->GetDescriptorBufferInfo( 3 * singleSize, singleSize ), m_CombinedStateBuffer[1]->GetDescriptorBufferInfo( 3 * singleSize, singleSize ) } );
	shadeDescriptorSet->Bind( cPOTENTIAL_CONTRIBUTIONS, { m_PotentialContributionBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cMATERIALS, { m_Materials->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cSKYBOX, { m_SkyboxImage->GetDescriptorImageInfo() } );
	shadeDescriptorSet->Bind( cTRIANGLES, m_TriangleBufferInfos );
	shadeDescriptorSet->Bind( cTRIANGLE_BUFFER_INDICES, { m_InstanceMeshMappingBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cINVERSE_TRANSFORMS, { m_InvTransformsBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cTEXTURE_ARGB32, { m_ARGB32Buffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cTEXTURE_ARGB128, { m_ARGB128Buffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cTEXTURE_NRM32, { m_NRM32Buffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cBLUENOISE, { m_BlueNoiseBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->UpdateSetContents();
	finalizeDescriptorSet->Bind( fACCUMULATION_BUFFER, { m_AccumulationBuffer->GetDescriptorBufferInfo() } );
	finalizeDescriptorSet->Bind( fUNIFORM_CONSTANTS, { m_UniformFinalizeParams->GetDescriptorBufferInfo() } );
	finalizeDescriptorSet->Bind( fOUTPUT, { m_OffscreenImage->GetDescriptorImageInfo() } );
	finalizeDescriptorSet->UpdateSetContents();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  Initialization.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
	VulkanShader::BaseFolder = R"(../../lib/RenderCore_Vulkan_RT/shaders/)";
	VulkanShader::BSDFFolder = R"(../../lib/sharedBSDFs/)";

	m_Initialized = false;
	RenderCore::instance = this;
	CreateInstance();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture *target, const uint spp )
{
	glFlush();
	glFinish();

	m_SamplesPP = spp;
	m_SamplesTaken = 0;
	m_ScrWidth = target->width;
	m_ScrHeight = target->height;

	if (!m_Initialized)
	{
		CreateDevice();
		InitRenderer();
		m_Initialized = true;
	}

	m_Device.WaitIdle();
	if (!m_InteropTexture)
	{
		// Create a bigger buffer than needed to prevent reallocating often
		m_InteropTexture = new VulkanGLTextureInterop( m_Device, m_ScrWidth, m_ScrHeight );
		auto cmdBuffer = m_Device.CreateOneTimeCmdBuffer();
		auto queue = m_Device.GetGraphicsQueue();
		m_InteropTexture->TransitionImageToInitialState( cmdBuffer, queue );
		cmdBuffer.Submit( queue, true );
		CreateOffscreenBuffers();   // Create off-screen render image
		ResizeBuffers();			// Resize path trace storage buffer
		InitializeDescriptorSets(); // Update descriptor sets with new target
	}
	else
	{
		m_InteropTexture->Resize( m_ScrWidth, m_ScrHeight );
		CreateOffscreenBuffers(); // Create off-screen render image
		ResizeBuffers();		  // Resize path trace storage buffer
	}

	glDeleteTextures( 1, &target->ID );
	target->ID = m_InteropTexture->GetID();
	CheckGL();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4 *vertexData, const int vertexCount, const int triangleCount, const CoreTri *triangles, const uint *alphaFlags )
{
	if (meshIdx >= m_Meshes.size())
	{
		m_Meshes.push_back( new CoreMesh( m_Device ) );
		m_MeshChanged.push_back( false );
	}

	m_Meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles, alphaFlags );
	m_MeshChanged[meshIdx] = true;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetInstance                                                    |
//  |  Set instance details.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4 &matrix )
{
	if (instanceIdx >= m_Instances.size())
	{
		m_Instances.emplace_back();

		if (m_InvTransformsBuffer->GetElementCount() < m_Instances.size())
		{
			auto newInvTransformsBuffer = new VulkanCoreBuffer<mat4>( m_Device, NEXTMULTIPLEOF( m_Instances.size(), 32 ),
				m_InvTransformsBuffer->GetMemoryProperties(),
				m_InvTransformsBuffer->GetBufferUsageFlags(),
				ON_DEVICE | ON_HOST );
			memcpy( newInvTransformsBuffer->GetHostBuffer(), m_InvTransformsBuffer->GetHostBuffer(), m_InvTransformsBuffer->GetSize() );
			delete m_InvTransformsBuffer;
			m_InvTransformsBuffer = newInvTransformsBuffer;
		}
		if (m_InstanceMeshMappingBuffer->GetElementCount() < m_Instances.size())
		{
			auto newInstanceMappingBuffer = new VulkanCoreBuffer<uint>( m_Device, NEXTMULTIPLEOF( m_Instances.size(), 32 ),
				m_InstanceMeshMappingBuffer->GetMemoryProperties(),
				m_InstanceMeshMappingBuffer->GetBufferUsageFlags(),
				ON_DEVICE | ON_HOST );
			memcpy( newInstanceMappingBuffer->GetHostBuffer(), m_InstanceMeshMappingBuffer->GetHostBuffer(), m_InstanceMeshMappingBuffer->GetSize() );
			delete m_InstanceMeshMappingBuffer;
			m_InstanceMeshMappingBuffer = newInstanceMappingBuffer;
		}
	}
	auto &curInstance = m_Instances.at( instanceIdx );

	curInstance.instanceId = instanceIdx;
	curInstance.mask = 0xFF;
	curInstance.instanceOffset = 0;
	curInstance.flags = (uint32_t)vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable;

	mat4 m = matrix.Inverted();

	// Update matrix
	m_InvTransformsBuffer->GetHostBuffer()[instanceIdx] = m;
	memcpy( curInstance.transform.data(), &matrix, 12 * sizeof( float ) );

	// Update mesh index
	m_InstanceMeshMappingBuffer->GetHostBuffer()[instanceIdx] = meshIdx;

	// Update acceleration structure handle
	curInstance.accelerationStructureHandle = m_Meshes.at( meshIdx )->accelerationStructure->GetHandle();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
	for (uint i = 0; i < m_Instances.size(); i++)
	{
		// Meshes might have changed in the mean time
		const auto meshIdx = m_InstanceMeshMappingBuffer->GetHostBuffer()[i];
		if (!m_MeshChanged.at( meshIdx )) continue;

		auto &instance = m_Instances.at( i );
		auto *mesh = m_Meshes.at( meshIdx );
		// Update acceleration structure handle
		instance.accelerationStructureHandle = mesh->accelerationStructure->GetHandle();
		assert( instance.accelerationStructureHandle );
	}

	bool triangleBuffersDirty = false;					   // Initially we presume triangle buffers are up to date
	if (m_TriangleBufferInfos.size() != m_Meshes.size()) // Update every triangle buffer info
	{
		triangleBuffersDirty = true;
		m_TriangleBufferInfos.resize( m_Meshes.size() );
		for (uint i = 0; i < m_Meshes.size(); i++) m_TriangleBufferInfos[i] = m_Meshes[i]->triangles->GetDescriptorBufferInfo(), m_MeshChanged[i] = false;
	}
	else
	{
		for (uint i = 0; i < m_Meshes.size(); i++) // Update only those buffer infos that have changed
		{
			if (m_MeshChanged.at( i )) // Check if mesh triangle buffer actually changed
			{
				m_TriangleBufferInfos.at( i ) = m_Meshes.at( i )->triangles->GetDescriptorBufferInfo(); // Set new buffer info flag
				triangleBuffersDirty = true;															// Set triangle buffer write flag
				m_MeshChanged.at( i ) = false;															// Reset mesh dirty flag
			}
		}
	}

	// Update triangle buffers
	if (triangleBuffersDirty) shadeDescriptorSet->Bind( cTRIANGLES, m_TriangleBufferInfos );

	m_InvTransformsBuffer->CopyToDevice(); // Update inverse transforms
	shadeDescriptorSet->Bind( cINVERSE_TRANSFORMS, { m_InvTransformsBuffer->GetDescriptorBufferInfo() } );

	m_InstanceMeshMappingBuffer->CopyToDevice();
	shadeDescriptorSet->Bind( cTRIANGLE_BUFFER_INDICES, { m_InstanceMeshMappingBuffer->GetDescriptorBufferInfo() } );

	if (m_TopLevelAS->GetInstanceCount() < m_Instances.size()) // Recreate top level AS in case our number of instances changed
	{
		delete m_TopLevelAS;
		m_TopLevelAS = new TopLevelAS( m_Device, FastestTrace, (uint32_t)m_Instances.size() );
		m_TopLevelAS->UpdateInstances( m_Instances );
		m_TopLevelAS->Build();

		rtDescriptorSet->Bind( rtACCELERATION_STRUCTURE, { m_TopLevelAS->GetDescriptorBufferInfo() } );
	}
	else if (!m_TopLevelAS->CanUpdate()) // Recreate top level AS in case it cannot be updated
	{
		delete m_TopLevelAS;
		m_TopLevelAS = new TopLevelAS( m_Device, FastTrace, (uint32_t)m_Instances.size() );
		m_TopLevelAS->UpdateInstances( m_Instances );
		m_TopLevelAS->Build();

		// Update descriptor set
		rtDescriptorSet->Bind( rtACCELERATION_STRUCTURE, { m_TopLevelAS->GetDescriptorBufferInfo() } );
	}
	else // Rebuild (refit) our top level AS
	{
		m_TopLevelAS->UpdateInstances( m_Instances );
		assert( m_TopLevelAS->CanUpdate() );
		m_TopLevelAS->Rebuild();

		// No descriptor write needed, same acceleration structure object
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTextures                                                    |
//  |  Set the texture data.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTextures( const CoreTexDesc *tex, const int textures )
{
	// Store descriptors, we need them later to assign texture indices to materials
	m_TexDescs = std::vector<CoreTexDesc>( tex, tex + textures );

	// Get buffers
	std::vector<uint> ARGB32Data;
	std::vector<float4> ARGB128Data;
	std::vector<uint> NRM32Data;

	size_t texelTotal32 = 0;
	size_t texelTotal128 = 0;
	size_t texelTotalNRM32 = 0;
	for (const auto &tex : m_TexDescs)
	{
		switch (tex.storage)
		{
		case TexelStorage::ARGB32: texelTotal32 += tex.pixelCount; break;
		case TexelStorage::ARGB128: texelTotal128 += tex.pixelCount; break;
		case TexelStorage::NRM32: texelTotalNRM32 += tex.pixelCount; break;
		}
	}

	ARGB32Data.resize( texelTotal32 );
	ARGB128Data.resize( texelTotal128 );
	NRM32Data.resize( texelTotalNRM32 );

	// Copy data to buffers
	texelTotal32 = 0;
	texelTotal128 = 0;
	texelTotalNRM32 = 0;
	for (auto &tex : m_TexDescs)
	{
		switch (tex.storage)
		{
		case TexelStorage::ARGB32:
		{
			auto destination = ARGB32Data.data() + texelTotal32;
			memcpy( destination, tex.idata, tex.pixelCount * sizeof( uint ) );
			tex.firstPixel = (uint)texelTotal32;
			texelTotal32 += tex.pixelCount;
			break;
		}
		case TexelStorage::ARGB128:
		{
			auto destination = ARGB128Data.data() + texelTotal128;
			memcpy( destination, tex.idata, tex.pixelCount * sizeof( uint ) );
			tex.firstPixel = (uint)texelTotal128;
			texelTotal128 += tex.pixelCount;
			break;
		}
		case TexelStorage::NRM32:
		{
			auto destination = NRM32Data.data() + texelTotalNRM32;
			memcpy( destination, tex.idata, tex.pixelCount * sizeof( uint ) );
			tex.firstPixel = (uint)texelTotalNRM32;
			texelTotalNRM32 += tex.pixelCount;
			break;
		}
		}
	}

	delete m_ARGB32Buffer;
	delete m_ARGB128Buffer;
	delete m_NRM32Buffer;

	const auto ARGB32Size = std::max( ARGB32Data.size(), size_t( 1 ) );
	const auto ARGB128Size = std::max( ARGB128Data.size(), size_t( 1 ) );
	const auto NRM32Size = std::max( NRM32Data.size(), size_t( 1 ) );

	// Texture buffers
	m_ARGB32Buffer = new VulkanCoreBuffer<uint>( m_Device, ARGB32Size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_ARGB128Buffer = new VulkanCoreBuffer<float4>( m_Device, ARGB128Size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_NRM32Buffer = new VulkanCoreBuffer<uint>( m_Device, NRM32Size, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );

	if (!ARGB32Data.empty()) m_ARGB32Buffer->CopyToDevice( ARGB32Data.data(), m_ARGB32Buffer->GetSize() );
	if (!ARGB128Data.empty()) m_ARGB128Buffer->CopyToDevice( ARGB128Data.data(), m_ARGB128Buffer->GetSize() );
	if (!NRM32Data.empty()) m_NRM32Buffer->CopyToDevice( NRM32Data.data(), m_NRM32Buffer->GetSize() );

	shadeDescriptorSet->Bind( cTEXTURE_ARGB32, { m_ARGB32Buffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cTEXTURE_ARGB128, { m_ARGB128Buffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cTEXTURE_NRM32, { m_NRM32Buffer->GetDescriptorBufferInfo() } );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial *mat, const CoreMaterialEx *matEx, const int materialCount )
{
	delete m_Materials;
	std::vector<CoreMaterial> materialData( materialCount );
	materialData.resize( materialCount );
	memcpy( materialData.data(), mat, materialCount * sizeof( CoreMaterial ) );

	const std::vector<CoreTexDesc> &texDescs = m_TexDescs;

	for (int i = 0; i < materialCount; i++)
	{
		CoreMaterial &mat = materialData.at( i );
		const CoreMaterialEx &ids = matEx[i];
		if (ids.texture[0] != -1)
			mat.texaddr0 = texDescs[ids.texture[0]].firstPixel;
		if (ids.texture[1] != -1) mat.texaddr1 = texDescs[ids.texture[1]].firstPixel;
		if (ids.texture[2] != -1) mat.texaddr2 = texDescs[ids.texture[2]].firstPixel;
		if (ids.texture[3] != -1) mat.nmapaddr0 = texDescs[ids.texture[3]].firstPixel;
		if (ids.texture[4] != -1) mat.nmapaddr1 = texDescs[ids.texture[4]].firstPixel;
		if (ids.texture[5] != -1) mat.nmapaddr2 = texDescs[ids.texture[5]].firstPixel;
		if (ids.texture[6] != -1) mat.smapaddr = texDescs[ids.texture[6]].firstPixel;
		if (ids.texture[7] != -1) mat.rmapaddr = texDescs[ids.texture[7]].firstPixel;
		//if ( ids.texture[8] != -1 ) mat.texaddr0 = texDescs[ids.texture[8]].firstPixel; // second roughness map is not used
		if (ids.texture[9] != -1) mat.cmapaddr = texDescs[ids.texture[9]].firstPixel;
		if (ids.texture[10] != -1) mat.amapaddr = texDescs[ids.texture[10]].firstPixel;
	}

	m_Materials = new VulkanCoreBuffer<CoreMaterial>( m_Device, materialCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	m_Materials->CopyToDevice( materialData.data(), materialCount * sizeof( CoreMaterial ) );

	shadeDescriptorSet->Bind( cMATERIALS, { m_Materials->GetDescriptorBufferInfo() } );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetLights( const CoreLightTri *areaLights, const int areaLightCount,
	const CorePointLight *pointLights, const int pointLightCount,
	const CoreSpotLight *spotLights, const int spotLightCount,
	const CoreDirectionalLight *directionalLights, const int directionalLightCount )
{
	static_assert(sizeof( CoreLightTri ) == sizeof( CoreLightTri4 ));
	static_assert(sizeof( CorePointLight ) == sizeof( CorePointLight4 ));
	static_assert(sizeof( CoreSpotLight ) == sizeof( CoreSpotLight4 ));
	static_assert(sizeof( CoreDirectionalLight ) == sizeof( CoreDirectionalLight4 ));

	m_LightCounts = make_uint4(
		std::max( areaLightCount, 1 ),
		std::max( pointLightCount, 1 ),
		std::max( spotLightCount, 1 ),
		std::max( directionalLightCount, 1 ) );

	if (m_AreaLightBuffer->GetSize() < (areaLightCount * sizeof( CoreLightTri4 )))
	{
		delete m_AreaLightBuffer;
		m_AreaLightBuffer = new VulkanCoreBuffer<CoreLightTri>( m_Device, m_LightCounts.x, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	}
	if (m_PointLightBuffer->GetSize() < (pointLightCount * sizeof( CorePointLight4 )))
	{
		delete m_PointLightBuffer;
		m_PointLightBuffer = new VulkanCoreBuffer<CorePointLight>( m_Device, m_LightCounts.y, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	}
	if (m_SpotLightBuffer->GetSize() < (spotLightCount * sizeof( CoreSpotLight4 )))
	{
		delete m_SpotLightBuffer;
		m_SpotLightBuffer = new VulkanCoreBuffer<CoreSpotLight>( m_Device, m_LightCounts.z, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	}
	if (m_DirectionalLightBuffer->GetSize() < (directionalLightCount * sizeof( CoreDirectionalLight4 )))
	{
		delete m_DirectionalLightBuffer;
		m_DirectionalLightBuffer = new VulkanCoreBuffer<CoreDirectionalLight>( m_Device, m_LightCounts.w, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	}

	// Copy to device in case lights exist
	if (areaLightCount > 0) m_AreaLightBuffer->CopyToDevice( areaLights, m_AreaLightBuffer->GetSize() );
	if (pointLightCount > 0) m_PointLightBuffer->CopyToDevice( pointLights, m_PointLightBuffer->GetSize() );
	if (spotLightCount > 0) m_SpotLightBuffer->CopyToDevice( spotLights, m_SpotLightBuffer->GetSize() );
	if (directionalLightCount > 0) m_DirectionalLightBuffer->CopyToDevice( directionalLights, m_DirectionalLightBuffer->GetSize() );

	// Update descriptor set
	shadeDescriptorSet->Bind( cAREALIGHT_BUFFER, { m_AreaLightBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cPOINTLIGHT_BUFFER, { m_PointLightBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cSPOTLIGHT_BUFFER, { m_SpotLightBuffer->GetDescriptorBufferInfo() } );
	shadeDescriptorSet->Bind( cDIRECTIONALLIGHT_BUFFER, { m_DirectionalLightBuffer->GetDescriptorBufferInfo() } );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3 *pixels, const uint width, const uint height )
{
	std::vector<float4> data( size_t( width * height ) );
	for (uint i = 0; i < (width * height); i++) data[i] = make_float4( pixels[i].x, pixels[i].y, pixels[i].z, 0.0f );

	delete m_SkyboxImage;
	// Create a Vulkan image that can be sampled
	m_SkyboxImage = new VulkanImage( m_Device, vk::ImageType::e2D, vk::Format::eR32G32B32A32Sfloat,
		vk::Extent3D( width, height, 1 ), vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal );

	vk::ImageSubresourceRange range{};
	range.aspectMask = vk::ImageAspectFlagBits::eColor;
	range.baseMipLevel = 0;
	range.levelCount = 1;
	range.baseArrayLayer = 0;
	range.layerCount = 1;

	// Set image data
	m_SkyboxImage->SetData( data, width, height );
	// Create an image view that can be sampled
	m_SkyboxImage->CreateImageView( vk::ImageViewType::e2D, vk::Format::eR32G32B32A32Sfloat, range );
	// Create sampler to be used in shader
	m_SkyboxImage->CreateSampler( vk::Filter::eLinear, vk::Filter::eNearest, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge );
	// Make sure image is usable by shader
	m_SkyboxImage->TransitionToLayout( vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlags() );
	// Update descriptor set
	shadeDescriptorSet->Bind( cSKYBOX, { m_SkyboxImage->GetDescriptorImageInfo() } );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char *name, const float value )
{
	// we have no settings yet
	// TODO
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid &view, const Convergence converge, const float brightness, const float contrast )
{
	VulkanCamera &camera = m_UniformCamera->GetData()[0];
	Counters &c = m_Counters->GetHostBuffer()[0];

	auto queue = m_Device.GetGraphicsQueue();
	if (converge == Restart || m_FirstConvergingFrame)
	{
		m_SamplesTaken = 0;
		m_FirstConvergingFrame = true; // if we switch to converging, it will be the first converging frame.
	}
	if (converge == Converge) m_FirstConvergingFrame = false;
	const bool recordCommandBuffers = rtDescriptorSet->IsDirty() || shadeDescriptorSet->IsDirty() || finalizeDescriptorSet->IsDirty() || m_First; // Before we render we potentially have to update our command buffers
	m_First = false;
	if (recordCommandBuffers)
	{
		queue.waitIdle();
		rtDescriptorSet->UpdateSetContents();						   // Update ray trace descriptor set if needed
		shadeDescriptorSet->UpdateSetContents();					   // Update shade descriptor set if needed
		finalizeDescriptorSet->UpdateSetContents();					   // Update finalize descriptor set if needed
		if (recordCommandBuffers || m_First) RecordCommandBuffers(); // Record command buffers if descriptor set generator was dirty
	}

	// Get queue and command buffer for this frame
	OneTimeCommandBuffer cmdBuffer = m_Device.CreateOneTimeCmdBuffer();

	uint pathCount = m_ScrWidth * m_ScrHeight * 1;
	uint32_t pushConstant[3];

	// Reset stats
	coreStats.primaryRayCount = 0;
	coreStats.deepRayCount = 0;
	coreStats.totalExtensionRays = 0;
	coreStats.totalShadowRays = 0;
	coreStats.bounce1RayCount = 0;
	coreStats.deepRayCount = 0;
	coreStats.traceTime0 = 0.0f;
	coreStats.traceTime1 = 0.0f;
	coreStats.traceTimeX = 0.0f;
	coreStats.shadeTime = 0.0f;
	coreStats.shadowTraceTime = 0.0f;
	coreStats.renderTime = 0.0f;

	Timer t, frameTime{};

	for (uint i = 0; i < m_SamplesPP; i++)
	{
		// Initialize out camera
		camera = VulkanCamera( view, m_SamplesTaken, STAGE_PRIMARY_RAY ); // Reset camera
		camera.scrwidth = m_ScrWidth;
		camera.scrheight = m_ScrHeight;
		m_UniformCamera->CopyToDevice();

		// Initialize counters
		c.Reset( m_LightCounts, m_ScrWidth, m_ScrHeight, 10.0f, 1e-4f );
		c.probePixelIdx = m_ProbePos.x + m_ProbePos.y * m_ScrWidth;
		m_Counters->CopyToDevice();

		if (i != 0) cmdBuffer.Begin();
		// Primary ray stage
		if (m_SamplesTaken <= 1) cmdBuffer->fillBuffer( *m_AccumulationBuffer, 0, m_ScrWidth * m_ScrHeight * sizeof( float4 ), 0 );
		pushConstant[0] = c.pathLength;
		pushConstant[1] = pathCount;
		pushConstant[2] = STAGE_PRIMARY_RAY;
		rtPipeline->RecordPushConstant( cmdBuffer, 0, 3 * sizeof( uint32_t ), pushConstant ); // Push intersection stage to shader
		rtPipeline->RecordTraceCommand( cmdBuffer, NEXTMULTIPLEOF( pathCount, 64 ) );

		// Submit primary rays to queue
		t.reset();
		cmdBuffer.Submit( queue, true );
		queue.waitIdle();
		coreStats.traceTime0 += t.elapsed();
		coreStats.primaryRayCount += pathCount;

		// Record shade stage
		cmdBuffer.Begin();
		shadePipeline->RecordPushConstant( cmdBuffer, 0, 2 * sizeof( uint32_t ), pushConstant );
		shadePipeline->RecordDispatchCommand( cmdBuffer, NEXTMULTIPLEOF( pathCount, 64 ) );
		// Make sure shading finished before copying counters
		cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {} );
		RecordCopyCommand( m_CounterTransferBuffer, m_Counters, cmdBuffer );

		// Submit command buffer
		cmdBuffer.Submit( queue, true );
		t.reset();
		queue.waitIdle();
		coreStats.shadeTime += t.elapsed();

		// Prepare extension rays
		auto *counters = m_CounterTransferBuffer->Map(); // Get Counters
		pathCount = counters->extensionRays;			 // Get number of extension rays generated
		c.extensionRays = 0;							 // Reset extension counter
		c.pathLength++;									 // Increment path length
		c.shadowRays = counters->shadowRays;			 // Make sure we keep count of the number of shadow rays

		coreStats.probedDist = c.probedDist;
		coreStats.probedInstid = c.probedInstid;
		coreStats.probedTriid = c.probedTriid;

		memcpy( counters, &c, sizeof( Counters ) ); // Reset counters
		m_CounterTransferBuffer->Unmap();			// Unmap buffer so that it can be copied from again
		coreStats.totalExtensionRays += pathCount;  // Update stats
		coreStats.bounce1RayCount += pathCount;

		for (uint i = 2; i <= MAXPATHLENGTH; i++)
		{
			if (pathCount > 0)
			{
				// Extension ray stage
				cmdBuffer.Begin();
				RecordCopyCommand( m_Counters, m_CounterTransferBuffer, cmdBuffer );
				//m_Counters->GetBuffer()->RecordCopyToDeviceCommand( m_CounterTransferBuffer, sizeof( Counters ), cmdBuffer ); // Copy counters
				pushConstant[0] = c.pathLength;
				pushConstant[1] = pathCount;
				pushConstant[2] = STAGE_SECONDARY_RAY;
				rtPipeline->RecordPushConstant( cmdBuffer, 0, 3 * sizeof( uint32_t ), pushConstant ); // Push intersection stage to shader
				cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eRayTracingShaderNV,
					{}, {}, {}, {} ); // Make sure counters update transfer finished
				rtPipeline->RecordTraceCommand( cmdBuffer, NEXTMULTIPLEOF( pathCount, 64 ) );

				t.reset();
				cmdBuffer.Submit( queue, true );		   // Run command buffer
				coreStats.totalExtensionRays += pathCount; // Update stats

				if (i == 2)
				{
					coreStats.bounce1RayCount = pathCount;
					coreStats.traceTime1 += t.elapsed();
				}
				else
				{
					coreStats.deepRayCount += pathCount;
					coreStats.traceTimeX += t.elapsed();
				}

				cmdBuffer.Begin();
				// Shade extension rays
				shadePipeline->RecordPushConstant( cmdBuffer, 0, 2 * sizeof( uint32_t ), pushConstant );
				shadePipeline->RecordDispatchCommand( cmdBuffer, NEXTMULTIPLEOF( pathCount, 64 ) );
				cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {} ); // Make sure shade stage finished
				// Copy counters to host
				RecordCopyCommand( m_CounterTransferBuffer, m_Counters, cmdBuffer );

				// Submit command buffer
				t.reset();
				cmdBuffer.Submit( queue, true ); // Run command buffer
				coreStats.shadeTime += t.elapsed();

				counters = (Counters *)m_CounterTransferBuffer->Map(); // Get Counters
				pathCount = counters->extensionRays;				   // Get number of extension rays generated
				c.pathCount = pathCount;
				c.extensionRays = 0;						// Reset extension counter
				c.pathLength++;								// Increment path length
				c.shadowRays = counters->shadowRays;		// Make sure we keep count of the number of shadow rays
				memcpy( counters, &c, sizeof( Counters ) ); // Reset counters
				m_CounterTransferBuffer->Unmap();			// Unmap buffer so that it can be copied from again
				coreStats.totalExtensionRays += pathCount;  // Update stats
			}
			else
			{
				break; // All paths were terminated
			}
		}

		// Prepare shadow rays
		counters = (Counters *)m_CounterTransferBuffer->Map(); // Get Counters
		pathCount = counters->shadowRays;					   // Get number of shadow rays generated
		m_CounterTransferBuffer->Unmap();					   // Unmap buffer so that it can be copied from again
		if (pathCount > 0)
		{
			cmdBuffer.Begin();
			pushConstant[0] = c.pathLength;
			pushConstant[1] = pathCount;
			pushConstant[2] = STAGE_SHADOW_RAY;
			rtPipeline->RecordPushConstant( cmdBuffer, 0, 3 * sizeof( uint32_t ), pushConstant ); // Push intersection stage to shader
			rtPipeline->RecordTraceCommand( cmdBuffer, NEXTMULTIPLEOF( pathCount, 64 ) );

			// Submit shadow rays
			t.reset();
			cmdBuffer.Submit( queue, true ); // Run command buffer
			coreStats.shadowTraceTime += t.elapsed();
			coreStats.totalShadowRays += pathCount;
		}

		m_SamplesTaken++;
	}

	// Initialize params for finalize stage
	VulkanFinalizeParams &params = m_UniformFinalizeParams->GetData()[0];
	params = VulkanFinalizeParams( m_ScrWidth, m_ScrHeight, m_SamplesTaken, brightness, contrast );
	m_UniformFinalizeParams->CopyToDevice();

	cmdBuffer.Begin();
	// Make sure off-screen render image is ready to be used
	const auto subresourceRange = vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 );
	const auto imageMemoryBarrier = vk::ImageMemoryBarrier( vk::AccessFlags(), vk::AccessFlagBits::eShaderWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
		VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_OffscreenImage->GetImage(), subresourceRange );
	cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier );
	// Dispatch finalize image shader
	finalizePipeline->RecordDispatchCommand( cmdBuffer, m_ScrWidth, m_ScrHeight );

	// Run command buffer
	cmdBuffer.Submit( queue, true );

	// Ensure OpenGL finished
	glFlush(), glFinish();

	// Blit image to OpenGL texture
	m_Device.SubmitCommandBuffer( m_BlitCommandBuffer, queue, nullptr, vk::PipelineStageFlagBits::eColorAttachmentOutput );
	queue.waitIdle();

	coreStats.renderTime = frameTime.elapsed();
	// OneTimeCommandBuffer automatically gets freed once it goes out of scope
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	glFlush(), glFinish();

	m_Device->waitIdle();

	if (m_CounterTransferBuffer) delete m_CounterTransferBuffer;

	if (rtPipeline) delete rtPipeline;
	if (rtDescriptorSet) delete rtDescriptorSet;

	if (shadePipeline) delete shadePipeline;
	if (shadeDescriptorSet) delete shadeDescriptorSet;

	if (finalizePipeline) delete finalizePipeline;
	if (finalizeDescriptorSet) delete finalizeDescriptorSet;

	if (m_BlitCommandBuffer) m_Device.FreeCommandBuffer( m_BlitCommandBuffer );
	if (m_TopLevelAS) delete m_TopLevelAS;
	for (auto *mesh : m_Meshes) delete mesh;

	if (m_OffscreenImage != nullptr) delete m_OffscreenImage;

	if (m_InvTransformsBuffer) delete m_InvTransformsBuffer;

	if (m_AreaLightBuffer) delete m_AreaLightBuffer;
	if (m_PointLightBuffer) delete m_PointLightBuffer;
	if (m_SpotLightBuffer) delete m_SpotLightBuffer;
	if (m_DirectionalLightBuffer) delete m_DirectionalLightBuffer;

	if (m_BlueNoiseBuffer) delete m_BlueNoiseBuffer;
	if (m_CombinedStateBuffer[0]) delete m_CombinedStateBuffer[0];
	if (m_CombinedStateBuffer[1]) delete m_CombinedStateBuffer[1];
	if (m_InstanceMeshMappingBuffer) delete m_InstanceMeshMappingBuffer;
	if (m_Counters) delete m_Counters;
	if (m_SkyboxImage) delete m_SkyboxImage;
	if (m_UniformCamera) delete m_UniformCamera;
	if (m_UniformFinalizeParams) delete m_UniformFinalizeParams;
	if (m_AccumulationBuffer) delete m_AccumulationBuffer;
	if (m_PotentialContributionBuffer) delete m_PotentialContributionBuffer;
	if (m_Materials) delete m_Materials;
	if (m_InteropTexture) delete m_InteropTexture;
	if (m_VkDebugMessenger) m_VkInstance.destroyDebugUtilsMessengerEXT( m_VkDebugMessenger, nullptr, dynamicDispatcher );
	// Vulkan device & Vulkan instance automatically get freed when this class gets destroyed
}

} // namespace lh2core

// EOF