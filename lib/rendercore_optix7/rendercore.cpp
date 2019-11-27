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
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace lh2core {

// forward declaration of cuda code
const surfaceReference* renderTargetRef();
void finalizeRender( const float4* accumulator, const int w, const int h, const int spp );
void shade( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, const float4* hits, float4* connections,
	const uint R0, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const float3 p1, const float3 p2, const float3 p3, const float3 pos );
void InitCountersForExtend( int pathCount );
void InitCountersSubsequent();

// setters / getters
void SetInstanceDescriptors( CoreInstanceDesc* p );
void SetMaterialList( CoreMaterial* p );
void SetAreaLights( CoreLightTri* p );
void SetPointLights( CorePointLight* p );
void SetSpotLights( CoreSpotLight* p );
void SetDirectionalLights( CoreDirectionalLight* p );
void SetLightCounts( int area, int point, int spot, int directional );
void SetARGB32Pixels( uint* p );
void SetARGB128Pixels( float4* p );
void SetNRM32Pixels( uint* p );
void SetSkyPixels( float3* p );
void SetSkySize( int w, int h );
void SetPathStates( PathState* p );
void SetDebugData( float4* p );
void SetGeometryEpsilon( float e );
void SetClampValue( float c );
void SetCounters( Counters* p );

} // namespace lh2core

using namespace lh2core;

OptixDeviceContext RenderCore::optixContext = 0;
struct SBTRecord { __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

const char *ParseOptixError( OptixResult r )
{
	switch ( r )
	{
	case 0: return "NO ERROR";
	case 7001: return "OPTIX_ERROR_INVALID_VALUE";
	case 7002: return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
	case 7003: return "OPTIX_ERROR_INVALID_OPERATION";
	case 7004: return "OPTIX_ERROR_FILE_IO_ERROR";
	case 7005: return "OPTIX_ERROR_INVALID_FILE_FORMAT";
	case 7010: return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";
	case 7011: return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";
	case 7012: return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";
	case 7013: return "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";
	case 7050: return "OPTIX_ERROR_LAUNCH_FAILURE";
	case 7051: return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";
	case 7052: return "OPTIX_ERROR_CUDA_NOT_INITIALIZED";
	case 7200: return "OPTIX_ERROR_INVALID_PTX";
	case 7201: return "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";
	case 7202: return "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";
	case 7203: return "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";
	case 7204: return "OPTIX_ERROR_INVALID_FUNCTION_USE";
	case 7205: return "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";
	case 7250: return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";
	case 7251: return "OPTIX_ERROR_PIPELINE_LINK_ERROR";
	case 7299: return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";
	case 7300: return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";
	case 7301: return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";
	case 7400: return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";
	case 7800: return "OPTIX_ERROR_NOT_SUPPORTED";
	case 7801: return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
	case 7802: return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
	case 7803: return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
	case 7804: return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
	case 7805: return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
	case 7900: return "OPTIX_ERROR_CUDA_ERROR";
	case 7990: return "OPTIX_ERROR_INTERNAL_ERROR";
	case 7999: return "OPTIX_ERROR_UNKNOWN";
	default: return "UNKNOWN ERROR";
	};
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetProbePos                                                    |
//  |  Set the pixel for which the triid will be captured.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetProbePos( int2 pos )
{
	probePos = pos; // triangle id for this pixel will be stored in coreStats
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::CreateOptixContext                                             |
//  |  Optix 7 initialization.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
	printf( "[%i][%s]: %s\n", level, tag, message );
}
void RenderCore::CreateOptixContext( int cc )
{
	// prepare the optix context
	cudaFree( 0 );
	CUcontext cu_ctx = 0; // zero means take the current context
	CHK_OPTIX( optixInit() );
	OptixDeviceContextOptions contextOptions = {};
	contextOptions.logCallbackFunction = &context_log_cb;
	contextOptions.logCallbackLevel = 4;
	CHK_OPTIX( optixDeviceContextCreate( cu_ctx, &contextOptions, &optixContext ) );
	cudaMalloc( (void**)(&d_params), sizeof( Params ) );

	// load and compile PTX
	string ptx;
	if (NeedsRecompile( "../../lib/RenderCore_Optix7/optix/", ".optix.turing.cu.ptx", ".optix.cu", "../../RenderSystem/common_settings.h", "../core_settings.h" ))
	{
		CUDATools::compileToPTX( ptx, TextFileRead( "../../lib/RenderCore_Optix7/optix/.optix.cu" ).c_str(), "../../lib/RenderCore_Optix7/optix", cc, 7 );
		if (cc / 10 == 7) TextFileWrite( ptx, "../../lib/RenderCore_Optix7/optix/.optix.turing.cu.ptx" );
		else if (cc / 10 == 6) TextFileWrite( ptx, "../../lib/RenderCore_Optix7/optix/.optix.pascal.cu.ptx" );
		else if (cc / 10 == 5) TextFileWrite( ptx, "../../lib/RenderCore_Optix7/optix/.optix.maxwell.cu.ptx" );
		printf( "recompiled .optix.cu.\n" );
	}
	else
	{
		const char *file = NULL;
		if (cc / 10 == 7) file = "../../lib/RenderCore_Optix7/optix/.optix.turing.cu.ptx";
		else if (cc / 10 == 6) file = "../../lib/RenderCore_Optix7/optix/.optix.pascal.cu.ptx";
		else if (cc / 10 == 5) file = "../../lib/RenderCore_Optix7/optix/.optix.maxwell.cu.ptx";
		FILE* f;
#ifdef _MSC_VER
		fopen_s( &f, file, "rb" );
#else
		f = fopen( file, "rb" );
#endif
		int len;
		fread( &len, 1, 4, f );
		char* t = new char[len];
		fread( t, 1, len, f );
		fclose( f );
		ptx = string( t );
		delete t;
	}

	// create the optix module
	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
	OptixPipelineCompileOptions pipeCompileOptions = {};
	pipeCompileOptions.usesMotionBlur = false;
	pipeCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipeCompileOptions.numPayloadValues = 4;
	pipeCompileOptions.numAttributeValues = 2;
	pipeCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipeCompileOptions.pipelineLaunchParamsVariableName = "params";
	char log[2048];
	size_t logSize = sizeof( log );
	CHK_OPTIX_LOG( optixModuleCreateFromPTX( optixContext, &module_compile_options, &pipeCompileOptions,
		ptx.c_str(), ptx.size(), log, &logSize, &ptxModule ) );

	// create program groups
	OptixProgramGroupOptions groupOptions = {};
	OptixProgramGroupDesc group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	group.raygen.module = ptxModule;
	group.raygen.entryFunctionName = "__raygen__rg";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAYGEN] ) );
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	group.miss.module = nullptr; // NULL miss program for extension rays
	group.miss.entryFunctionName = nullptr;
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAD_MISS] ) );
	group.miss.module = ptxModule;
	group.miss.entryFunctionName = "__miss__occlusion";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[OCC_MISS] ) );
	group = {};
	group.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	group.hitgroup.moduleCH = ptxModule;
	group.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[RAD_HIT] ) );
	group.hitgroup.moduleCH = nullptr;
	group.hitgroup.entryFunctionNameCH = nullptr; // NULL hit program for shadow rays
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixProgramGroupCreate( optixContext, &group, 1, &groupOptions, log, &logSize, &progGroup[OCC_HIT] ) );

	// create the pipeline
	OptixPipelineLinkOptions linkOptions = {};
	linkOptions.maxTraceDepth = 1;
	linkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
	linkOptions.overrideUsesMotionBlur = false;
	logSize = sizeof( log );
	CHK_OPTIX_LOG( optixPipelineCreate( optixContext, &pipeCompileOptions, &linkOptions, progGroup, 5, log, &logSize, &pipeline ) );
	// calculate the stack sizes, so we can specify all parameters to optixPipelineSetStackSize
	OptixStackSizes stack_sizes = {};
	for (int i = 0; i < 5; i++) optixUtilAccumulateStackSizes( progGroup[i], &stack_sizes );
	uint32_t ss0, ss1, ss2;
	CHK_OPTIX( optixUtilComputeStackSizes( &stack_sizes, 1, 0, 0, &ss0, &ss1, &ss2 ) );
	CHK_OPTIX( optixPipelineSetStackSize( pipeline, ss0, ss1, ss2, 2 ) );

	// create the shader binding table
	SBTRecord rsbt[5] = {}; // , ms_sbt[2], hg_sbt[2];
	for( int i = 0; i < 5; i++ ) optixSbtRecordPackHeader( progGroup[i], &rsbt[i] );
	sbt.raygenRecord = (CUdeviceptr)(new CoreBuffer<SBTRecord>( 1, ON_DEVICE, &rsbt[0] ))->DevPtr();
	sbt.missRecordBase = (CUdeviceptr)(new CoreBuffer<SBTRecord>( 2, ON_DEVICE, &rsbt[1] ))->DevPtr();
	sbt.hitgroupRecordBase = (CUdeviceptr)(new CoreBuffer<SBTRecord>( 2, ON_DEVICE, &rsbt[3] ))->DevPtr();
	sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sizeof( SBTRecord );
	sbt.missRecordCount = sbt.hitgroupRecordCount = 2;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  Initialization.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
	// select the fastest device
	uint device = CUDATools::FastestDevice();
	cudaSetDevice( device );
	cudaDeviceProp properties;
	cudaGetDeviceProperties( &properties, device );
	coreStats.SMcount = SMcount = properties.multiProcessorCount;
	coreStats.ccMajor = properties.major;
	coreStats.ccMinor = properties.minor;
	computeCapability = coreStats.ccMajor * 10 + coreStats.ccMinor;
	coreStats.VRAM = (uint)(properties.totalGlobalMem >> 20);
	coreStats.deviceName = new char[strlen( properties.name ) + 1];
	memcpy( coreStats.deviceName, properties.name, strlen( properties.name ) + 1 );
	printf( "running on GPU: %s (%i SMs, %iGB VRAM)\n", coreStats.deviceName, coreStats.SMcount, (int)(coreStats.VRAM >> 10) );
	// initialize Optix7
	CreateOptixContext( computeCapability );
	// render settings
	SetClampValue( 10.0f );
	// prepare counters for persistent threads
	counterBuffer = new CoreBuffer<Counters>( 1, ON_HOST | ON_DEVICE );
	SetCounters( counterBuffer->DevPtr() );
	// prepare the bluenoise data
	const uchar* data8 = (const uchar*)sob256_64; // tables are 8 bit per entry
	uint* data32 = new uint[65536 * 5]; // we want a full uint per entry
	for (int i = 0; i < 65536; i++) data32[i] = data8[i]; // convert
	data8 = (uchar*)scr256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 65536] = data8[i];
	data8 = (uchar*)rnk256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 3 * 65536] = data8[i];
	blueNoise = new CoreBuffer<uint>( 65536 * 5, ON_DEVICE, data32 );
	params.blueNoise = blueNoise->DevPtr();
	delete data32;
	// preallocate optix instance descriptor array
	instanceArray = new CoreBuffer<OptixInstance>( 16 /* will grow if needed */, ON_HOST | ON_DEVICE );
	// allow CoreMeshes to access the core
	CoreMesh::renderCore = this;
	// prepare timing events
	for( int i = 0; i < MAXPATHLENGTH; i++ )
	{
		cudaEventCreate( &shadeStart[i] );
		cudaEventCreate( &shadeEnd[i] );
		cudaEventCreate( &traceStart[i] );
		cudaEventCreate( &traceEnd[i] );
	}
	cudaEventCreate( &shadowStart );
	cudaEventCreate( &shadowEnd );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture* target, const uint spp )
{
	// synchronize OpenGL viewport
	scrwidth = target->width;
	scrheight = target->height;
	scrspp = spp;
	renderTarget.SetTexture( target );
	bool firstFrame = (maxPixels == 0);
	// notify CUDA about the texture
	renderTarget.LinkToSurface( renderTargetRef() );
	// see if we need to reallocate our buffers
	bool reallocate = false;
	if (scrwidth * scrheight > maxPixels || spp != currentSPP)
	{
		maxPixels = scrwidth * scrheight;
		maxPixels += maxPixels >> 4; // reserve a bit extra to prevent frequent reallocs
		currentSPP = spp;
		reallocate = true;
	}
	// notify OptiX about the new screen size
	params.scrsize = make_int3( scrwidth, scrheight, scrspp );
	if (reallocate)
	{
		// reallocate buffers
		delete connectionBuffer;
		delete accumulator;
		delete hitBuffer;
		delete pathStateBuffer;
		connectionBuffer = new CoreBuffer<float4>( maxPixels * scrspp * 3 * 2, ON_DEVICE );
		accumulator = new CoreBuffer<float4>( maxPixels, ON_DEVICE );
		hitBuffer = new CoreBuffer<float4>( maxPixels * scrspp, ON_DEVICE );
		pathStateBuffer = new CoreBuffer<float4>( maxPixels * scrspp * 3, ON_DEVICE );
		params.connectData = connectionBuffer->DevPtr();
		params.accumulator = accumulator->DevPtr();
		params.hitData = hitBuffer->DevPtr();
		params.pathStates = pathStateBuffer->DevPtr();
		printf( "buffers resized for %i pixels @ %i samples.\n", maxPixels, scrspp );
	}
	// clear the accumulator
	accumulator->Clear( ON_DEVICE );
	samplesTaken = 0;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags )
{
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new CoreMesh pointers being pushed into the meshes vector.
	// Subsequent mesh changes will be applied to existing CoreMeshes. This is deliberately
	// minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
	if (meshIdx >= meshes.size()) meshes.push_back( new CoreMesh() );
	meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles, alphaFlags );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetInstance                                                    |
//  |  Set instance details.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4& matrix )
{
	// A '-1' mesh denotes the end of the instance stream;
	// adjust the instances vector if we have more.
	if (meshIdx == -1)
	{
		if (instances.size() > instanceIdx) instances.resize( instanceIdx );
		return;
	}
	// For the first frame, instances are added to the instances vector.
	// For subsequent frames existing slots are overwritten / updated.
	if (instanceIdx >= instances.size())
	{
		// create a geometry instance
		CoreInstance* newInstance = new CoreInstance();
		memset( &newInstance->instance, 0, sizeof( OptixInstance ) );
		newInstance->instance.flags = OPTIX_INSTANCE_FLAG_NONE;
		newInstance->instance.instanceId = instanceIdx;
		newInstance->instance.sbtOffset = 0;
		newInstance->instance.visibilityMask = 255;
		newInstance->instance.traversableHandle = meshes[meshIdx]->gasHandle;
		memcpy( newInstance->transform, &matrix, 12 * sizeof( float ) );
		memcpy( newInstance->instance.transform, &matrix, 12 * sizeof( float ) );
		instances.push_back( newInstance );
	}
	// update the matrices for the transform
	memcpy( instances[instanceIdx]->transform, &matrix, 12 * sizeof( float ) );
	memcpy( instances[instanceIdx]->instance.transform, &matrix, 12 * sizeof( float ) );
	// set/update the mesh for this instance
	instances[instanceIdx]->mesh = meshIdx;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
	// resize instance array if more space is needed
	if (instances.size() > (size_t)instanceArray->GetSize())
	{
		delete instanceArray;
		instanceArray = new CoreBuffer<OptixInstance>( instances.size() + 4, ON_HOST | ON_DEVICE );
	}
	// copy instance descriptors to the array, sync with device
	for (int s = (int)instances.size(), i = 0; i < s; i++)
	{
		instances[i]->instance.traversableHandle = meshes[instances[i]->mesh]->gasHandle;
		instanceArray->HostPtr()[i] = instances[i]->instance;
	}
	instanceArray->CopyToDevice();
	// build the top-level tree
	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = (CUdeviceptr)instanceArray->DevPtr();
	buildInput.instanceArray.numInstances = (uint)instances.size();
	OptixAccelBuildOptions options = {};
	options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	options.operation = OPTIX_BUILD_OPERATION_BUILD;
	static size_t reservedTemp = 0, reservedTop = 0;
	static CoreBuffer<uchar> *temp, *topBuffer = 0;
	OptixAccelBufferSizes sizes;
	CHK_OPTIX( optixAccelComputeMemoryUsage( optixContext, &options, &buildInput, 1, &sizes ) );
	if (sizes.tempSizeInBytes > reservedTemp)
	{
		reservedTemp = sizes.tempSizeInBytes + 1024;
		delete temp;
		temp = new CoreBuffer<uchar>( reservedTemp, ON_DEVICE );
	}
	if (sizes.outputSizeInBytes > reservedTop)
	{
		reservedTop = sizes.outputSizeInBytes + 1024;
		delete topBuffer;
		topBuffer = new CoreBuffer<uchar>( reservedTop, ON_DEVICE );
	}
	CHK_OPTIX( optixAccelBuild( optixContext, 0, &options, &buildInput, 1, (CUdeviceptr)temp->DevPtr(),
		reservedTemp, (CUdeviceptr)topBuffer->DevPtr(), reservedTop, &bvhRoot, 0, 0 ) );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTextures                                                    |
//  |  Set the texture data.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTextures( const CoreTexDesc* tex, const int textures )
{
	// copy the supplied array of texture descriptors
	delete texDescs; texDescs = 0;
	textureCount = textures;
	if (textureCount == 0) return; // scene has no textures
	texDescs = new CoreTexDesc[textureCount];
	memcpy( texDescs, tex, textureCount * sizeof( CoreTexDesc ) );
	// copy texels for each type to the device
	SyncStorageType( TexelStorage::ARGB32 );
	SyncStorageType( TexelStorage::ARGB128 );
	SyncStorageType( TexelStorage::NRM32 );
	// Notes:
	// - the three types are copied from the original HostTexture pixel data (to which the
	//   descriptors point) straight to the GPU. There is no pixel storage on the host
	//   in the RenderCore.
	// - the types are copied one by one. Copying involves creating a temporary host-side
	//   buffer; doing this one by one allows us to delete host-side data for one type
	//   before allocating space for the next, thus reducing storage requirements.
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SyncStorageType                                                |
//  |  Copies texel data for one storage type (argb32, argb128 or nrm32) to the   |
//  |  device. Note that this data is obtained from the original HostTexture      |
//  |  texel arrays.                                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SyncStorageType( const TexelStorage storage )
{
	uint texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage) texelTotal += texDescs[i].pixelCount;
	texelTotal = max( 16, texelTotal ); // OptiX does not tolerate empty buffers...
	// construct the continuous arrays
	switch (storage)
	{
	case TexelStorage::ARGB32:
		delete texel32Buffer;
		texel32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE );
		SetARGB32Pixels( texel32Buffer->DevPtr() );
		coreStats.argb32TexelCount = texelTotal;
		break;
	case TexelStorage::ARGB128:
		delete texel128Buffer;
		SetARGB128Pixels( (texel128Buffer = new CoreBuffer<float4>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.argb128TexelCount = texelTotal;
		break;
	case TexelStorage::NRM32:
		delete normal32Buffer;
		SetNRM32Pixels( (normal32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.nrm32TexelCount = texelTotal;
		break;
	}
	// copy texel data to arrays
	texelTotal = 0;
	for (int i = 0; i < textureCount; i++) if (texDescs[i].storage == storage)
	{
		void* destination = 0;
		switch (storage)
		{
		case TexelStorage::ARGB32:  destination = texel32Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::ARGB128: destination = texel128Buffer->HostPtr() + texelTotal; break;
		case TexelStorage::NRM32:   destination = normal32Buffer->HostPtr() + texelTotal; break;
		}
		memcpy( destination, texDescs[i].idata, texDescs[i].pixelCount * sizeof( uint ) );
		texDescs[i].firstPixel = texelTotal;
		texelTotal += texDescs[i].pixelCount;
	}
	// move to device
	if (storage == TexelStorage::ARGB32) if (texel32Buffer) texel32Buffer->MoveToDevice();
	if (storage == TexelStorage::ARGB128) if (texel128Buffer) texel128Buffer->MoveToDevice();
	if (storage == TexelStorage::NRM32) if (normal32Buffer) normal32Buffer->MoveToDevice();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount )
{
	// Notes:
	// Call this after the textures have been set; CoreMaterials store the offset of each texture
	// in the continuous arrays; this data is valid only when textures are in sync.
	delete materialBuffer;
	delete hostMaterialBuffer;
	hostMaterialBuffer = new CoreMaterial[materialCount];
	memcpy( hostMaterialBuffer, mat, materialCount * sizeof( CoreMaterial ) );
	for (int i = 0; i < materialCount; i++)
	{
		CoreMaterial& m = hostMaterialBuffer[i];
		const CoreMaterialEx& e = matEx[i];
		if (e.texture[0] != -1) m.texaddr0 = texDescs[e.texture[0]].firstPixel;
		if (e.texture[1] != -1) m.texaddr1 = texDescs[e.texture[1]].firstPixel;
		if (e.texture[2] != -1) m.texaddr2 = texDescs[e.texture[2]].firstPixel;
		if (e.texture[3] != -1) m.nmapaddr0 = texDescs[e.texture[3]].firstPixel;
		if (e.texture[4] != -1) m.nmapaddr1 = texDescs[e.texture[4]].firstPixel;
		if (e.texture[5] != -1) m.nmapaddr2 = texDescs[e.texture[5]].firstPixel;
		if (e.texture[6] != -1) m.smapaddr = texDescs[e.texture[6]].firstPixel;
		if (e.texture[7] != -1) m.rmapaddr = texDescs[e.texture[7]].firstPixel;
		// if (e.texture[ 8] != -1) m.texaddr0 = texDescs[e.texture[ 8]].firstPixel; second roughness map is not used
		if (e.texture[9] != -1) m.cmapaddr = texDescs[e.texture[9]].firstPixel;
		if (e.texture[10] != -1) m.amapaddr = texDescs[e.texture[10]].firstPixel;
	}
	materialBuffer = new CoreBuffer<CoreMaterial>( materialCount, ON_DEVICE | ON_HOST /* on_host: for alpha mapped tris */, hostMaterialBuffer );
	SetMaterialList( materialBuffer->DevPtr() );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetLights( const CoreLightTri* areaLights, const int areaLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
	delete areaLightBuffer;
	delete pointLightBuffer;
	delete spotLightBuffer;
	delete directionalLightBuffer;
	SetAreaLights( (areaLightBuffer = new CoreBuffer<CoreLightTri>( areaLightCount, ON_DEVICE, areaLights ))->DevPtr() );
	SetPointLights( (pointLightBuffer = new CoreBuffer<CorePointLight>( pointLightCount, ON_DEVICE, pointLights ))->DevPtr() );
	SetSpotLights( (spotLightBuffer = new CoreBuffer<CoreSpotLight>( spotLightCount, ON_DEVICE, spotLights ))->DevPtr() );
	SetDirectionalLights( (directionalLightBuffer = new CoreBuffer<CoreDirectionalLight>( directionalLightCount, ON_DEVICE, directionalLights ))->DevPtr() );
	SetLightCounts( areaLightCount, pointLightCount, spotLightCount, directionalLightCount );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height )
{
	delete skyPixelBuffer;
	skyPixelBuffer = new CoreBuffer<float3>( width * height, ON_DEVICE, pixels );
	SetSkyPixels( skyPixelBuffer->DevPtr() );
	SetSkySize( width, height );
	skywidth = width;
	skyheight = height;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char* name, const float value )
{
	if (!strcmp( name, "epsilon" ))
	{
		if (vars.geometryEpsilon != value)
		{
			vars.geometryEpsilon = value;
			SetGeometryEpsilon( value );
			// context["geometryEpsilon"]->setFloat( value );
		}
	}
	else if (!strcmp( name, "clampValue" ))
	{
		if (vars.clampValue != value)
		{
			vars.clampValue = value;
			SetClampValue( value );
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge )
{
	// wait for OpenGL
	glFinish();
	Timer timer;
	// clean accumulator, if requested
	if (converge == Restart || firstConvergingFrame)
	{
		accumulator->Clear( ON_DEVICE );
		samplesTaken = 0;
		firstConvergingFrame = true; // if we switch to converging, it will be the first converging frame.
		camRNGseed = 0x12345678; // same seed means same noise.
	}
	if (converge == Converge) firstConvergingFrame = false;
	// update instance descriptor array on device
	// Note: we are not using the built-in OptiX instance system for shading. Instead,
	// we figure out which triangle we hit, and to what instance it belongs; from there,
	// we handle normal management and material acquisition in custom code.
	if (instancesDirty)
	{
		// prepare CoreInstanceDesc array. For any sane number of instances this should
		// be efficient while yielding supreme flexibility.
		vector<CoreInstanceDesc> instDescArray;
		for (auto instance : instances)
		{
			CoreInstanceDesc id;
			id.triangles = meshes[instance->mesh]->triangles->DevPtr();
			mat4 T, invT;
			if (instance->transform)
			{
				T = mat4::Identity();
				memcpy( &T, instance->transform, 12 * sizeof( float ) );
				invT = T.Inverted();
			}
			else
			{
				T = mat4::Identity();
				invT = mat4::Identity();
			}
			id.invTransform = *(float4x4*)&invT;
			instDescArray.push_back( id );
		}
		if (instDescBuffer == 0 || instDescBuffer->GetSize() < (int)instances.size())
		{
			delete instDescBuffer;
			// size of instance list changed beyond capacity.
			// Allocate a new buffer, with some slack, to prevent excessive reallocs.
			instDescBuffer = new CoreBuffer<CoreInstanceDesc>( instances.size() * 2, ON_HOST | ON_DEVICE );
			SetInstanceDescriptors( instDescBuffer->DevPtr() );
		}
		memcpy( instDescBuffer->HostPtr(), instDescArray.data(), instDescArray.size() * sizeof( CoreInstanceDesc ) );
		instDescBuffer->CopyToDevice();
		// instancesDirty = false;
	}
	// render image
	coreStats.totalExtensionRays = coreStats.totalShadowRays = 0;
	// jitter the view for TAA
	float3 right = view.p2 - view.p1, up = view.p3 - view.p1;
	// render an image using OptiX
	params.posLensSize = make_float4( view.pos.x, view.pos.y, view.pos.z, view.aperture );
	params.right = make_float3( right.x, right.y, right.z );
	params.up = make_float3( up.x, up.y, up.z );
	params.p1 = make_float3( view.p1.x, view.p1.y, view.p1.z );
	params.pass = samplesTaken;
	// loop
	params.bvhRoot = bvhRoot; // meshes[1]->gasHandle;
	Counters counters;
	coreStats.deepRayCount = 0;
	uint pathCount = scrwidth * scrheight * scrspp;
	int actualPathLength = 0;
	for (int pathLength = 1; pathLength <= MAXPATHLENGTH; pathLength++)
	{
		// generate / extend
		cudaEventRecord( traceStart[pathLength - 1] );
		if (pathLength == 1)
		{
			// spawn and extend camera rays
			params.phase = 0;
			coreStats.primaryRayCount = pathCount;
			InitCountersForExtend( pathCount );
			cudaMemcpyAsync( (void*)d_params, &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params, sizeof( Params ), &sbt, params.scrsize.x, params.scrsize.y * scrspp, 1 ) );
		}
		else
		{
			// extend bounced paths
			if (pathLength == 2) coreStats.bounce1RayCount = pathCount; else coreStats.deepRayCount += pathCount;
			params.phase = 1;
			InitCountersSubsequent();
			cudaMemcpyAsync( (void*)d_params, &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params, sizeof( Params ), &sbt, pathCount, 1, 1 ) );
		}
		cudaEventRecord( traceEnd[pathLength - 1] );
		// shade
		cudaEventRecord( shadeStart[pathLength - 1] );
		shade( pathCount, accumulator->DevPtr(), scrwidth * scrheight * scrspp,
			pathStateBuffer->DevPtr(), hitBuffer->DevPtr(), connectionBuffer->DevPtr(),
			RandomUInt( camRNGseed ) + pathLength * 91771, blueNoise->DevPtr(), samplesTaken,
			probePos.x + scrwidth * probePos.y, pathLength, scrwidth, scrheight,
			view.spreadAngle, view.p1, view.p2, view.p3, view.pos );
		cudaEventRecord( shadeEnd[pathLength - 1] );
		counterBuffer->CopyToHost();
		counters = counterBuffer->HostPtr()[0];
		pathCount = counters.extensionRays;
		actualPathLength = pathLength; // prevent timing loop iterations that we didn't execute
		if (pathCount == 0) break;
		// trace shadow rays now if the next loop iteration could overflow the buffer.
		uint maxShadowRays = connectionBuffer->GetSize() / 3;
		if ((pathCount + counters.shadowRays) >= maxShadowRays) if (counters.shadowRays > 0)
		{
			params.phase = 2;
			cudaMemcpyAsync( (void*)d_params, &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params, sizeof( Params ), &sbt, counters.shadowRays, 1, 1 ) );
			counterBuffer->HostPtr()[0].shadowRays = 0;
			counterBuffer->CopyToDevice();
			printf( "WARNING: connection buffer overflowed.\n" ); // we should not have to do this; handled here to be conservative.
		}
	}
	// connect to light sources
	cudaEventRecord( shadowStart );
	if (counters.shadowRays > 0)
	{
		params.phase = 2;
		cudaMemcpyAsync( (void*)d_params, &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
		CHK_OPTIX( optixLaunch( pipeline, 0, d_params, sizeof( Params ), &sbt, counters.shadowRays, 1, 1 ) );
	}
	cudaEventRecord( shadowEnd );
	// gather ray tracing statistics
	coreStats.totalShadowRays = counters.shadowRays;
	coreStats.totalExtensionRays = counters.totalExtensionRays;
	// present accumulator to final buffer
	renderTarget.BindSurface();
	samplesTaken += scrspp;
	finalizeRender( accumulator->DevPtr(), scrwidth, scrheight, samplesTaken );
	renderTarget.UnbindSurface();
	// finalize statistics
	cudaStreamSynchronize( 0 );
	coreStats.renderTime = timer.elapsed();
	coreStats.totalRays = coreStats.totalExtensionRays + coreStats.totalShadowRays;
	coreStats.traceTime0 = CUDATools::Elapsed( traceStart[0], traceEnd[0] );
	coreStats.traceTime1 = CUDATools::Elapsed( traceStart[1], traceEnd[1] );
	coreStats.shadowTraceTime = CUDATools::Elapsed( shadowStart, shadowEnd );
	coreStats.traceTimeX = coreStats.shadeTime = 0;
	for( int i = 2; i < actualPathLength; i++ ) coreStats.traceTimeX += CUDATools::Elapsed( traceStart[i], traceEnd[i] );
	for( int i = 0; i < actualPathLength; i++ ) coreStats.shadeTime += CUDATools::Elapsed( shadeStart[i], shadeEnd[i] );
	coreStats.probedInstid = counters.probedInstid;
	coreStats.probedTriid = counters.probedTriid;
	coreStats.probedDist = counters.probedDist;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	optixPipelineDestroy( pipeline );
	for (int i = 0; i < 5; i++) optixProgramGroupDestroy( progGroup[i] );
	optixModuleDestroy( ptxModule );
	optixDeviceContextDestroy( optixContext );
	cudaFree( (void*)sbt.raygenRecord );
	cudaFree( (void*)sbt.missRecordBase );
	cudaFree( (void*)sbt.hitgroupRecordBase );
}

// EOF