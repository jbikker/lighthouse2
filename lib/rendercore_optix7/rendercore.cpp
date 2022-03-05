/* rendercore.cpp - Copyright 2019/2021 Utrecht University

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

namespace lh2core
{

// forward declaration of cuda code
const surfaceReference* renderTargetRef();
void InitCountersForExtend( int pathCount );
void InitCountersSubsequent();
void shade( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, float4* hits, float4* connections,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle );
void finalizeRender( const float4* accumulator, const int w, const int h, const int spp );

} // namespace lh2core

using namespace lh2core;

struct SBTRecord { __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

const char* ParseOptixError( OptixResult r )
{
	switch (r)
	{
	case OPTIX_SUCCESS: return "NO ERROR";
	case OPTIX_ERROR_INVALID_VALUE: return "OPTIX_ERROR_INVALID_VALUE";
	case OPTIX_ERROR_HOST_OUT_OF_MEMORY: return "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
	case OPTIX_ERROR_INVALID_OPERATION: return "OPTIX_ERROR_INVALID_OPERATION";
	case OPTIX_ERROR_FILE_IO_ERROR: return "OPTIX_ERROR_FILE_IO_ERROR";
	case OPTIX_ERROR_INVALID_FILE_FORMAT: return "OPTIX_ERROR_INVALID_FILE_FORMAT";
	case OPTIX_ERROR_DISK_CACHE_INVALID_PATH: return "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";
	case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR: return "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";
	case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR: return "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";
	case OPTIX_ERROR_DISK_CACHE_INVALID_DATA: return "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";
	case OPTIX_ERROR_LAUNCH_FAILURE: return "OPTIX_ERROR_LAUNCH_FAILURE";
	case OPTIX_ERROR_INVALID_DEVICE_CONTEXT: return "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";
	case OPTIX_ERROR_CUDA_NOT_INITIALIZED: return "OPTIX_ERROR_CUDA_NOT_INITIALIZED";
	case OPTIX_ERROR_INVALID_PTX: return "OPTIX_ERROR_INVALID_PTX";
	case OPTIX_ERROR_INVALID_LAUNCH_PARAMETER: return "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";
	case OPTIX_ERROR_INVALID_PAYLOAD_ACCESS: return "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";
	case OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS: return "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";
	case OPTIX_ERROR_INVALID_FUNCTION_USE: return "OPTIX_ERROR_INVALID_FUNCTION_USE";
	case OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS: return "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";
	case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY: return "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";
	case OPTIX_ERROR_PIPELINE_LINK_ERROR: return "OPTIX_ERROR_PIPELINE_LINK_ERROR";
	case OPTIX_ERROR_INTERNAL_COMPILER_ERROR: return "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";
	case OPTIX_ERROR_DENOISER_MODEL_NOT_SET: return "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";
	case OPTIX_ERROR_DENOISER_NOT_INITIALIZED: return "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";
	case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE: return "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";
	case OPTIX_ERROR_NOT_SUPPORTED: return "OPTIX_ERROR_NOT_SUPPORTED";
	case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION: return "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
	case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH: return "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
	case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS: return "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
	case OPTIX_ERROR_LIBRARY_NOT_FOUND: return "OPTIX_ERROR_LIBRARY_NOT_FOUND";
	case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND: return "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
	case OPTIX_ERROR_CUDA_ERROR: return "OPTIX_ERROR_CUDA_ERROR";
	case OPTIX_ERROR_INTERNAL_ERROR: return "OPTIX_ERROR_INTERNAL_ERROR";
	case OPTIX_ERROR_UNKNOWN: return "OPTIX_ERROR_UNKNOWN";
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
	cudaMalloc( (void**)(&d_params[0]), sizeof( Params ) );
	cudaMalloc( (void**)(&d_params[1]), sizeof( Params ) );
	cudaMalloc( (void**)(&d_params[2]), sizeof( Params ) );
	// Note: we set up three sets of params, with the only difference being the 'phase' variable.
	// During wavefront path tracing this allows us to select the phase without a copyToDevice,
	// by passing the right param set for the Optix call. A bit ugly but it works.

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
		const char* file = NULL;
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
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
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
	for (int i = 0; i < 5; i++) optixSbtRecordPackHeader( progGroup[i], &rsbt[i] );
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
#ifdef _DEBUG
	printf( "Initializing Optix7 core - DEBUG build.\n" );
#else
	printf( "Initializing Optix7 core - RELEASE build.\n" );
#endif
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
	stageClampValue( 10.0f );
	// prepare counters for persistent threads
	counterBuffer = new CoreBuffer<Counters>( 1, ON_DEVICE | ON_HOST );
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
	for (int i = 0; i < MAXPATHLENGTH; i++)
	{
		cudaEventCreate( &shadeStart[i] );
		cudaEventCreate( &shadeEnd[i] );
		cudaEventCreate( &traceStart[i] );
		cudaEventCreate( &traceEnd[i] );
	}
	cudaEventCreate( &shadowStart );
	cudaEventCreate( &shadowEnd );
	// create events for worker thread communication
	startEvent = CreateEvent( NULL, false, false, NULL );
	doneEvent = CreateEvent( NULL, false, false, NULL );
	// create worker thread
	renderThread = new RenderThread();
	renderThread->Init( this );
	renderThread->start();
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
		cudaMemset( hitBuffer->DevPtr(), 255, maxPixels * scrspp * sizeof( float4 ) ); // set all hits to -1 for first frame.
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
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles )
{
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new CoreMesh pointers being pushed into the meshes vector.
	// Subsequent mesh changes will be applied to existing CoreMeshes. This is deliberately
	// minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
	if (meshIdx >= meshes.size()) meshes.push_back( new CoreMesh() );
	meshes[meshIdx]->SetGeometry( vertexData, vertexCount, triangleCount, triangles );
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
//  |  RenderCore::FinalizeInstances                                              |
//  |  Update instance descriptor array on device.                          LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::FinalizeInstances()
{
	// resize instance array if more space is needed
	if (instances.size() > (size_t)instanceArray->GetSize())
	{
		delete instanceArray;
		instanceArray = new CoreBuffer<OptixInstance>( instances.size() + 4, ON_HOST | ON_DEVICE | STAGED );
	}
	// copy instance descriptors to the array, sync with device
	for (int s = (int)instances.size(), i = 0; i < s; i++)
	{
		instances[i]->instance.traversableHandle = meshes[instances[i]->mesh]->gasHandle;
		instanceArray->HostPtr()[i] = instances[i]->instance;
	}
	instanceArray->StageCopyToDevice();
	// pass instance descriptors to the device; will be used during shading.
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
			else T = mat4::Identity(), invT = mat4::Identity();
			id.invTransform = *(float4x4*)&invT;
			instDescArray.push_back( id );
		}
		if (instDescBuffer == 0 || instDescBuffer->GetSize() < (int)instances.size())
		{
			delete instDescBuffer;
			// size of instance list changed beyond capacity.
			// Allocate a new buffer, with some slack, to prevent excessive reallocs.
			instDescBuffer = new CoreBuffer<CoreInstanceDesc>( instances.size() * 2, ON_HOST | ON_DEVICE );
			stageInstanceDescriptors( instDescBuffer->DevPtr() );
		}
		memcpy( instDescBuffer->HostPtr(), instDescArray.data(), instDescArray.size() * sizeof( CoreInstanceDesc ) );
		instDescBuffer->StageCopyToDevice();
		// instancesDirty = false; // TODO: for now we do this every frame.
	}
	// rendering is allowed from now on
	gpuHasSceneData = true;
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
		texel32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE | STAGED );
		stageARGB32Pixels( texel32Buffer->DevPtr() );
		coreStats.argb32TexelCount = texelTotal;
		break;
	case TexelStorage::ARGB128:
		delete texel128Buffer;
		stageARGB128Pixels( (texel128Buffer = new CoreBuffer<float4>( texelTotal, ON_HOST | ON_DEVICE | STAGED ))->DevPtr() );
		coreStats.argb128TexelCount = texelTotal;
		break;
	case TexelStorage::NRM32:
		delete normal32Buffer;
		stageNRM32Pixels( (normal32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE | STAGED ))->DevPtr() );
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
	if (storage == TexelStorage::ARGB32) if (texel32Buffer) texel32Buffer->StageCopyToDevice();
	if (storage == TexelStorage::ARGB128) if (texel128Buffer) texel128Buffer->StageCopyToDevice();
	if (storage == TexelStorage::NRM32) if (normal32Buffer) normal32Buffer->StageCopyToDevice();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial* mat, const int materialCount )
{
#define TOCHAR(a) ((uint)((a)*255.0f))
#define TOUINT4(a,b,c,d) (TOCHAR(a)+(TOCHAR(b)<<8)+(TOCHAR(c)<<16)+(TOCHAR(d)<<24))
	// Notes:
	// Call this after the textures have been set; CoreMaterials store the offset of each texture
	// in the continuous arrays; this data is valid only when textures are in sync.
	if (materialBuffer == 0 || materialCount > materialBuffer->GetSize())
	{
		delete hostMaterialBuffer;
		hostMaterialBuffer = new CUDAMaterial[materialCount + 512];
	}
	for (int i = 0; i < materialCount; i++)
	{
		// perform conversion to internal material format
		CoreMaterial& m = mat[i];
		CUDAMaterial& gpuMat = hostMaterialBuffer[i];
		memset( &gpuMat, 0, sizeof( CUDAMaterial ) );
		gpuMat.SetDiffuse( m.color.value );
		gpuMat.SetTransmittance( make_float3( 1 ) - m.absorption.value );
		gpuMat.parameters.x = TOUINT4( m.metallic.value, m.subsurface.value, m.specular.value, m.roughness.value );
		gpuMat.parameters.y = TOUINT4( m.specularTint.value, m.anisotropic.value, m.sheen.value, m.sheenTint.value );
		gpuMat.parameters.z = TOUINT4( m.clearcoat.value, m.clearcoatGloss.value, m.transmission.value, 0 );
		gpuMat.parameters.w = *((uint*)&m.eta);
		if (m.color.textureID != -1) gpuMat.tex0 = Map<CoreMaterial::Vec3Value>( m.color );
		if (m.detailColor.textureID != -1) gpuMat.tex1 = Map<CoreMaterial::Vec3Value>( m.detailColor );
		if (m.normals.textureID != -1) gpuMat.nmap0 = Map<CoreMaterial::Vec3Value>( m.normals );
		if (m.detailNormals.textureID != -1) gpuMat.nmap1 = Map<CoreMaterial::Vec3Value>( m.detailNormals );
		if (m.roughness.textureID != -1) gpuMat.rmap = Map<CoreMaterial::ScalarValue>( m.roughness ); /* also means metallic is mapped */
		if (m.specular.textureID != -1) gpuMat.smap = Map<CoreMaterial::ScalarValue>( m.specular );
		bool hdr = false;
		if (m.color.textureID != -1) if (texDescs[m.color.textureID].flags & 8 /* HostTexture::HDR */) hdr = true;
		gpuMat.flags =
			(m.eta.value < 1 ? ISDIELECTRIC : 0) + (hdr ? DIFFUSEMAPISHDR : 0) +
			(m.color.textureID != -1 ? HASDIFFUSEMAP : 0) +
			(m.normals.textureID != -1 ? HASNORMALMAP : 0) +
			(m.specular.textureID != -1 ? HASSPECULARITYMAP : 0) +
			(m.roughness.textureID != -1 ? HASROUGHNESSMAP : 0) +
			(m.detailNormals.textureID != -1 ? HAS2NDNORMALMAP : 0) +
			(m.detailColor.textureID != -1 ? HAS2NDDIFFUSEMAP : 0) +
			((m.flags & 1) ? HASSMOOTHNORMALS : 0) + ((m.flags & 2) ? HASALPHA : 0);
	}
	if (!materialBuffer)
	{
		materialBuffer = new CoreBuffer<CUDAMaterial>( materialCount + 512, ON_HOST | ON_DEVICE | STAGED, hostMaterialBuffer );
	}
	else if (materialCount <= materialBuffer->GetSize())
	{
		// just set the new material data
		materialBuffer->SetHostData( hostMaterialBuffer );
	}
	else if (materialCount > materialBuffer->GetSize())
	{
		// TODO: realloc, remove +512 during allocation
		delete materialBuffer;
		materialBuffer = new CoreBuffer<CUDAMaterial>( materialCount + 512, ON_HOST | ON_DEVICE | STAGED, hostMaterialBuffer );
	}
	materialBuffer->StageCopyToDevice();
	stageMaterialList( materialBuffer->DevPtr() );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateLightTree                                                |
//  |  Prepare the light BVH for stochastic lightcuts.                      LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateLightTreeNormals( const int node )
{
	LightCluster* treeData = lightTree->HostPtr();
	if (treeData[node].left > -1)
	{
		UpdateLightTreeNormals( treeData[node].left );
		UpdateLightTreeNormals( treeData[node].right );
		// check normals of children
		const float3 Nl = treeData[treeData[node].left].N;
		const float3 Nr = treeData[treeData[node].right].N;
		if (dot( Nl, Nr ) > 0.9f)
		{
			// left and right normals are similar enough to be useful
			treeData[node].N = normalize( Nl + Nr );
		}
		else
		{
			// store an impossible normal
			treeData[node].N = make_float3( 0 );
		}
		// store parent index in children
		treeData[treeData[node].left].parent = treeData[treeData[node].right].parent = node;
	}
	else
	{
		// get normal from light source
		const int lightIdx = treeData[node].light;
		if (lightIdx & ((1 << 30) + (1 << 29)))
		{
			// point or spot light
			treeData[node].N = make_float3( 0 );
		}
		else
		{
			// triangle light
			CoreLightTri& light = triLightBuffer->HostPtr()[lightIdx];
			treeData[node].N = light.N;
		}
	}
}
int RenderCore::FindBestMatch( int* todo, const int idx, const int N )
{
	float bestCost = 1e34f, bestIdx = 0;
	LightCluster* treeData = lightTree->HostPtr();
	for (int i = 0; i < N; i++) if (i != idx)
	{
		LightCluster tmp = treeData[todo[idx]];
		tmp.bounds.Grow( treeData[todo[i]].bounds );
		tmp.intensity += treeData[todo[i]].intensity;
		float cost = tmp.Cost();
		if (cost < bestCost) bestCost = cost, bestIdx = i;
	}
	return bestIdx;
}
void RenderCore::UpdateLightTree()
{
	// create an array of triLights
	delete lightTree;
	int N = triLightBuffer->GetSize(), remaining = N;
	int M = pointLightBuffer->GetSize();
	int O = spotLightBuffer->GetSize();
	lightTree = new CoreBuffer<LightCluster>( (N + M + O) * 2, ON_HOST | ON_DEVICE | STAGED );
	LightCluster* treeData = lightTree->HostPtr();
	int* todo = new int[N + M + O];
	for (int i = 0; i < N; i++)
		treeData[i + 1] = LightCluster( triLightBuffer->HostPtr()[i], i ), // leaf for light i has index i + 1
		todo[i] = i + 1;
	for (int i = 0; i < M; i++)
		treeData[i + 1 + N] = LightCluster( pointLightBuffer->HostPtr()[i], i ),
		todo[i + N] = i + 1 + N;
	for (int i = 0; i < O; i++)
		treeData[i + 1 + N + M] = LightCluster( spotLightBuffer->HostPtr()[i], i ),
		todo[i + N + M] = i + 1 + N + M;
	remaining += M + O;
	N += M + O;
	// build the BVH, agglomerative
	int A = 0;
	int B = FindBestMatch( todo, A, remaining );
	while (remaining > 1)
	{
		int C = FindBestMatch( todo, B, remaining );
		if (A == C)
		{
			// create a new cluster
			treeData[N + 1] = treeData[todo[A]];
			treeData[N + 1].bounds.Grow( treeData[todo[B]].bounds );
			treeData[N + 1].intensity += treeData[todo[B]].intensity;
			treeData[N + 1].left = todo[A];
			treeData[N + 1].right = todo[B];
			// delete A and B cluster indices from 'todo'
			for (int i = A; i < remaining - 1; i++) todo[i] = todo[i + 1]; // remove A
			if (B > A) B--;
			for (int i = B; i < remaining - 2; i++) todo[i] = todo[i + 1]; // remove B
			remaining -= 2;
			// add the new cluster index to 'todo'
			todo[remaining] = ++N;
			// prepare search for next couple
			A = remaining++;
			B = FindBestMatch( todo, A, remaining );
		}
		else A = B, B = C;
	}
	// finalize
	treeData[0] = treeData[todo[0]]; // put root in convenient place
	delete[] todo;
	UpdateLightTreeNormals( 0 );
	// copy to device
	stageLightTree( lightTree->DevPtr() );
	lightTree->StageCopyToDevice();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'20|
//  +-----------------------------------------------------------------------------+
template <class T> T* RenderCore::StagedBufferResize( CoreBuffer<T>*& lightBuffer, const int newCount, const T* sourceData )
{
	// helper function for (re)allocating light buffers with staged buffer and pointer update.
	if (lightBuffer == 0 || newCount > lightBuffer->GetSize())
	{
		delete lightBuffer;
		lightBuffer = new CoreBuffer<T>( newCount, ON_HOST | ON_DEVICE );
	}
	memcpy( lightBuffer->HostPtr(), sourceData, newCount * sizeof( T ) );
	lightBuffer->StageCopyToDevice();
	return lightBuffer->DevPtr();
}
void RenderCore::SetLights( const CoreLightTri* triLights, const int triLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
	stageTriLights( StagedBufferResize<CoreLightTri>( triLightBuffer, triLightCount, triLights ) );
	stagePointLights( StagedBufferResize<CorePointLight>( pointLightBuffer, pointLightCount, pointLights ) );
	stageSpotLights( StagedBufferResize<CoreSpotLight>( spotLightBuffer, spotLightCount, spotLights ) );
	stageDirectionalLights( StagedBufferResize<CoreDirectionalLight>( directionalLightBuffer, directionalLightCount, directionalLights ) );
	stageLightCounts( triLightCount, pointLightCount, spotLightCount, directionalLightCount );
	noDirectLightsInScene = (triLightCount + pointLightCount + spotLightCount + directionalLightCount) == 0;
	UpdateLightTree();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height, const mat4& worldToLight )
{
	delete skyPixelBuffer;
	skyPixelBuffer = new CoreBuffer<float4>( width * height + (width >> 6) * (height >> 6), ON_HOST | ON_DEVICE, 0 );
	for (uint i = 0; i < width * height; i++) skyPixelBuffer->HostPtr()[i] = make_float4( pixels[i], 0 );
	stageSkyPixels( skyPixelBuffer->DevPtr() );
	stageSkySize( width, height );
	stageWorldToSky( worldToLight );
	skywidth = width;
	skyheight = height;
	// calculate scaled-down version of the sky
	const uint w = width >> 6, h = height >> 6;
	float4* orig = skyPixelBuffer->HostPtr();
	float4* scaled = skyPixelBuffer->HostPtr() + width * height;
	for (uint y = 0; y < h; y++) for (uint x = 0; x < w; x++)
	{
		// average 64 * 64 pixels
		float4 total = make_float4( 0 );
		float4* tile = orig + x * 64 + y * 64 * width;
		for (int v = 0; v < 64; v++) for (int u = 0; u < 64; u++) total += tile[u + v * width];
		scaled[x + y * w] = total * (1.0f / (64 * 64));
	}
	// copy sky data to device
	skyPixelBuffer->CopyToDevice();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char* name, const float value )
{
	if (!strcmp( name, "epsilon" ))
	{
		if (vars.geometryEpsilon != value) stageGeometryEpsilon( vars.geometryEpsilon = value );
	}
	else if (!strcmp( name, "clampValue" ))
	{
		if (vars.clampValue != value) stageClampValue( vars.clampValue = value );
	}
	else if (!strcmp( name, "noiseShift" ))
	{
		noiseShift = fmod( value, 1.0f );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
	// build accstructs for modified meshes
	for (CoreMesh* m : meshes) if (m->accstrucNeedsUpdate) m->UpdateAccstruc();
	// build the top-level tree
	OptixBuildInput buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	buildInput.instanceArray.instances = (CUdeviceptr)instanceArray->DevPtr();
	buildInput.instanceArray.numInstances = (uint)instances.size();
	OptixAccelBuildOptions options = {};
	options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	options.operation = OPTIX_BUILD_OPERATION_BUILD;
	static size_t reservedTemp = 0, reservedTop = 0;
	static CoreBuffer<uchar>* temp, * topBuffer = 0;
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
//  |  RenderThread::run                                                          |
//  |  Main function of the render worker thread.                           LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderThread::run()
{
	while (1)
	{
		WaitForSingleObject( coreState.startEvent, INFINITE );
		// render a single frame
		coreState.RenderImpl( view );
		// we're done, go back to waiting
		SetEvent( coreState.doneEvent );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge, bool async )
{
	if (!gpuHasSceneData) return;
	// wait for OpenGL
	glFinish();
	// finalize staged writes
	pushStagedCopies();
	// handle converge restart
	if (converge == Restart || firstConvergingFrame)
	{
		samplesTaken = 0;
		firstConvergingFrame = true; // if we switch to converging, it will be the first converging frame.
		camRNGseed = 0x12345678; // same seed means same noise.
	}
	if (converge == Converge) firstConvergingFrame = false;
	// do the actual rendering
	renderTimer.reset();
	if (async)
	{
		asyncRenderInProgress = true;
		renderThread->Init( this, view );
		SetEvent( startEvent );
	}
	else
	{
		RenderImpl( view );
		FinalizeRender();
	}
}
void RenderCore::RenderImpl( const ViewPyramid& view )
{
	// update acceleration structure
	UpdateToplevel();
	// clean accumulator, if requested
	if (samplesTaken == 0) accumulator->Clear( ON_DEVICE );
	// render an image using OptiX
	RandomUInt( shiftSeed );
	coreStats.totalExtensionRays = coreStats.totalShadowRays = 0;
	float3 right = view.p2 - view.p1, up = view.p3 - view.p1;
	params.posLensSize = make_float4( view.pos.x, view.pos.y, view.pos.z, view.aperture );
	params.distortion = view.distortion;
	params.shift = shiftSeed;
	params.right = make_float3( right.x, right.y, right.z );
	params.up = make_float3( up.x, up.y, up.z );
	params.p1 = make_float3( view.p1.x, view.p1.y, view.p1.z );
	params.pass = samplesTaken;
	params.bvhRoot = bvhRoot;
	// sync params to device
	params.phase = Params::SPAWN_PRIMARY;
	cudaMemcpyAsync( (void*)d_params[0], &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
	params.phase = Params::SPAWN_SECONDARY;
	cudaMemcpyAsync( (void*)d_params[1], &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
	params.phase = Params::SPAWN_SHADOW;
	cudaMemcpyAsync( (void*)d_params[2], &params, sizeof( Params ), cudaMemcpyHostToDevice, 0 );
	// loop
	Counters counters;
	uint pathCount = scrwidth * scrheight * scrspp;
	coreStats.deepRayCount = 0;
	coreStats.primaryRayCount = pathCount;
	for (int pathLength = 1; pathLength <= MAXPATHLENGTH; pathLength++)
	{
		// generate / extend
		cudaEventRecord( traceStart[pathLength - 1] );
		if (pathLength == 1)
		{
			// spawn and extend camera rays
			InitCountersForExtend( pathCount );
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params[0], sizeof( Params ), &sbt, params.scrsize.x, params.scrsize.y * scrspp, 1 ) );
		}
		else
		{
			// extend bounced paths
			if (pathLength == 2) coreStats.bounce1RayCount = pathCount; else coreStats.deepRayCount += pathCount;
			InitCountersSubsequent();
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params[1], sizeof( Params ), &sbt, pathCount, 1, 1 ) );
		}
		cudaEventRecord( traceEnd[pathLength - 1] );
		// shade
		cudaEventRecord( shadeStart[pathLength - 1] );
		shade( pathCount, accumulator->DevPtr(), scrwidth * scrheight * scrspp,
			pathStateBuffer->DevPtr(), hitBuffer->DevPtr(), noDirectLightsInScene ? 0 : connectionBuffer->DevPtr(),
			RandomUInt( camRNGseed ) + pathLength * 91771, shiftSeed, blueNoise->DevPtr(), samplesTaken,
			probePos.x + scrwidth * probePos.y, pathLength, scrwidth, scrheight, view.spreadAngle );
		cudaEventRecord( shadeEnd[pathLength - 1] );
		counterBuffer->CopyToHost();
		counters = counterBuffer->HostPtr()[0];
		pathCount = counters.extensionRays;
		if (pathCount == 0) break;
		// trace shadow rays now if the next loop iteration could overflow the buffer.
		uint maxShadowRays = connectionBuffer->GetSize() / 3;
		if ((pathCount + counters.shadowRays) >= maxShadowRays) if (counters.shadowRays > 0)
		{
			CHK_OPTIX( optixLaunch( pipeline, 0, d_params[2], sizeof( Params ), &sbt, counters.shadowRays, 1, 1 ) );
			counterBuffer->HostPtr()[0].shadowRays = 0;
			counterBuffer->CopyToDevice();
			printf( "WARNING: connection buffer overflowed.\n" ); // we should not have to do this; handled here to be conservative.
		}
	}
	// connect to light sources
	cudaEventRecord( shadowStart );
	if (counters.shadowRays > 0)
	{
		CHK_OPTIX( optixLaunch( pipeline, 0, d_params[2], sizeof( Params ), &sbt, counters.shadowRays, 1, 1 ) );
	}
	cudaEventRecord( shadowEnd );
	// gather ray tracing statistics
	coreStats.totalShadowRays = counters.shadowRays;
	coreStats.totalExtensionRays = counters.totalExtensionRays;
	// finalize statistics
	cudaStreamSynchronize( 0 );
	coreStats.totalRays = coreStats.totalExtensionRays + coreStats.totalShadowRays;
	coreStats.traceTime0 = CUDATools::Elapsed( traceStart[0], traceEnd[0] );
	coreStats.traceTime1 = CUDATools::Elapsed( traceStart[1], traceEnd[1] );
	coreStats.shadowTraceTime = CUDATools::Elapsed( shadowStart, shadowEnd );
	// probe information
	coreStats.SetProbeInfo( counters.probedInstid, counters.probedTriid, counters.probedDist );
	const float3 P = RayTarget( probePos.x, probePos.y, 0.5f, 0.5f, make_int2( scrwidth, scrheight ), view.distortion, view.p1, right, up );
	const float3 D = normalize( P - view.pos );
	coreStats.probedWorldPos = view.pos + counters.probedDist * D;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::WaitForRender                                                  |
//  |  Wait for the render thread to finish.                                      |
//  |  Note: will deadlock if we didn't actually start a render.            LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::WaitForRender()
{
	// wait for the renderthread to complete
	if (!asyncRenderInProgress) return;
	WaitForSingleObject( doneEvent, INFINITE );
	asyncRenderInProgress = false;
	// get back the RenderCore state data changed by the thread
	coreStats = renderThread->coreState.coreStats;
	camRNGseed = renderThread->coreState.camRNGseed;
	shiftSeed = renderThread->coreState.shiftSeed;
	// copy the accumulator to the OpenGL texture
	FinalizeRender();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::FinalizeRender                                                 |
//  |  Fill the OpenGL rendertarget texture.                                LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::FinalizeRender()
{
	// present accumulator to final buffer
	renderTarget.BindSurface();
	samplesTaken += scrspp;
	finalizeRender( accumulator->DevPtr(), scrwidth, scrheight, samplesTaken );
	renderTarget.UnbindSurface();
	// timing statistics
	coreStats.renderTime = renderTimer.elapsed();
	coreStats.frameOverhead = max( 0.0f, frameTimer.elapsed() - coreStats.renderTime );
	frameTimer.reset();
	coreStats.traceTimeX = coreStats.shadeTime = 0;
	for (int i = 2; i < MAXPATHLENGTH; i++)
		coreStats.traceTimeX += CUDATools::Elapsed( renderThread->coreState.traceStart[i], renderThread->coreState.traceEnd[i] );
	for (int i = 0; i < MAXPATHLENGTH; i++)
		coreStats.shadeTime += CUDATools::Elapsed( renderThread->coreState.shadeStart[i], renderThread->coreState.shadeEnd[i] );
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

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::GetCoreStats                                                   |
//  |  Get a copy of the counters.                                          LH2'19|
//  +-----------------------------------------------------------------------------+
CoreStats RenderCore::GetCoreStats() const
{
	return coreStats;
}

// EOF