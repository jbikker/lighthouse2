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

   Implementation of the Optix Prime rendercore. This is a wavefront
   / streaming path tracer: CUDA code in camera.cu is used to
   generate a primary ray buffer, which is then traced by Optix. The
   resulting hitpoints are procressed using another CUDA kernel (in
   pathtracer.cu), which in turn generates extension rays and shadow
   rays. Path contributions are accumulated in an accumulator and
   finalized using code in finalize.cu.
*/

#include "core_settings.h"

namespace lh2core
{

// forward declaration of cuda code
const surfaceReference* renderTargetRef();
void generateEyeRays( Ray4* rayBuffer, float4* pathStateData, const uint R0, const uint* blueNoise, const float2* camSamples,
	const int pass, const ViewPyramid& view, const int4 screenParams );
void InitCountersForExtend( int pathCount );
void InitCountersSubsequent();
void shade( const int pathCount, float4* accumulator, const Ray4* extensionRays, const float4* extensionData,
	const Intersection* hits, Ray4* extensionRaysOut, float4* extensionDataOut, Ray4* shadowRays,
	float4* connectionT4, const uint R0, const uint* blueNoise, const int pass, const int2 probePos,
	const int pathLength, const int w, const int h, const ViewPyramid& view );
void finalizeConnections( int rayCount, float4* accumulator, uint* hitBuffer, float4* contributions );
void finalizeRender( const float4* accumulator, const int w, const int h, const int spp );

// rendertime getters/setters
void SetCounters( Counters* p );

} // namespace lh2core

using namespace lh2core;

RTPcontext RenderCore::context = 0;

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::GetScreenParams                                                |
//  |  Helper function - fills an int4 with values related to screen size.  LH2'19|
//  +-----------------------------------------------------------------------------+
int4 RenderCore::GetScreenParams()
{
	float e = 0.0001f; // RenderSettings::geoEpsilon;
	return make_int4( scrwidth + (scrheight << 16),					// .x : SCRHSIZE, SCRVSIZE
		scrspp + (1 /* RenderSettings::pathDepth */ << 8),			// .y : SPP, MAXDEPTH
		scrwidth * scrheight * scrspp,								// .z : PIXELCOUNT
		*((int*)&e) );												// .w : RenderSettings::geoEpsilon
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
//  |  RenderCore::Init                                                           |
//  |  CUDA / Optix / RenderCore initialization.                            LH2'19|
//  +-----------------------------------------------------------------------------+
static float2 RandomPointInTriangle( const float2& A, const float2& B, const float2& C, uint& seed )
{
	// float3 bary = RandomBarycentrics( r );
	return (A + B + C) / 3; // bary.x * A + bary.y * B + bary.z * C;
}
void RenderCore::Init()
{
#ifdef _DEBUG
	printf( "Initializing OptixPrime core - DEBUG build.\n" );
#else
	printf( "Initializing OptixPrime core - RELEASE build.\n" );
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
	// setup OptiX Prime
	CHK_PRIME( rtpContextCreate( RTP_CONTEXT_TYPE_CUDA, &context ) );
	const char* versionString;
	CHK_PRIME( rtpGetVersionString( &versionString ) );
	printf( "%s\n", versionString );
	CHK_PRIME( rtpContextSetCudaDeviceNumbers( context, 1, &device ) );
	// prepare the top-level 'model' node; instances will be added to this.
	topLevel = new RTPmodel();
	CHK_PRIME( rtpModelCreate( context, topLevel ) );
	CHK_PRIME( rtpQueryCreate( *topLevel, RTP_QUERY_TYPE_ANY, &shadowQuery ) );
	CHK_PRIME( rtpQueryCreate( *topLevel, RTP_QUERY_TYPE_CLOSEST, &extendQuery ) );
	// prepare counters for persistent threads
	counterBuffer = new CoreBuffer<Counters>( 16, ON_DEVICE | ON_HOST );
	SetCounters( counterBuffer->DevPtr() );
	// render settings
	stageClampValue( 10.0f );
	// prepare the bluenoise data
	const uchar* data8 = (const uchar*)sob256_64; // tables are 8 bit per entry
	uint* data32 = new uint[65536 * 5]; // we want a full uint per entry
	for (int i = 0; i < 65536; i++) data32[i] = data8[i]; // convert
	data8 = (uchar*)scr256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 65536] = data8[i];
	data8 = (uchar*)rnk256_64;
	for (int i = 0; i < (128 * 128 * 8); i++) data32[i + 3 * 65536] = data8[i];
	blueNoise = new CoreBuffer<uint>( 65536 * 5, ON_DEVICE, data32 );
	delete data32;
	// prepare data for aperture sampling
	camSamples = new CoreBuffer<float2>( 256, ON_HOST );
	float golden_angle = PI * (3 - sqrtf( 5 ));
	for (int i = 0; i < 256; i++)
	{
		float theta = i * golden_angle;
		float r = sqrtf( (float)i ) / sqrtf( 256.0f );
		camSamples->HostPtr()[i] = make_float2( r * cosf( theta ), r * sinf( theta ) );
	}
	for( int i = 0; i < 10240; i++ )
	{
		const uint idx0 = RandomUInt() & 255, idx1 = RandomUInt() & 255;
		swap( camSamples->HostPtr()[idx0], camSamples->HostPtr()[idx1] );
	}
	camSamples->CopyToDevice();
#if 0
	FILE* f = fopen( "points.dat", "wb" );
	fwrite( camSamples->HostPtr(), 8, 256, f );
	fclose( f );
#endif
	// allow CoreMeshes to access the core
	CoreMesh::renderCore = this;
	// timing events
	for (int i = 0; i < MAXPATHLENGTH; i++)
	{
		cudaEventCreate( &shadeStart[i] );
		cudaEventCreate( &shadeEnd[i] );
	}
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
		maxPixels = scrwidth * scrheight + maxPixels / 16; // reserve a bit extra to prevent frequent reallocs
		currentSPP = spp;
		reallocate = true;
	}
	if (reallocate)
	{
		// destroy previously created OptiX buffers
		if (!firstFrame)
		{
			rtpBufferDescDestroy( extensionRaysDesc[0] );
			rtpBufferDescDestroy( extensionRaysDesc[1] );
			rtpBufferDescDestroy( extensionHitsDesc );
			rtpBufferDescDestroy( shadowRaysDesc );
			rtpBufferDescDestroy( shadowHitsDesc );
		}
		// delete CoreBuffers
		delete extensionRayBuffer[0];
		delete extensionRayBuffer[1];
		delete extensionRayExBuffer[0];
		delete extensionRayExBuffer[1];
		delete extensionHitBuffer;
		delete shadowRayBuffer;
		delete shadowRayPotential;
		delete shadowHitBuffer;
		delete accumulator;
		const uint maxShadowRays = maxPixels * spp * 2;
		extensionHitBuffer = new CoreBuffer<Intersection>( maxPixels * spp, ON_DEVICE );
		shadowRayBuffer = new CoreBuffer<Ray4>( maxShadowRays, ON_DEVICE );
		shadowRayPotential = new CoreBuffer<float4>( maxShadowRays, ON_DEVICE ); // .w holds pixel index
		shadowHitBuffer = new CoreBuffer<uint>( (maxShadowRays + 31) >> 5 /* one bit per ray */, ON_DEVICE );
		accumulator = new CoreBuffer<float4>( maxPixels, ON_DEVICE );
		for (int i = 0; i < 2; i++)
		{
			extensionRayBuffer[i] = new CoreBuffer<Ray4>( maxPixels * spp, ON_DEVICE );
			extensionRayExBuffer[i] = new CoreBuffer<float4>( maxPixels * 2 * spp, ON_DEVICE );
			CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, extensionRayBuffer[i]->DevPtr(), &extensionRaysDesc[i] ) );
		}
		CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_HIT_T_TRIID_INSTID_U_V, RTP_BUFFER_TYPE_CUDA_LINEAR, extensionHitBuffer->DevPtr(), &extensionHitsDesc ) );
		CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX, RTP_BUFFER_TYPE_CUDA_LINEAR, shadowRayBuffer->DevPtr(), &shadowRaysDesc ) );
		CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_HIT_BITMASK, RTP_BUFFER_TYPE_CUDA_LINEAR, shadowHitBuffer->DevPtr(), &shadowHitsDesc ) );
		printf( "buffers resized for %i pixels @ %i samples.\n", maxPixels, spp );
	}
	// clear the accumulator
	accumulator->Clear( ON_DEVICE );
	samplesTaken = 0;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::TraceShadowRays                                                |
//  |  RenderCore::TraceExtensionRays                                             |
//  |  Trace rays using Optix Prime.                                        LH2'20|
//  +-----------------------------------------------------------------------------+
float RenderCore::TraceShadowRays( const int rayCount )
{
	Timer t;
	CHK_PRIME( rtpBufferDescSetRange( shadowRaysDesc, 0, rayCount ) );
	CHK_PRIME( rtpBufferDescSetRange( shadowHitsDesc, 0, rayCount ) );
	CHK_PRIME( rtpQuerySetRays( shadowQuery, shadowRaysDesc ) );
	CHK_PRIME( rtpQuerySetHits( shadowQuery, shadowHitsDesc ) );
	CHK_PRIME( rtpQueryExecute( shadowQuery, RTP_QUERY_HINT_NONE ) );
	return t.elapsed();
}
float RenderCore::TraceExtensionRays( const int rayCount )
{
	Timer t;
	CHK_PRIME( rtpBufferDescSetRange( extensionRaysDesc[inBuffer], 0, rayCount ) );
	CHK_PRIME( rtpBufferDescSetRange( extensionHitsDesc, 0, rayCount ) );
	CHK_PRIME( rtpQuerySetRays( extendQuery, extensionRaysDesc[inBuffer] ) );
	CHK_PRIME( rtpQuerySetHits( extendQuery, extensionHitsDesc ) );
	CHK_PRIME( rtpQueryExecute( extendQuery, RTP_QUERY_HINT_NONE ) );
	return t.elapsed();
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
	if (instanceIdx >= instances.size()) instances.push_back( new CoreInstance() );
	instances[instanceIdx]->mesh = meshIdx;
	instances[instanceIdx]->transform = matrix;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::FinalizeInstances                                              |
//  |  Update instance descriptor array on device.                          LH2'20|
//  +-----------------------------------------------------------------------------+
void RenderCore::FinalizeInstances()
{
	if (!instancesDirty) return;
	// prepare CoreInstanceDesc array. For any sane number of instances this should
	// be efficient while yielding supreme flexibility.
	vector<CoreInstanceDesc> instDescArray;
	for (auto instance : instances)
	{
		CoreInstanceDesc id;
		id.triangles = meshes[instance->mesh]->triangles->DevPtr();
		mat4 T = instance->transform.Inverted();
		id.invTransform = *(float4x4*)&T;
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
	// instancesDirty = false;
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
	// Notes:
	// - the three types are copied from the original HostTexture pixel data (to which the
	//   descriptors point) straight to the GPU. There is no pixel storage on the host
	//   in this RenderCore.
	// - the types are copied one by one. Copying involves creating a temporary host-side
	//   buffer; doing this one by one allows us to delete host-side data for one type
	//   before allocating space for the next, thus reducing runtime storage.
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
		stageARGB32Pixels( texel32Buffer->DevPtr() );
		coreStats.argb32TexelCount = texelTotal;
		break;
	case TexelStorage::ARGB128:
		delete texel128Buffer;
		stageARGB128Pixels( (texel128Buffer = new CoreBuffer<float4>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
		coreStats.argb128TexelCount = texelTotal;
		break;
	case TexelStorage::NRM32:
		delete normal32Buffer;
		stageNRM32Pixels( (normal32Buffer = new CoreBuffer<uint>( texelTotal, ON_HOST | ON_DEVICE ))->DevPtr() );
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
	delete materialBuffer;
	delete hostMaterialBuffer;
	hostMaterialBuffer = new CUDAMaterial[materialCount];
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
	materialBuffer = new CoreBuffer<CUDAMaterial>( materialCount, ON_HOST | ON_DEVICE | STAGED, hostMaterialBuffer );
	materialBuffer->StageCopyToDevice();
	stageMaterialList( materialBuffer->DevPtr() );
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
void RenderCore::SetLights( const CoreLightTri* areaLights, const int areaLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
	stageAreaLights( StagedBufferResize<CoreLightTri>( areaLightBuffer, areaLightCount, areaLights ) );
	stagePointLights( StagedBufferResize<CorePointLight>( pointLightBuffer, pointLightCount, pointLights ) );
	stageSpotLights( StagedBufferResize<CoreSpotLight>( spotLightBuffer, spotLightCount, spotLights ) );
	stageDirectionalLights( StagedBufferResize<CoreDirectionalLight>( directionalLightBuffer, directionalLightCount, directionalLights ) );
	stageLightCounts( areaLightCount, pointLightCount, spotLightCount, directionalLightCount );
	noDirectLightsInScene = (areaLightCount + pointLightCount + spotLightCount + directionalLightCount) == 0;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height, const mat4& worldToLight )
{
	delete skyPixelBuffer;
	skyPixelBuffer = new CoreBuffer<float3>( width * height, ON_DEVICE, pixels );
	stageSkyPixels( skyPixelBuffer->DevPtr() );
	stageSkySize( width, height );
	stageWorldToSky( worldToLight );
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
		if (vars.geometryEpsilon != value) stageGeometryEpsilon( vars.geometryEpsilon = value );
	}
	else if (!strcmp( name, "clampValue" ))
	{
		if (vars.clampValue != value) stageClampValue( vars.clampValue = value );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::UpdateToplevel                                                 |
//  |  After changing meshes, instances or instance transforms, we need to        |
//  |  rebuild the top-level structure.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::UpdateToplevel()
{
	// creates the top-level BVH over the supplied models.
	RTPbufferdesc instancesBuffer, transformBuffer;
	vector<RTPmodel> modelList;
	vector<mat4> transformList;
	for (auto instance : instances)
	{
		modelList.push_back( meshes[instance->mesh]->model );
		transformList.push_back( instance->transform );
	}
	CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_INSTANCE_MODEL, RTP_BUFFER_TYPE_HOST, modelList.data(), &instancesBuffer ) );
	CHK_PRIME( rtpBufferDescCreate( context, RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4, RTP_BUFFER_TYPE_HOST, transformList.data(), &transformBuffer ) );
	CHK_PRIME( rtpBufferDescSetRange( instancesBuffer, 0, instances.size() ) );
	CHK_PRIME( rtpBufferDescSetRange( transformBuffer, 0, instances.size() ) );
	CHK_PRIME( rtpModelSetInstances( *topLevel, instancesBuffer, transformBuffer ) );
	CHK_PRIME( rtpModelUpdate( *topLevel, RTP_MODEL_HINT_ASYNC /* blocking; try RTP_MODEL_HINT_ASYNC + rtpModelFinish for async version. */ ) );
	CHK_PRIME( rtpBufferDescDestroy( instancesBuffer ) /* no idea if this does anything relevant */ );
	CHK_PRIME( rtpBufferDescDestroy( transformBuffer ) /* no idea if this does anything relevant */ );
	instancesDirty = true; // sync instance list to device prior to next ray query
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
	// render image
	coreStats.totalExtensionRays = 0;
	InitCountersForExtend( scrwidth * scrheight * scrspp );
	generateEyeRays( extensionRayBuffer[inBuffer]->DevPtr(), extensionRayExBuffer[inBuffer]->DevPtr(),
		RandomUInt( camRNGseed ), blueNoise->DevPtr(), camSamples->DevPtr(), samplesTaken, view, GetScreenParams() );
	// start wavefront loop
	uint pathCount = scrwidth * scrheight * scrspp, pathLength = 1;
	Counters& counters = counterBuffer->HostPtr()[0];
	for (; pathLength <= MAXPATHLENGTH; pathLength++)
	{
		// extend
		float traceTime = TraceExtensionRays( pathCount );
		if (pathLength == 1) coreStats.traceTime0 = traceTime, coreStats.primaryRayCount = pathCount;
		else if (pathLength == 2)  coreStats.traceTime1 = traceTime, coreStats.bounce1RayCount = pathCount;
		else coreStats.traceTimeX = traceTime, coreStats.deepRayCount = pathCount;
		// shade
		cudaEventRecord( shadeStart[pathLength - 1] );
		shade( pathCount, accumulator->DevPtr(), extensionRayBuffer[inBuffer]->DevPtr(), extensionRayExBuffer[inBuffer]->DevPtr(),
			extensionHitBuffer->DevPtr(), extensionRayBuffer[outBuffer]->DevPtr(), extensionRayExBuffer[outBuffer]->DevPtr(),
			noDirectLightsInScene ? 0 : shadowRayBuffer->DevPtr(), shadowRayPotential->DevPtr(), samplesTaken * 7907 + pathLength * 91771, blueNoise->DevPtr(),
			samplesTaken, probePos, pathLength, scrwidth, scrheight, view );
		counterBuffer->CopyToHost(); // sadly this is needed; Optix Prime doesn't expose persistent threads
		cudaEventRecord( shadeEnd[pathLength - 1] );
		if (pathLength < MAXPATHLENGTH) // code below is not needed for the last path segment
		{
			pathCount = counters.extensionRays;
			swap( inBuffer, outBuffer );
			// handle overflowing shadow ray buffer
			uint maxShadowRays = shadowRayBuffer->GetSize();
			if ((counters.shadowRays + pathCount) >= maxShadowRays)
			{
				// trace the shadow rays using OptiX Prime
				TraceShadowRays( counters.shadowRays );
				finalizeConnections( counters.shadowRays, accumulator->DevPtr(), shadowHitBuffer->DevPtr(), shadowRayPotential->DevPtr() );
				// reset shadow ray counter
				counterBuffer->HostPtr()[0].shadowRays = 0;
				counterBuffer->CopyToDevice();
			}
			// prepare next path segment
			InitCountersSubsequent();
		}
	}
	// loop completed; handle gathered shadow rays
	if (counters.shadowRays > 0)
	{
		// trace the shadow rays using OptiX Prime
		coreStats.shadowTraceTime = TraceShadowRays( counters.shadowRays );
		finalizeConnections( counters.shadowRays, accumulator->DevPtr(), shadowHitBuffer->DevPtr(), shadowRayPotential->DevPtr() );
	}
	// finalize statistics
	coreStats.totalShadowRays = counters.shadowRays;
	coreStats.totalExtensionRays = counters.totalExtensionRays;
	coreStats.totalRays = coreStats.totalExtensionRays + coreStats.totalShadowRays;
	coreStats.SetProbeInfo( counters.probedInstid, counters.probedTriid, counters.probedDist );
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
	coreStats.shadeTime = 0;
	for (uint i = 0; i < MAXPATHLENGTH; i++)
		coreStats.shadeTime += CUDATools::Elapsed( renderThread->coreState.shadeStart[i], renderThread->coreState.shadeEnd[i] );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	// delete ray buffers
	delete extensionRayBuffer[0];
	delete extensionRayBuffer[1];
	delete extensionRayExBuffer[0];
	delete extensionRayExBuffer[1];
	delete extensionHitBuffer;
	delete shadowRayBuffer;
	delete shadowRayPotential;
	delete shadowHitBuffer;
	// delete internal data
	delete accumulator;
	delete counterBuffer;
	delete texDescs;
	delete texel32Buffer;
	delete texel128Buffer;
	delete normal32Buffer;
	delete materialBuffer;
	delete hostMaterialBuffer;
	delete skyPixelBuffer;
	delete instDescBuffer;
	// delete light data
	delete areaLightBuffer;
	delete pointLightBuffer;
	delete spotLightBuffer;
	delete directionalLightBuffer;
	// delete core scene representation
	for (auto mesh : meshes) delete mesh;
	for (auto instance : instances) delete instance;
	rtpQueryDestroy( shadowQuery );
	rtpQueryDestroy( extendQuery );
	delete topLevel;
	rtpBufferDescDestroy( extensionRaysDesc[0] );
	rtpBufferDescDestroy( extensionRaysDesc[1] );
	rtpBufferDescDestroy( extensionHitsDesc );
	rtpContextDestroy( context );
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