/* core_settings.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are core-specific:
   - avilable in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

// core-specific settings
#define CLAMPFIREFLIES		// suppress fireflies by clamping
#define MAXPATHLENGTH		3
// #define USE_LAMBERT_BSDF	// override default microfacet model
// #define USE_MULTISCATTER_BSDF // override default microfacet model
// #define GGXCONDUCTOR // alternative is the diffuse ggx brdf
#define SINGLEBOUNCE		// perform only a single diffuse bounce
#define CONSISTENTNORMALS	// consistent normal interpolation; don't use with filtering?

// low-level settings
#define USE_OPTIX_PERSISTENT_THREADS
#define SCATTERSTEPS 1		// max bounces in microfacet evaluation (multiscatter bsdf)
#define BLUENOISE			// use blue noise instead of uniform random numbers
#define TAA					// really basic temporal antialiasing
#define BILINEAR			// enable bilinear interpolation
// #define NOTEXTURES		// all texture reads will be white
#define COMBINEDSHADING		// direct and indirect are stored together for faster access

#define APPLYSAFENORMALS	if (dot( N, wi ) <= 0) pdf = 0;
#define NOHIT				-1

#ifndef __CUDACC__

#ifdef _DEBUG
#pragma comment(lib, "../platform/lib/debug/platform.lib" )
#else
#pragma comment(lib, "../platform/lib/release/platform.lib" )
#endif
#pragma comment(lib, "../OptiX/lib64/optix.6.0.0.lib" )

#define CUDABUILD			// signal system.h to include full CUDA headers
#include "helper_math.h"	// for vector types
#include "platform.h"
#include "system.h"			// for vector types
#undef APIENTRY				// get rid of an anoying warning
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "nvrtc.h"
#include "FreeImage.h"		// for loading blue noise
#include "shared_host_code/cudatools.h"
#include "shared_host_code/interoptexture.h"

#include <optixu/optixpp_namespace.h>
using namespace optix;
using namespace lighthouse2;

#include "core_mesh.h"

using namespace lh2core;

#endif

// for a full path state, we need a Ray, an Intersection, and
// the data specified in PathState.
struct PathState
{
	float3 O;				// normal at last path vertex
	uint data;				// holds pixel index << 8, path length in bits 0..3, flags in bits 4..7
	float3 D; int N;		// ray direction of the current path segment, encoded normal in w
	float3 throughput;		// path transport
	float bsdfPdf;			// postponed pdf
};
struct PathState4 { float4 O4, D4, T4; };

// PotentialContribution: besides a shadow ray, a connection needs
// a pixel index and the energy that will be depositied to that pixel
// if there is no occlusion. 
struct PotentialContribution
{
	float3 O;
	uint dummy1;
	float3 D;
	uint dummy2;
	float3 E;
	uint pixelIdx;
};
struct PotentialContribution4 { float4 O4, D4, E4; };

// counters and other global data, in device memory
struct Counters
{
	uint activePaths;
	uint shaded;
	uint generated;
	uint connected;
	uint extended;
	uint extensionRays;
	uint shadowRays;
	uint totalExtensionRays;
	uint totalShadowRays;
	int probedInstid;
	int probedTriid;
	float probedDist;
};

// ------------------------------------------------------------------------------
// Below this line: derived, low-level and internal.

// clamping
#ifdef CLAMPFIREFLIES
#define CLAMPINTENSITY		const float v=max(contribution.x,max(contribution.y,contribution.z)); \
							if(v>clampValue){const float m=clampValue/v;contribution.x*=m; \
							contribution.y*=m;contribution.z*=m; /* don't touch w */ }
#else
#define CLAMPINTENSITY
#endif

#ifndef __CUDACC__
#define OPTIXU_MATH_DEFINE_IN_NAMESPACE
#define _USE_MATH_DEFINES
#include "core_api_base.h"
#include "core_api.h"

namespace lh2core {

template <class T> class InteropBuffer
{
	// Note: the OptiX API uses a 'Buffer' object to synchronize data between host and device.
	// Theoretically, this facilitates setups more complex than ours, e.g. when using multiple
	// GPUs, in which case a single buffer should reside on multiple devices, which is then
	// transparently handled by OptiX (or at least, supposedly so). However, this comes at the
	// expense of transparency: when what data is copied to which device is not entirely clear,
	// and the 'dirty' system is something we would rather manage ourselves.
	// Since we use a blend of OptiX code and pure CUDA code (for wavefront PT / filtering),
	// and because the behavior of raw CoreBuffers feels more familiar, intuitive and low-level,
	// we construct OptiX buffer objects often using an existing device pointer obtained from
	// a CoreBuffer.
	// The InteropBuffer facilitates this method. An InteropBuffer is constructed much like a 
	// CoreBuffer, but it can now also be conveniently used to feed an rtBuffer object in OptiX
	// code.
	// Additionally, we can now create rtBuffers, and access the related device pointer for use
	// in CUDA. This behavior is assumed whenever bufferType is not RT_BUFFER_INPUT.
public:
	InteropBuffer( __int64 elementCount, __int64 loc, uint bufferType, RTformat rtFormat, const char* name, void* source = 0 )
	{
		// TODO: when rtFormat == RT_FORMAT_USER, the component size is deduced from T.
		// If this is as efficient as the built-in types, we can use RT_FORMAT_USER for
		// all buffers, which reduces the argument count of the constructor by 1.
		// Let's check impact on performance in OptiX 5.2 before doing this.
		if (bufferType == RT_BUFFER_INPUT)
		{
			// this buffer is for OptiX to read from; CUDA creates it
			// Note: RT_BUFFER_COPY_ON_DIRTY supposedly limits buffer syncs to those occasions where we explicitly 
			// marked the buffer as dirty. The idea is that this never happens, and the flag is intended to prevent 
			// smart behavior from OptiX. These assumptions are not carefully verified though.
			cudaBuffer = new CoreBuffer<T>( max( 1, elementCount /* OptiX buffers may not be nullptrs */ ), loc, source );
			optixBuffer = RenderCore::context->createBufferForCUDA( bufferType | RT_BUFFER_COPY_ON_DIRTY, rtFormat );
			if (rtFormat == RT_FORMAT_USER) optixBuffer->setElementSize( sizeof( T ) );
			optixBuffer->setDevicePointer( 0, cudaBuffer->DevPtr() );
			RenderCore::context[name]->setBuffer( optixBuffer );
			cudaOwned = true;
		}
		else // RT_BUFFER_INPUT or RT_BUFFER_INPUT_OUTPUT
		{
			// this buffer is for OptiX to write to; OptiX creates it
			assert( source == 0 /* no host-side init data for OptiX buffers */ );
			optixBuffer = RenderCore::context->createBuffer( bufferType | RT_BUFFER_COPY_ON_DIRTY, rtFormat, elementCount );
			if (rtFormat == RT_FORMAT_USER) optixBuffer->setElementSize( sizeof( T ) );
			RenderCore::context[name]->setBuffer( optixBuffer );
			cudaOwned = false;
		}
	}
	InteropBuffer( __int64 width, __int64 height, __int64 loc, uint bufferType, RTformat rtFormat, const char* name, void* source = 0 )
	{
		// A 2D buffer must be an OptiX buffer.
		assert( bufferType != RT_BUFFER_INPUT );
		assert( source == 0 /* no host-side init data for OptiX buffers */ );
		optixBuffer = RenderCore::context->createBuffer( bufferType | RT_BUFFER_COPY_ON_DIRTY, rtFormat, width, height );
		if (rtFormat == RT_FORMAT_USER) optixBuffer->setElementSize( sizeof( T ) );
		RenderCore::context[name]->setBuffer( optixBuffer );
		cudaOwned = false;
	}
	~InteropBuffer()
	{
		if (optixBuffer) optixBuffer->destroy();
		delete cudaBuffer;
		optixBuffer = 0;
		cudaBuffer = 0;
	}
	T* DevPtr()
	{
		if (cudaOwned) return cudaBuffer->DevPtr();
		T* payload;
		optixBuffer->getDevicePointer( 0 /* not considering multi-GPU */, (void**)&payload );
		return payload;
	}
	T* HostPtr()
	{
		assert( cudaOwned );
		return cudaBuffer->HostPtr();
	}
	T* DebugHostPtr()
	{
		if (cudaOwned)
		{
			cudaBuffer->CopyToHost();
			return cudaBuffer->HostPtr();
		}
		return (T*)optixBuffer->map(); // really just for debugging; we should unmap to continue safely.
	}
	void Clear( __int64 loc )
	{
		assert( loc == ON_DEVICE ); // for consistency with CoreBuffer
		if (cudaOwned) cudaBuffer->Clear( ON_DEVICE ); else
		{
			T* payload;
			optixBuffer->getDevicePointer( 0 /* not considering multi-GPU */, (void**)&payload );
			cudaMemset( payload, 0, GetSize() * sizeof( T ) );
		}
	}
	// TODO: buffers owned by OptiX can't be moved to/from host (although we can 'map' them);
	// for now we limit these functions to cuda-owned buffers, by means of an assert.
	void* MoveToDevice() { assert( cudaOwned ); return cudaBuffer->MoveToDevice(); }
	void* CopyToDevice() { assert( cudaOwned ); return cudaBuffer->CopyToDevice(); }
	void* CopyFromDevice() { assert( cudaOwned ); return cudaBuffer->CopyToHost(); }
	void* CopyToHost() { assert( cudaOwned ); return cudaBuffer->CopyToHost(); }
	long long GetSize()
	{
		if (cudaOwned) return cudaBuffer->GetSize(); else
		{
			RTsize size;
			optixBuffer->getSize( size );
			return size;
		}
	}
private:
	CoreBuffer<T>* cudaBuffer = 0;		// the device data, encapsulated as CoreBuffer
	optix::Buffer optixBuffer = 0;		// the device data, encapsualted as OptixBuffer
	bool cudaOwned;						// false if OptiX created the buffer
};

} // namespace lh2core

#include "rendercore.h"

#endif

// EOF