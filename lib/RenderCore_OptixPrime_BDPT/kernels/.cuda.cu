/* .cuda.cu - Copyright 2019 Utrecht University

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

#include ".cuda.h"

namespace lh2core
{

// path tracing buffers and global variables
__constant__ CoreInstanceDesc* instanceDescriptors;
__constant__ CUDAMaterial* materials;
__constant__ CoreLightTri* areaLights;
__constant__ CorePointLight* pointLights;
__constant__ CoreSpotLight* spotLights;
__constant__ CoreDirectionalLight* directionalLights;
__constant__ int4 lightCounts; // area, point, spot, directional
__constant__ uchar4* argb32;
__constant__ float4* argb128;
__constant__ uchar4* nrm32;
__constant__ float3* skyPixels;
__constant__ int skywidth;
__constant__ int skyheight;
__constant__ float4* debugData;

// path tracer settings
__constant__ __device__ float geometryEpsilon;
__constant__ __device__ float clampValue;

// access
__host__ void SetInstanceDescriptors( CoreInstanceDesc* p ) { cudaMemcpyToSymbol( instanceDescriptors, &p, sizeof( void* ) ); }
__host__ void SetMaterialList( CUDAMaterial* p ) { cudaMemcpyToSymbol( materials, &p, sizeof( void* ) ); }
__host__ void SetAreaLights( CoreLightTri* p ) { cudaMemcpyToSymbol( areaLights, &p, sizeof( void* ) ); }
__host__ void SetPointLights( CorePointLight* p ) { cudaMemcpyToSymbol( pointLights, &p, sizeof( void* ) ); }
__host__ void SetSpotLights( CoreSpotLight* p ) { cudaMemcpyToSymbol( spotLights, &p, sizeof( void* ) ); }
__host__ void SetDirectionalLights( CoreDirectionalLight* p ) { cudaMemcpyToSymbol( directionalLights, &p, sizeof( void* ) ); }
__host__ void SetLightCounts( int area, int point, int spot, int directional )
{
	const int4 counts = make_int4( area, point, spot, directional );
	cudaMemcpyToSymbol( lightCounts, &counts, sizeof( int4 ) );
}
__host__ void SetARGB32Pixels( uint* p ) { cudaMemcpyToSymbol( argb32, &p, sizeof( void* ) ); }
__host__ void SetARGB128Pixels( float4* p ) { cudaMemcpyToSymbol( argb128, &p, sizeof( void* ) ); }
__host__ void SetNRM32Pixels( uint* p ) { cudaMemcpyToSymbol( nrm32, &p, sizeof( void* ) ); }
__host__ void SetSkyPixels( float3* p ) { cudaMemcpyToSymbol( skyPixels, &p, sizeof( void* ) ); }
__host__ void SetSkySize( int w, int h ) { cudaMemcpyToSymbol( skywidth, &w, sizeof( int ) ); cudaMemcpyToSymbol( skyheight, &h, sizeof( int ) ); }
__host__ void SetDebugData( float4* p ) { cudaMemcpyToSymbol( debugData, &p, sizeof( void* ) ); }

// access
__host__ void SetGeometryEpsilon( float e ) { cudaMemcpyToSymbol( geometryEpsilon, &e, sizeof( float ) ); }
__host__ void SetClampValue( float c ) { cudaMemcpyToSymbol( clampValue, &c, sizeof( float ) ); }

// BDPT
/////////////////////////////////////////////////
/* LH2_DEVFUNC void copyPathState(const BiPathState orgin, BiPathState& target)
{
	memcpy(&target, &orgin, sizeof(BiPathState));
} */

__global__ void InitIndexForConstructionLight_Kernel( int pathCount, uint* construcLightBuffer )
{
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	construcLightBuffer[jobIndex] = jobIndex;
}
__host__ void InitIndexForConstructionLight( int pathCount, uint* construcLightBuffer )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	InitIndexForConstructionLight_Kernel << <gridDim.x, 256 >> > (pathCount, construcLightBuffer);
}
///////////////////////////////////////////////////
// counters for persistent threads
static __device__ Counters* counters;
__global__ void InitCountersForExtend_Kernel( int pathCount )
{
	if (threadIdx.x != 0) return;

	counters->constructionLightPos = pathCount;	// remaining active paths
	counters->constructionEyePos = 0;

	counters->extendEyePath = 0;
	counters->extendLightPath = 0;

	counters->randomWalkRays = 0;
	counters->visibilityRays = 0;
}
__host__ void InitCountersForExtend( int pathCount ) { InitCountersForExtend_Kernel << <1, 32 >> > (pathCount); }

__global__ void InitCountersForPixels_Kernel()
{
	if (threadIdx.x != 0) return;

	counters->contribution_count = 0;
}
__host__ void InitCountersForPixels() { InitCountersForPixels_Kernel << <1, 32 >> > (); }

__host__ void SetCounters( Counters* p ) { cudaMemcpyToSymbol( counters, &p, sizeof( void* ) ); }

// functional blocks
#include "tools_shared.h"
#include "sampling_shared.h"
#include "material_shared.h"
#include "lights_shared.h"
#include "finalize_shared.h"
#include "bsdf.h"

#include "constructionLightPos.h"
#include "constructionEyePos.h"

#include "extendEyePath.h"
#include "extendLightPath.h"

#include "connectionPath.h"
#include "finalizeContribution.h"
} // namespace lh2core

// EOF
