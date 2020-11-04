/* .cuda.cu - Copyright 2019/2020 Utrecht University

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
__constant__ CoreMaterialDesc* materialDescriptors;
// For simplicity PBRT materials receive the full-fat CoreMaterial instead of a slimmed-down CUDAMaterial
__constant__ CoreMaterial* pbrtMaterials;
__constant__ CoreLightTri* triLights;
__constant__ CorePointLight* pointLights;
__constant__ CoreSpotLight* spotLights;
__constant__ CoreDirectionalLight* directionalLights;
__constant__ int4 lightCounts; // area, point, spot, directional
__constant__ uchar4* argb32;
__constant__ float4* argb128;
__constant__ uchar4* nrm32;
__constant__ float4* skyPixels;
__constant__ int skywidth;
__constant__ int skyheight;
__constant__ float4* debugData;
__constant__ LightCluster* lightTree;

__constant__ mat4 worldToSky;

// path tracer settings
__constant__ __device__ float geometryEpsilon;
__constant__ __device__ float clampValue;

// faking staged copies for now
#define stagedcpy( d, a ) cudaMemcpyToSymbol( (d), &a, sizeof( a ) )

// staged render state access
__host__ void stageInstanceDescriptors( CoreInstanceDesc* p ) { stagedcpy( instanceDescriptors, p ); }
__host__ void stageMaterialList( CUDAMaterial* p ) { stagedcpy( materials, p ); }
__host__ void stageMaterialDescList( CoreMaterialDesc* p ) { stagedcpy( materialDescriptors, p ); }
__host__ void stagePbrtMaterialList( CoreMaterial* p ) { stagedcpy( pbrtMaterials, p ); }
__host__ void stageTriLights( CoreLightTri* p ) { stagedcpy( triLights, p ); }
__host__ void stagePointLights( CorePointLight* p ) { stagedcpy( pointLights, p ); }
__host__ void stageSpotLights( CoreSpotLight* p ) { stagedcpy( spotLights, p ); }
__host__ void stageDirectionalLights( CoreDirectionalLight* p ) { stagedcpy( directionalLights, p ); }
__host__ void stageLightTree( LightCluster* t ) { stagedcpy( lightTree, t ); }
__host__ void stageLightCounts( int tri, int point, int spot, int directional )
{
	const int4 counts = make_int4( tri, point, spot, directional );
	stagedcpy( lightCounts, counts );
}
__host__ void stageARGB32Pixels( uint* p ) { stagedcpy( argb32, p ); }
__host__ void stageARGB128Pixels( float4* p ) { stagedcpy( argb128, p ); }
__host__ void stageNRM32Pixels( uint* p ) { stagedcpy( nrm32, p ); }
__host__ void stageSkyPixels( float4* p ) { stagedcpy( skyPixels, p ); }
__host__ void stageSkySize( int w, int h ) { stagedcpy( skywidth, w ); stagedcpy( skyheight, h ); }
__host__ void stageWorldToSky( const mat4& worldToLight ) { stagedcpy( worldToSky, worldToLight ); }
__host__ void stageDebugData( float4* p ) { stagedcpy( debugData, p ); }
__host__ void stageGeometryEpsilon( float e ) { stagedcpy( geometryEpsilon, e ); }
__host__ void stageClampValue( float c ) { stagedcpy( clampValue, c ); }
__host__ void stageMemcpy( void* d, void* s, int n ) { cudaMemcpy( d, s, n, cudaMemcpyHostToDevice ); }

// counters for persistent threads
static __device__ Counters* counters;
__global__ void InitCountersForExtend_Kernel( int pathCount )
{
	if (threadIdx.x != 0) return;
	counters->activePaths = pathCount;	// remaining active paths
	counters->extensionRays = 0;		// compaction counter for extension rays
	counters->shadowRays = 0;			// compaction counter for connections
	counters->totalExtensionRays = pathCount;
	counters->totalShadowRays = 0;
}
__host__ void InitCountersForExtend( int pathCount ) { InitCountersForExtend_Kernel << <1, 32 >> > (pathCount); }
__global__ void InitCountersSubsequent_Kernel()
{
	if (threadIdx.x != 0) return;
	counters->totalExtensionRays += counters->extensionRays;
	counters->activePaths = counters->extensionRays;	// remaining active paths
	counters->extensionRays = 0;		// compaction counter for extension rays
}
__host__ void InitCountersSubsequent() { InitCountersSubsequent_Kernel << <1, 32 >> > (); }
__host__ void SetCounters( Counters* p ) { cudaMemcpyToSymbol( counters, &p, sizeof( void* ) ); }

// functional blocks
#include "tools_shared.h"
#include "tools.h"
#include "sampling_shared.h"
#include "material_shared.h"
#include "lights_shared.h"
#include "bsdf.h"
#include "material_intf.h"
#include "pathtracer.h"
#include "finalize_shared.h"
#include "camera.h"
#include "connections.h"

} // namespace lh2core

// EOF
