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

__constant__ mat4 worldToSky;

// path tracer settings
__constant__ __device__ float geometryEpsilon;
__constant__ __device__ float clampValue;

#if 1

// alternative staging method: copies will be batched and carried out after rendering
// completes, so allow the CPU to update the scene concurrently with GPU rendering.

enum { INSTS = 0, MATS, ALGHTS, PLGHTS, SLGHTS, DLGHTS, LCNTS, RGB32, RGBH, NRMLS, SKYPIX, SKYW, SKYH, SMAT, DBGDAT, GEPS, CLMPV };

// device pointers are not real pointers for nvcc, so we need a bit of a hack.

struct StagedPtr { void* p; int id; };
struct StagedInt { int v; int id; };
struct StagedInt4 { int4 v; int id; };
struct StagedMat { mat4 v; int id; };
struct StagedF32 { float v; int id; };
struct StagedCpy { void* d; void* s; int n; };
static std::vector<StagedPtr> stagedPtr;
static std::vector<StagedInt> stagedInt;
static std::vector<StagedInt4> stagedInt4;
static std::vector<StagedMat> stagedMat;
static std::vector<StagedF32> stagedF32;
static std::vector<StagedCpy> stagedCpy;

__host__ static void pushPtrCpy( int id, void* p )
{
	if (id == INSTS) cudaMemcpyToSymbol( instanceDescriptors, &p, sizeof( void* ) );
	if (id == MATS) cudaMemcpyToSymbol( materials, &p, sizeof( void* ) );
	if (id == ALGHTS) cudaMemcpyToSymbol( areaLights, &p, sizeof( void* ) );
	if (id == PLGHTS) cudaMemcpyToSymbol( pointLights, &p, sizeof( void* ) );
	if (id == SLGHTS) cudaMemcpyToSymbol( spotLights, &p, sizeof( void* ) );
	if (id == DLGHTS) cudaMemcpyToSymbol( directionalLights, &p, sizeof( void* ) );
	if (id == RGB32) cudaMemcpyToSymbol( argb32, &p, sizeof( void* ) );
	if (id == RGBH) cudaMemcpyToSymbol( argb128, &p, sizeof( void* ) );
	if (id == NRMLS) cudaMemcpyToSymbol( nrm32, &p, sizeof( void* ) );
	if (id == SKYPIX) cudaMemcpyToSymbol( skyPixels, &p, sizeof( void* ) );
	if (id == DBGDAT) cudaMemcpyToSymbol( debugData, &p, sizeof( void* ) );
}
__host__ static void pushIntCpy( int id, const int v )
{
	if (id == SKYW) cudaMemcpyToSymbol( skywidth, &v, sizeof( int ) );
	if (id == SKYH) cudaMemcpyToSymbol( skyheight, &v, sizeof( int ) );
}
__host__ static void pushF32Cpy( int id, const float v )
{
	if (id == GEPS) cudaMemcpyToSymbol( geometryEpsilon, &v, sizeof( float ) );
	if (id == CLMPV) cudaMemcpyToSymbol( clampValue, &v, sizeof( int ) );
}
__host__ static void pushMatCpy( int id, const mat4& m )
{
	if (id == SMAT) cudaMemcpyToSymbol( worldToSky, &m, sizeof( mat4 ) );
}
__host__ static void pushInt4Cpy( int id, const int4& v )
{
	if (id == LCNTS) cudaMemcpyToSymbol( lightCounts, &v, sizeof( int4 ) );
}

__host__ static void stagePtrCpy( int id, void* p ) { StagedPtr n = { p, id }; stagedPtr.push_back( n ); }
__host__ static void stageIntCpy( int id, const int v ) { StagedInt n = { v, id }; stagedInt.push_back( n ); }
__host__ static void stageF32Cpy( int id, const float v ) { StagedF32 n = { v, id }; stagedF32.push_back( n ); }
__host__ static void stageMatCpy( int id, const mat4& m ) { StagedMat n = { m, id }; stagedMat.push_back( n ); }
__host__ static void stageInt4Cpy( int id, const int4& v ) { StagedInt4 n = { v, id }; stagedInt4.push_back( n ); }

__host__ void stageMemcpy( void* d, void* s, int n ) { StagedCpy c = { d, s, n }; stagedCpy.push_back( c ); }

__host__ void stageInstanceDescriptors( CoreInstanceDesc* p ) { stagePtrCpy( INSTS /* instanceDescriptors */, p ); }
__host__ void stageMaterialList( CUDAMaterial* p ) { stagePtrCpy( MATS /* materials */, p ); }
__host__ void stageAreaLights( CoreLightTri* p ) { stagePtrCpy( ALGHTS /* areaLights */, p ); }
__host__ void stagePointLights( CorePointLight* p ) { stagePtrCpy( PLGHTS /* pointLights */, p ); }
__host__ void stageSpotLights( CoreSpotLight* p ) { stagePtrCpy( SLGHTS /* spotLights */, p ); }
__host__ void stageDirectionalLights( CoreDirectionalLight* p ) { stagePtrCpy( DLGHTS /* directionalLights */, p ); }
__host__ void stageARGB32Pixels( uint* p ) { stagePtrCpy( RGB32 /* argb32 */, p ); }
__host__ void stageARGB128Pixels( float4* p ) { stagePtrCpy( RGBH /* argb128 */, p ); }
__host__ void stageNRM32Pixels( uint* p ) { stagePtrCpy( NRMLS /* nrm32 */, p ); }
__host__ void stageSkyPixels( float3* p ) { stagePtrCpy( SKYPIX /* skyPixels */, p ); }
__host__ void stageSkySize( int w, int h ) { stageIntCpy( SKYW /* skywidth */, w ); stageIntCpy( SKYH /* skyheight */, h ); }
__host__ void stageWorldToSky( const mat4& worldToLight ) { stageMatCpy( SMAT /* worldToSky */, worldToLight ); }
__host__ void stageDebugData( float4* p ) { stagePtrCpy( DBGDAT /* debugData */, p ); }
__host__ void stageGeometryEpsilon( float e ) { stageF32Cpy( GEPS /* geometryEpsilon */, e ); }
__host__ void stageClampValue( float c ) { stageF32Cpy( CLMPV /* clampValue */, c ); }
__host__ void stageLightCounts( int area, int point, int spot, int directional )
{
	const int4 counts = make_int4( area, point, spot, directional );
	stageInt4Cpy( LCNTS /* lightCounts */, counts );
}

__host__ void pushStagedCopies()
{
	for (auto c : stagedCpy) cudaMemcpy( c.d, c.s, c.n, cudaMemcpyHostToDevice ); stagedCpy.clear();
	for (auto n : stagedPtr) pushPtrCpy( n.id, n.p ); stagedPtr.clear();
	for (auto n : stagedInt) pushIntCpy( n.id, n.v ); stagedInt.clear();
	for (auto n : stagedInt4) pushInt4Cpy( n.id, n.v ); stagedInt4.clear();
	for (auto n : stagedF32) pushF32Cpy( n.id, n.v ); stagedF32.clear();
	for (auto n : stagedMat) pushMatCpy( n.id, n.v ); stagedMat.clear();
}

#else

#if 0

// staged copies using streams - cannot work with OptixPrime, because BVH updates in
// prime are not guaranteed to be tied to a stream.

// streams
cudaStream_t stageStream, renderStream;

// staging:
// In order to allow for concurrent GPU-side rendering and CPU-side scene updates,
// all changes made by CPU-side code to the GPU-side data are batched in the stageStream
// and saved up until the renderStream completes.
#define stagedcpy( d, a ) cudaMemcpyToSymbolAsync( (d), &a, sizeof( a ), stageStream )

#else

// when using Optix Prime we will just run the 'staged' copies right away.
#define stagedcpy( d, a ) cudaMemcpyToSymbol( (d), &a, sizeof( a ) )

#endif

// render state access - staged
__host__ void stageInstanceDescriptors( CoreInstanceDesc* p ) { stagedcpy( instanceDescriptors, p ); }
__host__ void stageMaterialList( CUDAMaterial* p ) { stagedcpy( materials, p ); }
__host__ void stageAreaLights( CoreLightTri* p ) { stagedcpy( areaLights, p ); }
__host__ void stagePointLights( CorePointLight* p ) { stagedcpy( pointLights, p ); }
__host__ void stageSpotLights( CoreSpotLight* p ) { stagedcpy( spotLights, p ); }
__host__ void stageDirectionalLights( CoreDirectionalLight* p ) { stagedcpy( directionalLights, p ); }
__host__ void stageLightCounts( int area, int point, int spot, int directional )
{
	const int4 counts = make_int4( area, point, spot, directional );
	stagedcpy( lightCounts, counts );
}
__host__ void stageARGB32Pixels( uint* p ) { stagedcpy( argb32, p ); }
__host__ void stageARGB128Pixels( float4* p ) { stagedcpy( argb128, p ); }
__host__ void stageNRM32Pixels( uint* p ) { stagedcpy( nrm32, p ); }
__host__ void stageSkyPixels( float3* p ) { stagedcpy( skyPixels, p ); }
__host__ void stageSkySize( int w, int h ) { stagedcpy( skywidth, w ); stagedcpy( skyheight, h ); }
__host__ void stageWorldToSky( const mat4& worldToLight ) { stagedcpy( worldToSky, worldToLight ); }
__host__ void stageDebugData( float4* p ) { stagedcpy( debugData, p ); }
__host__ void stageGeometryEpsilon( float e ) { stagedcpy( geometryEpsilon, e ); }
__host__ void stageClampValue( float c ) { stagedcpy( clampValue, c ); }
__host__ void stageMemcpy( void* d, void* s, int n ) { cudaMemcpy( d, s, n, cudaMemcpyHostToDevice ); }

#endif

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
#include "sampling_shared.h"
#include "material_shared.h"
#include "lights_shared.h"
#include "bsdf.h"
#include "pathtracer.h"
#include "finalize_shared.h"
#include "camera.h"
#include "connections.h"

} // namespace lh2core

// EOF