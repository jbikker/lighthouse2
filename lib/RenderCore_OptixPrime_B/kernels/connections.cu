/* connections.cu - Copyright 2019 Utrecht University

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

#include "noerrors.h"

//  +-----------------------------------------------------------------------------+
//  |  finalizeConnectionKernel                                                   |
//  |  Finalize a connection.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
__device__ void finalizeConnectionKernel( int jobIndex, float4* accumulator, uint* hitBuffer, float4* contributions )
{
	const float3 E = make_float3( contributions[jobIndex] );
	const uint pixelIdx = __float_as_uint( contributions[jobIndex].w );
	const uint occluded = hitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));
	if (!occluded) accumulator[pixelIdx] += make_float4( E, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeConnectionsPersistent                                              |
//  |  Persistent kernel for finalizing connections.                        LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  void __launch_bounds__( 256 /* max block size */, 1 /* min blocks per sm */ )
finalizeConnectionsPersistent( int connections, float4* accumulator, uint* hitBuffer, float4* contributions )
{
	__shared__ volatile int baseIdx[32];
	int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	__syncthreads();
	while (1)
	{
		if (lane == 0) baseIdx[warp] = atomicAdd( &counters->connected, 32 );
		int jobIndex = baseIdx[warp] + lane;
		if (__all_sync( THREADMASK, jobIndex >= connections )) break; // need to do the path with all threads in the warp active
		const uint set = WangHash( jobIndex ) & (LDSETS - 1);
		if (jobIndex < connections) finalizeConnectionKernel( jobIndex, accumulator, hitBuffer, contributions );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeConnections                                                        |
//  |  Entry point for the persistent finalizeConnections kernel.           LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void finalizeConnections( int smcount, int connections, float4* accumulator,
	uint* hitBuffer, float4* contributions )
{
	finalizeConnectionsPersistent << < smcount, 256 >> > (connections, accumulator, hitBuffer, contributions);
}

// EOF