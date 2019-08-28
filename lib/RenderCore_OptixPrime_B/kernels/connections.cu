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
__global__  __launch_bounds__( 256 /* max block size */, 1 /* min blocks per sm */ )
void finalizeConnectionKernel( float4* accumulator, uint* hitBuffer, float4* contributions, const int pathCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// process shadow ray query result
	const float3 E = make_float3( contributions[jobIndex] );
	const uint pixelIdx = __float_as_uint( contributions[jobIndex].w );
	const uint occluded = hitBuffer[jobIndex >> 5] & (1 << (jobIndex & 31));
	if (!occluded) accumulator[pixelIdx] += make_float4( E, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeConnections                                                        |
//  |  Entry point for the persistent finalizeConnections kernel.           LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void finalizeConnections( int rayCount, float4* accumulator, uint* hitBuffer, float4* contributions )
{
	const dim3 gridDim( NEXTMULTIPLEOF( rayCount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	finalizeConnectionKernel << < gridDim.x, 256 >> > (accumulator, hitBuffer, contributions, rayCount);
}

// EOF