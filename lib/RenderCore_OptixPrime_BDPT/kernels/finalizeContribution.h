/* camera.cu - Copyright 2019 Utrecht University

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
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void finalizeContributionKernel( int smcount,
	uint* visibilityHitBuffer, float4* accumulatorOnePass,
	float4* contribution_buffer )
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= counters->contribution_count) return;


	const uint occluded = visibilityHitBuffer[gid >> 5] & (1 << (gid & 31));

	if (!occluded)
	{
		float4 color = contribution_buffer[gid];
		uint jobIndex = __float_as_uint( color.w );


		color.w = 0.0f;

		// accumulatorOnePass[jobIndex] += color;

		atomicAdd( &(accumulatorOnePass[jobIndex].x), color.x );
		atomicAdd( &(accumulatorOnePass[jobIndex].y), color.y );
		atomicAdd( &(accumulatorOnePass[jobIndex].z), color.z );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void finalizeContribution( int smcount,
	uint* visibilityHitBuffer, float4* accumulatorOnePass,
	float4* contribution_buffer )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	finalizeContributionKernel << < gridDim.x, 256 >> > (smcount,
		visibilityHitBuffer, accumulatorOnePass, contribution_buffer);
}

// EOF