/* finalize.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE:
   used in RenderCore_OptixPrime and RenderCore_OptixRTX.
*/

#include "noerrors.h"

//  +-----------------------------------------------------------------------------+
//  |  finalizeRenderKernel                                                       |
//  |  Presenting the accumulator; including brightness, contrast and gamma       |
//  |  correction.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void finalizeRenderKernel( const float4* accumulator, const int scrwidth, const int scrheight, const float pixelValueScale, const float brightness, const float contrastFactor )
{
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	// plot scaled pixel, gamma corrected
	float4 value = accumulator[x + y * scrwidth] * pixelValueScale;
	float r = sqrtf( max( 0.0f, (value.x - 0.5f) * contrastFactor + 0.5f + brightness ) );
	float g = sqrtf( max( 0.0f, (value.y - 0.5f) * contrastFactor + 0.5f + brightness ) );
	float b = sqrtf( max( 0.0f, (value.z - 0.5f) * contrastFactor + 0.5f + brightness ) );
	surf2Dwrite<float4>( make_float4( r, g, b, value.w ), renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeRender( const float4* accumulator, const int w, const int h, const int spp, const float brightness, const float contrast )
{
	const float pixelValueScale = 1.0f / (float)spp;
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	const float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	finalizeRenderKernel << < gridDim, blockDim >> > (accumulator, w, h, pixelValueScale, brightness, contrastFactor);
}

//  +-----------------------------------------------------------------------------+
//  |  prepareFilterKernel                                                        |
//  |  Split albedo and illumination for filtering. And:                          |
//  |  1st and 2nd moment of the luminance, for variance estimation.              |
//  |  Combined because the moments require the luminance, which is already       |
//  |  available during the split process. Saves array I/O.                 LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float WorldDistance( const int2& pixel, const float4& currentWorldPos, const float4* prevWorldPos, const int w, const int h )
{
	float4 p = ReadWorldPos( prevWorldPos, pixel, w, h );
	if ((__float_as_uint( p.w ) & 3) != 0) return 1e21f; // we are always searching for specular pixels
	return sqrLength( make_float3( currentWorldPos - p ) );
}
LH2_DEVFUNC int RefineHistoryPos( int2& currentScreenPos, const float4& localPos, float& bestDist, int step, const float4* prevWorldPos, const int w, const int h )
{
	int bestTap = 0;
	const int2 pixelPos0 = make_int2( currentScreenPos.x - step, currentScreenPos.y );
	const int2 pixelPos1 = make_int2( currentScreenPos.x + step, currentScreenPos.y );
	const int2 pixelPos2 = make_int2( currentScreenPos.x, currentScreenPos.y - step );
	const int2 pixelPos3 = make_int2( currentScreenPos.x, currentScreenPos.y + step );
	float d;
	d = WorldDistance( pixelPos0, localPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos0, bestTap = 1;
	d = WorldDistance( pixelPos1, localPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos1, bestTap = 2;
	d = WorldDistance( pixelPos2, localPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos2, bestTap = 3;
	d = WorldDistance( pixelPos3, localPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos3, bestTap = 4;
	return bestTap;
}
LH2_DEVFUNC void FinalRefine( int2& currentScreenPos, const float4& currentWorldPos, float& bestDist, const float4* prevWorldPos, const int w, const int h )
{
	const int2 pixelPos0 = make_int2( currentScreenPos.x - 1, currentScreenPos.y );
	const int2 pixelPos1 = make_int2( currentScreenPos.x + 1, currentScreenPos.y );
	const int2 pixelPos2 = make_int2( currentScreenPos.x, currentScreenPos.y - 1 );
	const int2 pixelPos3 = make_int2( currentScreenPos.x, currentScreenPos.y + 1 );
	const int2 pixelPos4 = make_int2( currentScreenPos.x - 1, currentScreenPos.y - 1 );
	const int2 pixelPos5 = make_int2( currentScreenPos.x + 1, currentScreenPos.y - 1 );
	const int2 pixelPos6 = make_int2( currentScreenPos.x - 1, currentScreenPos.y + 1 );
	const int2 pixelPos7 = make_int2( currentScreenPos.x + 1, currentScreenPos.y + 1 );
	float d;
	d = WorldDistance( pixelPos0, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos0;
	d = WorldDistance( pixelPos1, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos1;
	d = WorldDistance( pixelPos2, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos2;
	d = WorldDistance( pixelPos3, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos3;
	d = WorldDistance( pixelPos4, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos4;
	d = WorldDistance( pixelPos5, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos5;
	d = WorldDistance( pixelPos6, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos6;
	d = WorldDistance( pixelPos7, currentWorldPos, prevWorldPos, w, h ); if (d < bestDist) bestDist = d, currentScreenPos = pixelPos7;
}
__global__ void prepareFilterKernel( const float4* accumulator, uint4* features, const float4* worldPos, const float4* prevWorldPos,
	float4* shading, float2* motion, float4* moments, float4* prevMoments, const float4* deltaDepth,
	const float4 prevPos, const float4 prevE, const float4 prevRight, const float4 prevUp, const float j0, const float j1, const float prevj0, const float prevj1,
	const int scrwidth, const int scrheight, const float pixelValueScale, const float directClamp, const float indirectClamp, const int flags )
{
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	const int pixelIdx = x + y * scrwidth;
	// split direct and indirect light from albedo and clamp
	const float3 direct = make_float3( accumulator[pixelIdx] ) * pixelValueScale;
	const float3 albedo = RGB32toHDRmin1( features[pixelIdx].x );
	const float3 indirect = make_float3( accumulator[pixelIdx + scrwidth * scrheight] ) * pixelValueScale;
	const float3 reciAlbedo = make_float3( 1.0f / albedo.x, 1.0f / albedo.y, 1.0f / albedo.z );
	const float3 directLight = min3( direct * reciAlbedo, directClamp );
	const float3 indirectLight = min3( indirect * reciAlbedo, indirectClamp );
	shading[pixelIdx] = CombineToFloat4( directLight, indirectLight );
	// calculate location in screen space of the current pixel in the previous frame
	const float4 localPos = worldPos[pixelIdx];
	const float4 D = make_float4( normalize( make_float3( localPos - prevPos ) ), 0 );
	const float4 S = make_float4( make_float3( prevPos + D * (prevPos.w / dot( prevE, D )) ), 0 );
	float2 prevPixelPos = make_float2( dot( S, prevRight ) - prevRight.w - j0, dot( S, prevUp ) - prevUp.w - j1 );
	float lumDirect = Luminance( directLight ), lumDirect2 = lumDirect * lumDirect;
	float lumIndirect = Luminance( indirectLight ), lumIndirect2 = lumIndirect * lumIndirect;
	if ((__float_as_uint( localPos.w ) & 3) == 0)
	{
		// zero motion vectors for stationary camera, hack as proposed by Victor
		if (flags & 1) prevPixelPos = make_float2( x, y ); else
		{
			// for speculars, determine motion vector using diamond search
		#if 1
			int2 bestPos = make_int2( x, y );
			const float4 prevWorldPosA = ReadWorldPos( prevWorldPos, make_int2( x, y ), scrwidth, scrheight );	// current position
			float bestDist = sqrLength( make_float3( localPos - prevWorldPosA ) );
		#else
			int2 bestPos = make_int2( prevPixelPos.x, prevPixelPos.y );
			const float4 prevWorldPosA = ReadWorldPos( prevWorldPos, bestPos, scrwidth, scrheight );			// reprojected position
			const float4 prevWorldPosB = ReadWorldPos( prevWorldPos, make_int2( x, y ), scrwidth, scrheight );	// current position
			float bestDist = sqrLength( make_float3( localPos - prevWorldPosA ) );
			const float altDist = sqrLength( make_float3( localPos - prevWorldPosB ) );
			if ((altDist < bestDist || ((__float_as_uint( prevWorldPosA.w ) & 3) != 0) && (__float_as_uint( prevWorldPosB.w ) & 3) == 0))
			{
				bestPos = make_int2( x, y );
				bestDist = altDist;
			}
		#endif
			int iter = 0, stepSize = (1 + 2 + 3 + 4), stepDelta = 4;
			while (1)
			{
				const int tap = RefineHistoryPos( bestPos, localPos, bestDist, stepSize, prevWorldPos, scrwidth, scrheight );
				if (tap == 0) { stepSize -= stepDelta--; if (stepSize == 0) break; }
				if (++iter == 20) break;
			}
			// FinalRefine( bestPos, localPos, bestDist, prevWorldPos, scrwidth, scrheight );
			// sub-pixel refine
			const int2 p0 = make_int2( bestPos.x - 1, bestPos.y - 1 ); const float d0 = WorldDistance( p0, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p1 = make_int2( bestPos.x, bestPos.y - 1 ); const float d1 = WorldDistance( p1, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p2 = make_int2( bestPos.x + 1, bestPos.y - 1 ); const float d2 = WorldDistance( p2, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p3 = make_int2( bestPos.x + 1, bestPos.y ); const float d3 = WorldDistance( p3, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p4 = make_int2( bestPos.x + 1, bestPos.y + 1 ); const float d4 = WorldDistance( p4, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p5 = make_int2( bestPos.x, bestPos.y + 1 ); const float d5 = WorldDistance( p5, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p6 = make_int2( bestPos.x - 1, bestPos.y + 1 ); const float d6 = WorldDistance( p6, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p7 = make_int2( bestPos.x - 1, bestPos.y ); const float d7 = WorldDistance( p7, localPos, prevWorldPos, scrwidth, scrheight );
			const int2 p8 = make_int2( bestPos.x, bestPos.y ); const float d8 = bestDist;
			const float v0 = d0 > 1e20f ? 0 : 1, v1 = d1 > 1e20f ? 0 : 1, v2 = d2 > 1e20f ? 0 : 1;
			const float v3 = d3 > 1e20f ? 0 : 1, v4 = d4 > 1e20f ? 0 : 1, v5 = d5 > 1e20f ? 0 : 1;
			const float v6 = d6 > 1e20f ? 0 : 1, v7 = d7 > 1e20f ? 0 : 1;
			int bestQuad = 0;
			{
				const float score0 = (d0 * v0 + d1 * v1 + d7 * v7 + d8) / (v0 + v1 + v7 + 1);    //  0 1 2
				const float score1 = (d1 * v1 + d2 * v2 + d3 * v3 + d8) / (v1 + v2 + v3 + 1);    //  7 8 3
				const float score2 = (d3 * v3 + d5 * v5 + d4 * v4 + d8) / (v3 + v5 + v4 + 1);    //  6 5 4
				const float score3 = (d7 * v7 + d6 * v6 + d5 * v5 + d8) / (v7 + v6 + v5 + 1);
				if (score1 < score0) bestQuad = 1;
				if (score2 < score0 && score2 < score1) bestQuad = 2;
				if (score3 < score0 && score3 < score1 && score3 < score2) bestQuad = 3;
			}
			float2 subPixelPos;
			float4 weight;
			if (bestQuad == 0)
				weight = make_float4( v0 * d1 * d7 * d8, v1 * d0 * d7 * d8, v7 * d0 * d1 * d8, d0 * d1 * d7 ),
				subPixelPos = weight.x * make_float2( p0 ) + weight.y * make_float2( p1 ) + weight.z * make_float2( p7 ) + weight.w * make_float2( p8 );
			else if (bestQuad == 1)
				weight = make_float4( v1 * d2 * d8 * d3, v2 * d1 * d8 * d3, v3 * d1 * d2 * d8, d1 * d2 * d3 ),
				subPixelPos = weight.x * make_float2( p1 ) + weight.y * make_float2( p2 ) + weight.z * make_float2( p3 ) + weight.w * make_float2( p8 );
			else if (bestQuad == 2)
				weight = make_float4( v3 * d8 * d5 * d4, d3 * d5 * d4, v5 * d8 * d3 * d4, v4 * d8 * d3 * d5 ),
				subPixelPos = weight.x * make_float2( p3 ) + weight.y * make_float2( p8 ) + weight.z * make_float2( p5 ) + weight.w * make_float2( p4 );
			else // if (bestQuad == 3)
				weight = make_float4( v7 * d8 * d6 * d5, d7 * d6 * d5, v6 * d7 * d8 * d5, v5 * d7 * d8 * d6 ),
				subPixelPos = weight.x * make_float2( p7 ) + weight.y * make_float2( p8 ) + weight.z * make_float2( p6 ) + weight.w * make_float2( p5 );
			// overwrite motion vector
			prevPixelPos = subPixelPos * (1.0f / (weight.x + weight.y + weight.z + weight.w)) + make_float2( prevj0 - j0, prevj1 - j1 );
		}
	}
	// get history luminance moments
	prevPixelPos += make_float2( 0.5f, 0.5f );
	const float px = prevPixelPos.x;
	const float py = prevPixelPos.y;
	const uint4 localFeature = features[pixelIdx];
	if (px >= 0 && px < scrwidth && py >= 0 && py < scrheight)
	{
		const float localDdx = deltaDepth[pixelIdx].z;
		const float localDdy = deltaDepth[pixelIdx].w;
		const float allowedDist = max( 0.05f, fabs( localDdx ) + fabs( localDdy ) );
		const float3 localNormal = UnpackNormal2( localFeature.y );
		const float4 history = ReadTexelConsistent( prevMoments, prevWorldPos, localPos, allowedDist, localNormal, px, py, scrwidth, scrheight );
		if (history.x > -1)
		{
			lumDirect = 0.2f * lumDirect + 0.8f * history.x;
			lumDirect2 = 0.2f * lumDirect2 + 0.8f * history.y;
			lumIndirect = 0.2f * lumIndirect + 0.8f * history.z;
			lumIndirect2 = 0.2f * lumIndirect2 + 0.8f * history.w;
			int historySize = features[pixelIdx].w & 15;
			if (historySize < 15) features[pixelIdx].w++;
		}
		else features[pixelIdx].w &= 0xfffffff0; // reset history count
	}
	else features[pixelIdx].w &= 0xfffffff0; // reset history count
	// store motion vector and luminance moments
	motion[pixelIdx] = prevPixelPos;
	moments[pixelIdx] = make_float4( lumDirect, lumDirect2, lumIndirect, lumIndirect2 );
}
__host__ void prepareFilter( const float4* accumulator, uint4* features, const float4* worldPos, const float4* prevWorldPos,
	float4* shading, float2* motion, float4* moments, float4* prevMoments, const float4* deltaDepth,
	const ViewPyramid& prevView, const float j0, const float j1, const float prevj0, const float prevj1,
	const int w, const int h, const uint spp, const float directClamp, const float indirectClamp, const int flags )
{
	const float pixelValueScale = 1.0f / (float)spp;
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	// prepare data for reprojection
	const float3 centre = 0.5f * (prevView.p2 + prevView.p3);
	const float3 direction = normalize( centre - prevView.pos );
	const float3 right = normalize( prevView.p2 - prevView.p1 );
	const float3 up = normalize( prevView.p3 - prevView.p1 );
	const float focalDistance = length( centre - prevView.pos );
	const float screenSize = length( prevView.p3 - prevView.p1 );
	const float lenReci = h / screenSize;
	prepareFilterKernel << < gridDim, blockDim >> > (accumulator, features, worldPos, prevWorldPos, shading, motion, moments, prevMoments, deltaDepth,
		make_float4( prevView.pos, -(dot( prevView.pos, direction ) - dot( centre, direction )) ),
		make_float4( direction, 0 ),
		make_float4( right * lenReci, dot( prevView.p1, right ) * lenReci ),
		make_float4( up * lenReci, dot( prevView.p1, up ) * lenReci ),
		j0, j1, prevj0, prevj1, w, h, pixelValueScale, directClamp, indirectClamp, flags);
}

//  +-----------------------------------------------------------------------------+
//  |  applyFilterKernel                                                          |
//  |  Multi-phase SVGF filter kernel.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void __launch_bounds__( 64 /* max block size */, 6 /* min blocks per sm */ ) applyFilterKernel(
	const uint4* features, const float4* prevWorldPos, const float4* worldPos, const float4* deltaDepth, const float2* motion, const float4* moments,
	const float4* A, const float4* B, float4* C,
	const uint scrwidth, const uint scrheight, const int phase, const uint lastPass,
	const float brightness, const float contrastFactor )
{
	// float4 knob: { 10.0f, 5.0f, 7.5f, 0.5f }
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	const uint pixelIdx = x + y * scrwidth;
	// prepare reconstruction: gather info on local pixel
	const uint4 localFeature = features[pixelIdx];
	const float4 localPos = worldPos[pixelIdx];
	const float3 localNormal = UnpackNormal2( localFeature.y );
	const float3 localColor = RGB32toHDR( localFeature.x );
	const int localMatID = localFeature.w >> 4;
	float directlightWeightSum = 1;
	float indirectlightWeightSum = 1;
	const float4 combined = A[pixelIdx];
	float3 directLightSum = directlightWeightSum * GetDirectFromFloat4( combined );
	float3 indirectLightSum = indirectlightWeightSum * GetIndirectFromFloat4( combined );
	const float localDirect = Luminance( GetDirectFromFloat4( combined ) );
	const float localIndirect = Luminance( GetIndirectFromFloat4( combined ) );
	const float localDepth = __uint_as_float( localFeature.z );
	const float localDdx = deltaDepth[pixelIdx].z;
	const float localDdy = deltaDepth[pixelIdx].w;
	// determine variance
	float sigma_dir = 10.0f * oneoverpow2( phase - 1 );
	float sigma_ind = 10.0f * oneoverpow2( phase - 1 );
	const uint historySize = localFeature.w & 15;
	const float factor = historySize == 0 ? 400.0f : 1.0f;
	const float4 m = moments[pixelIdx];
	const float var_dir = m.y - m.x * m.x;
	const float var_ind = m.w - m.z * m.z;
	const float reci_sqrt_filt_var_dir_p = -1.0f / (sigma_dir * factor * sqrtf( var_dir + 0.00001f ) + 0.00001f);
	const float reci_sqrt_filt_var_ind_p = -1.0f / (sigma_ind * factor * sqrtf( var_ind + 0.00001f ) + 0.00001f);
	// reconstruct illumination
	const int step = 1 << (phase - 1);
	for (int vv = -2; vv <= 2; vv++)
	{
		const int v = vv * step + y;
		const int r = abs( vv ) == 2 ? 1 : 2;
		if (v >= 0 && v < scrheight) for (int uu = -r; uu <= r; uu++) if (uu != 0 || vv != 0)
		{
			const int u = clamp( uu * step + x, 0, (int)scrwidth - 1 );
			// edge stopping weights
			const uint localPixelIdx = u + v * scrwidth;
			const float4 combined = A[localPixelIdx];
			const uint4 neighborFeature = features[localPixelIdx];
			const float w_dist = (uu * uu + vv * vv) * (-1.0f / 7.5f);
			const float3 neighborDirect = GetDirectFromFloat4( combined );
			const float3 neighborIndirectLight = GetIndirectFromFloat4( combined );
			const float3 neighborNormal = UnpackNormal2( neighborFeature.y );
			float w_normal = powf( max( 0.0f, dot( neighborNormal, localNormal ) ), 128 );
			// depth weight. Don't set too aggressive or it will break with curved surfaces.
			const float expectedNeighborDepth = localDepth + localDdx * (float)(uu * step) + localDdy * (float)(vv * step);
			const float neighborDepthError = fabs( expectedNeighborDepth - __uint_as_float( neighborFeature.z ) );
			const float expectedDifference = fabs( expectedNeighborDepth - localDepth );
			const float w_depth = neighborDepthError / max( 0.00001f, (0.5f + phase * 0.5f) * expectedDifference );
			// minor weighting on albedo, different materials
			w_normal *= ((neighborFeature.w >> 4) != localMatID) ? 0.0001 : dot( localColor, RGB32toHDR( neighborFeature.x ) );
			// luminance weight, calculate separately for direct and indirect
			float w_dir = w_normal * __expf( fabs( localDirect - Luminance( neighborDirect ) ) * reci_sqrt_filt_var_dir_p + w_dist - w_depth );
			float w_ind = w_normal * __expf( fabs( localIndirect - Luminance( neighborIndirectLight ) ) * reci_sqrt_filt_var_ind_p + w_dist - w_depth );
			if (!isfinite( w_dir )) w_dir = 0;
			if (!isfinite( w_ind )) w_ind = 0;
			directLightSum += neighborDirect * w_dir, directlightWeightSum += w_dir;
			indirectLightSum += neighborIndirectLight * w_ind, indirectlightWeightSum += w_ind;
		}
	}
	float3 directFiltered = directLightSum * (1.0f / max( 0.0001f, directlightWeightSum ));
	float3 indirectFiltered = indirectLightSum * (1.0f / max( 0.0001f, indirectlightWeightSum ));
	if (phase == 1)
	{
		// average with filtered value from previous frame
		const float2 prevPixelPos = motion[pixelIdx];
		const int px = (int)prevPixelPos.x;
		const int py = (int)prevPixelPos.y;
		if (px >= 0 && px < scrwidth && py >= 0 && py < scrheight)
		{
			// look up the shading in the previous frame using a single (clamped) lookup.
			const float allowedDist2 = sqr( max( 0.05f, fabs( localDdx ) + fabs( localDdy ) ) );
			float3 prevDirect, prevIndirect;
			ReadTexelConsistent2( B, prevWorldPos, localPos, allowedDist2, localNormal, prevPixelPos.x, prevPixelPos.y, scrwidth, scrheight, prevDirect, prevIndirect );
			if (prevDirect.x != -1)
			{
			#if 1
				prevDirect = RGBToYCoCg( prevDirect ), prevIndirect = RGBToYCoCg( prevIndirect );
				float3 dirAvg = RGBToYCoCg( directFiltered ), dirVar = dirAvg * dirAvg, f;
				float3 indAvg = RGBToYCoCg( indirectFiltered ), indVar = indAvg * indAvg, g;
				float4 c4;
				if (x > 1)
				{
					if (y > 1) c4 = A[pixelIdx - scrwidth - 1], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
					c4 = A[pixelIdx - 1], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
					if (y < (scrheight - 1)) c4 = A[pixelIdx + scrwidth - 1], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
				}
				if (y > 1) c4 = A[pixelIdx - scrwidth], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
				if (y < (scrheight - 1)) c4 = A[pixelIdx + scrwidth], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
				if (x < (scrwidth - 1))
				{
					if (y > 1) c4 = A[pixelIdx + 1 - scrwidth], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
					c4 = A[pixelIdx + 1], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
					if (y < (scrheight - 1)) c4 = A[pixelIdx + 1 + scrwidth], f = RGBToYCoCg( GetDirectFromFloat4( c4 ) ), g = RGBToYCoCg( GetIndirectFromFloat4( c4 ) ), dirAvg += f, dirVar += f * f, indAvg += g, indVar += g * g;
				}
				dirAvg *= 1.0f / 9.0f, dirVar *= 1.0f / 9.0f, indAvg *= 1.0f / 9.0f, indVar *= 1.0f / 9.0f;
				float3 sigmaDir = max3( make_float3( 0.0f ), dirVar - dirAvg * dirAvg );
				float3 sigmaInd = max3( make_float3( 0.0f ), indVar - indAvg * indAvg );
				sigmaDir.x = sqrtf( sigmaDir.x ), sigmaDir.y = sqrtf( sigmaDir.y ), sigmaDir.z = sqrtf( sigmaDir.z );
				sigmaInd.x = sqrtf( sigmaInd.x ), sigmaInd.y = sqrtf( sigmaInd.y ), sigmaInd.z = sqrtf( sigmaInd.z );
				const float3 colorMinDir = dirAvg - 0.75f * sigmaDir;
				const float3 colorMaxDir = dirAvg + 0.75f * sigmaDir;
				const float3 colorMinInd = indAvg - 0.75f * sigmaInd;
				const float3 colorMaxInd = indAvg + 0.75f * sigmaInd;
				prevDirect = clamp( prevDirect, colorMinDir, colorMaxDir );
				prevIndirect = clamp( prevIndirect, colorMinInd, colorMaxInd );
				directFiltered = directFiltered * 0.1f + YCoCgToRGB( prevDirect ) * 0.9f;
				indirectFiltered = indirectFiltered * 0.1f + YCoCgToRGB( prevIndirect ) * 0.9f;
			#else
				directFiltered = 0.2f * directFiltered + 0.8f * prevDirect;
				indirectFiltered = 0.2f * indirectFiltered + 0.8f * prevIndirect;
			#endif
			}
		}
	}
	if (lastPass)
	{
		const float3 albedo = RGB32toHDR( localFeature.x );
		const float3 combined = (directFiltered + indirectFiltered) * albedo;
		// do brightness, contrast and gamma here, input for TAA
		const float r = sqrtf( max( 0.0f, (combined.x - 0.5f) * contrastFactor + 0.5f + brightness ) );
		const float g = sqrtf( max( 0.0f, (combined.y - 0.5f) * contrastFactor + 0.5f + brightness ) );
		const float b = sqrtf( max( 0.0f, (combined.z - 0.5f) * contrastFactor + 0.5f + brightness ) );
		C[pixelIdx] = make_float4( r, g, b, 1 );
	}
	else
	{
		// store the filtered value so we can reuse it in the next frame
		// if (isnan( directFiltered.x + directFiltered.y + directFiltered.z )) directFiltered = make_float3( 0 ); // happens...
		C[pixelIdx] = CombineToFloat4( directFiltered, indirectFiltered );
	}
}
__host__ void applyFilter(
	const uint4* features, const float4* prevWorldPos, const float4* worldPos, const float4* deltaDepth, const float2* motion, const float4* moments,
	const float4* A, const float4* B, float4* C, const uint w, const uint h, const int phase, const uint lastPass,
	const float brightness, const float contrast )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 2 ) / 2 ), blockDim( 32, 2 );
	// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	const float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	applyFilterKernel << < gridDim, blockDim >> > (features, prevWorldPos, worldPos, deltaDepth, motion, moments, A, B, C, w, h, phase, lastPass, brightness, contrastFactor);
}

//  +-----------------------------------------------------------------------------+
//  |  TAApassKernel                                                              |
//  |  Temporal antialiasing.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float maxComponent( const float3& a ) { return max( a.x, max( a.y, a.z ) ); }
LH2_DEVFUNC float3 slideTowardsAABB( float3 oldColor, float3 newColor, float3 minimum, float3 maximum, float maxVel )
{
	float overflow = max( 0.0f, max( maxComponent( minimum - oldColor ), maxComponent( oldColor - maximum ) ) );
	if (overflow <= 0.0) return oldColor;
	float ghost = max( 0.01f, 0.6f - max( maxVel * 3.0f, 5.0f * overflow * overflow ) );
	return lerp( newColor, oldColor, ghost );
}
__global__ void TAApassKernel(
	float4* pixels, float4* prevPixels, float j0, float j1, const float4* worldPos, const float4* prevWorldPos, const float2* motion,
	const uint scrwidth, const uint scrheight )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	const uint pixelIdx = x + y * scrwidth;
	float3 pixel = make_float3( pixels[pixelIdx] );
	// TAA
	const float2 prevPixelPos = motion[pixelIdx] - make_float2( 0.5f, 0.5f );
	if (prevPixelPos.x >= 0 && prevPixelPos.x < scrwidth && prevPixelPos.y >= 0 && prevPixelPos.y < scrheight)
	{
		// Marco Salvi's Implementation (by Chris Wyman)
		const float3 newPixel = RGBToYCoCg( pixel );
		float3 history = RGBToYCoCg( ReadTexelBmitchellNetravali( prevPixels, prevPixelPos.x, prevPixelPos.y, scrwidth, scrheight ) );
		float3 colorAvg = newPixel, colorVar = newPixel * newPixel, f;
		if (x > 1)
		{
			if (y > 1) f = RGBToYCoCg( make_float3( pixels[pixelIdx - scrwidth - 1] ) ), colorAvg += f, colorVar += f * f;
			f = RGBToYCoCg( make_float3( pixels[pixelIdx - 1] ) ), colorAvg += f, colorVar += f * f;
			if (y < (scrheight - 1)) f = RGBToYCoCg( make_float3( pixels[pixelIdx + scrwidth - 1] ) ), colorAvg += f, colorVar += f * f;
		}
		if (y > 1) f = RGBToYCoCg( make_float3( pixels[pixelIdx - scrwidth] ) ), colorAvg += f, colorVar += f * f;
		if (y < (scrheight - 1)) f = RGBToYCoCg( make_float3( pixels[pixelIdx + scrwidth] ) ), colorAvg += f, colorVar += f * f;
		if (x < (scrwidth - 1))
		{
			if (y > 1) f = RGBToYCoCg( make_float3( pixels[pixelIdx + 1 - scrwidth] ) ), colorAvg += f, colorVar += f * f;
			f = RGBToYCoCg( make_float3( pixels[pixelIdx + 1] ) ), colorAvg += f, colorVar += f * f;
			if (y < (scrheight - 1)) f = RGBToYCoCg( make_float3( pixels[pixelIdx + 1 + scrwidth] ) ), colorAvg += f, colorVar += f * f;
		}
		colorAvg *= 1.0f / 9.0f, colorVar *= 1.0f / 9.0f;
		float3 sigma = max3( make_float3( 0.0f ), colorVar - colorAvg * colorAvg );
		sigma.x = sqrtf( sigma.x ), sigma.y = sqrtf( sigma.y ), sigma.z = sqrtf( sigma.z );
		const float3 colorMin = colorAvg - 1.25f * sigma;
		const float3 colorMax = colorAvg + 1.25f * sigma;
		history = clamp( history, colorMin, colorMax );
		pixel = YCoCgToRGB( newPixel * 0.1f + history * 0.9f );
		if (isnan( pixel.x + pixel.y + pixel.z )) pixel = YCoCgToRGB( newPixel );
	}
	pixels[pixelIdx] = make_float4( min3( make_float3( 10 ), pixel ), 0 ); // for next frame
}
__host__ void TAApass(
	float4* pixels, float4* prevPixels, float pj0, float pj1, const float4* worldPos, const float4* prevWorldPos, const float2* motion,
	const uint w, const uint h )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	TAApassKernel << < gridDim, blockDim >> > (pixels, prevPixels, pj0, pj1, worldPos, prevWorldPos, motion, w, h);
}

//  +-----------------------------------------------------------------------------+
//  |  unsharpenTAAKernel                                                         |
//  |  Partial fix for the blur introduced by TAA.                          LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void unsharpenTAAKernel( const float4* pixels, const uint scrwidth, const uint scrheight )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x >= (scrwidth - 1) || y >= (scrheight - 1)) return;
#if 1
	const float4 p0 = pixels[x - 1 + (y - 1) * scrwidth];
	const float4 p1 = pixels[x + (y - 1) * scrwidth];
	const float4 p2 = pixels[x + 1 + (y - 1) * scrwidth];
	const float4 p3 = pixels[x + 1 + y * scrwidth];
	const float4 p4 = pixels[x + 1 + (y + 1) * scrwidth];
	const float4 p5 = pixels[x + (y + 1) * scrwidth];
	const float4 p6 = pixels[x - 1 + (y + 1) * scrwidth];
	const float4 p7 = pixels[x - 1 + y * scrwidth];
	const float4 pixel = max4( pixels[x + y * scrwidth], pixels[x + y * scrwidth] * 2.7f - 0.5f * (0.35f * p0 + 0.5f * p1 + 0.35f * p2 + 0.5f * p3 + 0.35f * p4 + 0.5f * p5 + 0.35f * p6 + 0.5f * p7) );
#else
	const float4 pixel = pixels[x + y * scrwidth];
#endif
	surf2Dwrite<float4>( make_float4( make_float3( pixel ), 0 ), renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void unsharpenTAA( const float4* pixels, const uint w, const uint h )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	unsharpenTAAKernel << < gridDim, blockDim >> > (pixels, w, h);
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeNoTAAKernel                                                        |
//  |  Finish the frame without TAA.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void finalizeNoTAAKernel( float4* pixels, const uint scrwidth, const uint scrheight, const float brightness, const float contrastFactor )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x >= (scrwidth - 1) || y >= (scrheight - 1)) return;
	float4 pixel = pixels[x + y * scrwidth];
	const float r = sqrtf( max( 0.0f, (pixel.x - 0.5f) * contrastFactor + 0.5f + brightness ) );
	const float g = sqrtf( max( 0.0f, (pixel.y - 0.5f) * contrastFactor + 0.5f + brightness ) );
	const float b = sqrtf( max( 0.0f, (pixel.z - 0.5f) * contrastFactor + 0.5f + brightness ) );
	surf2Dwrite<float4>( make_float4( r, g, b, 0 ), renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeNoTAA( float4* pixels, const uint w, const uint h, const float brightness, const float contrast )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	const float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	finalizeNoTAAKernel << < gridDim, blockDim >> > (pixels, w, h, brightness, contrastFactor);
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeFilterDebugKernel                                                  |
//  |  Raw dump of debug data.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void finalizeFilterDebugKernel( const uint scrwidth, const uint scrheight )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x >= (scrwidth - 1) || y >= (scrheight - 1)) return;
	float4 pixel = debugData[x + y * scrwidth];
	surf2Dwrite<float4>( pixel, renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeFilterDebug( const uint w, const uint h )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	finalizeFilterDebugKernel << < gridDim, blockDim >> > (w, h);
}

// EOF