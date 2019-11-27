/* finalize_shared.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE: used in
   - RenderCore_OptixPrime_b
   - RenderCore_Optix7
   - RenderCore_Optix7Filter
   - RenderCore_PrimeRef
*/

#include "noerrors.h"

//  +-----------------------------------------------------------------------------+
//  |  finalizeRenderKernel                                                       |
//  |  Presenting the accumulator. Gamma, brightness and contrast will be done    |
//  |  in postprocessing.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void finalizeRenderKernel( const float4* accumulator, const int scrwidth, const int scrheight, const float pixelValueScale )
{
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	// plot scaled pixel
	float4 value = accumulator[x + y * scrwidth] * pixelValueScale;
	surf2Dwrite<float4>( value, renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeRender( const float4* accumulator, const int w, const int h, const int spp )
{
	const float pixelValueScale = 1.0f / (float)spp;
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	finalizeRenderKernel << < gridDim, blockDim >> > (accumulator, w, h, pixelValueScale);
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
	float3 localNormal = UnpackNormal2( __float_as_uint( currentWorldPos.w ) );
	float3 prevNormal = UnpackNormal2( __float_as_uint( p.w ) );
	if ((__float_as_uint( p.w ) & 3) != 1) return 1e21f; // we are always searching for specular pixels
	if (dot( localNormal, prevNormal ) < 0.85f) return 1e21f; // normals should at least be somewhat simular
	return length( make_float3( currentWorldPos - p ) );
}
LH2_DEVFUNC float FineWorldDistance( const float2& pixel, const float4& currentWorldPos, const float4* prevWorldPos, const int w, const int h )
{
	const int2 p0 = make_int2( pixel.x, pixel.y );
	const int2 p1 = make_int2( p0.x + 1, p0.y );
	const int2 p2 = make_int2( p0.x, p0.y + 1 );
	const int2 p3 = make_int2( p0.x + 1, p0.y + 1 );
	const float fx = pixel.x - floor( pixel.x );
	const float fy = pixel.y - floor( pixel.y );
	const float w0 = (1 - fx) * (1 - fy), w1 = fx * (1 - fy), w2 = (1 - fx) * fy, w3 = fx * fy;
	float d0 = WorldDistance( p0, currentWorldPos, prevWorldPos, w, h );
	float d1 = WorldDistance( p1, currentWorldPos, prevWorldPos, w, h );
	float d2 = WorldDistance( p2, currentWorldPos, prevWorldPos, w, h );
	float d3 = WorldDistance( p3, currentWorldPos, prevWorldPos, w, h );
	float totalWeight = 0;
	float totalDist = 0;
	if (d0 < 1e20f) totalDist += d0 * w0, totalWeight += w0;
	if (d1 < 1e20f) totalDist += d1 * w1, totalWeight += w1;
	if (d2 < 1e20f) totalDist += d2 * w2, totalWeight += w2;
	if (d3 < 1e20f) totalDist += d3 * w3, totalWeight += w3;
	if (totalWeight == 0) return 1e20f; else return totalDist / totalWeight;
}
LH2_DEVFUNC int RefineHistoryPos( float2& currentScreenPos, const float2& offset, const float4& localWorldPos, float& bestDist, const float step, const float4* prevWorldPos, const int w, const int h )
{
	int bestTap = 0;
	const float2 pixelPos0 = make_float2( currentScreenPos.x - step, currentScreenPos.y );
	const float2 pixelPos1 = make_float2( currentScreenPos.x + step, currentScreenPos.y );
	const float2 pixelPos2 = make_float2( currentScreenPos.x, currentScreenPos.y - step );
	const float2 pixelPos3 = make_float2( currentScreenPos.x, currentScreenPos.y + step );
	float d;
	d = FineWorldDistance( pixelPos0 + offset, localWorldPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos0, bestTap = 1;
	d = FineWorldDistance( pixelPos1 + offset, localWorldPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos1, bestTap = 2;
	d = FineWorldDistance( pixelPos2 + offset, localWorldPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos2, bestTap = 3;
	d = FineWorldDistance( pixelPos3 + offset, localWorldPos, prevWorldPos, w, h );
	if (d < bestDist) bestDist = d, currentScreenPos = pixelPos3, bestTap = 4;
	return bestTap;
}
__global__ void prepareFilterKernel( const float4* accumulator, uint4* features, const float4* worldPos, const float4* prevWorldPos,
	float4* shading, float2* motion, float4* moments, float4* prevMoments, const float4* deltaDepth,
	const float4 prevPos, const float4 prevE, const float4 prevRight, const float4 prevUp, const float j0, const float j1, const float prevj0, const float prevj1,
	const int scrwidth, const int scrheight, const float pixelValueScale, const float directClamp, const float indirectClamp, const int camIsStationary )
{
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((x >= scrwidth) || (y >= scrheight)) return;
	const int pixelIdx = x + y * scrwidth;
	// split direct and indirect light from albedo and clamp
	const float3 direct = make_float3( accumulator[pixelIdx] ) * pixelValueScale;
	const uint4 localFeature = features[pixelIdx];
	const float4 localWorldPos = worldPos[pixelIdx];
	const float3 albedo = RGB32toHDRmin1( localFeature.x );
	const float3 indirect = make_float3( accumulator[pixelIdx + scrwidth * scrheight] ) * pixelValueScale;
	const float3 reciAlbedo = make_float3( 1.0f / albedo.x, 1.0f / albedo.y, 1.0f / albedo.z );
	const float3 directLight = min3( direct * reciAlbedo, directClamp );
	const float3 indirectLight = min3( indirect * reciAlbedo, indirectClamp );
	shading[pixelIdx] = CombineToFloat4( directLight, indirectLight );
	float lumDirect = Luminance( directLight ), lumDirect2 = lumDirect * lumDirect;
	float lumIndirect = Luminance( indirectLight ), lumIndirect2 = lumIndirect * lumIndirect;
	// reproject
	debugData[pixelIdx] = make_float4( x == 50 || y == 50 ? 1 : 0, 0, 0, 1 );
	float2 prevPixelPos;
	if (((localFeature.w >> 4) & 3) == 0 /* bit 4 and 5 of feature.w: zero = diffuse, non-zero is specular */)
	{
		// calculate location in screen space of the current pixel in the previous frame
		const float4 D = make_float4( normalize( make_float3( localWorldPos - prevPos ) ), 0 );
		const float4 S = make_float4( make_float3( prevPos + D * (prevPos.w / dot( prevE, D )) ), 0 );
		prevPixelPos = make_float2( dot( S, prevRight ) - prevRight.w - j0, dot( S, prevUp ) - prevUp.w - j1 );
	}
	else
	{
		prevPixelPos = make_float2( x, y ); // best option when camera doesn't move.
		if (!camIsStationary)
		{
			// perform a diamond search for the world space position of the current pixel in the previous frame
			float bestDist = length( make_float3( localWorldPos - prevWorldPos[pixelIdx] ) ), stepSize = 5.0f;
			const float2 offset = make_float2( j0 - prevj0, j1 - prevj1 );
			int iter = 0;
			while (1)
			{
				const int tap = RefineHistoryPos( prevPixelPos, offset, localWorldPos, bestDist, stepSize, prevWorldPos, scrwidth, scrheight );
				if (tap == 0) { stepSize *= 0.45f; if (stepSize < 0.05f) break; }
				if (++iter == 25) break;
			}
		}
	}
	// get history luminance moments
	prevPixelPos += make_float2( 0.5f );
	const float px = prevPixelPos.x;
	const float py = prevPixelPos.y;
	if (px >= 0 && px < scrwidth && py >= 0 && py < scrheight)
	{
		const float localDdx = deltaDepth[pixelIdx].z;
		const float localDdy = deltaDepth[pixelIdx].w;
		const float allowedDist = max( 0.05f, fabs( localDdx ) + fabs( localDdy ) );
		const float3 localNormal = UnpackNormal2( localFeature.y );
		const float4 history = ReadTexelConsistent( prevMoments, prevWorldPos, localWorldPos, allowedDist, localNormal, px, py, scrwidth, scrheight );
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
	const int w, const int h, const uint spp, const float directClamp, const float indirectClamp, const int camIsStationary )
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
		j0, j1, prevj0, prevj1, w, h, pixelValueScale, directClamp, indirectClamp, camIsStationary);
}

//  +-----------------------------------------------------------------------------+
//  |  applyFilterKernel                                                          |
//  |  Multi-phase SVGF filter kernel.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void __launch_bounds__( 64 /* max block size */, 6 /* min blocks per sm */ ) applyFilterKernel(
	const uint4* features, const float4* prevWorldPos, const float4* worldPos, const float4* deltaDepth, const float2* motion, const float4* moments,
	const float4* A, const float4* B, float4* C,
	const uint scrwidth, const uint scrheight, const int phase, const uint lastPass )
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
		// obtain best mapping of a pixel in the current frame to a pixel in the previous frame
		float2 prevPixelPos = motion[pixelIdx];
	#if 0
		// motion vector dilation
		float bestDepth = 1e20f;
		for( int v = 0; v < 3; v++ ) for( int u = 0; u < 3; u++ )
		{
			int uu = x + (u - 1);
			int vv = y + (v - 1);
			if (uu >= 0 && vv >= 0 && uu < scrwidth && vv < scrheight)
			{
				float depth = __uint_as_float( features[uu + vv * scrwidth].z );
				if (depth < bestDepth) bestDepth = depth, prevPixelPos = motion[uu + vv * scrwidth];
			}
		}
	#endif
		// average with filtered value from previous frame
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
		// we need gamma for the filter; brightness, contrast will be done in postproc
		const float r = sqrtf( combined.x );
		const float g = sqrtf( combined.y );
		const float b = sqrtf( combined.z );
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
	const float4* A, const float4* B, float4* C, const uint w, const uint h, const int phase, const uint lastPass )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 2 ) / 2 ), blockDim( 32, 2 );
	applyFilterKernel << < gridDim, blockDim >> > (features, prevWorldPos, worldPos, deltaDepth, motion, moments, A, B, C, w, h, phase, lastPass);
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
	// SVGF needs gamma corrected pixels for TAA, but we also apply gamma in the postproc, after
	// tonemapping. So, the gamma corrected data is uncorrected here, so we can correct it again later...
	float3 squared = make_float3( pixel.x * pixel.x, pixel.y * pixel.y, pixel.z * pixel.z );
	surf2Dwrite<float4>( make_float4( squared, 0 ), renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
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
__global__ void finalizeNoTAAKernel( float4* pixels, const uint scrwidth, const uint scrheight )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x >= (scrwidth - 1) || y >= (scrheight - 1)) return;
	float4 pixel = pixels[x + y * scrwidth];
	surf2Dwrite<float4>( make_float4( sqrtf( pixel.x ), sqrtf( pixel.y ), sqrtf( pixel.z ), 0 ), renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeNoTAA( float4* pixels, const uint w, const uint h )
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	finalizeNoTAAKernel << < gridDim, blockDim >> > (pixels, w, h);
}

//  +-----------------------------------------------------------------------------+
//  |  finalizeFilterDebugKernel                                                  |
//  |  Raw dump of debug data.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void finalizeFilterDebugKernel( const uint scrwidth, const uint scrheight, const uint4* features, const float4* worldPos,
	const float4* prevWorldPos, const float4* deltaDepth, const float2* motion, const float4* moments, const float4* shading )
{
	// get x and y for pixel
	const uint x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x >= (scrwidth - 1) || y >= (scrheight - 1)) return;
#if 1
	// extract data for a multi-view of the various buffers we have at this point
	int pixelIdx = x + y * scrwidth;
	float3 albedo = RGB32toHDR( features[pixelIdx].x );
	// uint matID = features[pixelIdx].w >> 6;
	float3 deltaWorldPos = make_float3( worldPos[pixelIdx] - prevWorldPos[pixelIdx] );
	float3 N = UnpackNormal2( features[pixelIdx].y /* or __float_as_uint( worldPos[pixelIdx].w ) */ );
	float dist = __uint_as_float( features[pixelIdx].z );
	float3 directLight = GetDirectFromFloat4( shading[pixelIdx] );
	float3 indirectLight = GetIndirectFromFloat4( shading[pixelIdx] );
	// prepare the multi-view
	float4 pixel;
	if (x < (scrwidth / 2))
	{
		if (y < (scrheight / 2)) pixel = make_float4( (N + make_float3( 1 )) * 0.5f, 1 );
		else pixel = make_float4( (x < (scrwidth / 4) ? directLight : indirectLight), 1 );
	}
	else
	{
		// if (y < (scrheight / 2)) pixel = dist < 1000 ? make_float4( (deltaWorldPos + make_float3( 1 )) * 0.5f, 1 ) : make_float4( 0 );
		if (y < (scrheight / 2)) pixel = dist < 1000 ? make_float4( motion[pixelIdx].x, motion[pixelIdx].y, 0, 1 ) : make_float4( 0 );
		else pixel = make_float4( albedo, 1 );
	}
#else
	// simply show the contents of debugData for this pixel
	float4 pixel = debugData[x + y * scrwidth];
#endif
	surf2Dwrite<float4>( pixel, renderTarget, x * sizeof( float4 ), y, cudaBoundaryModeClamp );
}
__host__ void finalizeFilterDebug( const uint w, const uint h,
	// data available after gathering frame data in shade (pathtracer.h)
	const uint4* features,		// x: albedo; y: packed normal; z: dist from cam; w: 4-bit history count, 2-bit specularity flags, (matid << 6)
	const float4* worldPos,		// xyz: world pos first non-specular vertex; w: packed normal (again)
	const float4* prevWorldPos, // worldPos data for the previous frame
	const float4* deltaDepth,	// xy: unused; zw: depth derivatives over x and y
	// data available after processing gathered frame data in prepareFilter
	const float2* motion,		// per-pixel motion vectors
	const float4* moments,		// xyzw: lumDirect, lumDirect2, lumIndirect, lumIndirect2
	const float4* shading		// indirect and indirect shading; unpack using Get(Ind|D)irectFromFloat4
)
{
	const dim3 gridDim( NEXTMULTIPLEOF( w, 32 ) / 32, NEXTMULTIPLEOF( h, 8 ) / 8 ), blockDim( 32, 8 );
	finalizeFilterDebugKernel << < gridDim, blockDim >> > (w, h, features, worldPos, prevWorldPos, deltaDepth, motion, moments, shading);
}

// EOF