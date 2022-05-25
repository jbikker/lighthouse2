/* common_settings.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are global:
   - available in host and device code
   - the same for each core.
   Settings that can be configured per core can be found in core_settings.h.
*/

#pragma once

// global settings
#define CACHEIMAGES					// imported images will be saved to bin files (faster)

// default screen size
#define SCRWIDTH			1600
#define SCRHEIGHT			920

// skydome defines
// #define IBL						// calculate pdf and cdf for ibl renderer
// #define TESTSKY					// red/green/blue area lights for debugging
#define IBLWIDTH			512
#define IBLHEIGHT			256
#define IBLWBITS			9
#define IBLHBITS			8

// PNEE settings
#define PHOTONCOUNT			5000000
#define GRIDDIMX			128
#define GRIDDIMY			128
#define GRIDDIMZ			128
#define CDFSIZE				16		// Note: CUDA code assumes 16 (hardcoded)
#define CDFFLOOR			0.1f

// low level settings
#define PI					3.14159265358979323846264f
#define INVPI				0.31830988618379067153777f
#define INV2PI				0.15915494309189533576888f
#define TWOPI				6.28318530717958647692528f
#define SQRT_PI_INV			0.56418958355f
#define LARGE_FLOAT			1e34f
#define EPSILON				0.0001f
#define MINROUGHNESS		0.0001f	// minimal GGX roughness
#define BLACK				make_float3( 0 )
#define WHITE				make_float3( 1 )
#define MIPLEVELCOUNT		5

// file format versions
#define BINTEXFILEVERSION	0x10001001

// tools

// nan chasing
#ifndef __OPENCLCC__
#define FIXNAN_FLOAT3(a)	{if(!isfinite(a.x+a.y+a.z))a=make_float3(0);}
#define FIXNAN_FLOAT4(a)	{if(!isfinite(a.x+a.y+a.z+a.w))a=make_float4(0);}
#define REPORTNAN_FLOAT3(a)	{if(!isfinite(a.x+a.y+a.z))printf("getting NaNs here!");}
#define REPORTNAN_FLOAT4(a)	{if(!isfinite(a.x+a.y+a.z))printf("getting NaNs here!");}
#else
#define FIXNAN_FLOAT3(a)	{if(isnan(a.x+a.y+a.z))a=make_float3(0);}
#define FIXNAN_FLOAT4(a)	{if(isnan(a.x+a.y+a.z+a.w))a=make_float4(0);}
#define REPORTNAN_FLOAT3(a)	{if(isnan(a.x+a.y+a.z))printf("getting NaNs here!");}
#define REPORTNAN_FLOAT4(a)	{if(isnan(a.x+a.y+a.z))printf("getting NaNs here!");}
#endif

// Get the log2 for an integer using the preprocessor.
// https://stackoverflow.com/questions/27581671/how-to-compute-log-with-the-preprocessor
#define NB_(N,B) (((unsigned long)N >> B) > 0)
#define BITS_TO_REPRESENT( N )                                       \
        (NB_((N),  0) + NB_((N),  1) + NB_((N),  2) + NB_((N),  3) + \
         NB_((N),  4) + NB_((N),  5) + NB_((N),  6) + NB_((N),  7) + \
         NB_((N),  8) + NB_((N),  9) + NB_((N), 10) + NB_((N), 11) + \
         NB_((N), 12) + NB_((N), 13) + NB_((N), 14) + NB_((N), 15) + \
         NB_((N), 16) + NB_((N), 17) + NB_((N), 18) + NB_((N), 19) + \
         NB_((N), 20) + NB_((N), 21) + NB_((N), 22) + NB_((N), 23) + \
         NB_((N), 24) + NB_((N), 25) + NB_((N), 26) + NB_((N), 27) + \
         NB_((N), 28) + NB_((N), 29) + NB_((N), 30) + NB_((N), 31) )

// EOF