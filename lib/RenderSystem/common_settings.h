/* common_settings.h - Copyright 2019 Utrecht University

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
   - avilable in host and device code
   - the same for each core.
   Settings that can be configured per core can be found in core_settings.h.
*/

#pragma once

// global settings
#define CACHEIMAGES					// imported images will be saved to bin files (faster)
// #define ZIPIMGBINS				// cached images will be zipped (slower but smaller)

// default screen size
#define SCRWIDTH			1600
#define SCRHEIGHT			900

// skydome defines
// #define IBL						// calculate pdf and cdf for ibl renderer
// #define TESTSKY					// red/green/blue area lights for debugging
#define IBLWIDTH			512
#define IBLHEIGHT			256
#define IBLWBITS			9
#define IBLHBITS			8

// low discrepancy sampling
#define LDSETS				256		// number of full low discrepancy sets
#define LDDIMENSIONS		16		// number of dimensions per ld set
#define LDSAMPLES			128		// number of samples per pixel before we start using random floats

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
#else
#define FIXNAN_FLOAT3(a)	{if(isnan(a.x+a.y+a.z))a=make_float3(0);}
#define FIXNAN_FLOAT4(a)	{if(isnan(a.x+a.y+a.z+a.w))a=make_float4(0);}
#endif

// EOF