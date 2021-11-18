/* core_settings.h - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The settings and classes in this file are core-specific:
   - available in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

// core-specific settings
#define CLAMPFIREFLIES		// suppress fireflies by clamping
#define CONSISTENTNORMALS	// consistent normal interpolation; don't use with filtering?

// low-level settings
#define BLUENOISE			// use blue noise instead of uniform random numbers
#define TAA					// really basic temporal antialiasing
#define BILINEAR			// enable bilinear interpolation
// #define NOTEXTURES		// all texture reads will be white

#define APPLYSAFENORMALS	if (dot( N, wi ) <= 0) pdf = 0;
#define NOHIT				-1

//#define FLAGS_ON
#define BLUENOISER_ON (true) // only for the first extension of eye path

#define NEW_PATH            0
#define EXTEND_EYEPATH      1
#define EXTEND_LIGHTPATH    2
#define DEAD                3

// The length of eye and light must be the same, otherwise, the sum of MIS is not correct
#define MAXPATHLENGTH	5               // [1,32)
#define MAX_EYEPATH     (MAXPATHLENGTH)   // eye starts from 0
#define MAX_LIGHTPATH   (MAXPATHLENGTH)   // light starts from 1
#define NKK (MAX_LIGHTPATH * 0.85f)

// for directional light
#define DIRECTIONAL_LIGHT

#define SCENE_CENTER (make_float3(0.0f))
#define SCENE_RADIUS  100.0f
#define SCENE_AREA (PI * SCENE_RADIUS * SCENE_RADIUS)

#define RadiusFactor 0.01f

#define VIS_BUFFERSIZE 5;

#ifndef __CUDACC__

#define CUDABUILD
#include "helper_math.h"	// for vector types
#include "platform.h"
#include "system.h"
#undef APIENTRY				// get rid of an annoying warning
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "nvrtc.h"
#include "FreeImage.h"		// for loading blue noise
#include "shared_host_code/cudatools.h"
#include "shared_host_code/interoptexture.h"
#include <optix_prime/optix_prime.h>

#include <optixu/optixpp_namespace.h>
using namespace optix;
using namespace lighthouse2;

#include "core_mesh.h"

using namespace lh2core;

#endif

// for a full path state, we need a Ray, an Intersection, and
// the data specified in PathState.
struct PathState
{
	float3 O;				// normal at last path vertex
	uint data;				// holds pixel index << 8, path length in bits 0..3, flags in bits 4..7
	float3 D; int N;		// ray direction of the current path segment, encoded normal in w
	float3 throughput;		// path transport
	float bsdfPdf;			// postponed pdf
};
struct PathState4 { float4 O4, D4, T4; };

struct BiPathState
{
	// light vertex
	float4 data0; // light_throughput light_dL
	float4 data1; // light_beta light_p
	float4 data2; // light_pos light_pdf_solid
	float4 data3; // light_dir queryId

	// eye vertex
	float4 data4; // eye_throughput eye_dE
	float4 data5; // eye_beta eye_p
	float4 data6; // eye_pos eye_pdf_solid
	float4 data7; // eye_dir queryId

	float4 light_intersection;
	float4 eye_intersection;

	float4 light_normal; // normal + jobIndex + flag
	float4 eye_normal; // normal + w: s_t_type_pass

	float4 currentLight_hitData;
	float4 pre_light_dir;
};

// counters and other global data, in device memory
struct Counters
{
	uint constructionLightPos;
	uint constructionEyePos;

	uint extendEyePath;
	uint extendLightPath;

	uint visibilityRays;
	uint randomWalkRays;

	int probedInstid;
	int probedTriid;
	float probedDist;

	uint contribution_count;
};

// internal material representation
struct CUDAMaterial
{
#ifndef OPTIX_CU
	void SetDiffuse( float3 d ) { diffuse_r = d.x, diffuse_g = d.y, diffuse_b = d.z; }
	void SetTransmittance( float3 t ) { transmittance_r = t.x, transmittance_g = t.y, transmittance_b = t.z; }
	struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; };
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b; uint flags;
	uint4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
	// texture / normal map descriptors; exactly 128-bit each
	Map tex0, tex1, nmap0, nmap1, smap, rmap;
#endif
};

struct CUDAMaterial4
{
	uint4 baseData4;
	uint4 parameters;
	uint4 t0data4;
	uint4 t1data4;
	uint4 n0data4;
	uint4 n1data4;
	uint4 sdata4;
	uint4 rdata4;
	// flag query macros
#define ISDIELECTRIC				(1 << 0)
#define DIFFUSEMAPISHDR				(1 << 1)
#define HASDIFFUSEMAP				(1 << 2)
#define HASNORMALMAP				(1 << 3)
#define HASSPECULARITYMAP			(1 << 4)
#define HASROUGHNESSMAP				(1 << 5)
#define ISANISOTROPIC				(1 << 6)
#define HAS2NDNORMALMAP				(1 << 7)
#define HAS2NDDIFFUSEMAP			(1 << 9)
#define HASSMOOTHNORMALS			(1 << 11)
#define HASALPHA					(1 << 12)
#define HASMETALNESSMAP				(1 << 13)
#define MAT_ISDIELECTRIC			(flags & ISDIELECTRIC)
#define MAT_DIFFUSEMAPISHDR			(flags & DIFFUSEMAPISHDR)
#define MAT_HASDIFFUSEMAP			(flags & HASDIFFUSEMAP)
#define MAT_HASNORMALMAP			(flags & HASNORMALMAP)
#define MAT_HASSPECULARITYMAP		(flags & HASSPECULARITYMAP)
#define MAT_HASROUGHNESSMAP			(flags & HASROUGHNESSMAP)
#define MAT_ISANISOTROPIC			(flags & ISANISOTROPIC)
#define MAT_HAS2NDNORMALMAP			(flags & HAS2NDNORMALMAP)
#define MAT_HAS2NDDIFFUSEMAP		(flags & HAS2NDDIFFUSEMAP)
#define MAT_HASSMOOTHNORMALS		(flags & HASSMOOTHNORMALS)
#define MAT_HASALPHA				(flags & HASALPHA)
#define MAT_HASMETALNESSMAP			(flags & HASMETALNESSMAP)
};

// ------------------------------------------------------------------------------
// Below this line: derived, low-level and internal.

// clamping
#ifdef CLAMPFIREFLIES
#define CLAMPINTENSITY		const float v=max(contribution.x,max(contribution.y,contribution.z)); \
							if(v>clampValue){const float m=clampValue/v;contribution.x*=m; \
							contribution.y*=m;contribution.z*=m; /* don't touch w */ }
#else
#define CLAMPINTENSITY
#endif

// OptiX Prime interface structs
struct Ray { float3 O; float tmin; float3 D; float tmax; };
struct Ray4 { float4 O4, D4; };
struct Intersection { float t; int triid, instid; float u, v; };

#ifndef __CUDACC__

#include "core_api_base.h"
#include "rendercore.h"

#include "../RenderSystem/common_bluenoise.h"

#endif

// EOF