/* core_settings.h - Copyright 2019 Utrecht University

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
   - avilable in host and device code
   - specific to this particular core.
   Global settings can be configured shared.h.
*/

#pragma once

// core-specific settings
#define CLAMPFIREFLIES		// suppress fireflies by clamping
#define MAXPATHLENGTH		3
// #define USE_LAMBERT_BSDF	// override default microfacet model
// #define USE_MULTISCATTER_BSDF // override default microfacet model
// #define GGXCONDUCTOR // alternative is the diffuse ggx brdf
#define SINGLEBOUNCE		// perform only a single diffuse bounce
#define CONSISTENTNORMALS	// consistent normal interpolation; don't use with filtering?

// low-level settings
#define SCATTERSTEPS 1		// max bounces in microfacet evaluation (multiscatter bsdf)
#define BLUENOISE			// use blue noise instead of uniform random numbers
#define TAA					// really basic temporal antialiasing
#define BILINEAR			// enable bilinear interpolation
// #define NOTEXTURES		// all texture reads will be white

#define APPLYSAFENORMALS	if (dot( N, wi ) <= 0) pdf = 0;
#define NOHIT				-1

#ifndef __CUDACC__

#ifdef _DEBUG
#pragma comment(lib, "../platform/lib/debug/platform.lib" )
#else
#pragma comment(lib, "../platform/lib/release/platform.lib" )
#endif
#pragma comment(lib, "../OptiX/lib64/optix_prime.1.lib" )

#define CUDABUILD
#include "helper_math.h"	// for vector types
#include "platform.h"
#include "system.h"
#undef APIENTRY				// get rid of an anoying warning
#include "cuda.h"
#include "cuda_gl_interop.h"
#include "nvrtc.h"
#include "FreeImage.h"		// for loading blue noise
#include "shared_host_code/cudatools.h"
#include "shared_host_code/interoptexture.h"
#include "optix_prime/optix_prime.h"

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

// counters and other global data, in device memory
struct Counters
{
	uint activePaths;
	uint shaded;
	uint generated;
	uint connected;
	uint extended;
	uint extensionRays;
	uint shadowRays;
	uint totalExtensionRays;
	uint totalShadowRays;
	int probedInstid;
	int probedTriid;
	float probedDist;
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
#include "core_api.h"
#include "rendercore.h"

#endif

// EOF