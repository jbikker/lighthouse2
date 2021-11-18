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

// need to use GLFW from its DLL to be able to use GLFW from both app and render core
#define GLFW_DLL

#define INCLUDE_BLUENOISE

#ifdef WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#ifdef _DEBUG
#pragma comment(lib, "../platform/lib/debug/platform.lib" )
#else
#pragma comment(lib, "../platform/lib/release/platform.lib" )
#endif

#include "platform.h"

#define NV_EXTENSIONS

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#ifdef WIN32
#include <direct.h>
#endif

struct GeometryInstance;

using namespace lighthouse2;

// clang-format off
#include "bindings.h"
#include <vulkan/vulkan.hpp>
#include <shaderc/shaderc.hpp>

#define CheckVK( x ) _CheckVK( __LINE__, __FILE__, static_cast<vk::Result>( x ) )
static void _CheckVK( int line, const char *file, vk::Result x )
{
	const char *error;

	switch (x)
	{
	case (vk::Result::eSuccess):
		return;
	case (vk::Result::eNotReady):
		error = "VK_NOT_READY";
		break;
	case (vk::Result::eTimeout):
		error = "VK_TIMEOUT";
		break;
	case (vk::Result::eEventSet):
		error = "VK_EVENT_SET";
		break;
	case (vk::Result::eIncomplete):
		error = "VK_INCOMPLETE";
		break;
	case (vk::Result::eErrorOutOfHostMemory):
		error = "VK_ERROR_OUT_OF_HOST_MEMORY";
		break;
	case (vk::Result::eErrorOutOfDeviceMemory):
		error = "VK_ERROR_OUT_OF_DEVICE_MEMORY";
		break;
	case (vk::Result::eErrorInitializationFailed):
		error = "VK_ERROR_INITIALIZATION_FAILED";
		break;
	case (vk::Result::eErrorDeviceLost):
		error = "VK_ERROR_DEVICE_LOST";
		break;
	case (vk::Result::eErrorMemoryMapFailed):
		error = "VK_ERROR_MEMORY_MAP_FAILED";
		break;
	case (vk::Result::eErrorLayerNotPresent):
		error = "VK_ERROR_LAYER_NOT_PRESENT";
		break;
	case (vk::Result::eErrorExtensionNotPresent):
		error = "VK_ERROR_EXTENSION_NOT_PRESENT";
		break;
	case (vk::Result::eErrorFeatureNotPresent):
		error = "VK_ERROR_FEATURE_NOT_PRESENT";
		break;
	case (vk::Result::eErrorIncompatibleDriver):
		error = "VK_ERROR_INCOMPATIBLE_DRIVER";
		break;
	case (vk::Result::eErrorTooManyObjects):
		error = "VK_ERROR_TOO_MANY_OBJECTS";
		break;
	case (vk::Result::eErrorFormatNotSupported):
		error = "VK_ERROR_FORMAT_NOT_SUPPORTED";
		break;
	case (vk::Result::eErrorFragmentedPool):
		error = "VK_ERROR_FRAGMENTED_POOL";
		break;
	case (vk::Result::eErrorOutOfPoolMemory):
		error = "VK_ERROR_OUT_OF_POOL_MEMORY";
		break;
	case (vk::Result::eErrorInvalidExternalHandle):
		error = "VK_ERROR_INVALID_EXTERNAL_HANDLE";
		break;
	case (vk::Result::eErrorSurfaceLostKHR):
		error = "VK_ERROR_SURFACE_LOST_KHR";
		break;
	case (vk::Result::eErrorNativeWindowInUseKHR):
		error = "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
		break;
	case (vk::Result::eSuboptimalKHR):
		error = "VK_SUBOPTIMAL_KHR";
		break;
	case (vk::Result::eErrorOutOfDateKHR):
		error = "VK_ERROR_OUT_OF_DATE_KHR";
		break;
	case (vk::Result::eErrorIncompatibleDisplayKHR):
		error = "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR";
		break;
	case (vk::Result::eErrorValidationFailedEXT):
		error = "VK_ERROR_VALIDATION_FAILED_EXT";
		break;
	case (vk::Result::eErrorInvalidShaderNV):
		error = "VK_ERROR_INVALID_SHADER_NV";
		break;
	case (vk::Result::eErrorInvalidDrmFormatModifierPlaneLayoutEXT):
		error = "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT";
		break;
	case (vk::Result::eErrorFragmentationEXT):
		error = "VK_ERROR_FRAGMENTATION_EXT";
		break;
	case (vk::Result::eErrorNotPermittedEXT):
		error = "VK_ERROR_NOT_PERMITTED_EXT";
		break;
	case (vk::Result::eErrorInvalidDeviceAddressEXT):
		error = "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT";
		break;
	case (vk::Result::eErrorFullScreenExclusiveModeLostEXT):
		error = "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT";
		break;
	default:
		error = "UNKNOWN";
		break;
	}

	FATALERROR( "Vulkan error on line %d in %s: %u = %s", line, file, uint( x ), error );
}

#include <memory>
#include <unordered_set>
#include <map>
#include <set>

#include "core_api_base.h"

#include "vulkan_device.h"

#include "vulkan_core_buffer.h"
#include "vulkan_gl_texture_interop.h"

#include "vulkan_core_buffer.h"
#include "uniform_objects.h"

#include "vulkan_shader.h"
#include "vulkan_image.h"
#include "vulkan_descriptor_set.h"
#include "vulkan_shader_binding_table_generator.h"
#include "vulkan_ray_trace_nv_pipeline.h"
#include "vulkan_compute_pipeline.h"

#include "core_mesh.h"

#include "top_level_as.h"
#include "bottom_level_as.h"
#include "vulkan_gl_texture_interop.h"

#define NEXTMULTIPLEOF(a,b)	(((a)+((b)-1))&(0x7fffffff-((b)-1)))

struct GeometryInstance
{
	GeometryInstance() = default;
	std::array<float, 12> transform;	  // Transform matrix, containing only the top 3 rows
	uint32_t instanceId : 24;			  // Instance index
	uint32_t mask : 8;					  // Visibility mask
	uint32_t instanceOffset : 24;		  // Index of the hit group which will be invoked when a ray hits the instance
	uint32_t flags : 8;					  // Instance flags, such as culling
	uint64_t accelerationStructureHandle; // Opaque handle of the bottom-level acceleration structure
};

static_assert(sizeof( GeometryInstance ) == 64);

//struct PathState
//{
//	float3 Normal; // normal at last path vertex
//	uint data;// holds pixel index << 8, path length in bits 0..3, flags in bits 4..7
//	float3 Direction; // ray direction of the current path segment
//	int N; // Encoded normal in w
//	float3 Throughput; // path transport
//	float bsdfPdf; // postponed pdf
//
//};
//struct PathState4 { float4 O4, D4, T4; };

// PotentialContribution: besides a shadow ray, a connection needs
// a pixel index and the energy that will be deposited to that pixel
// if there is no occlusion.
struct PotentialContribution
{
	float4 Origin;
	float4 Direction;
	float3 Emission;
	uint pixelIdx;
};
struct PotentialContribution4 { float4 O4, D4, E4; };

// counters and other global data, in device memory
struct Counters // 14 counters
{
	void Reset( uint4 LightCounts, uint scrwidth, uint scrheight, float ClampValue = 10.0f, float GeometryEpsilon = 0.0001f )
	{
		pathLength = 1;
		scrWidth = scrwidth;
		scrHeight = scrheight;
		pathCount = scrwidth * scrheight;
		generated = 0;
		extensionRays = 0;
		shadowRays = 0;
		probePixelIdx = 0;
		probedInstid = 0;
		probedTriid = 0;
		probedDist = 0;
		clampValue = ClampValue;
		geometryEpsilon = GeometryEpsilon;
		lightCounts = LightCounts;
	}

	uint pathLength;
	uint scrWidth;
	uint scrHeight;
	uint pathCount;
	uint generated;
	uint extensionRays;
	uint shadowRays;
	uint probePixelIdx;
	int probedInstid;
	int probedTriid;
	float probedDist;
	float clampValue;
	float geometryEpsilon;
	uint4 lightCounts;
};

// internal material representation
struct VulkanMaterial
{
	struct Map { short width, height; half uscale, vscale, uoffs, voffs; uint addr; };
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b; uint flags;
	uint4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
	// texture / normal map descriptors; exactly 128-bit each
	Map tex0, tex1, nmap0, nmap1, smap, rmap;
};

struct VulkanMaterial4
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

#include "rendercore.h"

// clang-format on
using namespace lh2core;

#include "../RenderSystem/common_bluenoise.h"

// EOF