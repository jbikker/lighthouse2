/* core_api_base.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file contains the declaration of the base class for core APIs,
   which defines the interface between the RenderSystem and each of the
   cores.
*/

#pragma once

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  CoreStats                                                                  |
//  |  Container for various statistics, filled by the core. Obtain a const ref   |
//  |  to this data by calling CoreAPI::GetCoreStats().                     LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreStats
{
	// device
	char* deviceName = 0;				// device name; TODO: will leak
	uint SMcount = 0;					// number of shading multiprocessors on device
	uint ccMajor = 0, ccMinor = 0;		// compute capability
	uint VRAM = 0;						// device memory, in MB
	// storage
	uint argb32TexelCount = 0;			// number of uint texels
	uint argb128TexelCount = 0;			// number of float4 texels
	uint nrm32TexelCount = 0;			// number of normal map texels
	// bvh
	float bvhBuildTime = 0;				// overall accstruc build time
	// rendering
	uint totalRays = 0;					// total number of rays cast
	uint totalExtensionRays = 0;		// total extension rays cast
	uint totalShadowRays = 0;			// total shadow rays cast
	float renderTime;					// overall render time
	uint primaryRayCount;				// # primary rays
	float traceTime0;					// time spent tracing primary rays
	uint bounce1RayCount;				// # rays after first bounce
	float traceTime1;					// time spent tracing first bounce
	uint deepRayCount;					// # rays after multiple bounces
	float traceTimeX;					// time spent tracing subsequent bounces
	float shadowTraceTime;				// time spent tracing shadow rays
	float shadeTime;					// time spent in shading code
	float filterTime = 0;				// time spent in filter code
	// probe
	int probedInstid;					// id of the instance at probe position
	int probedTriid;					// id of triangle at probe position
	float probedDist;					// distance of triangle at probe position
};

//  +-----------------------------------------------------------------------------+
//  |  CoreStats                                                                  |
//  |  Container for various statistics, filled by the render system. Obtain a    |
//  |  const ref to this data by calling CoreAPI::GetSystemStats().         LH2'19|
//  +-----------------------------------------------------------------------------+
struct SystemStats
{
	// scene
	float sceneUpdateTime = 0;			// time spent updating the scene graph
};

//  +-----------------------------------------------------------------------------+
//  |  CoreAPI_Base                                                               |
//  |  Interface between the RenderCore and the RenderSystem.               LH2'19|
//  +-----------------------------------------------------------------------------+
class CoreAPI_Base
{
public:
	// CreateCoreAPI: instantiate and initialize a RenderCore object and obtain an interface to it.
	static CoreAPI_Base* CreateCoreAPI( const char* dllName );
	// GetCoreStats: obtain a const ref to the CoreStats object, which provides statistics on the rendering process.
	virtual CoreStats GetCoreStats() = 0;
	// Init: initialize the core
	virtual void Init() = 0;
	// SetProbePos: set a pixel for which the triangle and instance id will be captured, e.g. for object picking.
	virtual void SetProbePos( const int2 pos ) = 0;
	// SetTarget: specify an OpenGL texture as a render target for the path tracer.
	virtual void SetTarget( GLTexture* target, const uint spp ) = 0;
	// Setting: modify a render setting
	virtual void Setting( const char* name, float value ) = 0;
	// Render: produce one frame. Convergence can be 'Converge' or 'Restart'.
	virtual void Render( const ViewPyramid& view, const Convergence converge ) = 0;
	// Shutdown: destroy the RenderCore and free all resources.
	virtual void Shutdown() = 0;
	// SetTextures: update the texture data in the RenderCore using the supplied data.
	virtual void SetTextures( const CoreTexDesc* tex, const int textureCount ) = 0;
	// SetMaterials: update the material list used by the RenderCore. Textures referenced by the materials must be set in advance.
	virtual void SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount ) = 0;
	// SetLights: update the point lights, spot lights and directional lights.
	virtual void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount ) = 0;
	// SetSkyData: specify the data required for sky dome rendering.
	virtual void SetSkyData( const float3* pixels, const uint width, const uint height ) = 0;
	// SetGeometry: update the geometry for a single mesh.
	virtual void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 ) = 0;
	// SetInstance: update the data on a single instance.
	virtual void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform = mat4::Identity() ) = 0;
	// UpdateTopLevel: trigger a top-level BVH update.
	virtual void UpdateToplevel() = 0;
};

} // namespace lighthouse2

// EOF