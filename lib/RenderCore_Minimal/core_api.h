/* core_api.h - Copyright 2019 Utrecht University

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

#pragma once

namespace lh2core
{

//  +-----------------------------------------------------------------------------+
//  |  CoreAPI                                                                    |
//  |  Interface between the RenderCore and the RenderSystem.               LH2'19|
//  +-----------------------------------------------------------------------------+
class CoreAPI : public CoreAPI_Base
{
public:
	// Init: initialize the core
	void Init();
	// GetCoreStats_: obtain a const ref to the CoreStats object, which provides statistics on the rendering process.
	CoreStats GetCoreStats();
	// SetProbePos: set a pixel for which the triangle and instance id will be captured, e.g. for object picking.
	void SetProbePos( const int2 pos ) { /* not implemented for the minimal core. */ }
	// SetTarget: specify an OpenGL texture as a render target for the path tracer.
	void SetTarget( GLTexture* target, const uint spp );
	// Setting: modify a render setting
	void Setting( const char* name, float value ) { /* the minimal core ignores all settings. */ }
	// Render: produce one frame. Convergence can be 'Converge' or 'Restart'.
	void Render( const ViewPyramid& view, const Convergence converge );
	// Shutdown: destroy the RenderCore and free all resources.
	void Shutdown();
	// SetTextures: update the texture data in the RenderCore using the supplied data.
	void SetTextures( const CoreTexDesc* tex, const int textureCount );
	// SetMaterials: update the material list used by the RenderCore. Textures referenced by the materials must be set in advance.
	void SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount );
	// SetLights: update the point lights, spot lights and directional lights.
	void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount ) { /* ignore lights for now. */ }
	// SetSkyData: specify the data required for sky dome rendering.
	void SetSkyData( const float3* pixels, const uint width, const uint height ) { /* ignore sky data for the minimal core. */ }
	// SetGeometry: update the geometry for a single mesh.
	void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 );
	// SetInstance: update the data on a single instance.
	void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform = mat4::Identity() );
	// UpdateTopLevel: trigger a top-level BVH update.
	void UpdateToplevel() { /* not implemented for the minimal core. */ }
};

} // namespace lh2core

extern "C" COREDLL_API CoreAPI_Base* CreateCore();
extern "C" COREDLL_API void DestroyCore();

// EOF