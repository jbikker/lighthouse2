/* rendercore.h - Copyright 2019 Utrecht University

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
//  |  DeviceVars                                                                 |
//  |  Copy of device-side variables, to detect changes.                    LH2'19|
//  +-----------------------------------------------------------------------------+
struct DeviceVars
{
	// impossible values to trigger an update in the first frame
	float clampValue = -1.0f;
	float geometryEpsilon = 1e34f;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore
{
public:
	// methods
	void Init();
	void Render( const ViewPyramid& view, const Convergence converge );
	void Setting( const char* name, const float value );
	void SetTarget( GLTexture* target, const uint spp );
	void Shutdown();
	// passing data. Note: RenderCore always copies what it needs; the passed data thus remains the
	// property of the caller, and can be safely deleted or modified as soon as these calls return.
	void SetTextures( const CoreTexDesc* tex, const int textureCount );
	void SetMaterials( CoreMaterial* mat, const int materialCount ); // textures must be in sync when calling this
	void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount );
	void SetSkyData( const float3* pixels, const uint width, const uint height );
	// geometry and instances:
	// a scene is setup by first passing a number of meshes (geometry), then a number of instances.
	// note that stored meshes can be used zero, one or multiple times in the scene.
	// also note that, when using alpha flags, materials must be in sync.
	void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 );
	void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform );
	void UpdateToplevel() { /* nothing here for a rasterizer */ }
	void SetProbePos( const int2 pos );
	// internal methods
private:
	// data members
	int scrwidth = 0, scrheight = 0;				// current screen width and height
	Surface* renderTarget = 0;						// screen pixels
	int targetTextureID = 0;						// ID of the target OpenGL texture
	int skywidth = 0, skyheight = 0;				// size of the skydome texture
	int maxPixels = 0;								// max screen size buffers can accomodate without a realloc
	int2 probePos = make_int2( 0 );					// triangle picking; primary ray for this pixel copies its triid to coreStats.probedTriid
	int textureCount = 0;							// size of texture descriptor array
	Rasterizer rasterizer;							// rasterization functionality
	vector<Mesh*> meshes;							// list of meshes, for easy access in SetGeometry
public:
	CoreStats coreStats;							// rendering statistics
};

} // namespace lh2core

// EOF