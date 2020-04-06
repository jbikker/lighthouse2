/* rendercore.h - Copyright 2019/2020 Utrecht University

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
//  |  Mesh                                                                       |
//  |  Minimalistic mesh storage.                                           LH2'19|
//  +-----------------------------------------------------------------------------+
class Mesh
{
public:
	float4* vertices = 0;							// vertex data received via SetGeometry
	int vcount = 0;									// vertex count
	CoreTri* triangles = 0;							// 'fat' triangle data
};

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore : public CoreAPI_Base
{
public:
	// methods
	void Init();
	void SetTarget( GLTexture* target, const uint spp );
	void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles );
	void Render( const ViewPyramid& view, const Convergence converge, bool async );
	void WaitForRender() { /* this core does not support asynchronous rendering yet */ }
	CoreStats GetCoreStats() const override;
	void Shutdown();

	// unimplemented for the minimal core
	inline void SetProbePos( const int2 pos ) override {}
	inline void Setting( const char* name, float value ) override {}
	inline void SetTextures( const CoreTexDesc* tex, const int textureCount ) override {}
	inline void SetMaterials( CoreMaterial* mat, const int materialCount ) override {}
	inline void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount ) override
	{
	}
	inline void SetSkyData( const float3* pixels, const uint width, const uint height, const mat4& worldToLight ) override {}
	inline void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform ) override {}
	inline void FinalizeInstances() override {}

	// internal methods
private:

	// data members
	Bitmap* screen = 0;								// temporary storage of RenderCore output; will be copied to render target
	int targetTextureID = 0;						// ID of the target OpenGL texture
	vector<Mesh> meshes;							// mesh data storage
public:
	CoreStats coreStats;							// rendering statistics
};

} // namespace lh2core

// EOF