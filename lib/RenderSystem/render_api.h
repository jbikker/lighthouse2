/* render_api.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Declaration of the RenderSystem API.
*/

#pragma once

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  RenderAPI                                                                  |
//  |  Interface between the RenderSystem and the application.              LH2'19|
//  +-----------------------------------------------------------------------------+
struct RenderSettings;
class RenderAPI
{
public:
	// CreateRenderAPI: instantiate and initialize a RenderSystem object and obtain an interface to it.
	static RenderAPI* CreateRenderAPI( const char* dllName );
	// Methods
	void SerializeMaterials( const char* xmlFile );
	void DeserializeMaterials( const char* xmlFile );
	void Shutdown();
	void DeserializeCamera( const char* camera );
	void SerializeCamera( const char* camera );
	int AddMesh( const char* file, const char* dir, const float scale = 1.0f, const bool flatShaded = false );
	int AddMesh( const char* file, const float scale = 1.0f, const bool flatShaded = false );
	int AddScene( const char* file, const char* dir, const mat4& transform = mat4::Identity() );
	int AddScene( const char* file, const mat4& transform = mat4::Identity() );
	int AddMesh( const int triCount );
	void AddTriToMesh( const int meshId, const float3& v0, const float3& v1, const float3& v2, const int matId );
	int AddQuad( const float3 N, const float3 pos, const float width, const float height, const int material, const int meshID = -1 );
	int AddInstance( const int meshId, const mat4& transform = mat4() );
	void RemoveNode( const int nodeId );
	void SetNodeTransform( const int nodeId, const mat4& transform );
	const mat4& GetNodeTransform( const int nodeId );
	void ResetAnimation( const int animId );
	void UpdateAnimation( const int animId, const float dt );
	int AnimationCount();
	void SynchronizeSceneData();
	void Render( Convergence converge, bool async = false );
	void WaitForRender();
	Camera* GetCamera();
	RenderSettings* GetSettings();
	void Setting( const char* name, const float value );
	int GetTriangleNode( const int coreInstId, const int coreTriId );
	int GetTriangleMesh( const int coreInstId, const int coreTriId );
	HostScene* GetScene();
	int GetTriangleMaterialID( const int coreInstId, const int coreTriId );
	HostMaterial* GetTriangleMaterial( const int coreInstId, const int coreTriId );
	HostMaterial* GetMaterial( const int matId );
	const vector<HostMaterial*>& GetMaterials();
	int FindNode( const char* name );
	int FindMaterialID( const char* name );
	int AddMaterial( const float3 color, const char* name = 0 );
	int AddPointLight( const float3 pos, const float3 radiance, bool enabled = true );
	int AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled = true );
	int AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled = true );
	void SetTarget( GLTexture* tex, const uint spp );
	void SetProbePos( const int2 pos );
	CoreStats GetCoreStats() const;
	SystemStats GetSystemStats();
};

} // namespace lighthouse2

// EOF