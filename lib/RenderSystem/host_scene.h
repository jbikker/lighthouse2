/* host_scene.h - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  HostScene                                                                  |
//  |  Module for scene I/O and host-side management.                             |
//  |  This is a pure static class; we will not have more than one scene.   LH2'19|
//  +-----------------------------------------------------------------------------+
class HostScene
{
public:
	// constructor / destructor
	HostScene();
	~HostScene();
	// serialization / deserialization
	static void SerializeMaterials( const char* xmlFile );
	static void DeserializeMaterials( const char* xmlFile );
	// methods
	static void Init();
	static int FindOrCreateTexture( const string& origin, const uint modFlags = 0 );
	static int CreateTexture( const string& origin, const uint modFlags = 0 );
	static int FindOrCreateMaterial( const string& name );
	static int GetTriangleMaterial( const int instid, const int triid );
	static int FindMaterialID( const char* name );
	// scene construction / maintenance
	static int AddMesh( const char* objFile, const char* dataDir, const float scale = 1.0f );
	static int AddQuad( const float3 N, const float3 pos, const float width, const float height, const int material );
	static int AddInstance( const int meshId, const mat4& transform = mat4() );
	static void SetInstanceTransform( const int instId, const mat4& T );
	static int AddMaterial( const float3 color );
	static int AddPointLight( const float3 pos, const float3 radiance, bool enabled = true );
	static int AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled = true );
	static int AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled = true );
	// data members
	static HostSkyDome* sky;
	static vector<HostMesh*> meshes;
	static vector<HostInstance*> instances;
	static vector<HostMaterial*> materials;
	static vector<HostTexture*> textures;
	static vector<HostAreaLight*> areaLights;
	static vector<HostPointLight*> pointLights;
	static vector<HostSpotLight*> spotLights;
	static vector<HostDirectionalLight*> directionalLights;
	static Camera* camera;
};

} // namespace lighthouse2

// EOF