/* host_mesh.h - Copyright 2019 Utrecht University

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

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  HostMesh                                                                   |
//  |  Host-side mesh data storage.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
class HostMesh
{
public:
	struct Pose
	{
		vector<float3> positions;
		vector<float3> normals;
		vector<float3> tangents;
	};
	// constructor / destructor
	HostMesh() = default;
	HostMesh( const char* name, const char* dir, const float scale = 1.0f );
	~HostMesh();
	// methods
	void LoadGeometry( const char* file, const char* dir, const float scale = 1.0f );
	void LoadGeometryFromOBJ( const string& fileName, const char* directory, const mat4& transform );
	void BuildMaterialList();
	void UpdateAlphaFlags();
	// data members
	string name = "unnamed";					// name for the mesh						
	int ID = -1;								// unique ID for the mesh: position in mesh array
	vector<float4> vertices;					// model vertices
	vector<uint3> indices;						// connectivity data
	vector<HostTri> triangles;					// full triangles
	vector<int> materialList;					// list of materials used by the mesh; used to efficiently track light changes
	vector<uint> alphaFlags;					// list containing 1 for each triangle that is flagged as HASALPHA, 0 otherwise 
	vector<Pose*> poses;						// morph target data
	bool isAnimated;							// true when this mesh has animation data
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
#ifdef RENDERSYSTEMBUILD
	// this is ugly, but otherwise apps that include host_mesh.h need to know what tinygltf is.
	friend class HostScene;
protected:
	HostMesh( tinygltf::Mesh& gltfMesh, tinygltf::Model& gltfModel, const int matIdxOffset );
	void ConvertFromGTLFMesh( tinygltf::Mesh& gltfMesh, tinygltf::Model& gltfModel, const int matIdxOffset );
#endif
	// Note: design decision:
	// Vertices and indices can be deduced from the list of HostTris, obviously. However, efficient intersection
	// (e.g. in OptiX) requires only vertices and connectivity data. Shading on the other hand requires the full
	// HostTris. The cores will thus benefit from having both structures. Now, we could let the core build the
	// vertex and index lists. However, building these efficiently is non-trivial, therefore the 'smart' split 
	// logic stays in the RenderSystem.
};

} // namespace lighthouse2

// EOF