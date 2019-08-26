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
//  |  Host-side model data storage.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
class HostMesh
{
public:
	// constructor / destructor
	HostMesh() = default;
	HostMesh( const char* name, const char* dir, const float scale = 1.0f );
	~HostMesh();
	// methods
	void LoadGeometry( const char* file, const char* dir, const float scale = 1.0f );
	void LoadGeometryFromOBJ( const string& fileName, const char* directory, const mat4& transform );
	void LoadGeometryFromGLTF( const string& fileName, const mat4& transform );
	void LoadGeometryFromFBX( const string& fileName, const char* directory, const mat4& transform );
	vector<HostMesh*> FBXImport( const string& fileName );
	void BuildMaterialList();
	void UpdateAlphaFlags();
	// data members
	string name = "unnamed";					// name for the mesh						
	int ID = -1;								// unique ID for the model: position in model array
	vector<float4> vertices;					// model vertices
	vector<uint3> indices;						// connectivity data
	vector<HostTri> triangles;					// full triangles
	vector<int> materialList;					// list of materials used by the mesh; used to efficiently track light changes
	vector<uint> alphaFlags;					// list containing 1 for each triangle that is flagged as HASALPHA, 0 otherwise 
	bool isAnimated;							// true when this mesh has animation data
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
	// Note: design decision:
	// Vertices and indices can be deduced from the list of HostTris, obviously. However, efficient intersection
	// (e.g. in OptiX) requires only vertices and connectivity data. Shading on the other hand requires the full
	// HostTris. The cores will thus benefit from having both structures. Now, we could let the core build the
	// vertex and index lists. However, building these efficiently is non-trivial, therefore the 'smart' split 
	// logic stays in the RenderSystem.
};

//  +-----------------------------------------------------------------------------+
//  |  HostInstance                                                               |
//  |  Host-side instance definition.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
class HostInstance
{
public:
	// constructor / destructor
	HostInstance() = default;
	HostInstance( int meshID, mat4 T = mat4() ) : meshIdx( meshID ), transform( T ) {}
	~HostInstance();
	// data members
	int ID = -1;								// unique ID for the instance: position in instance array
	int meshIdx = -1;							// id of the mesh this instance refers to
	mat4 transform = mat4();					// transform for this instance
	bool hasLTris = false;						// true if this instance uses an emissive material
	TRACKCHANGES;
};

} // namespace lighthouse2

// EOF