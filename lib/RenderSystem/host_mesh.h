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

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  HostSkin                                                                   |
//  |  Skin data storage.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
class HostSkin
{
public:
	HostSkin( const tinygltfSkin& gltfSkin, const tinygltfModel& gltfModel, const int nodeBase );
	void ConvertFromGLTFSkin( const tinygltfSkin& gltfSkin, const tinygltfModel& gltfModel, const int nodeBase );
	string name;
	int skeletonRoot = 0;
	vector<mat4> inverseBindMatrices;
	vector<mat4> jointMat;
	vector<int> joints; // node indices of the joints
};

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
	HostMesh( const int triCount );
	HostMesh( const char* name, const char* dir, const float scale = 1.0f );
	HostMesh( const tinygltfMesh& gltfMesh, const tinygltfModel& gltfModel, const int matIdxOffset, const int materialOverride = -1 );
	~HostMesh();
	// methods
	void LoadGeometry( const char* file, const char* dir, const float scale = 1.0f );
	void LoadGeometryFromOBJ( const string& fileName, const char* directory, const mat4& transform );
	void ConvertFromGTLFMesh( const tinygltfMesh& gltfMesh, const tinygltfModel& gltfModel, const int matIdxOffset, const int materialOverride );
	void BuildFromIndexedData( const vector<int>& tmpIndices, const vector<float3>& tmpVertices,
		const vector<float3>& tmpNormals, const vector<float2>& tmpUvs, const vector<Pose>& tmpPoses,
		const vector<uint4>& tmpJoints, const vector<float4>& tmpWeights, const int materialIdx );
	void BuildMaterialList();
	void UpdateAlphaFlags();
	void SetPose( const vector<float>& weights );
	void SetPose( const HostSkin* skin );
	// data members
	string name = "unnamed";					// name for the mesh						
	int ID = -1;								// unique ID for the mesh: position in mesh array
	vector<float4> vertices;					// model vertices
	vector<float3> vertexNormals;				// vertex normals
	vector<float4> original;					// skinning: base pose; will be transformed into vector vertices
	vector<float3> origNormal;					// skinning: base pose normals
	vector<HostTri> triangles;					// full triangles
	vector<int> materialList;					// list of materials used by the mesh; used to efficiently track light changes
	vector<uint> alphaFlags;					// list containing 1 for each triangle that is flagged as HASALPHA, 0 otherwise 
	vector<uint4> joints;						// skinning: joints
	vector<float4> weights;						// skinning: joint weights
	vector<Pose> poses;							// morph target data
	bool isAnimated;							// true when this mesh has animation data
	bool excludeFromNavmesh = false;			// prevents mesh from influencing navmesh generation (e.g. curtains)
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
	// Note: design decision:
	// Vertices and indices can be deduced from the list of HostTris, obviously. However, efficient intersection
	// (e.g. in OptiX) requires only vertices and connectivity data. Shading on the other hand requires the full
	// HostTris. The cores will thus benefit from having both structures. Now, we could let the core build the
	// vertex and index lists. However, building these efficiently is non-trivial, therefore the 'smart' split 
	// logic stays in the RenderSystem.
};

} // namespace lighthouse2

// EOF