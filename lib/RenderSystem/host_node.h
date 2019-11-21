/* host_node.h - Copyright 2019 Utrecht University

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

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  HostNode                                                                   |
//  |  Simple node for construction of a scene graph for the scene.               |
//  |  Designed to mirror the properties stored in a GLTF file.             LH2'19|
//  +-----------------------------------------------------------------------------+
class HostNode
{
public:
	// constructor / destructor
	HostNode() = default;
	HostNode( const int meshIdx, const mat4& transform );
	HostNode( const tinygltfNode& gltfNode, const int nodeBase, const int meshBase, const int skinBase );
	~HostNode();
	// methods
	void ConvertFromGLTFNode( const tinygltfNode& gltfNode, const int nodeBase, const int meshBase, const int skinBase );
	bool Update( mat4& T, vector<int>& instances, int& instanceIdx );	// recursively update the transform of this node and its children
	void UpdateTransformFromTRS();		// process T, R, S data to localTransform
	void PrepareLights();				// detects emissive triangles and creates light triangles for them
	void UpdateLights();				// when the transform changes, this fixes the light triangles
	// data members
	string name;						// node name as specified in the GLTF file
	mat4 combinedTransform;				// transform combined with ancestor transforms
	mat4 localTransform;				// = matrix * T * R * S, in case of animation.
	float3 translation = make_float3( 0 );
	quat rotation;
	float3 scale = make_float3( 1 );
	mat4 matrix;
	int ID = -1;						// unique ID for the node: position in node array
	int meshID = -1;					// id of the mesh this node refers to (if any, -1 otherwise)
	int skinID = -1;					// id of the skin this node refers to (if any, -1 otherwise)
	vector<float> weights;				// morph target weights
	bool hasLTris = false;				// true if this instance uses an emissive material
	bool morphed = false;				// node mesh should update pose
	bool transformed = false;			// local transform of node should be updated
	bool treeChanged = false;			// this node or one of its children got updated
	vector<int> childIdx;				// child nodes of this node
	TRACKCHANGES;
protected:
	friend class RenderSystem;
	int instanceID = -1;				// for mesh nodes: location in the instance array. For internal use only.
};

} // namespace lighthouse2

// EOF