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

namespace lighthouse2 {

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
	~HostNode();
	// methods
	bool Update( mat4& T, int& instanceIdx );	// recursively update the transform of this node and its children
	void PrepareLights();				// detects emissive triangles and creates light triangles for them
	void UpdateLights();				// when the transform changes, this fixes the light triangles
	// data members
	string name;						// node name as specified in the GLTF file
	mat4 combinedTransform;				// transform combined with ancestor transforms
	mat4 localTransform;				// = T * R * S, in case of animation.
	float3 translation = make_float3( 0 );
	quat rotation;
	float3 scale = make_float3( 1 );
	int ID = -1;						// unique ID for the node: position in node array
	int instanceID = -1;				// for mesh nodes: location in the instance array
	int meshID = -1;					// id of the mesh this node refers to (if any)
	int skinID = -1;					// TODO
	bool hasLTris = false;				// true if this instance uses an emissive material
	vector<int> childIdx;				// child nodes of this node
#ifdef RENDERSYSTEMBUILD
	// this is ugly, but otherwise apps that include host_scene.h need to know what tinygltf is.
	friend class HostScene;
protected:
	HostNode( tinygltf::Node& gltfNode );
	void ConvertFromGLTFNode( tinygltf::Node& gltfNode );
#endif
	TRACKCHANGES;
};

} // namespace lighthouse2

// EOF