/* host_node.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

// helper function
static HostTri TransformedHostTri( HostTri* tri, mat4 T )
{
	HostTri transformedTri = *tri;
	transformedTri.vertex0 = make_float3( make_float4( transformedTri.vertex0, 1 ) * T );
	transformedTri.vertex1 = make_float3( make_float4( transformedTri.vertex1, 1 ) * T );
	transformedTri.vertex2 = make_float3( make_float4( transformedTri.vertex2, 1 ) * T );
	const float4 N = normalize( make_float4( transformedTri.Nx, transformedTri.Ny, transformedTri.Nz, 0 ) * T );
	transformedTri.Nx = N.x;
	transformedTri.Ny = N.y;
	transformedTri.Nz = N.z;
	return transformedTri;
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::HostNode                                                         |
//  |  Constructors.                                                        LH2'19|
//  +-----------------------------------------------------------------------------+
HostNode::HostNode( const tinygltfNode& gltfNode, const int nodeBase, const int meshBase, const int skinBase )
{
	ConvertFromGLTFNode( gltfNode, nodeBase, meshBase, skinBase );
}

HostNode::HostNode( const int meshIdx, const mat4& transform )
{
	// setup a node based on a mesh index and a transform
	meshID = meshIdx;
	localTransform = transform;
	// process light emitting surfaces
	PrepareLights();
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::~HostNode                                                        |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
HostNode::~HostNode()
{
	if ((meshID > -1) && hasLTris)
	{
		// this node is an instance and has emissive materials;
		// remove the relevant area lights.
		HostMesh* mesh = HostScene::meshPool[meshID];
		for (auto materialIdx : mesh->materialList)
		{
			HostMaterial* material = HostScene::materials[materialIdx];
			if (material->IsEmissive())
			{
				// mesh contains an emissive material; remove related area lights
				vector<HostAreaLight*>& lightList = HostScene::areaLights;
				for (int s = (int)lightList.size(), i = 0; i < s; i++)
					if (lightList[i]->instIdx == ID) lightList.erase( lightList.begin() + i-- );
			}
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::ConvertFromGLTFNode                                              |
//  |  Create a node from a GLTF node.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostNode::ConvertFromGLTFNode( const tinygltfNode& gltfNode, const int nodeBase, const int meshBase, const int skinBase )
{
	// copy node name
	name = gltfNode.name;
	// set mesh / skin ID
	meshID = gltfNode.mesh == -1 ? -1 : (gltfNode.mesh + meshBase);
	skinID = gltfNode.skin == -1 ? -1 : (gltfNode.skin + skinBase);
	// if the mesh has morph targets, the node should have weights for them
	if (meshID != -1)
	{
		const int morphTargets = (int)HostScene::meshPool[meshID]->poses.size() - 1;
		if (morphTargets > 0) weights.resize( morphTargets, 0.0f );
	}
	// copy child node indices
	for (int s = (int)gltfNode.children.size(), i = 0; i < s; i++) childIdx.push_back( gltfNode.children[i] + nodeBase );
	// obtain matrix
	bool buildFromTRS = false;
	if (gltfNode.matrix.size() == 16)
	{
		// we get a full matrix
		for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) matrix.cell[i * 4 + j] = gltfNode.matrix[j * 4 + i];
		buildFromTRS = true;
	}
	if (gltfNode.translation.size() == 3)
	{
		// the GLTF node contains a translation
		translation = make_float3( gltfNode.translation[0], gltfNode.translation[1], gltfNode.translation[2] );
		buildFromTRS = true;
	}
	if (gltfNode.rotation.size() == 4)
	{
		// the GLTF node contains a rotation
		rotation = quat( gltfNode.rotation[3], gltfNode.rotation[0], gltfNode.rotation[1], gltfNode.rotation[2] );
		buildFromTRS = true;
	}
	if (gltfNode.scale.size() == 3)
	{
		// the GLTF node contains a scale
		scale = make_float3( gltfNode.scale[0], gltfNode.scale[1], gltfNode.scale[2] );
		buildFromTRS = true;
	}
	// if we got T, R and/or S, reconstruct final matrix
	if (buildFromTRS) UpdateTransformFromTRS();
	// process light emitting surfaces
	PrepareLights();
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::UpdateTransformFromTRS                                           |
//  |  Process T, R, S data to localTransform.                              LH2'19|
//  +-----------------------------------------------------------------------------+
void HostNode::UpdateTransformFromTRS()
{
	mat4 T = mat4::Translate( translation );
	mat4 R = rotation.toMatrix();
	mat4 S = mat4::Scale( scale );
	localTransform = T * R * S * matrix;
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::Update                                                           |
//  |  Calculates the combined transform for this node and recurses into the      |
//  |  child nodes. If a change is detected, the light triangles are updated      |
//  |  as well.                                                             LH2'19|
//  +-----------------------------------------------------------------------------+
bool HostNode::Update( mat4& T, vector<int>& instances, int& posInInstanceArray )
{
	// update the combined transform for this node
	bool thisWasModified = Changed();
	bool instancesChanged = thisWasModified;
	treeChanged = thisWasModified;
	if (transformed)
	{
		UpdateTransformFromTRS();
		transformed = false;
	}
	combinedTransform = T * localTransform;
	// update the combined transforms of the children
	for (int s = (int)childIdx.size(), i = 0; i < s; i++)
	{
		HostNode* child = HostScene::nodePool[childIdx[i]];
		bool childChanged = child->Update( combinedTransform, instances, posInInstanceArray );
		instancesChanged |= childChanged;
		treeChanged |= childChanged;
	}
	// update animations
	if (meshID > -1)
	{
		if (morphed)
		{
			HostScene::meshPool[meshID]->SetPose( weights );
			morphed = false;
		}
		if (thisWasModified && hasLTris) UpdateLights();
		if (instanceID != posInInstanceArray)
		{
			instancesChanged = true;
			if (posInInstanceArray < instances.size())
				instances[posInInstanceArray] = ID;
			else
				instances.push_back( ID );
		}
		if (skinID > -1)
		{
			HostSkin* skin = HostScene::skins[skinID];
			mat4 meshTransform = combinedTransform;
			mat4 meshTransformInverted = meshTransform.Inverted();
			for (int s = (int)skin->joints.size(), j = 0; j < s; j++)
			{
				HostNode* jointNode = HostScene::nodePool[skin->joints[j]];
				skin->jointMat[j] = meshTransformInverted * jointNode->combinedTransform * skin->inverseBindMatrices[j];
			}
			HostScene::meshPool[meshID]->SetPose( skin );
		}
		posInInstanceArray++;
	}
	// all done.
	return instancesChanged;
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::PrepareLights                                                    |
//  |  Detects emissive triangles and creates light triangles for them.     LH2'19|
//  +-----------------------------------------------------------------------------+
void HostNode::PrepareLights()
{
	if (meshID > -1)
	{
		HostMesh* mesh = HostScene::meshPool[meshID];
		for (int s = (int)mesh->triangles.size(), i = 0; i < s; i++)
		{
			HostTri* tri = &mesh->triangles[i];
			HostMaterial* mat = HostScene::materials[tri->material];
			if (mat->IsEmissive())
			{
				tri->UpdateArea();
				HostTri transformedTri = TransformedHostTri( tri, localTransform );
				HostAreaLight* light = new HostAreaLight( &transformedTri, i, ID );
				tri->ltriIdx = (int)HostScene::areaLights.size(); // TODO: can't duplicate a light due to this.
				HostScene::areaLights.push_back( light );
				hasLTris = true;
				// Note: TODO: 
				// 1. if a mesh is deleted it should scan the list of area lights
				//    to delete those that no longer exist.
				// 2. if a material is changed from emissive to non-emissive,
				//    meshes using the material should remove their light emitting
				//    triangles from the list of area lights.
				// 3. if a material is changed from non-emissive to emissive,
				//    meshes using the material should update the area lights list.
				// Item 1 can be done efficiently. Items 2 and 3 require a list
				// of materials per mesh to be efficient.
			}
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostNode::UpdateLights                                                     |
//  |  Update light triangles belonging to this instance after the tansform for   |
//  |  the node changed.                                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostNode::UpdateLights()
{
	if (!hasLTris) return;
	HostMesh* mesh = HostScene::meshPool[meshID];
	for (int s = (int)mesh->triangles.size(), i = 0; i < s; i++)
	{
		HostTri* tri = &mesh->triangles[i];
		if (tri->ltriIdx == -1) continue;
		tri->UpdateArea();
		HostTri transformedTri = TransformedHostTri( tri, combinedTransform );
		*HostScene::areaLights[tri->ltriIdx] = HostAreaLight( &transformedTri, i, ID );
	}
}

// EOF