/* host_material.cpp - Copyright 2019 Utrecht University

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

// Lighthouse2 data model design principles:
// 
// - If a data member can be given an obvious name, access is public, saving on LOC and repetition.
// - (Almost) all data members are initialized to promote deterministic behavior in debug and release.
// - Textures, materials, meshes etc. have IDs; use of pointers is minimized.
// - Textures, materials, meshes etc. also have an optional name and origin string (not a char*).
// - A struct is either host-oriented or GPU-oriented; the name (as well as the file name) indicates 
//   this. If both are needed, both will be created, even if they are the same.
// - The RenderCore is the only system communicating with the GPU. It is thus also the only owner of
//   device-side data. The RenderSystem is responsible for setting this data before rendering starts.
//   Data in RenderCore reached its final destination; it can thus not be queried, and internal
//   operations are minimal.

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertFrom                                                  |
//  |  Converts a tinyobjloader material to a HostMaterial.                 LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertFrom( const tinyobjMaterial& original )
{
	// properties
	name = original.name;
	color = make_float3( original.diffuse[0], original.diffuse[1], original.diffuse[2] ); // Kd
	absorption = make_float3( original.transmittance[0], original.transmittance[1], original.transmittance[2] ); // Kt
	roughness = min( 1 - original.shininess, 1.0f );
	// maps
	if (original.diffuse_texname != "")
	{
		int diffuseTextureID = color.textureID = HostScene::FindOrCreateTexture( original.diffuse_texname, HostTexture::LINEARIZED | HostTexture::FLIPPED );
		color = make_float3( 1 ); // we have a texture now; default modulation to white
		if (HostScene::textures[diffuseTextureID]->flags & HASALPHA) flags |= HASALPHA;
	}
	if (original.normal_texname != "")
	{
		normals.textureID = HostScene::FindOrCreateTexture( original.normal_texname, HostTexture::FLIPPED );
		HostScene::textures[normals.textureID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	else if (original.bump_texname != "")
	{
		int bumpMapID = normals.textureID = HostScene::CreateTexture( original.bump_texname, HostTexture::FLIPPED ); // cannot reuse, height scale may differ
		float heightScaler = 1.0f;
		auto heightScalerIt = original.unknown_parameter.find( "bump_height" );
		if (heightScalerIt != original.unknown_parameter.end()) heightScaler = static_cast<float>(atof( (*heightScalerIt).second.c_str() ));
		HostScene::textures[bumpMapID]->BumpToNormalMap( heightScaler );
		HostScene::textures[bumpMapID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	if (original.specular_texname != "")
	{
		roughness.textureID = HostScene::FindOrCreateTexture( original.specular_texname.c_str(), HostTexture::FLIPPED );
		roughness() = 1.0f;
	}
	// finalize
	auto shadingIt = original.unknown_parameter.find( "shading" );
	if (shadingIt != original.unknown_parameter.end() && shadingIt->second == "flat") flags &= ~SMOOTH; else flags |= SMOOTH;
}

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertFrom                                                  |
//  |  Converts a tinygltf material to a HostMaterial.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertFrom( const tinygltfMaterial& original, const tinygltfModel& model, const int textureBase )
{
	name = original.name;
	for (const auto& value : original.values)
	{
		if (value.first == "baseColorFactor")
		{
			tinygltf::Parameter p = value.second;
			color = make_float3( p.number_array[0], p.number_array[1], p.number_array[2] );
		}
		if (value.first == "metallicFactor") if (value.second.has_number_value)
		{
			metallic = (float)value.second.number_value;
		}
		if (value.first == "roughnessFactor") if (value.second.has_number_value)
		{
			roughness = (float)value.second.number_value;
		}
		if (value.first == "baseColorTexture") for (auto& item : value.second.json_double_value)
		{
			if (item.first == "index") color.textureID = (int)item.second + textureBase;
		}
		// TODO: do a better automatic conversion.
	}
}

// EOF