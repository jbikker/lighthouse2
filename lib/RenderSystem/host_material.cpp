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
		int diffuseTextureID = map[TEXTURE0].textureID = HostScene::FindOrCreateTexture( original.diffuse_texname, HostTexture::LINEARIZED | HostTexture::FLIPPED );
		color = make_float3( 1 ); // we have a texture now; default modulation to white
		if (HostScene::textures[diffuseTextureID]->flags & HASALPHA) flags |= HASALPHA;
	}
	if (original.normal_texname != "")
	{
		map[NORMALMAP0].textureID = HostScene::FindOrCreateTexture( original.normal_texname, HostTexture::FLIPPED );
		HostScene::textures[map[NORMALMAP0].textureID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	else if (original.bump_texname != "")
	{
		int bumpMapID = map[NORMALMAP0].textureID = HostScene::CreateTexture( original.bump_texname, HostTexture::FLIPPED ); // cannot reuse, height scale may differ
		float heightScaler = 1.0f;
		auto heightScalerIt = original.unknown_parameter.find( "bump_height" );
		if (heightScalerIt != original.unknown_parameter.end()) heightScaler = static_cast<float>(atof( (*heightScalerIt).second.c_str() ));
		HostScene::textures[bumpMapID]->BumpToNormalMap( heightScaler );
		HostScene::textures[bumpMapID]->flags |= HostTexture::NORMALMAP; // TODO: what if it's also used as regular texture?
	}
	if (original.specular_texname != "")
	{
		map[ROUGHNESS0].textureID = HostScene::FindOrCreateTexture( original.specular_texname.c_str(), HostTexture::FLIPPED );
		roughness = 1.0f;
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
			if (item.first == "index") map[TEXTURE0].textureID = (int)item.second + textureBase;
		}
		// TODO: do a better automatic conversion.
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertTo                                                    |
//  |  Converts a host material representation to the CoreMaterial representation |
//  |  suitable for rendering. This code is here so the RenderCore never needs to |
//  |  know about HostMaterials.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
#define TOCHAR(a) ((uint)((a)*255.0f))
#define TOUINT4(a,b,c,d) (TOCHAR(a)+(TOCHAR(b)<<8)+(TOCHAR(c)<<16)+(TOCHAR(d)<<24))
void HostMaterial::ConvertTo( CoreMaterial& gpuMat, CoreMaterialEx& gpuMatEx ) const
{
	// base properties
	memset( &gpuMat, 0, sizeof( CoreMaterial ) );
	gpuMat.diffuse_r = color.x;
	gpuMat.diffuse_g = color.y;
	gpuMat.diffuse_b = color.z;
	gpuMat.transmittance_r = 1 - absorption.x;
	gpuMat.transmittance_g = 1 - absorption.y;
	gpuMat.transmittance_b = 1 - absorption.z;
	gpuMat.parameters.x = TOUINT4( metallic, subsurface, specular, roughness );
	gpuMat.parameters.y = TOUINT4( specularTint, anisotropic, sheen, sheenTint );
	gpuMat.parameters.z = TOUINT4( clearcoat, clearcoatGloss, transmission, 0 );
	gpuMat.parameters.w = *((uint*)&eta);
	const HostTexture* t0 = map[TEXTURE0].textureID == -1 ? 0 : HostScene::textures[map[TEXTURE0].textureID];
	const HostTexture* t1 = map[TEXTURE1].textureID == -1 ? 0 : HostScene::textures[map[TEXTURE1].textureID];
	const HostTexture* t2 = map[TEXTURE2].textureID == -1 ? 0 : HostScene::textures[map[TEXTURE2].textureID];
	const HostTexture* nm0 = map[NORMALMAP0].textureID == -1 ? 0 : HostScene::textures[map[NORMALMAP0].textureID];
	const HostTexture* nm1 = map[NORMALMAP1].textureID == -1 ? 0 : HostScene::textures[map[NORMALMAP1].textureID];
	const HostTexture* nm2 = map[NORMALMAP2].textureID == -1 ? 0 : HostScene::textures[map[NORMALMAP2].textureID];
	const HostTexture* r = map[ROUGHNESS0].textureID == -1 ? 0 : HostScene::textures[map[ROUGHNESS0].textureID];
	const HostTexture* s = map[SPECULARITY].textureID == -1 ? 0 : HostScene::textures[map[SPECULARITY].textureID];
	const HostTexture* cm = map[COLORMASK].textureID == -1 ? 0 : HostScene::textures[map[COLORMASK].textureID];
	const HostTexture* am = map[ALPHAMASK].textureID == -1 ? 0 : HostScene::textures[map[ALPHAMASK].textureID];
	bool hdr = false;
	if (t0) if (t0->flags & HostTexture::HDR) hdr = true;
	gpuMat.flags =
		(eta < 1 ? (1 << 0) : 0) +							// is dielectric
		(hdr ? (1 << 1) : 0) +								// diffuse map is hdr
		(t0 ? (1 << 2) : 0) +								// has diffuse map
		(nm0 ? (1 << 3) : 0) +								// has normal map
		(s ? (1 << 4) : 0) +								// has specularity map
		(r ? (1 << 5) : 0) +								// has roughness map
		((flags & ANISOTROPIC) ? (1 << 6) : 0) +			// is anisotropic
		(nm1 ? (1 << 7) : 0) +								// has 2nd normal map
		(nm2 ? (1 << 8) : 0) +								// has 3rd normal map
		(t1 ? (1 << 9) : 0) +								// has 2nd diffuse map
		(t2 ? (1 << 10) : 0) +								// has 3rd diffuse map
		((flags & SMOOTH) ? (1 << 11) : 0) +				// has smooth normals
		((flags & HASALPHA) ? (1 << 12) : 0);				// has alpha
	// copy maps array to CoreMaterialEx instance
	for (int i = 0; i < 11; i++) gpuMatEx.texture[i] = map[i].textureID;
	// maps
	if (t0) // texture layer 0
		gpuMat.texwidth0 = t0->width, gpuMat.texheight0 = t0->height,
		gpuMat.uoffs0 = map[TEXTURE0].uvoffset.x, gpuMat.voffs0 = map[TEXTURE0].uvoffset.y,
		gpuMat.uscale0 = map[TEXTURE0].uvscale.x, gpuMat.vscale0 = (half)map[TEXTURE0].uvscale.y;
	if (t1) // texture layer 1
		gpuMat.texwidth1 = t1->width, gpuMat.texheight1 = t1->height,
		gpuMat.uoffs1 = map[TEXTURE1].uvoffset.x, gpuMat.voffs1 = map[TEXTURE1].uvoffset.y,
		gpuMat.uscale1 = map[TEXTURE1].uvscale.x, gpuMat.vscale1 = (half)map[TEXTURE1].uvscale.y;
	if (t2) // texture layer 2
		gpuMat.texwidth2 = t2->width, gpuMat.texheight2 = t2->height,
		gpuMat.uoffs2 = map[TEXTURE2].uvoffset.x, gpuMat.voffs2 = map[TEXTURE2].uvoffset.y,
		gpuMat.uscale2 = map[TEXTURE2].uvscale.x, gpuMat.vscale2 = (half)map[TEXTURE2].uvscale.y;
	if (nm0) // normal map layer 0
		gpuMat.nmapwidth0 = nm0->width, gpuMat.nmapheight0 = nm0->height,
		gpuMat.nuoffs0 = map[NORMALMAP0].uvoffset.x, gpuMat.nvoffs0 = map[NORMALMAP0].uvoffset.y,
		gpuMat.nuscale0 = map[NORMALMAP0].uvscale.x, gpuMat.nvscale0 = map[NORMALMAP0].uvscale.y;
	if (nm1) // normal map layer 1
		gpuMat.nmapwidth1 = nm1->width, gpuMat.nmapheight1 = nm1->height,
		gpuMat.nuoffs1 = map[NORMALMAP1].uvoffset.x, gpuMat.nvoffs1 = map[NORMALMAP1].uvoffset.y,
		gpuMat.nuscale1 = map[NORMALMAP1].uvscale.x, gpuMat.nvscale1 = map[NORMALMAP1].uvscale.y;
	if (nm2) // normal map layer 2
		gpuMat.nmapwidth2 = nm2->width, gpuMat.nmapheight2 = nm2->height,
		gpuMat.nuoffs2 = map[NORMALMAP2].uvoffset.x, gpuMat.nvoffs2 = map[NORMALMAP2].uvoffset.y,
		gpuMat.nuscale2 = map[NORMALMAP2].uvscale.x, gpuMat.nvscale2 = map[NORMALMAP2].uvscale.y;
	if (r) // roughness map
		gpuMat.rmapwidth = r->width, gpuMat.rmapheight = r->height,
		gpuMat.ruoffs = map[ROUGHNESS0].uvoffset.x, gpuMat.rvoffs = map[ROUGHNESS0].uvoffset.y,
		gpuMat.ruscale = map[ROUGHNESS0].uvscale.x, gpuMat.rvscale = map[ROUGHNESS0].uvscale.y;
	if (s) // specularity map
		gpuMat.smapwidth = s->width, gpuMat.smapheight = s->height,
		gpuMat.suoffs = map[SPECULARITY].uvoffset.x, gpuMat.svoffs = map[SPECULARITY].uvoffset.y,
		gpuMat.suscale = map[SPECULARITY].uvscale.x, gpuMat.svscale = map[SPECULARITY].uvscale.y;
}

// EOF