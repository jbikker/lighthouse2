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
void HostMaterial::ConvertFrom( tinyobj::material_t& original )
{
	// properties
	name = original.name;
	color = make_float3( original.diffuse[0], original.diffuse[1], original.diffuse[2] ); // Kd
	absorption = make_float3( original.transmittance[0], original.transmittance[1], original.transmittance[2] ); // Kt
	roughness[0].x = min( original.shininess, 1.0f );
	roughness[1].x = (original.ambient[0] + original.ambient[1] + original.ambient[2]) / 3.0f; //Ka
	if (original.ior > 1.0f) // d & Ni
	{
		roughness[0].x = 0.0f;
		eta = original.ior;
	}
	else eta = 0;
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
		roughness[0].x = 1.0f;
	}
	// finalize
	auto shadingIt = original.unknown_parameter.find( "shading" );
	if (shadingIt != original.unknown_parameter.end() && shadingIt->second == "flat") flags &= ~SMOOTH; else flags |= SMOOTH;
}

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertFrom                                                  |
//  |  Converts a tinygltf material to a HostMaterial.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertFrom( tinygltf::Material& original, tinygltf::Model& model )
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
			metalness = (float)value.second.number_value;
		}
		if (value.first == "roughnessFactor") if (value.second.has_number_value)
		{
			roughness[0] = roughness[1] = make_float2( (float)value.second.number_value );
		}
		if (value.first == "baseColorTexture") for (auto& item : value.second.json_double_value)
		{
			if (item.first == "index") map[TEXTURE0].textureID = (int)item.second;
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostMaterial::ConvertTo                                                    |
//  |  Converts a host material representation to the CoreMaterial representation |
//  |  suitable for rendering. This code is here so the RenderCore never needs to |
//  |  know about HostMaterials.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
void HostMaterial::ConvertTo( CoreMaterial& gpuMat, CoreMaterialEx& gpuMatEx )
{
	// base properties
	memset( &gpuMat, 0, sizeof( CoreMaterial ) );
	gpuMat.diffuse_r = color.x, gpuMat.diffuse_g = color.y, gpuMat.diffuse_b = color.z;
	gpuMat.specularColor = specularColor;
	gpuMat.eta = (uchar)(255.0f * (eta - 1));		// assume range is [1..2], slider enforces this
	gpuMat.roughness0 = (uchar)(255.0f * min( 1.0f, sqr( roughness[0].x ) ));
	gpuMat.roughness1 = (uchar)(255.0f * min( 1.0f, sqr( roughness[1].x ) ));
	gpuMat.specularity = specularity;
	gpuMat.nscale0 = (uchar)roundf( copysignf( min( 127.0f, 10.0f * logf( 10000.0f * (0.0001f + fabsf( map[NORMALMAP0].valueScale )) ) ), map[NORMALMAP0].valueScale ) ) + 128;
	gpuMat.nscale1 = (uchar)roundf( copysignf( min( 127.0f, 10.0f * logf( 10000.0f * (0.0001f + fabsf( map[NORMALMAP1].valueScale )) ) ), map[NORMALMAP1].valueScale ) ) + 128;
	gpuMat.nscale2 = (uchar)roundf( copysignf( min( 127.0f, 10.0f * logf( 10000.0f * (0.0001f + fabsf( map[NORMALMAP2].valueScale )) ) ), map[NORMALMAP2].valueScale ) ) + 128;
	gpuMat.absorption = absorption;
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
		(eta > 0 ? (1 << 0) : 0) +							// is dielectric
		(hdr ? (1 << 1) : 0) +								// diffuse map is hdr
		(t0 ? (1 << 2) : 0) +								// has diffuse map
		(t1 ? (1 << 11) : 0) +								// has 2nd diffuse map
		(t2 ? (1 << 12) : 0) +								// has 3rd diffuse map
		(nm0 ? (1 << 3) : 0) +								// has normal map
		(nm1 ? (1 << 9) : 0) +								// has 2nd normal map
		(nm2 ? (1 << 10) : 0) +								// has 3rd normal map
		(s ? (1 << 4) : 0) +								// has specularity map
		(r ? (1 << 5) : 0) +								// has roughness map
		((flags & ANISOTROPIC) ? (1 << 6) : 0) +			// is anisotropic
		((flags & PLANE) ? (1 << 7) : 0) +					// is ground shadow plane
		((flags & SKYSPHERE) ? (1 << 8) : 0) +				// is skysphere
		((flags & SMOOTH) ? (1 << 13) : 0) +				// has smooth normals
		((flags & HASALPHA) ? (1 << 14) : 0) +				// has alpha
		(specularity >= 0.0f ? (1 << 15) : 0) +				// has unity specularity
		((flags & UNLIT) ? (1 << 16) : 0) +					// is unlit
		(cm ? (1 << 17) : 0) +								// has color mask map
		(am ? (1 << 18) : 0) +								// has alpha mask map
		((flags & INDIRECTONLY) ? 0 : (1 << 19)) +			// is directly visible (or only indirectly)
		(((uint)(metalness * 255.0f)) << 24) +				// metalness, encoded as 8-bit value
		((flags & ISCONDUCTOR) ? (1 << 20) : 0) +			// use the rough conductor brdf
		((flags & ISDIELECTRIC) ? (1 << 21) : 0);			// use the rough dielectric brdf
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
	if (cm) // color mask map
		gpuMat.cmapwidth = cm->width, gpuMat.cmapheight = cm->height,
		gpuMat.cuoffs = map[COLORMASK].uvoffset.x, gpuMat.cvoffs = map[COLORMASK].uvoffset.y,
		gpuMat.cuscale = map[COLORMASK].uvscale.x, gpuMat.cvscale = map[COLORMASK].uvscale.y;
	if (am) // alpha mask map
		gpuMat.amapwidth = am->width, gpuMat.amapheight = am->height,
		gpuMat.auoffs = map[ALPHAMASK].uvoffset.x, gpuMat.avoffs = map[ALPHAMASK].uvoffset.y,
		gpuMat.auscale = map[ALPHAMASK].uvscale.x, gpuMat.avscale = map[ALPHAMASK].uvscale.y;
}

// EOF