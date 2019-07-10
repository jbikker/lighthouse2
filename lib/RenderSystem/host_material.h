/* host_material.h - Copyright 2019 Utrecht University

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
//  |  HostMaterial                                                               |
//  |  Host-side material definition.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
class HostMaterial
{
public:
	struct MapProps
	{
		int textureID = -1;						// texture ID; -1 denotes empty slot
		float valueScale = 1;					// texture value scale, only sensible for normal maps
		float2 uvscale = make_float2( 1 );		// uv coordinate scale
		float2 uvoffset = make_float2( 0 );		// uv coordinate offset
	};
	// constructor / destructor
	HostMaterial() = default;
	// methods
#ifdef RENDERSYSTEMBUILD
	// methods visible to the RenderSystem only
	void ConvertFrom( tinyobj::material_t& );
	void ConvertFrom( tinygltf::Material&, tinygltf::Model& );
	void ConvertTo( CoreMaterial&, CoreMaterialEx& );
#endif
	// data members
	enum
	{
		SMOOTH = 1,								// material uses normal interpolation
		HASALPHA = 2,							// material textures use alpha channel
		ANISOTROPIC = 4,						// material has anisotropic roughness
		PLANE = 8,								// special purpose flag; shadow-receiving plane
		SKYSPHERE = 16,							// special purpose flag; sky sphere material
		UNLIT = 32,								// special purpose flag; material is unlit
		INDIRECTONLY = 64,						// special purpose flag; material is only visible indirectly
		FROM_MTL = 128,							// changes are persistent for these, not for others
		ISCONDUCTOR = 256,						// rough conductor
		ISDIELECTRIC = 512						// rough dielectric. If 256 and 512 not specified: diffuse.
	};
	// identifier and name
	string name = "unnamed";					// material name, not for unique identification
	string origin;								// origin: file from which the data was loaded, with full path
	int ID = -1;								// unique integer ID of this material
	uint flags = SMOOTH;						// material properties
	uint refCount = 1;							// the number of models that use this material
	// maps and colors
	float3 color = make_float3( 1 );			// diffuse color / reflectivity (conductors); texture overrides this
	float3 specularColor = make_float3( 1 );	// specular color / extinction coefficient (conductors)
	float2 roughness[2] =						// anisotropic roughness for each of the two layers
	{ make_float2( 1 ), make_float2( 1 ) };
	MapProps map[11];							// bitmap data
	// pbr data
	float eta = 1.0f;							// index of refraction for rough dielectrics
	float3 absorption = make_float3( 0 );		// dielectric absorption according to Beer's law
	// custom data
	float metalness = 1;						// metalness; metals use the diffuse colors for specular reflections, non-metals use white
	float specularity = -1;						// unity specularity slider (if not -1: affects diffuse color, eta, and roughness2)
	// field for the BuildMaterialList method of HostMesh
	bool visited = false;						// last mesh that checked this material
	bool AlphaChanged()
	{
		// A change to the alpha flag should trigger a change to any mesh using this flag as
		// well. This method allows us to track this.
		const bool changed = (flags & HASALPHA) != (prevFlags & HASALPHA);
		prevFlags = flags;
		return changed;
	}
private:
	uint prevFlags = SMOOTH;					// initially identical to flags
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF