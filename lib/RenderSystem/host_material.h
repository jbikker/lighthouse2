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

namespace lighthouse2
{

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
	void ConvertFrom( const tinyobjMaterial& );
	void ConvertFrom( const tinygltfMaterial&, const tinygltfModel&, const int textureBase );
	void ConvertTo( CoreMaterial&, CoreMaterialEx& ) const;
	// data members
	enum
	{
		SMOOTH = 1,								// material uses normal interpolation
		HASALPHA = 2,							// material textures use alpha channel
		ANISOTROPIC = 4,						// material has anisotropic roughness
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
	float3 color = make_float3( 1 );
	float3 absorption = make_float3( 0 );
	float metallic = 0.0f;
	float subsurface = 0.0f;
	float specular = 0.5f;
	float roughness = 0.5f;
	float specularTint = 0.0f;
	float anisotropic = 0.0f;
	float sheen = 0.0f;
	float sheenTint = 0.0f;
	float clearcoat = 0.0f;
	float clearcoatGloss = 1.0f;
	float transmission = 0.0f;
	float eta = 1.0f;
	float custom0 = 0.0f;
	float custom1 = 0.0f;
	float custom2 = 0.0f;
	float custom3 = 0.0f;
	MapProps map[11];							// bitmap data
	// field for the BuildMaterialList method of HostMesh
	bool visited = false;						// last mesh that checked this material
	bool AlphaChanged()
	{
		// A change to the alpha flag should trigger a change to any mesh using this flag as
		// well. This method allows us to track this.
		const bool dirty = IsDirty();
		const bool changed = (flags & HASALPHA) != (prevFlags & HASALPHA);
		prevFlags = flags;
		if (!dirty) MarkAsNotDirty(); // checking if alpha changed should not make the object dirty.
		return changed;
	}
private:
	uint prevFlags = SMOOTH;					// initially identical to flags
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF