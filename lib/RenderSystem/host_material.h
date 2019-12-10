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
	enum
	{
		DISNEYBRDF = 1,
		LAMBERTBSDF,
		// add additional material IDs here
	};
	struct Vec3Value
	{
		// Vec3Value / ScalarValue: all material parameters can be spatially variant or invariant.
		// If a map is used, this map may have an offset and scale. The map values may also be
		// scaled. This facilitates parameter map reuse.
		// A parameter that has not been specified has a -1 textureID and a 1e-32f value. Such
		// parameters should not be used when converting the HostMaterial to a CoreMaterial in
		// the core implementation.
		Vec3Value() = default;
		Vec3Value( const float f ) : value( make_float3( f ) ) {}
		Vec3Value( const float3 f ) : value( f ) {}
		float3 value = make_float3( 1e-32f );	// default value if map is absent; 1e-32 means: not set
		int textureID = -1;						// texture ID; 'value'field is used if -1
		float scale = 1;						// map values will be scaled by this
		float2 uvscale = make_float2( 1 );		// uv coordinate scale
		float2 uvoffset = make_float2( 0 );		// uv coordinate offset
		bool Specified() { return (value.x != 1e32f) || (value.y != 1e32f) || (value.z != 1e32f) || (textureID != -1); }
		float3& operator()() { return value; }
	};
	struct ScalarValue
	{
		ScalarValue() = default;
		ScalarValue( const float f ) : value( f ) {}
		float value = 1e-32f;					// default value if map is absent; 1e32 means: not set
		int textureID = -1;						// texture ID; -1 denotes empty slot
		int component = 0;						// 0 = x, 1 = y, 2 = z, 3 = w
		float scale = 1;						// map values will be scaled by this
		float2 uvscale = make_float2( 1 );		// uv coordinate scale
		float2 uvoffset = make_float2( 0 );		// uv coordinate offset
		bool Specified() { return (value != 1e32f) || (textureID != -1); }
		float& operator()() { return value; }
	};
	enum
	{
		SMOOTH = 1,								// material uses normal interpolation
		HASALPHA = 2,							// material textures use alpha channel
		FROM_MTL = 4							// changes are persistent for these, not for others
	};

	// constructor / destructor
	HostMaterial() = default;

	// methods
	void ConvertFrom( const tinyobjMaterial& );
	void ConvertFrom( const tinygltfMaterial&, const tinygltfModel&, const int textureBase );
	bool IsEmissive() { float3& c = color(); return c.x > 1 || c.y > 1 || c.z > 1; /* ignores vec3map */ }

	// START OF DATA THAT WILL BE COPIED TO COREMATERIAL

	// Note: cores receive an exact copy of the material properties. The differences 
	// between a HostMaterial and a CoreMaterial are:
	// 1. A CoreMaterial instance does not initialize any fields;
	// 2. Data other than material parameters is not included in the CoreMaterial.
	// Cores are expected to take the data as-is, or to convert it to a core-specific 
	// format.

	// material properties
	Vec3Value color = Vec3Value( 1 );			// universal material property: base color
	Vec3Value detailColor;						// universal material property: detail texture
	Vec3Value normals;							// universal material property: normal map
	Vec3Value detailNormals;					// universal material property: detail normal map			
	uint flags = SMOOTH;						// material flags: 1 = SMOOTH, 2 = HASALPHA

	// Disney BRDF properties
	// Data for the Disney Principled BRDF.
	Vec3Value absorption;
	ScalarValue metallic;
	ScalarValue subsurface;
	ScalarValue specular;
	ScalarValue roughness;
	ScalarValue specularTint;
	ScalarValue anisotropic;
	ScalarValue sheen;
	ScalarValue sheenTint;
	ScalarValue clearcoat;
	ScalarValue clearcoatGloss;
	ScalarValue transmission;
	ScalarValue eta;

	// lambert bsdf properties
	// Data for a basic Lambertian BRDF, augmented with pure specular reflection and
	// refraction. Assumptions:
	// diffuse component = 1 - (reflectionm + refraction); 
	// (reflection + refraction) < 1;
	// ior is the index of refraction of the medium below the shading point.
	// Vec3Value absorption;					// shared with disney brdf
	ScalarValue reflection;
	ScalarValue refraction;
	ScalarValue ior;

	// additional bxdf properties
	// Add data for new BxDFs here. Add '//' to values that are shared with previously
	// specified material parameter sets.
	// ...

	// END OF DATA THAT WILL BE COPIED TO COREMATERIAL

	// identifier and name
	string name = "unnamed";					// material name, not for unique identification
	string origin;								// origin: file from which the data was loaded, with full path
	int ID = -1;								// unique integer ID of this material
	uint refCount = 1;							// the number of models that use this material

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

	// internal
private:
	uint prevFlags = SMOOTH;					// initially identical to flags
	TRACKCHANGES;								// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF