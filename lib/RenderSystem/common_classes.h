/* common_classes.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   The class defined in this header file are accessible thoughout the
   system and in all cores.

   Note that the vector data types are defined in common_types.h, except
   for the CUDA-based render cores, which include the CUDA headers
   directly. The types are interchangable though so this can be ignored.
*/

#pragma once

#ifndef __OPENCLCC__
namespace lighthouse2
{
#endif
#ifdef __OPENCLCC__
#define __CLORCUDA__
#endif
#ifdef __CUDACC__
#undef __CLORCUDA__
#define __CLORCUDA__
#endif

// to converge or not
enum Convergence
{
	Converge = 0,
	Restart = 1
};

//  +-----------------------------------------------------------------------------+
//  |  CoreTri - see HostTri for the host-side version.                           |
//  |  Complete data for a single triangle with:                                  |
//  |  - three uv layers                                                          |
//  |  - vertex normals and a face normal                                         |
//  |  - a material index                                                         |
//  |  - tangent and bitangent vectors                                            |
//  |  - alpha values for "consistent normal interpolation"                       |
//  |  - area and inverse area.                                                   |
//  |  The layout is tuned to minimize quad-float reads in GPU shading code,      |
//  |  at the expense of (some) wasted space.                               LH2'19|
//  +-----------------------------------------------------------------------------+
#ifndef __OPENCLCC__
class CoreTri
{
public:
#ifndef __CUDACC__
	// on the host, instantiated classes should be initialized
	CoreTri() { memset( this, 0, sizeof( CoreTri ) ); ltriIdx = -1; }
#endif
	float u0, u1, u2;		// 12
	int ltriIdx;			// 4, set only for emissive triangles, used for MIS
	float v0, v1, v2;		// 12
	uint material;			// 4
	float3 vN0;				// 12
	float Nx;				// 4
	float3 vN1;				// 12
	float Ny;				// 4
	float3 vN2;				// 12
	float Nz;				// 4
	float3 T;				// 12
	float area;				// 4
	float3 B;				// 12
	float invArea;			// 4
	float3 alpha;			// vertex0..2.w looks like a good place, but GetMaterial never needs it, so this extra field saves us two float4 reads.
	float LOD;				// for MIP mapping
	float3 vertex0; float dummy0;
	float3 vertex1; float dummy1;
	float3 vertex2; float dummy2; // total 11 * 16 = 176 bytes.
	void UpdateArea()
	{
		const float a = length( vertex1 - vertex0 );
		const float b = length( vertex2 - vertex1 );
		const float c = length( vertex0 - vertex2 );
		const float s = (a + b + c) * 0.5f;
		area = sqrtf( s * (s - a) * (s - b) * (s - c) ); // Heron's formula
	}
};
#else
// OpenCL float3 has 4-byte padding
struct CoreTri
{
	float u0, u1, u2;		// 12
	int ltriIdx;			// 4, set only for emissive triangles, used for MIS
	float v0, v1, v2;		// 12
	uint material;			// 4
	float4 vN0;				// 12 + 4, w: Nx
	float4 vN1;				// 12 + 4, w: Ny
	float4 vN2;				// 12 + 4, w: Nz
	float4 T;				// 12 + 4, w: area
	float4 B;				// 12 + 4, w: invArea
	float4 alpha;			// vertex0..2.w looks like a good place, but GetMaterial never needs it, so this extra field saves us two float4 reads. + 4 LOD
	float3 vertex0;			// 12 + 4, w: dummy1
	float3 vertex1;			// 12 + 4, w: dummy2
	float3 vertex2;			// 12 + 4, w: dummy3, total 11 * 16 = 176 bytes.
};
void UpdateArea( struct CoreTri* tri )
{
	const float a = length( tri->vertex1 - tri->vertex0 );
	const float b = length( tri->vertex2 - tri->vertex1 );
	const float c = length( tri->vertex0 - tri->vertex2 );
	const float s = (a + b + c) * 0.5f;
	tri->T.w = sqrtf( s * (s - a) * (s - b) * (s - c) ); // Heron's formula
}
#endif

//  +-----------------------------------------------------------------------------+
//  |  CoreTri4                                                                   |
//  |  Set of quadfloats with the same size as a single CoreTri.                  |
//  |  GPU code reads the required fields into tdata0..9, and may use the         |
//  |  included macros to efficiently access individual items.              LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreTri4
{
	float4 u4;				// w: light tri idx		tdata0
	float4 v4;				// w: material			tdata1
	float4 vN0;				// w: Nx				tdata2
	float4 vN1;				// w: Ny				tdata3
	float4 vN2;				// w: Nz				tdata4
	float4 T4;				// w: area				tdata5
	float4 B4;				// w: invArea			tdata6
	float4 alpha4;			// w: triLOD
	float4 vertex[3];		// 48					tdata7, tdata8, tdata9
#define TRI_U0			tdata0.x
#define TRI_U1			tdata0.y
#define TRI_U2			tdata0.z
#define TRI_LTRIIDX		__float_as_int( tdata0.w )
#define TRI_V0			tdata1.x
#define TRI_V1			tdata1.y
#define TRI_V2			tdata1.z
#define TRI_MATERIAL	__float_as_int( tdata1.w )
#define TRI_N0			make_float3( tdata2.x, tdata2.y, tdata2.z )
#define TRI_N1			make_float3( tdata3.x, tdata3.y, tdata3.z )
#define TRI_N2			make_float3( tdata4.x, tdata4.y, tdata4.z )
#define TRI_N			make_float3( tdata2.w, tdata3.w, tdata4.w )
#define TRI_T			make_float3( tdata5.x, tdata5.y, tdata5.z )
#define TRI_B			make_float3( tdata6.x, tdata6.y, tdata6.z )
#define TRI_AREA		tdata5.w
#define TRI_INVAREA		tdata6.w
#define TRI_LOD			vertexAlpha.w
};

//  +-----------------------------------------------------------------------------+
//  |  CoreInstanceDesc                                                           |
//  |  Instance descriptor. We will pass an array of these to the shading code,   |
//  |  so the path tracers can easily find triangles and a matrix for the         |
//  |  intersected mesh. Note that multiple instances may reference the same      |
//  |  triangle array.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
struct float4x4 { float4 A, B, C, D; };
#ifndef __OPENCLCC__ // OpenCL host and GPU instance descriptor structure is different and is defined inside OpenCL's core settings.
struct CoreInstanceDesc
{
	CoreTri4* triangles;						// device pointer to model triangle array
	int dummy1, dummy2;							// padding; 80 byte object
	float4x4 invTransform;						// inverse transform for the instance
};
#endif

//  +-----------------------------------------------------------------------------+
//  |  CoreMaterial - keep this in sync with the HostMaterial class, as these     |
//  |  will be copied into CoreMaterials before being passed to the cores.  LH2'19|
//  +-----------------------------------------------------------------------------+
class CoreMaterial
{
public:
	struct Vec3Value
	{
		float3 value;							// default value if map is absent; 1e-32 means: not set
		int textureID;							// texture ID; 'value'field is used if -1
		float scale;							// map values will be scaled by this
		float2 uvscale, uvoffset;				// uv coordinate scale and offset
	};
	struct ScalarValue
	{
		float value;							// default value if map is absent; 1e32 means: not set
		int textureID;							// texture ID; -1 denotes empty slot
		int component;							// 0 = x, 1 = y, 2 = z, 3 = w
		float scale;							// map values will be scaled by this
		float2 uvscale, uvoffset;				// uv coordinate scale and offset
	};

	// START OF HOSTMATERIAL DATA COPY

	// material properties
	Vec3Value color;							// universal material property: base color
	Vec3Value detailColor;						// universal material property: detail texture
	Vec3Value normals;							// universal material property: normal map
	Vec3Value detailNormals;					// universal material property: detail normal map			
	uint flags;									// material flags: 1 = SMOOTH, 2 = HASALPHA

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
};

enum TexelStorage
{
	ARGB32 = 0,									// regular texture data, RenderCore::texel32data
	ARGB128,									// hdr texture data, RenderCore::texel128data
	NRM32										// int32 encoded normal map data, RenderCore::normal32data
};
struct CoreTexDesc
{
	// This structure will never be stored on the GPU. RenderCore will use this to free the RenderSystem of
	// the burden of maintaining the continuous arrays of texel data, which really is a RenderCore job.
	union { float4* fdata; uchar4* idata; }; // points to the texel data in the original texture
#ifdef __CLORCUDA__
	// skip initial values in device code
	uint pixelCount;							// width and height are irrelevant; already stored with material
	uint firstPixel;							// start in continuous storage of the texture
	uint MIPlevels;								// number of MIP levels
#ifdef __CUDACC__
	TexelStorage storage;
#else
	enum TexelStorage storage;
#endif
#else
	uint width = 0, height = 0;					// width and height of the texture
	uint flags = 0;								// texture flags
	uint pixelCount = 0;						// width * height
	uint firstPixel = 0;						// start in continuous storage of the texture
	uint MIPlevels = 1;							// number of MIP levels
	TexelStorage storage = ARGB32;
#endif
};

//  +-----------------------------------------------------------------------------+
//  |  CoreLightTri - see HostLightTri for host-side version.                     |
//  |  Data layout for a light emitting triangle.                           LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreLightTri
{
#ifndef __OPENCLCC__
	float3 centre; float energy;				// data0 / centre4
	float3 N; float area;						// data1 / N4
	float3 radiance; int dummy2;				// data2
	float3 vertex0; int triIdx;					// data3
	float3 vertex1; int instIdx;				// data4
	float3 vertex2; int dummy1;					// data5
#else
	// OpenCL float3 has 4-byte padding
	float4 centre;								// w: float energy;		// data0 / centre4
	float4 N;									// w: float area;		// data1 / N4
	float4 radiance;							// w: int dummy2;		// data2
	float4 vertex0;								// w: int triIdx;		// data3
	float4 vertex1;								// w: int instIdx;		// data4
	float4 vertex2;								// w: int dummy1;		// data5
#endif
	// float4 access helper
#define AREALIGHT_ENERGY centre4.w
};
struct CoreLightTri4 { float4 data0, data1, data2, data3, data4, data5; };

//  +-----------------------------------------------------------------------------+
//  |  CorePointLight - see HostPointLight for host-side version.                 |
//  |  Data layout for a point light.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
struct CorePointLight
{
#ifndef __OPENCLCC__
	float3 position; float energy;				// data0 / position4
	float3 radiance; int dummy;					// data1
#else
	// OpenCL float3 has 4-byte padding
	float4 position;							// w: float energy;		// data0 / position4
	float4 radiance;							// w: int dummy;		// data1
#endif
	// float4 access helper
#define POINTLIGHT_ENERGY position4.w
};
struct CorePointLight4 { float4 data0, data1; };

//  +-----------------------------------------------------------------------------+
//  |  CoreSpotLight - see HostSpotLight for host-side version.                   |
//  |  Data layout for a spot light.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreSpotLight
{
#ifndef __OPENCLCC__
	float3 position; float cosInner;			// data0 / position4
	float3 radiance; float cosOuter;			// data1 / radiance4
	float3 direction; int dummy;				// data2 / direction4
#else
	// OpenCL float3 has 4-byte padding
	float4 position;							// w: float cosInner;	// data0 / position4
	float4 radiance;							// w: float cosOuter;	// data1 / radiance4
	float4 direction;							// w: int dummy;		// data2 / direction4
#endif
	// float4 access helpers
#define SPOTLIGHT_INNER	position4.w
#define SPOTLIGHT_OUTER	radiance4.w
};
struct CoreSpotLight4 { float4 data0, data1, data2; };

//  +-----------------------------------------------------------------------------+
//  |  CoreDirectionalLight- see HostDirectionalLight for host-side version.      |
//  |  Data layout for a directional light.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreDirectionalLight
{
#ifndef __OPENCLCC__
	float3 direction; float energy;				// data0 / direction4
	float3 radiance; int dummy;					// data1
#else
	// OpenCL float3 has 4-byte padding
	float4 direction;							// w: float energy;		// data0 / direction4
	float4 radiance;							// w: int dummy;		// data1
#endif
	// float4 access helper
#define DIRLIGHT_ENERGY direction4.w
};
struct CoreDirectionalLight4 { float4 data0, data1; };

//  +-----------------------------------------------------------------------------+
//  |  ViewPyramid                                                                |
//  |  Defines a camera view. Used for rendering and reprojection.          LH2'19|
//  +-----------------------------------------------------------------------------+
struct ViewPyramid
{
#ifdef __CLORCUDA__
	float3 pos;
	float3 p1;
	float3 p2;
	float3 p3;
	float aperture;
	float spreadAngle;
	float imagePlane;
	float focalDistance;
	float distortion;
#else
	float3 pos = make_float3( 0 );
	float3 p1 = make_float3( -1, -1, -1 );
	float3 p2 = make_float3( 1, -1, -1 );
	float3 p3 = make_float3( -1, 1, -1 );
	float aperture = 0;
	float spreadAngle = 0.01f; // spread angle of center pixel
	float imagePlane = 0.01f;
	float focalDistance = 0.01f;
	float distortion = 0.05f; // subtle barrel distortion
#endif
};

#ifndef __OPENCLCC__
typedef CoreTri HostTri; // these are identical
} // namespace lighthouse2
using namespace lighthouse2;
#endif

// EOF