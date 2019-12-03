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
namespace lighthouse2 {
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
	float4 u4;				// w: light tri idx			tdata0
	float4 v4;				// w: material				tdata1
	float4 vN0;				// w: Nx					tdata2
	float4 vN1;				// w: Ny					tdata3
	float4 vN2;				// w: Nz					tdata4
	float4 T4;				// w: area					tdata5
	float4 B4;				// w: invArea				tdata6
	float4 alpha4;			// w: triLOD
	float4 vertex[3];		// 48						tdata7, tdata8, tdata9
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
	CoreTri4* triangles;					// device pointer to model triangle array
	int dummy1, dummy2;					// padding; 80 byte object
	float4x4 invTransform;				// inverse transform for the instance
};
#endif

//  +-----------------------------------------------------------------------------+
//  |  CoreMaterial - see HostMaterial for host-side version.CoreTri4             |
//  |  Material layout optimized for rendering, created from a HostMaterial.      |
//  |  The structure is heavily optimized to require exactly a quadfloat worth    |
//  |  of data for each feature / data layer.                               LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreMaterial
{
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b; uint flags;
	uint4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
	// texture / normal map descriptors; exactly 128-bit each
	/* read if bit  2 set */ short texwidth0, texheight0; half uscale0, vscale0, uoffs0, voffs0; uint texaddr0;
	/* read if bit 11 set */ short texwidth1, texheight1; half uscale1, vscale1, uoffs1, voffs1; uint texaddr1;
	/* read if bit 12 set */ short texwidth2, texheight2; half uscale2, vscale2, uoffs2, voffs2; uint texaddr2;
	/* read if bit  3 set */ short nmapwidth0, nmapheight0; half nuscale0, nvscale0, nuoffs0, nvoffs0; uint nmapaddr0;
	/* read if bit  9 set */ short nmapwidth1, nmapheight1; half nuscale1, nvscale1, nuoffs1, nvoffs1; uint nmapaddr1;
	/* read if bit 10 set */ short nmapwidth2, nmapheight2; half nuscale2, nvscale2, nuoffs2, nvoffs2; uint nmapaddr2;
	/* read if bit  4 set */ short smapwidth, smapheight; half suscale, svscale, suoffs, svoffs; uint smapaddr;
	/* read if bit  5 set */ short rmapwidth, rmapheight; half ruscale, rvscale, ruoffs, rvoffs; uint rmapaddr;
#if 1
	// WUT
	/* read if bit 17 set */ short cmapwidth, cmapheight; half cuscale, cvscale, cuoffs, cvoffs; uint cmapaddr;
	/* read if bit 18 set */ short amapwidth, amapheight; half auscale, avscale, auoffs, avoffs; uint amapaddr;
#else
	// TODO: to match CoreMaterial4
	/* read if bit 17 set */ short m0mapwidth, m0mapheight; half m0uscale, m0vscale, m0uoffs, m0voffs; uint m0mapaddr;
	/* read if bit 18 set */ short m1mapwidth, m1mapheight; half m1uscale, m1vscale, m1uoffs, m1voffs; uint m1mapaddr;
#endif
#ifndef __CLORCUDA__
	#define CHAR2FLT(a,s) (((float)(((a)>>s)&255))*(1.0f/255.0f))
	float metallic() { return CHAR2FLT( parameters.x, 0 ); }
	float subsurface() { return CHAR2FLT( parameters.x, 8 ); }
	float specular() { return CHAR2FLT( parameters.x, 16 ); }
	float roughness() { return (max( 0.001f, CHAR2FLT( parameters.x, 24 ) )); }
	float spectint() { return CHAR2FLT( parameters.y, 0 ); }
	float anisotropic() { return CHAR2FLT( parameters.y, 8 ); }
	float sheen() { return CHAR2FLT( parameters.y, 16 ); }
	float sheentint() { return CHAR2FLT( parameters.y, 24 ); }
	float clearcoat() { return CHAR2FLT( parameters.z, 0 ); }
	float clearcoatgloss() { return CHAR2FLT( parameters.z, 8 ); }
	float transmission() { return CHAR2FLT( parameters.z, 16 ); }
	float eta() { return *(uint*)&parameters.w; }
#endif
};
// texture layers in HostMaterial and CoreMaterialEx
#define TEXTURE0		0
#define TEXTURE1		1
#define TEXTURE2		2
#define NORMALMAP0		3
#define NORMALMAP1		4
#define NORMALMAP2		5
#define SPECULARITY		6
#define ROUGHNESS0		7
#define ROUGHNESS1		8
#define COLORMASK		9
#define ALPHAMASK		10
struct CoreMaterialEx
{
	// This structure contains the texture IDs used by the HostMaterial this CoreMaterial is based on.
	// These are needed by the RenderCore to set the texaddr fields for the material.
	int texture[11];
};
enum TexelStorage
{
	ARGB32 = 0,							// regular texture data, RenderCore::texel32data
	ARGB128,							// hdr texture data, RenderCore::texel128data
	NRM32								// int32 encoded normal map data, RenderCore::normal32data
};
struct CoreTexDesc
{
	// This structure will never be stored on the GPU. RenderCore will use this to free the RenderSystem of
	// the burden of maintaining the continuous arrays of texel data, which really is a RenderCore job.
	union { float4* fdata; uchar4* idata; }; // points to the texel data in the original texture
#ifdef __CLORCUDA__
	// skip initial values in device code
	uint pixelCount;					// width and height are irrelevant; already stored with material
	uint firstPixel;					// start in continuous storage of the texture
	uint MIPlevels;						// number of MIP levels
#ifdef __CUDACC__
	TexelStorage storage;
#else
	enum TexelStorage storage;
#endif
#else
	uint pixelCount = 0;				// width and height are irrelevant; already stored with material
	uint firstPixel = 0;				// start in continuous storage of the texture
	uint MIPlevels = 1;					// number of MIP levels
	TexelStorage storage = ARGB32;
#endif
};

//  +-----------------------------------------------------------------------------+
//  |  CoreMaterial4                                                              |
//  |  Set of quadfloats with the same size as a single CoreMaterial.       LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreMaterial4
{
	uint4 baseData4;
	uint4 parameters;
	uint4 t0data4;
	uint4 t1data4;
	uint4 t2data4;
	uint4 n0data4;
	uint4 n1data4;
	uint4 n2data4;
	uint4 sdata4;
	uint4 rdata4;
	uint4 m0data4;
	uint4 m1data4;
	// float4 dielec4;
	// float4 spec4;
	// flag query macros
#define MAT_ISDIELECTRIC			(flags & (1 << 0))
#define MAT_DIFFUSEMAPISHDR			(flags & (1 << 1))
#define MAT_HASDIFFUSEMAP			(flags & (1 << 2))
#define MAT_HASNORMALMAP			(flags & (1 << 3))
#define MAT_HASSPECULARITYMAP		(flags & (1 << 4))
#define MAT_HASROUGHNESSMAP			(flags & (1 << 5))
#define MAT_ISANISOTROPIC			(flags & (1 << 6))
#define MAT_HAS2NDNORMALMAP			(flags & (1 << 7))
#define MAT_HAS3RDNORMALMAP			(flags & (1 << 8))
#define MAT_HAS2NDDIFFUSEMAP		(flags & (1 << 9))
#define MAT_HAS3RDDIFFUSEMAP		(flags & (1 << 10))
#define MAT_HASSMOOTHNORMALS		(flags & (1 << 11))
#define MAT_HASALPHA				(flags & (1 << 12))
};

//  +-----------------------------------------------------------------------------+
//  |  CoreLightTri - see HostLightTri for host-side version.                     |
//  |  Data layout for a light emitting triangle.                           LH2'19|
//  +-----------------------------------------------------------------------------+
struct CoreLightTri
{
#ifndef __OPENCLCC__
	float3 centre; float energy;		// data0 / centre4
	float3 N; float area;				// data1 / N4
	float3 radiance; int dummy2;		// data2
	float3 vertex0; int triIdx;			// data3
	float3 vertex1; int instIdx;		// data4
	float3 vertex2; int dummy1;			// data5
#else
	// OpenCL float3 has 4-byte padding
	float4 centre;						// w: float energy;		// data0 / centre4
	float4 N;							// w: float area;		// data1 / N4
	float4 radiance;					// w: int dummy2;		// data2
	float4 vertex0;						// w: int triIdx;		// data3
	float4 vertex1;						// w: int instIdx;		// data4
	float4 vertex2;						// w: int dummy1;		// data5
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
	float3 position; float energy;		// data0 / position4
	float3 radiance; int dummy;			// data1
#else
	// OpenCL float3 has 4-byte padding
	float4 position;					// w: float energy;		// data0 / position4
	float4 radiance;					// w: int dummy;		// data1
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
	float3 position; float cosInner;	// data0 / position4
	float3 radiance; float cosOuter;	// data1 / radiance4
	float3 direction; int dummy;		// data2 / direction4
#else
	// OpenCL float3 has 4-byte padding
	float4 position;					// w: float cosInner;	// data0 / position4
	float4 radiance;					// w: float cosOuter;	// data1 / radiance4
	float4 direction;					// w: int dummy;		// data2 / direction4
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
	float3 direction; float energy;		// data0 / direction4
	float3 radiance; int dummy;			// data1
#else
	// OpenCL float3 has 4-byte padding
	float4 direction;					// w: float energy;		// data0 / direction4
	float4 radiance;					// w: int dummy;		// data1
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