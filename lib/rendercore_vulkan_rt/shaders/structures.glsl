/* structures.glsl - Copyright 2019 Utrecht University

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

#ifndef STRUCTURES_GLSL
#define STRUCTURES_GLSL

// clang-format off
struct PotentialContribution
{
	vec4 Origin;
	vec4 Direction;
	vec4 Emission_pixelIdx;
};

struct Camera
{
	vec4 posLensSize;
	vec4 right_aperture;
	vec4 up_spreadAngle;
	vec4 p1;
	int samplesTaken, phase;
	int scrwidth, scrheight;
};

struct Material
{
	uvec4 baseData4;
	uvec4 parameters;
	ivec4 t0data4;
	ivec4 t1data4;
	ivec4 t2data4;
	ivec4 n0data4;
	ivec4 n1data4;
	ivec4 n2data4;
	ivec4 sdata4;
	ivec4 rdata4;
	ivec4 m0data4;
	ivec4 m1data4;
#define MAT_FLAGS					(uint(baseData.w))
#define MAT_ISDIELECTRIC			((flags & (1u << 0u)  ) != 0)
#define MAT_DIFFUSEMAPISHDR			((flags & (1u << 1u)  ) != 0)
#define MAT_HASDIFFUSEMAP			((flags & (1u << 2u)  ) != 0)
#define MAT_HASNORMALMAP			((flags & (1u << 3u)  ) != 0)
#define MAT_HASSPECULARITYMAP		((flags & (1u << 4u)  ) != 0)
#define MAT_HASROUGHNESSMAP			((flags & (1u << 5u)  ) != 0)
#define MAT_ISANISOTROPIC			((flags & (1u << 6u)  ) != 0)
#define MAT_HAS2NDNORMALMAP			((flags & (1u << 7u)  ) != 0)
#define MAT_HAS3RDNORMALMAP			((flags & (1u << 8u)  ) != 0)
#define MAT_HAS2NDDIFFUSEMAP		((flags & (1u << 9u)  ) != 0)
#define MAT_HAS3RDDIFFUSEMAP		((flags & (1u << 10u) ) != 0)
#define MAT_HASSMOOTHNORMALS		((flags & (1u << 11u) ) != 0)
#define MAT_HASALPHA				((flags & (1u << 12u) ) != 0)
};

struct CoreTri
{
	vec4 u4;				// w: light tri idx			tdata0
	vec4 v4;				// w: material				tdata1
	vec4 vN0;				// w: Nx					tdata2
	vec4 vN1;				// w: Ny					tdata3
	vec4 vN2;				// w: Nz					tdata4
	vec4 T4;				// w: area					tdata5
	vec4 B4;				// w: invArea				tdata6
	vec4 alpha4;			// w: triLOD
	vec4 vertex0;		    // 48						tdata7
	vec4 vertex1;			//							tdata8
	vec4 vertex2;			//							tdata9
#define TRI_U0			tdata0.x
#define TRI_U1			tdata0.y
#define TRI_U2			tdata0.z
#define TRI_LTRIIDX		floatBitsToUint( tdata0.w )
#define TRI_V0			tdata1.x
#define TRI_V1			tdata1.y
#define TRI_V2			tdata1.z
#define TRI_MATERIAL	floatBitsToUint( tdata1.w )
#define TRI_N0			vec3( tdata2.x, tdata2.y, tdata2.z )
#define TRI_N1			vec3( tdata3.x, tdata3.y, tdata3.z )
#define TRI_N2			vec3( tdata4.x, tdata4.y, tdata4.z )
#define TRI_N			vec3( tdata2.w, tdata3.w, tdata4.w )
#define TRI_T			vec3( tdata5.x, tdata5.y, tdata5.z )
#define TRI_B			vec3( tdata6.x, tdata6.y, tdata6.z )
#define TRI_AREA		tdata5.w
#define TRI_INVAREA		tdata6.w
#define TRI_LOD			alpha4.w
};

#define CHAR2FLT(x, s) ((float( ((x >> s) & 255)) ) * (1.0f / 255.0f))

struct ShadingData
{
	// This structure is filled for an intersection point. It will contain the spatially varying material properties.
	vec3 color; int flags;
	vec3 absorption; int matID;
	uvec4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
#define IS_SPECULAR (0)
#define IS_EMISSIVE (shadingData.color.x > 1.0f || shadingData.color.y > 1.0f || shadingData.color.z > 1.0f)
#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
#define ETA CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM0 CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM1 CHAR2FLT( shadingData.parameters.w, 0 )
#define CUSTOM2 CHAR2FLT( shadingData.parameters.w, 8 )
#define CUSTOM3 CHAR2FLT( shadingData.parameters.w, 16 )
#define CUSTOM4 CHAR2FLT( shadingData.parameters.w, 24 )
};

struct CoreLightTri4
{
	vec4 data0;
	vec4 data1;
	vec4 data2;
	vec4 data3;
	vec4 data4;
	vec4 data5;

#define AREALIGHT_ENERGY centre4.w
};

struct CorePointLight4
{
	vec4 data0;
	vec4 data1;
#define POINTLIGHT_ENERGY position4.w
};

struct CoreSpotLight4
{
	vec4 data0;
	vec4 data1;
	vec4 data2;
#define SPOTLIGHT_INNER	position4.w
#define SPOTLIGHT_OUTER	radiance4.w
};

struct CoreDirectionalLight4
{
	vec4 data0;
	vec4 data1;
#define DIRLIGHT_ENERGY direction4.w
};

// clang-format on
#endif