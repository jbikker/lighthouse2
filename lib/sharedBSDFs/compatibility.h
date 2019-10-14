#ifndef COMPATIBILITY_H
#define COMPATIBILITY_H

// Compatibility header accomodate 

#if defined(__CUDACC__) || defined(WIN32) || defined(__linux__)
#ifndef LH2_DEVFUNC
#define LH2_DEVFUNC __forceinline__ __device__
#endif
#ifndef LH2_KERNEL
#define LH2_KERNEL __global__
#endif
#define REFERENCE_OF(x) x&
#else
#define LH2_DEVFUNC
#define LH2_KERNEL
#define REFERENCE_OF(x) inout x
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4
#define make_float2 vec2
#define make_float3 vec3
#define make_float4 vec4

#define fabs abs
#endif

#ifndef PI
#define PI					3.14159265358979323846264f
#endif
#ifndef INVPI
#define INVPI				0.31830988618379067153777f
#endif
#ifndef INV2PI
#define INV2PI				0.15915494309189533576888f
#endif
#ifndef TWOPI
#define TWOPI				6.28318530717958647692528f
#endif
#ifndef SQRT_PI_INV
#define SQRT_PI_INV			0.56418958355f
#endif
#ifndef LARGE_FLOAT
#define LARGE_FLOAT			1e34f
#endif
#ifndef EPSILON
#define EPSILON				0.0001f
#endif
#ifndef MINROUGHNESS
#define MINROUGHNESS		0.0001f	// minimal GGX roughness
#endif
#ifndef BLACK
#define BLACK				make_float3( 0 )
#endif
#ifndef WHITE
#define WHITE				make_float3( 1 )
#endif
#ifndef MIPLEVELCOUNT
#define MIPLEVELCOUNT		5
#endif

#endif
