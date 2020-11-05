/*
* Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

/**
* @file   optix_7_device_impl.h
* @author NVIDIA Corporation
* @brief  OptiX public API
*
* OptiX public API Reference - Device side implementation
*/

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_device_impl.h is an internal header file and must not be used directly.  Please use optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_device_impl_h__
#define __optix_optix_7_device_impl_h__

#include "internal/optix_7_device_impl_exception.h"
#include "internal/optix_7_device_impl_transformations.h"

static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex )
{
    float ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    asm volatile(
        "call _optix_trace_0"
        ", (%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14"
        ");"
        :
        /* no return value */
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ), "r"( missSBTIndex )
        : );
}

static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out;
    asm volatile(
        "call (%0), _optix_trace_1"
        ", (%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15"
        ", %16"
        ");"
        : "=r"( p0_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ), "f"( rayTime ),
          "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ), "r"( missSBTIndex ), "r"( p0 )
        : );
    p0 = p0_out;
}

static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out;
    asm volatile(
        "call (%0, %1), _optix_trace_2"
        ", (%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16"
        ", %17, %18"
        ");"
        : "=r"( p0_out ), "=r"( p1_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 )
        : );
    p0 = p0_out;
    p1 = p1_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out;
    asm volatile(
        "call (%0, %1, %2), _optix_trace_3"
        ", (%3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17"
        ", %18, %19, %20"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2,
                                                   unsigned int&          p3 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out, p3_out;
    asm volatile(
        "call (%0, %1, %2, %3), _optix_trace_4"
        ", (%4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18"
        ", %19, %20, %21, %22"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out ), "=r"( p3_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 ), "r"( p3 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
    p3 = p3_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2,
                                                   unsigned int&          p3,
                                                   unsigned int&          p4 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out, p3_out, p4_out;
    asm volatile(
        "call (%0, %1, %2, %3, %4), _optix_trace_5"
        ", (%5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19"
        ", %20, %21, %22, %23, %24"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out ), "=r"( p3_out ), "=r"( p4_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 ), "r"( p3 ), "r"( p4 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
    p3 = p3_out;
    p4 = p4_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2,
                                                   unsigned int&          p3,
                                                   unsigned int&          p4,
                                                   unsigned int&          p5 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out, p3_out, p4_out, p5_out;
    asm volatile(
        "call (%0, %1, %2, %3, %4, %5), _optix_trace_6"
        ", (%6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20"
        ", %21, %22, %23, %24, %25, %26"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out ), "=r"( p3_out ), "=r"( p4_out ), "=r"( p5_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 ), "r"( p3 ), "r"( p4 ), "r"( p5 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
    p3 = p3_out;
    p4 = p4_out;
    p5 = p5_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2,
                                                   unsigned int&          p3,
                                                   unsigned int&          p4,
                                                   unsigned int&          p5,
                                                   unsigned int&          p6 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out, p3_out, p4_out, p5_out, p6_out;
    asm volatile(
        "call (%0, %1, %2, %3, %4, %5, %6), _optix_trace_7"
        ", (%7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21"
        ", %22, %23, %24, %25, %26, %27, %28"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out ), "=r"( p3_out ), "=r"( p4_out ), "=r"( p5_out ), "=r"( p6_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 ), "r"( p3 ), "r"( p4 ), "r"( p5 ), "r"( p6 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
    p3 = p3_out;
    p4 = p4_out;
    p5 = p5_out;
    p6 = p6_out;
}
static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
                                                   float3                 rayOrigin,
                                                   float3                 rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   unsigned int&          p0,
                                                   unsigned int&          p1,
                                                   unsigned int&          p2,
                                                   unsigned int&          p3,
                                                   unsigned int&          p4,
                                                   unsigned int&          p5,
                                                   unsigned int&          p6,
                                                   unsigned int&          p7 )
{
    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p0_out, p1_out, p2_out, p3_out, p4_out, p5_out, p6_out, p7_out;
    asm volatile(
        "call (%0, %1, %2, %3, %4, %5, %6, %7), _optix_trace_8"
        ", (%8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22"
        ", %23, %24, %25, %26, %27, %28, %29, %30"
        ");"
        : "=r"( p0_out ), "=r"( p1_out ), "=r"( p2_out ), "=r"( p3_out ), "=r"( p4_out ), "=r"( p5_out ),
          "=r"( p6_out ), "=r"( p7_out )
        : "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ), "f"( tmax ),
          "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( p0 ), "r"( p1 ), "r"( p2 ), "r"( p3 ), "r"( p4 ), "r"( p5 ), "r"( p6 ), "r"( p7 )
        : );
    p0 = p0_out;
    p1 = p1_out;
    p2 = p2_out;
    p3 = p3_out;
    p4 = p4_out;
    p5 = p5_out;
    p6 = p6_out;
    p7 = p7_out;
}

#define OPTIX_DEFINE_optixSetPayload_BODY( which )                                                                     \
    asm volatile( "call _optix_set_payload_" #which ", (%0);" : : "r"( p ) : );

#define OPTIX_DEFINE_optixGetPayload_BODY( which )                                                                     \
    unsigned int result;                                                                                               \
    asm volatile( "call (%0), _optix_get_payload_" #which ", ();" : "=r"( result ) : );                                         \
    return result;

static __forceinline__ __device__ void optixSetPayload_0( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 0 )
}

static __forceinline__ __device__ void optixSetPayload_1( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 1 )
}

static __forceinline__ __device__ void optixSetPayload_2( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 2 )
}

static __forceinline__ __device__ void optixSetPayload_3( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 3 )
}

static __forceinline__ __device__ void optixSetPayload_4( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 4 )
}

static __forceinline__ __device__ void optixSetPayload_5( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 5 )
}

static __forceinline__ __device__ void optixSetPayload_6( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 6 )
}

static __forceinline__ __device__ void optixSetPayload_7( unsigned int p )
{
    OPTIX_DEFINE_optixSetPayload_BODY( 7 )
}

static __forceinline__ __device__ unsigned int optixGetPayload_0()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 0 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_1()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 1 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_2()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 2 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_3()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 3 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_4()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 4 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_5()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 5 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_6()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 6 );
}

static __forceinline__ __device__ unsigned int optixGetPayload_7()
{
    OPTIX_DEFINE_optixGetPayload_BODY( 7 );
}

#undef OPTIX_DEFINE_optixSetPayload_BODY
#undef OPTIX_DEFINE_optixGetPayload_BODY

static __forceinline__ __device__ unsigned int optixUndefinedValue()
{
    unsigned int u0;
    asm( "call (%0), _optix_undef_value, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ float3 optixGetWorldRayOrigin()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_origin_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetWorldRayDirection()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_direction_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetObjectRayOrigin()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_object_ray_origin_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_object_ray_origin_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_object_ray_origin_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetObjectRayDirection()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_object_ray_direction_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_object_ray_direction_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_object_ray_direction_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float optixGetRayTmin()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmin, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetRayTmax()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmax, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetRayTime()
{
    float f0;
    asm( "call (%0), _optix_get_ray_time, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ unsigned int optixGetRayFlags()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_ray_flags, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetRayVisibilityMask()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_ray_visibility_mask, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ void optixGetTriangleVertexData( OptixTraversableHandle gas,
                                                                   unsigned int           primIdx,
                                                                   unsigned int           sbtGASIndex,
                                                                   float                  time,
                                                                   float3                 data[3] )
{
    asm( "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), _optix_get_triangle_vertex_data, "
         "(%9, %10, %11, %12);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[1].x ), "=f"( data[1].y ),
           "=f"( data[1].z ), "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetLinearCurveVertexData( OptixTraversableHandle gas,
                                                                      unsigned int           primIdx,
                                                                      unsigned int           sbtGASIndex,
                                                                      float                  time,
                                                                      float4                 data[2] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7), _optix_get_linear_curve_vertex_data, "
         "(%8, %9, %10, %11);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ),
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetQuadraticBSplineVertexData( OptixTraversableHandle gas,
                                                                           unsigned int         primIdx,
                                                                           unsigned int         sbtGASIndex,
                                                                           float                time,
                                                                           float4               data[3] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7,  %8, %9, %10, %11), _optix_get_quadratic_bspline_vertex_data, "
         "(%12, %13, %14, %15);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ), 
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w ),
           "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z ), "=f"( data[2].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ void optixGetCubicBSplineVertexData( OptixTraversableHandle gas,
                                                                       unsigned int         primIdx,
                                                                       unsigned int         sbtGASIndex,
                                                                       float                time,
                                                                       float4               data[4] )
{
    asm( "call (%0, %1, %2, %3,  %4, %5, %6, %7,  %8, %9, %10, %11,  %12, %13, %14, %15), "
         "_optix_get_cubic_bspline_vertex_data, "
         "(%16, %17, %18, %19);"
         : "=f"( data[0].x ), "=f"( data[0].y ), "=f"( data[0].z ), "=f"( data[0].w ), 
           "=f"( data[1].x ), "=f"( data[1].y ), "=f"( data[1].z ), "=f"( data[1].w ),
           "=f"( data[2].x ), "=f"( data[2].y ), "=f"( data[2].z ), "=f"( data[2].w ),
           "=f"( data[3].x ), "=f"( data[3].y ), "=f"( data[3].z ), "=f"( data[3].w )
         : "l"( gas ), "r"( primIdx ), "r"( sbtGASIndex ), "f"( time )
         : );
}

static __forceinline__ __device__ OptixTraversableHandle optixGetGASTraversableHandle()
{
    unsigned long long handle;
    asm( "call (%0), _optix_get_gas_traversable_handle, ();" : "=l"( handle ) : );
    return (OptixTraversableHandle)handle;
}

static __forceinline__ __device__ float optixGetGASMotionTimeBegin( OptixTraversableHandle handle )
{
    float f0;
    asm( "call (%0), _optix_get_gas_motion_time_begin, (%1);" : "=f"( f0 ) : "l"( handle ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetGASMotionTimeEnd( OptixTraversableHandle handle )
{
    float f0;
    asm( "call (%0), _optix_get_gas_motion_time_end, (%1);" : "=f"( f0 ) : "l"( handle ) : );
    return f0;
}

static __forceinline__ __device__ unsigned int optixGetGASMotionStepCount( OptixTraversableHandle handle )
{
    unsigned int u0;
    asm( "call (%0), _optix_get_gas_motion_step_count, (%1);" : "=r"( u0 ) : "l"( handle ) : );
    return u0;
}

static __forceinline__ __device__ void optixGetWorldToObjectTransformMatrix( float m[12] )
{
    if( optixGetTransformListSize() == 0 )
    {
        m[0]  = 1.0f;
        m[1]  = 0.0f;
        m[2]  = 0.0f;
        m[3]  = 0.0f;
        m[4]  = 0.0f;
        m[5]  = 1.0f;
        m[6]  = 0.0f;
        m[7]  = 0.0f;
        m[8]  = 0.0f;
        m[9]  = 0.0f;
        m[10] = 1.0f;
        m[11] = 0.0f;
        return;
    }

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    m[0]  = m0.x;
    m[1]  = m0.y;
    m[2]  = m0.z;
    m[3]  = m0.w;
    m[4]  = m1.x;
    m[5]  = m1.y;
    m[6]  = m1.z;
    m[7]  = m1.w;
    m[8]  = m2.x;
    m[9]  = m2.y;
    m[10] = m2.z;
    m[11] = m2.w;
}

static __forceinline__ __device__ void optixGetObjectToWorldTransformMatrix( float m[12] )
{
    if( optixGetTransformListSize() == 0 )
    {
        m[0]  = 1.0f;
        m[1]  = 0.0f;
        m[2]  = 0.0f;
        m[3]  = 0.0f;
        m[4]  = 0.0f;
        m[5]  = 1.0f;
        m[6]  = 0.0f;
        m[7]  = 0.0f;
        m[8]  = 0.0f;
        m[9]  = 0.0f;
        m[10] = 1.0f;
        m[11] = 0.0f;
        return;
    }

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    m[0]  = m0.x;
    m[1]  = m0.y;
    m[2]  = m0.z;
    m[3]  = m0.w;
    m[4]  = m1.x;
    m[5]  = m1.y;
    m[6]  = m1.z;
    m[7]  = m1.w;
    m[8]  = m2.x;
    m[9]  = m2.y;
    m[10] = m2.z;
    m[11] = m2.w;
}

static __forceinline__ __device__ float3 optixTransformPointFromWorldToObjectSpace( float3 point )
{
    if( optixGetTransformListSize() == 0 )
        return point;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformPoint( m0, m1, m2, point );
}

static __forceinline__ __device__ float3 optixTransformVectorFromWorldToObjectSpace( float3 vec )
{
    if( optixGetTransformListSize() == 0 )
        return vec;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformVector( m0, m1, m2, vec );
}

static __forceinline__ __device__ float3 optixTransformNormalFromWorldToObjectSpace( float3 normal )
{
    if( optixGetTransformListSize() == 0 )
        return normal;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );  // inverse of optixGetWorldToObjectTransformMatrix()
    return optix_impl::optixTransformNormal( m0, m1, m2, normal );
}

static __forceinline__ __device__ float3 optixTransformPointFromObjectToWorldSpace( float3 point )
{
    if( optixGetTransformListSize() == 0 )
        return point;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformPoint( m0, m1, m2, point );
}

static __forceinline__ __device__ float3 optixTransformVectorFromObjectToWorldSpace( float3 vec )
{
    if( optixGetTransformListSize() == 0 )
        return vec;

    float4 m0, m1, m2;
    optix_impl::optixGetObjectToWorldTransformMatrix( m0, m1, m2 );
    return optix_impl::optixTransformVector( m0, m1, m2, vec );
}

static __forceinline__ __device__ float3 optixTransformNormalFromObjectToWorldSpace( float3 normal )
{
    if( optixGetTransformListSize() == 0 )
        return normal;

    float4 m0, m1, m2;
    optix_impl::optixGetWorldToObjectTransformMatrix( m0, m1, m2 );  // inverse of optixGetObjectToWorldTransformMatrix()
    return optix_impl::optixTransformNormal( m0, m1, m2, normal );
}

static __forceinline__ __device__ unsigned int optixGetTransformListSize()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_transform_list_size, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ OptixTraversableHandle optixGetTransformListHandle( unsigned int index )
{
    unsigned long long u0;
    asm( "call (%0), _optix_get_transform_list_handle, (%1);" : "=l"( u0 ) : "r"( index ) : );
    return u0;
}

static __forceinline__ __device__ OptixTransformType optixGetTransformTypeFromHandle( OptixTraversableHandle handle )
{
    int i0;
    asm( "call (%0), _optix_get_transform_type_from_handle, (%1);" : "=r"( i0 ) : "l"( handle ) : );
    return (OptixTransformType)i0;
}

static __forceinline__ __device__ const OptixStaticTransform* optixGetStaticTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_static_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixStaticTransform*)ptr;
}

static __forceinline__ __device__ const OptixSRTMotionTransform* optixGetSRTMotionTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_srt_motion_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixSRTMotionTransform*)ptr;
}

static __forceinline__ __device__ const OptixMatrixMotionTransform* optixGetMatrixMotionTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_matrix_motion_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const OptixMatrixMotionTransform*)ptr;
}

static __forceinline__ __device__ unsigned int optixGetInstanceIdFromHandle( OptixTraversableHandle handle )
{
    int i0;
    asm( "call (%0), _optix_get_instance_id_from_handle, (%1);" : "=r"( i0 ) : "l"( handle ) : );
    return i0;
}

static __forceinline__ __device__ const float4* optixGetInstanceTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_instance_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const float4*)ptr;
}

static __forceinline__ __device__ const float4* optixGetInstanceInverseTransformFromHandle( OptixTraversableHandle handle )
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_instance_inverse_transform_from_handle, (%1);" : "=l"( ptr ) : "l"( handle ) : );
    return (const float4*)ptr;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_0"
        ", (%1, %2);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_1"
        ", (%1, %2, %3);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_2"
        ", (%1, %2, %3, %4);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float hitT, unsigned int hitKind, unsigned int a0, unsigned int a1, unsigned int a2 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_3"
        ", (%1, %2, %3, %4, %5);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_4"
        ", (%1, %2, %3, %4, %5, %6);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_5"
        ", (%1, %2, %3, %4, %5, %6, %7);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_6"
        ", (%1, %2, %3, %4, %5, %6, %7, %8);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_7"
        ", (%1, %2, %3, %4, %5, %6, %7, %8, %9);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 ), "r"( a6 )
        : );
    return ret;
}

static __forceinline__ __device__ bool optixReportIntersection( float        hitT,
                                                                unsigned int hitKind,
                                                                unsigned int a0,
                                                                unsigned int a1,
                                                                unsigned int a2,
                                                                unsigned int a3,
                                                                unsigned int a4,
                                                                unsigned int a5,
                                                                unsigned int a6,
                                                                unsigned int a7 )
{
    int ret;
    asm volatile(
        "call (%0), _optix_report_intersection_8"
        ", (%1, %2, %3, %4, %5, %6, %7, %8, %9, %10);"
        : "=r"( ret )
        : "f"( hitT ), "r"( hitKind ), "r"( a0 ), "r"( a1 ), "r"( a2 ), "r"( a3 ), "r"( a4 ), "r"( a5 ), "r"( a6 ), "r"( a7 )
        : );
    return ret;
}

#define OPTIX_DEFINE_optixGetAttribute_BODY( which )                                                                   \
    unsigned int ret;                                                                                                  \
    asm( "call (%0), _optix_get_attribute_" #which ", ();" : "=r"( ret ) : );                                          \
    return ret;

static __forceinline__ __device__ unsigned int optixGetAttribute_0()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 0 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_1()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 1 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_2()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 2 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_3()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 3 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_4()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 4 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_5()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 5 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_6()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 6 );
}

static __forceinline__ __device__ unsigned int optixGetAttribute_7()
{
    OPTIX_DEFINE_optixGetAttribute_BODY( 7 );
}

#undef OPTIX_DEFINE_optixGetAttribute_BODY

static __forceinline__ __device__ void optixTerminateRay()
{
    asm volatile( "call _optix_terminate_ray, ();" );
}

static __forceinline__ __device__ void optixIgnoreIntersection()
{
    asm volatile( "call _optix_ignore_intersection, ();" );
}

static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_primitive_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetSbtGASIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_sbt_gas_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetInstanceId()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_id, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetInstanceIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetHitKind()
{
    unsigned int u0;
    asm( "call (%0), _optix_get_hit_kind, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType(unsigned int hitKind)
{
    unsigned int u0;
    asm( "call (%0), _optix_get_primitive_type_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    return (OptixPrimitiveType)u0;
}

static __forceinline__ __device__ bool optixIsBackFaceHit( unsigned int hitKind )
{
    unsigned int u0;
    asm( "call (%0), _optix_get_backface_from_hit_kind, (%1);" : "=r"( u0 ) : "r"( hitKind ) );
    return (u0 == 0x1);
}

static __forceinline__ __device__ bool optixIsFrontFaceHit( unsigned int hitKind )
{
    return !optixIsBackFaceHit( hitKind );
}


static __forceinline__ __device__ OptixPrimitiveType optixGetPrimitiveType()
{
    return optixGetPrimitiveType( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsBackFaceHit()
{
    return optixIsBackFaceHit( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsFrontFaceHit()
{
    return optixIsFrontFaceHit( optixGetHitKind() );
}

static __forceinline__ __device__ bool optixIsTriangleHit()
{
    return optixIsTriangleFrontFaceHit() || optixIsTriangleBackFaceHit();
}

static __forceinline__ __device__ bool optixIsTriangleFrontFaceHit()
{
    return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE;
}

static __forceinline__ __device__ bool optixIsTriangleBackFaceHit()
{
    return optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE;
}

static __forceinline__ __device__ float optixGetCurveParameter()
{
    return __int_as_float( optixGetAttribute_0() );
}

static __forceinline__ __device__ float2 optixGetTriangleBarycentrics()
{
    float f0, f1;
    asm( "call (%0, %1), _optix_get_triangle_barycentrics, ();" : "=f"( f0 ), "=f"( f1 ) : );
    return make_float2( f0, f1 );
}

static __forceinline__ __device__ uint3 optixGetLaunchIndex()
{
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_index_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_index_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_index_z, ();" : "=r"( u2 ) : );
    return make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ uint3 optixGetLaunchDimensions()
{
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_dimension_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_dimension_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_dimension_z, ();" : "=r"( u2 ) : );
    return make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ CUdeviceptr optixGetSbtDataPointer()
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_sbt_data_ptr_64, ();" : "=l"( ptr ) : );
    return (CUdeviceptr)ptr;
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode )
{
    asm volatile(
        "call _optix_throw_exception_0, (%0);"
        : /* no return value */
        : "r"( exceptionCode )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0 )
{
    asm volatile(
        "call _optix_throw_exception_1, (%0, %1);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1 )
{
    asm volatile(
        "call _optix_throw_exception_2, (%0, %1, %2);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2 )
{
    asm volatile(
        "call _optix_throw_exception_3, (%0, %1, %2, %3);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3 )
{
    asm volatile(
        "call _optix_throw_exception_4, (%0, %1, %2, %3, %4);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4 )
{
    asm volatile(
        "call _optix_throw_exception_5, (%0, %1, %2, %3, %4, %5);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5 )
{
    asm volatile(
        "call _optix_throw_exception_6, (%0, %1, %2, %3, %4, %5, %6);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5, unsigned int exceptionDetail6 )
{
    asm volatile(
        "call _optix_throw_exception_7, (%0, %1, %2, %3, %4, %5, %6, %7);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 ), "r"( exceptionDetail6 )
        : );
}

static __forceinline__ __device__ void optixThrowException( int exceptionCode, unsigned int exceptionDetail0, unsigned int exceptionDetail1, unsigned int exceptionDetail2, unsigned int exceptionDetail3, unsigned int exceptionDetail4, unsigned int exceptionDetail5, unsigned int exceptionDetail6, unsigned int exceptionDetail7 )
{
    asm volatile(
        "call _optix_throw_exception_8, (%0, %1, %2, %3, %4, %5, %6, %7, %8);"
        : /* no return value */
        : "r"( exceptionCode ), "r"( exceptionDetail0 ), "r"( exceptionDetail1 ), "r"( exceptionDetail2 ), "r"( exceptionDetail3 ), "r"( exceptionDetail4 ), "r"( exceptionDetail5 ), "r"( exceptionDetail6 ), "r"( exceptionDetail7 )
        : );
}

static __forceinline__ __device__ int optixGetExceptionCode()
{
    int s0;
    asm( "call (%0), _optix_get_exception_code, ();" : "=r"( s0 ) : );
    return s0;
}

#define OPTIX_DEFINE_optixGetExceptionDetail_BODY( which )                                                             \
    unsigned int ret;                                                                                                  \
    asm( "call (%0), _optix_get_exception_detail_" #which ", ();" : "=r"( ret ) : );                                   \
    return ret;

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_0()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 0 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_1()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 1 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_2()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 2 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_3()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 3 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_4()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 4 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_5()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 5 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_6()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 6 );
}

static __forceinline__ __device__ unsigned int optixGetExceptionDetail_7()
{
    OPTIX_DEFINE_optixGetExceptionDetail_BODY( 7 );
}

#undef OPTIX_DEFINE_optixGetExceptionDetail_BODY

static __forceinline__ __device__ OptixTraversableHandle optixGetExceptionInvalidTraversable()
{
    unsigned long long handle;
    asm( "call (%0), _optix_get_exception_invalid_traversable, ();" : "=l"( handle ) : );
    return (OptixTraversableHandle)handle;
}

static __forceinline__ __device__ int optixGetExceptionInvalidSbtOffset()
{
    int s0;
    asm( "call (%0), _optix_get_exception_invalid_sbt_offset, ();" : "=r"( s0 ) : );
    return s0;
}

static __forceinline__ __device__ OptixInvalidRayExceptionDetails optixGetExceptionInvalidRay()
{
    float rayOriginX, rayOriginY, rayOriginZ, rayDirectionX, rayDirectionY, rayDirectionZ, tmin, tmax, rayTime;
    asm( "call (%0, %1, %2, %3, %4, %5, %6, %7, %8), _optix_get_exception_invalid_ray, ();"
         : "=f"( rayOriginX ), "=f"( rayOriginY ), "=f"( rayOriginZ ), "=f"( rayDirectionX ), "=f"( rayDirectionY ),
           "=f"( rayDirectionZ ), "=f"( tmin ), "=f"( tmax ), "=f"( rayTime )
         : );
    OptixInvalidRayExceptionDetails ray;
    ray.origin    = make_float3( rayOriginX, rayOriginY, rayOriginZ );
    ray.direction = make_float3( rayDirectionX, rayDirectionY, rayDirectionZ );
    ray.tmin      = tmin;
    ray.tmax      = tmax;
    ray.time      = rayTime;
    return ray;
}

static __forceinline__ __device__ OptixParameterMismatchExceptionDetails optixGetExceptionParameterMismatch()
{
    unsigned int expected, actual, sbtIdx;
    unsigned long long calleeName;
    asm(
        "call (%0, %1, %2, %3), _optix_get_exception_parameter_mismatch, ();"
        : "=r"(expected), "=r"(actual), "=r"(sbtIdx), "=l"(calleeName) : );
    OptixParameterMismatchExceptionDetails details;
    details.expectedParameterCount = expected;
    details.passedArgumentCount = actual;
    details.sbtIndex = sbtIdx;
    details.callableName = (char*)calleeName;
    return details;
}

static __forceinline__ __device__ char* optixGetExceptionLineInfo()
{
    unsigned long long ptr;
    asm( "call (%0), _optix_get_exception_line_info, ();" : "=l"(ptr) : );
    return (char*)ptr;
}

template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixDirectCall( unsigned int sbtIndex, ArgTypes... args )
{
    unsigned long long func;
    asm( "call (%0), _optix_call_direct_callable,(%1);" : "=l"( func ) : "r"( sbtIndex ) : );
    using funcT = ReturnT ( * )( ArgTypes... );
    funcT call  = ( funcT )( func );
    return call( args... );
}

template <typename ReturnT, typename... ArgTypes>
static __forceinline__ __device__ ReturnT optixContinuationCall( unsigned int sbtIndex, ArgTypes... args )
{
    unsigned long long func;
    asm( "call (%0), _optix_call_continuation_callable,(%1);" : "=l"( func ) : "r"( sbtIndex ) : );
    using funcT = ReturnT ( * )( ArgTypes... );
    funcT call  = ( funcT )( func );
    return call( args... );
}
#endif
