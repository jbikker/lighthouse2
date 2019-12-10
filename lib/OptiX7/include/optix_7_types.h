
/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header
///
/// OptiX types include file -- defines types and enums used by the API.
/// For the math library routines include optix_math.h

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_types.h is an internal header file and must not be used directly.  Please use optix_types.h, optix_host.h, optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_types_h__
#define __optix_optix_7_types_h__

#if 1
// JB: prevent inclusion of stddef.h so we don't have to search vs directories.
// typedef unsigned long long size_t;
#else
#include <stddef.h> /* for size_t */
#endif

// This typedef should match the one in cuda.h in order to avoid compilation errors.
#if defined( _WIN64 ) || defined( __LP64__ )
/// CUDA device pointer
typedef unsigned long long CUdeviceptr;
#else
/// CUDA device pointer
typedef unsigned int CUdeviceptr;
#endif

/// Opaque type representing a device context
typedef struct OptixDeviceContext_t* OptixDeviceContext;

/// Opaque type representing a module
typedef struct OptixModule_t* OptixModule;

/// Opaque type representing a program group
typedef struct OptixProgramGroup_t* OptixProgramGroup;

/// Opaque type representing a pipeline
typedef struct OptixPipeline_t* OptixPipeline;

/// Opaque type representing a denoiser instance
typedef struct OptixDenoiser_t* OptixDenoiser;

/// Traversable handle
typedef unsigned long long OptixTraversableHandle;

/// Visibility mask
typedef unsigned int OptixVisibilityMask;

/// Size of the SBT record headers.
#define OPTIX_SBT_RECORD_HEADER_SIZE ( (size_t)32 )

/// Alignment requirement for device pointers in OptixShaderBindingTable.
#define OPTIX_SBT_RECORD_ALIGNMENT 16ull

/// Alignment requirement for output and temporay buffers for acceleration structures.
#define OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT 128ull

/// Alignment requirement for OptixBuildInputInstanceArray::instances.
#define OPTIX_INSTANCE_BYTE_ALIGNMENT 16ull

/// Alignment requirement for OptixBuildInputCustomPrimitiveArray::aabbBuffers and OptixBuildInputInstanceArray::aabbs.
#define OPTIX_AABB_BUFFER_BYTE_ALIGNMENT 8ull

/// Alignment requirement for OptixBuildInputTriangleArray::preTransform
#define OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT 16ull

/// Alignment requirement for OptixStaticTransform, OptixMatrixMotionTransform, OptixSRTMotionTransform.
#define OPTIX_TRANSFORM_BYTE_ALIGNMENT 64ull

/// Maximum number of registers allowed. Defaults to no explicit limit.
#define OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0


/// Result codes
typedef enum OptixResult
{
    OPTIX_SUCCESS                               = 0,
    OPTIX_ERROR_INVALID_VALUE                   = 7001,
    OPTIX_ERROR_HOST_OUT_OF_MEMORY              = 7002,
    OPTIX_ERROR_INVALID_OPERATION               = 7003,
    OPTIX_ERROR_FILE_IO_ERROR                   = 7004,
    OPTIX_ERROR_INVALID_FILE_FORMAT             = 7005,
    OPTIX_ERROR_DISK_CACHE_INVALID_PATH         = 7010,
    OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR     = 7011,
    OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR       = 7012,
    OPTIX_ERROR_DISK_CACHE_INVALID_DATA         = 7013,
    OPTIX_ERROR_LAUNCH_FAILURE                  = 7050,
    OPTIX_ERROR_INVALID_DEVICE_CONTEXT          = 7051,
    OPTIX_ERROR_CUDA_NOT_INITIALIZED            = 7052,
    OPTIX_ERROR_INVALID_PTX                     = 7200,
    OPTIX_ERROR_INVALID_LAUNCH_PARAMETER        = 7201,
    OPTIX_ERROR_INVALID_PAYLOAD_ACCESS          = 7202,
    OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS        = 7203,
    OPTIX_ERROR_INVALID_FUNCTION_USE            = 7204,
    OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS      = 7205,
    OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY = 7250,
    OPTIX_ERROR_PIPELINE_LINK_ERROR             = 7251,
    OPTIX_ERROR_INTERNAL_COMPILER_ERROR         = 7299,
    OPTIX_ERROR_DENOISER_MODEL_NOT_SET          = 7300,
    OPTIX_ERROR_DENOISER_NOT_INITIALIZED        = 7301,
    OPTIX_ERROR_ACCEL_NOT_COMPATIBLE            = 7400,
    OPTIX_ERROR_NOT_SUPPORTED                   = 7800,
    OPTIX_ERROR_UNSUPPORTED_ABI_VERSION         = 7801,
    OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH    = 7802,
    OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS  = 7803,
    OPTIX_ERROR_LIBRARY_NOT_FOUND               = 7804,
    OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND          = 7805,
    OPTIX_ERROR_CUDA_ERROR                      = 7900,
    OPTIX_ERROR_INTERNAL_ERROR                  = 7990,
    OPTIX_ERROR_UNKNOWN                         = 7999,
} OptixResult;

/// Parameters used for optixDeviceContextGetProperty
typedef enum OptixDeviceProperty
{
    /// Maximum value for OptixPipelineLinkOptions::maxTraceDepth. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH = 0x2001,

    /// Maximum value to pass into optixPipelineSetStackSize for parameter
    /// maxTraversableGraphDepth.v sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH = 0x2002,

    /// The maximum number of primitives (over all build inputs) as input to a single
    /// Geometry Acceleration Structure (GAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS = 0x2003,

    /// The maximum number for the sum of the number of SBT records of all
    /// build inputs to a single Geometry Acceleration Structure (GAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS = 0x2004,

    /// The RT core version supported by the device (0 for no support, 10 for version
    /// 1.0). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_RTCORE_VERSION = 0x2005,

    /// The maximum value for OptixInstance::instanceId. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID = 0x2006,

    /// The number of bits available for the OptixInstance::visibilityMask.
    /// Higher bits must be set to zero. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK = 0x2007,

    /// The maximum number of instances that can be added to a single Instance
    /// Acceleration Structure (IAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS = 0x2008,

    /// The maximum value for OptixInstance::sbtOffset. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET = 0x2009,
} OptixDeviceProperty;

/// Type of the callback function used for log messages.
///
/// \param[in] level      The log level indicates the severity of the message. See below for
///                       possible values.
/// \param[in] tag        A terse message category description (e.g., 'SCENE STAT').
/// \param[in] message    Null terminated log message (without newline at the end).
/// \param[in] cbdata     Callback data that was provided with the callback pointer.
///
/// It is the users responsibility to ensure thread safety within this function.
///
/// The following log levels are defined.
///
///   0   disable   Setting the callback level will disable all messages.  The callback
///                 function will not be called in this case.
///   1   fatal     A non-recoverable error. The context and/or OptiX itself might no longer
///                 be in a usable state.
///   2   error     A recoverable error, e.g., when passing invalid call parameters.
///   3   warning   Hints that OptiX might not behave exactly as requested by the user or
///                 may perform slower than expected.
///   4   print     Status or progress messages.
///
/// Higher levels might occur.
typedef void ( *OptixLogCallback )( unsigned int level, const char* tag, const char* message, void* cbdata );

typedef struct OptixDeviceContextOptions
{
    OptixLogCallback logCallbackFunction;
    void*            logCallbackData;
    int              logCallbackLevel;
} OptixDeviceContextOptions;

/// Flags used for OptixBuildInputTriangleArray and OptixBuildInputCustomPrimitiveArray.
typedef enum OptixGeometryFlags
{
    /// No flags set
    OPTIX_GEOMETRY_FLAG_NONE = 0,

    /// Disables the invocation of the anyhit program.
    /// Can be overridden by OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT and OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT = 1u << 0,

    /// If set, an intersection with the primitive will trigger one and only one
    /// invocation of the anyhit program.  Otherwise, the anyhit program may be invoked
    /// more than once.
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1u << 1
} OptixGeometryFlags;

typedef enum OptixHitKind
{
    OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE = 0xFE,
    OPTIX_HIT_KIND_TRIANGLE_BACK_FACE  = 0xFF
} OptixHitKind;

typedef enum OptixIndicesFormat
{
    OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 = 0x2102,
    OPTIX_INDICES_FORMAT_UNSIGNED_INT3   = 0x2103
} OptixIndicesFormat;

/// Triangle inputs
typedef enum OptixVertexFormat
{
    OPTIX_VERTEX_FORMAT_FLOAT3    = 0x2121,
    OPTIX_VERTEX_FORMAT_FLOAT2    = 0x2122,
    OPTIX_VERTEX_FORMAT_HALF3     = 0x2123,
    OPTIX_VERTEX_FORMAT_HALF2     = 0x2124,
    OPTIX_VERTEX_FORMAT_SNORM16_3 = 0x2125,
    OPTIX_VERTEX_FORMAT_SNORM16_2 = 0x2126
} OptixVertexFormat;

typedef struct OptixBuildInputTriangleArray
{
    /// Pointer to host array of device pointers.
    /// The host array's size must match number of motion keys as set
    /// in OptixMotionOptions if motion is used.
    /// Must be of size one otherwise (if OptixMotionOptions::numKeys == 0).
    /// Each per motion key-device pointer must point to an array of floats
    /// (the vertices of the triangles).
    const CUdeviceptr* vertexBuffers;

    unsigned int numVertices;

    OptixVertexFormat vertexFormat;

    /// Stride between vertices. If set to zero, vertices are assumed to be tightly
    /// packed and stride is inferred from vertexFormat.
    unsigned int vertexStrideInBytes;

    /// Optional pointer to array of 16 or 32-bit int triplets, one triplet per triangle.
    CUdeviceptr indexBuffer;

    // Needs to be zero if indexBuffer==nullptr
    unsigned int numIndexTriplets;

    OptixIndicesFormat indexFormat;

    /// Stride between triplets of indices. If set to zero, indices are assumed to be tightly
    /// packed and stride is inferred from indexFormat.
    unsigned int indexStrideInBytes;

    /// Optional pointer to array of floats
    /// representing a 3x4 row major affine
    /// transformation matrix. This pointer must be a multiple of OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT
    CUdeviceptr preTransform;

    /// Array of flags, to specify flags per sbt record,
    /// combinations of OptixGeometryFlags describing the
    /// primitive behavior, size must match numSbtRecords
    const unsigned int* flags;

    /// Number of sbt records available to the sbt index offset override.
    unsigned int numSbtRecords;

    /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
    /// Every entry must be in range [0,numSbtRecords-1].
    /// Size needs to be the number of primitives.
    CUdeviceptr sbtIndexOffsetBuffer;

    /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
    unsigned int sbtIndexOffsetSizeInBytes;

    /// Stride between the index offsets. If set to zero, the offsets are assumed to be tightly
    /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
    unsigned int sbtIndexOffsetStrideInBytes;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of triangles must not overflow 32bits.
    unsigned int primitiveIndexOffset;
} OptixBuildInputTriangleArray;

/// AABB inputs
typedef struct OptixAabb
{
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
} OptixAabb;

typedef struct OptixBuildInputCustomPrimitiveArray
{
    /// Points to host array of device pointers to AABBs (type OptixAabb) per motion step.
    /// (host array size must match number of motion keys as set in OptixMotionOptions).
    /// Each device pointer must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.
    const CUdeviceptr* aabbBuffers;
    unsigned int       numPrimitives;

    /// Stride between AABBs (per motion key). If set to zero, the aabbs are assumed to be tightly
    /// packed and the stride is assumed to be sizeof( OptixAabb ).
    unsigned int strideInBytes;

    /// Array of flags, to specify flags per sbt record,
    /// combinations of OptixGeometryFlags describing the
    /// primitive behavior, size must match numSbtRecords
    const unsigned int* flags;

    /// Number of sbt records available to the sbt index offset override.
    unsigned int numSbtRecords;

    /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
    /// Every entry must be in range [0,numSbtRecords-1].
    /// Size needs to be the number of primitives.
    CUdeviceptr sbtIndexOffsetBuffer;

    /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
    unsigned int sbtIndexOffsetSizeInBytes;

    /// Stride between the index offsets. If set to zero, the offsets are assumed to be tightly
    /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
    unsigned int sbtIndexOffsetStrideInBytes;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of primitive must not overflow 32bits.
    unsigned int primitiveIndexOffset;
} OptixBuildInputCustomPrimitiveArray;

/// Instance inputs
typedef struct OptixBuildInputInstanceArray
{
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS instances and
    /// aabbs should be interpreted as arrays of pointers instead of arrays of structs.
    ///
    /// This pointer must be a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT if
    /// OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCES. The array elements must
    /// be a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT if OptixBuildInput::type is
    /// OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS.
    CUdeviceptr  instances;
    unsigned int numInstances;

    /// Optional AABBs.  In OptixAabb format.
    ///
    /// Required for traversables (OptixMatrixMotionTransform, OptixSRTMotionTransform,
    /// OptixStaticTransform) and certain configurations of motion AS as instance.
    /// Will be ignored for non-motion AS, since no AABBs are required. May be NULL in that case.
    ///
    /// The following table illustrates this (IAS is Instance Acceleration Structure)
    /// instance type          |  traversable | motion AS | static AS
    /// building a motion  IAS |   required   | required  | ignored
    /// building a static  IAS |   required   | ignored   | ignored
    ///
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS the unused
    /// pointers for unused aabbs may be set to NULL.
    ///
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCES this pointer must be a
    /// multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.  If OptixBuildInput::type is
    /// OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS the array elements must be a multiple of
    /// OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.
    ///
    /// Motion:
    ///
    /// In case of motion (OptixMotionOptions::numKeys>=2), OptixMotionOptions::numKeys
    /// aabbs are expected per instance, e.g., for N instances and M motion keys:
    /// aabb[inst0][t0], aabb[inst0][t1], ..., aabb[inst0][tM-1], ..., aabb[instN-1][t0],
    /// aabb[instN-1][t1],..., aabb[instN-1][tM-1].
    ///
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCES aabbs must be a device
    /// pointer to an array of N * M * 6 floats.
    ///
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS aabbs must be
    /// a device pointer to an array of N device pointers, each pointing to an array of M *
    /// 6 floats in OptixAabb format.  Pointers may be NULL if the aabbs are not required.
    /// Hence, if the second instance (inst1) points to a static GAS, aabbs are not
    /// required for that instance.  While being ignored, aabbs must still be a device
    /// pointer to an array of N elements.
    ///
    /// In case of OPTIX_BUILD_INPUT_TYPE_INSTANCES, the second element (with a size of M *
    /// 6 floats) will be ignored.  In case of OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS,
    /// the second element (with a size of pointer to M * 6 floats) can be NULL.
    CUdeviceptr aabbs;

    /// number of aabbs, in case of motion, this needs to match
    /// numInstances multiplied with OptixMotionOptions::numKeys
    unsigned int numAabbs;

} OptixBuildInputInstanceArray;

typedef enum OptixBuildInputType
{
    OPTIX_BUILD_INPUT_TYPE_TRIANGLES         = 0x2141,
    OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES = 0x2142,
    OPTIX_BUILD_INPUT_TYPE_INSTANCES         = 0x2143,
    OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS = 0x2144
} OptixBuildInputType;

typedef struct OptixBuildInput
{
    OptixBuildInputType type;
    union
    {
        /// all of them support motion and the size of the data arrays needs to match the number of motion steps
        OptixBuildInputTriangleArray        triangleArray;
        OptixBuildInputCustomPrimitiveArray aabbArray;
        OptixBuildInputInstanceArray        instanceArray;
        char                                pad[1024];  // TODO Why is the padding so huge? Why is there padding at all?
        // Why is there an explicit padding, but not for other types like  OptixProgramGroupDesc?
    };
} OptixBuildInput;

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixBuildInput ) == 8 + 1024, "OptixBuildInput has wrong size" );
#endif

/// Flags set on the OptixInstance::flags.
///
/// These can be or'ed together to combine multiple flags.
typedef enum OptixInstanceFlags
{
    /// No special flag set
    OPTIX_INSTANCE_FLAG_NONE = 0,

    /// Prevent triangles from getting culled due to their orientation.
    /// Effectively ignores ray flags
    /// OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES and OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES.
    OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING = 1u << 0,

    /// Flip triangle orientation.
    /// This affects front/backface culling as well as the reported face in case of a hit.
    OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1,

    /// Disable anyhit programs for all geometries of the instance.
    /// Can be overridden by OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT.
    OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2,

    /// Enables anyhit programs for all geometries of the instance.
    /// Overrides OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    /// Can be overridden by OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT.
    OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3,

    /// Disable the instance transformation
    OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM = 1u << 6,
} OptixInstanceFlags;

typedef struct OptixInstance
{
    /// affine world-to-object transformation as 3x4 matrix in row-major layout
    float transform[12];

    /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
    unsigned int instanceId;

    /// SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
    /// Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. The maximal SBT offset
    /// can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.
    unsigned int sbtOffset;

    /// Visibility mask. If rayMask & instanceMask == 0 the instance is culled. The number of available bits can be
    /// queried using OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
    unsigned int visibilityMask;

    /// Any combination of OptixInstanceFlags is allowed.
    unsigned int flags;

    /// Set with an OptixTraversableHandle.
    OptixTraversableHandle traversableHandle;

    /// round up to 80-byte, to ensure 16-byte alignment
    unsigned int pad[2];
} OptixInstance;

// Builder Options

/// Used for OptixAccelBuildOptions::buildFlags. Can be or'ed together.
typedef enum OptixBuildFlags
{
    /// No special flags set.
    OPTIX_BUILD_FLAG_NONE = 0,

    /// Allow updating the build with new vertex positions with subsequent calls to
    /// optixAccelBuild.
    OPTIX_BUILD_FLAG_ALLOW_UPDATE = 1u << 0,

    OPTIX_BUILD_FLAG_ALLOW_COMPACTION = 1u << 1,

    OPTIX_BUILD_FLAG_PREFER_FAST_TRACE = 1u << 2,

    OPTIX_BUILD_FLAG_PREFER_FAST_BUILD = 1u << 3,

    /// Allow access to random baked vertex in closesthit
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS = 1u << 4,
} OptixBuildFlags;

typedef enum OptixBuildOperation
{
    OPTIX_BUILD_OPERATION_BUILD  = 0x2161,
    OPTIX_BUILD_OPERATION_UPDATE = 0x2162,
} OptixBuildOperation;

typedef enum OptixMotionFlags
{
    OPTIX_MOTION_FLAG_NONE         = 0,
    OPTIX_MOTION_FLAG_START_VANISH = 1u << 0,
    OPTIX_MOTION_FLAG_END_VANISH   = 1u << 1
} OptixMotionFlags;

typedef struct OptixMotionOptions
{
    /// If numKeys > 1, motion is enabled. timeBegin,
    /// timeEnd and flags are all ignored when motion is disabled.
    unsigned short numKeys;

    /// Combinations of OptixMotionFlags
    unsigned short flags;

    float timeBegin;

    float timeEnd;
} OptixMotionOptions;

typedef struct OptixAccelBuildOptions
{
    /// Combinations of OptixBuildFlags
    unsigned int buildFlags;

    /// If OPTIX_BUILD_OPERATION_UPDATE the output buffer is assumed to contain the result
    /// of a full build with OPTIX_BUILD_FLAG_ALLOW_UPDATE set and using the same number of
    /// primitives.  It is updated incrementally to reflect the current position of the
    /// primitives.
    OptixBuildOperation operation;

    OptixMotionOptions motionOptions;
} OptixAccelBuildOptions;

/// Builder memory usage (CPU synchronous)
typedef struct OptixAccelBufferSizes
{
    size_t outputSizeInBytes;

    /// size of temp buffer required for building the bvh
    size_t tempSizeInBytes;

    /// size of temp buffer required for updating the bvh, only non-zero if Allow_Update flag is set
    size_t tempUpdateSizeInBytes;
} OptixAccelBufferSizes;

/// Build (Async)
typedef enum OptixAccelPropertyType
{
    /// uint64
    OPTIX_PROPERTY_TYPE_COMPACTED_SIZE = 0x2181,

    /// OptixAabb * numMotionSteps
    OPTIX_PROPERTY_TYPE_AABBS = 0x2182,
} OptixAccelPropertyType;

/// Specifies a type and output destination for emitted post-build properties.
typedef struct OptixAccelEmitDesc
{
    /// output buffer for the properties
    CUdeviceptr result;

    /// type of information requested
    OptixAccelPropertyType type;
} OptixAccelEmitDesc;

/// Used with optixAccelGetRelocationInfo and optixAccelCheckRelocationCompatibility.  The
/// values are used internally to OptiX and should not be modified.
typedef struct OptixAccelRelocationInfo
{
    unsigned long long info[4];
} OptixAccelRelocationInfo;


// Traversables


/// Static transform
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
typedef struct OptixStaticTransform
{
    OptixTraversableHandle child;

    /// padding to make the transformations 16 byte aligned
    unsigned int pad[2];

    /// affine world-to-object transformation as 3x4 matrix in row-major layout
    float transform[12];

    /// affine object-to-world transformation as 3x4 matrix in row-major layout
    float invTransform[12];
} OptixStaticTransform;

/// Represents a matrix motion transformation.
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
///
/// This struct, as defined here, handles only N=2 motion keys due to the fixed array length of its transform member.
/// The following example shows how to create instances for an arbitrary number N of motion keys:
///
/// \code
/// float matrixData[N][12];
/// ... // setup matrixData
///
/// size_t transformSizeInBytes = sizeof( OptixMatrixMotionTransform ) + ( N-2 ) * 12 * sizeof( float );
/// OptixMatrixMotionTransform* matrixMoptionTransform = (OptixMatrixMotionTransform*) malloc( transformSizeInBytes );
/// memset( matrixMoptionTransform, 0, transformSizeInBytes );
///
/// ... // setup other members of matrixMoptionTransform
/// matrixMoptionTransform->motionOptions.numKeys/// = N;
/// memcpy( matrixMoptionTransform->transform, matrixData, N * 12 * sizeof( float ) );
///
/// ... // copy matrixMoptionTransform to device memory
/// free( matrixMoptionTransform )
/// \endcode
typedef struct OptixMatrixMotionTransform
{
    OptixTraversableHandle child;
    OptixMotionOptions     motionOptions;

    /// padding to make the transformation 16 byte aligned
    unsigned int pad[3];

    /// affine object-to-world transformation as 3x4 matrix in row-major layout
    float transform[2][12];
} OptixMatrixMotionTransform;

/// Represents an SRT transformation.
///
/// An SRT transformation can represent a smooth rotation with fewer motion keys than a matrix transformation. Each
/// motion key is constructed from elements taken from a matrix S, a quaternion R, and a translation T.
///
///                        [  sx   a   b  pvx ]
/// The scaling matrix S = [   0  sy   c  pvy ] defines an affine transformation that can include scale, shear, and a
///                        [   0   0  sz  pvz ]
///
/// translation. The translation allows to define the pivot point for the subsequent rotation.
///
/// The quaternion R = [ qx, qy, qz, qw ] describes a rotation  with angular component qw = cos(theta/2) and other
/// components [ qx, qy, qz ] = sin(theta/2) * [ ax, ay, az ] where the axis [ ax, ay, az ] is normalized.
///
///                     [  1  0  0 tx ]
/// The translation T = [  0  1  0 ty ] defines another translation that is applied after the rotation. Typically, this
///                     [  0  0  1 tz ]
///
/// translation includes the inverse translation from the matrix S to reverse its effect.
///
/// To obtain the effective transformation at time t, the elements of the components of S, R, and T will be interpolated
/// linearly. The components are then multiplied to obtain the combined transformation C = T * R * S. The transformation
/// C is the effective object-to-world transformations at time t, and C^(-1) is the effective world-to-object
/// transformation at time t. */
typedef struct OptixSRTData
{
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
} OptixSRTData;

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixSRTData ) == 16 * 4, "OptixSRTData has wrong size" );
#endif

/// Represents an SRT motion transformation.
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
///
/// This struct, as defined here, handles only N=2 motion keys due to the fixed array length of its srtData member.
/// The following example shows how to create instances for an arbitrary number N of motion keys:
///
/// \code
/// OptixSRTData srtData[N];
/// ... // setup srtData
///
/// size_t transformSizeInBytes = sizeof( OptixSRTMotionTransform ) + ( N-2 ) * sizeof( OptixSRTData );
/// OptixSRTMotionTransform* srtMotionTransform = (OptixSRTMotionTransform*) malloc( transformSizeInBytes );
/// memset( srtMotionTransform, 0, transformSizeInBytes );
///
/// ... // setup other members of srtMotionTransform
/// srtMotionTransform->motionOptions.numKeys   = N;
/// memcpy( srtMotionTransform->srtData, srtData, N * sizeof( OptixSRTData ) );
///
/// ... // copy srtMotionTransform to device memory
/// free( srtMotionTransform )
/// \endcode
typedef struct OptixSRTMotionTransform
{
    OptixTraversableHandle child;
    OptixMotionOptions     motionOptions;
    /// padding to make the SRT data 16 byte aligned
    unsigned int pad[3];
    OptixSRTData srtData[2];
} OptixSRTMotionTransform;

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixSRTMotionTransform ) == 8 + 12 + 12 + 2 * 16 * 4, "OptixSRTMotionTransform has wrong size" );
#endif

/// Traversable Handles
typedef enum OptixTraversableType
{
    OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM        = 0x21C1,
    OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM = 0x21C2,
    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM    = 0x21C3,
} OptixTraversableType;


/// Images
typedef enum OptixPixelFormat
{
    OPTIX_PIXEL_FORMAT_HALF3  = 0x2201,
    OPTIX_PIXEL_FORMAT_HALF4  = 0x2202,
    OPTIX_PIXEL_FORMAT_FLOAT3 = 0x2203,
    OPTIX_PIXEL_FORMAT_FLOAT4 = 0x2204,
    OPTIX_PIXEL_FORMAT_UCHAR3 = 0x2205,
    OPTIX_PIXEL_FORMAT_UCHAR4 = 0x2206
} OptixPixelFormat;

typedef struct OptixImage2D
{
    CUdeviceptr  data;
    unsigned int width;
    unsigned int height;
    unsigned int rowStrideInBytes;
    /// either 0 or the value that corresponds to a dense packing of format
    unsigned int     pixelStrideInBytes;
    OptixPixelFormat format;
} OptixImage2D;

/// Denoiser
typedef enum OptixDenoiserInputKind
{
    OPTIX_DENOISER_INPUT_RGB               = 0x2301,
    OPTIX_DENOISER_INPUT_RGB_ALBEDO        = 0x2302,
    OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL = 0x2303,
} OptixDenoiserInputKind;

/// Used as the argument to optixDenoiserSetModel.
typedef enum OptixDenoiserModelKind
{
    /// Use the model provided by the associated pointer.  See the programming guide for a
    /// description of how to format the data.
    OPTIX_DENOISER_MODEL_KIND_USER = 0x2321,

    /// Use the built-in model appropriate for low dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_LDR = 0x2322,

    /// Use the built-in model appropriate for high dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_HDR = 0x2323,
} OptixDenoiserModelKind;

typedef struct OptixDenoiserOptions
{
    OptixDenoiserInputKind inputKind;
    OptixPixelFormat       pixelFormat;
} OptixDenoiserOptions;

typedef struct OptixDenoiserParams
{
    unsigned int denoiseAlpha;
    CUdeviceptr  hdrIntensity;
    float        blendFactor;
} OptixDenoiserParams;

typedef struct OptixDenoiserSizes
{
    size_t       stateSizeInBytes;
    size_t       minimumScratchSizeInBytes;
    size_t       recommendedScratchSizeInBytes;
    unsigned int overlapWindowSizeInPixels;
} OptixDenoiserSizes;

/// Ray flags passed to the device function optixTrace().  These affect the behavior of
/// traversal per invocation.
typedef enum OptixRayFlags
{
    /// No change from the behavior configured for the individual AS.
    OPTIX_RAY_FLAG_NONE = 0u,

    /// Disables anyhit programs for the ray. Overrides
    /// OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
    /// OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT, OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT.
    OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1u << 0,

    /// Forces anyhit program execution for the ray. Overrides
    /// OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as well as
    /// OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    /// OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT, OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT.
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1u << 1,

    /// Terminates the ray after the first hit and executes
    /// the closesthit program of that hit.
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1u << 2,

    /// Disables closesthit and miss programs for the ray.
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = 1u << 3,

    /// Do not intersect triangle back faces
    /// (respects a possible face change due to instance flag
    /// OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES.
    OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 1u << 4,

    /// Do not intersect triangle front faces
    /// (respects a possible face change due to instance flag
    /// OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES.
    OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 1u << 5,

    /// Do not intersect geometry which disables anyhit programs
    /// (due to setting geometry flag OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT or
    /// instance flag OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT,
    /// OPTIX_RAY_FLAG_ENFORCE_ANYHIT, OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT = 1u << 6,

    /// Do not intersect geometry which have an enabled anyhit program
    /// (due to not setting geometry flag OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT or
    /// setting instance flag OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_CULL_DISABLED_ANYHIT,
    /// OPTIX_RAY_FLAG_ENFORCE_ANYHIT, OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    OPTIX_RAY_FLAG_CULL_ENFORCED_ANYHIT = 1u << 7
} OptixRayFlags;

/// Transform
///
/// OptixTransformType is used by the device function optixGetTransformTypeFromHandle() to
/// determine the type of the OptixTraversableHandle returned from
/// optixGetTransformListHandle().
typedef enum OptixTransformType
{
    OPTIX_TRANSFORM_TYPE_NONE                    = 0,
    OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM        = 1,
    OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM = 2,
    OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM    = 3,
    OPTIX_TRANSFORM_TYPE_INSTANCE                = 4,
} OptixTransformType;

/// OptixTraversableGraphFlags specify the set of valid traversable graphs that may be
/// passed to invocation of optixTrace. Flags may be bitwise combined.
typedef enum OptixTraversableGraphFlags
{
    ///  Used to signal that any traversable graphs is valid.
    ///  This flag is mutually exclusive with all other flags.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0,

    ///  Used to signal that a traversable graph of a single Geometry Acceleration
    ///  Structure (GAS) without any transforms is valid. This flag may be combined with
    ///  other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0,

    ///  Used to signal that a traversable graph of a single Instance Acceleration
    ///  Structure (IAS) directly connected to Geometry Acceleration Structure (GAS)
    ///  traversables without transform traversables in between is valid.  This flag may
    ///  be combined with other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1,
} OptixTraversableGraphFlags;

typedef enum OptixCompileOptimizationLevel
{
    /// No optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 = 0,
    /// Some optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 = 1,
    /// Most optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 = 2,
    /// All optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 = 3,
    OPTIX_COMPILE_OPTIMIZATION_DEFAULT = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
} OptixCompileOptimizationLevel;

typedef enum OptixCompileDebugLevel
{
    /// no debug information
    OPTIX_COMPILE_DEBUG_LEVEL_NONE = 0,
    /// generate lineinfo only
    OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO = 1,
    /// generate dwarf debug information.
    OPTIX_COMPILE_DEBUG_LEVEL_FULL = 2,
} OptixCompileDebugLevel;

typedef struct OptixModuleCompileOptions
{
    /// Maximum number of registers allowed when compiling to SASS.
    /// Set to 0 for no explicit limit. May vary within a pipeline.
    int maxRegisterCount;

    /// Optimization level. May vary within a pipeline.
    OptixCompileOptimizationLevel optLevel;

    /// Generate debug information.
    OptixCompileDebugLevel debugLevel;
} OptixModuleCompileOptions;

typedef enum OptixProgramGroupKind
{
    OPTIX_PROGRAM_GROUP_KIND_RAYGEN    = 0x2421,
    OPTIX_PROGRAM_GROUP_KIND_MISS      = 0x2422,
    OPTIX_PROGRAM_GROUP_KIND_EXCEPTION = 0x2423,
    OPTIX_PROGRAM_GROUP_KIND_HITGROUP  = 0x2424,
    OPTIX_PROGRAM_GROUP_KIND_CALLABLES = 0x2425
} OptixProgramGroupKind;

typedef enum OptixProgramGroupFlags
{
    OPTIX_PROGRAM_GROUP_FLAGS_NONE = 0
} OptixProgramGroupFlags;

typedef struct OptixProgramGroupSingleModule
{
    OptixModule module;
    const char* entryFunctionName;
} OptixProgramGroupSingleModule;

typedef struct OptixProgramGroupHitgroup
{
    OptixModule moduleCH;
    const char* entryFunctionNameCH;
    OptixModule moduleAH;
    const char* entryFunctionNameAH;
    OptixModule moduleIS;
    const char* entryFunctionNameIS;
} OptixProgramGroupHitgroup;

typedef struct OptixProgramGroupCallables
{
    OptixModule moduleDC;
    const char* entryFunctionNameDC;
    OptixModule moduleCC;
    const char* entryFunctionNameCC;
} OptixProgramGroupCallables;

/// If kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN or kind == OPTIX_PROGRAM_GROUP_KIND_EXCEPTION, then the module and entry
/// function name need to be valid. If kind == OPTIX_PROGRAM_GROUP_KIND_CALLABLES, then the module and entryFunctionName
/// for at least one of the two callables types needs to be valid. For all other kinds, module and entryFunctionName
/// might both be nullptr.
typedef struct OptixProgramGroupDesc
{
    OptixProgramGroupKind kind;

    // See #OptixProgramGroupFlags
    unsigned int flags;

    union
    {
        OptixProgramGroupSingleModule raygen;
        OptixProgramGroupSingleModule miss;
        OptixProgramGroupSingleModule exception;
        OptixProgramGroupCallables    callables;
        OptixProgramGroupHitgroup     hitgroup;
    };
} OptixProgramGroupDesc;

typedef struct OptixProgramGroupOptions
{
    /// Currently no options, so include a placeholder
    int placeholder;
} OptixProgramGroupOptions;

/// The following values are used to indicate which exception was thrown.
typedef enum OptixExceptionCodes
{
    /// Stack overflow of the continuation stack.
    /// no exception details.
    OPTIX_EXCEPTION_CODE_STACK_OVERFLOW = -1,

    /// The trace depth is exceeded.
    /// no exception details.
    OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED = -2,

    /// The traversal depth is exceeded.
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED = -3,

    /// Traversal encountered an invalid traversable type.
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    ///     optixGetExceptionInvalidTraversable()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE = -5,

    /// The miss SBT record index is out of bounds
    /// Exception details:
    ///     optixGetExceptionInvalidSbtOffset()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_MISS_SBT = -6,

    /// The traversal hit SBT record index out of bounds.
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    ///     optixGetExceptionInvalidSbtOffset()
    ///     optixGetPrimitiveIndex()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT = -7,
} OptixExceptionCodes;

/// Set OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW and OPTIX_EXCEPTION_FLAG_TRACE_DEPTH to enable
/// generation of these runtime checks.
///
/// See documentation for OptixExceptionCodes for what exception codes are returned.
///
/// If OPTIX_EXCEPTION_FLAG_USER must be specified for all OptixModule objects in the
/// pipeline if any OptixModule calls optixThrow.
///
/// Set OPTIX_EXCEPTION_FLAG_DEBUG to enable all runtime checks.
typedef enum OptixExceptionFlags
{
    OPTIX_EXCEPTION_FLAG_NONE           = 0,
    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW = 1u << 0,
    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH    = 1u << 1,
    OPTIX_EXCEPTION_FLAG_USER           = 1u << 2,
    OPTIX_EXCEPTION_FLAG_DEBUG          = 1u << 3
} OptixExceptionFlags;

typedef struct OptixPipelineCompileOptions
{
    /// Boolean value indicating whether motion blur could be used
    int usesMotionBlur;

    /// Traversable graph bitfield. See OptixTraversableGraphFlags
    unsigned int traversableGraphFlags;

    /// How much storage, in 32b words, to make available for the payload, [0..8]
    int numPayloadValues;

    /// How much storage, in 32b words, to make available for the attributes. The
    /// minimum number is 2. Values below that will automatically be changed to 2. [2..8]
    int numAttributeValues;

    /// A bitmask of OptixExceptionFlags indicating which exceptions are enabled.
    unsigned int exceptionFlags;

    /// The name of the pipeline parameter variable.  If 0, no pipeline parameter
    /// will be available. This will be ignored if the launch param variable was
    /// optimized out or was not found in the modules linked to the pipeline.
    const char* pipelineLaunchParamsVariableName;
} OptixPipelineCompileOptions;

typedef struct OptixPipelineLinkOptions
{
    /// Maximum trace recursion depth. 0 means a ray generation shader can be
    /// launched, but can't trace any rays. The maximum allowed value is 31.
    unsigned int maxTraceDepth;

    /// Generate debug information.
    OptixCompileDebugLevel debugLevel;

    /// Boolean value that customizes the pipeline to enable or disable motion blur.  If
    /// enabled all modules must have specified the usesMotionBlur flag in
    /// OptixPipelineCompileOptions.
    int overrideUsesMotionBlur;
} OptixPipelineLinkOptions;

typedef struct OptixShaderBindingTable
{
    /// Device address of the SBT record of the ray gen program to start launch at. The address must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    CUdeviceptr raygenRecord;

    /// Device address of the SBT record of the exception program. The address must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    CUdeviceptr exceptionRecord;

    /// Arrays of SBT records for miss programs. The base address and the stride must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    CUdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;

    /// Arrays of SBT records for hit groups. The base address and the stride must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    CUdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;

    /// Arrays of SBT records for callable programs. If the base address is not null, the stride and count must not be
    /// zero. If the base address is null, then the count needs to zero. The base address and the stride must be a
    /// multiple of OPTIX_SBT_RECORD_ALIGNMENT.
    CUdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;

} OptixShaderBindingTable;

typedef struct OptixStackSizes
{
    /// Continuation stack size of RG programs in bytes
    unsigned int cssRG;
    /// Continuation stack size of MS programs in bytes
    unsigned int cssMS;
    /// Continuation stack size of CH programs in bytes
    unsigned int cssCH;
    /// Continuation stack size of AH programs in bytes
    unsigned int cssAH;
    /// Continuation stack size of IS programs in bytes
    unsigned int cssIS;
    /// Continuation stack size of CC programs in bytes
    unsigned int cssCC;
    /// Direct stack size of DC programs in bytes
    unsigned int dssDC;

} OptixStackSizes;

/// Options that can be passed to \c optixQueryFunctionTable()
typedef enum OptixQueryFunctionTableOptions
{
    /// Placeholder (there are no options yet)
    OPTIX_QUERY_FUNCTION_TABLE_OPTION_DUMMY = 0

} OptixQueryFunctionTableOptions;

/// Type of the function \c optixQueryFunctionTable()
typedef OptixResult( OptixQueryFunctionTable_t )( int          ABI_ID,
                                                  unsigned int numOptions,
                                                  OptixQueryFunctionTableOptions* /*optionKeys*/,
                                                  const void** /*optionValues*/,
                                                  void*  functionTable,
                                                  size_t sizeOfTable );

#endif  // __optix_optix_7_types_h__
