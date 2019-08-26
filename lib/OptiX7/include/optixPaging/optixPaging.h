//
//  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <cuda_runtime.h>

#include <utility>

#include <stdint.h>
#include <stdio.h>

inline bool checkCudaError( cudaError_t err )
{
    if( err != cudaSuccess )
    {
        printf( "CUDA error: %d\n", err );
        return false;
    }
    return true;
}

const int MAX_WORKER_THREADS = 32;

template <typename T>
__host__ __device__ T minimum( T lhs, T rhs )
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__host__ __device__ T maximum( T lhs, T rhs )
{
    return lhs > rhs ? lhs : rhs;
}

using MapType = std::pair<uint32_t, uint64_t>;

struct OptixPagingSizes
{
    uint32_t pageTableSizeInBytes;  // only one for all workers
    uint32_t usageBitsSizeInBytes;  // per worker
};

struct OptixPagingOptions
{
    uint32_t maxVaSizeInPages;
    uint32_t initialVaSizeInPages;
};

struct OptixPagingContext
{
    uint32_t  maxVaSizeInPages;
    uint32_t* usageBits;      // also beginning of referenceBits. [ referenceBits | residencesBits ]
    uint32_t* residenceBits;  // located half way into usasgeBits.
    uint64_t* pageTable;
};

__host__ void optixPagingCreate( OptixPagingOptions* options, OptixPagingContext** context );
__host__ void optixPagingDestroy( OptixPagingContext* context );
__host__ void optixPagingCalculateSizes( uint32_t vaSizeInPages, OptixPagingSizes& sizes );
__host__ void optixPagingSetup( OptixPagingContext* context, const OptixPagingSizes& sizes, int numWorkers );
__host__ void optixPagingPullRequests( OptixPagingContext* context,
                                       uint32_t*           devRequestedPages,
                                       uint32_t            numRequestedPages,
                                       uint32_t*           devStalePages,
                                       uint32_t            numStalePages,
                                       uint32_t*           devEvictablePages,
                                       uint32_t            numEvictablePages,
                                       uint32_t*           devNumPagesReturned );
__host__ void optixPagingPushMappings( OptixPagingContext* context,
                                       MapType*            devFilledPages,
                                       int                 filledPageCount,
                                       uint32_t*           devInvalidatedPages,
                                       int                 invalidatedPageCount );

#if defined( __CUDACC__ ) || defined( OPTIX_PAGING_BIT_OPS )
__device__ inline void atomicSetBit( uint32_t bitIndex, uint32_t* bitVector )
{
    const uint32_t wordIndex = bitIndex >> 5;
    const uint32_t bitOffset = bitIndex % 32;
    const uint32_t mask      = 1U << bitOffset;
    atomicOr( bitVector + wordIndex, mask );
}

__device__ inline void atomicUnsetBit( int bitIndex, uint32_t* bitVector )
{
    const int wordIndex = bitIndex / 32;
    const int bitOffset = bitIndex % 32;

    const int mask = ~( 1U << bitOffset );
    atomicAnd( bitVector + wordIndex, mask );
}

__device__ inline bool checkBitSet( uint32_t bitIndex, const uint32_t* bitVector )
{
    const uint32_t wordIndex = bitIndex >> 5;
    const uint32_t bitOffset = bitIndex % 32;
    return ( bitVector[wordIndex] & ( 1U << bitOffset ) ) != 0;
}

__device__ inline uint64_t optixPagingMapOrRequest( uint32_t* usageBits, uint32_t* residenceBits, uint64_t* pageTable, uint32_t page, bool* valid )
{
    bool requested = checkBitSet( page, usageBits );
    if( !requested )
        atomicSetBit( page, usageBits );

    bool mapped = checkBitSet( page, residenceBits );
    *valid      = mapped;

    return mapped ? pageTable[page] : 0;
}
#endif
