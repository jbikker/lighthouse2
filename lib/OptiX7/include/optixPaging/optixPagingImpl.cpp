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

#include <optixPaging/optixPaging.h>

__device__ __forceinline__ uint32_t countSetBitsAndCalcIndex( const uint32_t laneId, const uint32_t pageBits, uint32_t* pageCount )
{
    // Each thread gets sum of all values of numSetBits for entire warp also do
    // a prefix sum for help indexing later on.
    uint32_t numSetBits = __popc( pageBits );
    uint32_t index      = numSetBits;

#if defined( __CUDACC__ )
#pragma unroll
#endif
    for( uint32_t i = 1; i < 32; i *= 2 )
    {
        numSetBits += __shfl_xor_sync( 0xFFFFFFFF, numSetBits, i );
        uint32_t n = __shfl_up_sync( 0xFFFFFFFF, index, i );

        if( laneId >= i )
            index += n;
    }
    index = __shfl_up_sync( 0xFFFFFFFF, index, 1 );

    // One thread from each warp reserves its global index and updates the count
    // for other warps.
    int startingIndex;
    if( laneId == 0 )
    {
        index = 0;
        if( numSetBits )
            startingIndex = atomicAdd( pageCount, numSetBits );
    }
    index += __shfl_sync( 0xFFFFFFFF, startingIndex, 0 );

    return index;
}

__device__ __forceinline__ void addPagesToList( uint32_t startingIndex, uint32_t pageBits, uint32_t pageBitOffset, uint32_t maxCount, uint32_t* outputArray )
{
    while( pageBits != 0 && ( startingIndex < maxCount ) )
    {
        // Find index of least significant bit and clear it
        uint32_t bitIndex = __ffs( pageBits ) - 1;
        pageBits ^= ( 1U << bitIndex );

        // Add the requested page to the queue
        outputArray[startingIndex++] = pageBitOffset + bitIndex;
    }
}

__global__ void devicePullRequests( uint32_t* usageBits,
                                    uint32_t* residenceBits,
                                    uint32_t  maxVaSizeInPages,
                                    uint32_t* devRequestedPages,
                                    uint32_t  numRequestedPages,
                                    uint32_t* numRequestedPagesReturned,
                                    uint32_t* devStalePages,
                                    uint32_t  numStalePages,
                                    uint32_t* numStalePagesReturned,
                                    uint32_t* devEvictablePages,
                                    uint32_t  numEvictablePages,
                                    uint32_t* numEvictablePagesReturned )
{
    uint32_t globalIndex   = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t pageBitOffset = globalIndex * 32;

    const uint32_t laneId = globalIndex % 32;
    while( pageBitOffset < maxVaSizeInPages )
    {
        const uint32_t requestWord   = usageBits[globalIndex];
        const uint32_t residenceWord = residenceBits[globalIndex];

        // Gather the outstanding page requests.  A request is 'outstanding' if it
        // is requested but not resident; otherwise we don't need to return the request
        // to the host.
        const uint32_t outstandingRequests = ~residenceWord & requestWord;
        const uint32_t requestIndex = countSetBitsAndCalcIndex( laneId, outstandingRequests, numRequestedPagesReturned );
        addPagesToList( requestIndex, outstandingRequests, pageBitOffset, numRequestedPages, devRequestedPages );

        // Gather the stale pages, which are pages that are resident but not requested.
        const uint32_t stalePages = ~requestWord & residenceWord;
        const uint32_t staleIndex = countSetBitsAndCalcIndex( laneId, stalePages, numStalePagesReturned );
        addPagesToList( staleIndex, stalePages, pageBitOffset, numStalePages, devStalePages );

        globalIndex += gridDim.x * blockDim.x;
        pageBitOffset = globalIndex * 32;
    }

    // TODO: Gather the evictable pages? Or is that host-side work?

    // Clamp counts of returned pages, since they may have been over-incremented
    if( laneId == 0 )
    {
        atomicMin( numRequestedPagesReturned, numRequestedPages );
        atomicMin( numStalePagesReturned, numStalePages );
    }
}

__global__ void deviceFillPages( uint64_t* pageTable, uint32_t* residenceBits, MapType* devFilledPages, int filledPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < filledPageCount )
    {
        const MapType devFilledPage    = devFilledPages[globalIndex];
        pageTable[devFilledPage.first] = devFilledPage.second;
        atomicSetBit( devFilledPage.first, residenceBits );
        globalIndex += gridDim.x * blockDim.x;
    }
}

__global__ void deviceInvalidatePages( uint32_t* residenceBits, uint32_t* devInvalidatedPages, int invalidatedPageCount )
{
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    while( globalIndex < invalidatedPageCount )
    {
        atomicUnsetBit( devInvalidatedPages[globalIndex], residenceBits );
        globalIndex += gridDim.x * blockDim.x;
    }
}
