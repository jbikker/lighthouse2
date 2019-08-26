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

#include <optixPaging/optixPagingImpl.cpp>

__host__ void optixPagingPullRequests( OptixPagingContext* context,
                                       uint32_t*           devRequestedPages,
                                       uint32_t            numRequestedPages,
                                       uint32_t*           devStalePages,
                                       uint32_t            numStalePages,
                                       uint32_t*           devEvictablePages,
                                       uint32_t            numEvictablePages,
                                       uint32_t*           devNumPagesReturned )
{
    checkCudaError( cudaMemset( devRequestedPages, 0, numRequestedPages * sizeof( uint32_t ) ) );
    checkCudaError( cudaMemset( devStalePages, 0, numStalePages * sizeof( uint32_t ) ) );
    checkCudaError( cudaMemset( devEvictablePages, 0, numEvictablePages * sizeof( uint32_t ) ) );
    checkCudaError( cudaMemset( devNumPagesReturned, 0, 3 * sizeof( uint32_t ) ) );

    int numPagesPerThread = context->maxVaSizeInPages / 65536;
    numPagesPerThread     = ( numPagesPerThread + 31 ) & 0xFFFFFFE0;  // Round to nearest multiple of 32
    if( numPagesPerThread < 32 )
        numPagesPerThread = 32;

    const int numThreadsPerBlock = 64;
    const int numPagesPerBlock   = numPagesPerThread * numThreadsPerBlock;
    const int numBlocks          = ( context->maxVaSizeInPages + ( numPagesPerBlock - 1 ) ) / numPagesPerBlock;

    devicePullRequests<<<numBlocks, numThreadsPerBlock>>>( context->usageBits, context->residenceBits, context->maxVaSizeInPages,
                                                           devRequestedPages, numRequestedPages, devNumPagesReturned,
                                                           devStalePages, numStalePages, devNumPagesReturned + 1,
                                                           devEvictablePages, numEvictablePages, devNumPagesReturned + 2 );
}

__host__ void optixPagingPushMappings( OptixPagingContext* context,
                                       MapType*            devFilledPages,
                                       int                 filledPageCount,
                                       uint32_t*           devInvalidatedPages,
                                       int                 invalidatedPageCount )
{
    // Zero out the reference bits
    uint32_t referenceBitsSizeInBytes = sizeof( uint32_t ) * static_cast<uint32_t>( context->residenceBits - context->usageBits );
    checkCudaError( cudaMemset( context->usageBits, 0, referenceBitsSizeInBytes ) );

    const int numPagesPerThread = 2;
    const int numThreadsPerBlock = 128;
    const int numPagesPerBlock = numPagesPerThread * numThreadsPerBlock;
    if( filledPageCount != 0 )
    {
        const int numFilledPageBlocks = ( filledPageCount + numPagesPerBlock - 1 ) / numPagesPerBlock;
        deviceFillPages<<<numFilledPageBlocks, numThreadsPerBlock>>>( context->pageTable, context->residenceBits,
                                                                      devFilledPages, filledPageCount );
    }

    if( invalidatedPageCount != 0 )
    {
        const int numInvalidatedPageBlocks = ( invalidatedPageCount + numPagesPerBlock - 1 ) / numPagesPerBlock;
        deviceInvalidatePages<<<numInvalidatedPageBlocks, numThreadsPerBlock>>>( context->residenceBits, devInvalidatedPages,
                                                                                 invalidatedPageCount );
    }
}

__host__ void optixPagingCreate( OptixPagingOptions* options, OptixPagingContext** context )
{
    *context                       = new OptixPagingContext;
    ( *context )->maxVaSizeInPages = options->maxVaSizeInPages;
    ( *context )->usageBits        = nullptr;
    ( *context )->pageTable        = nullptr;
}

__host__ void optixPagingDestroy( OptixPagingContext* context )
{
    delete context;
}

__host__ void optixPagingCalculateSizes( uint32_t vaSizeInPages, OptixPagingSizes& sizes )
{
    //TODO: decide on limit for sizes, add asserts

    // Number of entries * 8 bytes per entry
    sizes.pageTableSizeInBytes = vaSizeInPages * sizeof( uint64_t );

    // Calc reference bits size with 128 byte alignnment, residence bits are same size.
    // Usage bits is the concatenation of the two.
    uint32_t referenceBitsSizeInBytes = ( ( vaSizeInPages + 1023 ) & 0xFFFFFC00 ) / 8;
    uint32_t residenceBitsSizeInBytes = referenceBitsSizeInBytes;
    sizes.usageBitsSizeInBytes        = referenceBitsSizeInBytes + residenceBitsSizeInBytes;
}

__host__ void optixPagingSetup( OptixPagingContext* context, const OptixPagingSizes& sizes, int numWorkers )
{
    // TODO: decide on limit for numWorkers, add asserts

    // This doubles as a memset and a check to make sure they allocated the device pointers
    checkCudaError( cudaMemset( context->pageTable, 0, sizes.pageTableSizeInBytes ) );
    checkCudaError( cudaMemset( context->usageBits, 0, sizes.usageBitsSizeInBytes * numWorkers ) );

    // Set residence bits pointer in context (index half way into usage bits)
    context->residenceBits = context->usageBits + ( sizes.usageBitsSizeInBytes / sizeof(uint32_t) / 2 );
}
