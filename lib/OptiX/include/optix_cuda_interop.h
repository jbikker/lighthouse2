
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
 * @file   optix_cuda_interop.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API declarations CUDAInterop
 *
 * OptiX public API declarations for CUDA interoperability
 */

#ifndef __optix_optix_cuda_interop_h__
#define __optix_optix_cuda_interop_h__

#include "optix.h"

#ifdef __cplusplus
extern "C" {
#endif

  /**
  * @brief Creates a new buffer object that will later rely on user-side CUDA allocation
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * Deprecated in OptiX 4.0. Now forwards to @ref rtBufferCreate.
  *
  * @param[in]   context          The context to create the buffer in
  * @param[in]   bufferdesc       Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[out]  buffer           The return handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferCreateForCUDA was introduced in OptiX 3.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferSetDevicePointer,
  * @ref rtBufferMarkDirty,
  * @ref rtBufferDestroy
  * 
  */
  RTresult RTAPI rtBufferCreateForCUDA (RTcontext context, unsigned int bufferdesc, RTbuffer *buffer);
  
  /**
  * @brief Gets the pointer to the buffer's data on the given device
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferGetDevicePointer returns the pointer to the data of \a buffer on device \a optix_device_ordinal in **\a device_pointer.
  *
  * If @ref rtBufferGetDevicePointer has been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is created with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  * 
  * @param[in]   buffer                          The buffer to be queried for its device pointer
  * @param[in]   optix_device_ordinal            The number assigned by OptiX to the device
  * @param[out]  device_pointer                  The return handle to the buffer's device pointer
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * 
  * <B>History</B>
  * 
  * @ref rtBufferGetDevicePointer was introduced in OptiX 3.0.
  * 
  * <B>See also</B>
  * @ref rtBufferMarkDirty,
  * @ref rtBufferSetDevicePointer
  * 
  */
  RTresult RTAPI rtBufferGetDevicePointer (RTbuffer buffer, int optix_device_ordinal, void** device_pointer);
  
  /**
  * @brief Sets a buffer as dirty
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * If @ref rtBufferSetDevicePointer or @ref rtBufferGetDevicePointer have been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch, unless the buffer is declared with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  *
  * Note that RT_BUFFER_COPY_ON_DIRTY currently only applies to CUDA interop buffers (buffers for which the application has a device pointer).
  * 
  * @param[in]   buffer                          The buffer to be marked dirty
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * 
  * <B>History</B>
  * 
  * @ref rtBufferMarkDirty was introduced in OptiX 3.0.
  * 
  * <B>See also</B>
  * @ref rtBufferGetDevicePointer,
  * @ref rtBufferSetDevicePointer,
  * @ref RT_BUFFER_COPY_ON_DIRTY
  * 
  */
  RTresult RTAPI rtBufferMarkDirty (RTbuffer buffer);
  
  /**
  * @brief Sets the pointer to the buffer's data on the given device
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferSetDevicePointer sets the pointer to the data of \a buffer on device \a optix_device_ordinal to \a device_pointer.
  *
  * If @ref rtBufferSetDevicePointer has been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is declared with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  * 
  * @param[in]   buffer                          The buffer for which the device pointer is to be set
  * @param[in]   optix_device_ordinal            The number assigned by OptiX to the device
  * @param[in]   device_pointer                  The pointer to the data on the specified device
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * 
  * <B>History</B>
  * 
  * @ref rtBufferSetDevicePointer was introduced in OptiX 3.0.
  * 
  * <B>See also</B>
  * @ref rtBufferMarkDirty,
  * @ref rtBufferGetDevicePointer
  * 
  */
  RTresult RTAPI rtBufferSetDevicePointer (RTbuffer buffer, int optix_device_ordinal, void* device_pointer);

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_cuda_interop_h__ */
