/* cudatools.h - Copyright 2019 Utrecht University

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

#pragma once 

enum { NOT_ALLOCATED = 0, ON_HOST = 1, ON_DEVICE = 2 };

#define CUDACHECK(x,y) CUDATools::CheckError( __FILE__, __LINE__, x, y )

#define STRINGIFY2(x) #x
#define CHK_NVRTC( func ) { nvrtcResult code = func; if (code != NVRTC_SUCCESS) \
	FatalError( __FILE__, __LINE__, nvrtcGetErrorString( code ) ); }

class CUDATools
{
public:
	static float Elapsed( cudaEvent_t start, cudaEvent_t end )
	{
		float elapsed;
		cudaEventElapsedTime( &elapsed, start, end );
		return max( 1e-20f, elapsed ) * 0.001f; // report in seconds
	}
	static int _ConvertSMVer2Cores( int major, int minor )
	{
		typedef struct { int SM, Cores; } sSMtoCores;
		sSMtoCores nGpuArchCoresPerSM[] = { { 0x30, 192 }, { 0x32, 192 }, { 0x35, 192 }, { 0x37, 192 },
		{ 0x50, 128 }, { 0x52, 128 }, { 0x53, 128 }, { 0x60, 64 }, { 0x61, 128 }, { 0x62, 128 },
		{ 0x70, 64 }, { 0x72, 64 }, { 0x75, 64 }, { -1, -1 } };
		int index = 0;
		while (nGpuArchCoresPerSM[index].SM != -1)
		{
			if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) return nGpuArchCoresPerSM[index].Cores;
			index++;
		}
		return nGpuArchCoresPerSM[index - 1].Cores;
	}
	static int FastestDevice() // from the CUDA 10.0 examples
	{
		int curdev = 0, smperproc = 0, fastest = 0, count = 0, prohibited = 0;
		uint64_t max_perf = 0;
		cudaDeviceProp deviceProp;
		cudaGetDeviceCount( &count );
		if (count == 0) exit( EXIT_FAILURE );
		while (curdev < count)
		{
			cudaGetDeviceProperties( &deviceProp, curdev );
			if (deviceProp.computeMode != cudaComputeModeProhibited)
			{
				if (deviceProp.major == 9999 && deviceProp.minor == 9999) smperproc = 1; else
					smperproc = _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor );
				uint64_t compute_perf = (uint64_t)deviceProp.multiProcessorCount * smperproc * deviceProp.clockRate;
				if (compute_perf > max_perf) 
				{
					max_perf = compute_perf;
					fastest = curdev;
				}
			}
			else prohibited++;
			++curdev;
		}
		if (prohibited == count) exit( EXIT_FAILURE );
		return fastest;
	}
	static void setfv( string& s, const char* fmt, va_list args )
	{
		static char* buffer = 0;
		if (!buffer) buffer = new char[16384];
		int len = _vscprintf( fmt, args );
		if (!len) return;
		vsprintf_s( buffer, len + 1, fmt, args );
		s = buffer;
	}
	static void fail( const char* t )
	{
		printf( t );
		while (1) exit( 0 );
	}
	static const char* decodeError( cudaError_t res )
	{
		static char error[128];
		switch (res)
		{
		default:                                        strcpy_s( error, "Unknown cudaError_t" ); break;
		case CUDA_SUCCESS:                              strcpy_s( error, "No error" ); break;
		case CUDA_ERROR_INVALID_VALUE:                  strcpy_s( error, "Invalid value" ); break;
		case CUDA_ERROR_OUT_OF_MEMORY:                  strcpy_s( error, "Out of memory" ); break;
		case CUDA_ERROR_NOT_INITIALIZED:                strcpy_s( error, "Not initialized" ); break;
		case CUDA_ERROR_DEINITIALIZED:                  strcpy_s( error, "Deinitialized" ); break;
		case CUDA_ERROR_NO_DEVICE:                      strcpy_s( error, "No device" ); break;
		case CUDA_ERROR_INVALID_DEVICE:                 strcpy_s( error, "Invalid device" ); break;
		case CUDA_ERROR_INVALID_IMAGE:                  strcpy_s( error, "Invalid image" ); break;
		case CUDA_ERROR_INVALID_CONTEXT:                strcpy_s( error, "Invalid context" ); break;
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        strcpy_s( error, "Context already current" ); break;
		case CUDA_ERROR_MAP_FAILED:                     strcpy_s( error, "Map failed" ); break;
		case CUDA_ERROR_UNMAP_FAILED:                   strcpy_s( error, "Unmap failed" ); break;
		case CUDA_ERROR_ARRAY_IS_MAPPED:                strcpy_s( error, "Array is mapped" ); break;
		case CUDA_ERROR_ALREADY_MAPPED:                 strcpy_s( error, "Already mapped" ); break;
		case CUDA_ERROR_NO_BINARY_FOR_GPU:              strcpy_s( error, "No binary for GPU" ); break;
		case CUDA_ERROR_ALREADY_ACQUIRED:               strcpy_s( error, "Already acquired" ); break;
		case CUDA_ERROR_NOT_MAPPED:                     strcpy_s( error, "Not mapped" ); break;
		case CUDA_ERROR_INVALID_SOURCE:                 strcpy_s( error, "Invalid source" ); break;
		case CUDA_ERROR_FILE_NOT_FOUND:                 strcpy_s( error, "File not found" ); break;
		case CUDA_ERROR_INVALID_HANDLE:                 strcpy_s( error, "Invalid handle" ); break;
		case CUDA_ERROR_NOT_FOUND:                      strcpy_s( error, "Not found" ); break;
		case CUDA_ERROR_NOT_READY:                      strcpy_s( error, "Not ready" ); break;
		case CUDA_ERROR_LAUNCH_FAILED:                  strcpy_s( error, "Launch failed" ); break;
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        strcpy_s( error, "Launch out of resources" ); break;
		case CUDA_ERROR_LAUNCH_TIMEOUT:                 strcpy_s( error, "Launch timeout" ); break;
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  strcpy_s( error, "Launch incompatible texturing" ); break;
		case CUDA_ERROR_UNKNOWN:                        strcpy_s( error, "Unknown error" ); break;
		case CUDA_ERROR_PROFILER_DISABLED:              strcpy_s( error, "Profiler disabled" ); break;
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       strcpy_s( error, "Profiler not initialized" ); break;
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:       strcpy_s( error, "Profiler already started" ); break;
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       strcpy_s( error, "Profiler already stopped" ); break;
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            strcpy_s( error, "Not mapped as array" ); break;
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          strcpy_s( error, "Not mapped as pointer" ); break;
		case CUDA_ERROR_ECC_UNCORRECTABLE:              strcpy_s( error, "ECC uncorrectable" ); break;
		case CUDA_ERROR_UNSUPPORTED_LIMIT:              strcpy_s( error, "Unsupported limit" ); break;
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         strcpy_s( error, "Context already in use" ); break;
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: strcpy_s( error, "Shared object symbol not found" ); break;
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      strcpy_s( error, "Shared object init failed" ); break;
		case CUDA_ERROR_OPERATING_SYSTEM:               strcpy_s( error, "Operating system error" ); break;
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    strcpy_s( error, "Peer access already enabled" ); break;
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        strcpy_s( error, "Peer access not enabled" ); break;
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         strcpy_s( error, "Primary context active" ); break;
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:           strcpy_s( error, "Context is destroyed" ); break;
		case CUDA_ERROR_ILLEGAL_ADDRESS:				strcpy_s( error, "Illegal address" ); break;
		case CUDA_ERROR_MISALIGNED_ADDRESS:				strcpy_s( error, "Misaligned address" ); break;
		}
		return error;
	}
	static void CheckError( const char* file, int line, const char* funcName, cudaError_t res )
	{
		if (res != CUDA_SUCCESS) 
		{
			if (!strcmp( funcName, "cudaGraphicsGLRegisterImage" ))
			{
				FatalError( file, line, decodeError( res ), 
					"cudaGraphicsGLRegisterImage\n(Are you running using the IGP?\nUse NVIDIA control panel to enable the high performance GPU.)" );
			}
			else
			{
				FatalError( file, line, decodeError( res ), funcName );
			}
		}
	}
	static void compileToPTX( string &ptx, const char* cuSource, const char* sourceDir, const int cc, const int optixVer )
	{
		// create program
		nvrtcProgram prog = 0;
		CHK_NVRTC( nvrtcCreateProgram( &prog, cuSource, 0, 0, NULL, NULL ) );
		// gather NVRTC options
		vector<const char*> options;
		if (optixVer > 6) options.push_back( "-I../../lib/Optix7/include/" ); else options.push_back( "-I../../lib/Optix/include/" );
		string optionString = "-I";
		optionString += string( sourceDir );
		options.push_back( optionString.c_str() );
		options.push_back( "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include/" );
		options.push_back( "-I../../lib/CUDA/" );
		// collect NVRTC options
		char versionString[64];
		sprintf_s( versionString, "compute_%i", cc >= 70 ? 70 : 50 );
		const char* compiler_options[] = { "-arch", versionString, "-restrict", "-std=c++11", "-use_fast_math", "-default-device", "-rdc", "true", "-D__x86_64", 0 };
		const size_t n_compiler_options = sizeof( compiler_options ) / sizeof( compiler_options[0] );
		for (size_t i = 0; i < n_compiler_options - 1; i++) options.push_back( compiler_options[i] );
		// JIT compile CU to PTX
		int versionMinor, versionMajor;
		nvrtcVersion( &versionMajor, &versionMinor );
		printf( "compiling cuda code using nvcrt %i.%i\n", versionMajor, versionMinor );
		const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );
		// retrieve log output
		size_t log_size = 0;
		CHK_NVRTC( nvrtcGetProgramLogSize( prog, &log_size ) );
		static string nvrtcLog;
		nvrtcLog.resize( log_size );
		if (log_size > 1) CHK_NVRTC( nvrtcGetProgramLog( prog, &nvrtcLog[0] ) );
		if (compileRes != NVRTC_SUCCESS) FatalError( "Compilation failed.\n%s", nvrtcLog.c_str() );
		// retrieve PTX code
		size_t ptx_size = 0;
		CHK_NVRTC( nvrtcGetPTXSize( prog, &ptx_size ) );
		ptx.resize( ptx_size );
		CHK_NVRTC( nvrtcGetPTX( prog, &ptx[0] ) );
		// cleanup
		CHK_NVRTC( nvrtcDestroyProgram( &prog ) );
	}
};

template <class T> class CoreBuffer
{
public:
	CoreBuffer() = default;
	CoreBuffer( __int64 elements, __int64 loc, const void* source = 0 ) : location( loc )
	{
		numElements = elements;
		sizeInBytes = elements * sizeof( T );
		if (elements > 0)
		{
			if (location & ON_DEVICE)
			{
				// location is ON_DEVICE; allocate room on device
				CUDACHECK( "cudaMalloc", cudaMalloc( &devPtr, sizeInBytes ) );
				owner |= ON_DEVICE;
			}
			if (location & ON_HOST)
			{
				// location is ON_HOST; use supplied pointer or allocate room if no source was specified
				if (source) 
				{
					hostPtr = (T*)source; 
					if (location & ON_DEVICE) CopyToDevice();
				}
				else 
				{
					hostPtr = (T*)_aligned_malloc( sizeInBytes, 64 ), owner |= ON_HOST;
				}
			}
			else if (source && (location & ON_DEVICE))
			{
				// location is ON_DEVICE only, and we have data, so send the data over
				hostPtr = (T*)source;
				CopyToDevice();
				hostPtr = 0;
			}
		}
	}
	~CoreBuffer()
	{
		if (sizeInBytes > 0)
		{
			if (owner & ON_HOST)
			{
				_aligned_free( hostPtr );
				hostPtr = 0;
				owner &= ~ON_HOST;
			}
			if (owner & ON_DEVICE)
			{
				CUDACHECK( "cudaFree", cudaFree( devPtr ) );
				owner &= ~ON_DEVICE;
			}
		}
	}
	void* CopyToDevice()
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_DEVICE))
			{
				CUDACHECK( "cudaMalloc", cudaMalloc( &devPtr, sizeInBytes ) );
				location |= ON_DEVICE;
				owner |= ON_DEVICE;
			}
			CUDACHECK( "cudaMemcpy", cudaMemcpy( devPtr, hostPtr, sizeInBytes, cudaMemcpyHostToDevice ) );
		}
		return devPtr;
	}
	void* CopyToDeviceAsync( cudaStream_t stream )
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_DEVICE))
			{
				CUDACHECK( "cudaMalloc", cudaMalloc( &devPtr, sizeInBytes ) );
				location |= ON_DEVICE;
				owner |= ON_DEVICE;
			}
			CUDACHECK( "cudaMemcpyAsync", cudaMemcpyAsync( devPtr, hostPtr, sizeInBytes, cudaMemcpyHostToDevice, stream ) );	
		}
		return devPtr;
	}
	void* MoveToDevice()
	{
		CopyToDevice();
		if (sizeInBytes > 0) _aligned_free( hostPtr );
		hostPtr = 0;
		owner &= ~ON_HOST;
		location &= ~ON_HOST;
		return devPtr;
	}
	T* CopyToHost()
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_HOST))
			{
				hostPtr = (T*)_aligned_malloc( sizeInBytes, 64 );
				location |= ON_HOST;
				owner |= ON_HOST;
			}
			CUDACHECK( "cudaMemcpy", cudaMemcpy( hostPtr, devPtr, sizeInBytes, cudaMemcpyDeviceToHost ) );
		}
		return hostPtr;
	}
	T* CopyToHostAsync( cudaStream_t stream )
	{
		if (sizeInBytes > 0)
		{
			if (!(location & ON_HOST))
			{
				hostPtr = (T*)_aligned_malloc( sizeInBytes, 64 );
				location |= ON_HOST;
				owner |= ON_HOST;
			}
			CUDACHECK( "cudaMemcpyAsync", cudaMemcpyAsync( hostPtr, devPtr, sizeInBytes, cudaMemcpyDeviceToHost, stream ) );
		}
		return hostPtr;
	}
	void Clear( int location, int overrideSize = -1 )
	{
		if (sizeInBytes > 0)
		{
			int bytesToClear = overrideSize == -1 ? sizeInBytes : overrideSize;
			if (location & ON_HOST) memset( hostPtr, 0, bytesToClear );
			if (location & ON_DEVICE) CUDACHECK( "cuMemset", cudaMemset( devPtr, 0, bytesToClear ) );
		}
	}
	__int64 GetSizeInBytes() const { return sizeInBytes; }
	__int64 GetSize() const { return numElements; }
	T* DevPtr() { return devPtr; }
	T** DevPtrPtr() { return &devPtr; /* Optix7 wants an array of pointers; this returns an array of 1 pointers. */ } 
	T* HostPtr() { return hostPtr; }
	void SetHostData( T* hostData ) { hostPtr = hostData; }
	// member data
private:
	__int64 location = NOT_ALLOCATED, owner = 0, sizeInBytes = 0, numElements = 0;
	T* devPtr = 0;
	T* hostPtr = 0;
};