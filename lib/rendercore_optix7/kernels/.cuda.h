/* .cuda.h - Copyright 2019 Utrecht University

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

// generic includes
#include <stdio.h>

// custom types
typedef unsigned int uint;
typedef unsigned char uchar;

// platform specific
#include "../CUDA/helper_math.h"
#ifdef __CUDACC__
#include "cuda_fp16.h"
#else
#include "half.hpp"
#endif
#include "../core_settings.h"
#include "common_settings.h"
#include "common_classes.h"
#if __CUDA_ARCH__ >= 700
#define THREADMASK	__activemask() // volta, turing
#else
#define THREADMASK	0xffffffff // pascal, kepler, fermi
#endif

// suppress errors outside nvcc
#include "noerrors2.h"

// convenience macros
#define NEXTMULTIPLEOF(a,b)	(((a)+((b)-1))&(0x7fffffff-((b)-1)))

// camera lens
#define APPERTURE_BLADES	9

// final pixel buffer for output
surface<void, cudaSurfaceType2D> renderTarget;
namespace lh2core {
__host__ const surfaceReference* renderTargetRef() { const surfaceReference* s; cudaGetSurfaceReference( &s, &renderTarget ); return s; }
} // namespace lh2core

// function defintion helper
#define LH2_DEVFUNC	static __forceinline__ __device__

// EOF