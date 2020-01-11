/* pbrt_wrap.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   LH2 variant of pbrt.h. Forward declarations, configuration and
   typedefs from common LH2 types to PBRT types.
*/

#pragma once

#include "stringprint.h"
#include <numeric>
#include <system.h>

namespace pbrt
{

struct Options
{
	bool cat = false, toPly = false;
};

extern Options PbrtOptions;

using Float = float;
using Vector3f = float3;
using Vector2f = float2;
using Normal3f = float3;
using Point3f = float3;
using Point2f = float2;
using Transform = mat4;

/**
 * traits-alike structure returning the desired vector type
 * for a templatable number of components.
 */
template <int c, typename T = Float>
struct VectorType
{
	// Incomplete but valid type for now
	typedef T* type;
};

template <>
struct VectorType<2, Float>
{
	typedef float2 type;
};
template <>
struct VectorType<3, Float>
{
	typedef float3 type;
};
template <>
struct VectorType<4, Float>
{
	typedef float4 type;
};

#define PBRT_CONSTEXPR constexpr

static PBRT_CONSTEXPR Float Infinity = std::numeric_limits<Float>::infinity();

template <typename... Args>
void Warning( const char* format, Args&&... args )
{
	std::string fmt = "Warning: ";
	fmt += format;
	fmt += '\n';
	printf( fmt.c_str(), std::forward<Args>( args )... );
}

template <typename... Args>
void Error( const char* format, Args&&... args )
{
	FatalError( format, std::forward<Args>( args )... );
}

template <int nSpectrumSamples>
class CoefficientSpectrum;

class RGBSpectrum;
class SampledSpectrum;
#ifdef PBRT_SAMPLED_SPECTRUM
// All sampled spectrums must be converted to RGB for GPU processing.
#error "LH2 Does not support SampledSpectrum!"
typedef SampledSpectrum Spectrum;
#else
typedef RGBSpectrum Spectrum;
#endif

class ParamSet;
template <typename T>
struct ParamSetItem;

#define CHECK_EQ( val1, val2 ) // CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE( val1, val2 ) // CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE( val1, val2 ) // CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT( val1, val2 ) // CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE( val1, val2 ) // CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT( val1, val2 ) // CHECK_OP(_GT, > , val1, val2)
#define DCHECK( val ) assert( val )
#define CHECK( val ) assert( val )

template <typename T, typename U, typename V>
inline T Clamp( T val, U low, V high )
{
	if ( val < low )
		return low;
	else if ( val > high )
		return high;
	else
		return val;
}

inline Float Lerp( Float t, Float v1, Float v2 ) { return ( 1 - t ) * v1 + t * v2; }

inline Float Radians( Float deg ) { return ( PI / 180.f ) * deg; }

inline Float Degrees( Float rad ) { return ( 180.f / PI ) * rad; }

template <typename Predicate>
int FindInterval( int size, const Predicate& pred )
{
	int first = 0, len = size;
	while ( len > 0 )
	{
		int half = len >> 1, middle = first + half;
		// Bisect range based on value of _pred_ at _middle_
		if ( pred( middle ) )
		{
			first = middle + 1;
			len -= half + 1;
		}
		else
			len = half;
	}
	return Clamp( first - 1, 0, size - 2 );
}

inline void CoordinateSystem( const float3& v1, float3& v2, float3& v3 )
{
	if ( std::abs( v1.x ) > std::abs( v1.y ) )
		v2 = make_float3( -v1.z, 0, v1.x ) / std::sqrt( v1.x * v1.x + v1.z * v1.z );
	else
		v2 = make_float3( 0, v1.z, -v1.y ) / std::sqrt( v1.y * v1.y + v1.z * v1.z );
	v3 = cross( v1, v2 );
}

}; // namespace pbrt
