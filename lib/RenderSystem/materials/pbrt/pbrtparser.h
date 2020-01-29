/*
	pbrt source code is Copyright(c) 1998-2016
						Matt Pharr, Greg Humphreys, and Wenzel Jakob.

	This file is part of pbrt.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are
	met:

	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.

	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// #if defined( _MSC_VER )
#pragma once
#define NOMINMAX
// #else
// #ifndef PBRT_CORE_PARSER_H
// #define PBRT_CORE_PARSER_H

#include <platform.h>
#include <rendersystem.h>

namespace pbrt
{

struct Options { bool cat = false, toPly = false; };

using Float = float;
using Vector3f = float3;
using Vector2f = float2;
using Normal3f = float3;
using Point3f = float3;
using Point2f = float2;
using Transform = mat4;

template <int c, typename T = Float> struct VectorType
{
	typedef T* type;
};

template <> struct VectorType<2, Float> { typedef float2 type; };
template <> struct VectorType<3, Float> { typedef float3 type; };
template <> struct VectorType<4, Float> { typedef float4 type; };

#define PBRT_CONSTEXPR constexpr

static PBRT_CONSTEXPR Float Infinity = std::numeric_limits<Float>::infinity();

template <typename... Args> void Warning( const char* format, Args&&... args )
{
	std::string fmt = "Warning: ";
	fmt += format, fmt += '\n';
	printf( fmt.c_str(), std::forward<Args>( args )... );
}

template <typename... Args> void Error( const char* format, Args&&... args )
{
	FatalError( format, std::forward<Args>( args )... );
}

template <int nSpectrumSamples> class CoefficientSpectrum;

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
template <typename T> struct ParamSetItem;

#define CHECK_EQ( val1, val2 ) // CHECK_OP(_EQ, ==, val1, val2)
#define CHECK_NE( val1, val2 ) // CHECK_OP(_NE, !=, val1, val2)
#define CHECK_LE( val1, val2 ) // CHECK_OP(_LE, <=, val1, val2)
#define CHECK_LT( val1, val2 ) // CHECK_OP(_LT, < , val1, val2)
#define CHECK_GE( val1, val2 ) // CHECK_OP(_GE, >=, val1, val2)
#define CHECK_GT( val1, val2 ) // CHECK_OP(_GT, > , val1, val2)
#define DCHECK( val ) assert( val )
#define CHECK( val ) assert( val )

template <typename T, typename U, typename V> inline T Clamp( T val, U low, V high )
{
	if (val < low) return low;
	else if (val > high) return high;
	else return val;
}

inline Float Lerp( Float t, Float v1, Float v2 ) { return (1 - t) * v1 + t * v2; }
inline Float Radians( Float deg ) { return (PI / 180.f) * deg; }
inline Float Degrees( Float rad ) { return (180.f / PI) * rad; }

template <typename Predicate> int FindInterval( int size, const Predicate& pred )
{
	int first = 0, len = size;
	while (len > 0)
	{
		int half = len >> 1, middle = first + half;
		// Bisect range based on value of _pred_ at _middle_
		if (pred( middle )) first = middle + 1, len -= half + 1; else len = half;
	}
	return Clamp( first - 1, 0, size - 2 );
}

inline void CoordinateSystem( const float3& v1, float3& v2, float3& v3 )
{
	if (std::abs( v1.x ) > std::abs( v1.y )) v2 = make_float3( -v1.z, 0, v1.x ) / std::sqrt( v1.x * v1.x + v1.z * v1.z );
	else v2 = make_float3( 0, v1.z, -v1.y ) / std::sqrt( v1.y * v1.y + v1.z * v1.z );
	v3 = cross( v1, v2 );
}

}; // namespace pbrt

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <numeric>
#include <system.h>
#include <functional>
#include <memory>
#include <vector>
#include "spectrum.h"
#include <map>
#include <cctype>
#include <climits>
#include <cstdlib>
#ifndef PBRT_IS_WINDOWS
#include <libgen.h>
#endif
#include <algorithm>
#include <cstdio>
#include <ctype.h>
#ifdef PBRT_HAVE_MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <iostream>
#include <utility>
#include <assert.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stddef.h>

namespace pbrt
{

bool ReadFloatFile( const char* filename, std::vector<Float>* values );
bool IsAbsolutePath( const std::string& filename );
std::string AbsolutePath( const std::string& filename );
std::string ResolveFilename( const std::string& filename );
std::string DirectoryContaining( const std::string& filename );
void SetSearchDirectory( const std::string& dirname );
inline bool HasExtension( const std::string& value, const std::string& ending )
{
	if (ending.size() > value.size()) return false;
	return std::equal( ending.rbegin(), ending.rend(), value.rbegin(),
		[]( char a, char b ) { return std::tolower( a ) == std::tolower( b ); } );
}

// ParamSet Declarations
class ParamSet
{
public:
	ParamSet() {}
	void AddFloat( const std::string&, std::unique_ptr<Float[]> v, int nValues = 1 );
	void AddInt( const std::string&, std::unique_ptr<int[]> v, int nValues );
	void AddBool( const std::string&, std::unique_ptr<bool[]> v, int nValues );
	void AddPoint2f( const std::string&, std::unique_ptr<Point2f[]> v, int nValues );
	void AddVector2f( const std::string&, std::unique_ptr<Vector2f[]> v, int nValues );
	void AddPoint3f( const std::string&, std::unique_ptr<Point3f[]> v, int nValues );
	void AddVector3f( const std::string&, std::unique_ptr<Vector3f[]> v, int nValues );
	void AddNormal3f( const std::string&, std::unique_ptr<Normal3f[]> v, int nValues );
	void AddString( const std::string&, std::unique_ptr<std::string[]> v, int nValues );
	void AddTexture( const std::string&, const std::string& );
	void AddRGBSpectrum( const std::string&, std::unique_ptr<Float[]> v, int nValues );
	void AddXYZSpectrum( const std::string&, std::unique_ptr<Float[]> v, int nValues );
	void AddBlackbodySpectrum( const std::string&, std::unique_ptr<Float[]> v, int nValues );
	void AddSampledSpectrumFiles( const std::string&, const char**, int nValues );
	void AddSampledSpectrum( const std::string&, std::unique_ptr<Float[]> v, int nValues );
	bool EraseInt( const std::string& );
	bool EraseBool( const std::string& );
	bool EraseFloat( const std::string& );
	bool ErasePoint2f( const std::string& );
	bool EraseVector2f( const std::string& );
	bool ErasePoint3f( const std::string& );
	bool EraseVector3f( const std::string& );
	bool EraseNormal3f( const std::string& );
	bool EraseSpectrum( const std::string& );
	bool EraseString( const std::string& );
	bool EraseTexture( const std::string& );
	Float FindOneFloat( const std::string&, Float d ) const;
	int FindOneInt( const std::string&, int d ) const;
	bool FindOneBool( const std::string&, bool d ) const;
	Point2f FindOnePoint2f( const std::string&, const Point2f& d ) const;
	Vector2f FindOneVector2f( const std::string&, const Vector2f& d ) const;
	Point3f FindOnePoint3f( const std::string&, const Point3f& d ) const;
	Vector3f FindOneVector3f( const std::string&, const Vector3f& d ) const;
	Normal3f FindOneNormal3f( const std::string&, const Normal3f& d ) const;
	Spectrum FindOneSpectrum( const std::string&, const Spectrum& d ) const;
	std::string FindOneString( const std::string&, const std::string& d ) const;
	std::string FindOneFilename( const std::string&, const std::string& d ) const;
	std::string FindTexture( const std::string& ) const;
	const Float* FindFloat( const std::string&, int* n ) const;
	const int* FindInt( const std::string&, int* nValues ) const;
	const bool* FindBool( const std::string&, int* nValues ) const;
	const Point2f* FindPoint2f( const std::string&, int* nValues ) const;
	const Vector2f* FindVector2f( const std::string&, int* nValues ) const;
	const Point3f* FindPoint3f( const std::string&, int* nValues ) const;
	const Vector3f* FindVector3f( const std::string&, int* nValues ) const;
	const Normal3f* FindNormal3f( const std::string&, int* nValues ) const;
	const Spectrum* FindSpectrum( const std::string&, int* nValues ) const;
	const std::string* FindString( const std::string&, int* nValues ) const;
	void ReportUnused() const;
	void Clear();
	std::string ToString() const;
	void Print( int indent ) const;
private:
	friend class TextureParams;
	friend bool shapeMaySetMaterialParameters( const ParamSet& ps );
	std::vector<std::shared_ptr<ParamSetItem<bool>>> bools;
	std::vector<std::shared_ptr<ParamSetItem<int>>> ints;
	std::vector<std::shared_ptr<ParamSetItem<Float>>> floats;
	std::vector<std::shared_ptr<ParamSetItem<Point2f>>> point2fs;
	std::vector<std::shared_ptr<ParamSetItem<Vector2f>>> vector2fs;
	std::vector<std::shared_ptr<ParamSetItem<Point3f>>> point3fs;
	std::vector<std::shared_ptr<ParamSetItem<Vector3f>>> vector3fs;
	std::vector<std::shared_ptr<ParamSetItem<Normal3f>>> normals;
	std::vector<std::shared_ptr<ParamSetItem<Spectrum>>> spectra;
	std::vector<std::shared_ptr<ParamSetItem<std::string>>> strings;
	std::vector<std::shared_ptr<ParamSetItem<std::string>>> textures;
	static std::map<std::string, Spectrum> cachedSpectra;
};

template <typename T> struct ParamSetItem
{
	ParamSetItem( const std::string& name, std::unique_ptr<T[]> val, int nValues = 1 );
	const std::string name;
	const std::unique_ptr<T[]> values;
	const int nValues;
	mutable bool lookedUp = false;
};

// ParamSetItem Methods
template <typename T>
ParamSetItem<T>::ParamSetItem( const std::string& name, std::unique_ptr<T[]> v, int nValues )
	: name( name ), values( std::move( v ) ), nValues( nValues )
{
}

// TextureParams Declarations
class TextureParams
{
public:
	inline TextureParams( const ParamSet& geomParams, const ParamSet& materialParams,
		std::map<std::string, HostMaterial::ScalarValue*>& fTex, std::map<std::string,
		HostMaterial::Vec3Value*>& sTex )
		: floatTextures( fTex ), spectrumTextures( sTex ), geomParams( geomParams ),
		materialParams( materialParams )
	{
	}
	HostMaterial::Vec3Value GetFloat3Texture( const std::string& name, const float3& def ) const;
	HostMaterial::Vec3Value GetFloat3Texture( const std::string& name, const Spectrum& def ) const;
	HostMaterial::Vec3Value* GetFloat3TextureOrNull( const std::string& name ) const;
	HostMaterial::ScalarValue GetFloatTexture( const std::string& name, Float def ) const;
	HostMaterial::ScalarValue* GetFloatTextureOrNull( const std::string& name ) const;
	Float FindFloat( const std::string& n, Float d ) const
	{
		return geomParams.FindOneFloat( n, materialParams.FindOneFloat( n, d ) );
	}
	std::string FindString( const std::string& n, const std::string& d = "" ) const
	{
		return geomParams.FindOneString( n, materialParams.FindOneString( n, d ) );
	}
	std::string FindFilename( const std::string& n, const std::string& d = "" ) const
	{
		return geomParams.FindOneFilename( n, materialParams.FindOneFilename( n, d ) );
	}
	int FindInt( const std::string& n, int d ) const
	{
		return geomParams.FindOneInt( n, materialParams.FindOneInt( n, d ) );
	}
	bool FindBool( const std::string& n, bool d ) const
	{
		return geomParams.FindOneBool( n, materialParams.FindOneBool( n, d ) );
	}
	Point3f FindPoint3f( const std::string& n, const Point3f& d ) const
	{
		return geomParams.FindOnePoint3f( n, materialParams.FindOnePoint3f( n, d ) );
	}
	Vector3f FindVector3f( const std::string& n, const Vector3f& d ) const
	{
		return geomParams.FindOneVector3f( n, materialParams.FindOneVector3f( n, d ) );
	}
	Normal3f FindNormal3f( const std::string& n, const Normal3f& d ) const
	{
		return geomParams.FindOneNormal3f( n, materialParams.FindOneNormal3f( n, d ) );
	}
	Spectrum FindSpectrum( const std::string& n, const Spectrum& d ) const
	{
		return geomParams.FindOneSpectrum( n, materialParams.FindOneSpectrum( n, d ) );
	}
	void ReportUnused() const;
	const ParamSet& GetGeomParams() const { return geomParams; }
	const ParamSet& GetMaterialParams() const { return materialParams; }
private:
	std::map<std::string, HostMaterial::ScalarValue*>& floatTextures;
	std::map<std::string, HostMaterial::Vec3Value*>& spectrumTextures;
	const ParamSet &geomParams, &materialParams;
};

// API Function Declarations
void pbrtInit( const Options& opt );
void pbrtCleanup();
void pbrtIdentity();
void pbrtTranslate( Float dx, Float dy, Float dz );
void pbrtRotate( Float angle, Float ax, Float ay, Float az );
void pbrtScale( Float sx, Float sy, Float sz );
void pbrtLookAt( Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux, Float uy, Float uz );
void pbrtConcatTransform( Float transform[16] );
void pbrtTransform( Float transform[16] );
void pbrtCoordinateSystem( const std::string& );
void pbrtCoordSysTransform( const std::string& );
void pbrtActiveTransformAll();
void pbrtActiveTransformEndTime();
void pbrtActiveTransformStartTime();
void pbrtTransformTimes( Float start, Float end );
void pbrtPixelFilter( const std::string& name, const ParamSet& params );
void pbrtFilm( const std::string& type, const ParamSet& params );
void pbrtSampler( const std::string& name, const ParamSet& params );
void pbrtAccelerator( const std::string& name, const ParamSet& params );
void pbrtIntegrator( const std::string& name, const ParamSet& params );
void pbrtCamera( const std::string&, const ParamSet& cameraParams );
void pbrtMakeNamedMedium( const std::string& name, const ParamSet& params );
void pbrtMediumInterface( const std::string& insideName, const std::string& outsideName );
void pbrtWorldBegin();
void pbrtAttributeBegin();
void pbrtAttributeEnd();
void pbrtTransformBegin();
void pbrtTransformEnd();
void pbrtTexture( const std::string& name, const std::string& type, const std::string& texname, const ParamSet& params );
void pbrtMaterial( const std::string& name, const ParamSet& params );
void pbrtMakeNamedMaterial( const std::string& name, const ParamSet& params );
void pbrtNamedMaterial( const std::string& name );
void pbrtLightSource( const std::string& name, const ParamSet& params );
void pbrtAreaLightSource( const std::string& name, const ParamSet& params );
void pbrtShape( const std::string& name, const ParamSet& params );
void pbrtReverseOrientation();
void pbrtObjectBegin( const std::string& name );
void pbrtObjectEnd();
void pbrtObjectInstance( const std::string& name );
void pbrtWorldEnd();
void pbrtParseFile( std::string filename );
void pbrtParseString( std::string str );

// Creating meshes
HostMesh* CreatePLYMesh( const Transform* o2w, const Transform* w2o, bool reverseOrientation,
	const ParamSet& params, const int materialIdx, std::map<std::string, 
	HostMaterial::ScalarValue*>* floatTextures = nullptr );
HostMesh* CreateTriangleMeshShape( const Transform* o2w, const Transform* w2o, bool reverseOrientation, 
	const ParamSet& params, const int materialIdx, std::map<std::string, 
	HostMaterial::ScalarValue*>* floatTextures = nullptr );

// Creating materials
HostMaterial* CreateDisneyMaterial( const TextureParams& mp );
HostMaterial* CreateGlassMaterial( const TextureParams& mp );
HostMaterial* CreateMatteMaterial( const TextureParams& mp );
HostMaterial* CreateMetalMaterial( const TextureParams& mp );
HostMaterial* CreateMirrorMaterial( const TextureParams& mp );
HostMaterial* CreatePlasticMaterial( const TextureParams& mp );
HostMaterial* CreateSubstrateMaterial( const TextureParams& mp );
HostMaterial* CreateUberMaterial( const TextureParams& mp );

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif // __GNUG__

inline void stringPrintfRecursive( std::string* s, const char* fmt )
{
	const char* c = fmt;
	while (*c)
	{
		if (*c == '%') { /* CHECK_EQ(c[1], '%'); */ ++c; }
		*s += *c++;
	}
}

inline std::string copyToFormatString( const char** fmt_ptr, std::string* s )
{
	const char*& fmt = *fmt_ptr;
	while (*fmt)
	{
		if (*fmt != '%') *s += *fmt, ++fmt; else
			if (fmt[1] == '%') *s += '%', *s += '%', fmt += 2;
			else /* fmt is at the start of a formatting directive. */ break;
	}
	std::string nextFmt;
	if (*fmt)
	{
		do { nextFmt += *fmt, ++fmt; } while (*fmt && *fmt != '%' && !isspace( *fmt ) && *fmt != ',' &&
			*fmt != '[' && *fmt != ']' && *fmt != '(' && *fmt != ')');
	}
	return nextFmt;
}

template <typename T> inline std::string formatOne( const char* fmt, T v )
{
	size_t size = snprintf( nullptr, 0, fmt, v ) + 1;
	std::string str;
	str.resize( size );
	snprintf( &str[0], size, fmt, v );
	str.pop_back(); // remove trailing NUL
	return str;
}

template <typename T, typename... Args> inline void stringPrintfRecursive( std::string* s, const char* fmt, T v, Args... args )
{
	std::string nextFmt = copyToFormatString( &fmt, s );
	*s += formatOne( nextFmt.c_str(), v );
	stringPrintfRecursive( s, fmt, args... );
}

template <typename... Args> inline void stringPrintfRecursive( std::string* s, const char* fmt, float v, Args... args )
{
	std::string nextFmt = copyToFormatString( &fmt, s );
	if (nextFmt == "%f") *s += formatOne( "%.9g", v );
	else *s += formatOne( nextFmt.c_str(), v );
	stringPrintfRecursive( s, fmt, args... );
}

template <typename... Args> inline void stringPrintfRecursive( std::string* s, const char* fmt, double v, Args... args )
{
	std::string nextFmt = copyToFormatString( &fmt, s );
	if (nextFmt == "%f") *s += formatOne( "%.17g", v );
	else *s += formatOne( nextFmt.c_str(), v );
	stringPrintfRecursive( s, fmt, args... );
}

template <typename... Args> inline std::string StringPrintf( const char* fmt, Args... args )
{
	std::string ret;
	stringPrintfRecursive( &ret, fmt, args... );
	return ret;
}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif // __GNUG__

struct Loc
{
	Loc() = default;
	Loc( const std::string& filename ) : filename( filename ) {}
	std::string filename;
	int line = 1, column = 0;
};

extern Loc* parserLoc;

class string_view
{
public:
	string_view( const char* start, size_t size ) : ptr( start ), length( size ) {}
	string_view() : ptr( nullptr ), length( 0 ) {}
	const char* data() const { return ptr; }
	size_t size() const { return length; }
	bool empty() const { return length == 0; }
	char operator[]( int index ) const { return ptr[index]; }
	char back() const { return ptr[length - 1]; }
	const char* begin() const { return ptr; }
	const char* end() const { return ptr + length; }
	bool operator==( const char* str ) const
	{
		int index;
		for (index = 0; *str; ++index, ++str)
		{
			if (index >= length) return false;
			if (*str != ptr[index]) return false;
		}
		return index == length;
	}
	bool operator!=( const char* str ) const { return !(*this == str); }
	void remove_prefix( int n ) { ptr += n; length -= n; }
	void remove_suffix( int n ) { length -= n; }
private:
	const char* ptr;
	size_t length;
};

class Tokenizer
{
public:
	static std::unique_ptr<Tokenizer> CreateFromFile( const std::string& filename,
		std::function<void( const char* )> errorCallback );
	static std::unique_ptr<Tokenizer> CreateFromString(
		std::string str, std::function<void( const char* )> errorCallback );
	~Tokenizer();
	string_view Next();
	Loc loc;
private:
	Tokenizer( std::string str, std::function<void( const char* )> errorCallback );
#if defined( PBRT_HAVE_MMAP ) || defined( PBRT_IS_WINDOWS )
	Tokenizer( void* ptr, size_t len, std::string filename, std::function<void( const char* )> errorCallback );
#endif
	int getChar()
	{
		if (pos == end) return EOF;
		int ch = *pos++;
		if (ch == '\n') ++loc.line, loc.column = 0; else ++loc.column;
		return ch;
	}
	void ungetChar()
	{
		--pos;
		if (*pos == '\n') --loc.line;
	}
	std::function<void( const char* )> errorCallback;
#if defined( PBRT_HAVE_MMAP ) || defined( PBRT_IS_WINDOWS )
	void* unmapPtr = nullptr;
	size_t unmapLength = 0;
#endif
	std::string contents;
	const char *pos, *end;
	std::string sEscaped;
};

} // namespace pbrt

// #endif // PBRT_CORE_PARSER_H