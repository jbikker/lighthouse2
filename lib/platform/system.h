/* system.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file:

   Platform-specific header-, class- and function declarations.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <half.hpp>
#include <ppl.h>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

using namespace std;
using namespace half_float;

#include "immintrin.h"
#include "emmintrin.h"
#include "common_types.h"
#include "common_settings.h"
#include "common_classes.h"
#include <GLFW/glfw3.h>		// needed for Timer class

// https://devblogs.microsoft.com/cppblog/msvc-preprocessor-progress-towards-conformance/
// MSVC _Should_ support this extended functionality for the token-paste operator:
#define FATALERROR( fmt, ... ) FatalError( "Error on line %d of %s: " fmt "\n", __LINE__, __FILE__, ##__VA_ARGS__ )
#define FATALERROR_IF( condition, fmt, ... ) do { if ( ( condition ) ) FATALERROR( fmt, ##__VA_ARGS__ ); } while ( 0 )

#define FATALERROR_IN( prefix, errstr, fmt, ... )                \
	FatalError( prefix " returned error '%s' at %s:%d" fmt "\n", \
				errstr, __FILE__, __LINE__,                      \
				##__VA_ARGS__ );

// Fatal error helper. Executes statement and throws fatal error on non-zero result.
// The result is converted to string by calling error_parser( ret )
#define FATALERROR_IN_CALL( stmt, error_parser, fmt, ... )                         \
	do                                                                             \
	{                                                                              \
		auto ret = ( stmt );                                                       \
		if ( ret ) FATALERROR_IN( #stmt, error_parser( ret ), fmt, ##__VA_ARGS__ ) \
	} while ( 0 )

#ifdef _MSC_VER
#define ALIGN( x ) __declspec( align( x ) )
#define MALLOC64( x ) ((x)==0?0:_aligned_malloc((x),64))
#define FREE64( x ) _aligned_free( x )
#else
#define ALIGN( x ) __attribute__( ( aligned( x ) ) )
#define MALLOC64( x ) ((x)==0?0:aligned_alloc(64, (x)))
#define FREE64( x ) free( x )
#endif

// threading
class Thread
{
public:
	void start();
	inline virtual void run() {};
	std::thread thread;
};
extern "C" { uint sthread_proc( void* param ); }

// timer
struct Timer
{
	Timer() { reset(); }
	float elapsed() const
	{
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - start);
		return (float)time_span.count();
	}
	void reset()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	std::chrono::high_resolution_clock::time_point start;
};

#define wrap(x,a,b) (((x)>=(a))?((x)<=(b)?(x):((x)-((b)-(a)))):((x)+((b)-(a))))
__inline float sqr( const float x ) { return x * x; }
template <class T> void Swap( T& x, T& y ) { T t; t = x; x = y; y = t; }

// crc64, from https://sourceforge.net/projects/crc64/
#define UINT64C(x) ((uint64_t) x##ULL)
#define CLEARCRC64 (UINT64C( 0xffffffffffffffff ))
const uint64_t crc64_table[256] = {
	UINT64C( 0x0000000000000000 ), UINT64C( 0x42F0E1EBA9EA3693 ), UINT64C( 0x85E1C3D753D46D26 ), UINT64C( 0xC711223CFA3E5BB5 ),
	UINT64C( 0x493366450E42ECDF ), UINT64C( 0x0BC387AEA7A8DA4C ), UINT64C( 0xCCD2A5925D9681F9 ), UINT64C( 0x8E224479F47CB76A ),
	UINT64C( 0x9266CC8A1C85D9BE ), UINT64C( 0xD0962D61B56FEF2D ), UINT64C( 0x17870F5D4F51B498 ), UINT64C( 0x5577EEB6E6BB820B ),
	UINT64C( 0xDB55AACF12C73561 ), UINT64C( 0x99A54B24BB2D03F2 ), UINT64C( 0x5EB4691841135847 ), UINT64C( 0x1C4488F3E8F96ED4 ),
	UINT64C( 0x663D78FF90E185EF ), UINT64C( 0x24CD9914390BB37C ), UINT64C( 0xE3DCBB28C335E8C9 ), UINT64C( 0xA12C5AC36ADFDE5A ),
	UINT64C( 0x2F0E1EBA9EA36930 ), UINT64C( 0x6DFEFF5137495FA3 ), UINT64C( 0xAAEFDD6DCD770416 ), UINT64C( 0xE81F3C86649D3285 ),
	UINT64C( 0xF45BB4758C645C51 ), UINT64C( 0xB6AB559E258E6AC2 ), UINT64C( 0x71BA77A2DFB03177 ), UINT64C( 0x334A9649765A07E4 ),
	UINT64C( 0xBD68D2308226B08E ), UINT64C( 0xFF9833DB2BCC861D ), UINT64C( 0x388911E7D1F2DDA8 ), UINT64C( 0x7A79F00C7818EB3B ),
	UINT64C( 0xCC7AF1FF21C30BDE ), UINT64C( 0x8E8A101488293D4D ), UINT64C( 0x499B3228721766F8 ), UINT64C( 0x0B6BD3C3DBFD506B ),
	UINT64C( 0x854997BA2F81E701 ), UINT64C( 0xC7B97651866BD192 ), UINT64C( 0x00A8546D7C558A27 ), UINT64C( 0x4258B586D5BFBCB4 ),
	UINT64C( 0x5E1C3D753D46D260 ), UINT64C( 0x1CECDC9E94ACE4F3 ), UINT64C( 0xDBFDFEA26E92BF46 ), UINT64C( 0x990D1F49C77889D5 ),
	UINT64C( 0x172F5B3033043EBF ), UINT64C( 0x55DFBADB9AEE082C ), UINT64C( 0x92CE98E760D05399 ), UINT64C( 0xD03E790CC93A650A ),
	UINT64C( 0xAA478900B1228E31 ), UINT64C( 0xE8B768EB18C8B8A2 ), UINT64C( 0x2FA64AD7E2F6E317 ), UINT64C( 0x6D56AB3C4B1CD584 ),
	UINT64C( 0xE374EF45BF6062EE ), UINT64C( 0xA1840EAE168A547D ), UINT64C( 0x66952C92ECB40FC8 ), UINT64C( 0x2465CD79455E395B ),
	UINT64C( 0x3821458AADA7578F ), UINT64C( 0x7AD1A461044D611C ), UINT64C( 0xBDC0865DFE733AA9 ), UINT64C( 0xFF3067B657990C3A ),
	UINT64C( 0x711223CFA3E5BB50 ), UINT64C( 0x33E2C2240A0F8DC3 ), UINT64C( 0xF4F3E018F031D676 ), UINT64C( 0xB60301F359DBE0E5 ),
	UINT64C( 0xDA050215EA6C212F ), UINT64C( 0x98F5E3FE438617BC ), UINT64C( 0x5FE4C1C2B9B84C09 ), UINT64C( 0x1D14202910527A9A ),
	UINT64C( 0x93366450E42ECDF0 ), UINT64C( 0xD1C685BB4DC4FB63 ), UINT64C( 0x16D7A787B7FAA0D6 ), UINT64C( 0x5427466C1E109645 ),
	UINT64C( 0x4863CE9FF6E9F891 ), UINT64C( 0x0A932F745F03CE02 ), UINT64C( 0xCD820D48A53D95B7 ), UINT64C( 0x8F72ECA30CD7A324 ),
	UINT64C( 0x0150A8DAF8AB144E ), UINT64C( 0x43A04931514122DD ), UINT64C( 0x84B16B0DAB7F7968 ), UINT64C( 0xC6418AE602954FFB ),
	UINT64C( 0xBC387AEA7A8DA4C0 ), UINT64C( 0xFEC89B01D3679253 ), UINT64C( 0x39D9B93D2959C9E6 ), UINT64C( 0x7B2958D680B3FF75 ),
	UINT64C( 0xF50B1CAF74CF481F ), UINT64C( 0xB7FBFD44DD257E8C ), UINT64C( 0x70EADF78271B2539 ), UINT64C( 0x321A3E938EF113AA ),
	UINT64C( 0x2E5EB66066087D7E ), UINT64C( 0x6CAE578BCFE24BED ), UINT64C( 0xABBF75B735DC1058 ), UINT64C( 0xE94F945C9C3626CB ),
	UINT64C( 0x676DD025684A91A1 ), UINT64C( 0x259D31CEC1A0A732 ), UINT64C( 0xE28C13F23B9EFC87 ), UINT64C( 0xA07CF2199274CA14 ),
	UINT64C( 0x167FF3EACBAF2AF1 ), UINT64C( 0x548F120162451C62 ), UINT64C( 0x939E303D987B47D7 ), UINT64C( 0xD16ED1D631917144 ),
	UINT64C( 0x5F4C95AFC5EDC62E ), UINT64C( 0x1DBC74446C07F0BD ), UINT64C( 0xDAAD56789639AB08 ), UINT64C( 0x985DB7933FD39D9B ),
	UINT64C( 0x84193F60D72AF34F ), UINT64C( 0xC6E9DE8B7EC0C5DC ), UINT64C( 0x01F8FCB784FE9E69 ), UINT64C( 0x43081D5C2D14A8FA ),
	UINT64C( 0xCD2A5925D9681F90 ), UINT64C( 0x8FDAB8CE70822903 ), UINT64C( 0x48CB9AF28ABC72B6 ), UINT64C( 0x0A3B7B1923564425 ),
	UINT64C( 0x70428B155B4EAF1E ), UINT64C( 0x32B26AFEF2A4998D ), UINT64C( 0xF5A348C2089AC238 ), UINT64C( 0xB753A929A170F4AB ),
	UINT64C( 0x3971ED50550C43C1 ), UINT64C( 0x7B810CBBFCE67552 ), UINT64C( 0xBC902E8706D82EE7 ), UINT64C( 0xFE60CF6CAF321874 ),
	UINT64C( 0xE224479F47CB76A0 ), UINT64C( 0xA0D4A674EE214033 ), UINT64C( 0x67C58448141F1B86 ), UINT64C( 0x253565A3BDF52D15 ),
	UINT64C( 0xAB1721DA49899A7F ), UINT64C( 0xE9E7C031E063ACEC ), UINT64C( 0x2EF6E20D1A5DF759 ), UINT64C( 0x6C0603E6B3B7C1CA ),
	UINT64C( 0xF6FAE5C07D3274CD ), UINT64C( 0xB40A042BD4D8425E ), UINT64C( 0x731B26172EE619EB ), UINT64C( 0x31EBC7FC870C2F78 ),
	UINT64C( 0xBFC9838573709812 ), UINT64C( 0xFD39626EDA9AAE81 ), UINT64C( 0x3A28405220A4F534 ), UINT64C( 0x78D8A1B9894EC3A7 ),
	UINT64C( 0x649C294A61B7AD73 ), UINT64C( 0x266CC8A1C85D9BE0 ), UINT64C( 0xE17DEA9D3263C055 ), UINT64C( 0xA38D0B769B89F6C6 ),
	UINT64C( 0x2DAF4F0F6FF541AC ), UINT64C( 0x6F5FAEE4C61F773F ), UINT64C( 0xA84E8CD83C212C8A ), UINT64C( 0xEABE6D3395CB1A19 ),
	UINT64C( 0x90C79D3FEDD3F122 ), UINT64C( 0xD2377CD44439C7B1 ), UINT64C( 0x15265EE8BE079C04 ), UINT64C( 0x57D6BF0317EDAA97 ),
	UINT64C( 0xD9F4FB7AE3911DFD ), UINT64C( 0x9B041A914A7B2B6E ), UINT64C( 0x5C1538ADB04570DB ), UINT64C( 0x1EE5D94619AF4648 ),
	UINT64C( 0x02A151B5F156289C ), UINT64C( 0x4051B05E58BC1E0F ), UINT64C( 0x87409262A28245BA ), UINT64C( 0xC5B073890B687329 ),
	UINT64C( 0x4B9237F0FF14C443 ), UINT64C( 0x0962D61B56FEF2D0 ), UINT64C( 0xCE73F427ACC0A965 ), UINT64C( 0x8C8315CC052A9FF6 ),
	UINT64C( 0x3A80143F5CF17F13 ), UINT64C( 0x7870F5D4F51B4980 ), UINT64C( 0xBF61D7E80F251235 ), UINT64C( 0xFD913603A6CF24A6 ),
	UINT64C( 0x73B3727A52B393CC ), UINT64C( 0x31439391FB59A55F ), UINT64C( 0xF652B1AD0167FEEA ), UINT64C( 0xB4A25046A88DC879 ),
	UINT64C( 0xA8E6D8B54074A6AD ), UINT64C( 0xEA16395EE99E903E ), UINT64C( 0x2D071B6213A0CB8B ), UINT64C( 0x6FF7FA89BA4AFD18 ),
	UINT64C( 0xE1D5BEF04E364A72 ), UINT64C( 0xA3255F1BE7DC7CE1 ), UINT64C( 0x64347D271DE22754 ), UINT64C( 0x26C49CCCB40811C7 ),
	UINT64C( 0x5CBD6CC0CC10FAFC ), UINT64C( 0x1E4D8D2B65FACC6F ), UINT64C( 0xD95CAF179FC497DA ), UINT64C( 0x9BAC4EFC362EA149 ),
	UINT64C( 0x158E0A85C2521623 ), UINT64C( 0x577EEB6E6BB820B0 ), UINT64C( 0x906FC95291867B05 ), UINT64C( 0xD29F28B9386C4D96 ),
	UINT64C( 0xCEDBA04AD0952342 ), UINT64C( 0x8C2B41A1797F15D1 ), UINT64C( 0x4B3A639D83414E64 ), UINT64C( 0x09CA82762AAB78F7 ),
	UINT64C( 0x87E8C60FDED7CF9D ), UINT64C( 0xC51827E4773DF90E ), UINT64C( 0x020905D88D03A2BB ), UINT64C( 0x40F9E43324E99428 ),
	UINT64C( 0x2CFFE7D5975E55E2 ), UINT64C( 0x6E0F063E3EB46371 ), UINT64C( 0xA91E2402C48A38C4 ), UINT64C( 0xEBEEC5E96D600E57 ),
	UINT64C( 0x65CC8190991CB93D ), UINT64C( 0x273C607B30F68FAE ), UINT64C( 0xE02D4247CAC8D41B ), UINT64C( 0xA2DDA3AC6322E288 ),
	UINT64C( 0xBE992B5F8BDB8C5C ), UINT64C( 0xFC69CAB42231BACF ), UINT64C( 0x3B78E888D80FE17A ), UINT64C( 0x7988096371E5D7E9 ),
	UINT64C( 0xF7AA4D1A85996083 ), UINT64C( 0xB55AACF12C735610 ), UINT64C( 0x724B8ECDD64D0DA5 ), UINT64C( 0x30BB6F267FA73B36 ),
	UINT64C( 0x4AC29F2A07BFD00D ), UINT64C( 0x08327EC1AE55E69E ), UINT64C( 0xCF235CFD546BBD2B ), UINT64C( 0x8DD3BD16FD818BB8 ),
	UINT64C( 0x03F1F96F09FD3CD2 ), UINT64C( 0x41011884A0170A41 ), UINT64C( 0x86103AB85A2951F4 ), UINT64C( 0xC4E0DB53F3C36767 ),
	UINT64C( 0xD8A453A01B3A09B3 ), UINT64C( 0x9A54B24BB2D03F20 ), UINT64C( 0x5D45907748EE6495 ), UINT64C( 0x1FB5719CE1045206 ),
	UINT64C( 0x919735E51578E56C ), UINT64C( 0xD367D40EBC92D3FF ), UINT64C( 0x1476F63246AC884A ), UINT64C( 0x568617D9EF46BED9 ),
	UINT64C( 0xE085162AB69D5E3C ), UINT64C( 0xA275F7C11F7768AF ), UINT64C( 0x6564D5FDE549331A ), UINT64C( 0x279434164CA30589 ),
	UINT64C( 0xA9B6706FB8DFB2E3 ), UINT64C( 0xEB46918411358470 ), UINT64C( 0x2C57B3B8EB0BDFC5 ), UINT64C( 0x6EA7525342E1E956 ),
	UINT64C( 0x72E3DAA0AA188782 ), UINT64C( 0x30133B4B03F2B111 ), UINT64C( 0xF7021977F9CCEAA4 ), UINT64C( 0xB5F2F89C5026DC37 ),
	UINT64C( 0x3BD0BCE5A45A6B5D ), UINT64C( 0x79205D0E0DB05DCE ), UINT64C( 0xBE317F32F78E067B ), UINT64C( 0xFCC19ED95E6430E8 ),
	UINT64C( 0x86B86ED5267CDBD3 ), UINT64C( 0xC4488F3E8F96ED40 ), UINT64C( 0x0359AD0275A8B6F5 ), UINT64C( 0x41A94CE9DC428066 ),
	UINT64C( 0xCF8B0890283E370C ), UINT64C( 0x8D7BE97B81D4019F ), UINT64C( 0x4A6ACB477BEA5A2A ), UINT64C( 0x089A2AACD2006CB9 ),
	UINT64C( 0x14DEA25F3AF9026D ), UINT64C( 0x562E43B4931334FE ), UINT64C( 0x913F6188692D6F4B ), UINT64C( 0xD3CF8063C0C759D8 ),
	UINT64C( 0x5DEDC41A34BBEEB2 ), UINT64C( 0x1F1D25F19D51D821 ), UINT64C( 0xD80C07CD676F8394 ), UINT64C( 0x9AFCE626CE85B507 )
};
__inline uint64_t calccrc64( unsigned char* pbData, int len )
{
	uint64_t crc = CLEARCRC64;
	unsigned char* p = pbData;
	unsigned int t, l = len;
	while (l-- > 0)
		t = ((uint)(crc >> 56) ^ *p++) & 255,
		crc = crc64_table[t] ^ (crc << 8);
	return crc ^ CLEARCRC64;
}
#define TRACKCHANGES public: bool Changed() { uint64_t currentcrc = crc64; \
crc64 = CLEARCRC64; uint64_t newcrc = calccrc64( (uchar*)this, sizeof( *this ) ); \
bool changed = newcrc != currentcrc; crc64 = newcrc; return changed; } \
bool IsDirty() { uint64_t t = crc64; bool c = Changed(); crc64 = t; return c; } \
void MarkAsDirty() { dirty++; } \
void MarkAsNotDirty() { Changed(); } \
private: uint64_t crc64 = CLEARCRC64; uint dirty = 0; \

// rng
uint RandomUInt();
uint RandomUInt( uint& seed );
float RandomFloat();
float RandomFloat( uint& seed );
float Rand( float range );

// forward declaration of the helper functions
void FatalError( const char* fmt, ... );
void OpenConsole();
bool FileIsNewer( const char* file1, const char* file2 );
bool NeedsRecompile( const char* path, const char* target, const char* s1, const char* s2 = 0, const char* s3 = 0, const char* s4 = 0 );
bool FileExists( const char* f );
bool RemoveFile( const char* f);
string TextFileRead( const char* _File );
void TextFileWrite( const string& text, const char* _File );
string LowerCase( string s );
void SerializeString( string s, FILE* f );
string DeserializeString( FILE* f );

// globally accessible classes
namespace lighthouse2
{

class Bitmap
{
public:
	Bitmap() = default;
	Bitmap( const char* f );
	Bitmap( uint w, uint h ) : pixels( new uint[w * h] ), width( w ), height( h ) {}
	~Bitmap() { delete pixels; }
	void Plot( uint x, uint y, uint c ) { if (x < width && y < height) pixels[x + y * width] = c; }
	void Clear() { memset( pixels, 0, width * height * 4 ); }
	uint* pixels = nullptr;
	uint width = 0, height = 0;
};

class GLTexture
{
public:
	enum
	{
		DEFAULT = 0,
		FLOAT = 1
	};
	// constructor / destructor
	GLTexture( uint width, uint height, uint type = DEFAULT );
	GLTexture( const char* fileName, int filter = GL_NEAREST );
	~GLTexture();
	// methods
	void Bind();
	void CopyFrom( Bitmap* src );
	void CopyTo( Bitmap* dst );
	// public data members
public:
	GLuint ID = 0;
	uint width = 0, height = 0;
};

} // namespace lighthouse2

// library namespace
using namespace lighthouse2;

// https://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux
#if defined(_MSC_VER)
	//  Microsoft
#define COREDLL_EXPORT __declspec(dllexport)
#define COREDLL_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
	//  GCC
#define COREDLL_EXPORT __attribute__((visibility("default")))
#define COREDLL_IMPORT
#else
	//  do nothing and hope for the best?
#define COREDLL_EXPORT
#define COREDLL_IMPORT
#pragma warning Unknown dynamic link import/export semantics.
#endif

#ifdef COREDLL_EXPORTS
#define COREDLL_API COREDLL_EXPORT
#else
#define COREDLL_API COREDLL_IMPORT
#endif

// EOF