/* system.cpp - Copyright 2019 Utrecht University

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

   Implementation of various generic, system-wide helper functions.
*/

#include "platform.h"

//  +-----------------------------------------------------------------------------+
//  |  RNG - Marsaglia's xor32.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
static uint seed = 0x12345678;
uint RandomUInt() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
float RandomFloat() { return RandomUInt() * 2.3283064365387e-10f; }
float Rand( float range ) { return RandomFloat() * range; }
// local seed
uint RandomUInt( uint& seed ) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
float RandomFloat( uint& seed ) { return RandomUInt( seed ) * 2.3283064365387e-10f; }

//  +-----------------------------------------------------------------------------+
//  |  Math implementations.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
mat4 operator * ( const mat4& a, const mat4& b )
{
	mat4 r;
	for (uint i = 0; i < 16; i += 4) for (uint j = 0; j < 4; ++j)
	{
		r[i + j] =
			(a.cell[i + 0] * b.cell[j + 0]) +
			(a.cell[i + 1] * b.cell[j + 4]) +
			(a.cell[i + 2] * b.cell[j + 8]) +
			(a.cell[i + 3] * b.cell[j + 12]);
	}
	return r;
}
mat4 operator * ( const mat4& a, const float s )
{
	mat4 r;
	for (uint i = 0; i < 16; i += 4) r.cell[i] = a.cell[i] * s;
	return r;
}
mat4 operator * ( const float s, const mat4& a )
{
	mat4 r;
	for (uint i = 0; i < 16; i++) r.cell[i] = a.cell[i] * s;
	return r;
}
mat4 operator + ( const mat4& a, const mat4& b )
{
	mat4 r;
	for (uint i = 0; i < 16; i += 4) r.cell[i] = a.cell[i] + b.cell[i];
	return r;
}
bool operator == ( const mat4& a, const mat4& b ) { for (uint i = 0; i < 16; i++) if (a.cell[i] != b.cell[i]) return false; return true; }
bool operator != ( const mat4& a, const mat4& b ) { return !(a == b); }
float4 operator * ( const mat4& a, const float4& b )
{
	return make_float4( a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3] * b.w,
		a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7] * b.w,
		a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11] * b.w,
		a.cell[12] * b.x + a.cell[13] * b.y + a.cell[14] * b.z + a.cell[15] * b.w );
}
float4 operator * ( const float4& b, const mat4& a )
{
	return make_float4( a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3] * b.w,
		a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7] * b.w,
		a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11] * b.w,
		a.cell[12] * b.x + a.cell[13] * b.y + a.cell[14] * b.z + a.cell[15] * b.w );
}

//  +-----------------------------------------------------------------------------+
//  |  Helper functions.                                                    LH2'19|
//  +-----------------------------------------------------------------------------+
bool FileIsNewer( const char* file1, const char* file2 )
{
	HANDLE fh1 = CreateFile( file1, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
	HANDLE fh2 = CreateFile( file2, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
	if (fh1 == INVALID_HANDLE_VALUE) FatalError( __FILE__, __LINE__, file1, "file not found" );
	if (fh2 == INVALID_HANDLE_VALUE)
	{
		CloseHandle( fh1 );
		return true; // second file does not exist
	}
	FILETIME ft1, ft2;
	GetFileTime( fh1, NULL, NULL, &ft1 );
	GetFileTime( fh2, NULL, NULL, &ft2 );
	int result = CompareFileTime( &ft1, &ft2 );
	CloseHandle( fh1 );
	CloseHandle( fh2 );
	return (result != -1);
}

bool FileExists( const char* f )
{
	HANDLE fh = CreateFile( f, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
	if (fh == INVALID_HANDLE_VALUE) return false;
	CloseHandle( fh );
	return true;
}

uint FileSize( string filename )
{
	struct stat stat_buf;
	int rc = stat( filename.c_str(), &stat_buf );
	return rc == 0 ? stat_buf.st_size : -1;
}

string TextFileRead( const char* _File )
{
	char line[16384];
	string d;
	FILE* f;
	fopen_s( &f, _File, "r" );
	while (!feof( f ))
	{
		fgets( line, 16382, f );
		d.append( line );
	}
	fclose( f );
	return d;
}

void TextFileWrite( const string& text, const char* _File )
{
	FILE* f;
	fopen_s( &f, _File, "wb" );
	int len = (int)strlen( text.c_str() ) + 1;
	fwrite( &len, 1, 4, f );
	fwrite( text.c_str(), 1, len, f );
	fclose( f );
}

string LowerCase( string s )
{
	transform( s.begin(), s.end(), s.begin(), ::tolower );
	return s;
}

void SerializeString( string s, FILE* f )
{
	uint stringLength = (uint)s.size();
	fwrite( &stringLength, 4, 1, f );
	if (stringLength > 0) fwrite( s.c_str(), 1, stringLength, f );
}

string DeserializeString( FILE* f )
{
	uint stringLength;
	fread( &stringLength, 4, 1, f );
	if (stringLength == 0) return string();
	char* rawText = new char[stringLength + 1];
	fread( rawText, 1, stringLength, f );
	rawText[stringLength] = 0;
	string retVal( rawText );
	delete rawText;
	return retVal;
}

bool NeedsRecompile( const char* path, const char* target, const char* s1, const char* s2, const char* s3, const char* s4 )
{
	string t( path );
	t.append( string( target ) );
	string d1( path );
	d1.append( string( s1 ) );
	if (!FileExists( d1.c_str() )) return false; // probably a demo without sources
	if (FileIsNewer( d1.c_str(), t.c_str() )) return true;
	if (s2) { string d2( path ); d2.append( string( s2 ) ); if (FileIsNewer( d2.c_str(), t.c_str() )) return true; }
	if (s3) { string d3( path ); d3.append( string( s3 ) ); if (FileIsNewer( d3.c_str(), t.c_str() )) return true; }
	if (s4) { string d4( path ); d4.append( string( s4 ) ); if (FileIsNewer( d4.c_str(), t.c_str() )) return true; }
	return false;
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

void FatalError( const char* source, const int line, const char* message, const char* part2 )
{
	printf( "Error executing line %i of file %s:\n%s", line, source, message );
	char t[16384];
	sprintf_s( t, 16384, "Error executing line %i of file %s:\n%s", line, source, message );
	if (part2)
	{
		strcat_s( t, "\n" );
		strcat_s( t, part2 );
	}
	MessageBox( NULL, t, "Fatal error", MB_OK );
	assert( false );
	while (1) exit( 0 );
}

void FatalError( const char* message, const char* part2 )
{
	printf( "Error: %s\n(%s)", message, part2 );
	char t[16384];
	sprintf_s( t, 16384, "Error: %s\n(%s)", message, part2 );
	MessageBox( NULL, t, "Fatal error", MB_OK );
	assert( false );
	while (1) exit( 0 );
}

// EOF