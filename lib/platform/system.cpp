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

// Enable usage of dedicated GPUs in notebooks
#ifdef WIN32
extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

extern "C"
{
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif
#include <ft2build.h>
#include FT_FREETYPE_H

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
//  |  Bitmap functions.                                                    LH2'19|
//  +-----------------------------------------------------------------------------+
Bitmap::Bitmap( const char* f )
{
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType( f, 0 );
	if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename( f );
	FIBITMAP* tmp = FreeImage_Load( fif, f );
	FIBITMAP* dib = FreeImage_ConvertTo32Bits( tmp );
	FreeImage_Unload( tmp );
	width = FreeImage_GetWidth( dib );
	height = FreeImage_GetHeight( dib );
	pixels = (uint*)MALLOC64( width * height * sizeof( uint ) );
	for (uint y = 0; y < height; y++)
	{
		unsigned const char* line = FreeImage_GetScanLine( dib, height - 1 - y );
		memcpy( pixels + y * width, line, width * sizeof( uint ) );
	}
	FreeImage_Unload( dib );
}

//  +-----------------------------------------------------------------------------+
//  |  GLTextRenderer implementation.                                       LH2'20|
//  +-----------------------------------------------------------------------------+
int GLTextRenderer::scrwidth = SCRWIDTH;
int GLTextRenderer::scrheight = SCRHEIGHT;
GLTextRenderer::GLTextRenderer( const int size, const char* font )
{
	// initialize FreeType2, based on https://learnopengl.com/In-Practice/Text-Rendering
	FT_Library ft;
	FT_Face face;
	if (FT_Init_FreeType( &ft )) FatalError( "Could not initialize FreeType2." );
	if (FT_New_Face( ft, font, 0, &face )) FatalError( "Could not load font." );
	FT_Set_Pixel_Sizes( face, 0, size );
	shader = new Shader( "shaders/freetype2.vert", "shaders/freetype2.frag" );
	// prepare charset textures
	glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ); // disable byte-alignment restriction
	for (GLubyte c = 0; c < 128; c++)
	{
		if (FT_Load_Char( face, c, FT_LOAD_RENDER )) FatalError( "failed to load glyph" );
		GLuint texture;
		glGenTextures( 1, &texture );
		glBindTexture( GL_TEXTURE_2D, texture );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows,
			0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		Character character = { texture,
			make_int2( face->glyph->bitmap.width, face->glyph->bitmap.rows ),
			make_int2( face->glyph->bitmap_left, face->glyph->bitmap_top ),
			(uint)face->glyph->advance.x };
		Characters.insert( pair<GLchar, Character>( c, character ) );
	}
	glBindTexture( GL_TEXTURE_2D, 0 );
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4 ); // restore default byte-alignment
	// destroy FreeType once we're finished
	FT_Done_Face( face );
	FT_Done_FreeType( ft );
	// configure VAO/VBO for texture quads
	glGenVertexArrays( 1, &vao );
	glGenBuffers( 1, &vbo );
	glBindVertexArray( vao );
	glBindBuffer( GL_ARRAY_BUFFER, vbo );
	glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * 6 * 4, NULL, GL_DYNAMIC_DRAW );
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof( GLfloat ), 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );
}

void GLTextRenderer::RenderR( string text, GLfloat x, GLfloat y, GLfloat scale, const float3 color )
{
	Render( text, x, y, scale, color, true );
}

void GLTextRenderer::Render( string text, GLfloat x, GLfloat y, GLfloat scale, const float3 color, bool rightAlign )
{
	// activate corresponding render state	
	shader->Bind();
	mat4 projection = mat4::Ortho( 0.0f, (float)scrwidth, 0.0f, (float)scrheight, 0, 1 );
	glUniformMatrix4fv( glGetUniformLocation( shader->ID, "projection" ), 1, GL_FALSE, (GLfloat*)&projection );
	glUniform3f( glGetUniformLocation( shader->ID, "textColor" ), color.x, color.y, color.z );
	glActiveTexture( GL_TEXTURE0 );
	glBindVertexArray( vao );
	// calculate start x when aligning to the right
	string::const_iterator c;
	if (rightAlign) 
	{
		for (c = text.begin(); c != text.end(); c++) x -= (Characters[*c].advance >> 6)* scale;
	}
	// iterate through characters
	for (c = text.begin(); c != text.end(); c++)
	{
		Character ch = Characters[*c];
		GLfloat xpos = x + ch.bearing.x * scale;
		GLfloat ypos = scrheight - 48 * scale - y + (ch.bearing.y - ch.size.y) * scale;
		GLfloat w = ch.size.x * scale;
		GLfloat h = ch.size.y * scale;
		// update VBO for each character
		GLfloat vertices[6][4] = {
			{ xpos, ypos + h, 0.0, 0.0 }, { xpos, ypos, 0.0, 1.0 },
		{ xpos + w, ypos, 1.0, 1.0 }, { xpos, ypos + h, 0.0, 0.0 },
		{ xpos + w, ypos, 1.0, 1.0 }, { xpos + w, ypos + h, 1.0, 0.0 } };
		// render glyph texture over quad
		glBindTexture( GL_TEXTURE_2D, ch.ID );
		// update content of VBO memory
		glBindBuffer( GL_ARRAY_BUFFER, vbo );
		glBufferSubData( GL_ARRAY_BUFFER, 0, sizeof( vertices ), vertices );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
		// render quad
		glDrawArrays( GL_TRIANGLES, 0, 6 );
		// advance cursor for next glyph (in 1/64 pixels)
		x += (ch.advance >> 6)* scale;
	}
	glBindVertexArray( 0 );
	glBindTexture( GL_TEXTURE_2D, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  Helper functions.                                                    LH2'19|
//  +-----------------------------------------------------------------------------+
bool FileIsNewer( const char* file1, const char* file2 )
{
	struct stat f1;
	struct stat f2;

	auto ret = stat( file1, &f1 );
	FATALERROR_IF( ret, "File %s not found!", file1 );

	if (stat( file2, &f2 ))
		return true; // second file does not exist

#ifdef _MSC_VER
	return f1.st_mtime >= f2.st_mtime;
#else
	if (f1.st_mtim.tv_sec >= f2.st_mtim.tv_sec)
		return true;
	return f1.st_mtim.tv_nsec >= f2.st_mtim.tv_nsec;
#endif
}

bool FileExists( const char* f )
{
	std::ifstream s( f );
	return s.good();
}

bool RemoveFile( const char* f )
{
	if (!FileExists( f )) return false;
	return !remove( f );
}

uint FileSize( string filename )
{
	std::ifstream s( filename );
	return s.good();
}

string TextFileRead( const char* _File )
{
	std::ifstream s( _File );
	std::string str( (std::istreambuf_iterator<char>( s )), std::istreambuf_iterator<char>() );
	return str;
}

void TextFileWrite( const string& text, const char* _File )
{
	std::ofstream s( _File, std::ios::binary );
	int len = (int)text.size();
	s.write( (const char*)&len, sizeof( len ) );
	s.write( text.c_str(), len );
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

void FatalError( const char* fmt, ... )
{
	char t[16384];
	va_list args;
	va_start( args, fmt );
	vsnprintf( t, sizeof( t ), fmt, args );
	va_end( args );

#ifdef _MSC_VER
	MessageBox( NULL, t, "Fatal error", MB_OK );
#else
	fprintf( stderr, t );
#endif
	assert( false );
	while (1) exit( EXIT_FAILURE );
}

// EOF