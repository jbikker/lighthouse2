/* host_skydome.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

#include <fstream>

#define SKYCDF(x,y) cdf[RadicalInverse8bit( y ) + x * (IBLHEIGHT + 1)] // columns stored sequentially for better cache coherence
#define COLCDF(x) columncdf[RadicalInverse9bit( x )]

static int RadicalInverse8bit( const int v )
{
	int x = ((v & 0xaa) >> 1) | ((v & 0x55) << 1);
	x = ((x & 0xcc) >> 2) | ((x & 0x33) << 2);
	x = ((x & 0xf0) >> 4) | ((x & 0x0f) << 4);
	return x + (1 - (v >> 8)); // so 256 = 0
}

static int RadicalInverse9bit( const int v )
{
	int x = ((v & 0xaa) >> 1) | ((v & 0x55) << 1);
	x = ((x & 0xcc) >> 2) | ((x & 0x33) << 2);
	x = ((x & 0xf0) >> 4) | ((x & 0x0f) << 4);
	return ((x << 1) | ((v >> 8) & 1)) + (1 - (v >> 9)); // so 512 = 0
}

//  +-----------------------------------------------------------------------------+
//  |  HostSkyDome::HostSkyDome                                                   |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostSkyDome::HostSkyDome()
{
	pdf = (float*)MALLOC64( IBLWIDTH * IBLHEIGHT * sizeof( float ) );
	cdf = (float*)MALLOC64( IBLWIDTH * (IBLHEIGHT + 1) * sizeof( float ) );
	columncdf = (float*)MALLOC64( (IBLWIDTH + 1) * sizeof( float ) );
}

//  +-----------------------------------------------------------------------------+
//  |  HostSkyDome::~HostSkyDome                                                  |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
HostSkyDome::~HostSkyDome()
{
	FREE64( pdf );
	FREE64( cdf );
	FREE64( columncdf );
}

//  +-----------------------------------------------------------------------------+
//  |  HostSkyDome::Load                                                          |
//  |  Load a skydome.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostSkyDome::Load()
{
	Timer timer;
	timer.reset();
	FREE64( pixels ); // just in case we're reloading
	pixels = 0;
	char t[] = "data/sky_15.hdr" /* skyBoxPath */, *p;
#ifdef TESTSKY
	// red / green / blue test environment
	width = 5120, height = 2560;
	pixels = (float3*)MALLOC64( 5120 * 2560 * sizeof( float3 ) );
	memset( pixels, 0, 5120 * 2560 * sizeof( float3 ) );
	for (int x = 0; x < 5120; x++) for (int y = 0; y < 2560; y++) pixels[x + y * 5120] = make_float3( 0.1f, 0.1f, 0.1f );
	for (int x = 0; x < 200; x++) for (int y = 900; y < 1100; y++) pixels[x + y * 5120] = make_float3( 10, 0, 0 );
	for (int x = 2000; x < 2200; x++) for (int y = 900; y < 1100; y++) pixels[x + y * 5120] = make_float3( 0, 10, 0 );
	for (int x = 4000; x < 4200; x++) for (int y = 900; y < 1100; y++) pixels[x + y * 5120] = make_float3( 0, 0, 10 );
#else
	if (p = strstr( t, ".hdr" ))
	{
		// attempt to load skydome from binary file
		memcpy( strstr( t, ".hdr" ), ".bin", 4 );
		std::ifstream f( t, std::ios::binary );
		if (f)
		{
			printf( "loading cached hdr data... " );
			f.read( (char*)&width, sizeof( width ) );
			f.read( (char*)&height, sizeof( height ) );
			// TODO: Mmap
			pixels = (float3*)MALLOC64( width * height * sizeof( float3 ) );
			f.read( (char*)pixels, sizeof( float3 ) * width * height );
		}
		else memcpy( strstr( t, ".bin" ), ".hdr", 4 );
	}
	else
	{
		printf( "bad skydome filename.\n" );
		return;
	}
#endif
	if (!pixels)
	{
		// load skydome from original .hdr file
		printf( "loading original hdr data... " );
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		fif = FreeImage_GetFileType( t, 0 );
		if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename( t );
		FIBITMAP* dib = FreeImage_Load( fif, t );
		if (!dib) return;
		width = FreeImage_GetWidth( dib );
		height = FreeImage_GetHeight( dib );
		pixels = (float3*)MALLOC64( width * height * sizeof( float3 ) );
		for (int y = 0; y < height; y++) memcpy( pixels + y * width, FreeImage_GetScanLine( dib, height - 1 - y ), width * sizeof( float3 ) );
		FreeImage_Unload( dib );
		// save skydome to binary file, .hdr is slow to load
		memcpy( strstr( t, ".hdr" ), ".bin", 4 );
		std::ofstream f( t, std::ios::binary );
		f.write( (char*)&width, sizeof( width ) );
		f.write( (char*)&height, sizeof( height ) );
		f.write( (char*)pixels, sizeof( float3 ) * width * height );
	}
#ifdef IBL
	// convert to pdf
	// see: https://www.scribd.com/document/134001376/Importance-Sampling-with-Infinite-Area-Light-Source
	// summarized in: http://cgg.mff.cuni.cz/~jaroslav/teaching/2011-pg3/ibl-writeup.pdf
	printf( "calculating sky pdf... " );
	float stepTheta = (2.0f * PI) / IBLWIDTH;			// theta: -PI...PI
	float stepPhi = PI / IBLHEIGHT;						// phi:     0...PI
	for (int p = 0; p < IBLHEIGHT; p++) // loop over rows
	{
		float scale = sinf( (float)p * (PI / IBLHEIGHT) );
		for (int t = 0; t < IBLWIDTH; t++) // loop over columns
		{
			// register scaled value
			int u = max( 0, min( width - 1, (t * width) / IBLWIDTH ) );
			int v = max( 0, min( height - 1, (p * height) / IBLHEIGHT ) );
			float3 texel = pixels[u + v * width];
			// eq. 55, http://www.igorsklyar.com/system/documents/papers/4/fiscourse.comp.pdf
			float luminance = texel.x * 0.2126f + texel.y * 0.7152f + texel.z * 0.0722f;
			pdf[t + p * IBLWIDTH] = luminance * scale;
		}
	}
	// calculate cdf
	float sum = 0;
	COLCDF( 0 ) = 0;
	for (int x = 0; x < IBLWIDTH; x++)
	{
		float columnSum = 0;
		SKYCDF( x, 0 ) = 0;
		for (int y = 0; y < IBLHEIGHT; y++) columnSum += pdf[x + y * IBLWIDTH], SKYCDF( x, y + 1 ) = columnSum;
		sum += columnSum;
		COLCDF( x + 1 ) = sum;
	}
#endif
	// done
	dirty = true;
	printf( "sky ready in %5.3fs.\n", timer.elapsed() );
}

// EOF