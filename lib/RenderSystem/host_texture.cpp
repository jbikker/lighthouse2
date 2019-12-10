/* host_texture.cpp - Copyright 2019 Utrecht University

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

// Local hack:
#ifndef _MSC_VER
#define strcat_s strcat
#endif

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::HostTexture                                                   |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostTexture::HostTexture( const char* fileName, const uint modFlags )
{
	Load( fileName, modFlags );
	origin = string( fileName );
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::ConvertToCoreTexDesc                                          |
//  |  Constructs a GPUTexture based on a HostTexture.                            |
//  |  Note that this is only a partial conversion:                               |
//  |  - texture dimensions are (also) stored in the CoreMaterial object,         |
//  |    for fast access;                                                         |
//  |  - texture pixels are stored in continuous arrays.                          |
//  |  The GPUTexture structure will be used by RenderCore to build and maintain  |
//  |  these continuous arrays however.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
CoreTexDesc HostTexture::ConvertToCoreTexDesc() const
{
	CoreTexDesc gpuTex;
	gpuTex.width = width;
	gpuTex.height = height;
	gpuTex.flags = flags;
	assert( (fdata != 0) | (idata != 0) );
	if (fdata)
	{
		gpuTex.fdata = fdata;
		gpuTex.storage = TexelStorage::ARGB128;
		gpuTex.pixelCount = PixelsNeeded( width, height, 1 );
		gpuTex.MIPlevels = 1;
		assert( (flags & NORMALMAP) == 0 );
	}
	else
	{
		gpuTex.idata = idata;
		if (flags & NORMALMAP) gpuTex.storage = TexelStorage::NRM32;
		/* else gpuTex.storage = TexelStorage::ARGB32; default */
		gpuTex.pixelCount = PixelsNeeded( width, height, MIPLEVELCOUNT );
		gpuTex.MIPlevels = MIPLEVELCOUNT;
	}
	return gpuTex;
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::sRGBtoLinear                                                  |
//  |  Convert sRGB data to linear color space.                             LH2'19|
//  +-----------------------------------------------------------------------------+
void HostTexture::sRGBtoLinear( uchar* pixels, const uint size, const uint stride )
{
	for (uint j = 0; j < size; j++)
	{
		pixels[j * stride + 0] = (pixels[j * stride + 0] * pixels[j * stride + 0]) >> 8;
		pixels[j * stride + 1] = (pixels[j * stride + 1] * pixels[j * stride + 1]) >> 8;
		pixels[j * stride + 2] = (pixels[j * stride + 2] * pixels[j * stride + 2]) >> 8;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::InverseGammaCorrect                                           |
//  |  Bring a gamma-corrected texture to linear color space.               LH2'19|
//  +-----------------------------------------------------------------------------+
float HostTexture::InverseGammaCorrect( float value )
{
	if (value <= 0.04045f) return value * 1.f / 12.92f;
	return powf( (value + 0.055f) * 1.f / 1.055f, 2.4f );
}
float4 HostTexture::InverseGammaCorrect( const float4& color )
{
	return make_float4(
		InverseGammaCorrect( color.x ),
		InverseGammaCorrect( color.y ),
		InverseGammaCorrect( color.z ), color.w );
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::Equals                                                        |
//  |  Returns true if the fields that identify the texture are identical to the  |
//  |  supplied values. Used for texture reuse by the HostScene object.     LH2'19|
//  +-----------------------------------------------------------------------------+
bool HostTexture::Equals( const string& o, const uint m )
{
	if (mods != m) return false;
	if (o.compare( origin )) return false;
	return true;
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::PixelsNeeded                                                  |
//  |  Helper function that determines the number of pixels that should be        |
//  |  allocated for the given width, height and MIP level count.           LH2'19|
//  +-----------------------------------------------------------------------------+
int HostTexture::PixelsNeeded( const int width, const int height, const int MIPlevels /* >= 1; includes base layer */ ) const
{
	int w = width, h = height, needed = 0;
	for (int i = 0; i < MIPlevels; i++) needed += w * h, w >>= 1, h >>= 1;
	return needed;
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::ConstructMIPmaps                                              |
//  |  Generate MIP levels for a loaded texture.                            LH2'19|
//  +-----------------------------------------------------------------------------+
void HostTexture::ConstructMIPmaps()
{
	uint* src = (uint*)idata;
	uint* dst = src + width * height;
	int pw = width, w = width >> 1, ph = height, h = height >> 1;
	for (int i = 1; i < MIPLEVELCOUNT; i++)
	{
		// reduce
		for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
		{
			const uint src0 = src[x * 2 + (y * 2) * pw];
			const uint src1 = src[x * 2 + 1 + (y * 2) * pw];
			const uint src2 = src[x * 2 + (y * 2 + 1) * pw];
			const uint src3 = src[x * 2 + 1 + (y * 2 + 1) * pw];
			const uint a = min( min( (src0 >> 24) & 255, (src1 >> 24) & 255 ), min( (src2 >> 24) & 255, (src3 >> 24) & 255 ) );
			const uint r = ((src0 >> 16) & 255) + ((src1 >> 16) & 255) + ((src2 >> 16) & 255) + ((src3 >> 16) & 255);
			const uint g = ((src0 >> 8) & 255) + ((src1 >> 8) & 255) + ((src2 >> 8) & 255) + ((src3 >> 8) & 255);
			const uint b = (src0 & 255) + (src1 & 255) + (src2 & 255) + (src3 & 255);
			dst[x + y * w] = (a << 24) + ((r >> 2) << 16) + ((g >> 2) << 8) + (b >> 2);
		}
		// next layer
		src = dst, dst += w * h, pw = w, ph = h, w >>= 1, h >>= 1;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::Load                                                          |
//  |  Load texture data from disk.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void HostTexture::Load( const char* fileName, const uint modFlags, bool normalMap )
{
	// check if texture exists
	FATALERROR_IF( !FileExists( fileName ), "File %s not found", fileName );

#ifdef CACHEIMAGES
	// see if we can fetch a binary blob; faster than most FreeImage formats
	if (strlen( fileName ) > 4) if (fileName[strlen( fileName ) - 4] == '.')
	{
		char binFile[1024];
		memcpy( binFile, fileName, strlen( fileName ) + 1 );
		binFile[strlen( fileName ) - 4] = 0;
		strcat_s( binFile, ".bin" );
		FILE* f;
	#ifdef _MSC_VER
		fopen_s( &f, binFile, "rb" );
	#else
		f = fopen( binFile, "rb" );
	#endif
		if (f)
		{
			uint version;
			fread( &version, 1, 4, f );
			if (version == BINTEXFILEVERSION)
			{
				fread( &width, 4, 1, f );
				fread( &height, 4, 1, f );
				int dataType;
				fread( &dataType, 4, 1, f );
				fread( &mods, 4, 1, f );
				fread( &flags, 4, 1, f );
				fread( &MIPlevels, 4, 1, f );
				if (dataType == 0)
				{
					int pixelCount = PixelsNeeded( width, height, 1 /* no MIPS for HDR textures */ );
					fdata = (float4*)MALLOC64( sizeof( float4 ) * pixelCount );
					fread( fdata, sizeof( float4 ), pixelCount, f );
				}
				else
				{
					int pixelCount = PixelsNeeded( width, height, MIPLEVELCOUNT );
					idata = (uchar4*)MALLOC64( sizeof( uchar4 ) * pixelCount );
					fread( idata, 4, pixelCount, f );
				}
				fclose( f );
				mods = modFlags;
				return;
			}
		}
	}
#endif
	// get filetype
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType( fileName, 0 );
	if (fif == FIF_UNKNOWN) fif = FreeImage_GetFIFFromFilename( fileName );
	FATALERROR_IF( fif == FIF_UNKNOWN, "%s contains an unsupported texture filetype", fileName );
	// load image
	FIBITMAP* tmp = FreeImage_Load( fif, fileName );
	FIBITMAP* img = FreeImage_ConvertTo32Bits( tmp ); // converts 1 4 8 16 24 32 48 64 bpp to 32 bpp, fails otherwise
	if (!img) img = tmp;
	width = FreeImage_GetWidth( img );
	height = FreeImage_GetHeight( img );
	mods = modFlags;
	uint pitch = FreeImage_GetPitch( img );
	BYTE* bytes = (BYTE*)FreeImage_GetBits( img );
	uint bpp = FreeImage_GetBPP( img );
	FIBITMAP* alpha = FreeImage_GetChannel( img, FICC_ALPHA );
	if (alpha)
	{
		// set alpha rendering for this texture to true if it contains a meaningful alpha channel
		DWORD histogram[256];
		if (FreeImage_GetHistogram( alpha, histogram )) if (histogram[0] > 0) flags |= HASALPHA;
		FreeImage_Unload( alpha );
	}
	// iterate image pixels and write to LightHouse internal format
	if (bpp == 32) // LDR
	{
		// invert image if requested
		if (mods & INVERTED) FreeImage_Invert( img );
		// read pixels
		idata = (uchar4*)MALLOC64( sizeof( uchar4 ) * PixelsNeeded( width, height, MIPLEVELCOUNT ) );
		flags |= LDR;
		for (uint y = 0; y < height; y++, bytes += pitch) for (uint x = 0; x < width; x++)
		{
			// convert from FreeImage's 32-bit image format (usually BGRA) to 32-bit RGBA
			uchar *pixel = &((uchar*)bytes)[x * 4];
			uchar4 rgba = make_uchar4( pixel[FI_RGBA_RED], pixel[FI_RGBA_GREEN], pixel[FI_RGBA_BLUE], pixel[FI_RGBA_ALPHA] );
			(mods & FLIPPED) ? idata[(y * width) + x] = rgba : idata[((height - 1 - y) * width) + x] = rgba;  // FreeImage stores the data upside down by default
		}
		// perform sRGB -> linear conversion if requested
		if (mods & LINEARIZED) sRGBtoLinear( (uchar*)idata, width * height, 4 );
		// produce the MIP maps
		ConstructMIPmaps();
	}
	else // HDR
	{
		fdata = (float4*)MALLOC64( sizeof( float4 ) * PixelsNeeded( width, height, 1 /* no MIPs for HDR for now */ ) );
		flags |= HDR;
		for (uint y = 0; y < height; y++, bytes += pitch) for (uint x = 0; x < width; x++)
		{
			float4 rgba;
			if (bpp == 96) rgba = make_float4( ((float3*)bytes)[x], 1.0f );	// 96-bit RGB, append alpha channel
			else if (bpp == 128) rgba = ((float4*)bytes)[x];				// 128-bit RGBA
			(mods & FLIPPED) ? fdata[(y * width) + x] = rgba : fdata[((height - 1 - y) * width) + x] = rgba; // FreeImage stores the data upside down by default
		}
	}
	// mark normal map
	if (normalMap) flags |= NORMALMAP;
	// unload
	FreeImage_Unload( img ); if (bpp == 32) FreeImage_Unload( tmp );

	// perform gamma correction
	if (mods & GAMMACORRECTION)
	{
		for (uint p = 0; p < width * height; ++p)
		{
			if (flags & HDR)
				fdata[p] = InverseGammaCorrect( fdata[p] );
			else
			{
				const auto convert = []( const uchar in ) {
					return (uchar)clamp( InverseGammaCorrect( in / 255.f ) * 255.f, 0.f, 255.f );
				};

				const auto in = idata[p];
				idata[p] = make_uchar4(
					convert( in.x ),
					convert( in.y ),
					convert( in.z ),
					in.w );
			}
		}
	}

#ifdef CACHEIMAGES
	// prepare binary blob to be faster next time
	if (strlen( fileName ) > 4) if (fileName[strlen( fileName ) - 4] == '.')
	{
		char binFile[1024];
		memcpy( binFile, fileName, strlen( fileName ) + 1 );
		binFile[strlen( fileName ) - 4] = 0;
		strcat_s( binFile, ".bin" );
		FILE* f;
	#ifdef _MSC_VER
		fopen_s( &f, binFile, "rb" );
	#else
		f = fopen( binFile, "rb" );
	#endif
		if (f)
		{
			uint version = BINTEXFILEVERSION;
			fwrite( &version, 4, 1, f );
			fwrite( &width, 4, 1, f );
			fwrite( &height, 4, 1, f );
			int dataType = fdata ? 0 : 1;
			fwrite( &dataType, 4, 1, f );
			fwrite( &mods, 4, 1, f );
			fwrite( &flags, 4, 1, f );
			fwrite( &MIPlevels, 4, 1, f );
			if (dataType == 0) fwrite( fdata, sizeof( float4 ), PixelsNeeded( width, height, 1 ), f );
			else fwrite( idata, 4, PixelsNeeded( width, height, MIPLEVELCOUNT ), f );
			fclose( f );
		}
	}
#endif
	// all done, mark for sync with core
}

//  +-----------------------------------------------------------------------------+
//  |  HostTexture::BumpToNormalMap                                               |
//  |  Convert a bumpmap to a normalmap.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostTexture::BumpToNormalMap( float heightScale )
{
	uchar* normalMap = new uchar[width * height * 4];
	const float stepZ = 1.0f / 255.0f;
	for (uint i = 0; i < width * height; i++)
	{
		uint xCoord = i % width, yCoord = i / width;
		float xPrev = xCoord > 0 ? idata[i - 1].x * stepZ : idata[i].x * stepZ;
		float xNext = xCoord < width - 1 ? idata[i + 1].x * stepZ : idata[i].x * stepZ;
		float yPrev = yCoord < height - 1 ? idata[i + width].x * stepZ : idata[i].x * stepZ;
		float yNext = yCoord > 0 ? idata[i - width].x * stepZ : idata[i].x * stepZ;
		float3 normal;
		normal.x = (xPrev - xNext) * heightScale;
		normal.y = (yPrev - yNext) * heightScale;
		normal.z = 1;
		normal = normalize( normal );
		normalMap[i * 4 + 0] = (uchar)round( (normal.x * 0.5 + 0.5) * 255 );
		normalMap[i * 4 + 1] = (uchar)round( (normal.y * 0.5 + 0.5) * 255 );
		normalMap[i * 4 + 2] = (uchar)round( (normal.z * 0.5 + 0.5) * 255 );
		normalMap[i * 4 + 3] = 255;
	}
	if (width * height > 0) memcpy( idata, normalMap, width * height * 4 );
	delete normalMap;
}

// EOF