/* interoptexture.cpp - Copyright 2019 Utrecht University

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

#include <core_settings.h>

//  +-----------------------------------------------------------------------------+
//  |  InteropTexture implementation.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
InteropTexture::~InteropTexture()
{
	if (bound) CHK_CUDA( cudaGraphicsUnmapResources( 1, &res, 0 ) );
	if (linked) CHK_CUDA( cudaGraphicsUnregisterResource( res ) );
}

void InteropTexture::SetTexture( GLTexture* t )
{
	if (bound)
	{
		CHK_CUDA( cudaGraphicsUnmapResources( 1, &res, 0 ) );
		bound = false;
	}
	if (linked)
	{
		CHK_CUDA( cudaGraphicsUnregisterResource( res ) );
		linked = false;
	}
	texture = t;
}

void InteropTexture::LinkToSurface( const surfaceReference* s )
{
	surfRef = s;
	assert( !bound );
	if (linked)
	{
		// we were already linked; unlink first
		CHK_CUDA( cudaGraphicsUnregisterResource( res ) );
	}
	CHK_CUDA( cudaGraphicsGLRegisterImage( &res, texture->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ) );
	linked = true;
}

void InteropTexture::BindSurface()
{
	assert( !bound );
	cudaArray* ar;
	CHK_CUDA( cudaGraphicsMapResources( 1, &res, 0 /* DEFAULTSTREAM */ ) );
	CHK_CUDA( cudaGraphicsSubResourceGetMappedArray( &ar, res, 0, 0 ) );
	cudaChannelFormatDesc desc;
	CHK_CUDA( cudaGetChannelDesc( &desc, ar ) );
	CHK_CUDA( cudaBindSurfaceToArray( surfRef, ar, &desc ) );
	bound = true;
}

void InteropTexture::UnbindSurface()
{
	assert( bound );
	CHK_CUDA( cudaGraphicsUnmapResources( 1, &res, 0 ) );
	bound = false;
}

// EOF