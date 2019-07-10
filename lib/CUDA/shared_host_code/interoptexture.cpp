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

#include "core_settings.h"

//  +-----------------------------------------------------------------------------+
//  |  InteropTexture implementation.                                             |
//  |  TODO: move to its own file.                                          LH2'19|
//  +-----------------------------------------------------------------------------+
InteropTexture::~InteropTexture()
{
	if (bound) CUDACHECK( "cudaGraphicsUnmapResources", cudaGraphicsUnmapResources( 1, &res, 0 ) );
	if (linked) CUDACHECK( "cudaGraphicsUnregisterResource", cudaGraphicsUnregisterResource( res ) );
}

void InteropTexture::SetTexture( GLTexture* t )
{
	if (bound) 
	{
		CUDACHECK( "cudaGraphicsUnmapResources", cudaGraphicsUnmapResources( 1, &res, 0 ) );
		bound = false;
	}
	if (linked) 
	{
		CUDACHECK( "cudaGraphicsUnregisterResource", cudaGraphicsUnregisterResource( res ) );
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
		CUDACHECK( "cudaGraphicsUnregisterResource", cudaGraphicsUnregisterResource( res ) );
	}
	CUDACHECK( "cudaGraphicsGLRegisterImage", cudaGraphicsGLRegisterImage( &res, texture->ID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore ) );
	linked = true;
}

void InteropTexture::BindSurface()
{
	assert( !bound );
	cudaArray* ar;
	CUDACHECK( "cudaGraphicsMapResources", cudaGraphicsMapResources( 1, &res, 0 /* DEFAULTSTREAM */ ) );
	CUDACHECK( "cudaGraphicsSubResourceGetMappedArray", cudaGraphicsSubResourceGetMappedArray( &ar, res, 0, 0 ) );
	cudaChannelFormatDesc desc;
	CUDACHECK( "cudaGetChannelDesc", cudaGetChannelDesc( &desc, ar ) );
	CUDACHECK( "cudaBindSurfaceToArray", cudaBindSurfaceToArray( surfRef, ar, &desc ) );
	bound = true;
}

void InteropTexture::UnbindSurface()
{
	assert( bound );
	CUDACHECK( "cudaGraphicsUnmapResources", cudaGraphicsUnmapResources( 1, &res, 0 ) );
	bound = false;
}

// EOF