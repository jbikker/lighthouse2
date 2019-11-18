/* rendercore.cpp - Copyright 2019 Utrecht University

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

using namespace lh2core;

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  Initialization.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
	// initialize core
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture* target )
{
	// synchronize OpenGL viewport
	scrwidth = target->width;
	scrheight = target->height;
	targetTextureID = target->ID;
	// see if we need to reallocate our buffers
	bool reallocate = false;
	if (scrwidth * scrheight > maxPixels)
	{
		maxPixels = scrwidth * scrheight;
		maxPixels += maxPixels >> 4; // reserve a bit extra to prevent frequent reallocs
		delete screenPixels;
		screenPixels = new uint[maxPixels];
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangleData, const uint* alphaFlags )
{
	Mesh newMesh;
	// copy the supplied vertices; we cannot assume that the render system does not modify
	// the original data after we leave this function.
	newMesh.vertices = new float4[vertexCount];
	newMesh.vcount = vertexCount;
	memcpy( newMesh.vertices, vertexData, vertexCount * sizeof( float4 ) );
	// copy the supplied 'fat triangles'
	newMesh.triangles = new CoreTri[vertexCount / 3];
	memcpy( newMesh.triangles, triangleData, (vertexCount / 3) * sizeof( CoreTri ) );
	meshes.push_back( newMesh );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge, const float brightness, const float contrast )
{
	// render
	memset( screenPixels, 0, scrwidth * scrheight * sizeof( uint ) );
	for( Mesh& mesh : meshes )
	{
		for( int i = 0; i < mesh.vcount; i++ )
		{
			// convert a vertex position to a screen coordinate
			int screenx = mesh.vertices[i].x / 80 * (float)scrwidth + scrwidth / 2;
			int screeny = mesh.vertices[i].z / 80 * (float)scrheight + scrheight / 2;
			// plot the vertex if it is within the screen boundaries
			if (screenx >= 0 && screeny >= 0 && screenx <= scrwidth && screeny <= scrheight)
			{
				screenPixels[screenx + screeny * scrwidth] = 0xffffff /* white */;
			}
		}
	}
	// copy pixel buffer to OpenGL render target texture
	glBindTexture( GL_TEXTURE_2D, targetTextureID );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, scrwidth, scrheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, screenPixels );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	delete screenPixels;
}

// EOF