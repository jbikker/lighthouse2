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
//  |  RenderCore::SetProbePos                                                    |
//  |  Set the pixel for which the triid will be captured.                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetProbePos( int2 pos )
{
	probePos = pos; // triangle id for this pixel will be stored in coreStats
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Init                                                           |
//  |  Initialization.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Init()
{
	// initialize scene
	rasterizer.Init();
	rasterizer.scene.root = new SGNode();
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTarget                                                      |
//  |  Set the OpenGL texture that serves as the render target.             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTarget( GLTexture* target, const uint spp )
{
	// synchronize OpenGL viewport
	scrwidth = target->width;
	scrheight = target->height;
	// see if we need to reallocate our buffers
	bool reallocate = false;
	if (scrwidth * scrheight > maxPixels)
	{
		maxPixels = scrwidth * scrheight;
		maxPixels += maxPixels >> 4; // reserve a bit extra to prevent frequent reallocs
		delete renderTarget;
		renderTarget = new Surface();
		renderTarget->pixels = (uint*)MALLOC64( maxPixels * sizeof( uint ) );
	}
	renderTarget->width = scrwidth;
	renderTarget->height = scrheight;
	targetTextureID = target->ID;
	// inform rasterizer
	rasterizer.Reinit( scrwidth, scrheight, renderTarget );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetGeometry                                                    |
//  |  Set the geometry data for a model.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags )
{
	// Note: for first-time setup, meshes are expected to be passed in sequential order.
	// This will result in new Mesh pointers being pushed into the meshes vector.
	// Subsequent mesh changes will be applied to existing Meshes. This is deliberately
	// minimalistic; RenderSystem is responsible for a proper (fault-tolerant) interface.
	assert( vertexCount == 3 * triangleCount );
	Mesh* mesh;
	if (meshIdx >= meshes.size()) meshes.push_back( mesh = new Mesh( vertexCount, triangleCount ) );
	else mesh = meshes[meshIdx]; // overwrite geometry data; assume vertex/face count does not change
	float3 bmin = make_float3( 1e34f ), bmax = -bmin;
	for (int i = 0; i < vertexCount; i++)
		mesh->pos[i] = make_float3( vertexData[i] ),
		bmin.x = min( bmin.x, vertexData[i].x ), bmin.y = min( bmin.y, vertexData[i].y ), bmin.z = min( bmin.z, vertexData[i].z ),
		bmax.x = max( bmax.x, vertexData[i].x ), bmax.y = max( bmax.y, vertexData[i].y ), bmax.z = max( bmax.z, vertexData[i].z );
	mesh->bounds[0] = bmin, mesh->bounds[1] = bmax;
	for (int i = 0; i < triangleCount * 3; i++) mesh->tri[i] = i;
	for (int i = 0; i < triangleCount; i++)
		mesh->norm[i * 3 + 0] = triangles[i].vN0, mesh->norm[i * 3 + 1] = triangles[i].vN1, mesh->norm[i * 3 + 2] = triangles[i].vN2,
		mesh->uv[i * 3 + 0] = make_float2( triangles[i].u0, triangles[i].v0 ),
		mesh->uv[i * 3 + 1] = make_float2( triangles[i].u1, triangles[i].v1 ),
		mesh->uv[i * 3 + 2] = make_float2( triangles[i].u2, triangles[i].v2 ),
		mesh->N[i] = make_float3( triangles[i].Nx, triangles[i].Ny, triangles[i].Nz ),
		mesh->material[i] = triangles[i].material;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetInstance                                                    |
//  |  Set instance details.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetInstance( const int instanceIdx, const int meshIdx, const mat4& matrix )
{
	// A '-1' mesh denotes the end of the instance stream;
	// adjust the instances vector if we have more.
	if (meshIdx == -1)
	{
		if (rasterizer.scene.root->child.size() > instanceIdx) 
			rasterizer.scene.root->child.resize( instanceIdx );
		return;
	}
	// For the first frame, instances are added to the instances vector.
	// For subsequent frames existing slots are overwritten / updated.
	if (instanceIdx >= rasterizer.scene.root->child.size())
	{
		// Note: for first-time setup, meshes are expected to be passed in sequential order.
		// This will result in new CoreInstance pointers being pushed into the instances vector.
		// Subsequent instance changes (typically: transforms) will be applied to existing CoreInstances.
		assert( instanceIdx == rasterizer.scene.root->child.size() );
		rasterizer.scene.root->child.push_back( meshes[meshIdx] );
	}
	else rasterizer.scene.root->child[instanceIdx] = meshes[meshIdx];
	rasterizer.scene.root->child[instanceIdx]->localTransform = matrix;
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetTextures                                                    |
//  |  Set the texture data.                                                LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetTextures( const CoreTexDesc* tex, const int textures )
{
	// copy the supplied array of texture descriptors
	for (int i = 0; i < textures; i++)
	{
		Texture* t;
		if (i < rasterizer.scene.texList.size()) t = rasterizer.scene.texList[i];
		else rasterizer.scene.texList.push_back( t = new Texture() );
		t->pixels = (uint*)MALLOC64( tex[i].pixelCount * sizeof( uint ) );
		if (tex[i].idata) memcpy( t->pixels, tex[i].idata, tex[i].pixelCount * sizeof( uint ) );
		else memcpy( t->pixels, 0, tex[i].pixelCount * sizeof( uint ) /* assume integer textures */ );
		// Note: texture width and height are not known yet, will be set when we get the materials.
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetMaterials                                                   |
//  |  Set the material data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount )
{
	// copy the supplied array of materials
	for (int i = 0; i < materialCount; i++)
	{
		Material* m;
		if (i < rasterizer.scene.matList.size()) m = rasterizer.scene.matList[i];
		else rasterizer.scene.matList.push_back( m = new Material() );
		m->texture = 0;
		int texID = matEx[i].texture[TEXTURE0];
		if (texID == -1)
		{
			float r = mat[i].diffuse_r, g = mat[i].diffuse_g, b = mat[i].diffuse_b;
			m->diffuse = ((int)(b * 255.0f) << 16) + ((int)(g * 255.0f) << 8) + (int)(r * 255.0f);
		}
		else
		{
			m->texture = rasterizer.scene.texList[texID];
			m->texture->width = mat[i].texwidth0; // we know this only now, so set it properly
			m->texture->height = mat[i].texheight0;
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetLights                                                      |
//  |  Set the light data.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetLights( const CoreLightTri* areaLights, const int areaLightCount,
	const CorePointLight* pointLights, const int pointLightCount,
	const CoreSpotLight* spotLights, const int spotLightCount,
	const CoreDirectionalLight* directionalLights, const int directionalLightCount )
{
	// not supported yet
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::SetSkyData                                                     |
//  |  Set the sky dome data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::SetSkyData( const float3* pixels, const uint width, const uint height )
{
	// not supported yet
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Setting                                                        |
//  |  Modify a render setting.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Setting( const char* name, const float value )
{
	// we have no settings yet
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Render                                                         |
//  |  Produce one image.                                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Render( const ViewPyramid& view, const Convergence converge )
{
	// render
	mat4 transform;
	const float3 X = normalize( view.p2 - view.p1 ), Y = normalize( view.p1 - view.p3 );
	const float3 Z = normalize( view.pos - 0.5f * (view.p2 + view.p3) );
	transform[0] = X.x, transform[4] = X.y, transform[8] = X.z;
	transform[1] = Y.x, transform[5] = Y.y, transform[9] = Y.z;
	transform[2] = Z.x, transform[6] = Z.y, transform[10] = Z.z;
	rasterizer.Render( mat4::Translate( view.pos ) * transform );
	// copy cpu surface to OpenGL render target texture
	glBindTexture( GL_TEXTURE_2D, targetTextureID );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, scrwidth, scrheight, 0, GL_RGBA, GL_UNSIGNED_BYTE, renderTarget->pixels );
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	delete renderTarget;
}

// EOF