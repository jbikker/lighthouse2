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
#include <iostream>

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
	targetTextureID = target->ID;
	if (screen != 0 && target->width == screen->width && target->height == screen->height) return; // nothing changed
	delete screen;
	screen = new Bitmap( target->width, target->height );
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
//void RenderCore::Render( const ViewPyramid& view, const Convergence converge, const float brightness, const float contrast )
//{
//	// render
//	screen->Clear();
//	for( Mesh& mesh : meshes ) for( int i = 0; i < mesh.vcount; i++ )
//	{
//		// convert a vertex position to a screen coordinate
//		int screenx = mesh.vertices[i].x / 80 * (float)screen->width + screen->width / 2;
//		int screeny = mesh.vertices[i].z / 80 * (float)screen->height + screen->height / 2;
//		screen->Plot( screenx, screeny, 0xffffff /* white */ );
//	}
//	// copy pixel buffer to OpenGL render target texture
//	glBindTexture( GL_TEXTURE_2D, targetTextureID );
//	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, screen->width, screen->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, screen->pixels );
//}

void RenderCore::Render(const ViewPyramid& view, const Convergence converge, const float brightness, const float contrast)
{
	// render
	screen->Clear();

	float3 xDirection = (view.p2 - view.p1) / screen->width;
	float3 yDirection = (view.p3 - view.p1) / screen->height;

	float3 p1 = view.p1 - view.pos;

	Ray ray;

	for (int u = 0; u < screen->width; u++) {
		cout << u << "/" << screen->width << endl;
		//cout << int(u/screen->width * 100) << "%" << endl;
		for (int v = 0; v < screen->height; v++) {
			ray.direction = normalize(p1 + u * xDirection + v * yDirection);
			ray.origin = view.pos;

			float3 color = Trace(ray);
			int colorHex = (int(0xff * color.x) << 16) + (int(0xff * color.y) << 8) + int(0xff * color.z);
			screen->Plot(u, v, colorHex);
		}
	}
	
	// copy pixel buffer to OpenGL render target texture
	glBindTexture(GL_TEXTURE_2D, targetTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screen->width, screen->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, screen->pixels);
}

//  +-----------------------------------------------------------------------------+
//  |  RenderCore::Shutdown                                                       |
//  |  Free all resources.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void RenderCore::Shutdown()
{
	delete screen;
}

float3 RenderCore::Trace(Ray r) {
	Intersection intersection;
	bool hasIntersection = NearestIntersection(r, intersection);

	if (hasIntersection) {
		return float3{ 0, 0, 0 };
	}

	return float3{ 0.4, 0.4, 0 };
}

bool RenderCore::NearestIntersection(Ray r, Intersection &intersection) {
	//float nearestT;
	//Mesh nearestMesh;
	float t;
	for (Mesh& mesh : meshes) for (int i = 0; i < mesh.vcount; i += 3) {
		float4 a = mesh.vertices[i];
		float4 b = mesh.vertices[i + 1];
		float4 c = mesh.vertices[i + 2];
		bool hasIntersection = GeometricTriangleIntersection(r, make_float3(a), make_float3(b), make_float3(c), t);

		if (hasIntersection) {
			return true;
		}
	}

	return false;
}

// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
bool RenderCore::GeometricTriangleIntersection(Ray r, float3 v0, float3 v1, float3 v2, float &t) {
	// compute plane's normal
	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;
	// no need to normalize
	float3 N = cross(v0v1, v0v2); // N 
	float area2 = length(N);

	float3 dir = r.direction;
	float3 orig = r.origin;

	// Step 1: finding P

	// check if ray and plane are parallel ?
	float NdotRayDirection = dot(N, dir);
	//if (fabs(NdotRayDirection) < kEpsilon) // almost 0 
	//	return false; // they are parallel so they don't intersect ! 

	// compute d parameter using equation 2
	float d = dot(N, v0);

	// compute t (equation 3)
	t = (dot(N,orig) + d) / NdotRayDirection;
	// check if the triangle is in behind the ray
	if (t < 0) return false; // the triangle is behind 

	// compute the intersection point using equation 1
	float3 P = orig + t * dir;

	// Step 2: inside-outside test
	float3 C; // vector perpendicular to triangle's plane 

	// edge 0
	float3 edge0 = v1 - v0;
	float3 vp0 = P - v0;
	C = cross(edge0,vp0);
	if (dot(N, C) < 0) return false; // P is on the right side 

	// edge 1
	float3 edge1 = v2 - v1;
	float3 vp1 = P - v1;
	C = cross(edge1,vp1);
	if (dot(N,C) < 0)  return false; // P is on the right side 

	// edge 2
	float3 edge2 = v0 - v2;
	float3 vp2 = P - v2;
	C = cross(edge2,vp2);
	if (dot(N, C) < 0) return false; // P is on the right side; 

	return true; // this ray hits the triangle
}

void printFloat3(float3 value) {
	cout << "{ x:" << value.x << " y: " << value.y << " z: " << value.z << "}";
}

// EOF