/* rasterizer.cpp - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Ancient code, originally written for a graphics course at the NHTV
   University of Applied Sciences / IGAD program.
*/

#include "core_settings.h"

uint ScaleColor( uint c, int scale )
{
	unsigned int rb = (((c & 0xff00ff) * scale) >> 8) & 0xff00ff;
	unsigned int g = (((c & 0xff00) * scale) >> 8) & 0xff00;
	return rb + g;
}

// -----------------------------------------------------------
// static data for the rasterizer
// -----------------------------------------------------------
Surface* Mesh::screen = 0;
float* Mesh::xleft, *Mesh::xright, *Mesh::uleft, *Mesh::uright;
float* Mesh::vleft, *Mesh::vright, *Mesh::zleft, *Mesh::zright;
Scene Rasterizer::scene;
float* Rasterizer::zbuffer;
float4 Rasterizer::frustum[5];
static float3 raxis[3] = { make_float3( 1, 0, 0 ), make_float3( 0, 1, 0 ), make_float3( 0, 0, 1 ) };

// -----------------------------------------------------------
// Mesh constructor
// input: vertex count & face count
// allocates room for mesh data:
// - pos:  vertex positions
// - tpos: transformed vertex positions
// - norm: vertex normals
// - spos: vertex screen space positions
// - uv:   vertex uv coordinates
// - N:    face normals
// - tri:  connectivity data
// -----------------------------------------------------------
Mesh::Mesh( int vcount, int tcount ) : verts( vcount ), tris( tcount )
{
	pos = new float3[vcount * 3], tpos = pos + vcount, norm = pos + 2 * vcount;
	spos = new float2[vcount * 2], uv = spos + vcount, N = new float3[tcount];
	tri = new int[tcount * 3];
	material = new int[tcount];
}

// -----------------------------------------------------------
// Mesh render function
// input: final matrix for scene graph node
// renders a mesh using software rasterization.
// stages:
// 1. mesh culling: checks the mesh against the view frustum
// 2. vertex transform: calculates world space coordinates
// 3. triangle rendering loop. substages:
//    a) backface culling
//    b) clipping (Sutherland-Hodgeman)
//    c) shading (using pre-scaled palettes for speed)
//    d) projection: world-space to 2D screen-space
//    e) span construction
//    f) span filling
// -----------------------------------------------------------
void Mesh::Render( const mat4& T )
{
	// cull mesh
	float3 c[8];
	for (int i = 0; i < 8; i++) c[i] = make_float3( T * make_float4( bounds[i & 1].x, bounds[(i >> 1) & 1].y, bounds[i >> 2].z, 1 ) );
	for (int i, p = 0; p < 5; p++)
	{
		for (i = 0; i < 8; i++) if ((dot( make_float3( Rasterizer::frustum[p] ), c[i] ) - Rasterizer::frustum[p].w) > 0) break;
		if (i == 8) return;
	}
	// transform vertices
	for (int i = 0; i < verts; i++) tpos[i] = make_float3( make_float4( pos[i], 1 ) * T );
	// draw triangles
	for (int i = 0; i < tris; i++)
	{
		Material* mat = Rasterizer::scene.matList[material[i]];
		static uint p;
		uint* src = mat->texture ? mat->texture->pixels : &p;
		if (!mat->texture) p = mat->diffuse;
		float* zbuffer = Rasterizer::zbuffer, f;
		const float tw = mat->texture ? (float)mat->texture->width : 1;
		const float th = mat->texture ? (float)mat->texture->height : 1;
		const int umask = (int)tw, vmask = (int)th;
		// cull triangle
		float3 Nt = make_float3( make_float4( N[i], 0 ) * T );
		if (dot( tpos[tri[i * 3 + 0]], Nt ) > 0) continue;
		// clip
		float3 cpos[2][8], *pos;
		float2 cuv[2][8], *tuv;
		int nin = 3, nout = 0, from = 0, to = 1, miny = screen->height - 1, maxy = 0, h;
		for (int v = 0; v < 3; v++) cpos[0][v] = tpos[tri[i * 3 + v]], cuv[0][v] = uv[tri[i * 3 + v]];
		for (int p = 0; p < 2; p++, from = 1 - from, to = 1 - to, nin = nout, nout = 0) for (int v = 0; v < nin; v++)
		{
			const float3 A = cpos[from][v], B = cpos[from][(v + 1) % nin];
			const float2 Auv = cuv[from][v], Buv = cuv[from][(v + 1) % nin];
			const float4 plane = Rasterizer::frustum[p];
			const float t1 = dot( make_float3( plane ), A ) - plane.w, t2 = dot( make_float3( plane ), B ) - plane.w;
			if ((t1 < 0) && (t2 >= 0))
				f = t1 / (t1 - t2),
				cuv[to][nout] = Auv + (Buv - Auv) * f, cpos[to][nout++] = A + f * (B - A),
				cuv[to][nout] = Buv, cpos[to][nout++] = B;
			else if ((t1 >= 0) && (t2 >= 0)) cuv[to][nout] = Buv, cpos[to][nout++] = B;
			else if ((t1 >= 0) && (t2 < 0))
				f = t1 / (t1 - t2),
				cuv[to][nout] = Auv + (Buv - Auv) * f, cpos[to][nout++] = A + f * (B - A);
		}
		if (nin == 0) continue;
		// project
		pos = cpos[from], tuv = cuv[from];
		for (int v = 0; v < nin; v++)
			pos[v].x = ((pos[v].x * screen->width) / -pos[v].z) + screen->width / 2,
			pos[v].y = ((pos[v].y * screen->width) / pos[v].z) + screen->height / 2;
		// draw
		uint shade = (uint)((N[i].z + 1) * 64.0f + 127.9f);
		for (int j = 0; j < nin; j++)
		{
			int vert0 = j, vert1 = (j + 1) % nin;
			if (pos[vert0].y > pos[vert1].y) h = vert0, vert0 = vert1, vert1 = h;
			const float y0 = pos[vert0].y, y1 = pos[vert1].y, rydiff = 1.0f / (y1 - y0);
			if ((y0 == y1) || (y0 >= screen->height) || (y1 < 1)) continue;
			const int iy0 = max( 1, (int)y0 + 1 ), iy1 = min( screen->height - 2, (int)y1 );
			float x0 = pos[vert0].x, dx = (pos[vert1].x - x0) * rydiff;
			float z0 = 1.0f / pos[vert0].z, z1 = 1.0f / pos[vert1].z, dz = (z1 - z0) * rydiff;
			float u0 = tuv[vert0].x * z0, du = (tuv[vert1].x * z1 - u0) * rydiff;
			float v0 = tuv[vert0].y * z0, dv = (tuv[vert1].y * z1 - v0) * rydiff;
			const float f = (float)iy0 - y0;
			x0 += dx * f, u0 += du * f, v0 += dv * f, z0 += dz * f;
			for (int y = iy0; y <= iy1; y++)
			{
				if (x0 < xleft[y]) xleft[y] = x0, uleft[y] = u0, vleft[y] = v0, zleft[y] = z0;
				if (x0 > xright[y]) xright[y] = x0, uright[y] = u0, vright[y] = v0, zright[y] = z0;
				x0 += dx, u0 += du, v0 += dv, z0 += dz;
			}
			miny = min( miny, iy0 ), maxy = max( maxy, iy1 );
		}
		for (int y = miny; y <= maxy; xleft[y] = screen->width - 1, xright[y++] = 0)
		{
			float x0 = xleft[y], x1 = xright[y], rxdiff = 1.0f / (x1 - x0);
			float u0 = uleft[y], du = (uright[y] - u0) * rxdiff;
			float v0 = vleft[y], dv = (vright[y] - v0) * rxdiff;
			float z0 = zleft[y], dz = (zright[y] - z0) * rxdiff;
			const int ix0 = (int)x0 + 1, ix1 = min( screen->width - 2, (int)x1 );
			const float f = (float)ix0 - x0;
			u0 += f * du, v0 += f * dv, z0 += f * dz;
			uint* dest = screen->pixels + y * screen->width;
			float* zbuf = zbuffer + y * screen->width;
			for (int x = ix0; x <= ix1; x++, u0 += du, v0 += dv, z0 += dz) // plot span
			{
				if (z0 >= zbuf[x]) continue;
				const float z = 1.0f / z0;
				const uint u = (uint)(u0 * z * tw) % umask, v = (uint)(v0 * z * th) % vmask;
				dest[x] = ScaleColor( src[u + v * umask], shade ), zbuf[x] = z0;
			}
		}
	}
}

// -----------------------------------------------------------
// Scene destructor
// -----------------------------------------------------------
Scene::~Scene()
{
	delete root;
	for (auto tex : texList) delete tex;
	for (auto mat : matList) delete mat;
}

// -----------------------------------------------------------
// SGNode::Render
// recursive rendering of a scene graph node and its child nodes
// input: (inverse) camera transform
// -----------------------------------------------------------
void SGNode::Render( const mat4& transform )
{
	mat4 M = transform * localTransform;
	if (GetType() == SG_MESH) ((Mesh*)this)->Render( M );
	for (uint s = (uint)child.size(), i = 0; i < s; i++) child[i]->Render( M );
}

// -----------------------------------------------------------
// Rasterizer::Init
// initialization of the rasterizer
// input: surface to draw to
// -----------------------------------------------------------
void Rasterizer::Init()
{
	// setup outline tables & zbuffer
	Mesh::xleft = new float[8192], Mesh::xright = new float[8192]; // that should do for pretty much any resolution
	Mesh::uleft = new float[8192], Mesh::uright = new float[8192];
	Mesh::vleft = new float[8192], Mesh::vright = new float[8192];
	Mesh::zleft = new float[8192], Mesh::zright = new float[8192];
}

void Rasterizer::Reinit( int w, int h, Surface* screen )
{
	// initialization that depends on screen size
	for (int y = 0; y < h; y++) Mesh::xleft[y] = w - 1, Mesh::xright[y] = 0;
	delete zbuffer;
	zbuffer = new float[w * h];
	// calculate view frustum planes
	float C = -1.0f, x1 = 0.5f, x2 = w - 1.5f, y1 = 0.5f, y2 = h - 1.5f;
	float3 p0 = { 0, 0, 0 };
	float3 p1 = { ((x1 - w * 0.5f) * C) / w, ((y1 - h * 0.5f) * C) / w, 1.0f };
	float3 p2 = { ((x2 - w * 0.5f) * C) / w, ((y1 - h * 0.5f) * C) / w, 1.0f };
	float3 p3 = { ((x2 - w * 0.5f) * C) / w, ((y2 - h * 0.5f) * C) / w, 1.0f };
	float3 p4 = { ((x1 - w * 0.5f) * C) / w, ((y2 - h * 0.5f) * C) / w, 1.0f };
	frustum[0] = { 0, 0, -1, 0.2f };
	float3 a( normalize( cross( p1 - p0, p4 - p1 ) ) ); frustum[1] = make_float4( a, 0 ); // left plane
	float3 b( normalize( cross( p2 - p0, p1 - p2 ) ) ); frustum[2] = make_float4( b, 0 ); // top plane
	float3 c( normalize( cross( p3 - p0, p2 - p3 ) ) ); frustum[3] = make_float4( c, 0 ); // right plane
	float3 d( normalize( cross( p4 - p0, p3 - p4 ) ) ); frustum[4] = make_float4( d, 0 ); // bottom plane
	// store screen pointer
	Mesh::screen = screen;
}

// -----------------------------------------------------------
// Rasterizer::Render
// render the scene
// input: camera to render with
// -----------------------------------------------------------
void Rasterizer::Render( const mat4& transform )
{
	memset( Mesh::screen->pixels, 0, Mesh::screen->width * Mesh::screen->height * sizeof( uint ) );
	memset( zbuffer, 0, Mesh::screen->width * Mesh::screen->height * sizeof( float ) );
	scene.root->Render( transform.Inverted() );
}

// EOF