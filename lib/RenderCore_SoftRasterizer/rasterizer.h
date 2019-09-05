/* rasterizer.h - Copyright 2019 Utrecht University

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

#pragma once

namespace lh2core {

// -----------------------------------------------------------
// Surface class
// bare minimum
// -----------------------------------------------------------
class Surface
{
public:
	Surface() = default;
	~Surface() { FREE64( pixels ); /* assuming we used MALLOC64 to create the buffer */ }
	int width = 0, height = 0;
	uint* pixels = 0;
};

// -----------------------------------------------------------
// Texture class
// encapsulates a palettized pixel surface with pre-scaled
// palettes for fast shading
// -----------------------------------------------------------
class Texture
{
public:
	// constructor / destructor
	Texture() = default;
	Texture( int w, int h ) : width( w ), height( h ) { pixels = (uint*)MALLOC64( w * h * sizeof( uint ) ); }
	~Texture() { FREE64( pixels ); }
	// data members
	int width = 0, height = 0;
	uint* pixels = 0;
};

// -----------------------------------------------------------
// Material class
// basic material properties
// -----------------------------------------------------------
class Material
{
public:
	// constructor / destructor
	Material() = default;
	// data members
	uint diffuse = 0xffffffff;		// diffuse material color
	Texture* texture = 0;			// texture
};

// -----------------------------------------------------------
// SGNode class
// scene graph node, with convenience functions for translate
// and transform; base class for Mesh
// -----------------------------------------------------------
class SGNode
{
public:
	enum { SG_TRANSFORM = 0, SG_MESH };
	// constructor / destructor
	~SGNode()
	{
		for (int s = (int)child.size(), i = 0; i < s; i++)
		{
			for (int j = i + 1; j < s; j++) if (child[j] == child[i]) child[j] = 0;
			delete child[i];
		}
	}
	// methods
	void SetPosition( float3& pos ) { mat4& M = localTransform; M[3] = pos.x, M[7] = pos.y, M[11] = pos.z; }
	float3 GetPosition() { mat4& M = localTransform; return make_float3( M[3], M[7], M[11] ); }
	void Render( mat4& transform );
	virtual int GetType() { return SG_TRANSFORM; }
	// data members
	mat4 localTransform;
	vector<SGNode*> child;
};

// -----------------------------------------------------------
// Mesh class
// represents a mesh
// -----------------------------------------------------------
class Mesh : public SGNode
{
public:
	// constructor / destructor
	Mesh() : verts( 0 ), tris( 0 ), pos( 0 ), uv( 0 ), spos( 0 ) {}
	Mesh( int vcount, int tcount );
	~Mesh() { delete pos; delete N; delete spos; delete tri; }
	// methods
	void Render( mat4& transform );
	virtual int GetType() { return SG_MESH; }
	// data members
	float3* pos = 0;				// object-space vertex positions
	float3* tpos = 0;				// world-space positions
	float2* uv = 0;					// vertex uv coordinates
	float2* spos = 0;				// screen positions
	float3* norm = 0;				// vertex normals
	float3* N = 0;					// triangle plane
	int* tri = 0;					// connectivity data
	int verts = 0, tris = 0;		// vertex & triangle count
	int* material = 0;				// per-face material ID
	float3 bounds[2];				// mesh bounds
	static Surface* screen;
	static float* xleft, *xright;	// outline tables for rasterization
	static float* uleft, *uright;
	static float* vleft, *vright;
	static float* zleft, *zright;
};

// -----------------------------------------------------------
// Scene class
// owner of the scene graph;
// owner of the material and texture list
// -----------------------------------------------------------
class Scene
{
public:
	// constructor / destructor
	Scene() = default;
	~Scene();
	// data members
public:
	SGNode* root = 0;
	vector<Material*> matList;
	vector<Texture*> texList;
};

// -----------------------------------------------------------
// Rasterizer class
// rasterizer
// implements a basic, but fast & accurate software rasterizer
// -----------------------------------------------------------
class Rasterizer
{
public:
	// constructor / destructor
	Rasterizer() = default;
	// methods
	void Init();
	void Reinit( int w, int h, Surface* screen );
	void Render( mat4& transform );
	// data members
	static Scene scene;
	static float* zbuffer;
	static float4 frustum[5];
};

} // namespace lh2core

// EOF