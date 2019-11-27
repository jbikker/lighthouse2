/* core_api.cpp - Copyright 2019 Utrecht University

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

static CoreAPI_Base* coreInstance = NULL;

extern "C" COREDLL_API CoreAPI_Base* CreateCore()
{
	assert( coreInstance == NULL );
	gladLoadGL(); // the dll needs its own OpenGL function pointers
	coreInstance = new CoreAPI();
	coreInstance->Init();
	return coreInstance;
}

extern "C" COREDLL_API void DestroyCore()
{
	assert( coreInstance );
	delete coreInstance;
	coreInstance = NULL;
}

namespace lh2core {
static lh2core::RenderCore* core = 0;
};

void CoreAPI::Init()
{
	if (!core)
	{
		core = new RenderCore();
		core->Init();
	}
}

CoreStats CoreAPI::GetCoreStats()
{
	return core->coreStats;
}

void CoreAPI::SetTarget( GLTexture* target, const uint spp )
{
	// we received an OpenGL texture as a render target; forward to the SetTarget method in rendercore.cpp.
	core->SetTarget( target /* ignore spp parameter */ );
}

void CoreAPI::Render( const ViewPyramid& view, const Convergence converge )
{
	// forward the render request to the Render method in rendercore.cpp
	core->Render( view, converge );
}

void CoreAPI::Shutdown()
{
	core->Shutdown();
	delete core;
	core = 0;
}

void CoreAPI::SetTextures( const CoreTexDesc* tex, const int textureCount )
{
	// core->SetTextures( tex, textureCount ); TODO
}

void CoreAPI::SetMaterials( CoreMaterial* mat, const CoreMaterialEx* matEx, const int materialCount )
{
	// core->SetMaterials( mat, matEx, materialCount ); TODO
}

void CoreAPI::SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags )
{
	core->SetGeometry( meshIdx, vertexData, vertexCount, triangleCount, triangles, alphaFlags );
}

void CoreAPI::SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform )
{
	// core->SetInstance( instanceIdx, modelIdx, transform ); // TODO; we will just render the meshes for now.
}

// EOF