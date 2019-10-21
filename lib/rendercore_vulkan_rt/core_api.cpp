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

static CoreAPI_Base *coreInstance = nullptr;

extern "C" COREDLL_API CoreAPI_Base *CreateCore()
{
	assert( coreInstance == nullptr );
	gladLoadGL(); // the dll needs its own OpenGL function pointers
	coreInstance = new CoreAPI();
	coreInstance->Init();
	return coreInstance;
}

extern "C" COREDLL_API void DestroyCore()
{
	assert( coreInstance );
	delete coreInstance;
	coreInstance = nullptr;
}

namespace lh2core
{
static lh2core::RenderCore *core = 0;
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

void CoreAPI::SetProbePos( const int2 pos )
{
	core->SetProbePos( pos );
}

void CoreAPI::SetTarget( GLTexture *target, const uint spp )
{
	core->SetTarget( target, spp );
}

void CoreAPI::Setting( const char *name, float value )
{
	core->Setting( name, value );
}

void CoreAPI::Render( const ViewPyramid &view, const Convergence converge, const float brightness, const float contrast )
{
	core->Render( view, converge, brightness, contrast );
}

void CoreAPI::Shutdown()
{
	core->Shutdown();
	delete core;
	core = 0;
}

void CoreAPI::SetTextures( const CoreTexDesc *tex, const int textureCount )
{
	core->SetTextures( tex, textureCount );
}

void CoreAPI::SetMaterials( CoreMaterial *mat, const CoreMaterialEx *matEx, const int materialCount )
{
	core->SetMaterials( mat, matEx, materialCount );
}

void CoreAPI::SetLights( const CoreLightTri *areaLights, const int areaLightCount,
	const CorePointLight *pointLights, const int pointLightCount,
	const CoreSpotLight *spotLights, const int spotLightCount,
	const CoreDirectionalLight *directionalLights, const int directionalLightCount )
{
	core->SetLights( areaLights, areaLightCount,
		pointLights, pointLightCount,
		spotLights, spotLightCount,
		directionalLights, directionalLightCount );
}

void CoreAPI::SetSkyData( const float3 *pixels, const uint width, const uint height )
{
	core->SetSkyData( pixels, width, height );
}

void CoreAPI::SetGeometry( const int meshIdx, const float4 *vertexData, const int vertexCount, const int triangleCount, const CoreTri *triangles, const uint *alphaFlags )
{
	core->SetGeometry( meshIdx, vertexData, vertexCount, triangleCount, triangles, alphaFlags );
}

void CoreAPI::SetInstance( const int instanceIdx, const int modelIdx, const mat4 &transform )
{
	core->SetInstance( instanceIdx, modelIdx, transform );
}

void CoreAPI::UpdateToplevel()
{
	core->UpdateToplevel();
}

// EOF