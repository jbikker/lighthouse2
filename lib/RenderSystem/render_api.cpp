/* render_api.cpp - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file implements the RebderSystem API, which is the interface
   between the application and the RenderSystem.
*/

#include "rendersystem.h"

static RenderSystem* renderer = nullptr;
static RenderAPI api;

RenderAPI* RenderAPI::CreateRenderAPI( const char* dllName )
{
	if (!renderer) 
	{
		renderer = new RenderSystem();
		renderer->Init( dllName );
	}
	return &api;
}

void RenderAPI::SerializeMaterials( const char* xmlFile )
{
	renderer->scene->SerializeMaterials( xmlFile );
}

void RenderAPI::DeserializeMaterials( const char* xmlFile )
{
	renderer->scene->DeserializeMaterials( xmlFile );
}

void RenderAPI::Shutdown()
{
	renderer->Shutdown();
}

void RenderAPI::DeserializeCamera( const char* xmlFile )
{
	renderer->scene->camera->Deserialize( xmlFile );
}

void RenderAPI::SerializeCamera( const char* xmlFile )
{
	renderer->scene->camera->Serialize( xmlFile );
}

int RenderAPI::AddMesh( const char* file, const char* dir, const float scale )
{
	return renderer->scene->AddMesh( file, dir, scale );
}

int RenderAPI::AddQuad( const float3 N, const float3 pos, const float width, const float height, const int material )
{
	return renderer->scene->AddQuad( N, pos, width, height, material );
}

int RenderAPI::AddInstance( const int meshId, const mat4& transform )
{
	return renderer->scene->AddInstance( meshId, transform );
}

void RenderAPI::SetInstanceTransform( const int instId, const mat4& transform )
{
	renderer->scene->SetInstanceTransform( instId, transform );
}

void RenderAPI::SynchronizeSceneData()
{
	renderer->SynchronizeSceneData();
}

void RenderAPI::Render( Convergence converge )
{
	renderer->Render( renderer->scene->camera->GetView(), converge );
}

Camera* RenderAPI::GetCamera()
{
	return renderer->scene->camera;
}

RenderSettings* RenderAPI::GetSettings()
{
	return &renderer->settings;
}

int RenderAPI::GetTriangleMaterialID( const int triId, const int instId )
{
	return renderer->scene->GetTriangleMaterial( triId, instId );
}

HostMaterial* RenderAPI::GetTriangleMaterial( const int triId, const int instId )
{
	int matId = renderer->scene->GetTriangleMaterial( triId, instId );
	return GetMaterial( matId );
}

HostMaterial* RenderAPI::GetMaterial( const int matId )
{
	return renderer->scene->materials[matId];
}

int RenderAPI::FindMaterialID( const char* name )
{
	return renderer->scene->FindMaterialID( name );
}

int RenderAPI::AddMaterial( const float3 color )
{
	return renderer->scene->AddMaterial( color );
}

int RenderAPI::AddPointLight( const float3 pos, const float3 radiance, bool enabled )
{
	return renderer->scene->AddPointLight( pos, radiance, enabled );
}

int RenderAPI::AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled )
{
	return renderer->scene->AddSpotLight( pos, direction, inner, outer, radiance, enabled );
}

int RenderAPI::AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled )
{
	return renderer->scene->AddDirectionalLight( direction, radiance, enabled );
}

void RenderAPI::SetTarget( GLTexture* tex, const uint spp )
{
	renderer->SetTarget( tex, spp );
}

void RenderAPI::SetProbePos( const int2 pos )
{
	renderer->SetProbePos( pos );
}

CoreStats RenderAPI::GetCoreStats()
{
	return renderer->GetCoreStats();
}

// EOF