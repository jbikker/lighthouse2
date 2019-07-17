/* host_scene.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

// static scene data
HostSkyDome* HostScene::sky = 0;
vector<HostMesh*> HostScene::meshes;
vector<HostInstance*> HostScene::instances;
vector<HostMaterial*> HostScene::materials;
vector<HostTexture*> HostScene::textures;
vector<HostAreaLight*> HostScene::areaLights;
vector<HostPointLight*> HostScene::pointLights;
vector<HostSpotLight*> HostScene::spotLights;
vector<HostDirectionalLight*> HostScene::directionalLights;
Camera* HostScene::camera = 0;

// helper functions
static HostTri TransformedHostTri( HostTri* tri, mat4 T )
{
	HostTri transformedTri = *tri;
	transformedTri.vertex0 = make_float3( make_float4( transformedTri.vertex0, 1 ) * T );
	transformedTri.vertex1 = make_float3( make_float4( transformedTri.vertex1, 1 ) * T );
	transformedTri.vertex2 = make_float3( make_float4( transformedTri.vertex2, 1 ) * T );
	float4 N = make_float4( transformedTri.Nx, transformedTri.Ny, transformedTri.Nz, 0 ) * T;
	transformedTri.Nx = N.x;
	transformedTri.Ny = N.y;
	transformedTri.Nz = N.z;
	return transformedTri;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::HostScene                                                       |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostScene::HostScene()
{
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::~HostScene                                                      |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
HostScene::~HostScene()
{
	// clean up allocated objects
	for (auto instance : instances) delete instance;
	for (auto mesh : meshes) delete mesh;
	for (auto material : materials) delete material;
	for (auto texture : textures) delete texture;
	delete sky;
	delete camera;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::SerializeMaterials                                              |
//  |  Write the list of materials to a file.                               LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::SerializeMaterials( const char* xmlFile )
{
	XMLDocument doc;
	XMLNode* root = doc.NewElement( "materials" );
	doc.InsertFirstChild( root );
	int materialCount = (int)materials.size();
	((XMLElement*)root->InsertEndChild( doc.NewElement( "material_count" ) ))->SetText( materialCount );
	for (uint i = 0; i < materials.size(); i++)
	{
		// skip materials that were created at runtime
		if ((materials[i]->flags & HostMaterial::FROM_MTL) == 0) continue;
		// create a new entry for the material
		char entryName[128];
		sprintf_s( entryName, "material_%i", i );
		XMLNode* materialEntry = doc.NewElement( entryName );
		root->InsertEndChild( materialEntry );
		// store material properties
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "name" ) ))->SetText( materials[i]->name.c_str() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "origin" ) ))->SetText( materials[i]->origin.c_str() );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "id" ) ))->SetText( materials[i]->ID );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "flags" ) ))->SetText( materials[i]->flags );
		XMLElement* diffuse = doc.NewElement( "color" );
		diffuse->SetAttribute( "b", materials[i]->color.z );
		diffuse->SetAttribute( "g", materials[i]->color.y );
		diffuse->SetAttribute( "r", materials[i]->color.x );
		XMLElement* absorption = doc.NewElement( "absorption" );
		absorption->SetAttribute( "b", materials[i]->absorption.z );
		absorption->SetAttribute( "g", materials[i]->absorption.y );
		absorption->SetAttribute( "r", materials[i]->absorption.x );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "metallic" ) ))->SetText( materials[i]->metallic );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "subsurface" ) ))->SetText( materials[i]->subsurface );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "specular" ) ))->SetText( materials[i]->specular );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "roughness" ) ))->SetText( materials[i]->roughness );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "specularTint" ) ))->SetText( materials[i]->specularTint );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "anisotropic" ) ))->SetText( materials[i]->anisotropic );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "sheen" ) ))->SetText( materials[i]->sheen );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "sheenTint" ) ))->SetText( materials[i]->sheenTint );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "clearcoat" ) ))->SetText( materials[i]->clearcoat );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "clearcoatGloss" ) ))->SetText( materials[i]->clearcoatGloss );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "transmission" ) ))->SetText( materials[i]->transmission );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "eta" ) ))->SetText( materials[i]->eta );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "custom0" ) ))->SetText( materials[i]->custom0 );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "custom1" ) ))->SetText( materials[i]->custom1 );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "custom2" ) ))->SetText( materials[i]->custom2 );
		((XMLElement*)materialEntry->InsertEndChild( doc.NewElement( "custom3" ) ))->SetText( materials[i]->custom3 );
	}
	doc.SaveFile( xmlFile );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::DeserializeMaterials                                            |
//  |  Restore the materials from a file.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::DeserializeMaterials( const char* xmlFile )
{
	XMLDocument doc;
	XMLError result = doc.LoadFile( xmlFile );
	if (result != XML_SUCCESS) return;
	XMLNode* root = doc.FirstChild();
	if (root == nullptr) return;
	XMLElement* countElement = root->FirstChildElement( "material_count" );
	if (!countElement) return;
	int materialCount;
	sscanf_s( countElement->GetText(), "%i", &materialCount );
	if (materialCount != materials.size()) return;
	for (int i = 0; i < materialCount; i++)
	{
		// find the entry for the material
		HostMaterial* m /* for brevity */ = materials[i];
		char entryName[128];
		sprintf_s( entryName, "material_%i", i );
		XMLNode* entry = root->FirstChildElement( entryName );
		if (!entry) continue;
		// set the properties
		const char* materialName = entry->FirstChildElement( "name" )->GetText();
		const char* materialOrigin = entry->FirstChildElement( "origin" )->GetText();
		m->name = string( materialName ? materialName : "" );
		m->origin = string( materialOrigin ? materialOrigin : "" );
		if (entry->FirstChildElement( "id" )) entry->FirstChildElement( "id" )->QueryIntText( &m->ID );
		if (entry->FirstChildElement( "flags" )) entry->FirstChildElement( "flags" )->QueryUnsignedText( &m->flags );
		XMLElement* color = entry->FirstChildElement( "color" );
		if (color)
			color->QueryFloatAttribute( "r", &m->color.x ),
			color->QueryFloatAttribute( "g", &m->color.y ),
			color->QueryFloatAttribute( "b", &m->color.z );
		XMLElement* absorption = entry->FirstChildElement( "absorption" );
		if (absorption)
			absorption->QueryFloatAttribute( "r", &m->absorption.x ),
			absorption->QueryFloatAttribute( "g", &m->absorption.y ),
			absorption->QueryFloatAttribute( "b", &m->absorption.z );
		if (entry->FirstChildElement( "metallic" )) entry->FirstChildElement( "metallic" )->QueryFloatText( &m->metallic );
		if (entry->FirstChildElement( "subsurface" )) entry->FirstChildElement( "subsurface" )->QueryFloatText( &m->subsurface );
		if (entry->FirstChildElement( "specular" )) entry->FirstChildElement( "specular" )->QueryFloatText( &m->specular );
		if (entry->FirstChildElement( "roughness" )) entry->FirstChildElement( "roughness" )->QueryFloatText( &m->roughness );
		if (entry->FirstChildElement( "specularTint" )) entry->FirstChildElement( "specularTint" )->QueryFloatText( &m->specularTint );
		if (entry->FirstChildElement( "anisotropic" )) entry->FirstChildElement( "anisotropic" )->QueryFloatText( &m->anisotropic );
		if (entry->FirstChildElement( "sheen" )) entry->FirstChildElement( "sheen" )->QueryFloatText( &m->sheen );
		if (entry->FirstChildElement( "sheenTint" )) entry->FirstChildElement( "sheenTint" )->QueryFloatText( &m->sheenTint );
		if (entry->FirstChildElement( "clearcoat" )) entry->FirstChildElement( "clearcoat" )->QueryFloatText( &m->clearcoat );
		if (entry->FirstChildElement( "clearcoatGloss" )) entry->FirstChildElement( "clearcoatGloss" )->QueryFloatText( &m->clearcoatGloss );
		if (entry->FirstChildElement( "transmission" )) entry->FirstChildElement( "transmission" )->QueryFloatText( &m->transmission );
		if (entry->FirstChildElement( "eta" )) entry->FirstChildElement( "eta" )->QueryFloatText( &m->eta );
		if (entry->FirstChildElement( "custom0" )) entry->FirstChildElement( "custom0" )->QueryFloatText( &m->custom0 );
		if (entry->FirstChildElement( "custom1" )) entry->FirstChildElement( "custom1" )->QueryFloatText( &m->custom1 );
		if (entry->FirstChildElement( "custom2" )) entry->FirstChildElement( "custom2" )->QueryFloatText( &m->custom2 );
		if (entry->FirstChildElement( "custom3" )) entry->FirstChildElement( "custom3" )->QueryFloatText( &m->custom3 );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::Init                                                            |
//  |  Prepare scene geometry for rendering.                                LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::Init()
{
	// initialize skydome
	sky = new HostSkyDome();
	sky->Load();
	// initialize the camera
	camera = new Camera();
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMesh                                                         |
//  |  Create a mesh specified by a file name and data dir, apply a scale, add    |
//  |  the mesh to the list of meshes and return the mesh ID.               LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMesh( const char* objFile, const char* dataDir, const float scale )
{
	HostMesh* newMesh = new HostMesh( objFile, dataDir, scale );
	newMesh->ID = (int)meshes.size();
	meshes.push_back( newMesh );
	return newMesh->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddQuad                                                         |
//  |  Create a mesh that consists of two triangles, described by a normal, a     |
//  |  centroid position and a material. Typically used to add an area light      |
//  |  to a scene.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddQuad( float3 N, const float3 pos, const float width, const float height, const int material )
{
	HostMesh* newMesh = new HostMesh();
	N = normalize( N ); // let's not assume the normal is normalized.
	// "Building an Orthonormal Basis, Revisited"
	const float sign = copysignf( 1.0f, N.z ), a = -1.0f / (sign + N.z), b = N.x * N.y * a;
	const float3 B = 0.5f * width * make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	const float3 T = 0.5f * height * make_float3( b, sign + N.y * N.y * a, -N.y );
	// calculate corners
	newMesh->vertices.push_back( make_float4( pos - B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos - B + T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B - T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos + B + T, 1 ) );
	newMesh->vertices.push_back( make_float4( pos - B + T, 1 ) );
	// connectivity data for two triangles
	newMesh->indices.push_back( make_uint3( 0, 1, 2 ) );
	newMesh->indices.push_back( make_uint3( 3, 4, 5 ) );
	// triangles
	HostTri tri1, tri2;
	tri1.material = tri2.material = material;
	tri1.vN0 = tri1.vN1 = tri1.vN2 = N;
	tri2.vN0 = tri2.vN1 = tri2.vN2 = N;
	tri1.Nx = N.x, tri1.Ny = N.y, tri1.Nz = N.z;
	tri2.Nx = N.x, tri2.Ny = N.y, tri2.Nz = N.z;
	tri1.u0 = tri1.u1 = tri1.u2 = tri1.v0 = tri1.v1 = tri1.v2 = 0;
	tri2.u0 = tri2.u1 = tri2.u2 = tri2.v0 = tri2.v1 = tri2.v2 = 0;
	tri1.vertex0 = make_float3( newMesh->vertices[0] );
	tri1.vertex1 = make_float3( newMesh->vertices[1] );
	tri1.vertex2 = make_float3( newMesh->vertices[2] );
	tri2.vertex0 = make_float3( newMesh->vertices[3] );
	tri2.vertex1 = make_float3( newMesh->vertices[4] );
	tri2.vertex2 = make_float3( newMesh->vertices[5] );
	newMesh->triangles.push_back( tri1 );
	newMesh->triangles.push_back( tri2 );
	// add mesh to scene mesh list
	newMesh->ID = (int)meshes.size();
	newMesh->materialList.push_back( material );
	meshes.push_back( newMesh );
	return newMesh->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddInstance                                                     |
//  |  Create an instance based on a mesh ID, set the transform,                  |
//  |  add the instance to the list of instances,                                 |
//  |  scan the geometry for light emitting triangles, and                        |
//  |  return the instance id.                                              LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddInstance( const int meshId, const mat4& transform )
{
	HostInstance* newInstance = new HostInstance( meshId, transform );
	newInstance->ID = (int)instances.size();
	instances.push_back( newInstance );
	// scan the mesh for light emitting triangles
	HostMesh* mesh = meshes[meshId];
	for (int i = 0; i < mesh->triangles.size(); i++)
	{
		HostTri* tri = &mesh->triangles[i];
		HostMaterial* mat = materials[tri->material];
		if (mat->color.x > 1 || mat->color.y > 1 || mat->color.z > 1)
		{
			tri->UpdateArea();
			HostTri transformedTri = TransformedHostTri( tri, transform );
			HostAreaLight* light = new HostAreaLight( &transformedTri, i, newInstance->ID );
			tri->ltriIdx = (int)areaLights.size(); // TODO: can't duplicate a light due to this.
			areaLights.push_back( light );
			newInstance->hasLTris = true;
			// Note: TODO: 
			// 1. if a mesh is deleted it should scan the list of area lights
			//    to delete those that no longer exist.
			// 2. if a material is changed from emissive to non-emissive,
			//    meshes using the material should remove their light emitting
			//    triangles from the list of area lights.
			// 3. if a material is changed from non-emissive to emissive,
			//    meshes using the material should update the area lights list.
			// Item 1 can be done efficiently. Items 2 and 3 require a list
			// of materials per mesh to be efficient.
		}
	}
	return newInstance->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::SetInstanceTransform                                            |
//  |  Set the transform for an instance. If the instance uses an emissive        |
//  |  material, also update the affected area lights.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostScene::SetInstanceTransform( const int instId, const mat4& T )
{
	instances[instId]->transform = T;
	if (instances[instId]->hasLTris)
	{
		HostMesh* mesh = meshes[instances[instId]->meshIdx];
		for (int i = 0; i < mesh->triangles.size(); i++)
		{
			HostTri* tri = &mesh->triangles[i];
			if (tri->ltriIdx == -1) continue;
			tri->UpdateArea();
			HostTri transformedTri = TransformedHostTri( tri, T );
			*areaLights[tri->ltriIdx] = HostAreaLight( &transformedTri, i, instId );
		}
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindOrCreateTexture                                             |
//  |  Return a texture: if it already exists, return the existing texture (after |
//  |  increasing its refCount), otherwise, create a new texture and return its   |
//  |  ID.                                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindOrCreateTexture( const string& origin, const uint modFlags )
{
	// search list for existing texture
	for (auto texture : textures) if (texture->Equals( origin, modFlags ))
	{
		texture->refCount++;
		return texture->ID;
	}
	// nothing found, create a new texture
	return CreateTexture( origin, modFlags );
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::FindOrCreateMaterial                                            |
//  |  Return a material: if it already exists, return the existing material      |
//  |  (after increasing its refCount), otherwise, create a new texture and       |
//  |  return its ID.                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindOrCreateMaterial( const string& name )
{
	// search list for existing texture
	for (auto material : materials) if (material->name.compare( name ) == 0)
	{
		material->refCount++;
		return material->ID;
	}
	// nothing found, create a new texture
	const int newID = AddMaterial( make_float3( 0 ) );
	materials[newID]->name = name;
	return newID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::GetTriangleMaterial                                             |
//  |  Retrieve the material ID for the specified triangle.                 LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::GetTriangleMaterial( const int instid, const int triid )
{
	if (triid == -1) return -1;
	return meshes[instances[instid]->meshIdx]->triangles[triid].material;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::GetTriangleMaterial                                             |
//  |  Find the ID of a material with the specified name.                   LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::FindMaterialID( const char* name )
{
	for (auto material : materials) if (material->name.compare( name ) == 0) return material->ID;
	return -1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::CreateTexture                                                   |
//  |  Return a texture. Create it anew, even if a texture with the same origin   |
//  |  already exists.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::CreateTexture( const string& origin, const uint modFlags )
{
	// create a new texture
	HostTexture* newTexture = new HostTexture( origin.c_str(), modFlags );
	textures.push_back( newTexture );
	return (int)textures.size() - 1;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddMaterial                                                     |
//  |  Create a material, with a limited set of parameters.                 LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddMaterial( const float3 color )
{
	HostMaterial* material = new HostMaterial();
	material->color = color;
	material->ID = (int)materials.size();
	materials.push_back( material );
	return material->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddPointLight                                                   |
//  |  Create a point light and add it to the scene.                        LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddPointLight( const float3 pos, const float3 radiance, bool enabled )
{
	HostPointLight* light = new HostPointLight();
	light->position = pos;
	light->radiance = radiance;
	light->enabled = enabled;
	light->ID = (int)pointLights.size();
	pointLights.push_back( light );
	return light->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddSpotLight                                                    |
//  |  Create a spot light and add it to the scene.                         LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddSpotLight( const float3 pos, const float3 direction, const float inner, const float outer, const float3 radiance, bool enabled )
{
	HostSpotLight* light = new HostSpotLight();
	light->position = pos;
	light->direction = direction;
	light->radiance = radiance;
	light->cosInner = inner;
	light->cosOuter = outer;
	light->enabled = enabled;
	light->ID = (int)spotLights.size();
	spotLights.push_back( light );
	return light->ID;
}

//  +-----------------------------------------------------------------------------+
//  |  HostScene::AddDirectionalLight                                             |
//  |  Create a directional light and add it to the scene.                  LH2'19|
//  +-----------------------------------------------------------------------------+
int HostScene::AddDirectionalLight( const float3 direction, const float3 radiance, bool enabled )
{
	HostDirectionalLight* light = new HostDirectionalLight();
	light->direction = direction;
	light->radiance = radiance;
	light->enabled = enabled;
	light->ID = (int)directionalLights.size();
	directionalLights.push_back( light );
	return light->ID;
}

// EOF