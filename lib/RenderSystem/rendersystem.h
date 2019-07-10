/* rendersystem.h - Copyright 2019 Utrecht University

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

#pragma once

#include "system.h"
#include "core_api_base.h"
#ifdef RENDERSYSTEMBUILD
// we will not expose these to the host application
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include "tiny_obj_loader.h"
#include "tinyxml2.h"
#include "FreeImage.h"
#include "fbxsdk.h"
#endif
#include "host_texture.h"
#include "host_material.h"
#include "host_mesh.h"
#include "host_light.h"
#include "host_skydome.h"
#include "camera.h"
#include "host_scene.h"
#include "render_api.h"

#ifdef RENDERSYSTEMBUILD
#include "host_mesh_fbx.h"
#undef APIENTRY
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // for dll loading
using namespace tinyxml2;
#endif

namespace lighthouse2 {

struct RenderSettings
{
	float geometryEpsilon = 1.0e-4f;
	float filterDirectClamp = 15.0f;
	float filterIndirectClamp = 2.5f;
	uint filterEnabled = 1;
	uint TAAEnabled = 1;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderSystem                                                               |
//  |  High-level API.                                                      LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderSystem
{
public:
	// methods
	void Init( const char* dllName );
	void SynchronizeSceneData();
	void Render( ViewPyramid& view, Convergence converge );
	void SetTarget( GLTexture* target, const uint spp );
	void SetProbePos( int2 pos ) { if (core) core->SetProbePos( pos ); }
	void Shutdown();
	CoreStats GetCoreStats() { return core ? core->GetCoreStats() : CoreStats(); }
private:
	// private methods
	void SynchronizeSky();
	void SynchronizeTextures();
	void SynchronizeMaterials();
	void SynchronizeMeshes();
	void SynchronizeInstances();
	void SynchronizeLights();
private:
	// private data members
	CoreAPI_Base* core = nullptr;			// low-level rendering functionality
	GLTexture* renderTarget = nullptr;		// CUDA will render to this OpenGL texture
public:
	// public data members
	HostScene* scene = nullptr;				// scene I/O and management module
	RenderSettings settings;				// render settings container
};

} // namespace lighthouse2

// EOF