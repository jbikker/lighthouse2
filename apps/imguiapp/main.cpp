/* main.cpp - Copyright 2019 Utrecht University

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

#include "platform.h"
#include "rendersystem.h"
#include <bitset>

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static Shader* shader = 0;
static uint scrwidth = 0, scrheight = 0, scrspp = 2;
static bool spaceDown = false, hasFocus = true, running = true, animPaused = false;
static std::bitset<1024> keystates;
static std::bitset<8> mbstates;
static string materialFile;

// material editing
HostMaterial currentMaterial;
int currentMaterialID = -1;
static CoreStats coreStats;

// frame timer
Timer frameTimer;
float frameTime = 0;


#include "main_tools.h"
#include "main_ui.h"

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// initialize scene
	auto scene = renderer->GetScene();
	auto sky = new HostSkyDome();
	sky->Load( "data/sky_15.hdr" );
	// Compensate for different evaluation in PBRT
	sky->worldToLight = mat4::RotateX( -PI / 2 );
	scene->SetSkyDome( sky );
#if 0
	// escher
	materialFile = string( "data/escher/materials.xml" );
	int mesh = renderer->AddMesh( "data/escher/escher2010.obj" );
	int inst = renderer->AddInstance( mesh );
	renderer->SetNodeTransform( inst, mat4::RotateX( PI / 4 ) * mat4::RotateZ( PI / 2 ) );
#else
#if 0
	// classic scene
	materialFile = string( "data/pica/pica_materials.xml" );
	int rootNode = renderer->AddScene( "data/pica/scene.gltf" );
	renderer->SetNodeTransform( rootNode, mat4::Translate( 0, -10.2f, 0 ) );
	rootNode = renderer->AddScene( "data/sphere/scene.gltf" );
	renderer->SetNodeTransform( rootNode, mat4::Scale( 0.02f ) );
	int whiteMat = renderer->AddMaterial( make_float3( 20 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 21, 0 ), 10.9f, 10.9f, whiteMat );
	renderer->AddInstance( lightQuad );
	renderer->AddScene( "data/drone/scene.gltf", mat4::Translate( 4.5f, -3.4f, -5.2f ) * mat4::Scale( 0.02f ) );
#else
	// book scene
	materialFile = string( "data/book/materials.xml" );
	renderer->AddScene( "data/book/scene.gltf" );
	int whiteMat = renderer->AddMaterial( make_float3( 20 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 21, 0 ), 10.9f, 10.9f, whiteMat );
	renderer->AddInstance( lightQuad );
#endif
#endif
	// optional animated models
	// renderer->AddScene( "data/CesiumMan.glb", mat4::Translate( 0, -2, -9 ) );
	// renderer->AddScene( "data/project_polly.glb", mat4::Translate( 4.5f, -5.45f, -5.2f ) * mat4::Scale( 2 ) );
	// test data for PNEE
	// int lightText = renderer->AddMesh( "data/lh2text.obj", 0.1f );
	// renderer->AddInstance( lightText, mat4::Translate( make_float3( -1, -3.7f, 0 ) ) * mat4::RotateX( PI / 2 ) );
	// for( int i = 0; i < 5; i++ ) renderer->AddPointLight( make_float3( (i - 2) * 8.0f, 6.0f, -18.82f ), make_float3( 100, 0, 0 ) );
	// load changed materials
	renderer->DeserializeMaterials( materialFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  HandleInput                                                                |
//  |  Process user input.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
bool HandleInput( float frameTime )
{
	// handle keyboard input
	float tspd = (keystates[GLFW_KEY_LEFT_SHIFT] ? 15.0f : 5.0f) * frameTime, rspd = 2.5f * frameTime;
	bool changed = false;
	Camera* camera = renderer->GetCamera();
	if (keystates[GLFW_KEY_A]) { changed = true; camera->TranslateRelative( make_float3( -tspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_D]) { changed = true; camera->TranslateRelative( make_float3( tspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_W]) { changed = true; camera->TranslateRelative( make_float3( 0, 0, tspd ) ); }
	if (keystates[GLFW_KEY_S]) { changed = true; camera->TranslateRelative( make_float3( 0, 0, -tspd ) ); }
	if (keystates[GLFW_KEY_R]) { changed = true; camera->TranslateRelative( make_float3( 0, tspd, 0 ) ); }
	if (keystates[GLFW_KEY_F]) { changed = true; camera->TranslateRelative( make_float3( 0, -tspd, 0 ) ); }
	if (keystates[GLFW_KEY_B]) changed = true; // force restart
	if (keystates[GLFW_KEY_UP]) { changed = true; camera->TranslateTarget( make_float3( 0, -rspd, 0 ) ); }
	if (keystates[GLFW_KEY_DOWN]) { changed = true; camera->TranslateTarget( make_float3( 0, rspd, 0 ) ); }
	if (keystates[GLFW_KEY_LEFT]) { changed = true; camera->TranslateTarget( make_float3( -rspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_RIGHT]) { changed = true; camera->TranslateTarget( make_float3( rspd, 0, 0 ) ); }
	if (!keystates[GLFW_KEY_SPACE]) spaceDown = false; else { if (!spaceDown) animPaused = !animPaused, changed = true; spaceDown = true; }
	// process left button click
	if (mbstates[GLFW_MOUSE_BUTTON_1] && keystates[GLFW_KEY_LEFT_SHIFT])
	{
		int selectedMaterialID = renderer->GetTriangleMaterialID( coreStats.probedInstid, coreStats.probedTriid );
		if (selectedMaterialID != -1)
		{
			currentMaterial = *renderer->GetMaterial( selectedMaterialID );
			currentMaterialID = selectedMaterialID;
			currentMaterial.Changed(); // update checksum so we can track changes
		}
		camera->focalDistance = coreStats.probedDist;
		changed = true;
	}
	// let the main loop know if the camera should update
	return changed;
}

//  +-----------------------------------------------------------------------------+
//  |  HandleMaterialChange                                                       |
//  |  Update a scene material based on AntTweakBar.                        LH2'19|
//  +-----------------------------------------------------------------------------+
bool HandleMaterialChange()
{
	if (currentMaterial.Changed() && currentMaterialID != -1)
	{
		// local copy of current material has been changed; put it back
		*renderer->GetMaterial( currentMaterialID ) = currentMaterial;
		renderer->GetMaterial( currentMaterialID )->MarkAsDirty();
		return true;
	}
	return false;
}

//  +-----------------------------------------------------------------------------+
//  |  Initialize                                                                 |
//  |  Setup the window, user interface and renderer.                       LH2'20|
//  +-----------------------------------------------------------------------------+
void Initialize()
{
	// initialize OpenGL and ImGui
	InitGLFW();
	InitImGui();

	// initialize renderer: pick one
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7filter" );			// OPTIX7 core, with filtering (static scenes only for now)
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7" );			// OPTIX7 core, best for RTX devices
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_B" );		// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_PrimeRef" );			// REFERENCE, for image validation
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_SoftRasterizer" );	// RASTERIZER, your only option if not on NVidia
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Minimal" );			// MINIMAL example, to get you started on your own core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Vulkan_RT" );			// Meir's Vulkan / RTX core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_BDPT" );	// Peter's OptixPrime / BDPT core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_PBRT" );	// Marijn's PBRT core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7Guiding" );		// OPTIX7 core with path guiding for next event estimation (under construction)

	renderer->DeserializeCamera( "camera.xml" );
	// set initial window size
	ReshapeWindowCallback( 0, SCRWIDTH, SCRHEIGHT );
	// initialize scene
	PrepareScene();
}

//  +-----------------------------------------------------------------------------+
//  |  Shutdown                                                                   |
//  |  Final words.                                                         LH2'20|
//  +-----------------------------------------------------------------------------+
void Shutdown()
{
	// save material changes
	renderer->SerializeMaterials( materialFile.c_str() );
	// save camera
	renderer->SerializeCamera( "camera.xml" );
	// clean up
	renderer->Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow( window );
	glfwTerminate();
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
int main()
{
	// prepare for rendering
	Initialize();
	// application main loop
	bool camMoved = true, firstFrame = true;
	while (!glfwWindowShouldClose( window ))
	{
		// poll glfw events
		glfwPollEvents();
		if (!running) break;
		// start renderering in a separate thread
		renderer->Render( camMoved ? Restart : Converge, false /* async */ );
		// camera and user input
		frameTime = frameTimer.elapsed();
		frameTimer.reset();
		camMoved = renderer->GetCamera()->Changed();
		if (hasFocus) if (HandleInput( frameTime )) camMoved = true;
		if (HandleMaterialChange()) camMoved = true;
		// update animations
		if (!animPaused) for (int i = 0; i < renderer->AnimationCount(); i++)
		{
			renderer->UpdateAnimation( i, frameTime );
			camMoved = true; // will remain false if scene has no animations
		}
		renderer->SynchronizeSceneData();
		// wait for rendering to complete
		renderer->WaitForRender();
		// postprocess
		shader->Bind();
		shader->SetInputTexture( 0, "color", renderTarget );
		shader->SetInputMatrix( "view", mat4::Identity() );
		shader->SetFloat( "contrast", renderer->GetCamera()->contrast );
		shader->SetFloat( "brightness", renderer->GetCamera()->brightness );
		shader->SetFloat( "gamma", renderer->GetCamera()->gamma );
		shader->SetInt( "method", renderer->GetCamera()->tonemapper );
		DrawQuad();
		shader->Unbind();
		// update user interface
		UpdateUI();
		glfwSwapBuffers( window );
		firstFrame = false;
	}
	Shutdown();
	return 0;
}

// EOF