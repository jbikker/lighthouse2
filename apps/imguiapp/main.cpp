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
static uint scrwidth = 0, scrheight = 0, scrspp = 1;
static bool camMoved = false, spaceDown = false, hasFocus = true, running = true, animPaused = false;
static std::bitset<1024> keystates;
static std::bitset<8> mbstates;
static string materialFile;

// material editing
HostMaterial currentMaterial;
int currentMaterialID = -1;
static CoreStats coreStats;

#include "main_tools.h"

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// initialize scene
#if 1
	// radio
	materialFile = string( "data/receiver/red_materials.xml" );
	renderer->AddScene( "scene.gltf", "data/receiver/", mat4::Scale( 0.2f ) * mat4::Translate( 0, 0, 0 ) );
	int rootNode = renderer->FindNode( "RootNode (gltf orientation matrix)" );
	renderer->SetNodeTransform( rootNode, mat4::RotateX( -PI / 2 ) );
	int floorMat = renderer->AddMaterial( make_float3( 0.5f, 0.5f, 0.6f ) );
	int floorQuad = renderer->AddQuad( make_float3( 0, 1, 0 ), make_float3( 0, -1.5f, 0 ), 40, 40, floorMat );
	renderer->AddInstance( floorQuad );
	animPaused = true;
#else
	// classic scene
	materialFile = string( "data/pica/pica_materials.xml" );
	renderer->AddScene( "scene.gltf", "data/pica/", mat4::Translate( 0, -10.2f, 0 ) );
	int rootNode = renderer->FindNode( "RootNode (gltf orientation matrix)" );
	renderer->SetNodeTransform( rootNode, mat4::RotateX( -PI / 2 ) );
#endif
#if 1
	// overhead light, use regular PT
	int lightMat = renderer->AddMaterial( make_float3( 100, 100, 80 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 26.0f, 0 ), 6.9f, 6.9f, lightMat );
#else
	// difficult light; use BDPT
	int lightMat = renderer->AddMaterial( make_float3( 500, 500, 400 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0.15188693, -0.32204545, 0.93446094 ), make_float3( -12.938412, -5.0068984, -25.725601 ), 1.9f, 1.9f, lightMat );
#endif
	int lightInst = renderer->AddInstance( lightQuad );
	// optional animated models
	// renderer->AddScene( "CesiumMan.glb", "data/", mat4::Translate( 0, -2, -9 ) );
	// renderer->AddScene( "project_polly.glb", "data/", mat4::Translate( 4.5f, -5.45f, -5.2f ) * mat4::Scale( 2 ) );
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
	Camera *camera = renderer->GetCamera();
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
		// camera->focalDistance = coreStats.probedDist;
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
		// put it back
		*renderer->GetMaterial( currentMaterialID ) = currentMaterial;
		renderer->GetMaterial( currentMaterialID )->MarkAsDirty();
		return true;
	}
	return false;
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
int main()
{
	// initialize OpenGL and ImGui
	InitGLFW();
	InitImGui();

	// initialize renderer: pick one
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7filter" );		// OPTIX7 core, with filtering (static scenes only for now)
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7" );				// OPTIX7 core, best for RTX devices
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_B" );		// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_PrimeRef" );			// REFERENCE, for image validation
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_SoftRasterizer" );	// RASTERIZER, your only option if not on NVidia
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Minimal" );			// MINIMAL example, to get you started on your own core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Vulkan_RT" );			// Meir's Vulkan / RTX core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_BDPT" );	// Peter's OptixPrime / BDPT core

	renderer->DeserializeCamera( "camera.xml" );
	// initialize scene
	PrepareScene();
	// set initial window size
	ReshapeWindowCallback( 0, SCRWIDTH, SCRHEIGHT );
	// enter main loop
	Timer timer;
	timer.reset();
	float deltaTime = 0;
	while (!glfwWindowShouldClose( window ))
	{
		// detect camera changes
		camMoved = false;
		deltaTime = timer.elapsed();
		if (HandleInput( deltaTime )) camMoved = true;
		// handle material changes
		if (HandleMaterialChange()) camMoved = true;
		// poll events, may affect probepos so needs to happen between HandleInput and Render
		glfwPollEvents();
		// update animations
		if (!animPaused) for (int i = 0; i < renderer->AnimationCount(); i++)
		{
			renderer->UpdateAnimation( i, deltaTime );
			camMoved = true; // will remain false if scene has no animations
		}
		renderer->SynchronizeSceneData();
		// render
		timer.reset();
		renderer->Render( camMoved ? Restart : Converge );
		// postprocess
		shader->Bind();
		shader->SetInputTexture( 0, "color", renderTarget );
		shader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		shader->Unbind();
		// gui
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::Begin( "Render statistics", 0 );
		coreStats = renderer->GetCoreStats();
		SystemStats systemStats = renderer->GetSystemStats();
		ImGui::Text( "Frame time:   %6.2fms", coreStats.renderTime * 1000 );
		ImGui::Text( "Scene update: %6.2fms", systemStats.sceneUpdateTime * 1000 );
		ImGui::Text( "Primary rays: %6.2fms", coreStats.traceTime0 * 1000 );
		ImGui::Text( "Secondary:    %6.2fms", coreStats.traceTime1 * 1000 );
		ImGui::Text( "Deep rays:    %6.2fms", coreStats.traceTimeX * 1000 );
		ImGui::Text( "Shadow rays:  %6.2fms", coreStats.shadowTraceTime * 1000 );
		ImGui::Text( "Shading time: %6.2fms", coreStats.shadeTime * 1000 );
		ImGui::Text( "Filter time:  %6.2fms", coreStats.filterTime * 1000 );
		ImGui::Text( "# primary:    %6ik (%6.1fM/s)", coreStats.primaryRayCount / 1000, coreStats.primaryRayCount / (max( 1.0f, coreStats.traceTime0 * 1000000 )) );
		ImGui::Text( "# secondary:  %6ik (%6.1fM/s)", coreStats.bounce1RayCount / 1000, coreStats.bounce1RayCount / (max( 1.0f, coreStats.traceTime1 * 1000000 )) );
		ImGui::Text( "# deep rays:  %6ik (%6.1fM/s)", coreStats.deepRayCount / 1000, coreStats.deepRayCount / (max( 1.0f, coreStats.traceTimeX * 1000000 )) );
		ImGui::Text( "# shadw rays: %6ik (%6.1fM/s)", coreStats.totalShadowRays / 1000, coreStats.totalShadowRays / (max( 1.0f, coreStats.shadowTraceTime * 1000000 )) );
		ImGui::End();
		ImGui::Begin( "Material parameters", 0 );
		ImGui::Text( "name:    %s", currentMaterial.name.c_str() );
		ImGui::ColorEdit3( "color", (float*)&currentMaterial.color );
		ImGui::ColorEdit3( "absorption", (float*)&currentMaterial.absorption );
		ImGui::SliderFloat( "metallic", &currentMaterial.metallic, 0, 1 );
		ImGui::SliderFloat( "subsurface", &currentMaterial.subsurface, 0, 1 );
		ImGui::SliderFloat( "specular", &currentMaterial.specular, 0, 1 );
		ImGui::SliderFloat( "roughness", &currentMaterial.roughness, 0, 1 );
		ImGui::SliderFloat( "specularTint", &currentMaterial.specularTint, 0, 1 );
		ImGui::SliderFloat( "anisotropic", &currentMaterial.anisotropic, 0, 1 );
		ImGui::SliderFloat( "sheen", &currentMaterial.sheen, 0, 1 );
		ImGui::SliderFloat( "sheenTint", &currentMaterial.sheenTint, 0, 1 );
		ImGui::SliderFloat( "clearcoat", &currentMaterial.clearcoat, 0, 1 );
		ImGui::SliderFloat( "clearcoatGloss", &currentMaterial.clearcoatGloss, 0, 1 );
		ImGui::SliderFloat( "transmission", &currentMaterial.transmission, 0, 1 );
		ImGui::SliderFloat( "eta (1/ior)", &currentMaterial.eta, 0.25f, 1.0f );
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
		// finalize
		glfwSwapBuffers( window );
		// terminate
		if (!running) break;
	}
	// save material changes
	renderer->SerializeMaterials( materialFile.c_str() );
	// clean up
	renderer->SerializeCamera( "camera.xml" );
	renderer->Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow( window );
	glfwTerminate();
	return 0;
}

// EOF