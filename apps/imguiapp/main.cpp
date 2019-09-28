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
#include "system.h"
#include "rendersystem.h"

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static Shader* shader = 0;
static uint scrwidth = 0, scrheight = 0, scrspp = 1;
static bool camMoved = false, hasFocus = true, running = true;

#include "main_tools.h"

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// initialize scene
	renderer->AddScene( "scene.gltf", "data\\pica\\", mat4::Translate( 0, -10.2f, 0 ) );
	// renderer->AddScene( "CesiumMan.glb", "data\\", mat4::Translate( 0, -2, -9 ) );
	// renderer->AddScene( "InterpolationTest.glb", "data\\", mat4::Translate( 0, 2, -5 ) );
	// renderer->AddScene( "AnimatedMorphCube.glb", "data\\", mat4::Translate( 0, 2, 9 ) );
	int rootNode = renderer->FindNode( "RootNode (gltf orientation matrix)" );
	renderer->SetNodeTransform( rootNode, mat4::RotateX( -PI / 2 ) );
	int lightMat = renderer->AddMaterial( make_float3( 100, 100, 80 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 26.0f, 0 ), 6.9f, 6.9f, lightMat );
	renderer->AddInstance( lightQuad );
}

//  +-----------------------------------------------------------------------------+
//  |  HandleInput                                                                |
//  |  Process user input.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
bool HandleInput( float frameTime )
{
	// handle keyboard input
	float translateSpeed = (GetAsyncKeyState( VK_SHIFT ) ? 15.0f : 5.0f) * frameTime, rotateSpeed = 2.5f * frameTime;
	bool changed = false;
	Camera* camera = renderer->GetCamera();
	if (GetAsyncKeyState( 'A' )) { changed = true; camera->TranslateRelative( make_float3( -translateSpeed, 0, 0 ) ); }
	if (GetAsyncKeyState( 'D' )) { changed = true; camera->TranslateRelative( make_float3( translateSpeed, 0, 0 ) ); }
	if (GetAsyncKeyState( 'W' )) { changed = true; camera->TranslateRelative( make_float3( 0, 0, translateSpeed ) ); }
	if (GetAsyncKeyState( 'S' )) { changed = true; camera->TranslateRelative( make_float3( 0, 0, -translateSpeed ) ); }
	if (GetAsyncKeyState( 'R' )) { changed = true; camera->TranslateRelative( make_float3( 0, translateSpeed, 0 ) ); }
	if (GetAsyncKeyState( 'F' )) { changed = true; camera->TranslateRelative( make_float3( 0, -translateSpeed, 0 ) ); }
	if (GetAsyncKeyState( 'B' )) changed = true; // force restart
	if (GetAsyncKeyState( VK_UP )) { changed = true; camera->TranslateTarget( make_float3( 0, -rotateSpeed, 0 ) ); }
	if (GetAsyncKeyState( VK_DOWN )) { changed = true; camera->TranslateTarget( make_float3( 0, rotateSpeed, 0 ) ); }
	if (GetAsyncKeyState( VK_LEFT )) { changed = true; camera->TranslateTarget( make_float3( -rotateSpeed, 0, 0 ) ); }
	if (GetAsyncKeyState( VK_RIGHT )) { changed = true; camera->TranslateTarget( make_float3( rotateSpeed, 0, 0 ) ); }
	// let the main loop know if the camera should update
	return changed;
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
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_optix7.dll" );			// OPTIX7 core, best for RTX devices
	renderer = RenderAPI::CreateRenderAPI( "rendercore_optixprime_b.dll" );			// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_primeref.dll" );			// REFERENCE, for image validation
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_optixrtx_b.dll" );		// OPTIX6 core, for reference
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_softrasterizer.dll" );	// RASTERIZER, your only option if not on NVidia

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
		renderer->SynchronizeSceneData();
		Convergence c = Converge;
		if (camMoved) c = Restart, camMoved = false;
		// detect camera changes
		if (renderer->GetCamera()->Changed()) camMoved = true;
		// poll events, may affect probepos so needs to happen between HandleInput and Render
		glfwPollEvents();
		// update animations
		for( int i = 0; i < renderer->AnimationCount(); i++ )
		{
			renderer->UpdateAnimation( i, deltaTime );
			camMoved = true; // will remain false if scene has no animations
		}
		// render
		deltaTime = timer.elapsed();
		timer.reset();
		renderer->Render( c );
		if (HandleInput( deltaTime )) camMoved = true;
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
		CoreStats coreStats = renderer->GetCoreStats();
		ImGui::Text( "Frame time:   %6.2fms", coreStats.renderTime * 1000 );
		ImGui::Text( "Primary rays: %6.2fms", coreStats.traceTime0 * 1000 );
		ImGui::Text( "Secondary:    %6.2fms", coreStats.traceTime1 * 1000 );
		ImGui::Text( "Deep rays:    %6.2fms", coreStats.traceTimeX * 1000 );
		ImGui::Text( "Shadow rays:  %6.2fms", coreStats.shadowTraceTime * 1000 );
		ImGui::Text( "Shading time: %6.2fms", coreStats.shadeTime * 1000 );
		ImGui::Text( "# primary:    %6ik (%6.1fM/s)", coreStats.primaryRayCount / 1000, coreStats.primaryRayCount / (max( 1.0f, coreStats.traceTime0 * 1000000 )) );
		ImGui::Text( "# secondary:  %6ik (%6.1fM/s)", coreStats.bounce1RayCount / 1000, coreStats.bounce1RayCount / (max( 1.0f, coreStats.traceTime1 * 1000000 )) );
		ImGui::Text( "# deep rays:  %6ik (%6.1fM/s)", coreStats.deepRayCount / 1000, coreStats.deepRayCount / (max( 1.0f, coreStats.traceTimeX * 1000000 )) );
		ImGui::Text( "# shadw rays: %6ik (%6.1fM/s)", coreStats.totalShadowRays / 1000, coreStats.totalShadowRays / (max( 1.0f, coreStats.shadowTraceTime * 1000000 )) );
		ImGui::End();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
		// finalize
		glfwSwapBuffers( window );
		// terminate
		if (!running) break;
	}
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