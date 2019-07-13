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
static uint scrwidth = 0, scrheight = 0;
static bool running = true, hasFocus = true;
static float r = 0;
static bool camMoved = false;
static string materialFile;

#define SCENE	1 // 1: Sponza, 2: Chamfer, 3: Waterbox
#define SPP		1

#include "main_tools.h"

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// initialize scene
	int worldID = renderer->AddMesh( "scene.gltf", "data\\pica\\", 10.0f );
	int lightMat = renderer->AddMaterial( make_float3( 250, 250, 200 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 26.0f, 0 ), 6.9f, 6.9f, lightMat );
	renderer->AddInstance( worldID );
	renderer->AddInstance( lightQuad );
	materialFile = string( "data\\pica\\pica_materials.xml" );
}

//  +-----------------------------------------------------------------------------+
//  |  HandleInput                                                                |
//  |  Process user input.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
bool HandleInput( float frameTime )
{
	if (!hasFocus) return false;
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
	// initialize OpenGL
	InitGLFW();

	// initialize renderer: pick one
	renderer = RenderAPI::CreateRenderAPI( "rendercore_optixprime_b.dll" );		// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_optixrtx_b.dll" );				// pure OPTIX, the only way to use RTX cores
	// renderer = RenderAPI::CreateRenderAPI( "rendercore_softrasterizer.dll" );	// RASTERIZER, your only option if not on NVidia

	renderer->LoadCamera( "camera.xml" );
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
		// render
		deltaTime = timer.elapsed();
		timer.reset();
		renderer->Render( c );
		if (HandleInput( deltaTime )) camMoved = true;
		// finalize and present
		shader->Bind();
		shader->SetInputTexture( 0, "color", renderTarget );
		shader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		shader->Unbind();
		// finalize
		glfwSwapBuffers( window );
		if (!running) break;
	}
	// clean up
	renderer->Shutdown();
	glfwDestroyWindow( window );
	glfwTerminate();
	return 0;
}

// EOF