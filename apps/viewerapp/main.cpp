/* main.cpp - Copyright 2019/2020 Utrecht University

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

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static Shader* shader = 0;
static uint scrwidth = 0, scrheight = 0, scrspp = 2;
static double mousex, mousey, x_start, y_start;
static bool spaceDown = false, hasFocus = true, running = true, animPaused = false, dragging = false;
static bool keystates[1024], mbstates[8];

// material editing
static HostMaterial currentMaterial;
static int currentMaterialID = -1;
static CoreStats coreStats;

// arcball
static float3 arcStart;
static int2 arcPivot;
static mat4 matStart;
static int arcInst = 0;
static int cloud, knight, plane;

#include "main_tools.h"
#include "main_ui.h"

float3 arcball( int x, int y, int2 pivot )
{
	float3 P = make_float3( (float)(x - pivot.x) / (scrwidth * 0.5f), (float)(y - pivot.y) / (scrheight * -0.5f), 0 );
	const float P2 = P.x * P.x + P.y * P.y;
	if (P2 <= 1) P.z = sqrt( 1 - P2 ); else P = normalize( P );
	return P;
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
			currentMaterial.Changed(); // update checksum so we can track ch		
		}
		// camera->focalDistance = coreStats.probedDist;
		changed = true;
	}
	else
	{
		// arcball
		if (mbstates[GLFW_MOUSE_BUTTON_1])
		{
			if (!dragging)
			{
				x_start = mousex, y_start = mousey;
				arcPivot = make_int2( scrwidth / 2, scrheight / 2 );
				matStart = camera->GetMatrix();
				arcStart = arcball( mousex, mousey, arcPivot );
				dragging = true;
			}
			else if (mousex != x_start || mousey != y_start)
			{
				float3 arcEnd = arcball( mousex, mousey, arcPivot );
				float angle = acosf( min( 1.0f, dot( arcStart, arcEnd ) ) );
				float3 a = cross( arcStart, arcEnd );
				mat4 rotation = mat4::Orthonormalize( mat4::Rotate( a, -angle ) ) * matStart;
				camera->SetMatrix( rotation );
				changed = true;
			}
		}
		else dragging = false;
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
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7" );			// OPTIX7 core, best for RTX devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_B" );		// OPTIX PRIME, best for pre-RTX CUDA devices

	renderer->DeserializeCamera( "camera.xml" );
	// set initial window size
	ReshapeWindowCallback( 0, SCRWIDTH, SCRHEIGHT );
	// initialize scene
	auto sky = new HostSkyDome();
	sky->Load( "../_shareddata/sky_15.hdr" );
	sky->worldToLight = mat4::RotateX( -PI / 2 ); // compensate for different evaluation in PBRT
	renderer->GetScene()->SetSkyDome( sky );
	cloud = renderer->AddScene( "../_shareddata/cloud3/scene.gltf" );
	knight = renderer->AddScene( "../_shareddata/knight/scene.gltf", mat4::Translate( 8, 0, 0 ) * mat4::Scale( 8 ) );
	int floorMat = renderer->AddMaterial( make_float3( 1 ), "floormaterial" );
	HostMaterial* m = renderer->GetMaterial( floorMat );
	plane = renderer->AddQuad( make_float3( 0, 1, 0 ), make_float3( 0, -2, 0 ), 100, 100, floorMat );
	renderer->AddInstance( plane );
	renderer->DeserializeMaterials( "materials.xml" );
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
void main()
{
	// prepare for rendering
	Initialize();
	// application main loop
	bool camMoved = true;
	Timer frameTimer;
	while (!glfwWindowShouldClose( window ))
	{
		// poll glfw events
		glfwPollEvents();
		if (!running) break;
		// start renderering in a separate thread
		renderer->Render( camMoved ? Restart : Converge, true /* async */ );
		// camera and user input
		const float frameTime = frameTimer.elapsed();
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
		coreStats = renderer->GetCoreStats();
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
	}
	// save material changes and camera
	renderer->SerializeMaterials( "materials.xml" );
	renderer->SerializeCamera( "camera.xml" );
	// clean up
	renderer->Shutdown();
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow( window );
	glfwTerminate();
}

// EOF