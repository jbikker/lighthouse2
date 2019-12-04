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

#include "navmesh_builder.h"
#include "navmesh_navigator.h"
#include "navmesh_shader.h"
#include "physics_placeholder.h"
#include "navmesh_agents.h"

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static Shader* shader = 0;
static NavMeshBuilder* navMeshBuilder = 0;
static PhysicsPlaceholder* rigidBodies = 0;
static NavMeshAgents* navMeshAgents = 0;

static uint scrwidth = 0, scrheight = 0, scrspp = 1;
static bool posChanges = false, camMoved = false;
static bool hasFocus = true, running = true, paused = false;
static bool leftClicked = false, rightClicked = false;
static int2 probeCoords;

static const float agentUpdateInterval = 2.0f;
static const int maxAgents = 50;
static const int maxAgentPathSize = 8;

#include "main_ui.h"
#include "main_tools.h"

//  +-----------------------------------------------------------------------------+
//  |  PrepareScene                                                               |
//  |  Initialize a scene.                                                  LH2'19|
//  +-----------------------------------------------------------------------------+
void PrepareScene()
{
	// the Recast test scene
	string materialFile = string( "data/nav_test_materials.xml" );
	int meshID = renderer->AddMesh( "nav_test.obj", "data/", 1.0f, true );
	HostScene::meshPool[meshID]->name = "Input Mesh";
	int instID = renderer->AddInstance( meshID, mat4::Identity() );
	int instID2 = renderer->AddInstance( meshID, mat4::Translate( 0, 0, 50 ) );
	int rootNode = renderer->FindNode( "RootNode (gltf orientation matrix)" );
	renderer->SetNodeTransform( rootNode, mat4::RotateX( -PI / 2 ) );
	int lightMat = renderer->AddMaterial( make_float3( 100, 100, 80 ) );
	int lightQuad = renderer->AddQuad( make_float3( 0, -1, 0 ), make_float3( 0, 66.0f, 0 ), 16, 16, lightMat );
	renderer->AddInstance( lightQuad );
	renderer->AddDirectionalLight( make_float3( -1 ), make_float3( 255 ) );

	// Navmesh builder
	navMeshBuilder = new NavMeshBuilder( "data/ai/" );
	NavMeshConfig* config = navMeshBuilder->GetConfig();
	config->SetCellSize( .3f, .2f );
	config->SetAgentInfo( 10.0f, 10, 2, 2 );
	config->SetPolySettings( 12, 1.3f, 8.0f, 20.0f, 6 );
	config->SetDetailPolySettings( 6.0f, 1.0f );
	config->m_printBuildStats = true;
	config->AddAreaType( "GROUND" );
	config->AddAreaType( "ROAD" );
	config->AddAreaType( "STAIRS" );
	config->AddAreaType( "WATER" );
	config->AddAreaType( "DOOR" );
	config->AddFlag( "WALK" );
	config->AddFlag( "CLIMB" );
	config->AddFlag( "JUMP" );
	config->AddFlag( "SWIM" );

	rigidBodies = new PhysicsPlaceholder( maxAgents );
	navMeshAgents = new NavMeshAgents( maxAgents, maxAgentPathSize, agentUpdateInterval );

	renderer->DeserializeMaterials( materialFile.c_str() );
	renderer->SerializeMaterials( materialFile.c_str() ); // so we can edit the materials in the xml file
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
int main()
{
	// initialize OpenGL and ImGui
	InitGLFW();

	// initialize renderer: pick one
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7filter" );			// OPTIX7 core, with filtering (static scenes only for now)
	renderer = RenderAPI::CreateRenderAPI( "rendercore_optix7.dll" );				// OPTIX7 core, best for RTX devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_B" );			// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_PrimeRef" );				// REFERENCE, for image validation
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_SoftRasterizer" );		// RASTERIZER, your only option if not on NVidia
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Minimal" );				// MINIMAL example, to get you started on your own core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Vulkan_RT" );				// Meir's Vulkan / RTX core
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_BDPT" );		// Peter's OptixPrime / BDPT core

	renderer->DeserializeCamera( "camera.xml" );
	// initialize scene
	PrepareScene();
	// initialize ui
	MAIN_UI::InitGUI( renderer, navMeshBuilder, rigidBodies, navMeshAgents, camMoved, hasFocus, leftClicked, rightClicked, paused, probeCoords, scrwidth, scrheight );
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
		for (int i = 0; i < renderer->AnimationCount(); i++)
		{
			renderer->UpdateAnimation( i, deltaTime );
			camMoved = true; // will remain false if scene has no animations
		}
		deltaTime = timer.elapsed();
		timer.reset();
		// Physics
		if (!paused) posChanges = rigidBodies->Update( deltaTime );
		if (posChanges) MAIN_UI::PostPhysicsUpdate( deltaTime );
		// render
		renderer->Render( c );
		MAIN_UI::PostRenderUpdate( deltaTime );
		// AI
		if (!paused) navMeshAgents->UpdateAgentMovement( deltaTime );
		if (!paused) navMeshAgents->UpdateAgentBehavior( deltaTime ); // only updates at a regular interval
		// UI
		if (MAIN_UI::HandleInput( deltaTime )) camMoved = true;
		// postprocess
		shader->Bind();
		shader->SetInputTexture( 0, "color", renderTarget );
		shader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		shader->Unbind();
		// draw ui
		MAIN_UI::DrawGUI( deltaTime );
		// finalize
		glfwSwapBuffers( window );
		// terminate
		if (!running) break;
	}
	// clean up
	MAIN_UI::ShutDown();
	renderer->SerializeCamera( "camera.xml" );
	renderer->Shutdown();
	glfwDestroyWindow( window );
	glfwTerminate();
	return 0;
}

// EOF