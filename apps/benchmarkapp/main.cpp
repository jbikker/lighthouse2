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
#include "direct.h" // for _chdir
#include <bitset>
#include <map>

#define DEMOMODE

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0;
static Shader* shader = 0;
static uint scrwidth = 0, scrheight = 0, scrspp = 2;
static bool spaceDown = false, enterDown = false, tabDown = false, hasFocus = true, running = true, animPaused = false;
static std::bitset<1024> keystates;
static std::bitset<8> mbstates;
static string materialFile;

// material editing
HostMaterial currentMaterial;
int currentMaterialID = -1;
static CoreStats coreStats;

// spline
vector<float3> camPos, camTarget;
float camTime;
int camSegment, camTrack;
bool camPlaying = false;
float delay;

// frame timer
Timer frameTimer;
float frameTime = 0;

// reporting
int convergedSamples = 0;
bool firstConvergingFrame = true;
bool firstFrameAfterConverge = false;
float reportTimer = 0;
Bitmap* rawFrame1 = 0, * rawFrame2 = 0; // noisy & converged frame, for comparison
uint rawFrameSize = 0;
float RMSE = 0;

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
	// book scene
	materialFile = string( "data/book/materials.xml" );
	renderer->AddScene( "data/book/scene.gltf" );
	// bird
	// renderer->AddScene( "data/bird/scene.gltf", mat4::Translate( 0, 4, 0 ) * mat4::Scale( 0.005f ) );
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
	if (!keystates[GLFW_KEY_ENTER]) enterDown = false; else
	{
		if (!enterDown)
		{
			// play current spline
			camPlaying = (camPos.size() > 1);
			FILE* f = fopen( "spline.txt", "w" );
			for (int i = 0; i < camPos.size(); i++) fprintf( f, "(%f,%f,%f) -> (%f,%f,%f)\n",
				camPos[i].x, camPos[i].y, camPos[i].z, camTarget[i].x, camTarget[i].y, camTarget[i].z );
			fclose( f );
			camTime = 0;
			camSegment = 0;
		}
		enterDown = true;
	}
	if (!keystates[GLFW_KEY_TAB]) tabDown = false; else
	{
		if (!tabDown)
		{
			// add a spline point
			camPos.push_back( camera->position );
			camTarget.push_back( camera->position + camera->direction );
			printf( "points in path: %i\n", (int)camPos.size() );
		}
		tabDown = true;
	}
	if (keystates[GLFW_KEY_BACKSPACE] && camPos.size() > 0)
	{
		// reset spline path data
		camera->position = camPos[0];
		camera->direction = normalize( camTarget[0] - camPos[0] );
		camPos.clear();
		camTarget.clear();
		camPlaying = false;
	}
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
	// play spline path if active
	if (camPlaying)
	{
		camTime += frameTime * 0.5f;
		if (camTime > 1)
		{
			camTime -= 1.0f;
			if (++camSegment == camPos.size() - 1) camPlaying = false, camSegment--, camTime = 0;
		}
		if (camPlaying)
		{
			float3 p1 = camPos[camSegment], p2 = camPos[camSegment + 1];
			float3 p0 = camSegment ? camPos[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
			float3 p3 = camSegment < (camPos.size() - 2) ? camPos[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
			camera->position = CatmullRom( p0, p1, p2, p3, camTime );
			p1 = camTarget[camSegment], p2 = camTarget[camSegment + 1];
			p0 = camSegment ? camTarget[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
			p3 = camSegment < (camTarget.size() - 2) ? camTarget[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
			float3 target = CatmullRom( p0, p1, p2, p3, camTime );
			camera->direction = normalize( target - camera->position );
		}
		else
		{
			// end of line
			camera->position = camPos[camPos.size() - 1];
			camera->direction = normalize( camTarget[camPos.size() - 1] - camera->position );
		}
		printf( "playback: segment %i, t=%5.3f\n", camSegment, camTime );
		changed = true;
	}
	// let the main loop know if the camera should update
	return changed;
}

struct Track { vector<float3> camPos, camTarget; vector<float2> focal; };
vector<Track> track;

bool Playback( float frameTime )
{
	static bool pathLoaded = false;
	if (!pathLoaded)
	{
		FILE* f = fopen( "spline_seq.txt", "r" );
		int tn = -1;
		while (!feof( f ))
		{
			char t[1024];
			fgets( t, 1023, f );
			if (t[0] == '#')
			{
				track.push_back( Track() );
				tn++;
				continue;
			}
			else
			{
				float3 P, T;
				float aperture, fdist;
				int r = sscanf( t, "(%f,%f,%f) -> (%f,%f,%f) %f %f", &P.x, &P.y, &P.z, &T.x, &T.y, &T.z, &aperture, &fdist );
				if (r != 8) continue;
				track[tn].camPos.push_back( P );
				track[tn].camTarget.push_back( T );
				track[tn].focal.push_back( make_float2( fdist, aperture ) );
			}
		}
		fclose( f );
		camTrack = 0;
		camSegment = camTime = 0;
		pathLoaded = true;
		camPlaying = true;
	}
	int N = (int)camPos.size();
	Camera* camera = renderer->GetCamera();
	camTime += frameTime * 0.5f;
	if (camTime > 1)
	{
		camTime -= 1.0f;
		if (++camSegment == track[camTrack].camPos.size() - 1) camPlaying = false, firstConvergingFrame = true, delay = 2.0f, camSegment--, camTime = 0;
	}
	if (camPlaying)
	{
		float3 p1 = track[camTrack].camPos[camSegment];
		float3 p2 = track[camTrack].camPos[camSegment + 1];
		float3 p0 = camSegment ? track[camTrack].camPos[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
		float3 p3 = camSegment < (track[camTrack].camPos.size() - 2) ? track[camTrack].camPos[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
		camera->position = CatmullRom( p0, p1, p2, p3, camTime );
		camera->focalDistance = (1 - camTime) * track[camTrack].focal[camSegment].x + camTime * track[camTrack].focal[camSegment + 1].x;
		camera->aperture = (1 - camTime) * track[camTrack].focal[camSegment].y + camTime * track[camTrack].focal[camSegment + 1].y;
		p1 = track[camTrack].camTarget[camSegment];
		p2 = track[camTrack].camTarget[camSegment + 1];
		p0 = camSegment ? track[camTrack].camTarget[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
		p3 = camSegment < (track[camTrack].camTarget.size() - 2) ? track[camTrack].camTarget[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
		float3 target = CatmullRom( p0, p1, p2, p3, camTime );
		camera->direction = normalize( target - camera->position );
	}
	else
	{
		camera->position = track[camTrack].camPos[track[camTrack].camPos.size() - 1];
		camera->direction = normalize( track[camTrack].camTarget[track[camTrack].camPos.size() - 1] - camera->position );
		delay -= frameTime;
		if (delay < 0)
		{
			camTrack = (camTrack + 1) % track.size();
			camSegment = 0;
			camTime = 0;
			camPlaying = true;
			firstFrameAfterConverge = true;
		}
		return false;
	}
	return true;
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
//  |  CheckRawFrameBuffers                                                       |
//  |  Make sure we have room for frame grabbing.                           LH2'20|
//  +-----------------------------------------------------------------------------+
void CheckRawFrameBuffers()
{
	if (rawFrameSize < scrwidth * scrheight)
	{
		delete rawFrame1;
		delete rawFrame2;
		rawFrame1 = new Bitmap( scrwidth, scrheight );
		rawFrame2 = new Bitmap( scrwidth, scrheight );
		rawFrameSize = scrwidth * scrheight;
	}
	rawFrame1->width = rawFrame2->width = scrwidth;
	rawFrame1->height = rawFrame2->height = scrheight;
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
int main( int argc, char *argv[] )
{
	// digest command line
	if (argc > 1)
	{
		// first argument is simply spp
		char* s = argv[1];
		int v = 0;
		while (*s) v = 10 * v + (*s - '0');
		scrspp = max( 1, min( 16, v ) );
		printf( "rendering with %i spp.\n", scrspp );
	}

	// get to the correct dir if exe is in root of project folder
	_chdir( "./apps/benchmarkapp" );

	// prepare for rendering
	Initialize();

	// prepare FreeType2
	GLTextRenderer* textRenderer = new GLTextRenderer( 24 );
	GLTextRenderer* smallText = new GLTextRenderer( 16 );

	// application main loop
	bool camMoved = true, firstFrame = true;
	while (!glfwWindowShouldClose( window ))
	{
		// poll glfw events
		glfwPollEvents();
		if (!running) break;
		// start renderering in a separate thread
		renderer->Render( camMoved ? Restart : Converge, true /* async */ );
		// camera and user input
		frameTime = frameTimer.elapsed();
		frameTimer.reset();
		camMoved = renderer->GetCamera()->Changed();
		if (hasFocus)
		{
		#ifdef DEMOMODE
			if (Playback( frameTime )) camMoved = true;
		#else
			if (HandleInput( frameTime )) camMoved = true;
		#endif
		}
		if (HandleMaterialChange()) camMoved = true;
		// update animations
	#ifdef DEMOMODE
		if (camPlaying)
		#else
		if (!animPaused)
		#endif
		{
			for (int i = 0; i < renderer->AnimationCount(); i++)
			{
				renderer->UpdateAnimation( i, frameTime );
				camMoved = true; // will remain false if scene has no animations
			}
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
	#ifndef DEMOMODE
		UpdateUI();
	#else
		// state for FreeType2 text rendering
		glEnable( GL_BLEND );
		glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		// render text
		char t[1024];
		static int frameIdx = 0;
		coreStats = renderer->GetCoreStats();
		sprintf( t, "Lighthouse 2 benchmark ALPHA - frame %05i: %5.2fms (%4.1ffps)", frameIdx++, coreStats.renderTime * 1000, 1.0f / coreStats.renderTime );
		textRenderer->Render( t, 0, scrheight - 54 );
		if (!camPlaying)
		{
			if (firstConvergingFrame)
			{
				// grab the output of the renderer so we can compare it to ground truth later.
				CheckRawFrameBuffers();
				renderTarget->CopyTo( rawFrame1 );
				firstConvergingFrame = false;
			}
			sprintf( t, "Rendering ground truth, %04i spp.", convergedSamples );
			smallText->Render( t, 0, scrheight - 84 );
			convergedSamples += scrspp;
		}
		else
		{
			convergedSamples = scrspp;
			if (firstFrameAfterConverge)
			{
				CheckRawFrameBuffers();
				renderTarget->CopyTo( rawFrame2 );
				firstFrameAfterConverge = false;
				// calculate RMSE
				uint* a = rawFrame1->pixels, *b = rawFrame2->pixels;
				double errorSum = 0;
				for( uint y = 0; y < scrheight; y++ ) for( uint x = 0; x < scrwidth; x++, a++, b++ )
				{
					double dr = (1.0f / 256.0f) * abs( (int)((*a >> 16) & 255) - (int)((*b >> 16) & 255) );
					double dg = (1.0f / 256.0f) * abs( (int)((*a >> 8) & 255) - (int)((*b >> 8) & 255) );
					double db = (1.0f / 256.0f) * abs( (int)(*a & 255) - (int)(*b & 255) );
					errorSum += dr * dr + dg * dg + db * db;
				}
				RMSE = (float)sqrt( errorSum / (scrwidth * scrheight) );
				reportTimer = 3.0f;
			}
			if (reportTimer > 0)
			{
				bool show = true;
				if (reportTimer < 0.5f) if ((int)(reportTimer * 18.0f) & 1) show = false;
				sprintf( t, "Last RMSE: %5.3f", RMSE );
				if (show) smallText->Render( t, 0, scrheight - 84 );
				reportTimer -= frameTime;
			}
		}
		// restore state for Lighthouse 2 rendering
		glDisable( GL_BLEND );
	#endif
		glfwSwapBuffers( window );
		firstFrame = false;
	}
	Shutdown();
	return 0;
}

// EOF