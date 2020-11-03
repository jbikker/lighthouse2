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
#include "direct.h" // for _chdir
#include <bitset>
#include <map>
#include "irrklang.h"
using namespace irrklang;

#define DEMOMODE
// #define OPTIX5FALLBACK

static RenderAPI* renderer = 0;
static GLTexture* renderTarget = 0, * overlayTarget = 0, * menuScreen;
static Shader* shader = 0, * overlayShader = 0;
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
float camTime, birbTime = 0;
int camSegment, camTrack, animCycle = 0, birbSegment = 0, birb = -1;
bool camPlaying = false, animsPlaying = true, fadeOut = false;
float delay, fadeTimer = 4.0f;
float3 birdSpline[] = {
	make_float3( -18.1f, -0.5f, -33.5f ),
	make_float3( -36.4f, -4.0f, -33.7f ),
	make_float3( -46.6f, -4.7f, -21.7f ),
	make_float3( -37.9f, -3.1f, -4.6f ),
	make_float3( -26.6f, -1.8f, 7.6f ),
	make_float3( -13.8f, -0.3f, 12.4f ),
	make_float3( 1.0f, -1.8f, 4.0f ),
	make_float3( 14.5f, -1.0f, 1.2f ),
	make_float3( 26.4f, -3.5f, -0.1f ),
	make_float3( 38.7f, -4.6f, 4.1f ),
	make_float3( 46.0f, -11.4f, 17.2f ),
	make_float3( 38.7f, -4.8f, 31.2f ),
	make_float3( 22.1f, -2.5f, 31.5f ),
	make_float3( 6.5f, 0.1f, 28.7f ),
	make_float3( -4.9f, -2.7f, 18.7f ),
	make_float3( -3.1f, -2.7f, 2.2f ),
	make_float3( -3.3f, -2.7f, -13.2f ),
	make_float3( -3.9f, -2.7f, -28.6f )
};

// clouds
int cloud1 = -1, cloud2 = -1, cloud3 = -1;
float c1pos = 0, c2pos = -25, c3pos = 30;

// frame timer
Timer frameTimer;
float frameTime = 0;
int frameIdx = 0;

// reporting
int convergedSamples = 0;
bool firstConvergingFrame = true;
bool firstFrameAfterConverge = false;
float reportTimer = 0;
Bitmap* rawFrame1 = 0, * rawFrame2 = 0; // noisy & converged frame, for comparison
Bitmap* overlay = 0;
uint rawFrameSize = 0;
float RMSE[6] = {}, lastRMSE = 0;
float fpsSum = 0;
int frameCount = 0;
float smoothedFps = 20, peakFps = 1;
float rpsSum = 0, rpsPeak = 0;
float shdSum = 0, shdPeak = 0;
int redirected = 0;

// sound
ISoundEngine* soundEngine = 0;

// application flow
int state = 0; // menu; 1 is demo, 2 is results
bool camMoved = true, firstFrame = true;

// text rendering
GLTextRenderer* textRenderer, * smallText, * tinyText, * digiText, * smallDigiText;

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
	sky->Load( "../_shareddata/sky_15.hdr" );
	// Compensate for different evaluation in PBRT
	sky->worldToLight = mat4::RotateX( -PI / 2 );
	scene->SetSkyDome( sky );
	// book scene
	materialFile = string( "data/materials.xml" );
	renderer->AddScene( "../_shareddata/book/scene.gltf" );
	// dragon statue
	renderer->AddScene( "../_shareddata/statue/scene.gltf", mat4::Translate( -14, -0.5f, 25 ) );
	// gems
	renderer->AddScene( "../_shareddata/crystal/scene.gltf", mat4::Translate( 27, -4, 6 ) * mat4::RotateZ( -0.25f ) * mat4::Scale( 0.5f ) );
	renderer->AddScene( "../_shareddata/crystal/scene.gltf", mat4::Translate( 31.5f, -4.75f, 6 ) * mat4::Scale( 0.5f ) );
	// knights
	renderer->AddScene( "../_shareddata/knight/scene.gltf", mat4::Translate( -16, 0.75f, -10 ) * mat4::RotateY( -1.2f ) );
	renderer->AddScene( "../_shareddata/knight/scene.gltf", mat4::Translate( -17, 0.75f, -18.5f ) * mat4::RotateY( PI / 2 ) );
	renderer->AddScene( "../_shareddata/knight/scene.gltf", mat4::Translate( -15, 0.75f, -18.5f ) * mat4::RotateY( PI / 2 ) );
	// bird
	birb = renderer->AddScene( "../_shareddata/bird/scene.gltf", mat4::Translate( 0, 14, 0 ) * mat4::Scale( 0.005f ) );
	// clouds
	cloud1 = renderer->AddScene( "../_shareddata/cloud1/scene.gltf", mat4::Translate( 0, 34, 0 ) * mat4::Scale( 4.0f ) );
	cloud2 = renderer->AddScene( "../_shareddata/cloud2/scene.gltf", mat4::Translate( 0, 34, 0 ) * mat4::Scale( 4.0f ) );
	cloud3 = renderer->AddScene( "../_shareddata/cloud3/scene.gltf", mat4::Translate( 0, 34, 0 ) * mat4::Scale( 4.0f ) );
	// light
	int whiteMat = renderer->AddMaterial( make_float3( 30 ) );
	int lightQuad = renderer->AddQuad( normalize( make_float3( -183.9f, -44.6f, -60.9f ) ),
		make_float3( 183.9f, 44.6f, 60.9f ), 30.0f, 80.0f, whiteMat );
	renderer->AddInstance( lightQuad );
	// load changed materials
	renderer->DeserializeMaterials( materialFile.c_str() );
	// aggressive clamping
	renderer->GetCamera()->clampValue = 2.5f;
	// prepare title / results texture
	menuScreen = new GLTexture();
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
	if (keystates[GLFW_KEY_A]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( -tspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_D]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( tspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_W]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( 0, 0, tspd ) ); }
	if (keystates[GLFW_KEY_S]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( 0, 0, -tspd ) ); }
	if (keystates[GLFW_KEY_R]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( 0, tspd, 0 ) ); }
	if (keystates[GLFW_KEY_F]) { changed = true, camTime += frameTime * 0.1f; camera->TranslateRelative( make_float3( 0, -tspd, 0 ) ); }
	camTime = fmod( camTime, 1 );
	if (keystates[GLFW_KEY_B]) changed = true; // force restart
	if (keystates[GLFW_KEY_UP]) { changed = true; camera->TranslateTarget( make_float3( 0, -rspd, 0 ) ); }
	if (keystates[GLFW_KEY_DOWN]) { changed = true; camera->TranslateTarget( make_float3( 0, rspd, 0 ) ); }
	if (keystates[GLFW_KEY_LEFT]) { changed = true; camera->TranslateTarget( make_float3( -rspd, 0, 0 ) ); }
	if (keystates[GLFW_KEY_RIGHT]) { changed = true; camera->TranslateTarget( make_float3( rspd, 0, 0 ) ); }
	if (!keystates[GLFW_KEY_SPACE]) spaceDown = false; else { if (!spaceDown) animPaused = !animPaused, changed = true; spaceDown = true; }
	if (!keystates[GLFW_KEY_ENTER]) enterDown = false; else if (!enterDown)
	{
		// play current spline
		camPlaying = (camPos.size() > 1);
		FILE* f = fopen( "spline.txt", "w" );
		for (int i = 0; i < camPos.size(); i++) fprintf( f, "(%f,%f,%f) -> (%f,%f,%f)\n",
			camPos[i].x, camPos[i].y, camPos[i].z, camTarget[i].x, camTarget[i].y, camTarget[i].z );
		fclose( f );
		camTime = 0, camSegment = 0, enterDown = true;
	}
	if (!keystates[GLFW_KEY_TAB]) tabDown = false; else if (!tabDown) // add a spline point
	{
		camPos.push_back( camera->transform.GetTranslation() );
		camTarget.push_back( camera->transform.GetTranslation() + camera->transform.GetForward() );
		printf( "points in path: %i\n", (int)camPos.size() );
		tabDown = true;
	}
	if (keystates[GLFW_KEY_BACKSPACE] && camPos.size() > 0)
	{
		// reset spline path data
		camera->LookAt( camPos[0], camTarget[0] );
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
			float3 pos = CatmullRom( p0, p1, p2, p3, camTime );
			p1 = camTarget[camSegment], p2 = camTarget[camSegment + 1];
			p0 = camSegment ? camTarget[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
			p3 = camSegment < (camTarget.size() - 2) ? camTarget[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
			float3 target = CatmullRom( p0, p1, p2, p3, camTime );
			camera->LookAt( pos, target );
		}
		else
		{
			// end of line
			camera->LookAt( camPos[camPos.size() - 1], camTarget[camPos.size() - 1] );
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
		int r, tn = -1;
		while (!feof( f ))
		{
			char t[1024];
			fgets( t, 1023, f );
			if (t[0] == '#') { track.push_back( Track() ); tn++; continue; }
			float3 P, T;
			float aperture, fdist;
			if ((r = sscanf( t, "(%f,%f,%f) -> (%f,%f,%f) %f %f", &P.x, &P.y, &P.z, &T.x, &T.y, &T.z, &aperture, &fdist )) != 8) continue;
			track[tn].camPos.push_back( P );
			track[tn].camTarget.push_back( T );
			track[tn].focal.push_back( make_float2( fdist, aperture ) );
		}
		fclose( f );
		camTrack = 0, camSegment = 1, camTime = 0, pathLoaded = true, camPlaying = animsPlaying = true;
	}
	int N = (int)camPos.size();
	Camera* camera = renderer->GetCamera();
	camTime += frameTime * 0.5f;
	if (camTime > 1)
	{
		if (camTrack == 0 && camSegment == 1 && animCycle == 1) fadeOut = true;
		camTime -= 1.0f;
		if (++camSegment == track[camTrack].camPos.size() - 1) camPlaying = animsPlaying = false, firstConvergingFrame = true, delay = 2.0f, camSegment--, camTime = 0;
	}
	if (camPlaying)
	{
		float3 p1 = track[camTrack].camPos[camSegment];
		float3 p2 = track[camTrack].camPos[camSegment + 1];
		float3 p0 = camSegment ? track[camTrack].camPos[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
		float3 p3 = camSegment < (track[camTrack].camPos.size() - 2) ? track[camTrack].camPos[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
		float3 pos = CatmullRom( p0, p1, p2, p3, camTime );
		camera->focalDistance = (1 - camTime) * track[camTrack].focal[camSegment].x + camTime * track[camTrack].focal[camSegment + 1].x;
		camera->aperture = (1 - camTime) * track[camTrack].focal[camSegment].y + camTime * track[camTrack].focal[camSegment + 1].y;
		p1 = track[camTrack].camTarget[camSegment];
		p2 = track[camTrack].camTarget[camSegment + 1];
		p0 = camSegment ? track[camTrack].camTarget[camSegment - 1] : (p1 - 0.01f * (p2 - p1));
		p3 = camSegment < (track[camTrack].camTarget.size() - 2) ? track[camTrack].camTarget[camSegment + 2] : (p2 + 0.01f * (p2 - 1));
		float3 target = CatmullRom( p0, p1, p2, p3, camTime );
		camera->LookAt( pos, target );
	}
	else
	{
		camera->LookAt( track[camTrack].camPos[track[camTrack].camPos.size() - 1],
			track[camTrack].camTarget[track[camTrack].camPos.size() - 1] );
		delay -= frameTime;
		if (delay < 0)
		{
			camTrack = camTrack + 1;
			if (camTrack == track.size()) camTrack = 0, animCycle++;
			camSegment = 0, camTime = 0, camPlaying = animsPlaying = true, delay = 0;
			firstFrameAfterConverge = true;
		}
		return false;
	}
	return true;
}

void UpdateBird( const float frameTime )
{
	if (birb == -1) return;
	birbTime += frameTime * 0.35f;
	if (birbTime > 1)
	{
		birbTime -= 1.0f;
		if (++birbSegment == 18) birbSegment = 0;
	}
	const float3 p0 = birdSpline[(birbSegment + 17) % 18];
	const float3 p1 = birdSpline[birbSegment];
	const float3 p2 = birdSpline[(birbSegment + 1) % 18];
	const float3 p3 = birdSpline[(birbSegment + 2) % 18];
	const float3 pos = CatmullRom( p0, p1, p2, p3, birbTime ) + make_float3( 0, 7, 0 );
	static float3 lastBirdPos = make_float3( 0 );
	renderer->SetNodeTransform( birb, mat4::LookAt( lastBirdPos, pos ) * mat4::Scale( 0.01f ) );
	lastBirdPos = pos;
}

void UpdateClouds( const float frameTime )
{
	if (cloud1 == -1) return;
	c1pos += frameTime * 1.0f;
	c2pos += frameTime * 1.0f;
	c3pos += frameTime * 1.0f;
	if (c1pos > 50) c1pos -= 100;
	if (c2pos > 50) c2pos -= 100;
	if (c3pos > 50) c3pos -= 100;
	float scale1 = 1.0f, scale2 = 1.0f, scale3 = 1.0f;
	if (c1pos < -40) scale1 = (c1pos + 50) * 0.1f; else if (c1pos > 40) scale1 = (50 - c1pos) * 0.1f;
	if (c2pos < -40) scale2 = (c2pos + 50) * 0.1f; else if (c2pos > 40) scale2 = (50 - c2pos) * 0.1f;
	if (c3pos < -40) scale3 = (c3pos + 50) * 0.1f; else if (c3pos > 40) scale3 = (50 - c3pos) * 0.1f;
	renderer->SetNodeTransform( cloud1, mat4::Translate( c1pos, 20, 0 ) * mat4::RotateZ( c1pos * -0.002f ) * mat4::Scale( 3.0f * scale1 ) );
	renderer->SetNodeTransform( cloud2, mat4::Translate( c2pos, 20, 25 ) * mat4::RotateZ( c2pos * -0.002f ) * mat4::Scale( 3.0f * scale2 ) );
	renderer->SetNodeTransform( cloud3, mat4::Translate( c3pos, 20, -32 ) * mat4::RotateZ( c3pos * -0.002f ) * mat4::Scale( 3.0f * scale3 ) );
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
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7filter" );	// OPTIX7 core, with filtering (static scenes only for now)
#ifndef OPTIX5FALLBACK
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_Optix7" );			// OPTIX7 core, best for RTX devices
#else
	renderer = RenderAPI::CreateRenderAPI( "RenderCore_OptixPrime_B" );		// OPTIX PRIME, best for pre-RTX CUDA devices
	// renderer = RenderAPI::CreateRenderAPI( "RenderCore_PrimeAdaptive" );	// OPTIX PRIME, with adaptive sampling
#endif
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
		delete overlay;
		rawFrame1 = new Bitmap( scrwidth, scrheight );
		rawFrame2 = new Bitmap( scrwidth, scrheight );
		overlay = new Bitmap( scrwidth, scrheight );
		rawFrameSize = scrwidth * scrheight;
	}
	rawFrame1->width = rawFrame2->width = overlay->width = scrwidth;
	rawFrame1->height = rawFrame2->height = overlay->height = scrheight;
}

//  +-----------------------------------------------------------------------------+
//  |  DrawOverlay                                                                |
//  |  Draw the overlay while converging.                                   LH2'20|
//  +-----------------------------------------------------------------------------+
void DrawOverlay()
{
	// render text
	char t[1024];
	sprintf( t, "Rendering ground truth, %04i spp.", convergedSamples );
	smallText->Render( t, 15, scrheight - 85 );
	// render overlay
	CheckRawFrameBuffers();
	overlay->Clear();
	// grid
	overlay->Box( 10, 10, scrwidth - 10, scrheight - 30, 0xaaffffff );
	overlay->Box( 11, 11, scrwidth - 11, scrheight - 29, 0xaaffffff );
	overlay->Box( 14, 14, scrwidth - 14, scrheight - 32, 0x88ffffff );
	overlay->VLine( 14 + (scrwidth - 28) / 3, 14, scrheight - 46, 0x88ffffff );
	overlay->VLine( 14 + (scrwidth - 28) * 2 / 3, 14, scrheight - 46, 0x88ffffff );
	overlay->HLine( 14, 14 + (scrheight - 46) / 3, scrwidth - 28, 0x88ffffff );
	overlay->HLine( 14, 14 + (scrheight - 46) * 2 / 3, scrwidth - 28, 0x88ffffff );
	overlay->Bar( 610, scrheight - 32, 720, scrheight - 28, 0 );
	// progress bar
	int range = (int)(sqr( sqr( 1 - 0.5f * delay ) ) * (float)((scrheight - 46) / 3));
	for (int y0 = 14 + (scrheight - 46) * 2 / 3, i = 0; i < range; i++, y0--) overlay->HLine( 14 + (scrwidth - 28) * 2 / 3 - 5, y0, 4, 0xaaffffff );
	// blinker
	if (delay < 0.25f)
	{
		if ((int)(delay * 16) & 1)
		{
			int x2 = 10 + (scrwidth - 28) / 3, x1 = x2 - scrwidth / 36;
			int y1 = 18 + (scrheight - 46) / 3, y2 = y1 + scrheight / 24;
			overlay->Bar( x1, y1, x2, y2, 0xbb6666ff );
		}
	}
	// text
	SystemStats systemStats = renderer->GetSystemStats();
	int x1 = 6 + (scrwidth - 28) / 3, y1 = (scrheight - 46) / 3 - 22;
	float3 camPos = renderer->GetCamera()->transform.GetTranslation();
	if (delay < 1.8f) { sprintf( t, "Scene update: %6.2fms", systemStats.sceneUpdateTime * 1000 ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 1.6f) { sprintf( t, "Primary ray time: %6.2fms", coreStats.traceTime0 * 1000 ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 1.5f) { sprintf( t, "Secondary ray time: %6.2fms", coreStats.traceTime1 * 1000 ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 1.2f) { sprintf( t, "Shadow ray time: %6.2fms", coreStats.shadowTraceTime * 1000 ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 1.0f) { sprintf( t, "Overhead: %6.2fms", coreStats.frameOverhead * 1000 ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.9f) { sprintf( t, "# primary: %6ik (%6.1fM/s)", coreStats.primaryRayCount / 1000, coreStats.primaryRayCount / (max( 1.0f, coreStats.traceTime0 * 1000000 )) ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.8f) { sprintf( t, "# secondary: %6ik (%6.1fM/s)", coreStats.bounce1RayCount / 1000, coreStats.bounce1RayCount / (max( 1.0f, coreStats.traceTime1 * 1000000 )) ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.6f) { sprintf( t, "# deep rays: %6ik (%6.1fM/s)", coreStats.deepRayCount / 1000, coreStats.deepRayCount / (max( 1.0f, coreStats.traceTimeX * 1000000 )) ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.4f) { sprintf( t, "# shadw rays: %6ik (%6.1fM/s)", coreStats.totalShadowRays / 1000, coreStats.totalShadowRays / (max( 1.0f, coreStats.shadowTraceTime * 1000000 )) ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.3f) { sprintf( t, "camera position: %5.2f, %5.2f, %5.2f", camPos.x, camPos.y, camPos.z ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	if (delay < 0.2f) { sprintf( t, "fdist: %5.2f", renderer->GetCamera()->focalDistance ); tinyText->RenderR( t, x1, y1 ); y1 += 13; }
	// present
	overlayTarget->CopyFrom( overlay );
	overlayShader->Bind();
	overlayShader->SetInputTexture( 0, "color", overlayTarget );
	overlayShader->SetInputMatrix( "view", mat4::Identity() );
	DrawQuad();
	overlayShader->Unbind();
}

//  +-----------------------------------------------------------------------------+
//  |  DemoState                                                                  |
//  |  Produce a frame for the demo.                                        LH2'20|
//  +-----------------------------------------------------------------------------+
void DemoState()
{
	// start renderering in a separate thread
	renderer->Setting( "noiseShift", camTime );
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
	if (animsPlaying) if (!animPaused)
	{
		UpdateBird( frameTime );
		UpdateClouds( frameTime );
	}
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
	coreStats = renderer->GetCoreStats();
	sprintf( t, "Lighthouse 2 benchmark ALPHA - frame %05i: %5.2fms         fps,  peak: ", frameIdx++, coreStats.renderTime * 1000 );
	textRenderer->Render( t, 0, scrheight - 54 );
	float currentFps = 1.0f / max( 0.001f, coreStats.renderTime );
	if (frameIdx > 5) peakFps = max( peakFps, currentFps );
	smoothedFps = 0.9f * smoothedFps + 0.1f * currentFps;
	sprintf( t, "%5.1f", smoothedFps );
	digiText->Render( t, 613, scrheight - 54 );
	sprintf( t, "%5.1f", peakFps );
	smallDigiText->Render( t, 850, scrheight - 54 );
	if (!camPlaying)
	{
		if (firstConvergingFrame)
		{
			// grab the output of the renderer so we can compare it to ground truth later.
			CheckRawFrameBuffers();
			renderTarget->CopyTo( rawFrame1 );
			firstConvergingFrame = false;
		}
		DrawOverlay();
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
			uint* a = rawFrame1->pixels, * b = rawFrame2->pixels;
			double errorSum = 0;
			for (uint y = 0; y < scrheight; y++) for (uint x = 0; x < scrwidth; x++, a++, b++)
			{
				double dr = (1.0f / 256.0f) * abs( (int)((*a >> 16) & 255) - (int)((*b >> 16) & 255) );
				double dg = (1.0f / 256.0f) * abs( (int)((*a >> 8) & 255) - (int)((*b >> 8) & 255) );
				double db = (1.0f / 256.0f) * abs( (int)(*a & 255) - (int)(*b & 255) );
				errorSum += dr * dr + dg * dg + db * db;
			}
			lastRMSE = RMSE[(camTrack + 5) % 6] = (float)sqrt( errorSum / (scrwidth * scrheight) );
			printf( "RMSE #%i: %6.4f\n", camTrack, lastRMSE );
			reportTimer = 3.0f;
		}
		else if (frameIdx > 5)
		{
			// gather data for final report
			fpsSum += currentFps;
			frameCount++;
			float primaryRayTime = coreStats.traceTime0;
			float secRayTime = coreStats.traceTime1;
			float shadowTime = coreStats.shadowTraceTime;
			float primaryCount = coreStats.primaryRayCount;
			float secRayCount = coreStats.bounce1RayCount;
			float deepRayCount = coreStats.deepRayCount;
			float shadowRayCount = coreStats.totalShadowRays;
			rpsSum += primaryCount / primaryRayTime;
			shdSum += shadowRayCount / shadowTime;
			rpsPeak = max( rpsPeak, primaryCount / primaryRayTime );
			shdPeak = max( shdPeak, shadowRayCount / shadowTime );
		}
		if (reportTimer > 0)
		{
			bool show = true;
			if (reportTimer < 0.5f) if ((int)(reportTimer * 18.0f) & 1) show = false;
			sprintf( t, "Last RMSE: %5.3f", lastRMSE );
			if (show) smallText->Render( t, 0, scrheight - 84 );
			reportTimer -= frameTime;
		}
	}
	// 'fade in' using the overlay
	if (frameIdx < 32)
	{
		CheckRawFrameBuffers();
		for (uint i = 0; i < scrwidth * scrheight; i++) overlay->pixels[i] = (31 - frameIdx) << 27;
		overlayTarget->CopyFrom( overlay );
		overlayShader->Bind();
		overlayShader->SetInputTexture( 0, "color", overlayTarget );
		overlayShader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		overlayShader->Unbind();
	}
	// fadeout at end of sequence
	if (fadeOut)
	{
		CheckRawFrameBuffers();
		fadeTimer -= frameTime;
		if (fadeTimer < 0) state = 2, frameIdx = 0;
		int f = 255 - ((int)(max( 0.0f, fadeTimer ) * 63));
		for (uint i = 0; i < scrwidth * scrheight; i++) overlay->pixels[i] = (f << 24) + 0xffffff;
		overlayTarget->CopyFrom( overlay );
		overlayShader->Bind();
		overlayShader->SetInputTexture( 0, "color", overlayTarget );
		overlayShader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		overlayShader->Unbind();
	}
#endif
}

//  +-----------------------------------------------------------------------------+
//  |  TitleState                                                                 |
//  |  Application entry point.                                             LH2'20|
//  +-----------------------------------------------------------------------------+
void TitleState( bool results )
{
	if (frameIdx == 0)
	{
		if (results)
		{
			// load results backdrop
			menuScreen->Load( "data/results.png", GL_LINEAR );
		}
		else
		{
			// load menu backdrop
			menuScreen->Load( "data/menu.png", GL_LINEAR );
		}
	}
	if (results) if (frameIdx++ < 60)
	{
		shader->Bind();
		for (uint i = 0; i < scrwidth * scrheight; i++) overlay->pixels[i] = 0xfffffff;
		overlayTarget->CopyFrom( overlay );
		shader->SetInputTexture( 0, "color", overlayTarget );
		shader->SetInputMatrix( "view", mat4::Identity() );
		DrawQuad();
		shader->Unbind();
		Sleep( 50 );
		return;
	}
	shader->Bind();
	shader->SetInputTexture( 0, "color", menuScreen );
	shader->SetFloat( "contrast", 0 );
	shader->SetFloat( "brightness", 0 );
	shader->SetFloat( "gamma", 1.0f );
	shader->SetInt( "method", 0 );
	shader->SetInputMatrix( "view", mat4::Identity() );
	DrawQuad();
	shader->Unbind();
	// state for FreeType2 text rendering
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	// construct title screen
	if (GetAsyncKeyState( VK_RETURN ))
	{
		if (results) exit( 0 );
		state = 1;
		soundEngine->play2D( "data/4088-movement-proposition-by-kevin-macleod.mp3" );
		frameTimer.reset();
	}
	char t[1024];
	coreStats = renderer->GetCoreStats();
	if (results)
	{
		textRenderer->Render( "RESULTS", 5, 50 );
		textRenderer->Render( "RESULTS", 6, 50 );
	}
	else
	{
		textRenderer->Render( "Lighthouse 2 Path Tracing Benchmark", 5, 50 );
		textRenderer->Render( "Lighthouse 2 Path Tracing Benchmark", 6, 50 );
	}
	int x = (int)(scrwidth * 0.168f), y = (int)(scrheight * 0.139f), d = (int)(scrheight * 0.03472f);
	sprintf( t, "Device: %s (#SM: %i)", coreStats.deviceName, coreStats.SMcount );
	smallText->Render( t, x, y );
	sprintf( t, "Compute capability: %i.%i", coreStats.ccMajor, coreStats.ccMinor );
	smallText->Render( t, x, y + d );
	sprintf( t, "Device RAM: %iGB", coreStats.VRAM >> 10 );
	smallText->Render( t, x, y + 2 * d );
	if (results)
	{
		// final performance report
		y = (int)(scrheight * 0.3f), x = scrwidth * 0.1f;
		smallText->Render( "REPORT", x, y );
		smallText->Render( "REPORT", x + 1, y );
		y = (int)(scrheight * 0.35f), x = scrwidth * 0.1f;
		smallText->Render( "PERFORMANCE", x, y );
		smallText->Render( "PERFORMANCE", x + 1, y ); y += 25;
		char t[1024];
		sprintf( t, "#frames recorded:    %5i", frameCount );
		smallText->Render( t, x, y ); y += 20;
		sprintf( t, "samples per pixel:   %5i", scrspp );
		smallText->Render( t, x, y ); y += 20;
		sprintf( t, "screen size:         %i x %i", scrwidth, scrheight );
		smallText->Render( t, x, y ); y += 25;
		sprintf( t, "average fps:         %05.2f", fpsSum / (float)frameCount );
		smallText->Render( t, x, y ); y += 20;
		sprintf( t, "peak fps:            %05.2f", peakFps );
		smallText->Render( t, x, y ); y += 30;
		smallText->Render( "primary rays", x, y ); y += 25;
		sprintf( t, "average rps:       %6.2fM", (rpsSum / 1000000) / (float)frameCount );
		smallText->Render( t, x, y ); y += 20;
		sprintf( t, "peak rps:          %6.2fM", rpsPeak / 1000000 );
		smallText->Render( t, x, y ); y += 30;
		smallText->Render( "shadow rays", x, y ); y += 25;
		sprintf( t, "average rps:       %6.2fM", (shdSum / 1000000) / (float)frameCount );
		smallText->Render( t, x, y ); y += 20;
		int lasty = y;
		sprintf( t, "peak rps:          %6.2fM", shdPeak / 1000000 );
		smallText->Render( t, x, y ); y += 20;
		// quality report
		y = (int)(scrheight * 0.35f), x = scrwidth * 0.4f;
		smallText->Render( "IMAGE QUALITY", x, y );
		smallText->Render( "IMAGE QUALITY", x + 1, y ); y += 25;
		float rmseSum = 0;
		for (int i = 0; i <= 5; i++)
		{
			sprintf( t, "Checkpoint #%i: %5.3f", i, RMSE[i] );
			smallText->Render( t, x, y ); y += 20;
			rmseSum += RMSE[i];
		}
		sprintf( t, "Average RMSE: %5.3f", rmseSum / 6 );
		smallText->Render( t, x, y + 5 ); y += 35;
	#ifdef OPTIX5FALLBACK
		smallText->Render( "Rendercore: RenderCore_OptixPrime_B.dll", x, y );
	#else
		smallText->Render( "Rendercore: RenderCore_Optix7.dll", x, y );
	#endif
		y = lasty; smallText->Render( "Results have been saved to results.txt.", x, y );
		static bool saved = false;
		if (!saved)
		{
			saved = true;
			FILE* f = (redirected == 0) ? fopen( "../../results.txt", "w" ) : fopen( "results.txt", "w" );
			fprintf( f, "Device: %s (#SM: %i)\n", coreStats.deviceName, coreStats.SMcount );
			fprintf( f, "Compute capability: %i.%i\n", coreStats.ccMajor, coreStats.ccMinor );
			fprintf( f, "Device RAM: %iGB\n", coreStats.VRAM >> 10 );
			fprintf( f, "Device RAM: %iGB\n", coreStats.VRAM >> 10 );
		#ifdef OPTIX5FALLBACK
			fprintf( f, "Rendercore: RenderCore_OptixPrime_B.dll\n--------------------------------\n" );
		#else
			fprintf( f, "Rendercore: RenderCore_Optix7.dll\n--------------------------------\n" );
		#endif
			fprintf( f, "samples per pixel:   %5i\n", scrspp );
			fprintf( f, "screen size:         %i x %i\n", scrwidth, scrheight );
			fprintf( f, "average fps:         %05.2f\n", fpsSum / (float)frameCount );
			fprintf( f, "peak fps:            %05.2f\n", peakFps );
			fprintf( f, "avg rps (primary): %6.2fM\n", (rpsSum / 1000000) / (float)frameCount );
			fprintf( f, "peak:              %6.2fM\n", rpsPeak / 1000000 );
			fprintf( f, "avg rps (shadow):  %6.2fM\n", (shdSum / 1000000) / (float)frameCount );
			fprintf( f, "peak:              %6.2fM\n", shdPeak / 1000000 );
			float rmseSum = 0;
			for (int i = 0; i <= 5; i++)
			{
				fprintf( f, "Checkpoint #%i: %5.3f\n", i, RMSE[i] );
				rmseSum += RMSE[i];
			}
			fprintf( f, "Average RMSE: %5.3f\n--------------------------------\n", rmseSum / 6 );
			fprintf( f, "#frames recorded:    %5i\n", frameCount );
			fprintf( f, "Demo version: March 31, 2020\n" );
			fclose( f );
		}
	}
	else
	{
		y = (int)(scrheight * 0.4167f);
		smallText->Render( "CREDITS", 15, y ); y += 20;
		tinyText->Render( """Medieval Fantasy Book"" scene by Robert Jordan", 15, y ); y += 15;
		tinyText->Render( "Music: ""Movement Proposition"", by Kevin MacLeod", 15, y ); y += 15;
		tinyText->Render( "Low poly bird animation by josluat91", 15, y ); y += 15;
		tinyText->Render( "Additional models by Alok and Benjamin Todd", 15, y ); y += 15;
		tinyText->Render( "Font: Source Code Pro by Google", 15, y ); y += 15;
		tinyText->Render( "LH2 uses GLFW, Glad, TinyGLTF, TinyXML2, zlib, imgui, FreeImage and FreeType2.", 15, y ); y += 20;
		tinyText->Render( "Lighthouse 2 engine and demo code by Jacco Bikker.", 15, y );
	}
	if (results)
	{
		if (frameTimer.elapsed() < 0.3f) textRenderer->Render( "Press [ENTER] to exit.", (int)(scrwidth * 0.355f), scrheight - 90 );
		else if (frameTimer.elapsed() > 0.6f) frameTimer.reset();
	}
	else
	{
		if (frameTimer.elapsed() < 0.3f) textRenderer->Render( "Press [ENTER] to start.", (int)(scrwidth * 0.355f), scrheight - 90 );
		else if (frameTimer.elapsed() > 0.6f) frameTimer.reset();
	}
}

//  +-----------------------------------------------------------------------------+
//  |  main                                                                       |
//  |  Application entry point.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
int main( int argc, char* argv[] )
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
	redirected = _chdir( "./apps/benchmarkapp" );

	// prepare for rendering
	Initialize();

	// prepare FreeType2
	textRenderer = new GLTextRenderer( 20, "data/sourcecodepro-regular.ttf" );
	smallText = new GLTextRenderer( 16, "data/sourcecodepro-regular.ttf" );
	tinyText = new GLTextRenderer( 11, "data/sourcecodepro-regular.ttf" );
	digiText = new GLTextRenderer( 48, "data/ds-digib.ttf" );
	smallDigiText = new GLTextRenderer( 32, "data/ds-digib.ttf" );

#ifdef DEMOMODE
	// music
	soundEngine = createIrrKlangDevice();
#endif

	// application main loop
	frameTimer.reset();
	while (!glfwWindowShouldClose( window ))
	{
		// poll glfw events
		glfwPollEvents();
		if (!running) break;
	#ifdef DEMOMODE
		if (state == 0) TitleState( false );
		if (state == 1) DemoState();
		if (state == 2) TitleState( true );
	#else
		DemoState();
	#endif
		// restore state for Lighthouse 2 rendering
		glDisable( GL_BLEND );
		glfwSwapBuffers( window );
		firstFrame = false;
	}
	Shutdown();
	return 0;
}

// EOF