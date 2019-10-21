/* main_ui.h - Copyright 2019 Utrecht University

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

#include <AntTweakBar.h>

// AntTweakBar data
float mraysincl = 0, mraysexcl = 0;
TwBar* bar = 0;
HostMaterial currentMaterial; // will contain a copy of the material we're editing
bool currentMaterialConductor, currentMaterialDielectric;
int currentMaterialID = -1;
static CoreStats coreStats;

//  +-----------------------------------------------------------------------------+
//  |  InitAntTweakBar                                                            |
//  |  Prepares a basic user interface.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
void RefreshUI();
void InitAntTweakBar()
{
	TwInit( TW_OPENGL_CORE, NULL );
	bar = TwNewBar( "settings" );
	RefreshUI();
}

//  +-----------------------------------------------------------------------------+
//  |  RefreshUI                                                                  |
//  |  AntTweakBar.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void RefreshUI()
{
	TwDefine( " settings size='200 400' color='50 120 50' alpha=220" );
	TwDefine( " GLOBAL help='LightHouse2 data' " );
	TwDefine( " settings resizable=true movable=true iconifiable=true refresh=0.05 " );
	int opened = 1, closed = 0;
	// create collapsed statistics block
	TwAddVarRO( bar, "rays", TW_TYPE_UINT32, &coreStats.totalRays, " group='statistics'" );
	TwAddVarRO( bar, "build time", TW_TYPE_FLOAT, &coreStats.bvhBuildTime, " group='statistics'" );
	TwAddVarRO( bar, "render time", TW_TYPE_FLOAT, &coreStats.renderTime, " group='statistics'" );
	TwAddVarRO( bar, "shade time", TW_TYPE_FLOAT, &coreStats.shadeTime, " group='statistics'" );
	TwAddVarRO( bar, "mrays inc", TW_TYPE_FLOAT, &mraysincl, " group='statistics'" );
	TwAddVarRO( bar, "mrays ex", TW_TYPE_FLOAT, &mraysexcl, " group='statistics'" );
	TwAddSeparator( bar, "separator0", "group='statistics'" );
	TwAddVarRO( bar, "probed tri", TW_TYPE_INT32, &coreStats.probedTriid, " group='statistics'" );
	TwSetParam( bar, "statistics", "opened", TW_PARAM_INT32, 1, &closed );
	// create collapsed material block
	TwAddVarRO( bar, "name", TW_TYPE_STDSTRING, &currentMaterial.name, "group='material'" );
	TwAddVarRO( bar, "origin", TW_TYPE_STDSTRING, &currentMaterial.origin, "group='material'" );
	TwAddVarRO( bar, "ID", TW_TYPE_INT32, &currentMaterial.ID, "group='material'" );
	TwAddVarRO( bar, "flags", TW_TYPE_UINT32, &currentMaterial.flags, "group='material'" );
	TwAddVarRW( bar, "color", TW_TYPE_COLOR3F, &currentMaterial.color, "group='material'" );
	TwAddVarRW( bar, "transmiss", TW_TYPE_COLOR3F, &currentMaterial.absorption, "group='material'" );
	TwAddVarRW( bar, "metallic", TW_TYPE_FLOAT, &currentMaterial.metallic, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "subsurface", TW_TYPE_FLOAT, &currentMaterial.subsurface, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "specular", TW_TYPE_FLOAT, &currentMaterial.specular, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "roughness", TW_TYPE_FLOAT, &currentMaterial.roughness, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "specularTint", TW_TYPE_FLOAT, &currentMaterial.specularTint, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "anisotropic", TW_TYPE_FLOAT, &currentMaterial.anisotropic, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "sheen", TW_TYPE_FLOAT, &currentMaterial.sheen, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "sheenTint", TW_TYPE_FLOAT, &currentMaterial.sheenTint, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "clearcoat", TW_TYPE_FLOAT, &currentMaterial.clearcoat, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "ccoatGloss", TW_TYPE_FLOAT, &currentMaterial.clearcoatGloss, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "transmission", TW_TYPE_FLOAT, &currentMaterial.transmission, "group='material' min=0 max=1 step=0.01" );
	TwAddVarRW( bar, "eta", TW_TYPE_FLOAT, &currentMaterial.eta, "group='material' min=0 max=2 step=0.01" );
	TwAddSeparator( bar, "separator2", "group='material'" );
	TwStructMember float2Members[] = { { "x", TW_TYPE_FLOAT, offsetof( float2, x ), "" },
		{ "y", TW_TYPE_FLOAT, offsetof( float2, y ), "" } };
	TwType float2Type = TwDefineStruct( "float2", float2Members, 2, sizeof( float2 ), NULL, NULL );
	TwStructMember float3Members[] = { { "x", TW_TYPE_FLOAT, offsetof( float3, x ), "" },
		{ "y", TW_TYPE_FLOAT, offsetof( float3, y ), "" },
		{ "z", TW_TYPE_FLOAT, offsetof( float3, z ), "" } };
	TwType float3Type = TwDefineStruct( "float3", float3Members, 3, sizeof( float3 ), NULL, NULL );
	TwStructMember mapMembers[] =
	{ { "ID", TW_TYPE_INT32, offsetof( HostMaterial::MapProps, textureID ), "" },
		{ "scale", float2Type, offsetof( HostMaterial::MapProps, uvscale ), "" },
		{ "offset", float2Type, offsetof( HostMaterial::MapProps, uvoffset ), "" } };
	TwType mapType = TwDefineStruct( "Texture0", mapMembers, 3, sizeof( HostMaterial::MapProps ), NULL, NULL );
	TwAddVarRW( bar, "difftex0", mapType, &currentMaterial.map[TEXTURE0], " group='material' " );
	TwAddVarRW( bar, "difftex1", mapType, &currentMaterial.map[TEXTURE1], " group='material' " );
	TwAddVarRW( bar, "difftex2", mapType, &currentMaterial.map[TEXTURE2], " group='material' " );
	TwAddVarRW( bar, "nrmlmap0", mapType, &currentMaterial.map[NORMALMAP0], " group='material' " );
	TwAddVarRW( bar, "nrmlmap1", mapType, &currentMaterial.map[NORMALMAP1], " group='material' " );
	TwAddVarRW( bar, "nrmlmap2", mapType, &currentMaterial.map[NORMALMAP2], " group='material' " );
	TwAddSeparator( bar, "separator3", "group='material'" );
	TwSetParam( bar, "material", "opened", TW_PARAM_INT32, 1, &closed );
	// create collapsed camera block
	TwAddVarRO( bar, "position", float3Type, &renderer->GetCamera()->position, "group='camera'" );
	TwAddVarRO( bar, "direction", float3Type, &renderer->GetCamera()->direction, "group='camera'" );
	TwAddVarRW( bar, "FOV", TW_TYPE_FLOAT, &renderer->GetCamera()->FOV, "group='camera' min=10 max=99 step=1" );
	TwAddVarRW( bar, "focaldist", TW_TYPE_FLOAT, &renderer->GetCamera()->focalDistance, "group='camera' min=0.1 max=100 step=0.01" );
	TwAddVarRW( bar, "aperture", TW_TYPE_FLOAT, &renderer->GetCamera()->aperture, "group='camera' min=0 max=1 step=0.001" );
	TwAddVarRW( bar, "brightness", TW_TYPE_FLOAT, &renderer->GetCamera()->brightness, "group='camera' min=-1 max=1 step=0.01" );
	TwAddVarRW( bar, "contrast", TW_TYPE_FLOAT, &renderer->GetCamera()->contrast, "group='camera' min=-1 max=1 step=0.01" );
	TwAddVarRW( bar, "clampValue", TW_TYPE_FLOAT, &renderer->GetCamera()->clampValue, "group='camera' min=1 max=100 step=1" );
	TwSetParam( bar, "camera", "opened", TW_PARAM_INT32, 1, &closed );
	// create the renderer block
	TwAddVarRW( bar, "epsilon", TW_TYPE_FLOAT, &renderer->GetSettings()->geometryEpsilon, "group='renderer'" );
	TwAddVarRW( bar, "maxDirect", TW_TYPE_FLOAT, &renderer->GetSettings()->filterDirectClamp, "group='renderer' min=1 max=50 step=0.5" );
	TwAddVarRW( bar, "maxIndirect", TW_TYPE_FLOAT, &renderer->GetSettings()->filterIndirectClamp, "group='renderer' min=1 max=50 step=0.5" );
}

GLTexture* digit[10], *hud;
Shader* plainShader = 0, *shadowShader = 0;
float smoothed = 1.0f, smoothFactor = 0.1f;

void InitFPSPrinter()
{
	// load digits
	for (int i = 0; i < 10; i++)
	{
		char t[128] = "data/system/digit0.png";
		t[strlen( t ) - 5] += i;
		digit[i] = new GLTexture( t, GL_LINEAR );
	}
	// load HUD
	hud = new GLTexture( "data/system/hud.png", GL_LINEAR );
	// load shaders
	plainShader = new Shader( "shaders/plain.vert", "shaders/plain.frag" );
	shadowShader = new Shader( "shaders/plain.vert", "shaders/plain_shadow.frag" );
}

void DrawDigit( int d, float x, float y, float scale = 1.0f )
{
	plainShader->SetInputTexture( 0, "color", digit[d] );
	mat4 T = mat4::Scale( make_float3( 0.06f * scale, 0.1f * scale, 1 ) );
	T.cell[12] = x, T.cell[13] = y;
	plainShader->SetInputMatrix( "view", T );
	DrawQuad();
}

void DrawHUD( float x, float y )
{
	plainShader->SetInputTexture( 0, "color", hud );
	float scale = 4.5f;
	mat4 T = mat4::Scale( scale * make_float3( 0.06f, 0.1f, 1 ) );
	T.cell[12] = x, T.cell[13] = y;
	plainShader->SetInputMatrix( "view", T );
	DrawQuad();
}

void PrintFPS( float deltaTime )
{
	float fps = (int)(1.0f / deltaTime);
	smoothed = (1 - smoothFactor) * smoothed + smoothFactor * fps;
	if (smoothFactor > 0.05f) smoothFactor -= 0.05f;
	int ifps = smoothed * 10, d1 = (ifps / 1000) % 10, d2 = (ifps / 100) % 10, d3 = (ifps / 10) % 10, d4 = ifps % 10;
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	shadowShader->Bind();
	float xpos = -0.91f, ypos = -0.81f;
	DrawDigit( d1, xpos, ypos ); xpos += 0.12f;
	DrawDigit( d2, xpos, ypos ); xpos += 0.12f;
	DrawDigit( d3, xpos, ypos ); xpos += 0.14f;
	DrawDigit( d4, xpos, ypos - 0.03f, 0.7f );
	shadowShader->Unbind();
	plainShader->Bind();
	xpos = -0.92f, ypos = -0.8f;
	DrawDigit( d1, xpos, ypos ); xpos += 0.12f;
	DrawDigit( d2, xpos, ypos ); xpos += 0.12f;
	DrawDigit( d3, xpos, ypos ); xpos += 0.14f;
	DrawDigit( d4, xpos, ypos - 0.03f, 0.7f );
	plainShader->Unbind();
	glDisable( GL_BLEND );
}

// EOF