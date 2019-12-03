/* fps_printer.cpp - Copyright 2019 Utrecht University

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

#include "platform.h" // Shader
#include "camera.h"

namespace MAIN_UI {

// FPS printer globals
GLTexture* digit[10], *hud;
Shader* plainShader = 0, *shadowShader = 0;
float smoothed = 1.0f, smoothFactor = 0.1f;

//  +-----------------------------------------------------------------------------+
//  |  InitFPSPrinter                                                             |
//  |  Initializes the FPS counter.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void InitFPSPrinter()
{
	// load digits
	for (int i = 0; i < 10; i++)
	{
		char t[128] = "data//system//digit0.png";
		t[strlen(t) - 5] += i;
		digit[i] = new GLTexture(t, GL_LINEAR);
	}
	// load HUD
	hud = new GLTexture("data//system//hud.png", GL_LINEAR);
	// load shaders
	plainShader = new Shader("shaders/plain.vert", "shaders/plain.frag");
	shadowShader = new Shader("shaders/plain.vert", "shaders/plain_shadow.frag");
}

//  +-----------------------------------------------------------------------------+
//  |  DrawDigit                                                                  |
//  |  Draws a digit onto the screen.                                       LH2'19|
//  +-----------------------------------------------------------------------------+
void DrawDigit(int d, float x, float y, float scale = 1.0f)
{
	plainShader->SetInputTexture(0, "color", digit[d]);
	mat4 T = mat4::Scale(make_float3(0.06f * scale, 0.1f * scale, 1));
	T.cell[12] = x, T.cell[13] = y;
	plainShader->SetInputMatrix("view", T);
	DrawQuad();
}

//  +-----------------------------------------------------------------------------+
//  |  PrintFPS                                                                   |
//  |  Prints the FPS counter onto the screen.                              LH2'19|
//  +-----------------------------------------------------------------------------+
void PrintFPS(float deltaTime)
{
	float fps = (int)(1.0f / deltaTime);
	smoothed = (1 - smoothFactor) * smoothed + smoothFactor * fps;
	if (smoothFactor > 0.05f) smoothFactor -= 0.05f;
	int ifps = smoothed * 10, d1 = (ifps / 1000) % 10, d2 = (ifps / 100) % 10, d3 = (ifps / 10) % 10, d4 = ifps % 10;
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	shadowShader->Bind();
	float xpos = -0.91f, ypos = -0.81f;
	DrawDigit(d1, xpos, ypos); xpos += 0.12f;
	DrawDigit(d2, xpos, ypos); xpos += 0.12f;
	DrawDigit(d3, xpos, ypos); xpos += 0.14f;
	DrawDigit(d4, xpos, ypos - 0.03f, 0.7f);
	shadowShader->Unbind();
	plainShader->Bind();
	xpos = -0.92f, ypos = -0.8f;
	DrawDigit(d1, xpos, ypos); xpos += 0.12f;
	DrawDigit(d2, xpos, ypos); xpos += 0.12f;
	DrawDigit(d3, xpos, ypos); xpos += 0.14f;
	DrawDigit(d4, xpos, ypos - 0.03f, 0.7f);
	plainShader->Unbind();
	glDisable(GL_BLEND);
}

} // namespace AI_UI

// EOF