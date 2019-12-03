/* main_tools.h - Copyright 2019 Utrecht University

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

GLFWwindow* window = 0;

//  +-----------------------------------------------------------------------------+
//  |  ...Callback                                                                |
//  |  Various GLFW callbacks, mostly just forwarded to AntTweakBar.        LH2'19|
//  +-----------------------------------------------------------------------------+
void ReshapeWindowCallback( GLFWwindow* window, int w, int h )
{
	// don't resize if nothing changed or the window was minimized
	if ((scrwidth == w && scrheight == h) || w == 0 || h == 0) return;
	scrwidth = w, scrheight = h;
	delete renderTarget;
	renderTarget = new GLTexture( scrwidth, scrheight, GLTexture::FLOAT );
	glViewport( 0, 0, scrwidth, scrheight );
	renderer->SetTarget( renderTarget, 1 );
	// forward to AntTweakBar
	TwWindowSize( scrwidth, scrheight );
}
void KeyEventCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
	if (key == GLFW_KEY_ESCAPE) running = false;
	TwEventKeyGLFW( key, action );
}
void CharEventCallback( GLFWwindow* window, uint code ) { TwEventCharGLFW( code, 1 ); }
void WindowFocusCallback( GLFWwindow* window, int focused ) { hasFocus = (focused == GL_TRUE); }
void MouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
	TwEventMouseButtonGLFW( button, action );
	if (button == GLFW_MOUSE_BUTTON_1)
	{
		if (action == GLFW_PRESS) // first press
			leftClicked = true;
		else if (action == GLFW_RELEASE)
			leftClicked = false;
		// else if(action == GLFW_REPEAT) // held press
	}
	else if (button == GLFW_MOUSE_BUTTON_2)
	{
		if (action == GLFW_PRESS) // first press
			rightClicked = true;
		else if (action == GLFW_RELEASE)
			rightClicked = false;
		// else if(action == GLFW_REPEAT) // held press
	}
}
void MousePosCallback( GLFWwindow* window, double x, double y )
{
	TwMouseMotion( (int)x, (int)y );
	// set pixel probe pos for triangle picking
	renderer->SetProbePos( make_int2( (int)x, (int)y ) );
	probeCoords = make_int2((int)x, (int)y);
}
void MouseWheelCallback( GLFWwindow* window, double x, double y )
{
	static double wheelPos = 0; // GLFW is relative, AntTweakBar expects absolute
	wheelPos += y;
	TwMouseWheel( (int)wheelPos );
}

//  +-----------------------------------------------------------------------------+
//  |  InitGLFW                                                                   |
//  |  Opens a GL window using GLFW.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
void OpenConsole();
void InitGLFW()
{
	// open a window
	if (!glfwInit()) exit( EXIT_FAILURE );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
	glfwWindowHint( GLFW_RESIZABLE, GL_TRUE );
	if (!(window = glfwCreateWindow( SCRWIDTH, SCRHEIGHT, "LightHouse v2.0", nullptr, nullptr ))) exit( EXIT_FAILURE );
	glfwMakeContextCurrent( window );
	// register callbacks
	glfwSetFramebufferSizeCallback( window, ReshapeWindowCallback );
	glfwSetKeyCallback( window, KeyEventCallback );
	glfwSetWindowFocusCallback( window, WindowFocusCallback );
	glfwSetMouseButtonCallback( window, MouseButtonCallback );
	glfwSetCursorPosCallback( window, MousePosCallback );
	glfwSetCharCallback( window, CharEventCallback );
	glfwSetScrollCallback( window, MouseWheelCallback );
	// initialize GLAD
	if (!gladLoadGLLoader( (GLADloadproc)glfwGetProcAddress )) exit( EXIT_FAILURE );
	// prepare OpenGL state
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );
	glDisable( GL_BLEND );
	// logo
	GLTexture* logo = new GLTexture( "data//system//logo.png", GL_LINEAR );
	shader = new Shader( "shaders/vignette.vert", "shaders/vignette.frag" );
	shader->Bind();
	shader->SetInputTexture( 0, "color", logo );
	float hscale = ((float)SCRHEIGHT / SCRWIDTH) * ((float)logo->width / logo->height);
	shader->SetInputMatrix( "view", mat4::Scale( make_float3( 0.1f * hscale, 0.1f, 1 ) ) );
	DrawQuad();
	shader->Unbind();
	glfwSwapBuffers( window );
	delete logo;
	// we want a console window for text output
	OpenConsole();
}

//  +-----------------------------------------------------------------------------+
//  |  OpenConsole                                                                |
//  |  Create the console window for text output.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void OpenConsole()
{
	CONSOLE_SCREEN_BUFFER_INFO coninfo;
	AllocConsole();
	GetConsoleScreenBufferInfo( GetStdHandle( STD_OUTPUT_HANDLE ), &coninfo );
	coninfo.dwSize.X = 1280;
	coninfo.dwSize.Y = 800;
	SetConsoleScreenBufferSize( GetStdHandle( STD_OUTPUT_HANDLE ), coninfo.dwSize );
	FILE* file = nullptr;
	freopen_s( &file, "CON", "w", stdout );
	freopen_s( &file, "CON", "w", stderr );
	SetWindowPos( GetConsoleWindow(), HWND_TOP, 0, 0, 1280, 800, 0 );
	glfwShowWindow( window );
}

// EOF