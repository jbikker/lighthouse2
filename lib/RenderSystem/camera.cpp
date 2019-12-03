/* camera.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

//  +-----------------------------------------------------------------------------+
//  |  Camera::Camera                                                             |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::Camera( const char* xmlFile )
{
	Deserialize( xmlFile );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::~Camera                                                            |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
Camera::~Camera()
{
	Serialize( xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::CalculateMatrix                                                    |
//  |  Helper function; constructs camera matrix.                           LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::CalculateMatrix( float3& x, float3& y, float3& z ) const
{
	if (fabs( direction.y ) > 0.99f)
	{
		// camera is looking straight down; use (1,0,0) as 'up' vector
		y = make_float3( 1, 0, 0 );
		z = direction;
		x = normalize( cross( z, y ) );
		y = cross( x, z );
	}
	else
	{
		y = make_float3( 0, 1, 0 );
		z = direction; // assumed to be normalized at all times
		x = normalize( cross( z, y ) );
		y = cross( x, z );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::LookAt                                                             |
//  |  Position and aim the camera.                                         LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::LookAt( const float3 O, const float3 T )
{
	position = O;
	direction = normalize( T - O );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateRelative                                                  |
//  |  Move the camera with respect to the current orientation.             LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateRelative( float3 T )
{
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	float3 delta = T.x * right + T.y * up + T.z * forward;
	position += delta;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::TranslateTarget                                                    |
//  |  Move the camera target with respect to the current orientation.      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::TranslateTarget( float3 T )
{
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	direction = normalize( direction + T.x * right + T.y * up + T.z * forward );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::GetView                                                            |
//  |  Create a ViewPyramid for rendering in the RenderCore layer.          LH2'19|
//  +-----------------------------------------------------------------------------+
ViewPyramid Camera::GetView() const
{
	ViewPyramid view;
	float3 right, up, forward;
	CalculateMatrix( right, up, forward );
	view.pos = position;
	view.spreadAngle = (FOV * PI / 180) / (float)pixelCount.y;
	const float screenSize = tanf( FOV / 2 / (180 / PI) );
	const float3 C = view.pos + focalDistance * forward;
	view.p1 = C - screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p2 = C + screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p3 = C - screenSize * right * focalDistance * aspectRatio - screenSize * focalDistance * up;
	view.aperture = aperture;
	view.focalDistance = focalDistance;
	view.distortion = distortion;
	// BDPT
	float3 unitP1 = C - screenSize * right * aspectRatio + screenSize * up;
	float3 unitP2 = C + screenSize * right * aspectRatio + screenSize * up;
	float3 unitP3 = C - screenSize * right * aspectRatio - screenSize * up;
	view.imagePlane = length( unitP1 - unitP2 ) * length( unitP1 - unitP3 );
	return view;
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::WorldToScreenPos                                                   |
//  |  Converts an array of world positions to screen positions.                  |
//  |  Helper function for Joram's navmesh code.                            LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::WorldToScreenPos( const float3* W, float2* S, int count ) const
{
	// calculate camera axis
	ViewPyramid p = GetView();
	float3 p1p2 = p.p2 - p.p1, p3p1 = p.p1 - p.p3;		// screen edges
	float3 f = ((p.p3 - p.pos) + (p.p2 - p.pos)) / 2;	// focal point
	float3 x = normalize( p1p2 );						// camera unit axis
	float3 y = normalize( p3p1 );						// camera unit axis
	float3 z = normalize( f );							// camera unit axis
	float invflen = 1 / length( f );					// the inversed focal distance
	float invxscrlen = 1 / (length( p1p2 ) * .5f);		// half the screen width inversed
	float invyscrlen = 1 / (length( p3p1 ) * .5f);		// half the screen height inversed

	// transform coordinates
	float3 dir;
	for (int i = 0; i < count; i++)
	{
		dir = W[i] - p.pos;								// vector from camera to pos
		dir = { dot( dir, x ), dot( dir, y ), dot( dir, z ) }; // make dir relative to camera
		if (dir.z < 0) dir *= {1000.0f, 1000.0f, 0.0f};	// prevent looking backwards (TODO: improve)
		dir *= (1 / (dir.z * invflen));					// trim dir to hit the screen
		dir.x *= invxscrlen; dir.y *= invyscrlen;		// convert to screen scale
		S[i] = make_float2( dir );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Serialize                                                          |
//  |  Save the camera data to the specified xml file.                      LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Serialize( const char* xmlFileName )
{
	XMLDocument doc;
	XMLNode* root = doc.NewElement( "camera" );
	doc.InsertFirstChild( root );
	XMLElement* campos = doc.NewElement( "position" );
	campos->SetAttribute( "x", position.x );
	campos->SetAttribute( "y", position.y );
	campos->SetAttribute( "z", position.z );
	root->InsertEndChild( campos );
	XMLElement* camdir = doc.NewElement( "direction" );
	camdir->SetAttribute( "x", direction.x );
	camdir->SetAttribute( "y", direction.y );
	camdir->SetAttribute( "z", direction.z );
	root->InsertEndChild( camdir );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "FOV" ) ))->SetText( FOV );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "brightness" ) ))->SetText( brightness );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "contrast" ) ))->SetText( contrast );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "gamma" ) ))->SetText( gamma );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "aperture" ) ))->SetText( aperture );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "distortion" ) ))->SetText( distortion );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "focalDistance" ) ))->SetText( focalDistance );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "clampValue" ) ))->SetText( clampValue );
	((XMLElement*)root->InsertEndChild( doc.NewElement( "tonemapper" ) ))->SetText( tonemapper );
	doc.SaveFile( xmlFileName ? xmlFileName : xmlFile.c_str() );
}

//  +-----------------------------------------------------------------------------+
//  |  Camera::Deserialize                                                        |
//  |  Load the camera data from the specified xml file.                    LH2'19|
//  +-----------------------------------------------------------------------------+
void Camera::Deserialize( const char* xmlFileName )
{
	xmlFile = xmlFileName;
	XMLDocument doc;
	XMLError result = doc.LoadFile( "camera.xml" );
	if (result != XML_SUCCESS) return;
	XMLNode* root = doc.FirstChild();
	if (root == nullptr) return;
	XMLElement* element = root->FirstChildElement( "position" );
	if (!element) return;
	element->QueryFloatAttribute( "x", &position.x );
	element->QueryFloatAttribute( "y", &position.y );
	element->QueryFloatAttribute( "z", &position.z );
	element = root->FirstChildElement( "direction" );
	if (!element) return;
	element->QueryFloatAttribute( "x", &direction.x );
	element->QueryFloatAttribute( "y", &direction.y );
	element->QueryFloatAttribute( "z", &direction.z );
	if (element = root->FirstChildElement( "FOV" )) element->QueryFloatText( &FOV );
	if (element = root->FirstChildElement( "brightness" )) element->QueryFloatText( &brightness );
	if (element = root->FirstChildElement( "contrast" )) element->QueryFloatText( &contrast );
	if (element = root->FirstChildElement( "gamma" )) element->QueryFloatText( &gamma );
	if (element = root->FirstChildElement( "aperture" )) element->QueryFloatText( &aperture );
	if (element = root->FirstChildElement( "distortion" )) element->QueryFloatText( &distortion );
	if (element = root->FirstChildElement( "focalDistance" )) element->QueryFloatText( &focalDistance );
	if (element = root->FirstChildElement( "clampValue" )) element->QueryFloatText( &clampValue );
	if (element = root->FirstChildElement( "tonemapper" )) element->QueryIntText( &tonemapper );
}

// EOF