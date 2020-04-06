/* camera.h - Copyright 2019/2020 Utrecht University

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

#pragma once

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  Camera                                                                     |
//  |  Camera class definition.                                             LH2'19|
//  +-----------------------------------------------------------------------------+
class Camera
{
public:
	// constructor / destructor
	Camera() = default;
	Camera( const char* xmlFile );
	~Camera();
	// data members
	mat4 transform;									// 4x4 camera matrix
	float focalDistance = 5.0f;						// distance of the focal plane
	float aperture = EPSILON;						// aperture size
	float distortion = 0.05f;						// barrel distortion
	float brightness = 0.0f;						// combined with contrast:
	float contrast = 0.0f;							// pragmatic representation of exposure
	float gamma = 2.2f;								// monitor gamma
	float FOV = 40;									// field of view, in degrees
	int tonemapper = 4;								// see tonemap.frag for options
	float aspectRatio = 1.0f;						// image plane aspect ratio
	float clampValue = 10.0f;						// firefly clamping
	int2 pixelCount = make_int2( 1, 1 );			// actual pixel count; needed for pixel spread angle
	// methods
	void LookAt( const float3 O, const float3 T );	// position the camera at O, looking at T
	void TranslateRelative( float3 T );				// move camera relative to orientation
	void TranslateTarget( float3 T );				// move camera target; used for rotating camera
	ViewPyramid GetView() const;					// calculate a view based on the setup
	mat4 GetMatrix() const;							// calculate a matrix for the camera
	void SetMatrix( mat4& T );						// set the camera view using a matrix
	int2 WorldToScreenPos( const float3& P );		// determine the screen space coordinate of a world space position
	void WorldToScreenPos( const float3* W, float2* S, int count ) const;	// convert world pos to screen pos
	float3 PrimaryHitPos( int2 pos, float dist );	// calculate hit pos along a ray through a pixel
	void Serialize( const char* xmlFile = 0 );		// save the camera to an xml file
	void Deserialize( const char* xmlFile );		// load the camera from an xml file
	// private methods
private:
	void CalculateMatrix( float3& x, float3& y, float3& z ) const;
	// private data
private:
	string xmlFile = "camera.dat";					// file the camera was loaded from, used for dtor
	TRACKCHANGES;									// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF