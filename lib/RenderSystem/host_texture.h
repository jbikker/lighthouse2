/* host_texture.h - Copyright 2019 Utrecht University

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
//  |  HostTexture                                                                |
//  |  Stores a texture, with either integer or floating point data.              |
//  |  Policy regarding texture reuse:                                            |
//  |  - The owner of the textures is the scene. The scene does not have a        |
//  |    texture manager, to prevent adding an extra layer for presumably simple  |
//  |    functionality.                                                           |
//  |  - Multiple materials (of single or multiple models) may use a texture.     |
//  |    A refCount keeps track of this.                                          |
//  |  - A file name does not uniquely identify a texture: the file may be        |
//  |    different between folders, and the texture may have been loaded with     |
//  |    'modFlags'. Instead, a texture is uniquely identified by its full file   |
//  |    name, including path, as well as the mods field.                         |
//  |                                                                       LH2'19|
//  +-----------------------------------------------------------------------------+
class HostTexture
{
public:
	enum
	{
		HASALPHA = 1,
		NORMALMAP = 2,
		LDR = 4,
		HDR = 8
	};
	enum
	{
		LINEARIZED = 1,
		FLIPPED = 2,
		INVERTED = 4,
		GAMMACORRECTION = 8,
	};
	// constructor / destructor / conversion
	HostTexture() = default;
	HostTexture( const char* fileName, const uint modFlags = 0 );
	CoreTexDesc ConvertToCoreTexDesc() const;
	// methods
	bool Equals( const string& o, const uint m );
	void Load( const char* fileName, const uint modFlags, bool normalMap = false );
	static void sRGBtoLinear( uchar* pixels, const uint size, const uint stride );
	static float InverseGammaCorrect( float value );
	static float4 InverseGammaCorrect( const float4& value );
	void BumpToNormalMap( float heightScale );
	uint* GetLDRPixels() { return (uint*)idata; }
	float4* GetHDRPixels() { return fdata; }
	// internal methods
	int PixelsNeeded( const int width, const int height, const int MIPlevels ) const;
	void ConstructMIPmaps();
	// public properties
public:
	uint width = 0;						// width in pixels
	uint height = 0;					// height in pixels
	uint MIPlevels = 1;					// number of MIPmaps
	uint ID = 0;						// unique integer ID of this texture
	string name;						// texture name, not for unique identification
	string origin;						// origin: file from which the data was loaded, with full path
	uint flags = 0;						// flags
	uint mods = 0;						// modifications to original data
	uint refCount = 1;					// the number of materials that use this texture
	uchar4* idata = nullptr;			// pointer to a 32-bit ARGB bitmap
	float4* fdata = nullptr;			// pointer to a 128-bit ARGB bitmap
	TRACKCHANGES;						// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF