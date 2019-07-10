/* interoptexture.h - Copyright 2019 Utrecht University

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

namespace lh2core {

//  +-----------------------------------------------------------------------------+
//  |  InteropTexture                                                             |
//  |  Extended texture object for CUDA interop.                            LH2'19|
//  +-----------------------------------------------------------------------------+
class InteropTexture
{
public:
	// constructor / destructor
	~InteropTexture();
	// get / set
	void SetTexture( GLTexture* t );
	cudaGraphicsResource** GetResID() { return &res; }
	void LinkToSurface( const surfaceReference* s );
	// methods
	void BindSurface();
	void UnbindSurface();
private:
	// data members
	GLTexture* texture = 0;
	cudaGraphicsResource* res = nullptr;
	const surfaceReference* surfRef = nullptr;
	bool linked = false, bound = false;
};

} // namespace lh2core

// EOF