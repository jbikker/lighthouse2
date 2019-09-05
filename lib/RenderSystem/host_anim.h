/* host_anim.h - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

namespace lighthouse2 {

class HostAnimation
{
class Sampler
{
public:
	vector<float> t;				// key frame times
	vector<float3> vec3Key;			// vec3 key frames (location or scale)
	vector<quat> vec4Key;			// vec4 key frames (rotation)
	vector<float> floatKey;			// float key frames (weight)
#ifdef RENDERSYSTEMBUILD
	Sampler( tinygltf::AnimationSampler& gltfSampler, tinygltf::Model& gltfModel );
	void ConvertFromGLTFSampler( tinygltf::AnimationSampler& gltfSampler, tinygltf::Model& gltfModel );
#endif
};
class Channel
{
public:
	int samplerIdx;					// sampler used by this channel
	int nodeIdx;					// index of the node this channel affects
	int target;						// 0: translation, 1: rotation, 2: scale, 3: weights
	void Reset() { t = 0; }
	void Update( const float t, const Sampler* sampler );	// apply this channel to the target nde for time t
	// data
	float t;						// animation timer
#ifdef RENDERSYSTEMBUILD
	Channel( tinygltf::AnimationChannel& gltfChannel, tinygltf::Model& gltfModel, const int nodeBase );
	void ConvertFromGLTFChannel( tinygltf::AnimationChannel& gltfChannel, tinygltf::Model& gltfModel, const int nodeBase );
#endif
};
public:
	vector<Sampler*> sampler;		// animation samplers
	vector<Channel*> channel;		// animation channels
	void Reset();					// reset all channels
	void Update( const float dt );	// advance and apply all channels
#ifdef RENDERSYSTEMBUILD
	HostAnimation( tinygltf::Animation& gltfAnim, tinygltf::Model& gltfModel, const int nodeBase );
	void ConvertFromGLTFAnim( tinygltf::Animation& gltfAnim, tinygltf::Model& gltfModel, const int nodeBase );
#endif
};

} // namespace lighthouse2

// EOF