/* host_anim.cpp - Copyright 2019 Utrecht University

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
//  |  HostAnimation::Sampler::Sampler                                            |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostAnimation::Sampler::Sampler( tinygltf::AnimationSampler& gltfSampler, tinygltf::Model& gltfModel )
{
	ConvertFromGLTFSampler( gltfSampler, gltfModel );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Sampler::ConvertFromGLTFSampler                             |
//  |  Convert a gltf animation sampler.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Sampler::ConvertFromGLTFSampler( tinygltf::AnimationSampler& gltfSampler, tinygltf::Model& gltfModel )
{
	// https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#animations
	// extract animation times
	auto inputAccessor = gltfModel.accessors[gltfSampler.input];
	assert( inputAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );
	auto bufferView = gltfModel.bufferViews[inputAccessor.bufferView];
	auto buffer = gltfModel.buffers[bufferView.buffer];
	const float* a = (const float*)(buffer.data.data() + bufferView.byteOffset + inputAccessor.byteOffset);
	size_t count = inputAccessor.count;
	for( int i = 0; i < count; i++ ) t.push_back( a[i] );
	
	// extract animation keys
	auto outputAccessor = gltfModel.accessors[gltfSampler.output];
	bufferView = gltfModel.bufferViews[outputAccessor.bufferView];
	buffer = gltfModel.buffers[bufferView.buffer];
	const uchar* b = (const uchar*)(buffer.data.data() + bufferView.byteOffset + outputAccessor.byteOffset);
	if (outputAccessor.type == TINYGLTF_TYPE_VEC3)
	{
		// b is an array of floats (for scale or translation)
		float* f = (float*)b;
		for( int i = 0; i < inputAccessor.count; i++ ) vec3Key.push_back( make_float3( f[i * 3], f[i * 3 + 1], f[i * 3 + 2] ) );
	}
	else if (outputAccessor.type == TINYGLTF_TYPE_SCALAR)
	{
		// b can be FLOAT, BYTE, UBYTE, SHORT or USHORT... (for weights)
		vector<float> fdata;
		const int N = (int)outputAccessor.count;
		switch (outputAccessor.componentType)
		{
		case TINYGLTF_COMPONENT_TYPE_FLOAT: for (int k = 0; k < N; k++, b += 4) fdata.push_back( *((float*)b) ); break;
		case TINYGLTF_COMPONENT_TYPE_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( max( *((char*)b) / 127.0f, -1 ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( *((char*)b) / 255.0f ); break;
		case TINYGLTF_COMPONENT_TYPE_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( max( *((char*)b) / 32767.0f, -1 ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( *((char*)b) / 65535.0f ); break;
		}
		for( int i = 0; i < inputAccessor.count; i++ ) floatKey.push_back( fdata[i] );
	}
	else if (outputAccessor.type == TINYGLTF_TYPE_VEC4)
	{
		// b can be FLOAT, BYTE, UBYTE, SHORT or USHORT... (for rotation)
		vector<float> fdata;
		const int N = (int)outputAccessor.count * 4;
		switch (outputAccessor.componentType)
		{
		case TINYGLTF_COMPONENT_TYPE_FLOAT: for (int k = 0; k < N; k++, b += 4) fdata.push_back( *((float*)b) ); break;
		case TINYGLTF_COMPONENT_TYPE_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( max( *((char*)b) / 127.0f, -1 ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( *((char*)b) / 255.0f ); break;
		case TINYGLTF_COMPONENT_TYPE_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( max( *((char*)b) / 32767.0f, -1 ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( *((char*)b) / 65535.0f ); break;
		}
		for( int i = 0; i < inputAccessor.count; i++ ) vec4Key.push_back( quat( fdata[i * 4], fdata[i * 4 + 1], fdata[i * 4 + 2], fdata[i * 4 + 3] ) );
	}
	else assert( false );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::Channel                                            |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostAnimation::Channel::Channel( tinygltf::AnimationChannel& gltfChannel, tinygltf::Model& gltfModel, const int nodeBase )
{
	ConvertFromGLTFChannel( gltfChannel, gltfModel, nodeBase );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::ConvertFromGLTFChannel                             |
//  |  Convert a gltf animation channel.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Channel::ConvertFromGLTFChannel( tinygltf::AnimationChannel& gltfChannel, tinygltf::Model& gltfModel, const int nodeBase )
{
	samplerIdx = gltfChannel.sampler;
	nodeIdx = gltfChannel.target_node + nodeBase;
	if (gltfChannel.target_path.compare( "translation" ) == 0) target = 0;
	if (gltfChannel.target_path.compare( "rotation" ) == 0) target = 1;
	if (gltfChannel.target_path.compare( "scale" ) == 0) target = 2;
	if (gltfChannel.target_path.compare( "weight" ) == 0) target = 3;
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::Update                                             |
//  |  Advance channel animation time.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Channel::Update( const float dt, const Sampler* sampler )
{
	// advance animation timer
	float animDuration = sampler->t[sampler->t.size() - 1];
	t = fmod( t + dt, animDuration );
	// determine interpolation parameters
	int keyIdx = 0;
	int keyCount = (int)sampler->t.size();
	float f = 0;
	for( ; keyIdx < keyCount; keyIdx++ )
	{
		// TODO: optimize this loop, this is silly.
		float t0 = sampler->t[keyIdx];
		float t1 = sampler->t[(keyIdx + 1) % keyCount];
		if (t >= t0 && t <= t1)
		{	
			float dt = t1 - t0;
			f = (t - t0) / dt;
			break;
		}
	}
	// apply anination key
	if (target == 0) // translation
	{
		float3 key0 = sampler->vec3Key[keyIdx];
		float3 key1 = sampler->vec3Key[(keyIdx + 1) % keyCount];
		float3 key = (1 - f) * key0 + f * key1;
		HostScene::nodes[nodeIdx]->scale = key;
	}
	else if (target == 1) // rotation
	{
		quat key0 = sampler->vec4Key[keyIdx];
		quat key1 = sampler->vec4Key[(keyIdx + 1) % keyCount];
		quat key = quat::slerp( key0, key1, f );
		HostScene::nodes[nodeIdx]->rotation = key;
	}
	else if (target == 2) // scale
	{
		float3 key0 = sampler->vec3Key[keyIdx];
		float3 key1 = sampler->vec3Key[(keyIdx + 1) % keyCount];
		float3 key = (1 - f) * key0 + f * key1;
		HostScene::nodes[nodeIdx]->translation = key;
	}
	else // target == 3, weight
	{
		/* float key0 = sampler->floatKey[keyIdx];
		float key1 = sampler->floatKey[(keyIdx + 1) % keyCount];
		float key = (1 - f) * key0 + f * key1;
		HostScene::nodes[nodeIdx]->weights[0] = key; */
	}
	HostScene::nodes[nodeIdx]->UpdateTransformFromTRS();
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::HostAnimation                                               |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostAnimation::HostAnimation( tinygltf::Animation& gltfAnim, tinygltf::Model& gltfModel, const int nodeBase )
{
	ConvertFromGLTFAnim( gltfAnim, gltfModel, nodeBase );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::ConvertFromGLTFAnim                                         |
//  |  Convert a gltf animation.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::ConvertFromGLTFAnim( tinygltf::Animation& gltfAnim, tinygltf::Model& gltfModel, const int nodeBase )
{
	for (int i = 0; i < gltfAnim.samplers.size(); i++) sampler.push_back( new Sampler( gltfAnim.samplers[i], gltfModel ) );
	for (int i = 0; i < gltfAnim.channels.size(); i++) channel.push_back( new Channel( gltfAnim.channels[i], gltfModel, nodeBase ) );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Reset                                                       |
//  |  Reset the animation timers of all channels.                          LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Reset()
{
	for( int i = 0; i < channel.size(); i++ ) channel[i]->Reset();
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Update                                                      |
//  |  Advance channel animation timers.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Update( const float dt )
{
	for( int i = 0; i < channel.size(); i++ ) channel[i]->Update( dt, sampler[channel[i]->samplerIdx] );
}

// EOF