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
HostAnimation::Sampler::Sampler( const tinygltfAnimationSampler& gltfSampler, const tinygltfModel& gltfModel )
{
	ConvertFromGLTFSampler( gltfSampler, gltfModel );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Sampler::ConvertFromGLTFSampler                             |
//  |  Convert a gltf animation sampler.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Sampler::ConvertFromGLTFSampler( const tinygltfAnimationSampler& gltfSampler, const tinygltfModel& gltfModel )
{
	// https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#animations
	// store interpolation type
	if (gltfSampler.interpolation == "STEP") interpolation = STEP;
	else if (gltfSampler.interpolation == "CUBICSPLINE") interpolation = SPLINE;
	else /* if (gltfSampler.interpolation == "LINEAR" ) */ interpolation = LINEAR;
	// extract animation times
	auto inputAccessor = gltfModel.accessors[gltfSampler.input];
	assert( inputAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );
	auto bufferView = gltfModel.bufferViews[inputAccessor.bufferView];
	auto buffer = gltfModel.buffers[bufferView.buffer];
	const float* a = (const float*)(buffer.data.data() + bufferView.byteOffset + inputAccessor.byteOffset);
	size_t count = inputAccessor.count;
	for (int i = 0; i < count; i++) t.push_back( a[i] );
	// extract animation keys
	auto outputAccessor = gltfModel.accessors[gltfSampler.output];
	bufferView = gltfModel.bufferViews[outputAccessor.bufferView];
	buffer = gltfModel.buffers[bufferView.buffer];
	const uchar* b = (const uchar*)(buffer.data.data() + bufferView.byteOffset + outputAccessor.byteOffset);
	if (outputAccessor.type == TINYGLTF_TYPE_VEC3)
	{
		// b is an array of floats (for scale or translation)
		float* f = (float*)b;
		const int N = (int)outputAccessor.count;
		for (int i = 0; i < N; i++) vec3Key.push_back( make_float3( f[i * 3], f[i * 3 + 1], f[i * 3 + 2] ) );
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
		for (int i = 0; i < N; i++) floatKey.push_back( fdata[i] );
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
		for (int i = 0; i < outputAccessor.count; i++) vec4Key.push_back( quat( fdata[i * 4 + 3], fdata[i * 4], fdata[i * 4 + 1], fdata[i * 4 + 2] ) );
	}
	else assert( false );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::Channel                                            |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostAnimation::Channel::Channel( const tinygltfAnimationChannel& gltfChannel, const tinygltfModel& gltfModel, const int nodeBase )
{
	ConvertFromGLTFChannel( gltfChannel, gltfModel, nodeBase );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::ConvertFromGLTFChannel                             |
//  |  Convert a gltf animation channel.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Channel::ConvertFromGLTFChannel( const tinygltfAnimationChannel& gltfChannel, const tinygltfModel& gltfModel, const int nodeBase )
{
	samplerIdx = gltfChannel.sampler;
	nodeIdx = gltfChannel.target_node + nodeBase;
	if (gltfChannel.target_path.compare( "translation" ) == 0) target = 0;
	if (gltfChannel.target_path.compare( "rotation" ) == 0) target = 1;
	if (gltfChannel.target_path.compare( "scale" ) == 0) target = 2;
	if (gltfChannel.target_path.compare( "weights" ) == 0) target = 3;
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Channel::Update                                             |
//  |  Advance channel animation time.                                      LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Channel::Update( const float dt, const Sampler* sampler )
{
	// advance animation timer
	t += dt;
	int keyCount = (int)sampler->t.size();
	float animDuration = sampler->t[keyCount - 1];
	while (t > animDuration) t -= animDuration, k = 0;
	while (t > sampler->t[(k + 1) % keyCount]) k++;
	// determine interpolation parameters
	float t0 = sampler->t[k % keyCount];
	float t1 = sampler->t[(k + 1) % keyCount];
	float f = (t - t0) / (t1 - t0);
	// apply anination key
	if (target == 0) // translation
	{
		switch (sampler->interpolation)
		{
		case Sampler::SPLINE:
		{
			float t = f, t2 = t * t, t3 = t2 * t;
			float3 p0 = sampler->vec3Key[k * 3 + 1];
			float3 m0 = (t1 - t0) * sampler->vec3Key[k * 3 + 2];
			float3 p1 = sampler->vec3Key[(k + 1) * 3 + 1];
			float3 m1 = (t1 - t0) * sampler->vec3Key[(k + 1) * 3];
			HostScene::nodes[nodeIdx]->translation =
				m0 * (t3 - 2 * t2 + t) +
				p0 * (2 * t3 - 3 * t2 + 1) +
				p1 * (-2 * t3 + 3 * t2) +
				m1 * (t3 - t2);
			break;
		}
		case Sampler::STEP:
			HostScene::nodes[nodeIdx]->translation = sampler->vec3Key[k];
			break;
		default:
		{
			float3 key1 = sampler->vec3Key[k];
			float3 key2 = sampler->vec3Key[k + 1];
			HostScene::nodes[nodeIdx]->translation = (1 - f) * key1 + f * key2;
			break;
		}
		};
		HostScene::nodes[nodeIdx]->transformed = true;
	}
	else if (target == 1) // rotation
	{
		switch (sampler->interpolation)
		{
		case Sampler::SPLINE:
		{
		#if 0
			quat key0 = sampler->vec4Key[k + keyCount];
			quat key1 = sampler->vec4Key[k + 1 + keyCount];
			quat interpolatedKey = quat::slerp( key0, key1, f );
			interpolatedKey.normalize();
			HostScene::nodes[nodeIdx]->rotation = interpolatedKey;
		#else
			float t = f, t2 = t * t, t3 = t2 * t;
			quat p0 = sampler->vec4Key[k * 3 + 1];
			quat m0 = sampler->vec4Key[k * 3 + 2] * (t1 - t0);
			quat p1 = sampler->vec4Key[(k + 1) * 3 + 1];
			quat m1 = sampler->vec4Key[(k + 1) * 3] * (t1 - t0);
			quat interpolatedKey =
				m0 * (t3 - 2 * t2 + t) +
				p0 * (2 * t3 - 3 * t2 + 1) +
				p1 * (-2 * t3 + 3 * t2) +
				m1 * (t3 - t2);
			interpolatedKey.normalize();
			HostScene::nodes[nodeIdx]->rotation = interpolatedKey;
		#endif
			break;
		}
		case Sampler::STEP:
		{
			quat interpolatedKey = sampler->vec4Key[k];
			interpolatedKey.normalize();
			HostScene::nodes[nodeIdx]->rotation = interpolatedKey;
			break;
		}
		default:
		{
			quat key0 = sampler->vec4Key[k];
			quat key1 = sampler->vec4Key[k + 1];
			quat interpolatedKey = quat::slerp( key0, key1, f );
			interpolatedKey.normalize();
			HostScene::nodes[nodeIdx]->rotation = interpolatedKey;
			break;
		};
		HostScene::nodes[nodeIdx]->transformed = true;
		}
	}
	else if (target == 2) // scale
	{
		switch (sampler->interpolation)
		{
		case Sampler::SPLINE:
		{
			float t = f, t2 = t * t, t3 = t2 * t;
			float3 p0 = sampler->vec3Key[k * 3 + 1];
			float3 m0 = (t1 - t0) * sampler->vec3Key[k * 3 + 2];
			float3 p1 = sampler->vec3Key[(k + 1) * 3 + 1];
			float3 m1 = (t1 - t0) * sampler->vec3Key[(k + 1) * 3];
			HostScene::nodes[nodeIdx]->scale =
				m0 * (t3 - 2 * t2 + t) +
				p0 * (2 * t3 - 3 * t2 + 1) +
				p1 * (-2 * t3 + 3 * t2) +
				m1 * (t3 - t2);
			break;
		};
		case Sampler::STEP:
			HostScene::nodes[nodeIdx]->scale = sampler->vec3Key[k];
			break;
		default:
			float3 key0 = sampler->vec3Key[k];
			float3 key1 = sampler->vec3Key[k + 1];
			float3 interpolatedKey = (1 - f) * key0 + f * key1;
			HostScene::nodes[nodeIdx]->scale = interpolatedKey;
			break;
		};
		HostScene::nodes[nodeIdx]->transformed = true;
	}
	else // target == 3, weight
	{
		int weightCount = (int)(sampler->floatKey.size() / sampler->t.size());
		for (int i = 0; i < weightCount; i++)
		{
			float key0 = sampler->floatKey[k * weightCount + i];
			float key1 = sampler->floatKey[(k + 1) * weightCount + i];
			float interpolatedKey = (1 - f) * key0 + f * key1;
			HostScene::nodes[nodeIdx]->weights[i] = interpolatedKey;
		}
		HostScene::nodes[nodeIdx]->morphed = true;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::HostAnimation                                               |
//  |  Constructor.                                                         LH2'19|
//  +-----------------------------------------------------------------------------+
HostAnimation::HostAnimation( tinygltfAnimation& gltfAnim, tinygltfModel& gltfModel, const int nodeBase )
{
	ConvertFromGLTFAnim( gltfAnim, gltfModel, nodeBase );
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::ConvertFromGLTFAnim                                         |
//  |  Convert a gltf animation.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::ConvertFromGLTFAnim( tinygltfAnimation& gltfAnim, tinygltfModel& gltfModel, const int nodeBase )
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
	for (int i = 0; i < channel.size(); i++) channel[i]->Reset();
}

//  +-----------------------------------------------------------------------------+
//  |  HostAnimation::Update                                                      |
//  |  Advance channel animation timers.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
void HostAnimation::Update( const float dt )
{
	for (int i = 0; i < channel.size(); i++) channel[i]->Update( dt, sampler[channel[i]->samplerIdx] );
}

// EOF