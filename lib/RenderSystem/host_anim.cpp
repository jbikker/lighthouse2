/* host_anim.cpp - Copyright 2019/2020 Utrecht University

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
		case TINYGLTF_COMPONENT_TYPE_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( max( *((char*)b) / 127.0f, -1.0f ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( *((char*)b) / 255.0f ); break;
		case TINYGLTF_COMPONENT_TYPE_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( max( *((char*)b) / 32767.0f, -1.0f ) ); break;
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
		case TINYGLTF_COMPONENT_TYPE_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( max( *((char*)b) / 127.0f, -1.0f ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: for (int k = 0; k < N; k++, b++) fdata.push_back( *((char*)b) / 255.0f ); break;
		case TINYGLTF_COMPONENT_TYPE_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( max( *((char*)b) / 32767.0f, -1.0f ) ); break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: for (int k = 0; k < N; k++, b += 2) fdata.push_back( *((char*)b) / 65535.0f ); break;
		}
		for (int i = 0; i < outputAccessor.count; i++) vec4Key.push_back( quat( fdata[i * 4 + 3], fdata[i * 4], fdata[i * 4 + 1], fdata[i * 4 + 2] ) );
	}
	else assert( false );
}

//  +-----------------------------------------------------------------------------+
//  |  Sampler::SampleFloat, SampleVec3, SampleVec4                               |
//  |  Get a value from the sampler.                                        LH2'19|
//  +-----------------------------------------------------------------------------+
float HostAnimation::Sampler::SampleFloat( float currentTime, int k, int i, int count ) const
{
	// handle valid out-of-bounds
	if (k == 0 && currentTime < t[0]) return interpolation == SPLINE ? floatKey[1] : floatKey[0];
	// determine interpolation parameters
	const float t0 = t[k], t1 = t[k + 1];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	if (f <= 0) return floatKey[0]; switch (interpolation)
	{
	case SPLINE:
	{
		const float t = f, t2 = t * t, t3 = t2 * t;
		const float p0 = floatKey[(k * count + i) * 3 + 1];
		const float m0 = (t1 - t0) * floatKey[(k * count + i) * 3 + 2];
		const float p1 = floatKey[((k + 1) * count + i) * 3 + 1];
		const float m1 = (t1 - t0) * floatKey[((k + 1) * count + i) * 3];
		return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
	}
	case Sampler::STEP:
		return floatKey[k];
	default:
		return (1 - f) * floatKey[k * count + i] + f * floatKey[(k + 1) * count + i];
	};
}
float3 HostAnimation::Sampler::SampleVec3( float currentTime, int k ) const
{
	// handle valid out-of-bounds
	if (k == 0 && currentTime < t[0]) return interpolation == SPLINE ? vec3Key[1] : vec3Key[0];
	// determine interpolation parameters
	const float t0 = t[k], t1 = t[k + 1];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	if (f <= 0) return vec3Key[0]; switch (interpolation)
	{
	case SPLINE:
	{
		const float t = f, t2 = t * t, t3 = t2 * t;
		const float3 p0 = vec3Key[k * 3 + 1];
		const float3 m0 = (t1 - t0) * vec3Key[k * 3 + 2];
		const float3 p1 = vec3Key[(k + 1) * 3 + 1];
		const float3 m1 = (t1 - t0) * vec3Key[(k + 1) * 3];
		return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
	}
	case Sampler::STEP: return vec3Key[k];
	default: return (1 - f) * vec3Key[k] + f * vec3Key[k + 1];
	};
}
quat HostAnimation::Sampler::SampleQuat( float currentTime, int k ) const
{
	// handle valid out-of-bounds
	if (k == 0 && currentTime < t[0]) return interpolation == SPLINE ? vec4Key[1] : vec4Key[0];
	// determine interpolation parameters
	const float t0 = t[k], t1 = t[k + 1];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	quat key;
	if (f <= 0) return vec4Key[0]; else switch (interpolation)
	{
#if 1
	case SPLINE:
	{
		const float t = f, t2 = t * t, t3 = t2 * t;
		const quat p0 = vec4Key[k * 3 + 1];
		const quat m0 = vec4Key[k * 3 + 2] * (t1 - t0);
		const quat p1 = vec4Key[(k + 1) * 3 + 1];
		const quat m1 = vec4Key[(k + 1) * 3] * (t1 - t0);
		key = m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
		key.normalize();
		break;
	}
#endif
	case STEP:
		key = vec4Key[k];
		break;
	default:
		key = quat::slerp( vec4Key[k], vec4Key[k + 1], f );
		key.normalize();
		break;
	};
	key.normalize();
	return key;
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
	if (animDuration == 0 /* book scene */ || keyCount == 1 /* bird */)
	{
		if (target == 0) // translation
		{
			HostScene::nodePool[nodeIdx]->translation = sampler->vec3Key[0];
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else if (target == 1) // rotation
		{
			HostScene::nodePool[nodeIdx]->rotation = sampler->vec4Key[0];
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else if (target == 2) // scale
		{
			HostScene::nodePool[nodeIdx]->scale = sampler->vec3Key[0];
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else // target == 3, weight
		{
			int weightCount = (int)HostScene::nodePool[nodeIdx]->weights.size();
			for (int i = 0; i < weightCount; i++)
				HostScene::nodePool[nodeIdx]->weights[i] = sampler->floatKey[0];
			HostScene::nodePool[nodeIdx]->morphed = true;
		}
	}
	else
	{
		while (t >= animDuration) t -= animDuration, k = 0;
		while (t >= sampler->t[(k + 1) % keyCount]) k++;
		// determine interpolation parameters
		assert( k < keyCount - 1 );
		assert( t >= 0 && t < animDuration );
		assert( t < sampler->t[k + 1] );
		// apply anination key
		if (target == 0) // translation
		{
			assert( sampler->t.size() == sampler->vec3Key.size() );
			HostScene::nodePool[nodeIdx]->translation = sampler->SampleVec3( t, k );
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else if (target == 1) // rotation
		{
			assert( sampler->t.size() == sampler->vec4Key.size() );
			HostScene::nodePool[nodeIdx]->rotation = sampler->SampleQuat( t, k );
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else if (target == 2) // scale
		{
			assert( sampler->t.size() == sampler->vec3Key.size() );
			HostScene::nodePool[nodeIdx]->scale = sampler->SampleVec3( t, k );
			HostScene::nodePool[nodeIdx]->transformed = true;
		}
		else // target == 3, weight
		{
			int weightCount = (int)HostScene::nodePool[nodeIdx]->weights.size();
			for (int i = 0; i < weightCount; i++)
				HostScene::nodePool[nodeIdx]->weights[i] = sampler->SampleFloat( t, k, i, weightCount );
			HostScene::nodePool[nodeIdx]->morphed = true;
		}
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