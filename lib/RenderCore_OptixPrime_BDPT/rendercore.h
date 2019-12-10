/* rendercore.h - Copyright 2019 Utrecht University

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

namespace lh2core
{

// from OptiX SDK, putil.h
#define CHK_PRIME(c){RTPresult r=c;if(r){const char*e;rtpContextGetLastErrorString(RenderCore::context,&e);\
FatalError( "Error at line %i of %s: %s", __LINE__,__FILE__,e);exit(1);}}

//  +-----------------------------------------------------------------------------+
//  |  DeviceVars                                                                 |
//  |  Copy of device-side variables, to detect changes.                    LH2'19|
//  +-----------------------------------------------------------------------------+
struct DeviceVars
{
	// impossible values to trigger an update in the first frame
	float clampValue = -1.0f;
	float geometryEpsilon = 1e34f;
};

//  +-----------------------------------------------------------------------------+
//  |  RenderCore                                                                 |
//  |  Encapsulates device code.                                            LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore
{
public:
	// methods
	void Init();
	void Render( const ViewPyramid& view, const Convergence converge );
	void Setting( const char* name, const float value );
	void SetTarget( GLTexture* target, const uint spp );
	void Shutdown();
	// passing data. Note: RenderCore always copies what it needs; the passed data thus remains the
	// property of the caller, and can be safely deleted or modified as soon as these calls return.
	void SetTextures( const CoreTexDesc* tex, const int textureCount );
	void SetMaterials( CoreMaterial* mat, const int materialCount ); // textures must be in sync when calling this
	void SetLights( const CoreLightTri* areaLights, const int areaLightCount,
		const CorePointLight* pointLights, const int pointLightCount,
		const CoreSpotLight* spotLights, const int spotLightCount,
		const CoreDirectionalLight* directionalLights, const int directionalLightCount );
	void SetSkyData( const float3* pixels, const uint width, const uint height );
	// geometry and instances:
	// a scene is setup by first passing a number of meshes (geometry), then a number of instances.
	// note that stored meshes can be used zero, one or multiple times in the scene.
	// also note that, when using alpha flags, materials must be in sync.
	void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const CoreTri* triangles, const uint* alphaFlags = 0 );
	void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform );
	void UpdateToplevel();
	int4 GetScreenParams();
	void SetProbePos( const int2 pos );
	// internal methods
private:
	void SyncStorageType( const TexelStorage storage );
	// helpers
	template <class T> CUDAMaterial::Map Map( T v )
	{
		CUDAMaterial::Map m;
		CoreTexDesc& t = texDescs[v.textureID];
		m.width = t.width, m.height = t.height, m.uscale = v.uvscale.x, m.vscale = v.uvscale.y;
		m.uoffs = v.uvoffset.x, m.voffs = v.uvoffset.y, m.addr = t.firstPixel;
		return m;
	}
	// data members
	int scrwidth = 0, scrheight = 0;				// current screen width and height
	int scrspp = 1;									// samples to be taken per screen pixel
	int skywidth = 0, skyheight = 0;				// size of the skydome texture
	int maxPixels = 0;								// max screen size buffers can accomodate without a realloc
	int currentSPP = 0;
	int samplesTaken = 0;// spp count which will be accomodated without a realloc
	uint maxVisNum = 1600 * 900 * VIS_BUFFERSIZE;
	int2 probePos = make_int2( 0 );					// triangle picking; primary ray for this pixel copies its triid to coreStats.probedTriid
	vector<CoreMesh*> meshes;						// list of meshes, to be referenced by the instances
	vector<CoreInstance*> instances;					// list of instances: model id plus transform
	bool instancesDirty = true;						// we need to sync the instance array to the device
	InteropTexture renderTarget;					// CUDA will render to this texture
	CoreBuffer<CUDAMaterial>* materialBuffer = 0;	// material array
	CUDAMaterial* hostMaterialBuffer = 0;			// core-managed copy of the materials
	CoreBuffer<CoreLightTri>* areaLightBuffer;		// area lights
	CoreBuffer<CorePointLight>* pointLightBuffer;	// point lights
	CoreBuffer<CoreSpotLight>* spotLightBuffer;		// spot lights
	CoreBuffer<CoreDirectionalLight>* directionalLightBuffer;	// directional lights
	CoreBuffer<float4>* texel128Buffer = 0;			// texel buffer 1: hdr ARGB128 texture data
	CoreBuffer<uint>* normal32Buffer = 0;			// texel buffer 2: integer-encoded normals
	CoreBuffer<float3>* skyPixelBuffer = 0;			// skydome texture data
	RTPmodel* topLevel = 0;							// the top-level node; combines all instances and is the entry point for ray queries
	CoreBuffer<CoreInstanceDesc>* instDescBuffer = 0; // instance descriptor array
	CoreBuffer<uint>* texel32Buffer = 0;			// texel buffer 0: regular ARGB32 texture data

	CoreBuffer<float4>* accumulatorOnePass = 0;
	CoreBuffer<Counters>* counterBuffer = 0;		// counters for persistent threads

	// BDPT
	///////////////////////////
	CoreBuffer<float4>* contributions = 0;

	CoreBuffer<uint>* constructEyeBuffer = 0;

	CoreBuffer<uint>* eyePathBuffer = 0;

	CoreBuffer<BiPathState>* pathDataBuffer = 0;	// additional path state data
	//CoreBuffer<BiPathState>* pathDataBufferExt = 0;
	CoreBuffer<Ray4>* visibilityRayBuffer = 0;         // buffer for visibility of connection
	CoreBuffer<uint>* visibilityHitBuffer = 0; // buffer for OptiX intersection results
	RTPbufferdesc visibilityRaysDesc;				// buffer descriptor for extension rays
	RTPbufferdesc visibilityHitsDesc;				// buffer descriptor for extension ray hits

	CoreBuffer<Ray4>* randomWalkRayBuffer = 0;         // buffer for visibility of connection
	CoreBuffer<Intersection>* randomWalkHitBuffer = 0; // buffer for OptiX intersection results
	RTPbufferdesc randomWalkRaysDesc;				// buffer descriptor for extension rays
	RTPbufferdesc randomWalkHitsDesc;				// buffer descriptor for extension ray hits

	///////////////////////////////
	CoreTexDesc* texDescs = 0;						// array of texture descriptors
	int textureCount = 0;							// size of texture descriptor array
	int SMcount = 0;								// multiprocessor count, used for persistent threads
	int computeCapability;							// device compute capability
	int inBuffer = 0, outBuffer = 1;				// extension ray buffers are double buffered
	uint camRNGseed = 0x12345678;					// seed for the RNG that feeds the renderer
	uint seed = 0x23456789;							// generic seed
	DeviceVars vars;								// copy of device-side variables, to detect changes
	// blue noise table: contains the three tables distributed by Heitz.
	// Offset 0: an Owen-scrambled Sobol sequence of 256 samples of 256 dimensions.
	// Offset 65536: scrambling tile of 128x128 pixels; 128 * 128 * 8 values.
	// Offset 65536 * 3: ranking tile of 128x128 pixels; 128 * 128 * 8 values. Total: 320KB.
	CoreBuffer<uint>* blueNoise = 0;
	// timing
	cudaEvent_t shadeStart[MAXPATHLENGTH], shadeEnd[MAXPATHLENGTH];	// events for timing CUDA code
public:
	static RTPcontext context;						// the OptiX prime context
	CoreStats coreStats;							// rendering statistics
};

} // namespace lh2core

// EOF