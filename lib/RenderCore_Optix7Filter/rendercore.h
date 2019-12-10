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

//  +-----------------------------------------------------------------------------+
//  |  DeviceVars                                                                 |
//  |  Copy of device-side variables, to detect changes.                    LH2'19|
//  +-----------------------------------------------------------------------------+
struct DeviceVars
{
	// impossible values to trigger an update in the first frame
	float clampValue = -1.0f;
	float geometryEpsilon = 1e34f;
	float filterClampDirect = 2.5f;
	float filterClampIndirect = 15.0f;
	uint filterEnabled = 1;
	uint TAAEnabled = 1;
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
	void SetProbePos( const int2 pos );
	// internal methods
private:
	void SyncStorageType( const TexelStorage storage );
	void CreateOptixContext( int cc );
	// helpers
	template <class T> CUDAMaterial::Map Map( T v )
	{
		CUDAMaterial::Map m;
		CoreTexDesc& t = texDescs[v.textureID];
		m.width = t.width, m.height = t.height, m.uscale = v.uvscale.x, m.vscale = v.uvscale.y;
		m.uoffs = v.uvoffset.x, m.voffs = v.uvoffset.y, m.addr = t.firstPixel;
		return m;
	}
	// cuda call abbreviation
	void applyFilter( const uint phase, CoreBuffer<float4>* A, CoreBuffer<float4>* B, CoreBuffer<float4>* C, const uint lastPass = 0 );
	// data members
	int scrwidth = 0, scrheight = 0;				// current screen width and height
	int scrspp = 1;									// samples to be taken per screen pixel
	int skywidth = 0, skyheight = 0;				// size of the skydome texture
	int maxPixels = 0;								// max screen size buffers can accomodate without a realloc
	int currentSPP = 0;								// spp count which will be accomodated without a realloc
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
	CoreBuffer<float4>* accumulator = 0;			// accumulator buffer for the path tracer
	CoreBuffer<Counters>* counterBuffer = 0;		// counters for wavefront path tracing
	CoreBuffer<CoreInstanceDesc>* instDescBuffer = 0; // instance descriptor array
	CoreBuffer<uint>* texel32Buffer = 0;			// texel buffer 0: regular ARGB32 texture data
	CoreBuffer<float4>* hitBuffer = 0;				// intersection results
	CoreBuffer<float4>* pathStateBuffer = 0;		// path state buffer
	CoreBuffer<float4>* connectionBuffer = 0;		// shadow rays
	CoreBuffer<OptixInstance>* instanceArray = 0;	// instance descriptors for Optix
	CoreBuffer<Params>* optixParams;				// parameters to be used in optix code
	CoreTexDesc* texDescs = 0;						// array of texture descriptors
	int textureCount = 0;							// size of texture descriptor array
	int SMcount = 0;								// multiprocessor count, used for persistent threads
	int computeCapability;							// device compute capability
	int samplesTaken = 0;							// number of accumulated samples in accumulator
	uint camRNGseed = 0x12345678;					// seed for the RNG that feeds the renderer
	uint frameCycle = 0;							// keeping track of the frame, for TAA
	float prevj0 = 0, prevj1 = 0;					// jittering values of the previous frame
	DeviceVars vars;								// copy of device-side variables, to detect changes
	bool firstConvergingFrame = false;				// to reset accumulator for first converging frame
	// filtering data
	CoreBuffer<uint4>* features = 0;				// features for filtering
	CoreBuffer<float4>* shading = 0;				// illumination to be filtered
	CoreBuffer<float2>* motion = 0;					// motion vectors (actually: position in history buffers)
	CoreBuffer<float4>* moments = 0;				// luminance moments for variance estimation in SVGF
	CoreBuffer<float4>* prevMoments = 0;			// luminance moments history
	CoreBuffer<float4>* prevPixels = 0;				// raw pixels from the previous frame, for TAA
	CoreBuffer<float4>* worldPos = 0;				// fragment worldspace positions, for filter
	CoreBuffer<float4>* prevWorldPos = 0;			// previous frame fragment worldspace positions
	CoreBuffer<float4>* filteredIN = 0;				// filtered data from the previous frame
	CoreBuffer<float4>* filteredOUT = 0;			// result of filtering in the current frame
	CoreBuffer<float4>* deltaDepth = 0;				// depth derivatives for filtering
	CoreBuffer<float4>* debugData = 0;				// used to store arbitrary data for inspection
	ViewPyramid prevView;							// previous view frustum, for reprojection
	// blue noise table: contains the three tables distributed by Heitz.
	// Offset 0: an Owen-scrambled Sobol sequence of 256 samples of 256 dimensions.
	// Offset 65536: scrambling tile of 128x128 pixels; 128 * 128 * 8 values.
	// Offset 65536 * 3: ranking tile of 128x128 pixels; 128 * 128 * 8 values. Total: 320KB.
	CoreBuffer<uint>* blueNoise = 0;
	int blueSlot = 0;
	// timing
	cudaEvent_t traceStart[MAXPATHLENGTH], traceEnd[MAXPATHLENGTH];
	cudaEvent_t shadeStart[MAXPATHLENGTH], shadeEnd[MAXPATHLENGTH];
	cudaEvent_t shadowStart, shadowEnd, filterStart, filterEnd;
public:
	CoreStats coreStats;							// rendering statistics
	static OptixDeviceContext optixContext;			// static, for access from CoreMesh
	enum { RAYGEN = 0, RAD_MISS, OCC_MISS, RAD_HIT, OCC_HIT };
	OptixShaderBindingTable sbt;
	OptixModule ptxModule;
	OptixPipeline pipeline;
	OptixProgramGroup progGroup[5];
	OptixTraversableHandle bvhRoot;
	Params params;
	CUdeviceptr d_params;
};

} // namespace lh2core

// EOF