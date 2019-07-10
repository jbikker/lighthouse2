#ifdef CORECOMMON_EXPORTS
#define CORECOMMON_API __declspec(dllexport)
#else
#define CORECOMMON_API __declspec(dllimport)
#endif

struct CoreStats_DLL
{
	// device
	char* deviceName = 0;				// device name; TODO: will leak
	uint SMcount = 0;					// number of shading multiprocessors on device
	uint ccMajor = 0, ccMinor = 0;		// compute capability
	uint VRAM = 0;						// device memory, in MB
	// storage
	uint argb32TexelCount = 0;			// number of uint texels
	uint argb128TexelCount = 0;			// number of float4 texels
	uint nrm32TexelCount = 0;			// number of normal map texels
	// bvh
	float bvhBuildTime = 0;				// overall accstruc build time
	// rendering
	uint totalRays = 0;					// total number of rays cast
	uint totalExtensionRays = 0;		// total extension rays cast
	uint totalShadowRays = 0;			// total shadow rays cast
	float renderTime;					// overall render time
	float traceTime;					// time spent in OptiX
	float filterTime;					// time spent in filter code
	float filterPrepTime;				// time spent in filter prep code
	float filterCoreTime;				// time spent in filter core code
	float filterTAATime;				// time spent in filter TAA code
	float shadeTime;					// time spent in shading code
	// probe
	int probedInstid;					// id of the instance at probe position
	int probedTriid;					// id of triangle at probe position
	float probedDist;					// distance of triangle at probe position
};

// This class is exported from the CommonDLL.dll
class CORECOMMON_API CoreAPI_DLL
{
public:
	enum CoreID
	{
		OPTIXPRIME_CORE = 0,		// wavefront path tracing using Prime for pre-RTX cards
		OPTIXRTX_CORE,				// wavefront path tracing using the RTX path in Optix
		SOFTRASTERIZER_CORE,		// software rasterizer core, CPU, single-threaded
		RTXAO_CORE,					// minimal RTX-based wavefront AO renderer
		CLASSICCUDA_CORE,			// TODO: core based on LH1, with custom BVH management and pure CUDA traversal
		EMBREE_CORE,				// TODO: CPU renderer for massive scenes
		OPENCL_CORE					// TODO: OpenCL core for AMD and Intel devices
	};
	// CreateCoreAPI: instantiate and initialize a RenderCore object and obtain an interface to it.
	static CoreAPI_DLL* CreateCoreAPI( const CoreID id );
	// GetCoreStats: obtain a const ref to the CoreStats object, which provides statistics on the rendering process.
	virtual CoreStats_DLL GetCoreStats() = 0;
	// SetName / GetName: to identify the core
	virtual void SetName( string name ) = 0;
	virtual string GetName() = 0;
	// Init: initialize the core
	virtual void Init() = 0;
	// SetProbePos: set a pixel for which the triangle and instance id will be captured, e.g. for object picking.
	virtual void SetProbePos( const int2 pos ) = 0;
	// SetTarget: specify an OpenGL texture as a render target for the path tracer.
	virtual void SetTarget( GLTexture* target, const uint spp ) = 0;
	// Setting: modify a render setting
	virtual void Setting( const char* name, float value ) = 0;
	// Render: produce one frame. Convergence can be 'Converge' or 'Restart'.
	virtual void Render( const ViewPyramid& view, const Convergence converge, const float brightness, const float contrast ) = 0;
	// Shutdown: destroy the RenderCore and free all resources.
	virtual void Shutdown() = 0;
	// SetTextures: update the texture data in the RenderCore using the supplied data.
	virtual void SetTextures( const GPUTexDesc* tex, const int textureCount ) = 0;
	// SetMaterials: update the material list used by the RenderCore. Textures referenced by the materials must be set in advance.
	virtual void SetMaterials( GPUMaterial* mat, const GPUMaterialEx* matEx, const int materialCount ) = 0;
	// SetLights: update the point lights, spot lights and directional lights.
	virtual void SetLights( const GPULightTri* areaLights, const int areaLightCount,
		const GPUPointLight* pointLights, const int pointLightCount,
		const GPUSpotLight* spotLights, const int spotLightCount,
		const GPUDirectionalLight* directionalLights, const int directionalLightCount ) = 0;
	// SetSkyData: specify the data required for sky dome rendering.
	virtual void SetSkyData( const float3* pixels, const uint width, const uint height ) = 0;
	// SetGeometry: update the geometry for a single mesh.
	virtual void SetGeometry( const int meshIdx, const float4* vertexData, const int vertexCount, const int triangleCount, const GPUTri* triangles, const uint* alphaFlags = 0 ) = 0;
	// SetInstance: update the data on a single instance.
	virtual void SetInstance( const int instanceIdx, const int modelIdx, const mat4& transform ) = 0;
	// UpdateTopLevel: trigger a top-level BVH update.
	virtual void UpdateToplevel() = 0;
};

class CoreManager
{
public:
	static CORECOMMON_API CoreManager& Instance();
	CORECOMMON_API CoreAPI_DLL* LoadCore( const string& coreName );
	CORECOMMON_API void UnloadCore( CoreAPI_DLL*& core );
private:
	CoreManager();
	~CoreManager();
	typedef map<string, CoreAPI_DLL*> CoreMap;
	typedef map<string, HMODULE> LibraryMap;
	CoreMap cores;
	LibraryMap libs;
};

extern CORECOMMON_API int nCommonDLL;

CORECOMMON_API int fnCommonDLL( void );

// EOF