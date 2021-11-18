/* api.cpp - Copyright 2019/2021 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   In this file, the PBRT "API" is implemented. This API is used to add
   geometry, material info, cameras and a hoist of other settings to the
   scene. Conversion from "pbrt" to LH2 "Host" representation happens here.
*/

#include "materials/pbrt/pbrtparser.h"

namespace pbrt
{

// API Local Classes
PBRT_CONSTEXPR int MaxTransforms = 2;
PBRT_CONSTEXPR int StartTransformBits = 1 << 0;
PBRT_CONSTEXPR int EndTransformBits = 1 << 1;
PBRT_CONSTEXPR int AllTransformsBits = (1 << MaxTransforms) - 1;
struct TransformSet
{
	// TransformSet Public Methods
	Transform& operator[]( int i )
	{
		CHECK_GE( i, 0 );
		CHECK_LT( i, MaxTransforms );
		return t[i];
	}
	const Transform& operator[]( int i ) const
	{
		CHECK_GE( i, 0 );
		CHECK_LT( i, MaxTransforms );
		return t[i];
	}
	friend TransformSet Inverse( const TransformSet& ts )
	{
		TransformSet tInv;
		for (int i = 0; i < MaxTransforms; ++i) tInv.t[i] = ts.t[i].Inverted();
		return tInv;
	}
	bool IsAnimated() const
	{
		for (int i = 0; i < MaxTransforms - 1; ++i) if (t[i] != t[i + 1]) return true;
		return false;
	}
private:
	Transform t[MaxTransforms];
};

// MaterialInstance represents both an instance of a material as well as
// the information required to create another instance of it (possibly with
// different parameters from the shape).
struct MaterialInstance
{
	MaterialInstance() = default;
	MaterialInstance( const std::string& name, HostMaterial* mtl, ParamSet params )
		: name( name ), material( mtl ), params( std::move( params ) )
	{
	}
	std::string name;
	HostMaterial* material;
	ParamSet params;
};

struct GraphicsState
{
	// Graphics State Methods
	GraphicsState() : floatTextures( std::make_shared<FloatTextureMap>() ),
		// spectrumTextures( std::make_shared<SpectrumTextureMap>() ),
		float3Textures( std::make_shared<Float3TextureMap>() ),
		namedMaterials( std::make_shared<NamedMaterialMap>() )
	{
		ParamSet empty;
		TextureParams tp( empty, empty, *floatTextures, *float3Textures );
		HostMaterial* mtl( CreateMatteMaterial( tp ) );
		currentMaterial = std::make_shared<MaterialInstance>( "matte", mtl, ParamSet() );
	}
	HostMaterial* GetMaterialForShape( const ParamSet& geomParams );
	// MediumInterface CreateMediumInterface();

	// Graphics State
	// std::string currentInsideMedium, currentOutsideMedium;

	// Updated after book publication: floatTextures, spectrumTextures, and
	// namedMaterials are all implemented using a "copy on write" approach
	// for more efficient GraphicsState management.  When state is pushed
	// in pbrtAttributeBegin(), we don't immediately make a copy of these
	// maps, but instead record that each one is shared.  Only if an item
	// is added to one is a unique copy actually made.
	using FloatTextureMap = std::map<std::string, HostMaterial::ScalarValue*>;
	std::shared_ptr<FloatTextureMap> floatTextures;
	bool floatTexturesShared = false;

	// using SpectrumTextureMap = std::map<std::string, HostMaterial::Vec3Value*>;
	// std::shared_ptr<SpectrumTextureMap> spectrumTextures;
	// bool spectrumTexturesShared = false;

	using Float3TextureMap = std::map<std::string, HostMaterial::Vec3Value*>;
	std::shared_ptr<Float3TextureMap> float3Textures;
	bool float3TexturesShared = false;

	using NamedMaterialMap = std::map<std::string, std::shared_ptr<MaterialInstance>>;
	std::shared_ptr<NamedMaterialMap> namedMaterials;
	bool namedMaterialsShared = false;

	std::shared_ptr<MaterialInstance> currentMaterial;
	ParamSet areaLightParams;
	std::string areaLight;
	bool reverseOrientation = false;
};

static TransformSet curTransform;
static uint32_t activeTransformBits = AllTransformsBits;
static std::map<std::string, TransformSet> namedCoordinateSystems;
GraphicsState graphicsState;
static std::vector<GraphicsState> pushedGraphicsStates;
static std::vector<TransformSet> pushedTransforms;
static std::vector<uint32_t> pushedActiveTransformBits;

Options PbrtOptions;

enum class APIState
{
	Uninitialized,
	OptionsBlock,
	WorldBlock
};
static APIState currentApiState = APIState::Uninitialized;
int catIndentCount = 0;

std::map<std::string, std::vector<HostNode*>> instances;
std::vector<HostNode*>* currentInstance = nullptr;

// API Macros
#define VERIFY_INITIALIZED( func )                         \
	if ( !( PbrtOptions.cat || PbrtOptions.toPly ) &&      \
		 currentApiState == APIState::Uninitialized )      \
	{                                                      \
		Error( "pbrtInit() must be before calling \"%s()\". Ignoring.", func );       \
		return;                                            \
	}                                                      \
	else /* swallow trailing semicolon */
#define VERIFY_OPTIONS( func )                           \
	VERIFY_INITIALIZED( func );                          \
	if ( !( PbrtOptions.cat || PbrtOptions.toPly ) &&    \
		 currentApiState == APIState::WorldBlock )       \
	{                                                    \
		Error(                                           \
			"Options cannot be set inside world block; " \
			"\"%s\" not allowed.  Ignoring.",            \
			func );                                      \
		return;                                          \
	}                                                    \
	else /* swallow trailing semicolon */
#define VERIFY_WORLD( func )                                 \
	VERIFY_INITIALIZED( func );                              \
	if ( !( PbrtOptions.cat || PbrtOptions.toPly ) &&        \
		 currentApiState == APIState::OptionsBlock )         \
	{                                                        \
		Error(                                               \
			"Scene description must be inside world block; " \
			"\"%s\" not allowed. Ignoring.",                 \
			func );                                          \
		return;                                              \
	}                                                        \
	else /* swallow trailing semicolon */
#define FOR_ACTIVE_TRANSFORMS( expr )           \
	for ( int i = 0; i < MaxTransforms; ++i )   \
		if ( activeTransformBits & ( 1 << i ) ) \
		{                                       \
			expr                                \
		}
#if 1
#define WARN_IF_ANIMATED_TRANSFORM( func )
#else
// TODO
#define WARN_IF_ANIMATED_TRANSFORM( func )                           \
	do                                                               \
	{                                                                \
		if ( curTransform.IsAnimated() )                             \
			Warning(                                                 \
				"Animated transformations set; ignoring for \"%s\" " \
				"and using the start transform only",                \
				func );                                              \
	} while ( false ) /* swallow trailing semicolon */
#endif

// Object Creation Function Definitions
static HostMesh* MakeShapes( const std::string& name,
	const Transform* object2world,
	const Transform* world2object,
	bool reverseOrientation,
	const ParamSet& params,
	const int materialIdx )
{
	if (name == "plymesh")
		return CreatePLYMesh( object2world, world2object, reverseOrientation, params, materialIdx, &*graphicsState.floatTextures );
	else if (name == "trianglemesh")
		return CreateTriangleMeshShape( object2world, world2object, reverseOrientation, params, materialIdx, &*graphicsState.floatTextures );
	else Warning( "Shape \"%s\" unknown.", name.c_str() );
	return nullptr;
}

static HostMaterial* MakeMaterial( const std::string& name,
	const TextureParams& mp )
{
	HostMaterial* material = nullptr;
	if (name == "" || name == "none") return nullptr;
	else if (name == "matte") material = CreateMatteMaterial( mp );
	else if (name == "plastic") material = CreatePlasticMaterial( mp );
	// else if ( name == "translucent" ) material = CreateTranslucentMaterial( mp );
	else if (name == "glass") material = CreateGlassMaterial( mp );
	else if (name == "mirror") material = CreateMirrorMaterial( mp );
	// else if ( name == "hair" ) material = CreateHairMaterial( mp );
	else if (name == "disney") material = CreateDisneyMaterial( mp );
#if 0 // Not implemented
	else if (name == "mix")
	{
		std::string m1 = mp.FindString( "namedmaterial1", "" );
		std::string m2 = mp.FindString( "namedmaterial2", "" );
		HostMaterial* mat1, mat2;
		if (graphicsState.namedMaterials->find( m1 ) == graphicsState.namedMaterials->end())
		{
			Error( "Named material \"%s\" undefined.  Using \"matte\"", m1.c_str() );
			mat1 = MakeMaterial( "matte", mp );
		}
		else mat1 = (*graphicsState.namedMaterials)[m1]->material;
		if (graphicsState.namedMaterials->find( m2 ) == graphicsState.namedMaterials->end())
		{
			Error( "Named material \"%s\" undefined.  Using \"matte\"", m2.c_str() );
			mat2 = MakeMaterial( "matte", mp );
		}
		else mat2 = (*graphicsState.namedMaterials)[m2]->material;
		material = CreateMixMaterial( mp, mat1, mat2 );
	}
#endif
	else if (name == "metal") material = CreateMetalMaterial( mp );
	else if (name == "substrate") material = CreateSubstrateMaterial( mp );
	else if (name == "uber") material = CreateUberMaterial( mp );
#if 0 // Not implemented
	else if (name == "subsurface") material = CreateSubsurfaceMaterial( mp );
	else if (name == "kdsubsurface") material = CreateKdSubsurfaceMaterial( mp );
	else if (name == "fourier") material = CreateFourierMaterial( mp );
#endif
	else
	{
		Warning( "Material \"%s\" unknown. Using \"matte\".", name.c_str() );
		material = CreateMatteMaterial( mp );
	}

#if 0 // All existing implementations are pathtracers
	if ((name == "subsurface" || name == "kdsubsurface") &&
		(renderOptions->IntegratorName != "path" && (renderOptions->IntegratorName != "volpath")))
		Warning( "Subsurface scattering material \"%s\" used, but \"%s\" "
			"integrator doesn't support subsurface scattering. Use \"path\" or \"volpath\".",
			name.c_str(), renderOptions->IntegratorName.c_str() );
#endif
	mp.ReportUnused();
	if (!material) Error( "Unable to create material \"%s\"", name.c_str() );
	return material;
}

static HostMaterial::ScalarValue* CreateImageFloatTexture( const Transform& tex2world, const TextureParams& tp )
{
	return nullptr;
}

static HostMaterial::Vec3Value* CreateImageSpectrumTexture(
	const Transform& tex2world, const TextureParams& tp )
{
#if 0 // TODO
	// Initialize 2D texture mapping _map_ from _tp_
	std::unique_ptr<TextureMapping2D> map;
	std::string type = tp.FindString( "mapping", "uv" );
	if (type == "uv")
	{
		Float su = tp.FindFloat( "uscale", 1. );
		Float sv = tp.FindFloat( "vscale", 1. );
		Float du = tp.FindFloat( "udelta", 0. );
		Float dv = tp.FindFloat( "vdelta", 0. );
		map.reset( new UVMapping2D( su, sv, du, dv ) );
	}
	else if (type == "spherical") map.reset( new SphericalMapping2D( Inverse( tex2world ) ) );
	else if (type == "cylindrical") map.reset( new CylindricalMapping2D( Inverse( tex2world ) ) );
	else if (type == "planar")
		map.reset( new PlanarMapping2D( tp.FindVector3f( "v1", Vector3f( 1, 0, 0 ) ),
			tp.FindVector3f( "v2", Vector3f( 0, 1, 0 ) ),
			tp.FindFloat( "udelta", 0.f ), tp.FindFloat( "vdelta", 0.f ) ) );
	else
	{
		Error( "2D texture mapping \"%s\" unknown", type.c_str() );
		map.reset( new UVMapping2D );
	}
#endif

	// Initialize _ImageTexture_ parameters
#if 0 // TODO
	Float maxAniso = tp.FindFloat( "maxanisotropy", 8.f );
	bool trilerp = tp.FindBool( "trilinear", false );
	std::string wrap = tp.FindString( "wrap", "repeat" );
	ImageWrap wrapMode = ImageWrap::Repeat;
	if (wrap == "black") wrapMode = ImageWrap::Black;
	else if (wrap == "clamp") wrapMode = ImageWrap::Clamp;
	Float scale = tp.FindFloat( "scale", 1.f );
#endif
	std::string filename = tp.FindFilename( "filename" );
	bool gamma = tp.FindBool( "gamma", HasExtension( filename, ".tga" ) || HasExtension( filename, ".png" ) );
	int flags = HostTexture::FLIPPED;
	if (gamma) flags |= HostTexture::GAMMACORRECTION;
	int texId = HostScene::FindOrCreateTexture( filename, flags );
	auto texPtr = new HostMaterial::Vec3Value();
	texPtr->textureID = texId;
	return texPtr;
	// return new ImageTexture<RGBSpectrum, Spectrum>(
	// 	std::move( map ), filename, trilerp, maxAniso, wrapMode, scale, gamma );
	// HostScene::FindOrCreateTexture();
}

static HostMaterial::ScalarValue* CreateCheckerboardFloatTexture( const Transform& tex2world, const TextureParams& tp )
{
	return nullptr;
}

static HostMaterial::Vec3Value* CreateCheckerboardSpectrumTexture( const Transform& tex2world, const TextureParams& tp )
{
	return nullptr;
}

static HostMaterial::ScalarValue* MakeFloatTexture( const std::string& name,
	const Transform& tex2world, const TextureParams& tp )
{
	HostMaterial::ScalarValue* tex = nullptr;
	// if ( name == "constant" ) tex = CreateConstantFloatTexture( tex2world, tp );
	// else if ( name == "scale" ) tex = CreateScaleFloatTexture( tex2world, tp );
	// else if ( name == "mix" ) tex = CreateMixFloatTexture( tex2world, tp );
	// else if ( name == "bilerp" ) tex = CreateBilerpFloatTexture( tex2world, tp );
	// else
	if (name == "imagemap") tex = CreateImageFloatTexture( tex2world, tp );
	// else if ( name == "uv" ) tex = CreateUVFloatTexture( tex2world, tp );
	else if (name == "checkerboard") tex = CreateCheckerboardFloatTexture( tex2world, tp );
	// else if ( name == "dots" ) tex = CreateDotsFloatTexture( tex2world, tp );
	// else if ( name == "fbm" ) tex = CreateFBmFloatTexture( tex2world, tp );
	// else if ( name == "wrinkled" ) tex = CreateWrinkledFloatTexture( tex2world, tp );
	// else if ( name == "marble" ) tex = CreateMarbleFloatTexture( tex2world, tp );
	// else if ( name == "windy" ) tex = CreateWindyFloatTexture( tex2world, tp );
	// else if ( name == "ptex" ) tex = CreatePtexFloatTexture( tex2world, tp );
	else Warning( "Float texture \"%s\" unknown.", name.c_str() );
	tp.ReportUnused();
	// return DynamicHostTexture<Float> * ( tex );
	return tex;
}

static HostMaterial::Vec3Value* MakeSpectrumTexture(
	const std::string& name, const Transform& tex2world, const TextureParams& tp )
{
	HostMaterial::Vec3Value* tex = nullptr;
	// if ( name == "constant" ) tex = CreateConstantSpectrumTexture( tex2world, tp );
	// else if ( name == "scale" ) tex = CreateScaleSpectrumTexture( tex2world, tp );
	// else if ( name == "mix" ) tex = CreateMixSpectrumTexture( tex2world, tp );
	// else if ( name == "bilerp" ) tex = CreateBilerpSpectrumTexture( tex2world, tp );
	// else
	if (name == "imagemap") tex = CreateImageSpectrumTexture( tex2world, tp );
	// else if ( name == "uv" ) tex = CreateUVSpectrumTexture( tex2world, tp );
	// else if ( name == "checkerboard" ) tex = CreateCheckerboardSpectrumTexture( tex2world, tp );
	// else if ( name == "dots" ) tex = CreateDotsSpectrumTexture( tex2world, tp );
	// else if ( name == "fbm" ) tex = CreateFBmSpectrumTexture( tex2world, tp );
	// else if ( name == "wrinkled" ) tex = CreateWrinkledSpectrumTexture( tex2world, tp );
	// else if ( name == "marble" ) tex = CreateMarbleSpectrumTexture( tex2world, tp );
	// else if ( name == "windy" ) tex = CreateWindySpectrumTexture( tex2world, tp );
	// else if ( name == "ptex" ) tex = CreatePtexSpectrumTexture( tex2world, tp );
	else Warning( "Spectrum texture \"%s\" unknown.", name.c_str() );
	tp.ReportUnused();
	// return DynamicHostTexture<Spectrum> * ( tex );
	return tex;
}

void pbrtInit()
{
	Options opt;
	pbrtInit( opt );
}

void pbrtInit( const Options& opt )
{
	PbrtOptions = opt;
	// API Initialization
	if (currentApiState != APIState::Uninitialized) Error( "pbrtInit() has already been called." );
	currentApiState = APIState::OptionsBlock;
	// renderOptions.reset( new RenderOptions );
	graphicsState = GraphicsState();
	catIndentCount = 0;
	// General \pbrt Initialization
	SampledSpectrum::Init();
}

void pbrtCleanup()
{
	// API Cleanup
	if (currentApiState == APIState::Uninitialized) Error( "pbrtCleanup() called without pbrtInit()." );
	else if (currentApiState == APIState::WorldBlock) Error( "pbrtCleanup() called while inside world block." );
	currentApiState = APIState::Uninitialized;
}

void pbrtIdentity()
{
	VERIFY_INITIALIZED( "Identity" );
	FOR_ACTIVE_TRANSFORMS( curTransform[i] = Transform(); )
		if (PbrtOptions.cat || PbrtOptions.toPly) printf( "%*sIdentity\n", catIndentCount, "" );
}

void pbrtTranslate( Float dx, Float dy, Float dz )
{
	VERIFY_INITIALIZED( "Translate" );
	FOR_ACTIVE_TRANSFORMS( curTransform[i] = curTransform[i] * mat4::Translate( dx, dy, dz ); )
		if (PbrtOptions.cat || PbrtOptions.toPly)
			printf( "%*sTranslate %.9g %.9g %.9g\n", catIndentCount, "", dx, dy,
				dz );
}

void pbrtRotate( Float angle, Float dx, Float dy, Float dz )
{
	VERIFY_INITIALIZED( "Rotate" );
	auto axis = normalize( make_float3( dx, dy, dz ) );
	FOR_ACTIVE_TRANSFORMS( curTransform[i] = curTransform[i] * mat4::Rotate( axis, Radians( angle ) ); )
		if (PbrtOptions.cat || PbrtOptions.toPly)
			printf( "%*sRotate %.9g %.9g %.9g %.9g\n", catIndentCount, "", angle, dx, dy, dz );
}

void pbrtScale( Float sx, Float sy, Float sz )
{
	VERIFY_INITIALIZED( "Scale" );
	FOR_ACTIVE_TRANSFORMS( curTransform[i] = curTransform[i] * mat4::Scale( make_float3( sx, sy, sz ) ); )
		if (PbrtOptions.cat || PbrtOptions.toPly) 
			printf( "%*sScale %.9g %.9g %.9g\n", catIndentCount, "", sx, sy, sz );
}

void pbrtLookAt( Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
	Float ux, Float uy, Float uz )
{
	VERIFY_INITIALIZED( "LookAt" );
	Transform lookAt =
		mat4::LookAt( make_float3( ex, ey, ez ), make_float3( lx, ly, lz ), make_float3( ux, uy, uz ) );
	FOR_ACTIVE_TRANSFORMS( curTransform[i] = curTransform[i] * lookAt; );
	if (PbrtOptions.cat || PbrtOptions.toPly)
		printf( "%*sLookAt %.9g %.9g %.9g\n%*s%.9g %.9g %.9g\n%*s%.9g %.9g %.9g\n",
			catIndentCount, "", ex, ey, ez, catIndentCount + 8, "", lx, ly, lz,
			catIndentCount + 8, "", ux, uy, uz );
}

void pbrtConcatTransform( Float tr[16] )
{
	VERIFY_INITIALIZED( "ConcatTransform" );
	FOR_ACTIVE_TRANSFORMS(
		curTransform[i] = curTransform[i] *
		Transform( mat4{ tr[0], tr[4], tr[8], tr[12], tr[1], tr[5],
						 tr[9], tr[13], tr[2], tr[6], tr[10], tr[14],
						 tr[3], tr[7], tr[11], tr[15] } ); )
		if (PbrtOptions.cat || PbrtOptions.toPly)
		{
			printf( "%*sConcatTransform [ ", catIndentCount, "" );
			for (int i = 0; i < 16; ++i) printf( "%.9g ", tr[i] );
			printf( " ]\n" );
		}
}

void pbrtTransform( Float tr[16] )
{
	VERIFY_INITIALIZED( "Transform" );
	FOR_ACTIVE_TRANSFORMS(
		curTransform[i] = Transform( mat4{
			tr[0], tr[4], tr[8], tr[12], tr[1], tr[5], tr[9], tr[13], tr[2],
			tr[6], tr[10], tr[14], tr[3], tr[7], tr[11], tr[15] } ); )
			if (PbrtOptions.cat || PbrtOptions.toPly)
			{
				printf( "%*sTransform [ ", catIndentCount, "" );
				for (int i = 0; i < 16; ++i) printf( "%.9g ", tr[i] );
				printf( " ]\n" );
			}
}

void pbrtCoordinateSystem( const std::string& ) { Warning( "pbrtCoordinateSystem is not implemented!" ); }
void pbrtCoordSysTransform( const std::string& ) { Warning( "pbrtCoordSysTransform is not implemented!" ); }
void pbrtActiveTransformAll() { Warning( "pbrtActiveTransformAll is not implemented!" ); }
void pbrtActiveTransformEndTime() { Warning( "pbrtActiveTransformEndTime is not implemented!" ); }
void pbrtActiveTransformStartTime() { Warning( "pbrtActiveTransformStartTime is not implemented!" ); }
void pbrtTransformTimes( Float start, Float end ) { Warning( "pbrtTransformTimes is not implemented!" ); }
void pbrtPixelFilter( const std::string& name, const ParamSet& params ) { Warning( "pbrtPixelFilter is not implemented!" ); }
void pbrtFilm( const std::string& type, const ParamSet& params ) { Warning( "pbrtFilm is not implemented!" ); }
void pbrtSampler( const std::string& name, const ParamSet& params ) { Warning( "pbrtSampler is not implemented!" ); }
void pbrtAccelerator( const std::string& name, const ParamSet& params ) { Warning( "pbrtAccelerator is not implemented!" ); }
void pbrtIntegrator( const std::string& name, const ParamSet& params ) { Warning( "pbrtIntegrator is not implemented!" ); }

void pbrtCamera( const std::string& name, const ParamSet& params )
{
	VERIFY_OPTIONS( "Camera" );
	auto CameraName = name;
	auto CameraParams = params;
	auto CameraToWorld = Inverse( curTransform );
	namedCoordinateSystems["camera"] = CameraToWorld;
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sCamera \"%s\" ", catIndentCount, "", name.c_str() );
		params.Print( catIndentCount );
		printf( "\n" );
	}
	// HostScene::camera->position = make_float3( CameraToWorld[0] * make_float4( 0, 0, 0, 1 ) );
	// HostScene::camera->direction = make_float3( CameraToWorld[0] * make_float4( 0, 0, 1, 0 ) );
	// HostScene::camera->transform = CameraToWorld; TODO
	HostScene::camera->FOV = params.FindOneFloat( "fov", 90.f );
	HostScene::camera->focalDistance = params.FindOneFloat( "focaldistance", 1e6f );
	// This should be `aperturediameter', but is hardly ever used.
	HostScene::camera->aperture = params.FindOneFloat( "lensradius", 0.f );
	// Reset distortion, PBRT does not pass any such information:
	HostScene::camera->distortion = 0.f;
}

void pbrtMakeNamedMedium( const std::string& name, const ParamSet& params ) { Warning( "pbrtMakeNamedMedium is not implemented!" ); }
void pbrtMediumInterface( const std::string& insideName, const std::string& outsideName ) { Warning( "pbrtMediumInterface is not implemented!" ); }

void pbrtWorldBegin()
{
	VERIFY_OPTIONS( "WorldBegin" );
	currentApiState = APIState::WorldBlock;
	for (int i = 0; i < MaxTransforms; ++i) curTransform[i] = Transform();
	activeTransformBits = AllTransformsBits;
	namedCoordinateSystems["world"] = curTransform;
	if (PbrtOptions.cat || PbrtOptions.toPly) printf( "\n\nWorldBegin\n\n" );
}

void pbrtAttributeBegin()
{
	VERIFY_WORLD( "AttributeBegin" );
	pushedGraphicsStates.push_back( graphicsState );
	graphicsState.floatTexturesShared = graphicsState.float3TexturesShared = graphicsState.namedMaterialsShared = true;
	pushedTransforms.push_back( curTransform );
	pushedActiveTransformBits.push_back( activeTransformBits );
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "\n%*sAttributeBegin\n", catIndentCount, "" );
		catIndentCount += 4;
	}
}

void pbrtAttributeEnd()
{
	VERIFY_WORLD( "AttributeEnd" );
	if (!pushedGraphicsStates.size())
	{
		Error( "Unmatched pbrtAttributeEnd() encountered. Ignoring it." );
		return;
	}
	graphicsState = std::move( pushedGraphicsStates.back() );
	pushedGraphicsStates.pop_back();
	curTransform = pushedTransforms.back();
	pushedTransforms.pop_back();
	activeTransformBits = pushedActiveTransformBits.back();
	pushedActiveTransformBits.pop_back();
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		catIndentCount -= 4;
		printf( "%*sAttributeEnd\n", catIndentCount, "" );
	}
}

void pbrtTransformBegin()
{
	VERIFY_WORLD( "TransformBegin" );
	pushedTransforms.push_back( curTransform );
	pushedActiveTransformBits.push_back( activeTransformBits );
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sTransformBegin\n", catIndentCount, "" );
		catIndentCount += 4;
	}
}

void pbrtTransformEnd()
{
	VERIFY_WORLD( "TransformEnd" );
	if (!pushedTransforms.size())
	{
		Error( "Unmatched pbrtTransformEnd() encountered. Ignoring it." );
		return;
	}
	curTransform = pushedTransforms.back();
	pushedTransforms.pop_back();
	activeTransformBits = pushedActiveTransformBits.back();
	pushedActiveTransformBits.pop_back();
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		catIndentCount -= 4;
		printf( "%*sTransformEnd\n", catIndentCount, "" );
	}
}

void pbrtTexture( const std::string& name, const std::string& type, const std::string& texname, const ParamSet& params )
{
	VERIFY_WORLD( "Texture" );
	TextureParams tp( params, params, *graphicsState.floatTextures, *graphicsState.float3Textures );
	if (type == "float")
	{
		// Create _Float_ texture and store in _floatTextures_
		if (graphicsState.floatTextures->find( name ) != graphicsState.floatTextures->end()) Warning( "Texture \"%s\" being redefined", name.c_str() );
		WARN_IF_ANIMATED_TRANSFORM( "Texture" );
		auto ft = MakeFloatTexture( texname, curTransform[0], tp );
		if (ft)
		{
			// TODO: move this to be a GraphicsState method, also don't
			// provide direct floatTextures access?
			if (graphicsState.floatTexturesShared)
			{
				graphicsState.floatTextures = std::make_shared<GraphicsState::FloatTextureMap>( *graphicsState.floatTextures );
				graphicsState.floatTexturesShared = false;
			}
			(*graphicsState.floatTextures)[name] = ft;
		}
	}
	else if (type == "color" || type == "spectrum")
	{
		// Create _color_ texture and store in _spectrumTextures_
		if (graphicsState.float3Textures->find( name ) != graphicsState.float3Textures->end())
			Warning( "Texture \"%s\" being redefined", name.c_str() );
		WARN_IF_ANIMATED_TRANSFORM( "Texture" );
		auto st = MakeSpectrumTexture( texname, curTransform[0], tp );
		if (st)
		{
			if (graphicsState.float3TexturesShared)
			{
				graphicsState.float3Textures = std::make_shared<GraphicsState::Float3TextureMap>( *graphicsState.float3Textures );
				graphicsState.float3TexturesShared = false;
			}
			(*graphicsState.float3Textures)[name] = st;
		}
	}
	else Error( "Texture type \"%s\" unknown.", type.c_str() );
}

void pbrtMaterial( const std::string& name, const ParamSet& params )
{
	ParamSet emptyParams;
	TextureParams mp( params, emptyParams, *graphicsState.floatTextures, *graphicsState.float3Textures );
	auto mtl = MakeMaterial( name, mp );
	graphicsState.currentMaterial = std::make_shared<MaterialInstance>( name, mtl, params );

	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sMaterial \"%s\" ", catIndentCount, "", name.c_str() );
		params.Print( catIndentCount );
		printf( "\n" );
	}
}

void pbrtMakeNamedMaterial( const std::string& name, const ParamSet& params )
{
	ParamSet emptyParams;
	TextureParams mp( params, emptyParams, *graphicsState.floatTextures, *graphicsState.float3Textures );
	std::string matName = mp.FindString( "type" );
	WARN_IF_ANIMATED_TRANSFORM( "MakeNamedMaterial" );
	if (matName == "") Error( "No parameter string \"type\" found in MakeNamedMaterial" );
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sMakeNamedMaterial \"%s\" ", catIndentCount, "", name.c_str() );
		params.Print( catIndentCount );
		printf( "\n" );
	}
	else
	{
		auto mtl = MakeMaterial( matName, mp );
		if (graphicsState.namedMaterials->find( name ) != graphicsState.namedMaterials->end())
			Warning( "Named material \"%s\" redefined.", name.c_str() );
		if (graphicsState.namedMaterialsShared)
		{
			graphicsState.namedMaterials = std::make_shared<GraphicsState::NamedMaterialMap>( *graphicsState.namedMaterials );
			graphicsState.namedMaterialsShared = false;
		}
		(*graphicsState.namedMaterials)[name] = std::make_shared<MaterialInstance>( matName, mtl, params );
	}
}

void pbrtNamedMaterial( const std::string& name )
{
	VERIFY_WORLD( "NamedMaterial" );
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sNamedMaterial \"%s\"\n", catIndentCount, "", name.c_str() );
		return;
	}
	auto iter = graphicsState.namedMaterials->find( name );
	if (iter == graphicsState.namedMaterials->end())
	{
		Error( "NamedMaterial \"%s\" unknown.", name.c_str() );
		return;
	}
	graphicsState.currentMaterial = iter->second;
}

void pbrtLightSource( const std::string& name, const ParamSet& params )
{
	if (name == "point")
	{
		const auto I = params.FindOneSpectrum( "I", Spectrum( 1.0 ) );
		const auto sc = params.FindOneSpectrum( "scale", Spectrum( 1.0 ) );
		const auto P = params.FindOnePoint3f( "from", make_float3( 0, 0, 0 ) );
		// Instead of creating a translation matrix from world to light,
		// transform the point to the desired position in space:
		const auto light2world = curTransform[0];
		const auto pos = light2world.TransformPoint( P );
		HostScene::AddPointLight( pos, (I * sc).vector() );
	}
	else if (name == "spot")
	{
		const auto I = params.FindOneSpectrum( "I", Spectrum( 1.0 ) );
		const auto sc = params.FindOneSpectrum( "scale", Spectrum( 1.0 ) );
		const auto coneangle = params.FindOneFloat( "coneangle", 30. );
		const auto conedelta = params.FindOneFloat( "conedeltaangle", 5. );
		// Compute spotlight world to light transformation
		const auto from = params.FindOnePoint3f( "from", make_float3( 0, 0, 0 ) );
		const auto to = params.FindOnePoint3f( "to", make_float3( 0, 0, 1 ) );
		const float3 dir = normalize( to - from );

		// PBRT Uses a transformation instead of separate from and direction:
		// float3 du, dv;
		// CoordinateSystem( dir, du, dv );
		// TBN:
		// mat4 dirToZ{du.x, du.y, du.z, 0.,
		// 			dv.x, dv.y, dv.z, 0.,
		// 			dir.x, dir.y, dir.z, 0.,
		// 			0, 0, 0, 1.};
		// const auto light2world = curTransform[0] * mat4::Translate( from ) * dirToZ.Inverted();

		// ERROR:
		// https://www.pbrt.org/fileformat-v3.html#lights specifies for _coneangle_:
		// > The angle that the spotlight's cone makes with its primary axis.
		// > For directions up to this angle from the main axis, the full radiant intensity
		// > given by "I" is emitted. After this angle and up to "coneangle" + "conedeltaangle",
		// > illumination falls off until it is zero.
		// HOWEVER:
		// The code computes and assigns the following:
		const auto totalWidth = coneangle;
		const auto falloffStart = coneangle - conedelta;
		// If the documentation is to be believed, this should instead be:
		// const auto totalWidth = coneangle + conedelta;
		// const auto falloffStart = coneangle;
		// This is perhaps worth filing an issue for.
		HostScene::AddSpotLight( from, dir, std::cos( Radians( falloffStart ) ), std::cos( Radians( totalWidth ) ), (I * sc).vector() );
	}
	else if (name == "distant")
	{
		const auto L = params.FindOneSpectrum( "L", Spectrum( 1.0 ) );
		const auto sc = params.FindOneSpectrum( "scale", Spectrum( 1.0 ) );
		const auto from = params.FindOnePoint3f( "from", make_float3( 0, 0, 0 ) );
		const auto to = params.FindOnePoint3f( "to", make_float3( 0, 0, 1 ) );
		const auto light2world = curTransform[0];
		// WARNING: In PBRT the dir vector points _towards_ the light,
		// while in LH2 this represents the direction the light is pointed towards
		const auto dir = normalize( light2world.TransformVector( to - from ) );
		HostScene::AddDirectionalLight( dir, (L * sc).vector() );
	}
	else if (name == "infinite" || name == "exinfinite")
	{
		Spectrum L = params.FindOneSpectrum( "L", Spectrum( 1.0 ) );
		Spectrum sc = params.FindOneSpectrum( "scale", Spectrum( 1.0 ) );
		std::string texmap = params.FindOneFilename( "mapname", "" );
		// int nSamples = params.FindOneInt( "samples",
		// 								  params.FindOneInt( "nsamples", 1 ) );
		auto sd = new HostSkyDome();
		const auto light2world = curTransform[0];
		sd->worldToLight = light2world.Inverted();
		sd->Load( texmap.c_str(), (L * sc).vector() );
		HostScene::SetSkyDome( sd );
	}
	// TODO: Implement other light types
	else Error( "LightSource: light type \"%s\" unknown.", name.c_str() );
}

void pbrtAreaLightSource( const std::string& name, const ParamSet& params )
{
	VERIFY_WORLD( "AreaLightSource" );
	graphicsState.areaLight = name;
	graphicsState.areaLightParams = params;
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sAreaLightSource \"%s\" ", catIndentCount, "", name.c_str() );
		params.Print( catIndentCount );
		printf( "\n" );
	}
}

void pbrtShape( const std::string& name, const ParamSet& params )
{
	VERIFY_WORLD( "Shape" );
	// std::vector<std::shared_ptr<Primitive>> prims;
	// std::vector<std::shared_ptr<AreaLight>> areaLights;
	if (PbrtOptions.cat || (PbrtOptions.toPly && name != "trianglemesh"))
	{
		printf( "%*sShape \"%s\" ", catIndentCount, "", name.c_str() );
		params.Print( catIndentCount );
		printf( "\n" );
	}
	if (curTransform.IsAnimated()) Error( "No animated transform loader yet!" );
	// Possibly create area light for shape
	int materialIdx;
	if (graphicsState.areaLight != "")
	{
		// area = MakeAreaLight( graphicsState.areaLight, curTransform[0],
		//                                        mi, graphicsState.areaLightParams, s );
		const auto& lparams = graphicsState.areaLightParams;
		const auto L = lparams.FindOneSpectrum( "L", Spectrum( 1.0 ) ).vector();
		const auto sc = lparams.FindOneSpectrum( "scale", Spectrum( 1.0 ) ).vector();
		const bool twoSided = lparams.FindOneBool( "twosided", false );
		auto mtl = new HostMaterial();
		mtl->color = L * sc;
		// Sanity, ensure the material is emissive within LH2 definitions:
		if (!mtl->IsEmissive()) Error( "None of the rgb components are larger than 1, material is not emissive!" );
		if (twoSided) mtl->flags |= HostMaterial::EMISSIVE_TWOSIDED;
		materialIdx = HostScene::AddMaterial( mtl );
	}
	else
	{
		auto mtl = graphicsState.GetMaterialForShape( params );
		materialIdx = HostScene::AddMaterial( mtl );
	}
	// Initialize _prims_ and _areaLights_ for static shape
	// Create shapes for shape _name_
	// Transform* ObjToWorld = transformCache.Lookup( curTransform[0] );
	// Transform* WorldToObj = transformCache.Lookup( Inverse( curTransform[0] ) );
	Transform ObjToWorld = curTransform[0];
	Transform WorldToObj = curTransform[0].Inverted();
	auto hostMesh = MakeShapes( name, &ObjToWorld, &WorldToObj,
		graphicsState.reverseOrientation, params, materialIdx );
	// if ( shapes.empty() ) return;
	params.ReportUnused();
	if (!hostMesh)
	{
		Warning( "No mesh created for %s", name.c_str() );
		return;
	}
	// TODO: Medium and area lights
#if 0
	MediumInterface mi = graphicsState.CreateMediumInterface();
	prims.reserve( shapes.size() );
	prims.push_back(
		std::make_shared<GeometricPrimitive>( s, mtl, area, mi ) );
#endif
	auto meshIdx = HostScene::AddMesh( hostMesh );
	HostNode* newNode = new HostNode( meshIdx, ObjToWorld );
	// Add _prims_ and _areaLights_ to scene or current instance
	if (currentInstance)
		// Abusing HostNode instance (this is not an instance of the mesh _yet_)
		// because it keeps track of the transform.
		currentInstance->push_back( newNode );
	else HostScene::AddInstance( newNode );
}

// Attempt to determine if the ParamSet for a shape may provide a value for
// its material's parameters. Unfortunately, materials don't provide an
// explicit representation of their parameters that we can query and
// cross-reference with the parameter values available from the shape.
//
// Therefore, we'll apply some "heuristics".
bool shapeMaySetMaterialParameters( const ParamSet& ps )
{
	for (const auto& param : ps.textures)
		// Any texture other than one for an alpha mask is almost certainly
		// for a Material (or is unused!).
		if (param->name != "alpha" && param->name != "shadowalpha") return true;
	// Special case spheres, which are the most common non-mesh primitive.
	for (const auto& param : ps.floats) if (param->nValues == 1 && param->name != "radius") return true;
	// Extra special case strings, since plymesh uses "filename", curve "type",
	// and loopsubdiv "scheme".
	for (const auto& param : ps.strings)
		if (param->nValues == 1 && param->name != "filename" && param->name != "type" && param->name != "scheme") return true;
	// For all other parameter types, if there is a single value of the
	// parameter, assume it may be for the material. This should be valid
	// (if conservative), since no materials currently take array
	// parameters.
	for (const auto& param : ps.bools) if (param->nValues == 1) return true;
	for (const auto& param : ps.ints) if (param->nValues == 1) return true;
	for (const auto& param : ps.point2fs) if (param->nValues == 1) return true;
	for (const auto& param : ps.vector2fs) if (param->nValues == 1) return true;
	for (const auto& param : ps.point3fs) if (param->nValues == 1) return true;
	for (const auto& param : ps.vector3fs) if (param->nValues == 1) return true;
	for (const auto& param : ps.normals) if (param->nValues == 1) return true;
	for (const auto& param : ps.spectra) if (param->nValues == 1) return true;
	return false;
}

HostMaterial* GraphicsState::GetMaterialForShape( const ParamSet& shapeParams )
{
	CHECK( currentMaterial );
	if (shapeMaySetMaterialParameters( shapeParams ))
	{
		// Only create a unique material for the shape if the shape's
		// parameters are (apparently) going to provide values for some of
		// the material parameters.
		TextureParams mp( shapeParams, currentMaterial->params, *floatTextures, *float3Textures );
		return MakeMaterial( currentMaterial->name, mp );
	}
	else return currentMaterial->material;
}

void pbrtReverseOrientation()
{
	Warning( "pbrtReverseOrientation is not implemented!" );
}

void pbrtObjectBegin( const std::string& name )
{
	VERIFY_WORLD( "ObjectBegin" );
	pbrtAttributeBegin();
	if (currentInstance) Error( "ObjectBegin called inside of instance definition" );
	instances[name] = std::vector<HostNode*>();
	currentInstance = &instances[name];
	if (PbrtOptions.cat || PbrtOptions.toPly) printf( "%*sObjectBegin \"%s\"\n", catIndentCount, "", name.c_str() );
}

void pbrtObjectEnd()
{
	VERIFY_WORLD( "ObjectEnd" );
	if (!currentInstance) Error( "ObjectEnd called outside of instance definition" );
	if (PbrtOptions.cat || PbrtOptions.toPly) printf( "%*sObjectEnd\n", catIndentCount, "" );
	currentInstance = nullptr;
	pbrtAttributeEnd();
}

void pbrtObjectInstance( const std::string& name )
{
	VERIFY_WORLD( "ObjectInstance" );
	if (PbrtOptions.cat || PbrtOptions.toPly)
	{
		printf( "%*sObjectInstance \"%s\"\n", catIndentCount, "", name.c_str() );
		return;
	}
	// Perform object instance error checking
	if (currentInstance)
	{
		Error( "ObjectInstance can't be called inside instance definition" );
		return;
	}
	if (instances.find( name ) == instances.end())
	{
		Error( "Unable to find instance named \"%s\"", name.c_str() );
		return;
	}
	auto& in = instances[name];
	if (in.empty()) return;
	// static_assert( MaxTransforms == 2,
	// 			   "TransformCache assumes only two transforms" );
	// Create _animatedInstanceToWorld_ transform for instance
	// Transform* InstanceToWorld[2] = {
	// 	transformCache.Lookup( curTransform[0] ),
	// 	transformCache.Lookup( curTransform[1] )};
	// AnimatedTransform animatedInstanceToWorld(
	// 	InstanceToWorld[0], transformStartTime,
	// 	InstanceToWorld[1], transformEndTime );
	// std::shared_ptr<Primitive> prim(
	// 	std::make_shared<TransformedPrimitive>( in[0], animatedInstanceToWorld ) );
	// primitives.push_back( prim );
	for (auto& mesh : in) HostScene::AddInstance( new HostNode( *mesh ) );
}

void pbrtWorldEnd()
{
	VERIFY_WORLD( "WorldEnd" );
	// Ensure there are no pushed graphics states
	while (pushedGraphicsStates.size())
	{
		Warning( "Missing end to pbrtAttributeBegin()" );
		pushedGraphicsStates.pop_back();
		pushedTransforms.pop_back();
	}
	while (pushedTransforms.size())
	{
		Warning( "Missing end to pbrtTransformBegin()" );
		pushedTransforms.pop_back();
	}
	graphicsState = GraphicsState();
	// transformCache.Clear();
	currentApiState = APIState::OptionsBlock;
	// ImageTexture<Float, Float>::ClearCache();
	// ImageTexture<RGBSpectrum, Spectrum>::ClearCache();
	// renderOptions.reset( new RenderOptions );
	for (int i = 0; i < MaxTransforms; ++i) curTransform[i] = Transform();
	activeTransformBits = AllTransformsBits;
	namedCoordinateSystems.erase( namedCoordinateSystems.begin(), namedCoordinateSystems.end() );
}

};