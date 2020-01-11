#pragma once

#include "Storage.h"

namespace deviceMaterials
{

enum /* class */ BxDFType : int
{
	BSDF_REFLECTION = 1 << 0,
	BSDF_TRANSMISSION = 1 << 1,
	BSDF_DIFFUSE = 1 << 2,
	BSDF_GLOSSY = 1 << 3,
	BSDF_SPECULAR = 1 << 4,
	BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
			   BSDF_TRANSMISSION,
	BSDF_ALL_EXCEPT_SPECULAR = BSDF_ALL & ~BSDF_SPECULAR,
};

enum class TransportMode
{
	Radiance,
	Importance,
};

class MaterialIntf : public HasPlacementNewOperator
{
  public:
	__device__ virtual void Setup(
		const float3 D,									   // IN:	incoming ray direction, used for consistent normals
		const float u, const float v,					   //		barycentric coordinates of intersection point
		const float coneWidth,							   //		ray cone width, for texture LOD
		const CoreTri4& tri,							   //		triangle data
		const int instIdx,								   //		instance index, for normal transform
		const int materialInstance,						   //		Material instance id/location
		float3& N, float3& iN, float3& fN,				   //		geometric normal, interpolated normal, final normal (normal mapped)
		float3& T,										   //		tangent vector
		const float waveLength = -1.0f,					   // IN:	wavelength (optional)
		const bool allowMultipleLobes = true,			   // IN:	Integrator samples multiple lobes (optional)
		const TransportMode mode = TransportMode::Radiance // IN:	Mode based on integrator (optional)
		) = 0;

	__device__ virtual void DisableTransmittance()
	{
		// Nothing at the moment
	}

	__device__ virtual bool IsEmissive() const = 0;
	__device__ virtual bool IsEmissiveTwosided() const { return false; }
	__device__ virtual bool IsAlpha() const = 0;
	/**
	 * Used to retrieve color for emissive surfaces.
	 */
	__device__ virtual float3 Color() const = 0;

	__device__ virtual float3 Evaluate( const float3 iN, const float3 T,
										const float3 woWorld, const float3 wiWorld,
										const BxDFType flags,
										float& pdf ) const = 0;
	__device__ virtual float3 Sample( float3 iN, const float3 N, const float3 T,
									  const float3 woWorld, const float distance,
									  const float r3, const float r4,
									  const BxDFType flags,
									  float3& wiWorld, float& pdf,
									  BxDFType& sampledType ) const = 0;
};

#include "material_bsdf_stack.h"

#include "material_disney.h"
#include "pbrt/materials.h"

template <MaterialType _MaterialType, typename _type>
struct Case
{
	static constexpr auto MaterialType = _MaterialType;
	using type = _type;
};

template <typename... Cases>
struct MaterialSwitch
{
	using MaterialStoreReq = StorageRequirement<typename Cases::type...>;
	using MaterialStore = typename MaterialStoreReq::type;

	__device__ static MaterialIntf* run( MaterialStore inplace, const MaterialType type )
	{
#if 1
		// initializer_list approach is pretty much as optimal as a hardcoded
		// switch-case. std::max - while more readable - produces less optimal PTX
		// https://cuda.godbolt.org/z/MxVcJY
		MaterialIntf* res = nullptr;
		std::initializer_list<MaterialIntf*>( {( type == Cases::MaterialType ? res = new ( inplace ) typename Cases::type() : nullptr )...} );
		return res;
#else
		return std::max( {( type == Cases::MaterialType ? (MaterialIntf*)new ( inplace ) typename Cases::type() : nullptr )...} );
#endif
	}
};

/**
 * List of supported materials and their enumeration value.
 *
 * If you add a new material, _list it here_.
 */
using Materials = MaterialSwitch<
#if 1
	// Implement the gltf-extracted material through PBRT BxDFs
	Case<MaterialType::DISNEY, pbrt::DisneyGltf>,
#else
	Case<MaterialType::DISNEY, DisneyMaterial>,
#endif
	Case<MaterialType::PBRT_DISNEY, pbrt::Disney>,
	Case<MaterialType::PBRT_GLASS, pbrt::Glass>,
	Case<MaterialType::PBRT_MATTE, pbrt::Matte>,
	Case<MaterialType::PBRT_METAL, pbrt::Metal>,
	Case<MaterialType::PBRT_MIRROR, pbrt::Mirror>,
	Case<MaterialType::PBRT_PLASTIC, pbrt::Plastic>,
	Case<MaterialType::PBRT_SUBSTRATE, pbrt::Substrate>,
	Case<MaterialType::PBRT_UBER, pbrt::Uber>,
	>;

using MaterialStore = Materials::MaterialStore;

// NOTE: Materialstore is a pointer-type (array) by design
static_assert( std::is_array<MaterialStore>::value, "MaterialStore must be an array" );
static_assert( std::is_pointer<std::decay_t<MaterialStore>>::value, "Decayed material store must be an array" );

LH2_DEVFUNC MaterialIntf* GetMaterial( MaterialStore inplace, const CoreMaterialDesc& matDesc )
{
	// Evaluate templated switch case
	return Materials::run( inplace, matDesc.type );
}
}; // namespace deviceMaterials
