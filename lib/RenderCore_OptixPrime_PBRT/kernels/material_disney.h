class DisneyMaterial : public MaterialIntf
{
	ShadingData shadingData;

  public:
	__device__ void Setup(
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
		) override
	{
		GetShadingData( D, u, v, coneWidth, tri, instIdx,
						shadingData, N, iN, fN, T, waveLength );
	}

	__device__ void DisableTransmittance() override
	{
		shadingData.transmittance = make_float3( 0 );
	}

	__device__ bool IsEmissive() const override
	{
		return shadingData.IsEmissive();
	}

	__device__ bool IsEmissiveTwosided() const override
	{
		return shadingData.flags &  ShadingDataFlags::EMISSIVE_TWOSIDED;
	}

	__device__ bool IsAlpha() const override
	{
		return shadingData.flags & ShadingDataFlags::ALPHA;
	}

	__device__ float3 Color() const override
	{
		return shadingData.color;
	}

	__device__ float3 Evaluate( const float3 iN, const float3 T,
								const float3 wo, const float3 wi,
								const BxDFType flags,
								float& pdf ) const override
	{
		return EvaluateBSDF( shadingData, iN, T, wo, wi, pdf );
		// pdf = fabs( dot( wi, iN ) ) * INVPI;
		// return shadingData.color * INVPI;
	}

	__device__ float3 Sample( float3 iN, const float3 N, const float3 T,
							  const float3 wo, const float distance,
							  const float r3, const float r4, const float r5,
							  const BxDFType flags,
							  float3& wi, float& pdf,
							  BxDFType& sampledType ) const override
	{
		bool specular = false;
		auto f = SampleBSDF( shadingData, iN, N, T, wo, distance, r3, r4, r5, wi, pdf, specular );

		sampledType = specular
						  ? BxDFType::BSDF_SPECULAR
						  /* TODO: Unknown, only specular is necessary: */
						  : BxDFType( 0 );

		return f;
	}
};
