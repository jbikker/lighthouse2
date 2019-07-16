// Lambert BSDF
// ----------------------------------------------------------------

// for debugging: Lambert brdf
__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, float& pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.baseColor * INVPI;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3 N, const float3 T, const float3 wo, 
	uint& seed, float3& wi, float& pdf )
{
	// specular and diffuse
	if (fabs( ROUGHNESS ) < 0.1f)
	{
		// pure specular
		wi = -reflect( wo, iN );
		pdf = 1;
		APPLYSAFENORMALS;
		return shadingData.baseColor * (1.0f / abs( dot( iN, wi ) ));
	}
	else
	{
		wi = normalize( Tangent2World( DiffuseReflectionCosWeighted( RandomFloat( seed ), RandomFloat( seed ) ), iN ) );
		pdf = max( 0.0f, dot( wi, iN ) ) * INVPI;
		APPLYSAFENORMALS;
		return shadingData.baseColor * INVPI;
	}
}

// EOF