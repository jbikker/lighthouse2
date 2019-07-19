// Lambert BSDF
// ----------------------------------------------------------------

// for debugging: Lambert brdf
__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, float& pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.color * INVPI;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3 N, const float3 T, const float3 wo, 
	const float r0, const float r1, float3& wi, float& pdf )
{
	// specular and diffuse
	if (fabs( ROUGHNESS ) < 0.1f)
	{
		// pure specular
		wi = -reflect( wo, iN );
		pdf = 1;
		APPLYSAFENORMALS;
		return shadingData.color * (1.0f / abs( dot( iN, wi ) ));
	}
	else
	{
		wi = normalize( Tangent2World( DiffuseReflectionCosWeighted( r0, r1 ), iN ) );
		pdf = max( 0.0f, dot( wi, iN ) ) * INVPI;
		APPLYSAFENORMALS;
		return shadingData.color * INVPI;
	}
}

// EOF