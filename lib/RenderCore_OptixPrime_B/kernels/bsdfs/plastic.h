__device__ inline float fresnelDielectric( float3 i, float3 m, float eta )
{
	float result = 1.0f;
	float cosThetaI = abs( dot( i, m ) );
	float sinThetaOSquared = (eta * eta) * (1.0f - cosThetaI * cosThetaI);
	if (sinThetaOSquared <= 1.0)
	{
		float cosThetaO = sqrt( saturate( 1.0f - sinThetaOSquared ) );
		float Rs = (cosThetaI - eta * cosThetaO) / (cosThetaI + eta * cosThetaO);
		float Rp = (eta * cosThetaI - cosThetaO) / (eta * cosThetaI + cosThetaO);
		result = 0.5f * (Rs * Rs + Rp * Rp);
	}
	return result;
}

__device__ inline SampledMaterial evaluate( const Material& material, float3 nO, float3 wI, float3 wO )
{
	SampledMaterial result;
	// ...
	float3 m = normalize( wO - wI );
	float NdotM = dot( nO, m );
	float MdotO = dot( m, wO );
	float a = material.roughness * material.roughness;
	float F = fresnelDielectric( wI, m, material.extIOR, material.intIOR );
	float D = ggxNormalDistribution( a, nO, m );
	float G = ggxVisibilityTerm( a, wI, wO, nO, m );
	float J = 1.0f / (4.0 * MdotO);
	result.bsdf =
		material.diffuse * INVERSE_PI * NdotO * (1.0f - F) +
		material.specular * (F * D * G / (4.0 * NdotI));
	result.pdf = 
		INVERSE_PI * NdotO * (1.0f - F) + 
		D * NdotM * J * F;
	result.weight = result.bsdf / result.pdf;
}

__device__ inline SampledMaterial sample( const Material& material,
	float3 nO, float3 wI, device const RandomSample& randomSample )
{
	// sample microfacet at given point
	float alphaSquared = material.roughness * material.roughness;
	float3 m = sampleGGXDistribution( nO, randomSample.bsdfSample, alphaSquared );

	// calculate Fresnel equation for given microfacet
	float F = fresnelDielectric( wI, m, material.extIOR, material.intIOR );

	float3 wO = {};
	if (randomSample.componentSample < F)
	{
		wO = reflect( wI, m );
	}
	else
	{
		wO = sampleCosineWeightedHemisphere( nO, randomSample.bsdfSample );
	}

	return evaluate( material, nO, wI, wO );
}
