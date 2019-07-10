// Lambert BSDF
// ----------------------------------------------------------------

// for debugging: Lambert brdf
__device__ static float EvaluateBalance( const ShadingData& shadingData, const float3 n, const float3 o, const float2 r, float3& m1, float3& m2 )
{
	m1 = make_float3( r.x, r.y, 0 ); // we will use these random numbers in SampleBSDF
	return 0;
}

__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 wo, const float3 wi, const float balance, float& pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.diffuse * INVPI;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3 N, const float3 wo, const float r0, const float r1, float3& wi, float& pdf )
{
	// dielectric
	if (shadingData.eta > 1)
	{
		float3 bsdf;
		// smooth glass
		float no = shadingData.nino.y, ni = shadingData.nino.x, eta = shadingData.eta, cosi = dot( wo, iN );
		if (cosi > 0) { float h = no; no = ni; ni = h; }
		else { eta = 1.0f / eta; cosi = -cosi; iN = -iN; }
#if 1
		eta = 1.0f / eta;
		float h = ni; ni = no; no = h;
#endif
		const float cost2 = 1.0f - eta * eta * (1 - cosi * cosi);
		wi = -reflect( wo, iN ), pdf = 1, bsdf = shadingData.diffuse; // defaults for TIR
		if (cost2 > 0)
		{
			const float a = ni - no, b = ni + no, R0 = (a * a) / (b * b), c = 1 - cosi, Fr = R0 + (1 - R0) * (c * c * c * c * c);
			if (r0 < Fr)
			{
				pdf = 1.0f, bsdf = shadingData.diffuse;
				APPLYSAFENORMALS;
			}
			else
			{
				pdf = 1.0f, wi = -eta * wo + ((eta * cosi - sqrtf( fabs( cost2 ) )) * iN);
				if (dot( wi, iN ) >= 0) { pdf = 0; return make_float3( 0 ); }
				bsdf = shadingData.diffuse * eta * eta;
			}
		}
		bsdf = bsdf * (1.0f / max( EPSILON, fabs( dot( iN, wi ) ) ));
		// apply absorption
#if 0
		const float3 absorb = {
			__expf( (1 - shadingData.absorption.x) * -distance ),
			__expf( (1 - shadingData.absorption.y) * -distance ),
			__expf( (1 - shadingData.absorption.z) * -distance )
		};
#else
		const float3 absorb = make_float3( 1 );
#endif
		return bsdf * absorb;
	}
	// specular and diffuse
	if (fabs( shadingData.roughness1 ) < 0.1f)
	{
		// pure specular
		wi = -reflect( wo, iN );
		pdf = 1;
		APPLYSAFENORMALS;
		return shadingData.diffuse * (1.0f / abs( dot( iN, wi ) ));
	}
	else
	{
		wi = normalize( Tangent2World( DiffuseReflectionCosWeighted( r0, r1 ), iN ) );
		pdf = max( 0.0f, dot( wi, iN ) ) * INVPI;
		APPLYSAFENORMALS;
		return shadingData.diffuse * INVPI;
	}
}

// EOF