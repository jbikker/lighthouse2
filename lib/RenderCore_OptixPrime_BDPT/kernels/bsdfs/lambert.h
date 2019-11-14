// Lambert BSDF
// ----------------------------------------------------------------

__device__ static float Fr_L(float VDotN, float eio)
{
    if (VDotN < 0.0f)
    {
        eio = 1.0f / eio;
        VDotN = fabs(VDotN);
    }

    const float SinThetaT2 = sqr(eio) * (1.0f - VDotN * VDotN);
    if (SinThetaT2 > 1.0f) return 1.0f; // TIR
    const float LDotN = sqrtf(1.0f - SinThetaT2);
    // todo: reformulate to remove this division
    const float r1 = (VDotN - eio * LDotN) / (VDotN + eio * LDotN);
    const float r2 = (LDotN - eio * VDotN) / (LDotN + eio * VDotN);
    return 0.5f * (sqr(r1) + sqr(r2));
}

__device__ static inline bool Refract_L(const float3 &wi, const float3 &n, const float eta, float3& wt)
{
    float cosThetaI = fabs(dot(n, wi));
    float sin2ThetaI = max(0.0f, 1.0f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;
    if (sin2ThetaT >= 1) return false; // TIR
    float cosThetaT = sqrtf(1.0f - sin2ThetaT);
    wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3(n);
    return true;
}

// for debugging: Lambert brdf
__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, float& pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.color * INVPI;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, 
    float3 N, const float3 iN, const float3 T, const float3 wo,
    const float r3, const float r4, float3& wi, float& pdf, const int type )
{
    /*
    float v = dot(N, wo);
    if (v < 0.0f)
    {
        N = N * -1.0f;
    }
    */
    if (r3 < TRANSMISSION)
    {
        // specular
        float F = Fr_L(dot(N, wo), ETA);
        if (r4 < F)
        {
            // pure specular
            wi = -reflect(wo, N);
            pdf = 1;
            APPLYSAFENORMALS;
            return shadingData.color * (1.0f / abs(dot(N, wi)));
        }
        else
        {
            bool entering = dot(wo, N) > 0.0f;

            float eio = entering ? ETA : 1.0f / ETA;

            if (Refract_L(wo, N, eio, wi))
            {
                pdf = 1.0f;

                float3 ft = shadingData.color;
                if (type == 1)
                {
                    ft *= (eio * eio);
                }

                return ft * (1.0f / abs(dot(N, wi)));
            }

            return make_float3(0.0f);
        }
    }
    else
    {
        // specular and diffuse
        if (fabs(ROUGHNESS) < 0.1f)
        {
            // pure specular
            wi = -reflect(wo, N);
            pdf = 1;
            APPLYSAFENORMALS;

            return shadingData.color * (1.0f / abs(dot(N, wi)));
        }
        else
        {
            wi = normalize(Tangent2World(DiffuseReflectionCosWeighted(r3, r4), N));
            pdf = max(0.0f, fabs(dot(wi, N))) * INVPI;

            APPLYSAFENORMALS;
            return shadingData.color * INVPI;
        }
    }
}

// EOF