#include "noerrors.h"

// BSDF code
// A BSDF must implement two methods:
// __device__ static float3 EvaluateBSDF( 
//    const ShadingData& shadingData,            spatially varying BSDF parameters at the shading point
//    const float3 iN,                           (interpolated / consistent) normal to use in the BSDF evaluation
//    const float3 T,                            tangent, for anisotropic materials
//    const float3 wo,                           outgoing direction (LH2 convention: always towards camera)
//    const float3 wi,                           incoming direction (LH2 convention: always towards light)
//    float& pdf                                 return value: probability density
// )
// This function is used to evaluate a BSDF given an incoming and outgoing direction (e.g. for NEE). 
// It returns the BSDF itself, so for Lambert shadingData.diffuse / pi. It also returns the probability density 
// for the specified direction, so for Lambert with a (cos theta)/pi pdf, we get dot(wi,iN) / pi.
// Note: all vectors point *away* from the surface.
//
// and:
// 
// __device__ static float3 SampleBSDF(          
//    const ShadingData& shadingData,            spatially varying BSDF parameters at the shading point
//    float3 iN,                                 (interpolated / consistent) normal to use in BSDF sampling
//    const float3 N,                            geometric normal
//    const float3 T,                            tangent, for anisotropic materials
//    const float3 wo,                           outgoing direction
//    uint& seed,                                seed for the random number generator
//    float3& wi,                                return value: sampled direction over hemisphere
//    float& pdf                                 return value: probability density of wi
// )
// This function is used to chose a random direction over the hemisphere proportional to some pdf.
// It returns the BSDF itself, and also the sampled direction and probability density for this direction.
// ----------------------------------------------------------------
/**/
//#include "bsdfs/lambert.h"
#include <sharedbsdf.h>
// EOF

// Lambert BSDF
// ----------------------------------------------------------------
/*
// for debugging: Lambert brdf
__device__ static float3 EvaluateBSDF(const ShadingData& shadingData, const float3 iN, const float3 T,
    const float3 wo, const float3 wi, float& pdf)
{
    pdf = fabs(dot(wi, iN)) * INVPI;
    return shadingData.color * INVPI;
}

__device__ static float3 SampleBSDF(const ShadingData& shadingData, float3 iN, const float3 N, const float3 T, const float3 wo,
    const float r3, const float r4, float3& wi, float& pdf, const int type)
{
    // specular and diffuse
    if (fabs(ROUGHNESS) < 0.1f)
    {
        // pure specular
        wi = -reflect(wo, iN);
        pdf = 1;
        APPLYSAFENORMALS;
        return shadingData.color * (1.0f / abs(dot(iN, wi)));
    }
    else
    {
        wi = normalize(Tangent2World(DiffuseReflectionCosWeighted(r3, r4), iN));
        pdf = max(0.0f, dot(wi, iN)) * INVPI;
        APPLYSAFENORMALS;
        return shadingData.color * INVPI;
    }
}
*/
// EOF
