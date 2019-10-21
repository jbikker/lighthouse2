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

// forward to Meir's API-agnostic sharedBRDFs folder.
#include <sharedbsdf.h>

// EOF