#ifndef BSDF_H
#define BSDF_H

#include "compatibility.h"

// BSDF code
// A BSDF must implement two methods:
// __device__ static float3 EvaluateBSDF(
//    const ShadingData shadingData,             spatially varying BSDF parameters at the shading point
//    const float3 iN,                           (interpolated / consistent) normal to use in the BSDF evaluation
//    const float3 T,                            tangent, for anisotropic materials
//    const float3 wo,                           outgoing direction (LH2 convention: always towards camera)
//    const float3 wi,                           incoming direction (LH2 convention: always towards light)
//    REFERENCE_OF(float) pdf                                 return value: probability density
// )
// This function is used to evaluate a BSDF given an incoming and outgoing direction (e.g. for NEE).
// It returns the BSDF itself, so for Lambert shadingData.diffuse / pi. It also returns the probability density
// for the specified direction, so for Lambert with a (cos theta)/pi pdf, we get dot(wi,iN) / pi.
// Note: all vectors point *away* from the surface.
//
// and:
//
// __device__ static float3 SampleBSDF(
//    const ShadingData shadingData,             spatially varying BSDF parameters at the shading point
//    float3 iN,                                 (interpolated / consistent) normal to use in BSDF sampling
//    const float3 N,                            geometric normal
//    const float3 T,                            tangent, for anisotropic materials
//    const float3 wo,                           outgoing direction
//    uint& seed,                                seed for the random number generator
//    REFERENCE_OF(float3) wi,                   return value: sampled direction over hemisphere
//    REFERENCE_OF(float) pdf                    return value: probability density of wi
// )
// This function is used to chose a random direction over the hemisphere proportional to some pdf.
// It returns the BSDF itself, and also the sampled direction and probability density for this direction.
//
// Do not use references (&) as this breaks compatibility with Vulkan GLSL.
// Instead, use REFERENCE_OF(__type__) as a workaround.
// Note that const references are not supported, compilers optimize the supposed ShadingData copy away
// ----------------------------------------------------------------

#if 0

// simple reference bsdf: Lambert plus specular reflection
#include "lambert.h"

#else

// Disney's principled BRDF, adapted from https://www.shadertoy.com/view/XdyyDd
#include "ggxmdf.h"
#include "disney.h"

#endif

#endif // BSDF_H

// EOF