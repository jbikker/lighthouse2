/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once

class Metal : public SimpleMaterial<MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelConductor>>
{

  public:
	__device__ void ComputeScatteringFunctions( const CoreMaterial& params,
												const float2 uv,
												const bool allowMultipleLobes,
												const TransportMode mode ) override
	{
		// TODO: Bumpmapping

		const auto k = SampleCoreTexture( params.absorption, uv );
		const auto eta = SampleCoreTexture( params.eta_rgb, uv );

		const FresnelConductor frMf( make_float3( 1.f ), eta, k );

		const auto urough = SampleCoreTexture( params.urough, uv );
		const auto vrough = SampleCoreTexture( params.vrough, uv );

		// NOTE: PBRT Doesn't make the optimization here to use a SpecularReflection like Glass does,
		// when u- and vrough are zero. This means a black output when the value is zero, and banding when near zero
		// (Unsure if this is a mathematic PBRT issue since the documentation says near-zero values should represent
		//  a mirror).
		// While the Mirror material would be a good alternative, it doesn't have an ETA nor K value. This material
		// on the other hand doesn't allow specifying the reflection color.

		const TrowbridgeReitzDistribution<> distrib( urough, vrough );
		bxdfs.emplace_back<MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelConductor>>( make_float3( 1.f ), distrib, frMf );
	}
};
