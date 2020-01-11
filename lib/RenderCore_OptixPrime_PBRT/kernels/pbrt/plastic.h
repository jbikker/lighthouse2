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

class Plastic : public SimpleMaterial<
					LambertianReflection,
					MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelDielectric>>
{
  public:
	__device__ void ComputeScatteringFunctions( const CoreMaterial& params,
												const float2 uv,
												const bool allowMultipleLobes,
												const TransportMode mode ) override
	{
		// TODO: Bumpmapping

		const auto Kd = SampleCoreTexture( params.color, uv );
		const auto Ks = SampleCoreTexture( params.Ks, uv );

		if ( !IsBlack( Kd ) )
			bxdfs.emplace_back<LambertianReflection>( Kd );

		if ( !IsBlack( Ks ) )
		{
			const float rough = SampleCoreTexture( params.roughness, uv );

			const FresnelDielectric fresnel( 1.5f, 1.f );
			const TrowbridgeReitzDistribution<> distrib( rough, rough );

			bxdfs.emplace_back<MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelDielectric>>( Ks, distrib, fresnel );
		}
	}
};
