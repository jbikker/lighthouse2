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

class Glass : public SimpleMaterial<
				  FresnelSpecular,
				  SpecularReflection<FresnelDielectric>,
				  MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelDielectric>,
				  SpecularTransmission,
				  MicrofacetTransmission<TrowbridgeReitzDistribution<>>>
{

  public:
	__device__ void ComputeScatteringFunctions( const CoreMaterial& params,
												const float2 uv,
												const bool allowMultipleLobes,
												const TransportMode mode ) override
	{
		// TODO: Bumpmapping

		const auto R = SampleCoreTexture( params.color, uv );
		const auto T = SampleCoreTexture( params.absorption, uv );

		if ( IsBlack( R ) && IsBlack( T ) )
			return;

		const auto urough = SampleCoreTexture( params.urough, uv );
		const auto vrough = SampleCoreTexture( params.vrough, uv );
		const auto eta = SampleCoreTexture( params.eta, uv );

		const bool isSpecular = urough == 0 && vrough == 0;

		if ( isSpecular && allowMultipleLobes )
		{
			bxdfs.emplace_back<FresnelSpecular>( R, T, 1.f, eta, mode );
		}
		else
		{
			const TrowbridgeReitzDistribution<> distrib( urough, vrough );

			if ( !IsBlack( R ) )
			{
				const FresnelDielectric fresnel( 1.f, eta );
				if ( isSpecular )
					bxdfs.emplace_back<SpecularReflection<FresnelDielectric>>( R, fresnel );
				else
					bxdfs.emplace_back<MicrofacetReflection<TrowbridgeReitzDistribution<>, FresnelDielectric>>( R, distrib, fresnel );
			}

			if ( !IsBlack( T ) )
			{
				if ( isSpecular )
					bxdfs.emplace_back<SpecularTransmission>( T, 1.f, eta, mode );
				else
					bxdfs.emplace_back<MicrofacetTransmission<TrowbridgeReitzDistribution<>>>( T, distrib, 1.f, eta, mode );
			}
		}
	}
};
