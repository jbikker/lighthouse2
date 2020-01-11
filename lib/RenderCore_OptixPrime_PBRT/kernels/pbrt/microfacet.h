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

// ----------------------------------------------------------------

// Microfacet Utility Functions

LH2_DEVFUNC float RoughnessToAlpha( float roughness )
{
	roughness = max( roughness, 1e-3f );
	float x = logf( roughness );
	return 1.62142f + 0.819955f * x + 0.1734f * x * x +
		   0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

template <bool sampleVisibleArea>
class MicrofacetDistribution
{
  public:
	// Interface methods
	__device__ virtual float D( const float3& wh ) const = 0;
	__device__ virtual float Lambda( const float3& w ) const = 0;
	__device__ virtual float3 Sample_wh( const float3& wo, const float r0, const float r1 ) const = 0;
	__device__ virtual float G( const float3& wo, const float3& wi ) const
	{
		return 1.f / ( 1.f + Lambda( wo ) + Lambda( wi ) );
	}

	__device__ float Pdf( const float3& wo, const float3& wh ) const
	{
		if ( sampleVisibleArea )
			return D( wh ) * G1( wo ) * AbsDot( wo, wh ) / AbsCosTheta( wo );
		else
			return D( wh ) * AbsCosTheta( wh );
	}

	__device__ float G1( const float3& w ) const
	{
		// if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
		return 1.f / ( 1.f + Lambda( w ) );
	}
};

template <bool sampleVisibleArea = true>
class BeckmannDistribution : public MicrofacetDistribution<sampleVisibleArea>
{
	const float alphax, alphay;

	__device__ float Lambda( const float3& w ) const override
	{
		float absTanTheta = abs( TanTheta( w ) );
		if ( isinf( absTanTheta ) ) return 0.;
		// Compute _alpha_ for direction _w_
		float alpha =
			sqrtf( Cos2Phi( w ) * alphax * alphax + Sin2Phi( w ) * alphay * alphay );
		float a = 1 / ( alpha * absTanTheta );
		if ( a >= 1.6f ) return 0;
		return ( 1 - 1.259f * a + 0.396f * a * a ) / ( 3.535f * a + 2.181f * a * a );
	}

	LH2_DEVFUNC void BeckmannSample11( const float cosThetaI, const float U1, const float U2,
									   float& slope_x, float& slope_y )
	{
		/* Special case (normal incidence) */
		if ( cosThetaI > .9999f )
		{
			float r = sqrtf( -logf( 1.0f - U1 ) );
			float sinPhi, cosPhi;
			__sincosf( 2.f * PI * U2, &sinPhi, &cosPhi );
			// float sinPhi = std::sin(2.f * PI * U2);
			// float cosPhi = std::cos(2.f * PI * U2);
			slope_x = r * cosPhi;
			slope_y = r * sinPhi;
			return;
		}

		/* The original inversion routine from the paper contained
	   discontinuities, which causes issues for QMC integration
	   and techniques like Kelemen-style MLT. The following code
	   performs a numerical inversion with better behavior */
		float sinThetaI =
			sqrtf( max( 0.f, 1.f - cosThetaI * cosThetaI ) );
		float tanThetaI = sinThetaI / cosThetaI;
		float cotThetaI = 1 / tanThetaI;

		/* Search interval -- everything is parameterized
	   in the Erf() domain */
		float a = -1.f, c = Erf( cotThetaI );
		float sample_x = max( U1, 1e-6f );

		/* Start with a good initial guess */
		// float b = (1-sample_x) * a + sample_x * c;

		/* We can do better (inverse of an approximation computed in
	 * Mathematica) */
		float thetaI = acosf( cosThetaI );
		float fit = 1 + thetaI * ( -0.876f + thetaI * ( 0.4265f - 0.0594f * thetaI ) );
		float b = c - ( 1 + c ) * powf( 1 - sample_x, fit );

		/* Normalization factor for the CDF */
		// static const float SQRT_PI_INV = 1.f / std::sqrt( PI );
		float normalization =
			1 /
			( 1 + c + SQRT_PI_INV * tanThetaI * expf( -cotThetaI * cotThetaI ) );

		int it = 0;
		while ( ++it < 10 )
		{
			/* Bisection criterion -- the oddly-looking
		   Boolean expression are intentional to check
		   for NaNs at little additional cost */
			if ( !( b >= a && b <= c ) ) b = 0.5f * ( a + c );

			/* Evaluate the CDF and its derivative
		   (i.e. the density function) */
			float invErf = ErfInv( b );
			float value =
				normalization *
					( 1 + b + SQRT_PI_INV * tanThetaI * expf( -invErf * invErf ) ) -
				sample_x;
			float derivative = normalization * ( 1 - invErf * tanThetaI );

			if ( abs( value ) < 1e-5f ) break;

			/* Update bisection intervals */
			if ( value > 0 )
				c = b;
			else
				a = b;

			b -= value / derivative;
		}

		/* Now convert back into a slope value */
		slope_x = ErfInv( b );

		/* Simulate Y component */
		slope_y = ErfInv( 2.0f * max( U2, 1e-6f ) - 1.0f );

		assert( !isinf( slope_x ) );
		assert( !isnan( slope_x ) );
		assert( !isinf( slope_y ) );
		assert( !isnan( slope_y ) );
	}

	LH2_DEVFUNC float3 BeckmannSample( const float3& wi, float alpha_x, float alpha_y,
									   float U1, float U2 )
	{
		// 1. stretch wi
		const float3 wiStretched =
			normalize( make_float3( alpha_x * wi.x, alpha_y * wi.y, wi.z ) );

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		BeckmannSample11( CosTheta( wiStretched ), U1, U2, slope_x, slope_y );

		// 3. rotate
		const float tmp = CosPhi( wiStretched ) * slope_x - SinPhi( wiStretched ) * slope_y;
		slope_y = SinPhi( wiStretched ) * slope_x + CosPhi( wiStretched ) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alpha_x * slope_x;
		slope_y = alpha_y * slope_y;

		// 5. compute normal
		return normalize( make_float3( -slope_x, -slope_y, 1.f ) );
	}

  public:
	__device__ BeckmannDistribution( float alphax, float alphay )
		: alphax( alphax ), alphay( alphay ) {}

	__device__ float D( const float3& wh ) const override
	{
		const float tan2Theta = Tan2Theta( wh );
		if ( isinf( tan2Theta ) ) return 0.;
		const float c2t = Cos2Theta( wh );
		const float cos4Theta = c2t * c2t;
		return expf( -tan2Theta * ( Cos2Phi( wh ) / ( alphax * alphax ) +
									Sin2Phi( wh ) / ( alphay * alphay ) ) ) /
			   ( PI * alphax * alphay * cos4Theta );
	}

	__device__ float3 Sample_wh( const float3& wo, const float r0, const float r1 ) const override
	{
		if ( !sampleVisibleArea )
		{
			// Sample full distribution of normals for Beckmann distribution

			// Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
			float tan2Theta, phi;
			if ( alphax == alphay )
			{
				const float logSample = logf( 1 - r0 );
				assert( !isinf( logSample ) );
				tan2Theta = -alphax * alphax * logSample;
				phi = r1 * 2 * PI;
			}
			else
			{
				// Compute _tan2Theta_ and _phi_ for anisotroPIc Beckmann
				// distribution
				const float logSample = logf( 1 - r0 );
				assert( !isinf( logSample ) );
				phi = atanf( alphay / alphax * tanf( 2 * PI * r1 + 0.5f * PI ) );
				if ( r1 > 0.5f ) phi += PI;
				const float sinPhi = sinf( phi ), cosPhi = cosf( phi );
				const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
				tan2Theta = -logSample /
							( cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2 );
			}

			// Map sampled Beckmann angles to normal direction _wh_
			const float cosTheta = 1 / sqrtf( 1 + tan2Theta );
			const float sinTheta = sqrtf( max( 0.f, 1.f - cosTheta * cosTheta ) );
			float3 wh = SphericalDirection( sinTheta, cosTheta, phi );
			if ( !SameHemisphere( wo, wh ) ) wh = -wh;
			return wh;
		}
		else
		{
			// Sample visible area of normals for Beckmann distribution
			const bool flip = wo.z < 0;
			float3 wh = BeckmannSample( flip ? -wo : wo, alphax, alphay, r0, r1 );
			if ( flip ) wh = -wh;
			return wh;
		}
	}
};

template <bool sampleVisibleArea = true>
class TrowbridgeReitzDistribution : public MicrofacetDistribution<sampleVisibleArea>
{
	const float alphax, alphay;

	__device__ float Lambda( const float3& w ) const override
	{
		const float absTanTheta = abs( TanTheta( w ) );
#ifdef _MSC_VER
		if ( isinf( absTanTheta ) ) return 0.;
#else
		if ( isinf( absTanTheta ) ) return 0.;
#endif
		// Compute _alpha_ for direction _w_
		const float alpha =
			sqrtf( Cos2Phi( w ) * alphax * alphax + Sin2Phi( w ) * alphay * alphay );
		const float alpha2Tan2Theta = ( alpha * absTanTheta ) * ( alpha * absTanTheta );
		return ( -1 + sqrtf( 1.f + alpha2Tan2Theta ) ) / 2;
	}

	LH2_DEVFUNC void TrowbridgeReitzSample11( float cosTheta, float U1, float U2,
											  float& slope_x, float& slope_y )
	{
		// special case (normal incidence)
		if ( cosTheta > .9999f )
		{
			float r = sqrt( U1 / ( 1 - U1 ) );
			float phi = 6.28318530718f * U2;
			slope_x = r * cos( phi );
			slope_y = r * sin( phi );
			return;
		}

		float sinTheta =
			sqrtf( max( (float)0, (float)1 - cosTheta * cosTheta ) );
		float tanTheta = sinTheta / cosTheta;
		float a = 1 / tanTheta;
		float G1 = 2 / ( 1 + sqrtf( 1.f + 1.f / ( a * a ) ) );

		// sample slope_x
		const float A = 2 * U1 / G1 - 1;
		const float tmp = min( 1e10f, 1.f / ( A * A - 1.f ) );
		const float B = tanTheta;
		const float D = sqrtf(
			max( float( B * B * tmp * tmp - ( A * A - B * B ) * tmp ), float( 0 ) ) );
		const float slope_x_1 = B * tmp - D;
		const float slope_x_2 = B * tmp + D;
		slope_x = ( A < 0 || slope_x_2 > 1.f / tanTheta ) ? slope_x_1 : slope_x_2;

		// sample slope_y
		float S;
		if ( U2 > 0.5f )
		{
			S = 1.f;
			U2 = 2.f * ( U2 - .5f );
		}
		else
		{
			S = -1.f;
			U2 = 2.f * ( .5f - U2 );
		}
		float z =
			( U2 * ( U2 * ( U2 * 0.27385f - 0.73369f ) + 0.46341f ) ) /
			( U2 * ( U2 * ( U2 * 0.093073f + 0.309420f ) - 1.000000f ) + 0.597999f );
		slope_y = S * z * sqrtf( 1.f + slope_x * slope_x );

		assert( !isinf( slope_y ) );
		assert( !isnan( slope_y ) );
	}

	LH2_DEVFUNC float3 TrowbridgeReitzSample( const float3& wi, float alpha_x,
											  float alpha_y, float U1, float U2 )
	{
		// 1. stretch wi
		const float3 wiStretched =
			normalize( make_float3( alpha_x * wi.x, alpha_y * wi.y, wi.z ) );

		// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
		float slope_x, slope_y;
		TrowbridgeReitzSample11( CosTheta( wiStretched ), U1, U2, slope_x, slope_y );

		// 3. rotate
		const float tmp = CosPhi( wiStretched ) * slope_x - SinPhi( wiStretched ) * slope_y;
		slope_y = SinPhi( wiStretched ) * slope_x + CosPhi( wiStretched ) * slope_y;
		slope_x = tmp;

		// 4. unstretch
		slope_x = alpha_x * slope_x;
		slope_y = alpha_y * slope_y;

		// 5. compute normal
		return normalize( make_float3( -slope_x, -slope_y, 1. ) );
	}

  public:
	__device__ TrowbridgeReitzDistribution( float alphax, float alphay )
		: alphax( alphax ), alphay( alphay ) {}

	__device__ float D( const float3& wh ) const override
	{
		float tan2Theta = Tan2Theta( wh );
#ifdef _MSC_VER
		if ( isinf( tan2Theta ) ) return 0.;
#else
		if ( isinf( tan2Theta ) ) return 0.;
#endif
		const float cos4Theta = Cos2Theta( wh ) * Cos2Theta( wh );
		const float e =
			( Cos2Phi( wh ) / ( alphax * alphax ) + Sin2Phi( wh ) / ( alphay * alphay ) ) *
			tan2Theta;
		return 1 / ( PI * alphax * alphay * cos4Theta * ( 1 + e ) * ( 1 + e ) );
	}

	__device__ float3 Sample_wh( const float3& wo, const float r0, const float r1 ) const override
	{
		float3 wh;
		if ( !sampleVisibleArea )
		{
			float cosTheta = 0, phi = ( 2 * PI ) * r1;
			if ( alphax == alphay )
			{
				float tanTheta2 = alphax * alphax * r0 / ( 1.0f - r0 );
				cosTheta = 1 / sqrtf( 1 + tanTheta2 );
			}
			else
			{
				phi = atanf( alphay / alphax * tanf( 2 * PI * r1 + .5f * PI ) );
				if ( r1 > .5f ) phi += PI;
				const float sinPhi = sinf( phi ), cosPhi = cosf( phi );
				const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
				const float alpha2 =
					1 / ( cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2 );
				float tanTheta2 = alpha2 * r0 / ( 1 - r0 );
				cosTheta = 1 / sqrtf( 1 + tanTheta2 );
			}
			float sinTheta =
				sqrtf( max( 0.f, 1.f - cosTheta * cosTheta ) );
			wh = SphericalDirection( sinTheta, cosTheta, phi );
			if ( !SameHemisphere( wo, wh ) ) wh = -wh;
		}
		else
		{
			const bool flip = wo.z < 0;
			wh = TrowbridgeReitzSample( flip ? -wo : wo, alphax, alphay, r0, r1 );
			if ( flip ) wh = -wh;
		}
		return wh;
	}
};
