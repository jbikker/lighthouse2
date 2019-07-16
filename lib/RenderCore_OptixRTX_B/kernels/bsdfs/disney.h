// Joe Schutte
// https://schuttejoe.github.io/post/disneybsdf
// https://github.com/schuttejoe/Selas/blob/dev/Source/Core/Shading/Disney.cpp

__device__ static float clamp( const float x, const float a, const float b ) { return min( max( x, a ), b ); }
__device__ static float saturate( const float x ) { return clamp( x, 0.0f, 1.0f ); }
__device__ static float square( const float x ) { return x * x; }
__device__ static float AbsCosTheta( const float3& v ) { return abs( v.y ); }
__device__ static float Absdot( const float3& a, const float3& b ) { return abs( dot( a, b ) ); }
__device__ static float lerp( const float x, const float y, const float a ) { return (1 - a) * x + a * y; }
__device__ static float3 lerp3( const float3 x, const float3 y, const float a ) { return (1 - a) * x + a * y; }
__device__ static float3 vlerp( const float3 x, const float3 y, const float3 a ) { return (make_float3( 1 ) - a) * x + (a * y); }
__device__ static float pow5( const float x ) { return (x * x) * (x * x) * x; }
__device__ static float CosTheta( const float3& w ) { return w.y; }
__device__ static float Cos2Theta( const float3& w ) { return w.y * w.y; }
__device__ static float Sin2Theta( const float3& w ) { return max( 0.0f, 1.0f - Cos2Theta( w ) ); }
__device__ static float SinTheta( const float3& w ) { return sqrtf( Sin2Theta( w ) ); }
__device__ static float TanTheta( const float3& w ) { return SinTheta( w ) / CosTheta( w ); }
__device__ static float SinPhi( const float3& w )
{
	const float s = SinTheta( w );
	return (s == 0) ? 1.0f : clamp( w.z / s, -1.0f, 1.0f );
}
__device__ float CosPhi( const float3& w )
{
	float s = SinTheta( w );
	return (s == 0) ? 1.0f : clamp( w.x / s, -1.0f, 1.0f );
}
__device__ float Sign( const float x ) { return x < 0 ? -1 : (x > 0 ? 1 : 0); }
__device__ static float Sin2Phi( const float3& w ) { return square( SinPhi( w ) ); }
__device__ float Cos2Phi( const float3& w ) { return square( CosPhi( w ) ); }
__device__ float3 Reflect( float3 n, float3 l ) { return 2.0f * dot( n, l ) * n - l; }
__device__ bool Transmit( float3 wm, float3 wi, float n, float3& wo )
{
	float c = dot( wi, wm );
	if (c < 0.0f) c = -c, wm = -wm;
	float root = 1.0f - n * n * (1.0f - c * c);
	if (root <= 0) return false;
	wo = (n * c - sqrtf( root )) * wm - n * wi;
	return true;
}

enum SurfaceEventFlags { eScatterEvent = 1, eTransmissionEvent, eDiracEvent };

struct SurfaceParameters
{
	// base properties
	float3 baseColor;
	float3 mediumColor;
	float metallic;
	float specTrans;
	float specularTint;
	float roughness;
	float diffTrans;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	// transmittance
	float ior;
	float scatterDistance;
	float relativeIOR;
	float flatness;
};

struct BsdfSample
{
	float forwardPdfW;
	float reversePdfW;
	float3 reflectance;
	float3 wi;
	uint flags;
};

namespace Fresnel {

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/

//=========================================================================================================================
__device__ float3 Schlick( const float3 r0, const float radians )
{
	const float exponential = pow5( 1.0f - radians );
	return r0 + (make_float3( 1 ) - r0) * exponential;
}

//=========================================================================================================================
__device__ float SchlickWeight( const float u )
{
	return pow5( saturate( 1.0f - u ) );
}

//=========================================================================================================================
__device__ float Schlick( const float r0, const float radians )
{
	return lerp( 1.0f, SchlickWeight( radians ), r0 );
}

//=========================================================================================================================
__device__ float SchlickR0FromRelativeIOR( const float eta )
{
	// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
	return square( eta - 1.0f ) / square( eta + 1.0f );
}

//=========================================================================================================================
__device__ float SchlickDielectic( const float cosThetaI, const float relativeIor )
{
	const float r0 = SchlickR0FromRelativeIOR( relativeIor );
	return r0 + (1.0f - r0) * SchlickWeight( cosThetaI );
}

//=========================================================================================================================
__device__ float Dielectric( float cosThetaI, float ni, float nt )
{
	// Copied from PBRT. This function calculates the full Fresnel term for a dielectric material.
	// See Sebastion Legarde's link above for details.

	cosThetaI = clamp( cosThetaI, -1.0f, 1.0f );

	// Swap index of refraction if this is coming from inside the surface
	if (cosThetaI < 0.0f)
	{
		float temp = ni;
		ni = nt, nt = temp;
		cosThetaI = -cosThetaI;
	}
	const float sinThetaI = sqrtf( max( 0.0f, 1.0f - cosThetaI * cosThetaI ) );
	const float sinThetaT = ni / nt * sinThetaI;

	// Check for total internal reflection
	if (sinThetaT >= 1) return 1;

	const float cosThetaT = sqrtf( max( 0.0f, 1.0f - sinThetaT * sinThetaT ) );
	const float rParallel = ((nt * cosThetaI) - (ni * cosThetaT)) / ((nt * cosThetaI) + (ni * cosThetaT));
	const float rPerpendicuar = ((ni * cosThetaI) - (nt * cosThetaT)) / ((ni * cosThetaI) + (nt * cosThetaT));
	return (rParallel * rParallel + rPerpendicuar * rPerpendicuar) / 2;
}

} // namespace Fresnel

namespace Bsdf {

//=============================================================================================================================
__device__ float SeparableSmithGGXG1( const float3& w, const float3& wm, const float ax, const float ay )
{
	const float absTanTheta = abs( TanTheta( w ) );
	if (!isfinite( absTanTheta )) return 0.0f;
	const float a = sqrtf( Cos2Phi( w ) * ax * ax + Sin2Phi( w ) * ay * ay );
	const float a2Tan2Theta = square( a * absTanTheta );
	const float lambda = 0.5f * (-1.0f + sqrtf( 1.0f + a2Tan2Theta ));
	return 1.0f / (1.0f + lambda);
}

//=========================================================================================================================
__device__ float SeparableSmithGGXG1( const float3& w, float a )
{
	const float a2 = a * a;
	const float absDotNV = AbsCosTheta( w );
	return 2.0f / (1.0f + sqrtf( a2 + (1 - a2) * absDotNV * absDotNV ));
}

//=========================================================================================================================
__device__ float HeightCorrelatedSmithGGXG2( const float3& wo, const float3& wi, float a )
{
	const float absDotNV = AbsCosTheta( wo );
	const float absDotNL = AbsCosTheta( wi );
	const float a2 = a * a;

	// height-correlated masking function
	// https://twvideo01.ubm-us.net/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
	const float denomA = absDotNV * sqrtf( a2 + (1.0f - a2) * absDotNL * absDotNL );
	const float denomB = absDotNL * sqrtf( a2 + (1.0f - a2) * absDotNV * absDotNV );

	return 2.0f * absDotNL * absDotNV / (denomA + denomB);
}

//=========================================================================================================================
__device__ float GgxIsotropicD( const float3& wm, const float a )
{
	const float a2 = a * a, dotNH2 = Cos2Theta( wm );
	const float sqrtdenom = dotNH2 * (a2 - 1) + 1;
	return a2 / (PI * sqrtdenom * sqrtdenom);
}

//=========================================================================================================================
__device__ float GgxAnisotropicD( const float3& wm, const float ax, const float ay )
{
	const float dotHX2 = square( wm.x );
	const float dotHY2 = square( wm.z );
	const float cos2Theta = Cos2Theta( wm );
	const float ax2 = square( ax );
	const float ay2 = square( ay );
	return 1.0f / (PI * ax * ay * square( dotHX2 / ax2 + dotHY2 / ay2 + cos2Theta ));
}

//=========================================================================================================================
__device__ float GgxVndfPdf( const float3& wo, const float3& wm, const float3& wi, const float a )
{
	const float absDotNL = AbsCosTheta( wi );
	const float absDotLH = abs( dot( wm, wi ) );
	const float G1 = Bsdf::SeparableSmithGGXG1( wo, a );
	const float D = Bsdf::GgxIsotropicD( wm, a );
	return G1 * absDotLH * D / absDotNL;
}

//=========================================================================================================================
__device__ float3 SampleGgxVndfAnisotropic( const float3& wo, const float ax, float const ay, float const u1, float const u2 )
{
	// -- Stretch the view vector so we are sampling as though roughness==1
	const float3 v = normalize( make_float3( wo.x * ax, wo.y, wo.z * ay ) );

	// -- Build an orthonormal basis with v, t1, and t2
	const float3 t1 = (v.y < 0.9999f) ? normalize( cross( v, make_float3( 0, 1, 0 ) ) ) : make_float3( 1, 0, 0 );
	const float3 t2 = cross( t1, v );

	// -- Choose a point on a disk with each half of the disk weighted proportionally to its projection onto direction v
	const float a = 1.0f / (1.0f + v.y);
	const float r = sqrtf( u1 );
	const float phi = (u2 < a) ? (u2 / a) * PI : PI + (u2 - a) / (1.0f - a) * PI;
	const float p1 = r * cosf( phi );
	const float p2 = r * sinf( phi ) * ((u2 < a) ? 1.0f : v.y);

	// -- Calculate the normal in this stretched tangent space
	const float3 n = p1 * t1 + p2 * t2 + sqrtf( max( 0.0f, 1.0f - p1 * p1 - p2 * p2 ) ) * v;

	// -- unstretch and normalize the normal
	return normalize( make_float3( ax * n.x, n.y, ay * n.z ) );
}

//=========================================================================================================================
// https://hal.archives-ouvertes.fr/hal-01509746/document
__device__ float3 SampleGgxVndf( const float3 wo, const float roughness, const float u1, const float u2 )
{
	return SampleGgxVndfAnisotropic( wo, roughness, roughness, u1, u2 );
}

//=========================================================================================================================
__device__ float GgxVndfAnisotropicPdf( const float3& wi, const float3& wm, const float3& wo, const float ax, const float ay )
{
	const float absDotNL = AbsCosTheta( wi );
	const float absDotLH = abs( dot( wm, wi ) );
	const float G1 = SeparableSmithGGXG1( wo, wm, ax, ay );
	const float D = GgxAnisotropicD( wm, ax, ay );
	return G1 * absDotLH * D / absDotNL;
}

//=========================================================================================================================
__device__ void GgxVndfAnisotropicPdf( const float3& wi, const float3& wm, const float3& wo, const float ax, const float ay,
	float& forwardPdfW, float& reversePdfW )
{
	const float D = GgxAnisotropicD( wm, ax, ay );
	float absDotNL = AbsCosTheta( wi );
	float absDotHL = abs( dot( wm, wi ) );
	float G1v = SeparableSmithGGXG1( wo, wm, ax, ay );
	forwardPdfW = G1v * absDotHL * D / absDotNL;
	float absDotNV = AbsCosTheta( wo );
	float absDotHV = abs( dot( wm, wo ) );
	float G1l = SeparableSmithGGXG1( wi, wm, ax, ay );
	reversePdfW = G1l * absDotHV * D / absDotNV;
}

} // namespace Bsdf

namespace JoesDisney {

//=============================================================================================================================
__device__ static void CalculateLobePdfs( const SurfaceParameters& surface,
	float& pSpecular, float& pDiffuse, float& pClearcoat, float& pSpecTrans )
{
	const float metallicBRDF = surface.metallic;
	const float specularBSDF = (1.0f - surface.metallic) * surface.specTrans;
	const float dielectricBRDF = (1.0f - surface.specTrans) * (1.0f - surface.metallic);
	const float specularWeight = metallicBRDF + dielectricBRDF;
	const float transmissionWeight = specularBSDF;
	const float diffuseWeight = dielectricBRDF;
	const float clearcoatWeight = 1.0f * saturate( surface.clearcoat );
	const float norm = 1.0f / (specularWeight + transmissionWeight + diffuseWeight + clearcoatWeight);
	pSpecular = specularWeight * norm;
	pSpecTrans = transmissionWeight * norm;
	pDiffuse = diffuseWeight * norm;
	pClearcoat = clearcoatWeight * norm;
}

//=============================================================================================================================
__device__ static float ThinTransmissionRoughness( const float ior, const float roughness )
{
	// -- Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR course notes. Based on their figure the results
	// -- match a geometrically thin solid fairly well but it is odd to me that roughness is decreased until an IOR of just
	// -- over 2.
	return saturate( (0.65f * ior - 0.35f) * roughness );
}

//=============================================================================================================================
__device__ static void CalculateAnisotropicParams( const float roughness, const float anisotropic, float& ax, float& ay )
{
	const float aspect = sqrtf( 1.0f - 0.9f * anisotropic );
	ax = max( 0.001f, square( roughness ) / aspect );
	ay = max( 0.001f, square( roughness ) * aspect );
}

//=============================================================================================================================
__device__ static float3 CalculateTint( const float3 baseColor )
{
	// -- The color tint is never mentioned in the SIGGRAPH presentations as far as I recall but it was done in the BRDF
	// -- Explorer so I'll replicate that here.
	// -- JB: the sum of Joe's weights (far) exceed 1; I'm using different ones.
	const float luminance = baseColor.x * 0.2126f + baseColor.y * 0.7152f + baseColor.z * 0.0722f;
	return (luminance > 0.0f) ? (baseColor * (1.0f / luminance)) : make_float3( 1 );
}

//=============================================================================================================================
// -- "generalized" Trowbridge-Reitz curve ungeneralized with a hard-coded exponent of 1
__device__ static float GTR1( const float absDotHL, const float a )
{
	if (a >= 1) return INVPI;
	const float a2 = a * a;
	return (a2 - 1.0f) / (PI * log2f( a2 ) * (1.0f + (a2 - 1.0f) * absDotHL * absDotHL));
}

//=============================================================================================================================
__device__ static float EvaluateDisneyClearcoat( const float clearcoat, const float alpha, const float3& wo, const float3& wm, const float3& wi,
	float& fPdfW, float& rPdfW )
{
	if (clearcoat <= 0.0f) return 0.0f;
	const float absDotNH = AbsCosTheta( wm );
	const float absDotNL = AbsCosTheta( wi );
	const float absDotNV = AbsCosTheta( wo );
	const float dotHL = dot( wm, wi );
	const float d = GTR1( absDotNH, lerp( 0.1f, 0.001f, alpha ) );
	const float f = Fresnel::Schlick( 0.04f, dotHL );
	const float gl = Bsdf::SeparableSmithGGXG1( wi, 0.25f );
	const float gv = Bsdf::SeparableSmithGGXG1( wo, 0.25f );
	fPdfW = d / (4.0f * Absdot( wo, wm ));
	rPdfW = d / (4.0f * Absdot( wi, wm ));
	return 0.25f * clearcoat * d * f * gl * gv;
}

//=============================================================================================================================
__device__ static float3 EvaluateSheen( const SurfaceParameters& surface, const float3& wo, const float3& wm, const float3& wi )
{
	return make_float3( 0 );
	const float dotHL = abs( dot( wm, wi ) );
	const float3 tint = CalculateTint( surface.baseColor );
	return surface.sheen * lerp3( make_float3( 1.0f ), tint, surface.sheenTint ) * Fresnel::SchlickWeight( dotHL );
}

//=============================================================================================================================
__device__ static float3 DisneyFresnel( const SurfaceParameters& surface, const float3& wo, const float3& wm, const float3& wi )
{
	const float dotHV = dot( wm, wo );
	const float3 tint = CalculateTint( surface.baseColor );

	// -- See section 3.1 and 3.2 of the 2015 PBR presentation + the Disney BRDF explorer (which does their 2012 remapping
	// -- rather than the SchlickR0FromRelativeIOR seen here but they mentioned the switch in 3.2).
	float3 R0 = Fresnel::SchlickR0FromRelativeIOR( surface.relativeIOR ) * lerp3( make_float3( 1.0f ), tint, surface.specularTint );
	R0 = lerp3( R0, surface.baseColor, surface.metallic );

	const float dielectricFresnel = Fresnel::Dielectric( dotHV, 1.0f, surface.ior );
	const float3 metallicFresnel = Fresnel::Schlick( R0, dot( wi, wm ) );
	return lerp3( make_float3( dielectricFresnel ), metallicFresnel, surface.metallic );
}

//=============================================================================================================================
__device__ static float3 EvaluateDisneyBRDF( const SurfaceParameters& surface, const float3& wo, const float3& wm, const float3& wi,
	float& fPdf, float& rPdf )
{
	fPdf = 0.0f;
	rPdf = 0.0f;

	const float dotNL = CosTheta( wi );
	const float dotNV = CosTheta( wo );
	if (dotNL <= 0.0f || dotNV <= 0.0f) return make_float3( 0 );

	float ax, ay;
	CalculateAnisotropicParams( surface.roughness, surface.anisotropic, ax, ay );

	const float d = Bsdf::GgxAnisotropicD( wm, ax, ay );
	const float gl = Bsdf::SeparableSmithGGXG1( wi, wm, ax, ay );
	const float gv = Bsdf::SeparableSmithGGXG1( wo, wm, ax, ay );

	float3 f = DisneyFresnel( surface, wo, wm, wi );

	Bsdf::GgxVndfAnisotropicPdf( wi, wm, wo, ax, ay, fPdf, rPdf );
	fPdf *= (1.0f / (4 * Absdot( wo, wm )));
	rPdf *= (1.0f / (4 * Absdot( wi, wm )));

	return d * gl * gv * f / (4.0f * dotNL * dotNV);
}

//=============================================================================================================================
__device__ static bool SampleDisneyBRDF( uint& seed, const SurfaceParameters& surface, float3 wo, BsdfSample& sample )
{
	// float3 wo = Normalize( MatrixMultiply( v, surface.worldToTangent ) );

	// -- Calculate Anisotropic params
	float ax, ay;
	CalculateAnisotropicParams( surface.roughness, surface.anisotropic, ax, ay );

	// -- Sample visible distribution of normals
	float r0 = RandomFloat( seed );
	float r1 = RandomFloat( seed );
	float3 wm = Bsdf::SampleGgxVndfAnisotropic( wo, ax, ay, r0, r1 );

	// -- Reflect over wm
	float3 wi = normalize( Reflect( wm, wo ) );
	if (CosTheta( wi ) <= 0.0f)
	{
		sample.forwardPdfW = 0.0f;
		sample.reversePdfW = 0.0f;
		sample.reflectance = make_float3( 0 );
		sample.wi = make_float3( 0 );
		return false;
	}

	// -- Fresnel term for this lobe is complicated since we're blending with both the metallic and the specularTint
	// -- parameters plus we must take the IOR into account for dielectrics
	float3 F = DisneyFresnel( surface, wo, wm, wi );

	// -- Since we're sampling the distribution of visible normals the pdf cancels out with a number of other terms.
	// -- We are left with the weight G2(wi, wo, wm) / G1(wi, wm) and since Disney uses a separable masking function
	// -- we get G1(wi, wm) * G1(wo, wm) / G1(wi, wm) = G1(wo, wm) as our weight.
	float G1v = Bsdf::SeparableSmithGGXG1( wo, wm, ax, ay );
	float3 specular = G1v * F;

	sample.flags = SurfaceEventFlags::eScatterEvent;
	sample.reflectance = specular;
	sample.wi = wi; // lets not go back to world space: normalize( MatrixMultiply( wi, MatrixTranspose( surface.worldToTangent ) ) );
	Bsdf::GgxVndfAnisotropicPdf( wi, wm, wo, ax, ay, sample.forwardPdfW, sample.reversePdfW );

	sample.forwardPdfW *= (1.0f / (4 * Absdot( wo, wm )));
	sample.reversePdfW *= (1.0f / (4 * Absdot( wi, wm )));

	return true;
}

//=============================================================================================================================
__device__ static float3 EvaluateDisneySpecTransmission( const SurfaceParameters& surface, const float3& wo, const float3& wm,
	const float3& wi, float ax, float ay, bool thin )
{
	float relativeIor = surface.relativeIOR;
	float n2 = relativeIor * relativeIor;

	float absDotNL = AbsCosTheta( wi );
	float absDotNV = AbsCosTheta( wo );
	float dotHL = dot( wm, wi );
	float dotHV = dot( wm, wo );
	float absDotHL = abs( dotHL );
	float absDotHV = abs( dotHV );

	float d = Bsdf::GgxAnisotropicD( wm, ax, ay );
	float gl = Bsdf::SeparableSmithGGXG1( wi, wm, ax, ay );
	float gv = Bsdf::SeparableSmithGGXG1( wo, wm, ax, ay );

	float f = Fresnel::Dielectric( dotHV, 1.0f, surface.ior );

	float3 color;
	if (thin)
		color = make_float3( sqrtf( surface.baseColor.x ), sqrtf( surface.baseColor.y ), sqrtf( surface.baseColor.z ) );
	else
		color = surface.baseColor;

	// Note that we are intentionally leaving out the 1/n2 spreading factor since for VCM we will be evaluating particles with
	// this. That means we'll need to model the air-[other medium] transmission if we ever place the camera inside a non-air
	// medium.
	float c = (absDotHL * absDotHV) / (absDotNL * absDotNV);
	float t = (n2 / square( dotHL + relativeIor * dotHV ));
	return color * c * t * (1.0f - f) * gl * gv * d;
}

//=============================================================================================================================
__device__ static float EvaluateDisneyRetroDiffuse( const SurfaceParameters& surface, const float3& wo, const float3& wm, const float3& wi )
{
	const float dotNL = AbsCosTheta( wi );
	const float dotNV = AbsCosTheta( wo );
	const float roughness = surface.roughness * surface.roughness;
	const float rr = 0.5f + 2.0f * dotNL * dotNL * roughness;
	const float fl = Fresnel::SchlickWeight( dotNL );
	const float fv = Fresnel::SchlickWeight( dotNV );
	return rr * (fl + fv + fl * fv * (rr - 1.0f));
}

//=============================================================================================================================
__device__ static float EvaluateDisneyDiffuse( const SurfaceParameters& surface, const float3& wo, const float3& wm, const float3& wi,
	bool thin )
{
	const float dotNL = AbsCosTheta( wi );
	const float dotNV = AbsCosTheta( wo );
	const float fl = Fresnel::SchlickWeight( dotNL );
	const float fv = Fresnel::SchlickWeight( dotNV );
	float hanrahanKrueger = 0.0f;
	if (thin && surface.flatness > 0.0f)
	{
		const float roughness = surface.roughness * surface.roughness;
		const float dotHL = dot( wm, wi );
		const float fss90 = dotHL * dotHL * roughness;
		const float fss = lerp( 1.0f, fss90, fl ) * lerp( 1.0f, fss90, fv );
		const float ss = 1.25f * (fss * (1.0f / (dotNL + dotNV) - 0.5f) + 0.5f);
		hanrahanKrueger = ss;
	}
	const float lambert = 1.0f;
	const float retro = EvaluateDisneyRetroDiffuse( surface, wo, wm, wi );
	const float subsurfaceApprox = lerp( lambert, hanrahanKrueger, thin ? surface.flatness : 0.0f );
	return INVPI * (retro + subsurfaceApprox * (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv));
}

//=============================================================================================================================
__device__ static bool SampleDisneyClearcoat( uint& seed, const SurfaceParameters& surface, const float3& wo, BsdfSample& sample )
{
	// float3 wo = Normalize( MatrixMultiply( v, surface.worldToTangent ) );

	float a = 0.25f;
	float a2 = a * a;

	float r0 = RandomFloat( seed );
	float r1 = RandomFloat( seed );
	float cosTheta = sqrtf( max( 0.0f, (1.0f - powf( a2, 1.0f - r0 )) / (1.0f - a2) ) );
	float sinTheta = sqrtf( max( 0.0f, 1.0f - cosTheta * cosTheta ) );
	float phi = 2.0f * PI * r1;

	float3 wm = make_float3( sinTheta * cosf( phi ), cosTheta, sinTheta * sinf( phi ) );
	if (dot( wm, wo ) < 0.0f) wm = -wm;

	float3 wi = Reflect( wm, wo );
	if (dot( wi, wo ) < 0.0f) return false;

	float clearcoatWeight = surface.clearcoat;
	float clearcoatGloss = surface.clearcoatGloss;

	float dotNH = CosTheta( wm );
	float dotLH = dot( wm, wi );

	float d = GTR1( abs( dotNH ), lerp( 0.1f, 0.001f, clearcoatGloss ) );
	float f = Fresnel::Schlick( 0.04f, dotLH );
	float g = Bsdf::SeparableSmithGGXG1( wi, 0.25f ) * Bsdf::SeparableSmithGGXG1( wo, 0.25f );

	float fPdf = d / (4.0f * dot( wo, wm ));

	sample.reflectance = make_float3( 0.25f * clearcoatWeight * g * f * d ) / fPdf;
	sample.wi = wi; // let's not go to world space: normalize( MatrixMultiply( wi, MatrixTranspose( surface.worldToTangent ) ) );
	sample.forwardPdfW = fPdf;
	sample.reversePdfW = d / (4.0f * dot( wi, wm ));

	return true;
}

//=============================================================================================================================
__device__ static float3 CalculateExtinction( float3 apparantColor, float scatterDistance )
{
	const float3 a = apparantColor;
	const float3 s = make_float3( 1.9f ) - a + 3.5f * (a - make_float3( 0.8f )) * (a - make_float3( 0.8f ));
	return 1.0f / (s * scatterDistance);
}

//=============================================================================================================================
__device__ static bool SampleDisneySpecTransmission( uint& seed, const SurfaceParameters& surface, float3 wo, bool thin,
	BsdfSample& sample )
{
	// float3 wo = MatrixMultiply( v, surface.worldToTangent );
	if (CosTheta( wo ) == 0.0)
	{
		sample.forwardPdfW = 0.0f;
		sample.reversePdfW = 0.0f;
		sample.reflectance = make_float3( 0 );
		sample.wi = make_float3( 0 );
		return false;
	}

	// -- Scale roughness based on IOR
	float rscaled = thin ? ThinTransmissionRoughness( surface.ior, surface.roughness ) : surface.roughness;

	float tax, tay;
	CalculateAnisotropicParams( rscaled, surface.anisotropic, tax, tay );

	// -- Sample visible distribution of normals
	float r0 = RandomFloat( seed );
	float r1 = RandomFloat( seed );
	float3 wm = Bsdf::SampleGgxVndfAnisotropic( wo, tax, tay, r0, r1 );

	float dotVH = dot( wo, wm );
	if (wm.y < 0.0f) dotVH = -dotVH;

	float ni = wo.y > 0.0f ? 1.0f : surface.ior;
	float nt = wo.y > 0.0f ? surface.ior : 1.0f;
	float relativeIOR = ni / nt;

	// -- Disney uses the full dielectric Fresnel equation for transmission. We also importance sample F to switch between
	// -- refraction and reflection at glancing angles.
	float F = Fresnel::Dielectric( dotVH, 1.0f, surface.ior );

	// -- Since we're sampling the distribution of visible normals the pdf cancels out with a number of other terms.
	// -- We are left with the weight G2(wi, wo, wm) / G1(wi, wm) and since Disney uses a separable masking function
	// -- we get G1(wi, wm) * G1(wo, wm) / G1(wi, wm) = G1(wo, wm) as our weight.
	float G1v = Bsdf::SeparableSmithGGXG1( wo, wm, tax, tay );

	float pdf;

	float3 wi;
	if (RandomFloat( seed ) <= F)
	{
		wi = normalize( Reflect( wm, wo ) );

		sample.flags = SurfaceEventFlags::eScatterEvent;
		sample.reflectance = G1v * surface.baseColor;

		float jacobian = (4 * Absdot( wo, wm ));
		pdf = F / jacobian;
	}
	else
	{
		if (thin)
		{
			// -- When the surface is thin so it refracts into and then out of the surface during this shading event.
			// -- So the ray is just reflected then flipped and we use the sqrt of the surface color.
			wi = Reflect( wm, wo );
			wi.y = -wi.y;
			sample.reflectance = G1v * make_float3( sqrtf( surface.baseColor.x ), sqrtf( surface.baseColor.y ), sqrtf( surface.baseColor.z ) );

			// -- Since this is a thin surface we are not ending up inside of a volume so we treat this as a scatter event.
			sample.flags = SurfaceEventFlags::eScatterEvent;
		}
		else
		{
			if (Transmit( wm, wo, relativeIOR, wi ))
			{
				sample.flags = SurfaceEventFlags::eTransmissionEvent;
				// sample.medium.phaseFunction = dotVH > 0.0f ? MediumPhaseFunction::eIsotropic : MediumPhaseFunction::eVacuum;
				// sample.medium.extinction = CalculateExtinction( surface.transmittanceColor, surface.scatterDistance );
			}
			else
			{
				sample.flags = SurfaceEventFlags::eScatterEvent;
				wi = Reflect( wm, wo );
			}

			sample.reflectance = G1v * surface.baseColor;
		}

		wi = normalize( wi );

		float dotLH = abs( dot( wi, wm ) );
		float jacobian = dotLH / (square( dotLH + surface.relativeIOR * dotVH ));
		pdf = (1.0f - F) / jacobian;
	}

	if (CosTheta( wi ) == 0.0f)
	{
		sample.forwardPdfW = 0.0f;
		sample.reversePdfW = 0.0f;
		sample.reflectance = make_float3( 0 );
		sample.wi = make_float3( 0 );
		return false;
	}

	if (surface.roughness < 0.01f)
	{
		sample.flags |= SurfaceEventFlags::eDiracEvent;
	}

	// -- calculate pdf terms
	Bsdf::GgxVndfAnisotropicPdf( wi, wm, wo, tax, tay, sample.forwardPdfW, sample.reversePdfW );
	sample.forwardPdfW *= pdf;
	sample.reversePdfW *= pdf;

	// -- convert wi back to world space
	sample.wi = wi; // lets not go back to world space: normalize( MatrixMultiply( wi, MatrixTranspose( surface.worldToTangent ) ) );

	return true;
}

//=============================================================================================================================
__device__ static float3 SampleCosineWeightedHemisphere( float r0, float r1 )
{
	float r = sqrtf( r0 );
	float theta = 2.0f * PI * r1;

	return make_float3( r * cosf( theta ), sqrtf( max( 0.0f, 1 - r0 ) ), r * sinf( theta ) );
}

//=============================================================================================================================
__device__ static bool SampleDisneyDiffuse( uint& seed, const SurfaceParameters& surface, float3 wo, bool thin, BsdfSample& sample )
{
	// float3 wo = MatrixMultiply( v, surface.worldToTangent );

	float sign = Sign( CosTheta( wo ) );

	// -- Sample cosine lobe
	float r0 = RandomFloat( seed );
	float r1 = RandomFloat( seed );
	float3 wi = sign * SampleCosineWeightedHemisphere( r0, r1 );
	float3 wm = normalize( wi + wo );

	float dotNL = CosTheta( wi );
	if (dotNL == 0.0f)
	{
		sample.forwardPdfW = 0.0f;
		sample.reversePdfW = 0.0f;
		sample.reflectance = make_float3( 0 );
		sample.wi = make_float3( 0 );
		return false;
	}

	float dotNV = CosTheta( wo );

	float pdf;

	SurfaceEventFlags eventType = SurfaceEventFlags::eScatterEvent;

	float3 color = surface.baseColor;

	float p = RandomFloat( seed );
	if (p <= surface.diffTrans)
	{
		wi = -wi;
		pdf = surface.diffTrans;

		if (thin)
			color = make_float3( sqrtf( color.x ), sqrtf( color.y ), sqrtf( color.z ) );
		else
		{
			eventType = SurfaceEventFlags::eTransmissionEvent;
			// sample.medium.phaseFunction = MediumPhaseFunction::eIsotropic;
			// sample.medium.extinction = CalculateExtinction( surface.transmittanceColor, surface.scatterDistance );
		}
	}
	else
	{
		pdf = (1.0f - surface.diffTrans);
	}

	float3 sheen = EvaluateSheen( surface, wo, wm, wi );

	float diffuse = EvaluateDisneyDiffuse( surface, wo, wm, wi, thin );

	// Assert_( pdf > 0.0f );
	sample.reflectance = sheen + color * (diffuse / pdf);
	sample.wi = wi; // lets not convert back to world space: normalize( MatrixMultiply( wi, MatrixTranspose( surface.worldToTangent ) ) );
	sample.forwardPdfW = abs( dotNL ) * pdf;
	sample.reversePdfW = abs( dotNV ) * pdf;
	sample.flags = eventType;
	return true;
}

//=============================================================================================================================
__device__ float3 EvaluateDisney( const SurfaceParameters& surface, float3 wo, float3 wi, bool thin, float& forwardPdf, float& reversePdf )
{
	float3 wm = normalize( wo + wi );

	float dotNV = CosTheta( wo );
	float dotNL = CosTheta( wi );

	float3 reflectance = make_float3( 0 );
	forwardPdf = 0.0f;
	reversePdf = 0.0f;

	float pBRDF, pDiffuse, pClearcoat, pSpecTrans;
	CalculateLobePdfs( surface, pBRDF, pDiffuse, pClearcoat, pSpecTrans );

	float metallic = surface.metallic;
	float specTrans = surface.specTrans;

	// calculate all of the anisotropic params
	float ax, ay;
	CalculateAnisotropicParams( surface.roughness, surface.anisotropic, ax, ay );

	float diffuseWeight = (1.0f - metallic) * (1.0f - specTrans);
	float transWeight = (1.0f - metallic) * specTrans;

	// -- Clearcoat
	bool upperHemisphere = dotNL > 0.0f && dotNV > 0.0f;
	if (upperHemisphere && surface.clearcoat > 0.0f)
	{
		float forwardClearcoatPdfW, reverseClearcoatPdfW;
		float clearcoat = EvaluateDisneyClearcoat( surface.clearcoat, surface.clearcoatGloss, wo, wm, wi, forwardClearcoatPdfW, reverseClearcoatPdfW );
		reflectance += make_float3( clearcoat );
		forwardPdf += pClearcoat * forwardClearcoatPdfW;
		reversePdf += pClearcoat * reverseClearcoatPdfW;
	}

	// -- Diffuse
	if (diffuseWeight > 0.0f)
	{
		float forwardDiffusePdfW = AbsCosTheta( wi );
		float reverseDiffusePdfW = AbsCosTheta( wo );
		float diffuse = EvaluateDisneyDiffuse( surface, wo, wm, wi, thin );

		float3 sheen = EvaluateSheen( surface, wo, wm, wi );

		reflectance += diffuseWeight * (diffuse * surface.baseColor + sheen);

		forwardPdf += pDiffuse * forwardDiffusePdfW;
		reversePdf += pDiffuse * reverseDiffusePdfW;
	}

	// -- transmission
	if (transWeight > 0.0f)
	{

		// Scale roughness based on IOR (Burley 2015, Figure 15).
		float rscaled = thin ? ThinTransmissionRoughness( surface.ior, surface.roughness ) : surface.roughness;
		float tax, tay;
		CalculateAnisotropicParams( rscaled, surface.anisotropic, tax, tay );

		float3 transmission = EvaluateDisneySpecTransmission( surface, wo, wm, wi, tax, tay, thin );
		reflectance += transWeight * transmission;

		float forwardTransmissivePdfW;
		float reverseTransmissivePdfW;
		Bsdf::GgxVndfAnisotropicPdf( wi, wm, wo, tax, tay, forwardTransmissivePdfW, reverseTransmissivePdfW );

		float dotLH = dot( wm, wi );
		float dotVH = dot( wm, wo );
		forwardPdf += pSpecTrans * forwardTransmissivePdfW / (square( dotLH + surface.relativeIOR * dotVH ));
		reversePdf += pSpecTrans * reverseTransmissivePdfW / (square( dotVH + surface.relativeIOR * dotLH ));
	}

	// -- specular
	if (upperHemisphere)
	{
		float forwardMetallicPdfW;
		float reverseMetallicPdfW;
		float3 specular = EvaluateDisneyBRDF( surface, wo, wm, wi, forwardMetallicPdfW, reverseMetallicPdfW );

		reflectance += specular;
		forwardPdf += pBRDF * forwardMetallicPdfW / (4 * Absdot( wo, wm ));
		reversePdf += pBRDF * reverseMetallicPdfW / (4 * Absdot( wi, wm ));
	}

	reflectance = reflectance * abs( dotNL );

	return reflectance;
}

//=============================================================================================================================
__device__ bool SampleDisney( uint& seed, const SurfaceParameters& surface, float3 wo, bool thin, BsdfSample& sample )
{
	float pSpecular;
	float pDiffuse;
	float pClearcoat;
	float pTransmission;
	CalculateLobePdfs( surface, pSpecular, pDiffuse, pClearcoat, pTransmission );
	bool success = false;
	float pLobe = 0.0f;
	float p = RandomFloat( seed );
	if (p <= pSpecular)
	{
		success = SampleDisneyBRDF( seed, surface, wo, sample );
		pLobe = pSpecular;
	}
	else if (p > pSpecular && p <= (pSpecular + pClearcoat))
	{
		success = SampleDisneyClearcoat( seed, surface, wo, sample );
		pLobe = pClearcoat;
	}
	else if (p > pSpecular + pClearcoat && p <= (pSpecular + pClearcoat + pDiffuse))
	{
		success = SampleDisneyDiffuse( seed, surface, wo, thin, sample );
		pLobe = pDiffuse;
	}
	else if (pTransmission >= 0.0f)
	{
		success = SampleDisneySpecTransmission( seed, surface, wo, thin, sample );
		pLobe = pTransmission;
	}
	else
	{
		// -- Make sure we notice if this is occurring.
		sample.reflectance = make_float3( 1000000.0f, 0.0f, 0.0f );
		sample.forwardPdfW = 0.000000001f;
		sample.reversePdfW = 0.000000001f;
	}
	if (pLobe > 0.0f)
	{
		sample.reflectance = sample.reflectance * (1.0f / pLobe);
		sample.forwardPdfW *= pLobe;
		sample.reversePdfW *= pLobe;
	}
	return success;
}

}

// ----------------------------------------------------------------

__device__ static float3 _Local2World( const float3 V, const float3 N, const float3 T, const float3 B )
{
	return normalize( V.x * T + V.y * N + V.z * B );
}

__device__ static float3 _World2Local( const float3 V, const float3 N, const float3 T, const float3 B )
{
	return normalize( make_float3( dot( V, T ), dot( V, N ), dot( V, B ) ) );
}

__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, float& pdf )
{
	// setup the tangent frame
	const float3 B = normalize( cross( T, iN ) );
	const float3 Tfinal = cross( B, iN );
	const float3 wo_local = _World2Local( wo, iN, Tfinal, B );
	const float3 wi_local = _World2Local( wi, iN, Tfinal, B );
	// convert data
	SurfaceParameters surface;
	surface.baseColor = shadingData.baseColor;
	surface.mediumColor = shadingData.mediumColor;
	const uint4 parameters = shadingData.parameters;
	const float scale = 1.0f / 256.0f;
	surface.metallic = scale * (float)(parameters.x & 255);
	surface.specTrans = scale * (float)((parameters.x >> 8) & 255);
	surface.specularTint = scale * (float)((parameters.x >> 16) & 255);
	surface.roughness = scale * (float)((parameters.x >> 24) & 255);
	surface.diffTrans = scale * (float)(parameters.y & 255);
	surface.anisotropic = scale * (float)((parameters.y >> 8) & 255);
	surface.sheen = scale * (float)((parameters.y >> 16) & 255);
	surface.sheenTint = scale * (float)((parameters.y >> 24) & 255);
	surface.clearcoat = scale * (float)(parameters.z & 255);
	surface.clearcoatGloss = scale * (float)((parameters.z >> 8) & 255);
	surface.ior = scale * (float)((parameters.z >> 16) & 255) * 2.0f + 1.0f;
	surface.scatterDistance = scale * (float)((parameters.z >> 24) & 255);
	surface.relativeIOR = scale * (float)(parameters.w & 255);
	surface.flatness = scale * (float)((parameters.w >> 8) & 255);
	// go
	float dummy; // for the reversePdf which we don't use
	float3 bsdf = JoesDisney::EvaluateDisney( surface, wo_local, wi_local, true, pdf, dummy );
	return bsdf;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3 N, const float3 T, const float3 wo,
	uint& seed, float3& wi, float& pdf )
{
	// setup the tangent frame
	const float3 B = normalize( cross( T, iN ) );
	const float3 Tfinal = cross( B, iN );
	const float3 wo_local = _World2Local( wo, iN, Tfinal, B );
	// convert data
	SurfaceParameters surface;
	surface.baseColor = shadingData.baseColor;
	surface.mediumColor = shadingData.mediumColor;
	const uint4 parameters = shadingData.parameters;
	const float scale = 1.0f / 256.0f;
	surface.metallic = scale * (float)(parameters.x & 255);
	surface.specTrans = scale * (float)((parameters.x >> 8) & 255);
	surface.specularTint = scale * (float)((parameters.x >> 16) & 255);
	surface.roughness = scale * (float)((parameters.x >> 24) & 255);
	surface.diffTrans = scale * (float)(parameters.y & 255);
	surface.anisotropic = scale * (float)((parameters.y >> 8) & 255);
	surface.sheen = scale * (float)((parameters.y >> 16) & 255);
	surface.sheenTint = scale * (float)((parameters.y >> 24) & 255);
	surface.clearcoat = scale * (float)(parameters.z & 255);
	surface.clearcoatGloss = scale * (float)((parameters.z >> 8) & 255);
	surface.ior = scale * (float)((parameters.z >> 16) & 255) * 2.0f + 1.0f;
	surface.scatterDistance = scale * (float)((parameters.z >> 24) & 255);
	surface.relativeIOR = scale * (float)(parameters.w & 255);
	surface.flatness = scale * (float)((parameters.w >> 8) & 255);
	// execute Joe
	BsdfSample sample;
	if (JoesDisney::SampleDisney( seed, surface, wo_local, true, sample ))
	{
		pdf = sample.forwardPdfW;
		wi = _Local2World( sample.wi, iN, Tfinal, B );
	#if 0
		// test
		float mag = sqrtf( dot( sample.reflectance, sample.reflectance ) );
		return mag * normalize( surface.baseColor );
	#else
		return sample.reflectance;
	#endif
	}
	else
	{
		pdf = 0;
		return make_float3( 0 );
	}
}

// EOF