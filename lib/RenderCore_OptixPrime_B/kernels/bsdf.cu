#include "noerrors.h"

// BSDF code
// ----------------------------------------------------------------

#if 1

#include "bsdfs/lambert.h"

#else

// IN PROGRESS, IGNORE

// Code adapted from https://www.shadertoy.com/view/XdyyDd
// Disney brdf's taken from here:: https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf

#define Le float3(25.)
#define clearCoatBoost 1.

// To disable multiple importance sampling
#define USE_MIS

struct MaterialInfo
{
	float3  baseColor;
	float subsurface;
	float roughness;
	float metallic;
	float specular;
	float specularTint;
	float clearcoat;
	float clearcoatGloss;
	float anisotropic;
	float sheen;
	float sheenTint;
};

struct SurfaceInteraction
{
	float id;
	float3 incomingRayDir;
	float3 point;
	float3 normal;
	float3 tangent;
	float3 binormal;
	float objId;
};

__device__ float3 linearToGamma( const float3& c ) 
{ 
	return make_float3( powf( c.x, 0.4545f ), powf( c.y, 0.4545f ), powf( c.z, 0.4545f ) ); 
}

__device__ float3 gammaToLinear( const float3& g ) 
{ 
	return make_float3( powf( g.x, 2.2f ), powf( g.y, 2.2f ), powf( g.z, 2.2f ) ); 
}

__device__ float distanceSq( float3 v1, float3 v2 ) { float3 d = v1 - v2; return dot( d, d ); }

__device__ float pow2( float x ) { return x * x; }

__device__ float mix( const float x, const float y, const float a ) { return (1 - a) * x + a * y; }
__device__ float3 mix( const float3 x, const float3 y, const float a ) { return (1 - a) * x + a * y; }

__device__ void createBasis( const float3& normal, float3& tangent, float3& binormal )
{
	if (abs( normal.x ) > abs( normal.y ))
		tangent = normalize( make_float3( 0.0f, normal.z, -normal.y ) );
	else
		tangent = normalize( make_float3( -normal.z, 0.0f, normal.x ) );
	binormal = cross( normal, tangent );
}

__device__ void directionOfAnisotropicity( const float3& normal, float3& tangent, float3& binormal )
{
	tangent = cross( normal, make_float3( 1.0f, 0.0f, 1.0f ) );
	binormal = normalize( cross( normal, tangent ) );
	tangent = normalize( cross( normal, binormal ) );
}

__device__ float3 sphericalDirection( float sinTheta, float cosTheta, float sinPhi, float cosPhi )
{
	return make_float3( sinTheta * cosPhi, sinTheta * sinPhi, cosTheta );
}

__device__ bool sameHemiSphere( const float3& wo, const float3& wi, const float3& normal )
{
	return dot( wo, normal ) * dot( wi, normal ) > 0.0;
}

__device__ float schlickWeight( float cosTheta )
{
	float m = clamp( 1.0f - cosTheta, 0.0f, 1.0f );
	return (m * m) * (m * m) * m;
}

__device__ float GTR1( float NdotH, float a )
{
	if (a >= 1.0f) return 1.0f / PI;
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
	return (a2 - 1.0f) / (PI * log( a2 ) * t);
}

__device__ float GTR2( float NdotH, float a )
{
	float a2 = a * a;
	float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
	return a2 / (PI * t * t);
}

__device__ float GTR2_aniso( float NdotH, float HdotX, float HdotY, float ax, float ay )
{
	return 1.0f / (PI * ax * ay * pow2( pow2( HdotX / ax ) + pow2( HdotY / ay ) + NdotH * NdotH ));
}

__device__ float smithG_GGX( float NdotV, float alphaG )
{
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0f / (abs( NdotV ) + max( sqrtf( a + b - a * b ), EPSILON ));
}

__device__ float smithG_GGX_aniso( float NdotV, float VdotX, float VdotY, float ax, float ay )
{
	return 1.0f / (NdotV + sqrtf( pow2( VdotX * ax ) + pow2( VdotY * ay ) + pow2( NdotV ) ));
}

__device__ float pdfLambertianReflection( const float3& wi, const float3& wo, const float3& normal )
{
	return sameHemiSphere( wo, wi, normal ) ? abs( dot( normal, wi ) ) / PI : 0;
}

__device__ float pdfMicrofacet( const float3& wi, const float3& wo, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	if (!sameHemiSphere( wo, wi, interaction.normal )) return 0;
	float3 wh = normalize( wo + wi );
	float NdotH = dot( interaction.normal, wh );
	float alpha2 = material.roughness * material.roughness;
	alpha2 *= alpha2;
	float cos2Theta = NdotH * NdotH;
	float denom = cos2Theta * (alpha2 - 1.0f) + 1.0f;
	if (denom == 0.0f) return 0.;
	float pdfDistribution = alpha2 * NdotH / (PI * denom * denom);
	return pdfDistribution / (4.0f * dot( wo, wh ));
}

__device__ float pdfMicrofacetAniso( const float3& wi, const float3& wo, const float3& X, const float3& Y, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	if (!sameHemiSphere( wo, wi, interaction.normal )) return 0.;
	float3 wh = normalize( wo + wi );
	float aspect = sqrt( 1.0f - material.anisotropic * 0.9f );
	float alphax = max( 0.001f, pow2( material.roughness ) / aspect );
	float alphay = max( 0.001f, pow2( material.roughness ) * aspect );
	float alphax2 = alphax * alphax;
	float alphay2 = alphax * alphay;
	float hDotX = dot( wh, X );
	float hDotY = dot( wh, Y );
	float NdotH = dot( interaction.normal, wh );
	float denom = hDotX * hDotX / alphax2 + hDotY * hDotY / alphay2 + NdotH * NdotH;
	if (denom == 0) return 0;
	float pdfDistribution = NdotH / (PI * alphax * alphay * denom * denom);
	return pdfDistribution / (4.0f * dot( wo, wh ));
}

__device__ float pdfClearCoat( const float3& wi, const float3& wo, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	if (!sameHemiSphere( wo, wi, interaction.normal )) return 0.0f;
	float3 wh = wi + wo;
	wh = normalize( wh );
	float NdotH = abs( dot( wh, interaction.normal ) );
	float Dr = GTR1( NdotH, mix( 0.1f, 0.001f, material.clearcoatGloss ) );
	return Dr * NdotH / (4.0f * dot( wo, wh ));
}

__device__ float3 disneyDiffuse( const float NdotL, const float NdotV, const float LdotH, const MaterialInfo& material )
{
	float FL = schlickWeight( NdotL ), FV = schlickWeight( NdotV );
	float Fd90 = 0.5f + 2.0f * LdotH * LdotH * material.roughness;
	float Fd = mix( 1.0f, Fd90, FL ) * mix( 1.0f, Fd90, FV );
	return (1.0f / PI) * Fd * material.baseColor;
}

__device__ float3 disneySubsurface( const float NdotL, const float NdotV, const float LdotH, const MaterialInfo& material )
{
	float FL = schlickWeight( NdotL ), FV = schlickWeight( NdotV );
	float Fss90 = LdotH * LdotH * material.roughness;
	float Fss = mix( 1.0f, Fss90, FL ) * mix( 1.0f, Fss90, FV );
	float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);
	return (1.0f / PI) * ss * material.baseColor;
}

__device__ float3 disneyMicrofacetIsotropic( float NdotL, float NdotV, float NdotH, float LdotH, const MaterialInfo& material )
{
	float Cdlum = 0.3f * material.baseColor.x + 0.6f * material.baseColor.y + 0.1f * material.baseColor.z; // luminance approx.
	float3 Ctint = Cdlum > 0.0f ? material.baseColor / Cdlum : make_float3( 1 ); // normalize lum. to isolate hue+sat
	float3 Cspec0 = mix( material.specular * 0.08f * mix( make_float3( 1 ), Ctint, material.specularTint ), material.baseColor, material.metallic );
	float a = max( 0.001f, pow2( material.roughness ) );
	float Ds = GTR2( NdotH, a );
	float FH = schlickWeight( LdotH );
	float3 Fs = mix( Cspec0, make_float3( 1 ), FH );
	float Gs = smithG_GGX( NdotL, a ) * smithG_GGX( NdotV, a );
	return Gs * Fs * Ds;
}

__device__ float3 disneyMicrofacetAnisotropic( float NdotL, float NdotV, float NdotH, float LdotH,
	const float3& L, const float3& V,
	const float3& H, const float3& X, const float3& Y,
	const MaterialInfo& material )
{
	float Cdlum = 0.3f * material.baseColor.x + 0.6f * material.baseColor.y + 0.1f * material.baseColor.z;
	float3 Ctint = Cdlum > 0.0f ? material.baseColor / Cdlum : make_float3( 1 );
	float3 Cspec0 = mix( material.specular * 0.08f * mix( make_float3( 1 ), Ctint, material.specularTint ), material.baseColor, material.metallic );
	float aspect = sqrt( 1.0f - material.anisotropic * 0.9f );
	float ax = max( 0.001f, pow2( material.roughness ) / aspect );
	float ay = max( 0.001f, pow2( material.roughness ) * aspect );
	float Ds = GTR2_aniso( NdotH, dot( H, X ), dot( H, Y ), ax, ay );
	float FH = schlickWeight( LdotH );
	float3 Fs = mix( Cspec0, make_float3( 1 ), FH );
	float Gs;
	Gs = smithG_GGX_aniso( NdotL, dot( L, X ), dot( L, Y ), ax, ay );
	Gs *= smithG_GGX_aniso( NdotV, dot( V, X ), dot( V, Y ), ax, ay );
	return Gs * Fs * Ds;
}

__device__ float disneyClearCoat( float NdotL, float NdotV, float NdotH, float LdotH, const MaterialInfo& material )
{
	float gloss = mix( 0.1f, 0.001f, material.clearcoatGloss );
	float Dr = GTR1( abs( NdotH ), gloss );
	float FH = schlickWeight( LdotH );
	float Fr = mix( 0.04f, 1.0f, FH );
	float Gr = smithG_GGX( NdotL, 0.25f ) * smithG_GGX( NdotV, 0.25f );
	return clearCoatBoost * material.clearcoat * Fr * Gr * Dr;
}

__device__ float3 disneySheen( float LdotH, const MaterialInfo& material )
{
	float FH = schlickWeight( LdotH );
	float Cdlum = 0.3f * material.baseColor.x + 0.6f * material.baseColor.y + 0.1f * material.baseColor.z;
	float3 Ctint = Cdlum > 0.0f ? material.baseColor / Cdlum : make_float3( 1.0f );
	float3 Csheen = mix( make_float3( 1.0f ), Ctint, material.sheenTint );
	float3 Fsheen = FH * material.sheen * Csheen;
	return FH * material.sheen * Csheen;
}

__device__ float3 bsdfEvaluate( const float3& wi, const float3& wo, const float3& X, const float3& Y, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	if (!sameHemiSphere( wo, wi, interaction.normal )) return make_float3( 0 );
	float NdotL = dot( interaction.normal, wo );
	float NdotV = dot( interaction.normal, wi );
	if (NdotL < 0.0f || NdotV < 0.0f) return make_float3( 0 );
	float3 H = normalize( wo + wi );
	float NdotH = dot( interaction.normal, H );
	float LdotH = dot( wo, H );
	float3 diffuse = disneyDiffuse( NdotL, NdotV, LdotH, material );
	float3 subSurface = disneySubsurface( NdotL, NdotV, LdotH, material );
	float3 glossy = disneyMicrofacetAnisotropic( NdotL, NdotV, NdotH, LdotH, wi, wo, H, X, Y, material );
	float clearCoat = disneyClearCoat( NdotL, NdotV, NdotH, LdotH, material );
	float3 sheen = disneySheen( LdotH, material );
	return (mix( diffuse, subSurface, material.subsurface ) + sheen) * (1.0f - material.metallic) + glossy + clearCoat;
}

__device__ void disneyDiffuseSample( float3& wi, const float3& wo, float& pdf, const float2& u, const float3& normal, const MaterialInfo& material )
{
	float3 wiLocal = DiffuseReflectionCosWeighted( u.x, u.y );
	float3 tangent = make_float3( 0 ), binormal = make_float3( 0 );
	createBasis( normal, tangent, binormal );
	wi = wiLocal.x * tangent + wiLocal.y * binormal + wiLocal.z * normal;
	if (dot( wo, normal ) < 0.0f) wi.z *= -1.0f;
}

__device__ float3 disneySubSurfaceSample( float3& wi, const float3& wo, float& pdf, const float2& u, const float3& normal, const MaterialInfo& material )
{
	float3 wiLocal = DiffuseReflectionCosWeighted( u.x, u.y );
	float3 tangent = make_float3( 0 ), binormal = make_float3( 0 );
	createBasis( normal, tangent, binormal );
	wi = wiLocal.x * tangent + wiLocal.y * binormal + wiLocal.z * normal;
	if (dot( wo, normal ) < 0.0f) wi.z *= -1.0f;
	float3 H = normalize( wo + wi );
	float NdotH = dot( normal, H );
	pdf = pdfLambertianReflection( wi, wo, normal );
	return make_float3( 0 );//disneySubsurface(NdotL, NdotV, NdotH, material) * material.subsurface;
}

__device__ float3 disneySheenSample( float3& wi, const float3& wo, float& pdf, const float2& u, const float3& normal, const MaterialInfo& material )
{
	float3 wiLocal = DiffuseReflectionCosWeighted( u.x, u.y );
	float3 tangent = make_float3( 0 ), binormal = make_float3( 0 );
	createBasis( normal, tangent, binormal );
	wi = wiLocal.x * tangent + wiLocal.y * binormal + wiLocal.z * normal;
	if (dot( wo, normal ) < 0.0f) wi.z *= -1.0f;
	float3 H = normalize( wo + wi );
	float LdotH = dot( wo, H );
	pdf = pdfLambertianReflection( wi, wo, normal );
	return disneySheen( LdotH, material );
}

__device__ float3 disneyMicrofacetSample( float3& wi, const float3& wo, float& pdf, const float2& u, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	float cosTheta = 0., phi = (2.0f * PI) * u.y;
	float alpha = material.roughness * material.roughness;
	float tanTheta2 = alpha * alpha * u.x / (1.0f - u.x);
	cosTheta = 1. / sqrt( 1.0f + tanTheta2 );
	float sinTheta = sqrt( max( EPSILON, 1.0f - cosTheta * cosTheta ) );
	float3 whLocal = sphericalDirection( sinTheta, cosTheta, sin( phi ), cos( phi ) );
	float3 tangent = make_float3( 0 ), binormal = make_float3( 0 );
	createBasis( interaction.normal, tangent, binormal );
	float3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * interaction.normal;
	if (!sameHemiSphere( wo, wh, interaction.normal )) wh *= -1.0f;
	wi = reflect( wo * -1.0f, wh );
	float NdotL = dot( interaction.normal, wo );
	float NdotV = dot( interaction.normal, wi );
	if (NdotL < 0.0f || NdotV < 0.0f) { pdf = 0.0f; return make_float3( 0 ); }
	float3 H = normalize( wo + wi );
	float NdotH = dot( interaction.normal, H );
	float LdotH = dot( wo, H );
	pdf = pdfMicrofacet( wi, wo, interaction, material );
	return disneyMicrofacetIsotropic( NdotL, NdotV, NdotH, LdotH, material );
}

__device__ void disneyMicrofacetAnisoSample( float3& wi, const float3& wo, const float3& X, const float3& Y, const float2& u, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	float cosTheta = 0., phi = 0.;
	float aspect = sqrtf( 1.0f - material.anisotropic*.9 );
	float alphax = max( 0.001f, pow2( material.roughness ) / aspect );
	float alphay = max( 0.001f, pow2( material.roughness ) * aspect );
	phi = atanf( alphay / alphax * tanf( 2.0f * PI * u.y + 0.5f * PI ) );
	if (u.y > 0.5f) phi += PI;
	float sinPhi = sin( phi ), cosPhi = cos( phi );
	float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
	float alpha2 = 1. / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
	float tanTheta2 = alpha2 * u.x / (1.0f - u.x);
	cosTheta = 1. / sqrt( 1.0f + tanTheta2 );
	float sinTheta = sqrt( max( 0.0f, 1.0f - cosTheta * cosTheta ) );
	float3 whLocal = sphericalDirection( sinTheta, cosTheta, sin( phi ), cos( phi ) );
	float3 wh = whLocal.x * X + whLocal.y * Y + whLocal.z * interaction.normal;
	if (!sameHemiSphere( wo, wh, interaction.normal )) wh *= -1.0f;
	wi = reflect( wo * -1.0f, wh );
}

__device__ void disneyClearCoatSample( float3& wi, const float3& wo, const float2& u, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	float gloss = mix( 0.1f, 0.001f, material.clearcoatGloss );
	float alpha2 = gloss * gloss;
	float cosTheta = sqrt( max( EPSILON, (1.0f - powf( alpha2, 1. - u.x )) / (1.0f - alpha2) ) );
	float sinTheta = sqrt( max( EPSILON, 1.0f - cosTheta * cosTheta ) );
	float phi = 2.0f * PI * u.y;
	float3 whLocal = sphericalDirection( sinTheta, cosTheta, sin( phi ), cos( phi ) );
	float3 tangent = make_float3( 0 ), binormal = make_float3( 0 );
	createBasis( interaction.normal, tangent, binormal );
	float3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * interaction.normal;
	if (!sameHemiSphere( wo, wh, interaction.normal )) wh *= -1.0f;
	wi = reflect( wo * -1.0f, wh );
}

__device__ float bsdfPdf( const float3& wi, const float3& wo, const float3& X, const float3& Y, const SurfaceInteraction& interaction, const MaterialInfo& material )
{
	float pdfDiffuse = pdfLambertianReflection( wi, wo, interaction.normal );
	float pdfMF = pdfMicrofacetAniso( wi, wo, X, Y, interaction, material );
	float pdfCC = pdfClearCoat( wi, wo, interaction, material );
	return (pdfDiffuse + pdfMF + pdfCC) * (1.0f / 3.0f);
}

__device__ float3 bsdfSample( float3& wi, const float3& wo, const float3& X, const float3& Y, float& pdf, const SurfaceInteraction& interaction, const MaterialInfo& material, uint& seed  )
{
	float3 f = make_float3( 0 );
	pdf = 0.0;
	wi = make_float3( 0 );
	float2 u = make_float2( RandomFloat( seed ), RandomFloat( seed ) );
	float rnd = RandomFloat( seed );
	if (rnd <= 0.3333) disneyDiffuseSample( wi, wo, pdf, u, interaction.normal, material );
	else if (rnd >= 0.3333 && rnd < 0.6666) disneyMicrofacetAnisoSample( wi, wo, X, Y, u, interaction, material );
	else disneyClearCoatSample( wi, wo, u, interaction, material );
	f = bsdfEvaluate( wi, wo, X, Y, interaction, material );
	pdf = bsdfPdf( wi, wo, X, Y, interaction, material );
	if (pdf < EPSILON) return make_float3( 0 );
	return f;
}

#endif