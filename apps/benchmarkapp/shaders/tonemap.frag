#version 330

uniform sampler2D color;
uniform float contrast = 0;
uniform float brightness = 0;
uniform float gamma = 2.2f;
uniform int method = 4;
uniform float vignetting = 0.35f;

in vec2 uv;
out vec3 pixel;

float luminance( vec3 v ) { return dot( v, vec3( 0.2126f, 0.7152f, 0.0722f ) ); }
vec3 change_luminance( vec3 c_in, float l_out ) { float l_in = luminance( c_in ); return c_in * (l_out / l_in); }

// from: https://github.com/64/64.github.io/blob/src/code/tonemapping/tonemap.cpp

vec3 reinhard(vec3 v)
{
	return v / (1.0f + v);
}

vec3 reinhard_extended( vec3 v, float max_white )
{
	vec3 numerator = v * (1.0f + (v / vec3( max_white * max_white )));
	return numerator / (1.0f + v);
}

vec3 reinhard_extended_luminance( vec3 v, float max_white_l )
{
	float l_old = luminance( v );
	float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
	float l_new = numerator / (1.0f + l_old);
	return change_luminance( v, l_new );
}

vec3 reinhard_jodie( vec3 v )
{
	float l = luminance( v );
	vec3 tv = v / (1.0f + v);
	return mix( v / (1.0f + l), tv, tv );
}

vec3 uncharted2_tonemap_partial( vec3 x )
{
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 uncharted2_filmic( vec3 v )
{
	float exposure_bias = 2.0f;
	vec3 curr = uncharted2_tonemap_partial( v * exposure_bias );
	vec3 W = vec3( 11.2f );
	vec3 white_scale = vec3( 1.0f ) / uncharted2_tonemap_partial( W );
	return curr * white_scale;
}

vec3 tonemap( vec3 v )
{
	switch (method)
	{
	case 0: return clamp( v, 0.0f, 1.0f );
	case 1: return reinhard( v );
	case 2: return reinhard_extended( v, 6.0f );
	case 3: return reinhard_extended_luminance( v, 1.5f );
	case 4: return reinhard_jodie( v );
	case 5: default: return uncharted2_filmic( v );
	}
}

vec3 gamma_correct( vec3 v )
{
	float r = 1.0f / gamma;
	return vec3( pow( v.x, r ), pow( v.y, r ), pow( v.z, r ) );
}

vec3 adjust( vec3 v )
{
	// https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	float r = max( 0.0f, (v.x - 0.5f) * contrastFactor + 0.5f + brightness );
	float g = max( 0.0f, (v.y - 0.5f) * contrastFactor + 0.5f + brightness );
	float b = max( 0.0f, (v.z - 0.5f) * contrastFactor + 0.5f + brightness );
	return vec3( r, g, b );
}

float vignette( vec2 uv )
{
	// based on Keijiro Takahashi's Kino Vignette: https://github.com/keijiro/KinoVignette, via https://www.shadertoy.com/view/4lSXDm
	vec2 coord = (uv - 0.5f) * 2.0f;
    float rf = sqrt( dot( coord, coord ) ) * vignetting;
    float rf2_1 = rf * rf + 1.0f;
    return 1.0f / (rf2_1 * rf2_1);
}

vec3 chromatic( vec2 uv )
{
	// chromatic abberation inspired by knifa (lsKSDz), via https://www.shadertoy.com/view/XlSBRW
	vec2 d = abs( (uv - 0.5f) * 2.0f);    
	d.x = pow( d.x, 1.5f );
	d.y *= 0.1f;
	float dScale = 0.01;
	vec4 r, g, b;
	r = g = b = vec4( 0.0 );
	for (int i = 0; i < 2; ++i )
	{
		float rnd = i * 0.03f;
		r += texture( color, uv + d * rnd * dScale );
		g += texture( color, uv );
		b += texture( color, uv - d * rnd * dScale );
	}
	return 0.5f * vec3( r.r, g.g, b.b );
}

void main()
{
	// retrieve input pixel
	pixel = gamma_correct( tonemap( adjust( vignette( uv ) * chromatic( uv ) ) ) );
}

// EOF