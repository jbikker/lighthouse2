#version 330

uniform sampler2D color;
uniform float contrast = 0;
uniform float brightness = 0;
uniform float gamma = 2.2f;

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
	// return clamp( v, 0.0f, 1.0f );
	// return reinhard( v );
	// return reinhard_extended( v, 6.0f );
	return reinhard_extended_luminance( v, 1.5f );
	// return reinhard_jodie( v );
	// return uncharted2_filmic( v );
	// return const_luminance_reinhard( v );
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

void main()
{
	// retrieve input pixel
	pixel = gamma_correct( tonemap( adjust( texture( color, uv ).rgb ) ) );
}

// EOF