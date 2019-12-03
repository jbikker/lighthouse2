#version 330

uniform sampler2D color;

in vec2 uv;
out vec4 pixel;

void main()
{
	vec4 p = texture( color, uv );
	pixel = vec4(p.rgb, p.r );
}

// EOF