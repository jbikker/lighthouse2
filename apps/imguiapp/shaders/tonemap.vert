#version 330

layout (location = 0) in vec4 pos;
layout (location = 1) in vec2 tuv;

uniform mat4 view;

out vec2 uv;
out vec2 P;

void main()
{
	uv = tuv;
	P = vec2( pos ) * 0.5 + vec2( 0.5, 0.5 );
	gl_Position = view * pos;
}

// EOF