#version 330

layout (location = 0) in vec4 pos;
layout (location = 1) in vec2 tuv;

uniform mat4 view;

out vec2 uv;

void main()
{
	uv = tuv;
	gl_Position = view * pos;
}

// EOF