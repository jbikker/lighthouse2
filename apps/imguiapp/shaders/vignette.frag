#version 330

uniform sampler2D color;

in vec2 uv;
out vec3 pixel;

void main()
{
	// retrieve input pixel
	pixel = texture( color, uv ).rgb;
}

// EOF