#include "light.h"
#include "triangle.h"

Light::Light(Triangle triangle, float emmitance)
{
	this->shape = triangle;
	this->emmitance = emmitance;
}
