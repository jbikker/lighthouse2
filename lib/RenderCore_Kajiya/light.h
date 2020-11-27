#pragma once
#include "core_settings.h"
#include "triangle.h"

class Light
{
public:
	float emmitance;
	Triangle* shape;
	Light(Triangle* triangle, float emmitance);
};

