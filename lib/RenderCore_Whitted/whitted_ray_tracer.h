#include "core_settings.h"
#include "primitive.h"
#include "ray.h"

#pragma once
class WhittedRayTracer
{
public:
	static void Render(const ViewPyramid& view, const Bitmap* screen);
private:
	static Primitive** scene;
	static Ray ray;
	static float3 GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const int x, const int y);
	static float4 GetRayDirection(const ViewPyramid& view, float3 point);
};

