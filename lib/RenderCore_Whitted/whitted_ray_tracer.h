#include "core_settings.h"
#include "primitive.h"
#include "ray.h"
#include "light.h"
#include "tuple"

#pragma once
class WhittedRayTracer
{
public:
	static void Render(const ViewPyramid& view, const Bitmap* screen);
private:
	static Primitive** scene;
	static Light** lights;
	static Ray primaryRay;
	static Ray shadowRay;
	static float3 GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const int x, const int y);
	static float4 GetRayDirection(const ViewPyramid& view, float3 point);
	static tuple<Primitive*, float> GetNearestIntersection(Ray& ray);
	static int ConvertColorToInt(float4 color);
	static float CalculateEnergyFromLights(float4 intersectionPoint, float4 normal);
};

