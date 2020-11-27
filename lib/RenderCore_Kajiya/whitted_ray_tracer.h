#pragma once
#include "core_settings.h"
#include "ray.h"
#include "light.h"
#include "tuple"
#include "vector"
#include "triangle.h";

class WhittedRayTracer
{
public:
	static int recursionThreshold;
	static vector<Triangle*> scene;
	static vector<Light*> lights;
	static vector<CoreMaterial> materials;

	static Ray primaryRay;
	static Ray shadowRay;

	static float4 globalIllumination;

	static void Initialise();
	static void AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex);
	static void Render(const ViewPyramid& view, const Bitmap* screen);
private:
	static float3 GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const int x, const int y);
	static float4 GetRayDirection(const ViewPyramid& view, float3 point);
	static int ConvertColorToInt(float4 color);
};

