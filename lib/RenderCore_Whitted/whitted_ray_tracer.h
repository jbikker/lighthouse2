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
	static vector<Triangle*> scene;
	static vector<Light*> lights;
	static vector<CoreMaterial> materials;
	
	static float4 globalIllumination;

	static Ray primaryRay;
	static Ray shadowRay;

	static int recursionThreshold;

	static void Initialise();
	static void AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex);
	static void Render(const ViewPyramid& view, const Bitmap* screen);
private:
	static int antiAliasingAmount;
	static bool applyPostProcessing;
	static float gammaCorrection;

	static float3 GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const float x, const float y);
	static float4 GetRayDirection(const ViewPyramid& view, float3 point);
	
	/** Post Processing */
	static void ChromaticAbberation(const Bitmap* screen, float4& color, uint* pixels, int x, int y, float u, float v);
	static void GammaCorrection(float4& color);
	static void Vignetting(float4& color, float u, float v);
	static void ApplyPostProcessing(const Bitmap* screen);

	/** Helper Functions */
	static int ConvertColorToInt(float4 color);
	static float4 ConvertIntToColor(int color);
};

