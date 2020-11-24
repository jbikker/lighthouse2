#pragma once
#include "core_settings.h"
#include "light.h"
#include "tuple"
#include "vector"

class Triangle;
class WhittedRayTracer;

class Ray
{
public:
	Ray(float4 _origin, float4 _direction);
	float4 origin;
	float4 direction;
	float4 GetIntersectionPoint(float intersectionDistance);
	float4 Trace(uint recursionDepth = 0);
	tuple<Triangle*, float> GetNearestIntersection();
	float4 DetermineColor(Triangle* triangle, CoreMaterial* material, float4 intersectionPoint, uint recursionDepth);
};
