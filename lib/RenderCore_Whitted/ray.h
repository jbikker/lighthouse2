#pragma once

#include "core_settings.h"
#include "tuple"
#include "vector"

class BVH;
class Triangle;
class Light;

class Ray
{
public:
	Ray(float4 _origin, float4 _direction);
	float4 origin;
	float4 direction;
	float4 GetIntersectionPoint(float intersectionDistance);
	bool IntersectionBounds(aabb &bounds, float &distance);
	float4 Trace(BVH* bvh, uint recursionDepth);
	float4 DetermineColor(Triangle* triangle, CoreMaterial* material, BVH* bvh, float4 intersectionPoint, uint recursionDepth);
	float4 GetRefractionDirection(Triangle* triangle, CoreMaterial* material);
};