#pragma once
#include "core_settings.h"
#include "light.h"
#include "bvhnode.h"
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
	tuple<Triangle*, float> GetNearestIntersection(vector<int> &triangleIndices);
	float4 DetermineColor(Triangle* triangle, CoreMaterial* material, float4 intersectionPoint, BVHNode* root, uint recursionDepth);
	float4 GetRefractionDirection(Triangle* triangle, CoreMaterial* material);
};
