#pragma once

#include "core_settings.h"
#include "tuple"
#include "vector"
#include "random"

class BVH;
class Triangle;
class Light;

class Ray
{

public:
	enum class HitType {
		Nothing,
		SceneObject,
		Light
	};

	Ray(float4 _origin, float4 _direction);
	float4 origin;
	float4 direction;
	float4 GetIntersectionPoint(float intersectionDistance);
	bool IntersectionBounds(aabb& bounds, float& distance);
	float4 Trace(BVH* bvh, uint recursionDepth = 0);
	tuple<Triangle*, float, HitType> IntersectLights(tuple<Triangle*, float, Ray::HitType> &intersection);
	float4 GetRefractionDirection(Triangle* triangle, CoreMaterial* material);
};
