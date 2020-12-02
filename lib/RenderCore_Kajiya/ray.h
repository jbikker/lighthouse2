#pragma once
#include "core_settings.h"
#include "tuple"
#include "vector"
#include "random"

class KajiyaPathTracer;
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
	float4 Trace(uint recursionDepth = 0);
	tuple<Triangle*, float, HitType> GetNearestIntersection();
	float4 GetRefractionDirection(Triangle* triangle, CoreMaterial* material);
};
