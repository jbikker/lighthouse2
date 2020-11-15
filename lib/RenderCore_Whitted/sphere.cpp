#include "sphere.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h";

Sphere::Sphere(float4 _origin, Material* _material, float _radius2) : Primitive(_origin, _material) {
	radius2 = _radius2;
}

float Sphere::Intersect(Ray& ray) {
	float4 directionSphere = this->origin - ray.origin;
	float distanceMappedOnRay = dot(directionSphere, ray.direction);
	float4 directionPerpendicular = directionSphere - (distanceMappedOnRay * ray.direction);
	float lengthPerpendicular = dot(directionPerpendicular, directionPerpendicular);
	
	if (lengthPerpendicular > this->radius2) return NULL;
	
	distanceMappedOnRay -= sqrt(this->radius2 - lengthPerpendicular);

	return distanceMappedOnRay;
}

float4 Sphere::GetNormal(float4 point)
{
	return normalize(point - origin);
}
