#include "sphere.h"
#include "core_settings.h"
#include "ray.h"

Sphere::Sphere(float4 _origin, float _radius2) : Primitive(_origin) {
	radius2 = _radius2;
}

void Sphere::Intersect(Ray& ray) {
	float4 directionSphere = this->origin - ray.origin;
	float distanceMappedOnRay = dot(directionSphere, ray.direction);
	float4 directionPerpendicular = directionSphere - (distanceMappedOnRay * ray.direction);
	float lengthPerpendicular = dot(directionPerpendicular, directionPerpendicular);
	if (lengthPerpendicular > this->radius2) return;
	distanceMappedOnRay -= sqrt(this->radius2 - lengthPerpendicular);
	if (
		(distanceMappedOnRay < ray.intersectionDistance) &&
		(distanceMappedOnRay > 0)
	) { 
		ray.intersectionDistance = distanceMappedOnRay;
	}
}