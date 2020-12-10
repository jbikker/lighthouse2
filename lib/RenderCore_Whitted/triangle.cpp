#include "triangle.h"
#include "whitted_ray_tracer.h"

Triangle::Triangle(float4 _v0, float4 _v1, float4 _v2, uint _material) {
	this->v0 = _v0;
	this->v1 = _v1;
	this->v2 = _v2;
	this->v0v2 = v2 - v0;
	this->v0v1 = v1 - v0;
	this->materialIndex = _material;
}

float4 Triangle::GetNormal() {
	return normalize(cross(v1 - v0, v2 - v0));
}

float Triangle::Intersect(Ray& ray) {
	float4 pvec = cross(ray.direction, v0v2);
	float det = dot(v0v1, pvec);

	if (det < EPSILON && det > -EPSILON) { return NULL; }

	float invDet = 1.0 / det;
	float4 tvec = ray.origin - v0;
	float u = dot(tvec, pvec) * invDet;

	if (u < 0 || u > 1) { return NULL; }

	float4 qvec = cross(tvec, v0v1);
	float v = dot(ray.direction, qvec) * invDet;

	if (v < 0 || u + v > 1) { return NULL; }

	float distance = dot(v0v2, qvec) * invDet;

	return distance;
}

bool Triangle::IsLightBlocked(float shadowRayLength) {
	for (int i = 0; i < WhittedRayTracer::scene.size(); i++) {
		Triangle* triangle = WhittedRayTracer::scene[i];
		float distance = triangle->Intersect(WhittedRayTracer::shadowRay);

		if (
			distance != NULL &&
			distance > EPSILON &&
			distance < shadowRayLength &&
			WhittedRayTracer::materials[triangle->materialIndex].refraction.value != 1
		) {
			return true;
		}
	}
	return false;
}

float Triangle::CalculateEnergyFromLights(const float4 intersectionPoint) {
	float energy = 0;
	float4 normal = this->GetNormal();

	for (int i = 0; i < WhittedRayTracer::lights.size(); i++) {
		Light* light = WhittedRayTracer::lights[i];
		/** Calculate the direction from the intersection point to the light */
		float4 shadowRayDirection = normalize(light->origin - intersectionPoint);
		float shadowRayLength = length(light->origin - WhittedRayTracer::shadowRay.origin) - (length(shadowRayDirection) * EPSILON);

		float distanceEnergy = light->intensity * (1 / (shadowRayLength * shadowRayLength));
		float angleFalloff = dot(normal, shadowRayDirection);

		/** check if there is enough energy to apply to the material */
		if (
			(angleFalloff > EPSILON) || (distanceEnergy > EPSILON)
		) {
			/** Adds additional length to prevent intersection with itself */
			WhittedRayTracer::shadowRay.origin = intersectionPoint + shadowRayDirection * EPSILON;
			WhittedRayTracer::shadowRay.direction = shadowRayDirection;

			if (Triangle::IsLightBlocked(shadowRayLength)) { continue; }

			energy += distanceEnergy * angleFalloff;
		}
	}
	return energy;
}