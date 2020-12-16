#include "triangle.h"
#include "whitted_ray_tracer.h"
#include "ray.h"
#include "light.h"
#include "bvh.h"
#include "bvhnode.h"
#include "tuple"
#include "core_settings.h"

Triangle::Triangle(float4 _v0, float4 _v1, float4 _v2, uint _material) {
	this->v0 = _v0;
	this->v1 = _v1;
	this->v2 = _v2;
	this->v0v2 = v2 - v0;
	this->v0v1 = v1 - v0;
	this->materialIndex = _material;
	this->centroid = (this->v0 + this->v1 + this->v2) / 3.0;
	this->bounds.Grow(this->v0);
	this->bounds.Grow(this->v1);
	this->bounds.Grow(this->v2);
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

bool Triangle::IsLightBlocked(const BVH* bvh, float shadowRayLength) {
	tuple<Triangle*, float> intersection = make_tuple<Triangle*, float>(NULL, NULL);
	bvh->root->Traverse(WhittedRayTracer::shadowRay, bvh->pool, bvh->triangleIndices, intersection);

	Triangle* intersectionTriangle = get<0>(intersection);
	float intersectionDist = get<1>(intersection);

	if (
		intersectionTriangle != NULL && 
		intersectionDist > EPSILON &&
		intersectionDist < shadowRayLength &&
		WhittedRayTracer::materials[intersectionTriangle->materialIndex].refraction.value != 1
	) {
		return true; 
	}
	return false;
}

float Triangle::CalculateEnergyFromLights(const BVH* bvh, const float4 intersectionPoint) {
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

			if (Triangle::IsLightBlocked(bvh, shadowRayLength)) { continue; }

			energy += distanceEnergy * angleFalloff;
		}
	}
	return energy;
}