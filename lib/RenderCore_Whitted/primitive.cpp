#include "primitive.h"
#include "core_settings.h"
#include "material.h";
#include "whitted_ray_tracer.h";
#include "ray.h";

Primitive::Primitive(float4 _origin, Material* _material) {
	origin = _origin;
	material = _material;
}

bool Primitive::IsLightBlocked(float shadowRayLength) {
	for (int i = 0; i < WhittedRayTracer::scene.size(); i++) {
		Primitive* primitive = WhittedRayTracer::scene[i];
		float distance = primitive->Intersect(WhittedRayTracer::shadowRay);

		if (
			distance != NULL &&
			distance > EPSILON &&
			distance < shadowRayLength
		) {
			return true;
		}
	}
	return false;
}

float Primitive::CalculateEnergyFromLights(const float4 intersectionPoint) {
	float energy = 0;
	float4 normal = this->GetNormal(intersectionPoint);

	for (int i = 0; i < WhittedRayTracer::lights.size(); i++) {
		Light* light = WhittedRayTracer::lights[i];
		float4 shadowRayDirection = normalize(light->origin - intersectionPoint);
		float shadowRayLength = length(light->origin - WhittedRayTracer::shadowRay.origin) - (length(shadowRayDirection) * EPSILON);

		float distanceEnergy = light->intensity * (1 / (shadowRayLength * shadowRayLength));
		float angleFalloff = dot(normal, shadowRayDirection);

		/** check if there is enough energy to apply to the material */
		if (
			(angleFalloff > EPSILON) || (distanceEnergy > EPSILON)
		) {
			/** Adds additional length to prevent intersection to itself */
			WhittedRayTracer::shadowRay.origin = intersectionPoint + shadowRayDirection * EPSILON;
			WhittedRayTracer::shadowRay.direction = shadowRayDirection;

			if (Primitive::IsLightBlocked(shadowRayLength)) { continue; }

			energy += distanceEnergy * angleFalloff;
		}
	}
	return energy;
}