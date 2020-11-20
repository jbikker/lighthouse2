#include "ray.h"
#include "core_settings.h"
#include "limits"
#include "whitted_ray_tracer.h"
#include "vector"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}

float4 Ray::GetIntersectionPoint(float intersectionDistance) {
	return origin + (direction * intersectionDistance);
}

float4 Ray::Trace(int recursionDepth) {

	/** check if we reached our recursion depth */
	if (recursionDepth > WhittedRayTracer::recursionThreshold) {
		return make_float4(0, 0, 0, 0);
	}

	tuple<Primitive*, float> nearestIntersection = GetNearestIntersection();
	Primitive* nearestPrimitive = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);

	if (intersectionDistance > 0) {
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);
		float4 normal = nearestPrimitive->GetNormal(intersectionPoint);

		if (nearestPrimitive->material->type == Material::Type::Diffuse) {
			float energy = this->CalculateEnergyFromLights(intersectionPoint, normal);
			float4 color = nearestPrimitive->material->color * energy;
			return color;
		}

		if (nearestPrimitive->material->type == Material::Type::Mirror) {
			/** TODO: may need epsilon */
			this->origin = intersectionPoint;
			this->direction = normalize(this->direction - 2.0f * normal * dot(normal, this->direction));
			return this->Trace(recursionDepth + 1);
		}

	}

	return make_float4(0,0,0,0);
}


tuple<Primitive*, float> Ray::GetNearestIntersection() {
	float minDistance = NULL;
	Primitive* nearestPrimitive = NULL;

	for (int i = 0; i < WhittedRayTracer::scene.size(); i++) {
		Primitive* primitive = WhittedRayTracer::scene[i];
		float distance = primitive->Intersect(*this);

		if (((minDistance == NULL) || (distance < minDistance))
			&& (distance > 0)
			) {
			minDistance = distance;
			nearestPrimitive = primitive;
		}
	}

	return make_tuple(nearestPrimitive, minDistance);
}

float Ray::CalculateEnergyFromLights(const float4 intersectionPoint, float4 normal) {
	float energy = 0;

	for (int i = 0; i < WhittedRayTracer::lights.size(); i++) {
		Light* light = WhittedRayTracer::lights[i];
		float4 shadowRayDirection = normalize(light->origin - intersectionPoint);
		float shadowRayLength = length(light->origin - WhittedRayTracer::shadowRay.origin) - EPSILON;

		float distanceEnergy = light->intensity * (1 / (shadowRayLength * shadowRayLength));
		float angleFalloff = dot(normal, shadowRayDirection);

		/** check if there is enough energy to apply to the material */
		if (
			(angleFalloff > EPSILON) || (distanceEnergy > EPSILON)
		) {

			/** Adds additional length to prevent intersection to itself */
			WhittedRayTracer::shadowRay.origin = intersectionPoint + shadowRayDirection * EPSILON;
			WhittedRayTracer::shadowRay.direction = shadowRayDirection;

			tuple<Primitive*, float> nearestIntersection = WhittedRayTracer::shadowRay.GetNearestIntersection();
			Primitive* nearestPrimitive = get<0>(nearestIntersection);
			float intersectionDistance = get<1>(nearestIntersection);

			if (intersectionDistance != NULL && intersectionDistance < shadowRayLength) { continue; }

			energy += distanceEnergy * angleFalloff;
		}
	}
	return energy;
}
