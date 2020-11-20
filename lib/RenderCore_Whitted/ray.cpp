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

	tuple<Primitive*, float> nearestIntersection = Ray::GetNearestIntersection();
	Primitive* nearestPrimitive = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);

	if (intersectionDistance > 0) {
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);

		switch (nearestPrimitive->material->type) {
			case Material::Type::Mirror:
				float4 normal = nearestPrimitive->GetNormal(intersectionPoint);
				float4 reflectDir = this->direction - 2.0f * normal * dot(normal, this->direction);
				this->origin = intersectionPoint + (reflectDir * EPSILON);
				this->direction = normalize(reflectDir);
				return this->Trace(recursionDepth + 1);
			
			default:
			case Material::Type::Diffuse:
				float energy = nearestPrimitive->CalculateEnergyFromLights(intersectionPoint);
				float4 color = nearestPrimitive->material->color * energy;
				return color;
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

