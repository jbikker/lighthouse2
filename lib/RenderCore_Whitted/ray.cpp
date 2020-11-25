#include "ray.h"
#include "core_settings.h"
#include "limits"
#include "triangle.h"
#include "whitted_ray_tracer.h"
#include "vector"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}

float4 Ray::GetIntersectionPoint(float intersectionDistance) {
	return origin + (direction * intersectionDistance);
}

float4 Ray::Trace(uint recursionDepth) {
	/** check if we reached our recursion depth */
	if (recursionDepth > WhittedRayTracer::recursionThreshold) {
		return make_float4(0, 0, 0, 0);
	}

	tuple<Triangle*, float> nearestIntersection = Ray::GetNearestIntersection();
	Triangle* nearestTriangle = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);

	if (intersectionDistance > 0) {
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);

		CoreMaterial* material = &WhittedRayTracer::materials[nearestTriangle->materialIndex];

		float4 globalIlluminationColor = WhittedRayTracer::globalIllumination * make_float4(material->color.value, 0);

		return Ray::DetermineColor(nearestTriangle, material, intersectionPoint, recursionDepth) + globalIlluminationColor;
	}

	return make_float4(0,0,0,0);
}

float4 Ray::DetermineColor(Triangle* triangle, CoreMaterial* material, float4 intersectionPoint, uint recursionDepth) {
	float reflection = material->reflection.value;
	float refraction = material->refraction.value;
	float diffuse    = 1 - (reflection + refraction);

	float4 materialColor = make_float4(material->color.value, 0);
	float4 color = make_float4(0,0,0,0);

	if (diffuse > EPSILON) {
		float energy = triangle->CalculateEnergyFromLights(intersectionPoint);
		float4 diffuseColor = materialColor * energy;
		color += diffuse * diffuseColor;
	}

	if (reflection > EPSILON) {
		float4 normal = triangle->GetNormal();
		float4 reflectDir = this->direction - 2.0f * normal * dot(normal, this->direction);
		this->origin = intersectionPoint + (reflectDir * EPSILON);
		this->direction = normalize(reflectDir);
		color += this->Trace(recursionDepth + 1) * reflection;
	}

	return color;
}

tuple<Triangle*, float> Ray::GetNearestIntersection() {
	float minDistance = NULL;
	Triangle* nearestPrimitive = NULL;

	for (int i = 0; i < WhittedRayTracer::scene.size(); i++) {
		Triangle* triangle = WhittedRayTracer::scene[i];
		float distance = triangle->Intersect(*this);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > 0)
		) {
			minDistance = distance;
			nearestPrimitive = triangle;
		}
	}

	return make_tuple(nearestPrimitive, minDistance);
}



