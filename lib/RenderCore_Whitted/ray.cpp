#include "ray.h"
#include "core_settings.h"
#include "limits"
#include "triangle.h"
#include "whitted_ray_tracer.h"
#include "bvhnode.h"
#include "vector"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}

float4 Ray::GetIntersectionPoint(float intersectionDistance) {
	return origin + (direction * intersectionDistance);
}

tuple<Triangle*, float> Ray::GetNearestIntersection(vector<int> &triangleIndices) {
	float minDistance = NULL;
	Triangle* nearestPrimitive = NULL;

	for (int i = 0; i < triangleIndices.size(); i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[i]];
		float distance = triangle->Intersect(*this);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > EPSILON)
			) {
			minDistance = distance;
			nearestPrimitive = triangle;
		}
	}

	return make_tuple(nearestPrimitive, minDistance);
}

float4 Ray::DetermineColor(Triangle* triangle, CoreMaterial* material, BVHNode* root, float4 intersectionPoint, uint recursionDepth) {
	float reflection = material->reflection.value;
	float refraction = material->refraction.value;
	float diffuse    = 1 - (reflection + refraction);

	float4 materialColor = make_float4(material->color.value, 0);
	float4 color = make_float4(0,0,0,0);
	float4 normal = triangle->GetNormal();

	/** If material = diffuse apply diffuse color */
	if (diffuse > EPSILON) {
		float4 globalIlluminationColor = WhittedRayTracer::globalIllumination * make_float4(material->color.value, 0);
		float energy = triangle->CalculateEnergyFromLights(intersectionPoint);
		float4 diffuseColor = materialColor * energy;
		color += diffuse * diffuseColor;
		color += globalIlluminationColor;
	}
	/** If material = reflection apply reflection color */
	if (reflection > EPSILON) {
		float4 reflectDir = this->direction - 2.0f * normal * dot(normal, this->direction);
		this->origin = intersectionPoint + (reflectDir * EPSILON);
		this->direction = normalize(reflectDir);
		color += this->Trace(recursionDepth + 1) * reflection;
	}

	/** If material = refraction apply refraction color */
	if (refraction > EPSILON) {
		float4 refractionDirection = this->GetRefractionDirection(triangle, material);
		if (length(refractionDirection) > 0) {
			this->origin = intersectionPoint + (refractionDirection * EPSILON);
			this->direction = refractionDirection;
			color += this->Trace(recursionDepth + 1) * refraction;
		}

	}

	return color;
}

float4 Ray::GetRefractionDirection(Triangle* triangle, CoreMaterial* material) {
	float4 normal = triangle->GetNormal();
	float cosi = clamp(-1.0, 1.0, dot(this->direction, normal));
	float etai = 1;
	float etat = material->ior.value;
	float4 normalRefraction = normal;

	/** Outside the surface */
	if (cosi < 0) {
		cosi = -cosi;
	}
	/** Inside the surface */
	else {
		normalRefraction = -normal;
		std::swap(etai, etat);
	}

	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);

	if (k < 0) {
		return make_float4(0, 0, 0, 0);
	} else {
		return normalize(eta * this->direction + (eta * cosi - sqrtf(k)) * normalRefraction);
	}
}
