#include "ray.h"
#include "core_settings.h"
#include "limits"
#include "triangle.h"
#include "kajiya_path_tracer.h"
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
	if (recursionDepth > KajiyaPathTracer::recursionThreshold) {
		return make_float4(0, 0, 0, 0);
	}

	tuple<Triangle*, float, Ray::HitType> nearestIntersection = Ray::GetNearestIntersection();
	Triangle* nearestTriangle = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);
	Ray::HitType hitType = get<2>(nearestIntersection);

	if (intersectionDistance > 0) {

		/** Hit a light */
		if (hitType == Ray::HitType::Light) {
			return make_float4(KajiyaPathTracer::materials[nearestTriangle->materialIndex].color.value, 0);
		}

		CoreMaterial material = KajiyaPathTracer::materials[nearestTriangle->materialIndex];
		float4 normal = nearestTriangle->GetNormal();
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);

		float randomChoice = ((float)rand()) / (float)RAND_MAX;

		/** Hit a mirror */
		float reflectionChance = KajiyaPathTracer::materials[nearestTriangle->materialIndex].reflection.value;
		if (randomChoice < reflectionChance) {
			this->direction = this->direction - 2.0f * normal * dot(normal, this->direction);
			this->origin = intersectionPoint + EPSILON * this->direction;
			return this->Trace(recursionDepth + 1);
		}

		/** Hit a glass */
		float refractionChance = KajiyaPathTracer::materials[nearestTriangle->materialIndex].refraction.value;
		if (randomChoice < refractionChance + reflectionChance) {
			float4 refractionDirection = this->GetRefractionDirection(nearestTriangle, &material);
			if (length(refractionDirection) > EPSILON) {
				this->origin = intersectionPoint + (refractionDirection * EPSILON);
				this->direction = refractionDirection;
				return this->Trace(recursionDepth + 1);
			}
		}

		/** hit a random point on the hemisphere */
		float x = ((float) rand()) / (float) RAND_MAX;
		float y = ((float) rand()) / (float) RAND_MAX;
		float4 uniformSample = normalize(make_float4(UniformSampleSphere(x, y)));
		float4 r = this->direction = (dot(uniformSample, normal) > 0) ? uniformSample : -uniformSample;
		this->origin = intersectionPoint + (this->direction * EPSILON);
		float4 hitColor = this->Trace(recursionDepth + 1);

		float4 BRDF = make_float4(material.color.value / PI, 0);
		return dot(r, normal) * BRDF * hitColor * 2.0 * PI;
	}

	return make_float4(0,0,0,0);
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
	}
	else {
		return normalize(eta * this->direction + (eta * cosi - sqrtf(k)) * normalRefraction);
	}

}

tuple<Triangle*, float, Ray::HitType> Ray::GetNearestIntersection() {
	float minDistance = NULL;
	Triangle* nearestPrimitive = NULL;
	Ray::HitType hitType = Ray::HitType::Nothing;

	/** Intersect scene objects */
	for (int i = 0; i < KajiyaPathTracer::scene.size(); i++) {
		Triangle* triangle = KajiyaPathTracer::scene[i];
		float distance = triangle->Intersect(*this);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > 0)
		) {
			minDistance = distance;
			nearestPrimitive = triangle;
			hitType = Ray::HitType::SceneObject;
		}
	}

	/** Intersect lights */
	for (int i = 0; i < KajiyaPathTracer::lights.size(); i++) {
		Triangle* triangle = KajiyaPathTracer::lights[i];
		float distance = triangle->Intersect(*this);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > 0)
			) {
			minDistance = distance;
			nearestPrimitive = triangle;
			hitType = Ray::HitType::Light;
		}
	}

	return make_tuple(nearestPrimitive, minDistance, hitType);
}



