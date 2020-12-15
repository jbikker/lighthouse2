#include "ray.h"
#include "core_settings.h"
#include "limits"
#include "triangle.h"
#include "kajiya_path_tracer.h"
#include "bvh.h"
#include "bvhnode.h"
#include "vector"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}

float4 Ray::GetIntersectionPoint(float intersectionDistance) {
	return origin + (direction * intersectionDistance);
}

bool Ray::IntersectionBounds(aabb& bounds, float& distance) {
	float4 invDir = 1.0 / this->direction;

	float4 bmin = make_float4(bounds.bmin[0], bounds.bmin[1], bounds.bmin[2], bounds.bmin[3]);
	float4 bmax = make_float4(bounds.bmax[0], bounds.bmax[1], bounds.bmax[2], bounds.bmax[3]);

	float4 t1 = (bmin - this->origin) * invDir;
	float4 t2 = (bmax - this->origin) * invDir;

	float4 tmin = fminf(t1, t2);
	float4 tmax = fmaxf(t1, t2);

	float dmin = max(tmin.x, max(tmin.y, tmin.z));
	float dmax = min(tmax.x, min(tmax.y, tmax.z));

	if (dmax < 0 || dmin > dmax) {
		return false;
	}
	distance = dmin;
	return true;
}

float4 Ray::Trace(BVH* bvh, uint recursionDepth) {
	/** check if we reached our recursion depth */
	if (recursionDepth > KajiyaPathTracer::recursionThreshold) {
		return make_float4(0, 0, 0, 0);
	}

	tuple<Triangle*, float> nearestIntersection = make_tuple<Triangle*, float>(NULL, NULL);
	bvh->root->Traverse(*this, bvh->pool, bvh->triangleIndices, nearestIntersection);

	//tuple<Triangle*, float, Ray::HitType> nearestIntersection = Ray::GetNearestIntersection();
	Triangle* nearestTriangle = get<0>(nearestIntersection);
	float intersectionDistance = get<1>(nearestIntersection);
	//Ray::HitType hitType = get<2>(nearestIntersection);
	Ray::HitType hitType = Ray::HitType::Light;

	if (intersectionDistance > 0) {
		/** Hit a light */
		if (hitType == Ray::HitType::Light) {
			return make_float4(KajiyaPathTracer::materials[nearestTriangle->materialIndex].color.value, 0);
		}

		CoreMaterial material = KajiyaPathTracer::materials[nearestTriangle->materialIndex];
		float4 normal = nearestTriangle->GetNormal();
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);

		float randomChoice = RandomFloat();

		/** If material = reflection, given a certain chance it calculates the reflection color */
		float reflectionChance = KajiyaPathTracer::materials[nearestTriangle->materialIndex].reflection.value;
		if (randomChoice < reflectionChance) {
			this->direction = this->direction - 2.0f * normal * dot(normal, this->direction);
			this->origin = intersectionPoint + EPSILON * this->direction;
			return this->Trace(bvh, recursionDepth + 1);
		}

		/** If material = refraction, given a certain chance it calculates the refraction color */
		float refractionChance = KajiyaPathTracer::materials[nearestTriangle->materialIndex].refraction.value;
		if (randomChoice < refractionChance + reflectionChance) {
			float4 refractionDirection = this->GetRefractionDirection(nearestTriangle, &material);
			if (length(refractionDirection) > EPSILON) {
				this->origin = intersectionPoint + (refractionDirection * EPSILON);
				this->direction = refractionDirection;
				return this->Trace(bvh, recursionDepth + 1);
			}
		}

		/** Calculate a random direction on the hempisphere */
		float x = RandomFloat();
		float y = RandomFloat();
		float4 uniformSample = normalize(make_float4(UniformSampleSphere(x, y)));
		
		/** Flips the direction away from the normal if needed */
		float4 r = this->direction = (dot(uniformSample, normal) > 0) ? uniformSample : -uniformSample;
		this->origin = intersectionPoint + (this->direction * EPSILON);
		
		float4 hitColor = this->Trace(bvh, recursionDepth + 1);

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



