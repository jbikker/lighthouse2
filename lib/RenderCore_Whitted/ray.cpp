#include "ray.h"
#include "core_settings.h"
#include "whitted_ray_tracer.h"
#include "triangle.h"
#include "bvh.h"
#include "bvhnode.h"

Ray::Ray(float4 _origin, float4 _direction) {
	origin = _origin;
	direction = _direction;
}

float4 Ray::GetIntersectionPoint(float intersectionDistance) {
	return origin + (direction * intersectionDistance);
}

bool Ray::IntersectionBounds(aabb &bounds, float &distance) {
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
	/** Check if we reached our recursion depth */
	if (recursionDepth > WhittedRayTracer::recursionThreshold) {
		return make_float4(0, 0, 0, 0);
	}
	
	tuple<Triangle*, float> intersection = make_tuple<Triangle*, float>(NULL, NULL);
	bvh->root->Traverse(*this, bvh->pool, bvh->triangleIndices, intersection);

	Triangle* nearestTriangle = get<0>(intersection);
	float intersectionDistance = get<1>(intersection);

	/** If a triangle is hit, determine the color of the triangle and return it */
	if (nearestTriangle != NULL && intersectionDistance != NULL) {
		float4 intersectionPoint = this->GetIntersectionPoint(intersectionDistance);

		CoreMaterial* material = &WhittedRayTracer::materials[nearestTriangle->materialIndex];

		return Ray::DetermineColor(nearestTriangle, material, bvh, intersectionPoint, recursionDepth);
	}

	/** If no triangle is hit, return black */
	return make_float4(0, 0, 0, 0);
}

float4 Ray::DetermineColor(Triangle* triangle, CoreMaterial* material, BVH* bvh, float4 intersectionPoint, uint recursionDepth) {
	float reflection = material->reflection.value;
	float refraction = material->refraction.value;
	float diffuse = 1 - (reflection + refraction);

	float4 materialColor = make_float4(material->color.value, 0);
	float4 color = make_float4(0, 0, 0, 0);
	float4 normal = triangle->GetNormal();

	/** If material = diffuse apply diffuse color */
	if (diffuse > EPSILON) {
		float4 globalIlluminationColor = WhittedRayTracer::globalIllumination * make_float4(material->color.value, 0);
		float energy = triangle->CalculateEnergyFromLights(bvh, intersectionPoint);
		float4 diffuseColor = materialColor * energy;
		color += diffuse * diffuseColor;
		color += globalIlluminationColor;
	}
	/** If material = reflection apply reflection color */
	if (reflection > EPSILON) {
		float4 reflectDir = this->direction - 2.0f * normal * dot(normal, this->direction);
		this->origin = intersectionPoint + (reflectDir * EPSILON);
		this->direction = normalize(reflectDir);
		color += this->Trace(bvh, recursionDepth + 1) * reflection;
	}

	/** If material = refraction apply refraction color */
	if (refraction > EPSILON) {
		float4 refractionDirection = this->GetRefractionDirection(triangle, material);
		if (length(refractionDirection) > 0) {
			this->origin = intersectionPoint + (refractionDirection * EPSILON);
			this->direction = refractionDirection;
			color += this->Trace(bvh, recursionDepth + 1) * refraction;
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
	}
	else {
		return normalize(eta * this->direction + (eta * cosi - sqrtf(k)) * normalRefraction);
	}
}