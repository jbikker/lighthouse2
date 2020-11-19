#include "triangle.h"

Triangle::Triangle(float4 _origin, Material* _material, float4 _v0, float4 _v1, float4 _v2) : Primitive(_origin, _material) {
	v0 = _origin + _v0;
	v1 = _origin + _v1;
	v2 = _origin + _v2;
	v0v2 = v2 - v0;
	v0v1 = v1 - v0;
}

float Triangle::Intersect(Ray& ray) {
	float4 pvec = make_float4(
		cross(make_float3(ray.direction), make_float3(v0v2)), 
		0
	);
	float det = dot(v0v1, pvec);

	if (det < EPSILON && det > -EPSILON) { return NULL; }

	float invDet = 1.0 / det;
	float4 tvec = ray.origin - v0;
	float u = dot(tvec, pvec) * invDet;

	if (u < 0 || u > 1) { return NULL; }

	float4 qvec = make_float4(
		cross(make_float3(tvec), make_float3(v0v1)), 
		0
	);
	float v = dot(ray.direction, qvec) * invDet;

	if (v < 0 || u + v > 1) { return NULL; }

	float distance = dot(v0v2, qvec) * invDet;

	return distance;
}

float4 Triangle::GetNormal(float4 point) {
	float3 q0 = make_float3(v0);
	float3 q1 = make_float3(v1);
	float3 q2 = make_float3(v2);

	return normalize(make_float4(cross(q1 - q0, q2 - q0), 0));
}
