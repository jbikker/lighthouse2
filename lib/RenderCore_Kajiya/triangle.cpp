#include "triangle.h"
#include "kajiya_path_tracer.h"

Triangle::Triangle(float4 _v0, float4 _v1, float4 _v2, uint _material) {
	this->v0 = _v0;
	this->v1 = _v1;
	this->v2 = _v2;
	this->v0v2 = v2 - v0;
	this->v0v1 = v1 - v0;
	this->materialIndex = _material;
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

float4 Triangle::GetNormal() {
	return normalize(cross(v1 - v0, v2 - v0));
}

/** Gets a random point on the surface of the triangle */
float4 Triangle::GetRandomPoint() {
	float a = RandomFloat();
	float b = 1 - a;
	return this->v0 + a * (this->v1 - this->v0) + b * (this->v2 - this->v0);
}

float Triangle::GetArea() {
	/** Area code from: common_classes.h */
	const float a = length(this->v1 - this->v0);
	const float b = length(this->v2 - this->v1);
	const float c = length(this->v0 - this->v2);
	const float s = (a + b + c) * 0.5f;
	return sqrtf(s * (s - a) * (s - b) * (s - c));
}