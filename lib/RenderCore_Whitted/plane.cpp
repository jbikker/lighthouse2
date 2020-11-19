#include "plane.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h";

Plane::Plane(float4 _origin, Material* _material, float4 _normal) : Primitive(_origin, _material) {
    normal = _normal;
}

float Plane::Intersect(Ray& ray) {
    float distDirection = dot(ray.direction, normal);
    float distPoints = dot(origin - ray.origin, normal);
    float distPlane = distPoints / distDirection;
    
    return distPlane;
}

float4 Plane::GetNormal(float4 point) {
    return normalize(normal);
}
