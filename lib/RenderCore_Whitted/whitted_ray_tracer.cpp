#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"
#include "sphere.h"

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	Ray ray = Ray(make_float4(view.pos, 0), make_float4(0, 0, 0, 0));
	
	Sphere sphere = Sphere(make_float4(view.pos, 0) + make_float4(0, 0, 10, 0), 3);
	Sphere sphere2 = Sphere(make_float4(view.pos, 0) + make_float4(2, 0, 10, 0), 3);

	for (int j = 0; j < screen->height; j++) {
		for (int i = 0; i < screen->width; i++) {

			float3 point = WhittedRayTracer::GetPointOnScreen(view, screen, i, j);
			float4 rayDirection = WhittedRayTracer::GetRayDirection(view, point);
			ray.direction = rayDirection;

			sphere.Intersect(ray);
			sphere2.Intersect(ray);

			int index = i + j * screen->width;
			if (ray.intersectionDistance > 0) {
				screen->pixels[index] = 255 << 8;
			} else {
				screen->pixels[index] = 0;
			}

			ray.intersectionDistance = NULL;
		}
	}
}

float3 WhittedRayTracer::GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const int x, const int y) {
	float u = (float)x / (float)screen->width;
	float v = (float)y / (float)screen->height;

	float3 point = view.p1 + u * (view.p2 - view.p1) + v * (view.p3 - view.p1);

	return point;
}

float4 WhittedRayTracer::GetRayDirection(const ViewPyramid& view, float3 point) {
	float3 originToPoint = point - view.pos;
	float3 rayDirection = normalize((originToPoint) / length(originToPoint));
	return make_float4(rayDirection, 0);
}