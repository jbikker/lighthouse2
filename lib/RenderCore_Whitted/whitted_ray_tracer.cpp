#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"
#include "sphere.h"
#include "primitive.h"

Primitive** WhittedRayTracer::scene = new Primitive*[2] {
	new Sphere(
		make_float4(0, 0, 10, 0),  
		new Material(make_float4(255, 0, 0, 0)),
		3
	),
	new Sphere(
		make_float4(2, 0, 10, 0), 
		new Material(make_float4(0, 255, 0, 0)),
		3
	)
};

Ray WhittedRayTracer::ray = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	ray.origin = make_float4(view.pos, 0);

	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			float3 point = WhittedRayTracer::GetPointOnScreen(view, screen, x, y);
			float4 rayDirection = WhittedRayTracer::GetRayDirection(view, point);
			ray.direction = rayDirection;

			for (int i = 0; i < 2; i++) {
				Primitive* primitive = WhittedRayTracer::scene[i];
				primitive->Intersect(ray);
			}

			int index = x + y * screen->width;
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