#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"
#include "triangle.h"
#include "light.h"
#include "tuple"
#include "vector"

vector<Triangle*> WhittedRayTracer::scene = vector<Triangle*>();
vector<Light*> WhittedRayTracer::lights = vector<Light*>();
vector<CoreMaterial> WhittedRayTracer::materials;

float4 WhittedRayTracer::globalIllumination = make_float4(0.05, 0.05, 0.05, 0);

void WhittedRayTracer::Initialise() {
	/** Lights */
	lights.push_back(new Light(
		make_float4(0, 20, 0, 0),
		100
	));
}

int WhittedRayTracer::recursionThreshold = 5;
Ray WhittedRayTracer::primaryRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
Ray WhittedRayTracer::shadowRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			/** Setup the ray from the screen */
			float3 point = WhittedRayTracer::GetPointOnScreen(view, screen, x, y);
			float4 rayDirection = WhittedRayTracer::GetRayDirection(view, point);

			/** Reset the primary, it can be used as a reflective ray */
			primaryRay.origin = make_float4(view.pos, 0);
			primaryRay.direction = rayDirection;

			/** Trace the ray */
			float4 color = primaryRay.Trace() + WhittedRayTracer::globalIllumination;

			int index = x + y * screen->width;
			screen->pixels[index] = WhittedRayTracer::ConvertColorToInt(color);
		}
	}
}

void WhittedRayTracer::AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex) {
	Triangle* triangle = new Triangle(v0, v1, v2, materialIndex);
	scene.push_back(triangle);
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

int WhittedRayTracer::ConvertColorToInt(float4 color) {
	int red = clamp((int)(color.x * 256), 0, 255);
	int green = clamp((int)(color.y * 256), 0, 255);
	int blue = clamp((int)(color.z * 256), 0, 255);
	return (blue << 16) + (green << 8) + red;
}
