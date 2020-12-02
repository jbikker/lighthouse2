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

float4 WhittedRayTracer::globalIllumination = make_float4(0.2, 0.2, 0.2, 0);

void WhittedRayTracer::Initialise() {
	/** Lights */
	lights.push_back(new Light(
		make_float4(-5, 20, 0, 0),
		200
	));
}

int WhittedRayTracer::recursionThreshold = 3;
int WhittedRayTracer::antiAliasingAmount = 1;
Ray WhittedRayTracer::primaryRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
Ray WhittedRayTracer::shadowRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			float4 pixelColor = make_float4(0, 0, 0, 0);

			for (int j = 0; j < WhittedRayTracer::antiAliasingAmount; j++) {
				for (int i = 0; i < WhittedRayTracer::antiAliasingAmount; i++) {
					/** Setup the ray from the screen */
					float u = (float)x + ((float)i / WhittedRayTracer::antiAliasingAmount);
					float v = (float)y + ((float)j / WhittedRayTracer::antiAliasingAmount);
					float3 point = WhittedRayTracer::GetPointOnScreen(view, screen, u, v);
					float4 rayDirection = WhittedRayTracer::GetRayDirection(view, point);

					/** Reset the primary, it can be used as a reflective ray */
					primaryRay.origin = make_float4(view.pos, 0);
					primaryRay.direction = rayDirection;

					/** Trace the ray */
					pixelColor += primaryRay.Trace();
				}
			}

			pixelColor /= WhittedRayTracer::antiAliasingAmount * WhittedRayTracer::antiAliasingAmount;
			WhittedRayTracer::ApplyPostProcessing(screen, x, y, pixelColor);

			int index = x + y * screen->width;
			screen->pixels[index] = WhittedRayTracer::ConvertColorToInt(pixelColor);
		}
	}

}

void WhittedRayTracer::AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex) {
	Triangle* triangle = new Triangle(v0, v1, v2, materialIndex);
	scene.push_back(triangle);
}

float3 WhittedRayTracer::GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const float x, const float y) {
	float u = x / (float)screen->width;
	float v = y / (float)screen->height;
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

void WhittedRayTracer::ApplyPostProcessing(const Bitmap* screen, int x, int y, float4& color) {
		float u = ((float) x / (float) screen->width) - 0.5;
		float v = ((float) y / (float) screen->height) - 0.5;
		
		/** Vignette */
		float vignette = abs(u * v);
		// vignette = pow(0.5, vignette);
		color -= vignette;
}
