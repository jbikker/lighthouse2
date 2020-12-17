#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "light.h"
#include "triangle.h";
#include "tuple"
#include "vector"

/** 
  * Variables
  */

/** Scene */
vector<Triangle*> WhittedRayTracer::scene = vector<Triangle*>();
vector<Light*> WhittedRayTracer::lights = vector<Light*>();
vector<CoreMaterial> WhittedRayTracer::materials;
vector<BVH*> WhittedRayTracer::bvhs;

/** Global Illumitation */
float4 WhittedRayTracer::globalIllumination = make_float4(0.2, 0.2, 0.2, 0);

/** Rays */
Ray WhittedRayTracer::primaryRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
Ray WhittedRayTracer::shadowRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

/** Whitted Ray Tracer Settings */
int WhittedRayTracer::recursionThreshold = 3;
int WhittedRayTracer::antiAliasingAmount = 1;
float WhittedRayTracer::gammaCorrection = 2.2;
bool WhittedRayTracer::applyPostProcessing = false;

/**
  * Setup
  */

/** Intialise Whitted Ray Tracer */
void WhittedRayTracer::Initialise() {
	/** Lights */
	lights.push_back(new Light(
		make_float4(5, 20, 0, 0),
		200
	));
}

void WhittedRayTracer::AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex) {
	Triangle* triangle = new Triangle(v0, v1, v2, materialIndex);
	scene.push_back(triangle);
}

/**
  * Rendering
  */

/** Calculates the point on the camera screen given the x and y position */
float3 WhittedRayTracer::GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const float x, const float y) {
	float u = x / (float)screen->width;
	float v = y / (float)screen->height;
	float3 point = view.p1 + u * (view.p2 - view.p1) + v * (view.p3 - view.p1);
	return point;
}

/** Calculates the ray direction from the camera to the screen */
float4 WhittedRayTracer::GetRayDirection(const ViewPyramid& view, float3 point) {
	float3 originToPoint = point - view.pos;
	float3 rayDirection = normalize((originToPoint) / length(originToPoint));
	return make_float4(rayDirection, 0);
}

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			float4 pixelColor = make_float4(0, 0, 0, 0);

			/** Loop additionally for anti aliasing */
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
					pixelColor += primaryRay.Trace(WhittedRayTracer::bvhs[0], 0);
				}
			}

			/** Divide the color by the amount of extra anti aliasing rays */
			pixelColor /= WhittedRayTracer::antiAliasingAmount * WhittedRayTracer::antiAliasingAmount;
			
			int index = x + y * screen->width;
			screen->pixels[index] = WhittedRayTracer::ConvertColorToInt(pixelColor);
		}
	}


	if (WhittedRayTracer::applyPostProcessing) {
		WhittedRayTracer::ApplyPostProcessing(screen);
	}
}

/** 
  * Post Processing
  */

void WhittedRayTracer::ChromaticAbberation(const Bitmap* screen, float4& color, uint* pixels, int x, int y, float u, float v) {
	int abberationStrength = 50;
	int abberationPixels = abs((u - 0.5) * (v - 0.5)) * abberationStrength;
	int min = 0;
	int max = screen->width - 1;
	int pixelR = clamp((x + abberationPixels), min, max) + y * screen->width;
	int pixelB = clamp((x - abberationPixels), min, max) + y * screen->width;
	color.x = WhittedRayTracer::ConvertIntToColor(pixels[pixelR]).x;
	color.z = WhittedRayTracer::ConvertIntToColor(pixels[pixelB]).z;
}

void WhittedRayTracer::GammaCorrection(float4& color) {
	float gamma = WhittedRayTracer::gammaCorrection;
	color.x = pow(color.x, 1.0 / gamma);
	color.y = pow(color.y, 1.0 / gamma);
	color.z = pow(color.z, 1.0 / gamma);
}

void WhittedRayTracer::Vignetting(float4& color, float u, float v) {
	float uVig = u * (1.0 - u);
	float vVig = v * (1.0 - v);
	float vignette = uVig * vVig * 15.0;
	vignette = pow(vignette, 0.25);
	color *= vignette;
}

void WhittedRayTracer::ApplyPostProcessing(const Bitmap* screen) {
	/** Copy the screen pixels to prevent changing the screen while reading from it */
	uint* pixels = new uint[screen->width * screen->height];
	memcpy(pixels, screen->pixels, screen->width * screen->height * sizeof(uint));

	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			int index = x + y * screen->width;
			float u = ((float) x / (float) screen->width);
			float v = ((float) y / (float) screen->height);
			float4 color = WhittedRayTracer::ConvertIntToColor(screen->pixels[index]);

			WhittedRayTracer::ChromaticAbberation(screen, color, pixels, x, y, u, v);

			WhittedRayTracer::GammaCorrection(color);

			WhittedRayTracer::Vignetting(color, u, v);

			screen->pixels[index] = WhittedRayTracer::ConvertColorToInt(color);
		}
	}
}

/**
  * Helper Functions
  */

int WhittedRayTracer::ConvertColorToInt(float4 color) {
	int red = clamp((int)(color.x * 256), 0, 255);
	int green = clamp((int)(color.y * 256), 0, 255);
	int blue = clamp((int)(color.z * 256), 0, 255);
	return (blue << 16) + (green << 8) + red;
}

float4 WhittedRayTracer::ConvertIntToColor(int color) {
	float red = color & 0xFF;
	float green = (color >> 8) & 0xFF;
	float blue = (color >> 16) & 0xFF;
	return make_float4(red, green, blue, 0) / 255;
}
