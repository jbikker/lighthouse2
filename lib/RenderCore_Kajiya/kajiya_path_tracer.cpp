#include "kajiya_path_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"
#include "triangle.h"
#include "light.h"
#include "tuple"
#include "vector"

vector<Triangle*> KajiyaPathTracer::scene = vector<Triangle*>();
vector<Triangle*> KajiyaPathTracer::lights = vector<Triangle*>();
vector<CoreMaterial> KajiyaPathTracer::materials;

float4 KajiyaPathTracer::globalIllumination = make_float4(0.2, 0.2, 0.2, 0);

void KajiyaPathTracer::Initialise() {
	/** Lights */
	lights.push_back(new Triangle(
		make_float4(0, 20, 0, 0),
		make_float4(100, 20, 0, 0),
		make_float4(0, 20, 100, 0),
		0
	));
}

int KajiyaPathTracer::stillFrames = 1;
float3 KajiyaPathTracer::oldCameraPos = make_float3(0, 0, 0);
float3 KajiyaPathTracer::oldCameraP1 = make_float3(0, 0, 0);
float3 KajiyaPathTracer::oldCameraP2 = make_float3(0, 0, 0);
float3 KajiyaPathTracer::oldCameraP3 = make_float3(0, 0, 0);


int KajiyaPathTracer::recursionThreshold = 3;
Ray KajiyaPathTracer::primaryRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
Ray KajiyaPathTracer::shadowRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

void KajiyaPathTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	bool cameraStill = view.pos == KajiyaPathTracer::oldCameraPos && view.p1 == KajiyaPathTracer::oldCameraP1 && view.p2 == KajiyaPathTracer::oldCameraP2 && view.p3 == KajiyaPathTracer::oldCameraP3;

	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			/** Setup the ray from the screen */
			float3 point = KajiyaPathTracer::GetPointOnScreen(view, screen, x, y);
			float4 rayDirection = KajiyaPathTracer::GetRayDirection(view, point);

			/** Reset the primary, it can be used as a reflective ray */
			primaryRay.origin = make_float4(view.pos, 0);
			primaryRay.direction = rayDirection;

			/** Trace the ray */
			float4 color = primaryRay.Trace();
			int index = x + y * screen->width;


			/** Camera moved */
			if (!cameraStill) {
				screen->pixels[index] = KajiyaPathTracer::ConvertColorToInt(color);
			}

			/** Converge */
			else {
				float4 oldColor = KajiyaPathTracer::ConvertIntToColor(screen->pixels[index]);
				screen->pixels[index] = KajiyaPathTracer::ConvertColorToInt(oldColor + ((color - oldColor) / (KajiyaPathTracer::stillFrames + 1)));
			}

		}
	}

	cout << "Amount of still frames: " << KajiyaPathTracer::stillFrames << "\n";
	KajiyaPathTracer::oldCameraPos = view.pos;
	KajiyaPathTracer::oldCameraP1 = view.p1;
	KajiyaPathTracer::oldCameraP2 = view.p2;
	KajiyaPathTracer::oldCameraP3 = view.p3;
	if (cameraStill) {
		KajiyaPathTracer::stillFrames++;
	}
	else {
		KajiyaPathTracer::stillFrames = 1;
	}

}

void KajiyaPathTracer::AddTriangle(float4 v0, float4 v1, float4 v2, uint materialIndex) {
	Triangle* triangle = new Triangle(v0, v1, v2, materialIndex);
	scene.push_back(triangle);
}

float3 KajiyaPathTracer::GetPointOnScreen(const ViewPyramid& view, const Bitmap* screen, const int x, const int y) {
	float u = (float)x / (float)screen->width;
	float v = (float)y / (float)screen->height;
	float3 point = view.p1 + u * (view.p2 - view.p1) + v * (view.p3 - view.p1);
	return point;
}

float4 KajiyaPathTracer::GetRayDirection(const ViewPyramid& view, float3 point) {
	float3 originToPoint = point - view.pos;
	float3 rayDirection = normalize((originToPoint) / length(originToPoint));
	return make_float4(rayDirection, 0);
}

int KajiyaPathTracer::ConvertColorToInt(float4 color) {
	int red = clamp((int)(color.x * 256), 0, 255);
	int green = clamp((int)(color.y * 256), 0, 255);
	int blue = clamp((int)(color.z * 256), 0, 255);
	return (blue << 16) + (green << 8) + red;
}

float4 KajiyaPathTracer::ConvertIntToColor(int color) {
	float red = color & 0xFF;
	float green = (color >> 8) & 0xFF;
	float blue = (color >> 16) & 0xFF;
	return make_float4(red, green, blue, 0) / 255;
}
