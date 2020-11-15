#include "whitted_ray_tracer.h"
#include "core_settings.h"
#include "ray.h"
#include "material.h"
#include "sphere.h"
#include "primitive.h"
#include "light.h"
#include "tuple"

const int PRIMITIVES_SIZE = 3;
Primitive** WhittedRayTracer::scene = new Primitive*[PRIMITIVES_SIZE] {
	new Sphere(
		make_float4(0, 0, 10, 0),  
		new Material(make_float4(1, 0, 0, 0)),
		3
	),
	new Sphere(
		make_float4(2, 0, 10, 0), 
		new Material(make_float4(0, 1, 0, 0)),
		3
	),
	new Sphere(
		make_float4(2, 2.5, 9, 0),
		new Material(make_float4(0, 0, 1, 0)),
		0.25
	)
};


const int LIGHTS_SIZE = 2;
Light** WhittedRayTracer::lights = new Light*[LIGHTS_SIZE]{
	new Light(
		make_float4(1, 3, 7, 0),
		100
	),
	new Light(
		make_float4(5, 3, 5, 0),
		100
	)
};

Ray WhittedRayTracer::primaryRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
Ray WhittedRayTracer::shadowRay = Ray(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));

void WhittedRayTracer::Render(const ViewPyramid& view, const Bitmap* screen) {
	primaryRay.origin = make_float4(view.pos, 0);

	for (int y = 0; y < screen->height; y++) {
		for (int x = 0; x < screen->width; x++) {
			float3 point = WhittedRayTracer::GetPointOnScreen(view, screen, x, y);
			float4 rayDirection = WhittedRayTracer::GetRayDirection(view, point);
			primaryRay.direction = rayDirection;

			tuple<Primitive*, float> nearestIntersection = WhittedRayTracer::GetNearestIntersection(primaryRay);
			Primitive* nearestPrimitive = get<0>(nearestIntersection);
			float intersectionDistance = get<1>(nearestIntersection);

			int index = x + y * screen->width;
			
			if (intersectionDistance > 0) {
				float4 intersectionPoint = primaryRay.GetIntersectionPoint(intersectionDistance);
				float4 normal = nearestPrimitive->GetNormal(intersectionPoint);
				float energy = WhittedRayTracer::CalculateEnergyFromLights(intersectionPoint, normal);
				float4 color = nearestPrimitive->material->color * energy;
				screen->pixels[index] = WhittedRayTracer::ConvertColorToInt(color);
			} else {
				screen->pixels[index] = 0;
			}
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

tuple<Primitive*, float> WhittedRayTracer::GetNearestIntersection(Ray& ray) {
	float minDistance = NULL;
	Primitive* nearestPrimitive = NULL;

	for (int i = 0; i < PRIMITIVES_SIZE; i++) {
		Primitive* primitive = WhittedRayTracer::scene[i];
		float distance = primitive->Intersect(ray);

		if (( (minDistance == NULL) || (distance < minDistance) )
			  && (distance > 0)
		) {
			minDistance = distance;
			nearestPrimitive = primitive;
		}
	}

	return make_tuple(nearestPrimitive, minDistance);
}

int WhittedRayTracer::ConvertColorToInt(float4 color) {
	int red = clamp((int)(color.x * 256), 0, 255);
	int green = clamp((int)(color.y * 256), 0, 255);
	int blue = clamp((int)(color.z * 256), 0, 255);
	// TODO: should be: red << 16 + green << 8 + blue
	return (blue << 16) + (green << 8) + red;
}

float WhittedRayTracer::CalculateEnergyFromLights(const float4 intersectionPoint, float4 normal)
{
	float energy = 0;
	for (int i = 0; i < LIGHTS_SIZE; i++) {
		Light* light = WhittedRayTracer::lights[i];
		float4 shadowRayDirection = normalize(light->origin - intersectionPoint);

		/** check if light can reach the surface */
		if (dot(normal, shadowRayDirection) > 0) {
			shadowRay.origin = intersectionPoint;
			shadowRay.direction = shadowRayDirection;
			float shadowRayLength = length(light->origin - shadowRay.origin);

			tuple<Primitive*, float> nearestIntersection = WhittedRayTracer::GetNearestIntersection(shadowRay);
			Primitive* nearestPrimitive = get<0>(nearestIntersection);
			float intersectionDistance = get<1>(nearestIntersection);

			 if (intersectionDistance < shadowRayLength) continue;

			/** energy based distance */
			float distanceEnergy = light->intensity * (1 / (4 * PI * (shadowRayLength * shadowRayLength)));
			float angleFalloff = dot(shadowRayDirection, normal);
			energy += distanceEnergy * angleFalloff;
		}
	}
	return energy;
}
