#include "bvhnode.h"
#include "whitted_ray_tracer.h"
#include "triangle.h"
#include "ray.h"
#include "tuple"

void BVHNode::SubdivideNode(BVHNode* pool, int* triangleIndices, int &poolPtr) {
	if (this->count <= 2) return;
	
	this->left = poolPtr;
	poolPtr += 2;
	
	this->PartitionTriangles(pool, triangleIndices);

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];
	
	if (left->count == 0 || right->count == 0) { return; }
	
	left->SubdivideNode(pool, triangleIndices, poolPtr);
	right->SubdivideNode(pool, triangleIndices, poolPtr);
	
	this->isLeaf = false;
}

void BVHNode::PartitionTriangles(BVHNode* pool, int* triangleIndices) {
	int axis = this->bounds.LongestAxis();
	this->splitAxis = axis;
	float splitPoint = this->bounds.Center(axis);

	int j = this->first;
	for (int i = this->first; i < this->first + this->count; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[i]];

		float trianglePoint;
		if (axis == 0) {
			trianglePoint = triangle->centroid.x;
		} else if (axis == 1) {
			trianglePoint = triangle->centroid.y;
		} else {
			trianglePoint = triangle->centroid.z;
		}

		if (trianglePoint < splitPoint) {
			this->Swap(triangleIndices, i, j);
			j++;
		}
	}

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];

	left->first = this->first;
	left->count = j - this->first;
	right->first = j;
	right->count = this->count - left->count;

	left->UpdateBounds(triangleIndices);
	right->UpdateBounds(triangleIndices);
}

void BVHNode::UpdateBounds(int* triangleIndices) {
	float4 bmin = make_float4(this->bounds.bmin[0], this->bounds.bmin[1], this->bounds.bmin[2], this->bounds.bmin[3]);
	float4 bmax = make_float4(this->bounds.bmax[0], this->bounds.bmax[1], this->bounds.bmax[2], this->bounds.bmax[3]);

	for (int i = 0; i < this->count; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[this->first + i]];

		bmin = fminf(bmin, fminf(triangle->v0, fminf(triangle->v1, triangle->v2)));
		bmax = fmaxf(bmax, fmaxf(triangle->v0, fmaxf(triangle->v1, triangle->v2)));
	}

	this->bounds.bmin[0] = bmin.x;
	this->bounds.bmin[1] = bmin.y;
	this->bounds.bmin[2] = bmin.z;

	this->bounds.bmax[0] = bmax.x;
	this->bounds.bmax[1] = bmax.y;
	this->bounds.bmax[2] = bmax.z;
}

void BVHNode::Traverse(Ray &ray, BVHNode* pool, int* triangleIndices, tuple<Triangle*, float> &intersection) {
	float intersect;
	if (!ray.IntersectionBounds(this->bounds, intersect)) { return; }
	
	Triangle* triangleIntersection = get<0>(intersection);
	float distanceIntersection = get<1>(intersection);
	if (triangleIntersection != NULL && intersect > distanceIntersection) { return; }

	if (this->isLeaf) {
		this->IntersectTriangles(ray, triangleIndices, intersection);
		return;
	}

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];

	float rayDirAxis;
	if (this->splitAxis == 0) {
		rayDirAxis = ray.direction.x;
	}
	else if (this->splitAxis == 1) {
		rayDirAxis = ray.direction.y;
	}
	else {
		rayDirAxis = ray.direction.z;
	}

	if (rayDirAxis > 0) {
		left->Traverse(ray, pool, triangleIndices, intersection);
		right->Traverse(ray, pool, triangleIndices, intersection);
	}
	else {
		right->Traverse(ray, pool, triangleIndices, intersection);
		left->Traverse(ray, pool, triangleIndices, intersection);
	}
}

void BVHNode::IntersectTriangles(Ray &ray, int* triangleIndices,  tuple<Triangle*, float> &intersection) {
	float minDistance = NULL;
	Triangle* nearestPrimitive = NULL;

	for (int i = 0; i < this->count; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[this->first + i]];
		float distance = triangle->Intersect(ray);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > EPSILON)
		) {
			minDistance = distance;
			nearestPrimitive = triangle;
		}
	}

	intersection = make_tuple(nearestPrimitive, minDistance);
}

void BVHNode::Swap(int* triangleIndices, int x, int y) {
	int prev = triangleIndices[x];
	triangleIndices[x] = triangleIndices[y];
	triangleIndices[y] = prev;
}
