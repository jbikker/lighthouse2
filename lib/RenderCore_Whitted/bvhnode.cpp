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
	this->bounds = aabb();

	for (int i = 0; i < this->count; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[this->first + i]];

		this->UpdateBounds(triangle->v0);
		this->UpdateBounds(triangle->v1);
		this->UpdateBounds(triangle->v2);
	}
}

void BVHNode::UpdateBounds(float4 point) {
	this->bounds.Grow(make_float3(point));
}

void BVHNode::Traverse(Ray &ray, BVHNode* pool, int* triangleIndices, tuple<Triangle*, float> &intersection) {
	float distBoundingBox;
	if (!ray.IntersectionBounds(this->bounds, distBoundingBox)) { return; }
	
	Triangle* triangleIntersection = get<0>(intersection);
	float distTriangle = get<1>(intersection);
	if (triangleIntersection != NULL && distBoundingBox > distTriangle) { return; }

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

	if (nearestPrimitive != NULL && (
		get<0>(intersection) == NULL ||
		(minDistance > EPSILON && minDistance < get<1>(intersection))
	)) {
		intersection = make_tuple(nearestPrimitive, minDistance);
	}
}

void BVHNode::Swap(int* triangleIndices, int x, int y) {
	int prev = triangleIndices[x];
	triangleIndices[x] = triangleIndices[y];
	triangleIndices[y] = prev;
}
