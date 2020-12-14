#include "bvhnode.h"
#include "whitted_ray_tracer.h"
#include "triangle.h"
#include "ray.h"
#include "tuple"

void BVHNode::SubdivideNode(BVHNode* pool, int* triangleIndices, int &poolPtr) {
	if (this->count < 5) return;
	
	//cout << "Amount BVHNode " << poolPtr << "\n";
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
	float splitPoint = this->bounds.Center(axis);
	
	int low = this->first;
	int high = this->first + this->count - 1;

	int j = low;
	for (int i = low; i < high; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[i]];

		float trianglePoint;
		if (axis == 0) {
			trianglePoint = triangle->centroid.x;
		} else if (axis == 1) {
			trianglePoint = triangle->centroid.z;
		} else {
			trianglePoint = triangle->centroid.y;
		}

		if (trianglePoint < splitPoint) {
			this->Swap(triangleIndices, i, j);
			j++;
		}
	}
	this->Swap(triangleIndices, j, high);

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];

	left->first = this->first;
	left->count = j - low;
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
	if (point.x < this->bounds.bmin[0]) {
		this->bounds.bmin[0] = point.x;
	}
	if (point.x > this->bounds.bmax[0]) {
		this->bounds.bmax[0] = point.x;
	}
	if (point.y < this->bounds.bmin[1]) {
		this->bounds.bmin[1] = point.y;
	}
	if (point.y > this->bounds.bmax[1]) {
		this->bounds.bmax[1] = point.y;
	}
	if (point.z < this->bounds.bmin[2]) {
		this->bounds.bmin[2] = point.z;
	}
	if (point.z > this->bounds.bmax[2]) {
		this->bounds.bmax[2] = point.z;
	}
}

void BVHNode::Traverse(Ray &ray, BVHNode* pool, int* triangleIndices, tuple<Triangle*, float> &bestIntersect) {
	float intersectDist;
	if (!ray.IntersectionBounds(this->bounds, intersectDist)) { return; }

	if (this->isLeaf) { 
		tuple<Triangle*, float> curIntersect = make_tuple<Triangle*, float>(NULL, NULL);
		this->IntersectTriangles(ray, triangleIndices, curIntersect);
		
		Triangle* curIntersectTriangle = get<0>(curIntersect);
		float curIntersectDist = get<1>(curIntersect);
		
		Triangle* bestIntersectTriangle = get<0>(bestIntersect);
		float bestIntersectDist = get<1>(bestIntersect);

		if (
			curIntersectTriangle != NULL && (
				bestIntersectTriangle == NULL ||
				(curIntersectDist > EPSILON && curIntersectDist < bestIntersectDist)
			)
		) {
			bestIntersect = curIntersect;
		}
		return; 
	}
	
	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];

	left->Traverse(ray, pool, triangleIndices, bestIntersect);
	right->Traverse(ray, pool, triangleIndices, bestIntersect);
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
