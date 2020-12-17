#pragma once

#include "core_settings.h"
#include "vector"
#include "tuple"
#include "ray.h"

class Ray;
class Triangle;

class BVHNode
{
public:

	aabb bounds;
	bool isLeaf = true;
	int left;
	int first;
	int count;
	int splitAxis;

	void SubdivideNode(BVHNode* pool, int* triangleIndices, int& poolPtr);
	bool PartitionTriangles(BVHNode* pool, int* triangleIndices);
	void UpdateBounds(int* triangleIndices);
	void Traverse(Ray& ray, BVHNode* pool, int* triangleIndices, tuple<Triangle*, float, Ray::HitType>& intersection);
	void IntersectTriangles(Ray& ray, int* triangleIndices, tuple<Triangle*, float, Ray::HitType>& intersection);
	void Swap(int* triangleIndices, int i, int j);
};

