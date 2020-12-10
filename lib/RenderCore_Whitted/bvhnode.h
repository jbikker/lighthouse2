#pragma once

#include "core_settings.h"
#include "ray.h"
#include "vector"
#include "triangle.h"

class BVHNode
{
public:
	aabb bounds;
	bool isLeaf;
	BVHNode *left, *right;
	vector<int> triangleIndices;

	void SubdivideNode(BVHNode* pool, int& poolPtr);
	void PartitionTriangles();
	void UpdateBounds();
	bool Traverse(Ray &ray, float4& color, uint& recursionDepth);
	bool IntersectTriangles(Ray &ray, float4& color, uint& recursionDepth);
};

