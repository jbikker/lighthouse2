#include "bvhnode.h"
#include "bvh.h"
#include "bin.h"
#include "whitted_ray_tracer.h"
#include "triangle.h"
#include "ray.h"
#include "tuple"

void BVHNode::SubdivideNode(BVHNode* pool, int* triangleIndices, int &poolPtr) {
	if (this->count <= 2) return;
	
	this->left = poolPtr;
	poolPtr += 2;
	
	bool shouldSubdivideNodeFurther = this->PartitionTriangles(pool, triangleIndices);
	if (!shouldSubdivideNodeFurther) { return; }

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];
	
	if (left->count == 0 || right->count == 0) { return; }
	
	left->SubdivideNode(pool, triangleIndices, poolPtr);
	right->SubdivideNode(pool, triangleIndices, poolPtr);
	
	this->isLeaf = false;
}

void UpdateBoundingBoxWithCentroid(aabb& aabb, Triangle* triangle) {
	aabb.Grow(make_float3(triangle->centroid));
}

void UpdateBoundingBoxWithTriangle(aabb& aabb, Triangle* triangle) {
	aabb.Grow(make_float3(triangle->v0));
	aabb.Grow(make_float3(triangle->v1));
	aabb.Grow(make_float3(triangle->v2));
}

float GetTrianglePoint(int axis, Triangle* triangle) {
	if (axis == 0) {
		return triangle->centroid.x;
	}
	else if (axis == 1) {
		return triangle->centroid.y;
	}
	else {
		return triangle->centroid.z;
	}
}


bool BVHNode::PartitionTriangles(BVHNode* pool, int* triangleIndices) {
	int binCount = 6;

	/** Generate bounding box over triangle centroids */
	aabb centroidBoundingBox = aabb();
	for (int i = this->first; i < this->first + this->count; i++) {
		Triangle* triangle = WhittedRayTracer::scene[triangleIndices[i]];
		UpdateBoundingBoxWithCentroid(centroidBoundingBox, triangle);
	}

	int axis = centroidBoundingBox.LongestAxis();
	this->splitAxis = axis;

	/** Split into bins */
	float bmin = centroidBoundingBox.bmin[axis];
	float bmax = centroidBoundingBox.bmax[axis];
	float totalWidth = abs(bmax - bmin);
	float widthPerBin = totalWidth / binCount;

	/** Dont compute the most left and most right bin */
	Bin* lowestCostBin = &BVH::bins[0];

	/** Create bins */
	for (int i = 1; i < binCount; i++) {
		float splitPoint = bmin + widthPerBin * i;
		Bin* bin = &BVH::bins[i - 1];
		bin->Clear();
		bin->splitPoint = splitPoint;
		for (int i = this->first; i < this->first + this->count; i++) {
			Triangle* triangle = WhittedRayTracer::scene[triangleIndices[i]];
			float trianglePoint = GetTrianglePoint(axis, triangle);
			if (trianglePoint < splitPoint) {
				bin->countLeft++;
				UpdateBoundingBoxWithTriangle(bin->boundsLeft, triangle);
			} else {
				bin->countRight++;
				UpdateBoundingBoxWithTriangle(bin->boundsRight, triangle);
			}
		}

		/** Calculate cost */
		bin->UpdateSurfaceAreaCost();
		if (bin->cost < lowestCostBin->cost) {
			lowestCostBin = bin;
		}
	}

	/** Check SAH */
	if (lowestCostBin->cost >= this->bounds.Area() * this->count) {
		return false;
	}


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

		if (trianglePoint < lowestCostBin->splitPoint) {
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

	return true;
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
	Triangle* nearestPrimitive = get<0>(intersection);
	float minDistance = get<1>(intersection);

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
