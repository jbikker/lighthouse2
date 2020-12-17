#include "bvhnode.h"
#include "bvh.h"
#include "bin.h"
#include "kajiya_path_tracer.h"
#include "triangle.h"
#include "ray.h"
#include "tuple"

void BVHNode::SubdivideNode(BVHNode* pool, int* triangleIndices, int &poolPtr) {
	if (this->count <= 2) { return; }
	
	this->left = poolPtr;
	poolPtr += 2;
	
	if (!this->PartitionTriangles(pool, triangleIndices)) { return; }

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];
	
	left->SubdivideNode(pool, triangleIndices, poolPtr);
	right->SubdivideNode(pool, triangleIndices, poolPtr);
	
	this->isLeaf = false;
}

float GetTriangleAxisValue(int axis, Triangle* triangle) {
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
	/** Generate bounding box over triangle centroids */
	aabb centroidBoundingBox = aabb();
	for (int i = this->first; i < this->first + this->count; i++) {
		Triangle* triangle = KajiyaPathTracer::scene[triangleIndices[i]];
		centroidBoundingBox.Grow(triangle->centroid);
	}

	int axis = centroidBoundingBox.LongestAxis();
	this->splitAxis = axis;

	/** Reset bins */
	for (int i = 0; i < BVH::binCount; i++) {
		BVH::bins[i].Clear();
		BVH::binsLeft[i].Clear();
		BVH::binsRight[i].Clear();
	}

	float cbmin = centroidBoundingBox.bmin[axis];
	float cbmax = centroidBoundingBox.bmax[axis];
	float k1 = (BVH::binCount * (1 - EPSILON)) / (cbmax - cbmin);

	/** Fill the bins with Triangles */
	for (int i = this->first; i < this->first + this->count; i++) {
		Triangle* triangle = KajiyaPathTracer::scene[triangleIndices[i]];
		float ci = GetTriangleAxisValue(axis, triangle);

		int binID = (int)(k1 * (ci - cbmin));
		
		Bin* bin = &BVH::bins[binID];

		bin->count++;
		bin->bounds.Grow(triangle->bounds);
	}

	/** Do a linear pass through the bins from the left */
	BVH::binsLeft[0] = BVH::bins[0];
	for (int i = 1; i < BVH::binCount; i++) {
		Bin* bin = &BVH::bins[i];
		Bin* binLeft = &BVH::binsLeft[i];
		Bin* prevBinLeft = &BVH::binsLeft[i - 1];

		binLeft->count = prevBinLeft->count + bin->count;
		binLeft->bounds.Grow(prevBinLeft->bounds);
		binLeft->bounds.Grow(bin->bounds);
	}

	/** Do a linear pass through the bins from the right */
	BVH::binsRight[BVH::binCount - 1] = BVH::bins[BVH::binCount - 1];
	for (int i = BVH::binCount - 2; i >= 0; i--) {
		Bin* bin = &BVH::bins[i];
		Bin* binRight = &BVH::binsRight[i];
		Bin* prevBinRight = &BVH::binsRight[i + 1];

		binRight->count = prevBinRight->count + bin->count;
		binRight->bounds.Grow(prevBinRight->bounds);
		binRight->bounds.Grow(bin->bounds);
	}

	int binIndex = -1;
	float bestCost = std::numeric_limits<float>::max();
	float curCost = this->bounds.Area() * this->count;

	for (int i = 0; i < BVH::binCount; i++) {
		Bin* binLeft = &BVH::binsLeft[i];
		Bin* binRight = &BVH::binsRight[i];

		float cost = binLeft->bounds.Area() * binLeft->count + binRight->bounds.Area() * binRight->count;

		if (
			(cost < curCost) && // SAH Termination
			(binLeft->count > 0 && binRight->count > 0) &&
			(binIndex == -1 || cost < bestCost)
		) {
			binIndex = i;
			bestCost = cost;
		}
	}

	if (binIndex == -1) { return false; }

	int j = this->first;
	for (int i = this->first; i < this->first + this->count; i++) {
		Triangle* triangle = KajiyaPathTracer::scene[triangleIndices[i]];
		float ci = GetTriangleAxisValue(axis, triangle);
		int binID = (int)(k1 * (ci - cbmin));

		if (binID < binIndex) {
			this->Swap(triangleIndices, i, j);
			j++;
		}
	}

	Bin* binLeft = &BVH::binsLeft[binIndex];
	Bin* binRight = &BVH::binsRight[binIndex];

	BVHNode* left = &pool[this->left];
	BVHNode* right = &pool[this->left + 1];

	left->first = this->first;
	left->count = j - this->first;
	right->first = j;
	right->count = this->count - left->count;

	left->bounds = binLeft->bounds;
	right->bounds = binRight->bounds;

	return true;
}

void BVHNode::UpdateBounds(int* triangleIndices) {
	this->bounds.Reset();

	for (int i = 0; i < this->count; i++) {
		Triangle* triangle = KajiyaPathTracer::scene[triangleIndices[this->first + i]];
		this->bounds.Grow(triangle->bounds);
	}
}

void BVHNode::Traverse(Ray &ray, BVHNode* pool, int* triangleIndices, tuple<Triangle*, float, Ray::HitType> &intersection) {
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

void BVHNode::IntersectTriangles(Ray &ray, int* triangleIndices,  tuple<Triangle*, float, Ray::HitType> &intersection) {
	Triangle* nearestPrimitive = get<0>(intersection);
	float minDistance = get<1>(intersection);
	Ray::HitType hitType = get<2>(intersection);

	for (int i = 0; i < this->count; i++) {
		Triangle* triangle = KajiyaPathTracer::scene[triangleIndices[this->first + i]];
		float distance = triangle->Intersect(ray);

		if (
			((minDistance == NULL) || (distance < minDistance))
			&& (distance > EPSILON)
		) {
			minDistance = distance;
			nearestPrimitive = triangle;
			hitType = Ray::HitType::SceneObject;
		}
	}

	intersection = make_tuple(nearestPrimitive, minDistance, hitType);
}

void BVHNode::Swap(int* triangleIndices, int x, int y) {
	int prev = triangleIndices[x];
	triangleIndices[x] = triangleIndices[y];
	triangleIndices[y] = prev;
}
