#pragma once
#include "core_settings.h"

class Bin {
public:
	float cost = 0;
	float splitPoint = 0;
	int countLeft = 0;
	int countRight = 0;
	aabb boundsLeft;
	aabb boundsRight;
	void UpdateSurfaceAreaCost();
	void Clear();
};

