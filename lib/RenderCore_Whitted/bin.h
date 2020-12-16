#pragma once
#include "core_settings.h"

class Bin {
public:
	int count = 0;
	aabb bounds = aabb();
	void Clear();
};

