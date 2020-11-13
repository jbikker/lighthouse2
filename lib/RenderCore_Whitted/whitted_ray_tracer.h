#pragma once
class WhittedRayTracer
{
public:
	void Render(const ViewPyramid& view);

private:
	float4 Trace(Ray& ray);

};

