#pragma once
#include "Filter.h"

using namespace cv;
using namespace std;

class EdgeDetector
{

private:
	Filter<Vec3i>* filter;
	Vec3i **gaussian = new Vec3i*[3];
public:
	
	EdgeDetector() {
		for (int i = 0; i < 3; i++) {
			gaussian[i] = new Vec3i[3];
		}
		gaussian[0][0] = Vec3i(1,1,1);
		gaussian[0][1] = Vec3i(2, 2, 2);
		gaussian[0][2] = Vec3i(1, 1, 1);

		gaussian[1][0] = Vec3i(2, 2, 2);
		gaussian[1][1] = Vec3i(4, 4, 4);
		gaussian[1][2] = Vec3i(2, 2, 2);

		gaussian[2][1] = Vec3i(1, 1, 1);
		gaussian[2][2] = Vec3i(2, 2, 2);
		gaussian[2][1] = Vec3i(1, 1, 1);
		this->filter = new Filter<Vec3i>(3, 3, 1.0 / 9, gaussian);
	};

	~EdgeDetector();
	double getXGradient(cv::Mat src, int x, int y);
	double getYGradient(cv::Mat src, int x, int y);
	cv::Mat get_sober_operator(cv::Mat source, bool debug = false);
};

