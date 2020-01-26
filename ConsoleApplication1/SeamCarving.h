#pragma once

using namespace std;
using namespace cv;

class SeamCarving
{
public:
	bool debug;
	SeamCarving();
	~SeamCarving();
	cv::Mat get_lowest_energy_Map(cv::Mat energy_map);
	int get_random_lowest_energy_index(cv::Mat minimal_energy_map);
	cv::Mat get_lowest_energy_with_seam(cv::Mat lowest_energy_map, int lowest_energy_index, cv::Mat colorTest);
	cv::Mat remove_seam(cv::Mat & result, cv::Mat & lowest_energy_map);
	cv::Mat get_cropped_image(cv::Mat source, bool debug = false);
};

