#include "pch.h"
#include "SeamCarving.h"
#include "EdgeDetector.h"
#include <random>
#include <chrono>
#include <thread>
#include <omp.h>

SeamCarving::SeamCarving()
{
}


SeamCarving::~SeamCarving()
{
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

cv::Mat SeamCarving::get_lowest_energy_Map(cv::Mat energy_map)
{
	double left, current, right;
	cv::Mat minimal_mat = cv::Mat(energy_map.size(), energy_map.type());
	energy_map.copyTo(minimal_mat);
	//cout << "Minimal map: " << endl << minimal_mat << endl;
	for (int y = 0; y < energy_map.rows; y++) {
#pragma omp parallel for private(left, current, right)
		for (int x = 0; x < energy_map.cols; x++) {
			int prevY = std::max<double>(y - 1, 0);
			left = minimal_mat.at<double>(prevY, std::max<int>(x - 1, 0));
			current = minimal_mat.at<double>(prevY, x);
			right = minimal_mat.at<double>(prevY, std::min<int>(x + 1, energy_map.cols-1));
			minimal_mat.at<double>(y, x) = energy_map.at<double>(y, x) + std::min<int>(left, min(current, right));
		}
	}

	return minimal_mat;
}

int SeamCarving::get_random_lowest_energy_index(cv::Mat minimal_energy_map) {

	int index = 0;
	std::mt19937 generator(123);
	std::vector<int> *indexes = new std::vector<int>(minimal_energy_map.cols);
	double minimum = -1;
	for (int x = 0; x < minimal_energy_map.cols; x++) {
		double current_value = minimal_energy_map.at<double>(minimal_energy_map.rows - 1, x);
		if (current_value < minimum || minimum == -1) {
			indexes->clear();
			indexes->push_back(x);
			minimum = current_value;
		} else {
			if (current_value == minimum) {
				indexes->push_back(x);
			}
		}
	}
	
	index = (int)indexes->front();
	int size = indexes->size();
	std::uniform_real_distribution<double> dis(0, size);
	int random = (int)dis(generator);
	index = indexes->at(random);
	
	return index;
}

cv::Mat SeamCarving::get_lowest_energy_with_seam(cv::Mat lowest_energy_map, int lowest_energy_index, cv::Mat colorTest) {
	double left, middle, right;
	int x = lowest_energy_index;
	for (int y = lowest_energy_map.rows - 1; y > 0; y--) {
		lowest_energy_map.at<double>(y, lowest_energy_index) = -1;
		if (this->debug) {
			colorTest.at<Vec3b>(y, lowest_energy_index) = Vec3b(0, 0, 255);
		}
		
		x = lowest_energy_index;
		
		int prevY = std::max<int>(y - 1, 0);
		left = lowest_energy_map.at<double>(prevY, std::max<int>(x - 1, 0));
		middle = lowest_energy_map.at<double>(prevY, x);
		right = lowest_energy_map.at<double>(prevY, std::min<int>(x + 1, lowest_energy_map.cols - 1));

		if (left < right) {
			if (left < middle) {
				lowest_energy_index = std::max<int>(x - 1, 0);
			}
			else {
				lowest_energy_index = x;
			}
		}
		else {
			if (middle < right) {
				lowest_energy_index = x;
			}
			else {
				lowest_energy_index = std::min<int>(x + 1, lowest_energy_map.cols - 1);
			}
		}

	}
	if (this->debug) {
		namedWindow("ColorDebug");
		imshow("ColorDebug", colorTest);
		waitKey(1);
	}

	return lowest_energy_map;
}

cv::Mat SeamCarving::remove_seam(cv::Mat& result, cv::Mat& lowest_energy_map) {
	cv::Mat newResult = cv::Mat(result.size().height, result.size().width - 1, result.type());
#pragma omp parallel for
	for (int y = 0; y < newResult.rows; y++) {
		int offset = 0;
		for (int x = 0; x < newResult.cols; x++) {
			double value = lowest_energy_map.at<double>(y, x + offset);
			if (value == -1) {
				offset++;
				if ((x + offset) >= result.cols) {
					break;
				}
			}
			//cv::Vec3b::cross()
			newResult.at<cv::Vec3b>(y, x) = result.at<cv::Vec3b>(y, x + offset);
		}
	}

	return newResult;
}

cv::Mat SeamCarving::get_cropped_image(cv::Mat source, bool debug) {
	this->debug = debug;
	cv::Mat energy_map, lowest_energy_map, colorTest;
	EdgeDetector *ed = new EdgeDetector();
	std::vector<int> *indexes = new std::vector<int>();
	uint removed_cols = 0;
	cout << "type" << type2str(source.type()) << endl;
	cv::Mat result = cv::Mat(source.size(), source.type());

	source.copyTo(result);
	while (removed_cols < 350)
	{
		removed_cols++;
		std::cout << removed_cols << std::endl;
		energy_map = ed->get_sober_operator(result, debug);
		lowest_energy_map = this->get_lowest_energy_Map(energy_map);
		int lowest_energy_index = this->get_random_lowest_energy_index(lowest_energy_map);
		if (this->debug) {
			result.copyTo(colorTest);
		}
		lowest_energy_map = this->get_lowest_energy_with_seam(lowest_energy_map, lowest_energy_index, colorTest);

		result = this->remove_seam(result, lowest_energy_map);
	}
	return result;
}

