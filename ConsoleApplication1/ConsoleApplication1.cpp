// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <windows.h>
#include "EdgeDetector.h"
#include "SeamCarving.h"

using namespace cv;
using namespace std;


void saveImg(cv::Mat src, const char* name) {
	cv::Mat m2;
	//src.convertTo(m2, CV_8UC1);
	string fileName = name;
	cv::imwrite(fileName, src);
}

int main(int argc, char* argv[])
{
	std::string argv_str(argv[0]);
	std::string base = argv_str.substr(0, argv_str.find_last_of("/"));
	
	string images_path = base + "\\..\\..\\..\\images\\";

	Mat src_image = imread(images_path + "Broadway_tower_edit.jpg", IMREAD_COLOR);
	//src_image = imread(images_path + "milka.jpg", IMREAD_COLOR);
	src_image = imread(images_path + "milka.jpg", IMREAD_COLOR);
	src_image = imread(images_path + "water.jpg", IMREAD_COLOR);
	src_image = imread(images_path + "cp002.jpg", IMREAD_COLOR);
	//saveImg(src_image, (images_path + "result.jpg").c_str());
	if (!src_image.empty()) {
		cout << "LOADED" << endl;
		cout << src_image.type() << endl;
		cv::resize(src_image, src_image, cv::Size(), 0.2, 0.2);
	
		SeamCarving *seamCarving = new SeamCarving();
		cv::Mat result = seamCarving->get_cropped_image(src_image, true);
		imshow("test", src_image);		
		imshow("test4", result);
		saveImg(result, (images_path + "result3.jpg").c_str());
	}
	waitKey(0);
}