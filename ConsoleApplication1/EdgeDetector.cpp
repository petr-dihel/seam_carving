#include "EdgeDetector.h"
#include "pch.h"

using namespace cv;
using namespace std;

EdgeDetector::~EdgeDetector()
{
}

double EdgeDetector::getXGradient(cv::Mat src, int x, int y) {
	bool sharr = true;
	double gx;
	int _x_1 = ((x - 1) < 0 ? (src.cols - 1) : (x - 1));
	int _y_1 = ((y - 1) < 0 ? (src.rows - 1) : (y - 1));
	int y_1 = ((y + 1) >= src.rows ? 0 : (y + 1));
	int x_1 = ((x + 1) >= src.cols ? 0 : (x + 1));


	if (sharr) {
		gx =
			3.0 * src.at<double>(_y_1, _x_1) +
			10.0 * src.at<double>(y, _x_1) +
			3.0 * src.at<double>(y_1, _x_1) -
			3.0 *src.at<double>(_y_1, x_1) -
			10.0 * src.at<double>(y, x_1) -
			3.0 * src.at<double>(y_1, x_1);
	}
	else {
		gx =
			src.at<double>(_y_1, _x_1) +
			2.0 * src.at<double>(y, _x_1) +
			src.at<double>(y_1, _x_1) -
			src.at<double>(_y_1, x_1) -
			2.0 * src.at<double>(y, x_1) -
			src.at<double>(y_1, x_1);
	}

	return gx;
}

double EdgeDetector::getYGradient(cv::Mat src, int x, int y) {
	bool sharr = true;
	int _x_1 = ((x - 1) < 0 ? (src.cols - 1) : (x - 1));
	int _y_1 = ((y - 1) < 0 ? (src.rows - 1) : (y - 1));
	int y_1 = ((y + 1) >= src.rows ? 0 : (y + 1));
	int x_1 = ((x + 1) >= src.cols ? 0 : (x + 1));
	double gY;

	if (sharr) {
		gY =
			3.0 * src.at<double>(_y_1, _x_1) +
			10.0 * src.at<double>(_y_1, x) +
			3.0 * src.at<double>(_y_1, x_1) -
			3.0 * src.at<double>(y_1, _x_1) -
			10.0 * src.at<double>(y_1, x) -
			3.0 * src.at<double>(y_1, x_1);
	}
	else {
		gY =
			src.at<double>(_y_1, _x_1) +
			2.0 * src.at<double>(_y_1, x) +
			src.at<double>(_y_1, x_1) -
			src.at<double>(y_1, _x_1) -
			2.0 * src.at<double>(y_1, x) -
			src.at<double>(y_1, x_1);
	}

	return gY;
}

string type3str(int type) {
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

cv::Mat EdgeDetector::get_sober_operator(cv::Mat source, bool debug)
{
	cv::Mat image_gray, image_blur, uchar_im;
	
	cv::Mat source_i3;
	source.convertTo(source_i3, CV_32SC3);
	//cout << "xxx: " << type3str(source_i3.type()) << endl;
	//lsource_i3 = this->filter->getFilteredImage(source_i3);
	//source_i3.convertTo(image_blur, source.type());
	//imshow("Kuk", source);
	//waitKey(1);
	cv::GaussianBlur(source, image_blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	cvtColor(image_blur, image_gray, cv::COLOR_BGR2GRAY);
	cv::Mat res_f = cv::Mat::zeros(image_gray.size(), CV_64FC1);
	cv::Mat image_gray_f;
	image_gray.convertTo(image_gray_f, CV_64FC1, 1.0/255.0);

#pragma omp parallel for
	for (int j = 0; j < image_gray.rows; j++) {
		for (int i = 0; i < image_gray.cols; i++) {
			double gX = this->getXGradient(image_gray_f, i, j) * 0.3;
			double gY = this->getYGradient(image_gray_f, i, j)* 0.3;
			double res = sqrtl(gX* gX + gY* gY);
			res_f.at<double>(j, i) = res;
		}
	}
	//cv::normalize(res_f, res_f, 0, 1, cv::NORM_MINMAX, CV_64FC1);
	if (debug) {
		cv::namedWindow("Energy Image2");
		cv::imshow("Energy Image2", res_f);
		cv::waitKey(1);
	}

	return res_f;
}

