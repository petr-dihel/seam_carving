#pragma once

using namespace cv;
using namespace std;

template <typename T>
class Filter
{
private:
	int mX;
	int mY;
	T** mat;
	double weigth;
public:
	Filter();
	~Filter();
	Filter(int x, int y, double wei, T** p) {
		this->mX = x;
		this->mY = y;
		this->weigth = wei;
		this->mat = p;
		/*for (int i = 0; i < mY; ++i) {
			this->mat[i] = new int[mX];
		}*/

		for (int y = 0; y < mY; ++y) {
			for (int x = 0; x < mY; ++x) {
				this->mat[y][x] = p[y][x];
			}
		}
	}

	T getPixel(cv::Mat image, int currentY, int currentX) {
		int minBorder = this->mX / 2;
		T pixel = image.at<T>(currentY, currentX);
		if (
			currentX < minBorder
			|| currentY < minBorder
			|| currentX >(image.cols - 1 - minBorder)
			|| currentY >(image.rows - 1 - minBorder)
			) {
			return pixel;
		}

		T sum = 0;

		for (int y = currentY - minBorder; y <= currentY + minBorder; y++) {
			for (int x = currentX - minBorder; x <= currentX + minBorder; x++) {
				int a = (currentY + minBorder) - y;
				int b = (currentX + minBorder) - x;
				T tmp = this->mat[a][b];
				T tmpPixel = image.at<T>(y, x);
				T tmpVec = tmp;
				sum += tmpPixel + tmpVec;
			}
		}
		sum *= this->weigth;
		return sum;
	}


	cv::Mat getFilteredImage(cv::Mat image) {
		cv::Mat result;
		image.copyTo(result);
	

#pragma omp parallel for
		for (int y = 0; y < image.rows; y++) {
			for (int x = 0; x < image.cols; x++) {
				T newPixel = this->getPixel(result, y, x);
				result.at<T>(y, x) = newPixel;
			}
		}
		return result;
	}

};

