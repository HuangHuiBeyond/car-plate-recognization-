#ifndef CORE_FUNC_H
#define CORE_FUNC_H
#include <opencv2\opencv.hpp>
//using namespace cv;
using namespace std;
namespace cpr {
	cv:: Mat ProjectedHistogram(cv::Mat img, int t, int threshold = 20);
	float countOfBigValue(cv::Mat &mat, int iValue);


}
#endif