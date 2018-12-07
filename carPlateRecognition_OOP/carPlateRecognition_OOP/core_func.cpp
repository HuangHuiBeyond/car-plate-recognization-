#include "core_func.h"
namespace cpr {
	cv::Mat ProjectedHistogram(cv::Mat img, int t, int threshold) {
		int sz = (t) ? img.rows : img.cols;
		cv::Mat mhist = cv::Mat::zeros(1, sz, CV_32F);

		for (int j = 0; j < sz; j++) {
			cv::Mat data = (t) ? img.row(j) : img.col(j);

			mhist.at<float>(j) = countOfBigValue(data, threshold);
		}

		// Normalize histogram
		double min, max;
		minMaxLoc(mhist, &min, &max);

		if (max > 0)
			mhist.convertTo(mhist, -1, 1.0f / max, 0);

		return mhist;
	}


	float countOfBigValue(cv::Mat &mat, int iValue) {
		float iCount = 0.0;
		if (mat.rows > 1) {
			for (int i = 0; i < mat.rows; ++i) {
				if (mat.data[i * mat.step[0]] > iValue) {
					iCount += 1.0;
				}
			}
			return iCount;

		}
		else {
			for (int i = 0; i < mat.cols; ++i) {
				if (mat.data[i] > iValue) {
					iCount += 1.0;
				}
			}

			return iCount;
		}
	}
}