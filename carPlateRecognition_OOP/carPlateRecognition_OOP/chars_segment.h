#ifndef CHARS_SEGMENT_H
#define CHARS_SEGMENT_H
#include <opencv2\opencv.hpp>
//using namespace cv;
using namespace std;
namespace cpr {
	class CCharsSegment {
	private:
	public:
		int charsSegment(cv::Mat plate, vector<cv::Mat>& plateChars);
		bool verifyCharSizes(cv::Mat r);
		void clearNoise(cv::Mat& thresholdImage);
		int getSpecificRect(const vector<cv::Rect>& vecRect);
		cv::Rect getChineseRect(const cv::Rect rectSpe);
		int rebuildRect(const vector<cv::Rect>& vecRect,
			vector<cv::Rect>& outRect, int specIndex);
		cv::Mat preprocessChar(cv::Mat in);
		bool comp(const cv::Rect a, const cv::Rect b);
	}; 
}

#endif
