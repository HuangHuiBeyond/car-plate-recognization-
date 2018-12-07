#ifndef CHARS_H
#define CHARS_H
#include <opencv2\opencv.hpp>
//using namespace cv;
using namespace std;
namespace cpr {
	class CChars {
	private:
		//获取到的7张字符
		vector<cv::Mat> m_charsMat;
		vector<cv::Mat> m_charsFeatures;
	public:
		vector<cv::Mat> getCharsMat() { return m_charsMat; }
		void setCharsMat(vector<cv::Mat> param) {
			for (auto it = param.begin(); it != param.end(); it++) {
				m_charsMat.push_back(*it);
			}
		}
		vector<cv::Mat> getFeaturesMat() { return m_charsFeatures; }
		void setCharsFeatures(vector<cv::Mat> param) {
			for (auto it = param.begin(); it != param.end(); it++) {
				m_charsFeatures.push_back(*it);
			}
		}

		cv::Mat getCharFeatures(cv::Mat in, int sizeData);
		void charsClassify();
	};
	

}
#endif
