#ifndef PLATE_H
#define PLATE_H
#include <opencv2/opencv.hpp>
#include <io.h>
using namespace std;
//using namespace cv;
namespace cpr{
class Plate {
private:
	//包含车牌的彩色图片
	cv::Mat m_img;
	//车牌的候选区域
	vector<cv::Mat> m_plateCandidate;
	//真正的车牌彩色图片
	cv::Mat m_plate;
	//特征
	cv::Mat m_features;



public:
	void setImg(cv::Mat param) { m_img = param; }
	cv::Mat getImg() { return m_img; }

	void setPlateCandidate(vector<cv::Mat> param) { m_plateCandidate = param; }
	vector<cv::Mat> getPlateCandidate() { return m_plateCandidate; }

	void setPlate(cv::Mat param) { m_plate = param; }
	cv::Mat getPlate() { return m_plate; }

	void setFeatures(cv::Mat param) { m_features = param; }
	cv::Mat getFeatures() { return m_features; }

	void getCandidatePlate(cv::Mat& img_rgb, vector<cv::Mat>& plate_Mat);
	void getLBPFeatures(const cv::Mat& image, cv::Mat& features);

};
}
#endif