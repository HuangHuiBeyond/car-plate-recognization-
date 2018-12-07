#include <opencv2/opencv.hpp>
#include "chars_segment.h"
#include "chars.h"
namespace cpr {
	/***************************************************************************
	input:RGB plate image Mat
	output:plate chars vector(resize)
	***************************************************************************/
	int CCharsSegment::charsSegment(cv::Mat plate, vector<cv::Mat>& plateCharsResize) {
		cv::Mat plateCopy = plate.clone();
		cv::Mat plateCopyAllBox = plate.clone();
		cv::Mat plateCopySelectBox = plate.clone();


		//Mat plateGaussianBlur;
		//GaussianBlur(plate, plateGaussianBlur, Size(3, 3), 0);
		cv::Mat plateGray;
		//灰度化
		cvtColor(plate, plateGray, CV_RGB2GRAY);
		//Mat plateCanny;
		//Sobel(plateGray, plateCanny, CV_8S, 1, 0, 3);
		//二值化
		cv::Mat plateThreshold;
		double ostuthreshVal =
			threshold(plateGray, plateThreshold, 0, 255, CV_THRESH_OTSU +
				CV_THRESH_BINARY);
		//去除噪声（铆钉等）
		clearNoise(plateThreshold);
		cv::Mat plateThresholdCopy = plateThreshold;//保存原始副本
		cv::Mat plateThreshlodCopyDebug = plateThreshold;//debug用
		imshow("plateThreshold", plateThreshold);
		//把单通道图像展成3通道图，每个通道保持一致，这样可以在二值化图显示有颜色的框
		cv::Mat plateThreshold3Channels;
		vector<cv::Mat> mv;
		mv.push_back(plateThreshold);
		mv.push_back(plateThreshold);
		mv.push_back(plateThreshold);
		merge(mv, plateThreshold3Channels);
		//GaussianBlur(imageSource, image, Size(3, 3), 0);
		//Canny(plateThreshold, plateThreshold, 100, 250);

		//画出所有外接轮廓,通过字符先验知识进行初步筛选
		vector<vector<cv::Point>> contours;
		findContours(plateThreshold,
			contours,
			CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_NONE);
		vector<vector<cv::Point>>::iterator itc = contours.begin();
		vector<cv::RotatedRect> first_rects;
		while (itc != contours.end()) {
			cv::RotatedRect mr = minAreaRect(cv::Mat(*itc));
			//float area = mr.size.height * mr.size.width;
			//float r = (float)mr.size.width / (float)mr.size.height;
			//if (r < 1) r = (float)mr.size.height / (float)mr.size.width;
			//int min = 34 * 8 * 1;
			//int max = 34 * 8 * 24 * 6;
			//float rmin = 4 - 4 * 0.9;
			//float rmax = 4 + 4 * 0.9;
			first_rects.push_back(mr);
			/*if ((area > min && area < max) && (r > rmin && r < rmax)) {
				first_rects.push_back(mr);

			}*/
			++itc;
		}
		//vector<Mat> plate_mat;
		vector<cv::Mat> chars;//存储经过筛选后的车牌图像
		vector<cv::Rect> charsRects;//存储经过筛选后的车牌区域
		for (size_t i = 0; i < first_rects.size(); i++) {
			cv::RotatedRect roi_rect = first_rects[i];
			cv::Rect roi_bounding_rect = roi_rect.boundingRect();
			//rectangle(img_close, roi_rect);
			rectangle(plateThreshold3Channels, roi_bounding_rect, cv::Scalar(0, 0, 255));
			rectangle(plateCopyAllBox, roi_bounding_rect, cv::Scalar(0, 0, 255));//显示全部框

			if (roi_bounding_rect.y < 0 || roi_bounding_rect.x < 0 || 
				plateThreshold.size().height < (roi_bounding_rect.y + roi_bounding_rect.height) ||
				plateThreshold.size().width < (roi_bounding_rect.x + roi_bounding_rect.width)) continue;
			cv::Mat charsCandidate = cv::Mat(plateThreshold, roi_bounding_rect);
			//rectangle(plateThreshold3Channels, roi_bounding_rect, Scalar(0, 0, 255));
			

			if (verifyCharSizes(charsCandidate)) {
				chars.push_back(charsCandidate);
				charsRects.push_back(roi_bounding_rect);
				rectangle(plateCopySelectBox, roi_bounding_rect, cv::Scalar(0, 0, 255));//显示筛选后的框
			}

			

			//if (roi_bounding_rect.y < 0 || roi_bounding_rect.x < 0 || plateThreshold.size().height < roi_bounding_rect.x) continue;
			//plate_mat.push_back(Mat(plateThreshold, roi_bounding_rect));
			//plate_candidate.push_back(Mat(36, 136, 16));



		}
		//for (size_t i = 0; i < plate_mat.size(); i++) {
		//	resize(plate_mat[i], plate_candidate[i], plate_candidate[i].size());
		//	//imwrite("plate_candidate" + to_string(i) + ".jpg", plate_candidate[i]);
		//	imwrite("plate_candidate" + to_string(i) + ".jpg", plate_candidate[i]);

		//}
		//获取特殊字符的位置
		int specificChar = 0;
		specificChar = getSpecificRect(charsRects);
		rectangle(plate, charsRects[specificChar], cv::Scalar(255, 0, 0));
		//根据特殊字符的位置找出中文字符
		cv::Rect chineseRect = getChineseRect(charsRects[specificChar]);
		vector<cv::Mat> plateChars;
		plateChars.push_back(cv::Mat(plateThresholdCopy, chineseRect));
		plateChars.push_back(cv::Mat(plateThresholdCopy, charsRects[specificChar]));
		rectangle(plate, chineseRect, cv::Scalar(0, 255, 0));
		//if (specificChar) {
		//	rectangle(plate, charsRects[specificChar-1], Scalar(0, 255, 0));
		//}
		
		//根据特殊字符按顺序找出之后的字符
		vector<cv::Rect> finalCharsRects;
		rebuildRect(charsRects, finalCharsRects, specificChar);
		for (size_t i = 0; i < finalCharsRects.size(); i++) {
			plateChars.push_back(cv::Mat(plateThresholdCopy,finalCharsRects[i]));

		}
		//进行放射变换和大小统一
		for (size_t i = 0; i < plateChars.size(); i++) {
			plateCharsResize.push_back(preprocessChar(plateChars[i]));
		}
		for (size_t i = 0; i < finalCharsRects.size(); i++) {
			rectangle(plateCopy, finalCharsRects[i], cv::Scalar(0, 0, 255));
		}

		for (int i = 0; i < plateCharsResize.size(); i++) {

			imwrite("chars" + to_string(i) + ".jpg", plateCharsResize[i]);
		}

		imshow("roi", plateThreshold3Channels);
		imshow("spe and chinese", plate);
		imshow("all box", plateCopyAllBox);
		imshow("selected box", plateCopySelectBox);
		imshow("5 chars", plateCopy);


		return 0;
	}


	/***************************************************************************
	input:plate chars candidate cv::Mat
	output:if  it is a plate chars
	***************************************************************************/
	bool CCharsSegment::verifyCharSizes(cv::Mat r) {
		// Char sizes 45x90
		float aspect = 45.0f / 90.0f;
		float charAspect = (float)r.cols / (float)r.rows;
		float error = 0.7f;
		float minHeight = 10.f;
		float maxHeight = 35.f;
		// We have a different aspect ratio for number 1, and it can be ~0.2
		float minAspect = 0.05f;
		float maxAspect = aspect + aspect * error;
		// area of pixels
		int area = cv::countNonZero(r);
		// bb area
		int bbArea = r.cols * r.rows;
		//% of pixel in area
		int percPixels = area / bbArea;

		if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
			r.rows >= minHeight && r.rows < maxHeight)
			return true;
		else
			return false;
	}


	/***************************************************************************
	input:plate threshold image with noise
	output:plate threshold image without noise
	***************************************************************************/
	void CCharsSegment::clearNoise(cv::Mat& thresholdImage) {
		vector<float> fJump;
		const int jumpMinimumTimes = 14;
		cv::Mat jump = cv::Mat::zeros(1, thresholdImage.rows, CV_32F);
		for (int i = 0; i < thresholdImage.rows; i++) {
			int jumpCount = 0;
			for (int j = 0; j < thresholdImage.cols - 1; j++) {
				if (thresholdImage.at<char>(i, j) != thresholdImage.at<char>(i, j + 1)) jumpCount++;
			

			}
			jump.at<float>(i) = (float)jumpCount;

		}
		for (int i = 0; i < thresholdImage.rows; i++) {
			if (jump.at<float>(i) <= jumpMinimumTimes) {
				for (int j = 0; j < thresholdImage.cols; j++) {
					thresholdImage.at<char>(i, j) = 0;
				}
			}
		}
	}


	/***************************************************************************
	input:vector of chars rect candidate
	output:specific char index(the first char after the chinese char)
	***************************************************************************/
	int CCharsSegment::getSpecificRect(const vector<cv::Rect>& vecRect) {
		vector<int> xpositions;
		int maxHeight = 0;
		int maxWidth = 0;

		for (size_t i = 0; i < vecRect.size(); i++) {
			xpositions.push_back(vecRect[i].x);

			if (vecRect[i].height > maxHeight) {
				maxHeight = vecRect[i].height;
			}
			if (vecRect[i].width > maxWidth) {
				maxWidth = vecRect[i].width;
			}
		}

		int specIndex = 0;
		for (size_t i = 0; i < vecRect.size(); i++) {
			cv::Rect mr = vecRect[i];
			int midx = mr.x + mr.width / 2;

			// use prior knowledage to find the specific character
			// position in 1/7 and 2/7
			if ((mr.width > maxWidth * 0.6 || mr.height > maxHeight * 0.6) &&
				(midx < int(136 / 7) *2 &&
					midx > int(136 / 7) * 1)) {
				specIndex = i;
			}
		}

		return specIndex;
	}


	/***************************************************************************
	input:specific char(the first char after the chinese char)
	output:chinese char rect
	***************************************************************************/
	cv::Rect CCharsSegment::getChineseRect(const cv::Rect rectSpe) {
		int height = rectSpe.height;
		float newwidth = rectSpe.width * 1.15f;
		int x = rectSpe.x;
		int y = rectSpe.y;

		int newx = x - int(newwidth * 1.15);
		newx = newx > 0 ? newx : 0;

		cv::Rect a(newx, y, int(newwidth), height);

		return a;
	}


	/***************************************************************************
	input:vector of chars candidate, specific char index
	output:chars after the specific char(included)
	***************************************************************************/
	//int CCharsSegment::rebuildRect(const vector<Rect>& vecRect,
	//	vector<Rect>& outRect, int specIndex) {
	//	int count = 6;
	//	if (specIndex < 4) {
	//		for (size_t i = specIndex; i < vecRect.size() && count; i++, --count) {
	//			outRect.push_back(vecRect[i]);
	//		}
	//	}
	//	else {
	//		for (size_t i = specIndex; i < vecRect.size() && count; i--, --count) {
	//					outRect.push_back(vecRect[i]);
	//		}
	//	}
	//	

	//	return 0;
	//}
	bool CCharsSegment::comp(const cv::Rect a, const cv::Rect b) {
		return a.x < b.x;
	}


	int CCharsSegment::rebuildRect(const vector<cv::Rect>& vecRect,
		vector<cv::Rect>& outRect, int specIndex) {
		vector<cv::Rect> sortedVecRect;
		for (auto it = vecRect.begin(); it != vecRect.end(); it++) {
			sortedVecRect.push_back(*it);
		}
		vector<int> rectX;
		int specificX = vecRect[specIndex].x;
		for (auto it = sortedVecRect.begin(); it != sortedVecRect.end(); it++) {
			if ((*it).x > specificX ) {
				rectX.push_back((*it).x);
			}
		}
		sort(rectX.begin(), rectX.end());
		for (auto it = rectX.begin(); it != rectX.end(); it++) {
			for (auto it2 = sortedVecRect.begin(); it2 != sortedVecRect.end(); it2++) {
				if ((*it) == (*it2).x && outRect.size() < 5){
					outRect.push_back(*it2);
				}
			}
		}
		return 0;
	}


	/***************************************************************************
	input:char cv::Mat
	output:resize char cv::Mat
	***************************************************************************/
	cv::Mat CCharsSegment::preprocessChar(cv::Mat in) {
		// Remap image
		int h = in.rows;
		int w = in.cols;

		int charSize = 20;

		cv::Mat transformMat = cv::Mat::eye(2, 3, CV_32F);
		int m = max(w, h);
		transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
		transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

		cv::Mat warpImage(m, m, in.type());
		warpAffine(in, warpImage, transformMat, warpImage.size(), cv::INTER_LINEAR,
			cv::BORDER_CONSTANT, cv::Scalar(0));

		cv::Mat out;
		resize(warpImage, out, cv::Size(charSize, charSize));

		return out;
	}
}