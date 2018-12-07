#include "plate.h"

namespace cpr{
/***************************************************************************
input:RGB plate image cv::Mat
output:LBPFeatures
***************************************************************************/
void Plate::getLBPFeatures(const cv::Mat& image, cv::Mat& features) {
	cv::Mat gray_image;
	cv::Mat LBPimage;
	cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	LBPimage.create(gray_image.rows - 2, gray_image.cols - 2, CV_8UC1);
	LBPimage.setTo(0);
	for (int i = 1; i<gray_image.rows - 1; i++) {
		for (int j = 1; j<gray_image.cols - 1; j++) {

			int center1 = gray_image.at<uchar>(i, j);
			unsigned char code = 0;
			code |= (gray_image.at<uchar>(i - 1, j - 1) >= center1) << 7;
			code |= (gray_image.at<uchar>(i - 1, j) >= center1) << 6;
			code |= (gray_image.at<uchar>(i - 1, j + 1) >= center1) << 5;
			code |= (gray_image.at<uchar>(i, j + 1) >= center1) << 4;
			code |= (gray_image.at<uchar>(i + 1, j + 1) >= center1) << 3;
			code |= (gray_image.at<uchar>(i + 1, j) >= center1) << 2;
			code |= (gray_image.at<uchar>(i + 1, j - 1) >= center1) << 1;
			code |= (gray_image.at<uchar>(i, j - 1) >= center1) << 0;
			LBPimage.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
	int grid_x = 4;
	int grid_y = 4;
	int numPatterns = 32;
	int width = LBPimage.cols / grid_x;
	int height = LBPimage.rows / grid_y;
	// allocate memory for the spatial histogram
	features = cv::Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given
	// initial result_row
	int resultRowIdx = 0;
	// iterate through grid
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			cv::Mat src_cell = cv::Mat(LBPimage, cv::Range(i*height, (i + 1)*height), cv::Range(j*width, (j + 1)*width));
			//Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), true);
			cv::Mat cell_hist;
			int maxVal = 31;
			int minVal = 0;
			bool normed = false;
			int histSize = maxVal - minVal + 1;
			// Set the ranges.
			float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
			const float* histRange = { range };
			// calc histogram
			calcHist(&src_cell, 1, 0, cv::Mat(), cell_hist, 1, &histSize, &histRange, true, false);
			if (normed) {
				cell_hist /= src_cell.total();
			}
			cell_hist.reshape(1, 1);
			// copy to the result matrix
			cv::Mat result_row = features.row(resultRowIdx);
			cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	features = features.reshape(1, 1);
}


/****************************************************************
input: RGB image cv::Mat
output:candidate plate cv::Mat vector
*****************************************************************/
void Plate:: getCandidatePlate(cv::Mat& img_rgb, vector<cv::Mat>& plate_candidate) {
	cv::Mat img_rgb_copy, img_rgb_contours, img_gray;
	img_rgb.copyTo(img_rgb_copy);
	img_rgb.copyTo(img_rgb_contours);

	//对图像进行高斯滤波，为Sobel算子计算去除干扰噪声；
	cv::Mat img_gaussian_blur;
	GaussianBlur(img_rgb, img_gaussian_blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	imshow("gaussian blur", img_gaussian_blur);
	//图像灰度化，提高运算速度
	cvtColor(img_gaussian_blur, img_gray, cv::COLOR_BGR2GRAY);
	imshow("car_picture_gray", img_gray);
	//对图像进行Sobel运算，得到图像的一阶水平方向导数；
	cv::Mat grad_x;
	cv::Mat abs_grad_x;
	Sobel(img_gray, grad_x, CV_16S, 1, 0, 3);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("sobel grad x", abs_grad_x);
	//通过otsu进行阈值分割；
	cv::Mat grad;
	addWeighted(abs_grad_x, 1, 0, 0, 0, grad);
	imshow("grad", grad);
	cv::Mat img_threshold;
	double ostu_thresh_val =
		threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU +
			CV_THRESH_BINARY);
	//imshow("img threshold", img_threshold);
	//通过形态学闭操作，连接车牌区域。
	cv::Mat img_close;
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)); //Size(15, 15)
	morphologyEx(img_threshold, img_close, cv::MORPH_CLOSE, element);
	imshow("img close", img_close);
	//画出所有外接轮廓,通过车牌先验知识进行初步筛选
	vector<vector<cv::Point>> contours;
	findContours(img_close,
		contours,
		CV_RETR_EXTERNAL,
		CV_CHAIN_APPROX_NONE);
	vector<vector<cv::Point>>::iterator itc = contours.begin();
	vector<cv::RotatedRect> first_rects;

	//显示所有外接轮廓
	vector<cv::RotatedRect> all_rects;//调试用，显示所有轮廓
	while (itc != contours.end()) {
		cv::RotatedRect mr = minAreaRect(cv::Mat(*itc));
		
		all_rects.push_back(mr);

		
		++itc;
	}
	for (size_t i = 0; i <all_rects.size(); i++) {
		cv::RotatedRect roi_rect = all_rects[i];
		cv::Rect roi_bounding_rect = roi_rect.boundingRect();
		//rectangle(img_close, roi_rect);
		cv::Scalar color(255, 255, 0);
		rectangle(img_rgb_contours, roi_bounding_rect, cv::Scalar(0, 0, 255));



	}
	imshow("all contours", img_rgb_contours);


	itc = contours.begin();
	while (itc != contours.end()) {
		cv::RotatedRect mr = minAreaRect(cv::Mat(*itc));
		float area = mr.size.height * mr.size.width;
		float r = (float)mr.size.width / (float)mr.size.height;
		if (r < 1) r = (float)mr.size.height / (float)mr.size.width;
		int min = 34 * 8 * 1;
		int max = 34 * 8 * 24 * 6;
		float rmin = 4 - 4 * 0.9;
		float rmax = 4 + 4 * 0.9;
		//first_rects.push_back(mr);
		if ((area > min && area < max) && (r > rmin && r < rmax)) {
			first_rects.push_back(mr);

		}
		++itc;
	}
	vector<cv::Mat> plate_mat;
	for (size_t i = 0; i < first_rects.size(); i++) {
		cv::RotatedRect roi_rect = first_rects[i];
		cv::Rect roi_bounding_rect = roi_rect.boundingRect();
		//rectangle(img_close, roi_rect);
		cv::Scalar color(255, 255, 0);
		rectangle(img_rgb, roi_bounding_rect, cv::Scalar(0, 0, 255));
		if (roi_bounding_rect.y < 0 || roi_bounding_rect.x < 0 || 
			img_rgb_copy.size().height < (roi_bounding_rect.y + roi_bounding_rect.height) ||
			img_rgb_copy.size().width  < (roi_bounding_rect.x + roi_bounding_rect.width ) )continue;
		plate_mat.push_back(cv::Mat(img_rgb_copy, roi_bounding_rect));
		plate_candidate.push_back(cv::Mat(36, 136, 16));
		//plate_candidate[i] = img_rgb(roi_bounding_rect);


	}
	for (size_t i = 0; i < plate_mat.size(); i++) {
		resize(plate_mat[i], plate_candidate[i], plate_candidate[i].size());
		//imwrite("plate_candidate" + to_string(i) + ".jpg", plate_candidate[i]);
		imwrite("plate_candidate" + to_string(i) + ".jpg", plate_candidate[i]);

	}
	imshow("roi_bounding_rect_plate", img_rgb);
	cv::waitKey(1);

}
}