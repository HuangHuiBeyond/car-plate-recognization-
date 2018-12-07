#include "chars.h"
#include "core_func.h"
namespace cpr{
	cv::Mat CChars::getCharFeatures(cv::Mat in, int sizeData) {
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;


	// Low data feature
	cv::Mat lowData;
	resize(in, lowData, cv::Size(sizeData, sizeData));

	// Histogram features
	cv::Mat vhist = ProjectedHistogram(lowData, VERTICAL);
	cv::Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

	cv::Mat out = cv::Mat::zeros(1, numCols, CV_32F);
	// Asign values to

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++) {
		for (int y = 0; y < lowData.rows; y++) {
			out.at<float>(j) += (float)lowData.at <unsigned char>(x, y);
			j++;
		}
	}

	//std::cout << out << std::endl;

	return out;
	}


	void CChars::charsClassify() {
		cv::Ptr<cv::ml::ANN_MLP> ann_ = cv::ml::ANN_MLP::load("ann.xml");
		cv::Ptr<cv::ml::ANN_MLP> annChinese_ = cv::ml::ANN_MLP::load("ann_chinese.xml");
		//初始化为指向char数组的指针
		//static const char *kChars[] = {
		//	"0", "1", "2",
		//	"3", "4", "5",
		//	"6", "7", "8",
		//	"9",
		//	/*  10  */
		//	"A", "B", "C",
		//	"D", "E", "F",
		//	"G", "H", /* {"I", "I"} */
		//	"J", "K", "L",
		//	"M", "N", /* {"O", "O"} */
		//	"P", "Q", "R",
		//	"S", "T", "U",
		//	"V", "W", "X",
		//	"Y", "Z",
		//	/*  24  */
		//	"zh_cuan" , "zh_e"    , "zh_gan"  ,
		//	"zh_gan1" , "zh_gui"  , "zh_gui1" ,
		//	"zh_hei"  , "zh_hu"   , "zh_ji"   ,
		//	"zh_jin"  , "zh_jing" , "zh_jl"   ,
		//	"zh_liao" , "zh_lu"   , "zh_meng" ,
		//	"zh_min"  , "zh_ning" , "zh_qing" ,
		//	"zh_qiong", "zh_shan" , "zh_su"   ,
		//	"zh_sx"   , "zh_wan"  , "zh_xiang",
		//	"zh_xin"  , "zh_yu"   , "zh_yu1"  ,
		//	"zh_yue"  , "zh_yun"  , "zh_zang" ,
		//	"zh_zhe"
		//	/*  31  */
		//};

		static const char *kChars[] = {
			"0", "1", "2",
			"3", "4", "5",
			"6", "7", "8",
			"9",
			/*  10  */
			"A", "B", "C",
			"D", "E", "F",
			"G", "H", /* {"I", "I"} */
			"J", "K", "L",
			"M", "N", /* {"O", "O"} */
			"P", "Q", "R",
			"S", "T", "U",
			"V", "W", "X",
			"Y", "Z",
			/*  24  */
			"川" , "鄂"    , "赣"  ,
			"甘" , "贵"  , "桂" ,
			"黑"  , "沪"   , "冀"   ,
			"津"  , "京" , "吉"   ,
			"辽" , "鲁"   , "蒙" ,
			"闽"  , "宁" , "青" ,
			"琼", "陕" , "苏"   ,
			"晋"   , "皖"  , "湘",
			"新"  , "豫"   , "渝"  ,
			"粤"  , "云"  , "藏" ,
			"浙"
			/*  31  */
		};
		cv::String finalResult = " " ;
		vector<cv::Mat> charsMat = getCharsMat();
		vector<cv::Mat> charsFeatures;
		for (int i = 0; i < charsMat.size(); i++) {
			if (i == 0) {
				charsFeatures.push_back(getCharFeatures(charsMat[0], 20));
			}
			else {
				charsFeatures.push_back(getCharFeatures(charsMat[i], 10));
			}
		}
		//得到了特征向量组成的向量
		setCharsFeatures(charsFeatures);
		cv::Mat output(1, 65, CV_32FC1);
		float maxVal = -2.f;
		char charIndex = 0;
		for (int i = 0; i < charsFeatures.size(); i++) {
			int j = 0;
			if (i != 0) {
				ann_->predict(charsFeatures[i], output);
				/*cout << output.size();
				cout << output << endl << endl;*/
				for (j = 0; j < 34; j++) {
					if (output.at<float>(j) > maxVal) {
						maxVal = output.at<float>(j);
						charIndex = j;
					}
				}
			}
			else {
				annChinese_->predict(charsFeatures[i], output);
				/*cout << output.size();
				cout << output << endl << endl;*/
				for (j =34; j < 65; j++) {
					if (output.at<float>(j-34) > maxVal) {
						maxVal = output.at<float>(j-34);
						charIndex = j;
					}
				}
			}
			//String word;
			//cout << *(kChars + charIndex);
			finalResult += *(kChars + charIndex);
			maxVal = -2.f;

		}

		cout << "This plate is recognized as: " << finalResult;


	}
}
