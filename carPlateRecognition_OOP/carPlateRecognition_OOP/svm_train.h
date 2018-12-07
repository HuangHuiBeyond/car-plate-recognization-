#ifndef SVM_TRAIN_H
#define SVM_TRAIN_H
#include <opencv2/opencv.hpp>
//using namespace cv;
using namespace std;
namespace cpr {
	class SvmTrain {
	private:
		cv::Mat m_trainingImages;
		vector<int> m_trainingLabels;
		cv::Ptr<cv::ml::TrainData> m_tdata;
		cv::Ptr<cv::ml::SVM> m_svm;
		const char* m_svm_xml;
		const char* m_plate_has_folder;
		const char* m_plate_no_folder;
		void getFiles(string path, vector<string>& files);



	public:
		SvmTrain();
		void getPlate(cv::Mat& trainingImages, vector<int>& trainingLabels);
		void getNoPlate(cv::Mat& trainingImages, vector<int>& trainingLabels);
		void makeTrainingData();
		void getSvmModel();

		void setTrainingImages(cv::Mat param) { m_trainingImages = param; }
		cv::Mat getTrainingImages() { return m_trainingImages; }

		void setTrainingLabels(vector<int> param) { m_trainingLabels = param; }
		vector<int> getTrainingLabels() { return m_trainingLabels; }

		void setTrainingdata(cv::Ptr<cv::ml::TrainData> param) { m_tdata = param; }
		cv::Ptr<cv::ml::TrainData> getTrainingData() { return m_tdata; }

		void setSvm(cv::Ptr<cv::ml::SVM> param) { m_svm = param; }
		cv::Ptr<cv::ml::SVM> getSvm() { return m_svm; }

		void setSvmXml(const char* param) { m_svm_xml = param; }
		const char* getSvmXml() { return m_svm_xml; }
	};
}
#endif
