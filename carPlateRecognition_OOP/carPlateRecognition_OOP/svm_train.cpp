#include "svm_train.h"
#include "plate.h"
using namespace cpr;


/****************************************************************
default SvmTrain constrtctor
*****************************************************************/
SvmTrain::SvmTrain()
{
	m_plate_has_folder = "train\\has";
	m_plate_no_folder = "train\\no";
}


/****************************************************************
from folder with plates which are plates to get plate data
*****************************************************************/
void SvmTrain::getPlate(cv::Mat & trainingImages, vector<int>& trainingLabels)
{
	
	vector<string> files;
	getFiles(m_plate_has_folder, files);

	int size = files.size();
	if (0 == size)
		cout << "No File Found in train\\has!" << endl;

	for (int i = 0; i < size; i++)
	{
		cout << files[i].c_str() << endl;
		cv::Mat img = cv::imread(files[i].c_str());
		cv::Mat features;
		Plate those_plate;
		those_plate.getLBPFeatures(img, features);

		trainingImages.push_back(features);
		trainingLabels.push_back(1);

	}
	setTrainingImages(trainingImages);
	setTrainingLabels(trainingLabels);
}


/****************************************************************
from folder with plates which are not plates to get no plate data 
*****************************************************************/
void SvmTrain::getNoPlate(cv::Mat & trainingImages, vector<int>& trainingLabels)
{
	vector<string> files;

	getFiles(m_plate_no_folder, files);
	int size = files.size();
	if (0 == size)
		cout << "No File Found in train\\no!" << endl;

	for (int i = 0; i < size; i++)
	{
		cout << files[i].c_str() << endl;
		cv::Mat img = cv::imread(files[i].c_str());
		cv::Mat features;
		Plate those_plate;
		those_plate.getLBPFeatures(img, features);
		trainingImages.push_back(features);
		trainingLabels.push_back(0);
	}
	setTrainingImages(trainingImages);
	setTrainingLabels(trainingLabels);

}


/****************************************************************
get training data 
*****************************************************************/
void SvmTrain::makeTrainingData()
{
	//准备训练数据
	cv::Mat classes;//(numPlates+numNoPlates, 1, CV_32FC1);
	cv::Mat trainingData;//(numPlates+numNoPlates, imageWidth*imageHeight, CV_32FC1 );

	cv::Mat trainingImages;
	vector<int> trainingLabels;

	getPlate(trainingImages, trainingLabels);
	getNoPlate(trainingImages, trainingLabels);

	cv::Mat(trainingImages).copyTo(trainingData);
	
	trainingData.convertTo(trainingData, CV_32FC1);
	cv::Mat(trainingLabels).copyTo(classes);
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingData, cv::ml::SampleTypes::ROW_SAMPLE, classes);
	setTrainingdata(tdata);
}

void SvmTrain::getSvmModel()
{
	cv::Ptr<cv::ml::SVM> model = cv::ml::SVM::create();//以下是设置SVM训练模型的配置
	model->setType(cv::ml::SVM::C_SVC);
	model->setKernel(cv::ml::SVM::LINEAR);
	model->setGamma(1);
	model->setC(1);
	model->setCoef0(0);
	model->setNu(0);
	model->setP(0);
	model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));
	setSvm(model);
}


/****************************************************************
input:file path
output:a vector of the file path's all file name
*****************************************************************/
void SvmTrain::getFiles(string path, vector<string>& files)
{
	//文件句柄
	long  long hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
