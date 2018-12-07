#include "plate.h"
#include "svm_train.h"
#include "chars_segment.h"
#include "chars.h"
#include <afxdlgs.h>
using namespace cpr;
void getXML(SvmTrain& my_svm);
bool getPlate(cv::String filePath, Plate& my_plate,  SvmTrain& my_svm); 
void testSvm(Plate& my_plate, SvmTrain& my_svm);
cv::String selectPicture();
int main(int argc, const char* argv[]) {



	cv::String filePath = selectPicture();
	//CFileDialog fileDig(true);
 	cpr::SvmTrain my_svm;
	cpr::Plate my_plate;
	cpr::CCharsSegment myCharsSegment;
	cpr::CChars myChars;
	//getXML(my_svm);
	//test
	//testSvm(my_plate, my_svm);
	getPlate(filePath, my_plate, my_svm);
	cv::Mat plate = my_plate.getPlate();
	//waitKey(0);
	vector<cv::Mat> plateChars;
	myCharsSegment.charsSegment(plate, plateChars);
	myChars.setCharsMat(plateChars);
	myChars.charsClassify();
	cv::waitKey(0);
	return 0;

}

/****************************************************************
input:
output:.xml file
*****************************************************************/
void getXML(SvmTrain& my_svm) {
	
	my_svm.getSvmModel();
	cv::Ptr<cv::ml::SVM> model = my_svm.getSvm();
	my_svm.makeTrainingData();
	cv::Ptr<cv::ml::TrainData> tdata = my_svm.getTrainingData();
	model->train(tdata);
	model->save("svm.xml");//保存
	my_svm.setSvmXml("svm.xml");
}


/****************************************************************
input:
output: a plate mat
*****************************************************************/
bool getPlate(cv::String  filePath, Plate& my_plate, SvmTrain& my_svm) {
	my_svm.setSvmXml("svm.xml");
	cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::load(my_svm.getSvmXml());
	//Mat img_rgb = imread("1.bmp", -1);
	//Mat img_rgb = imread("2.png", -1);
	//Mat img_rgb = imread("3.jpg", -1);
	//Mat img_rgb = imread("4.jpg", -1);
	//Mat img_rgb = imread("5.jpg", -1);
	//Mat img_rgb = imread("6.jpg", -1);
	//Mat img_rgb = imread("7.jpg", -1);
	//Mat img_rgb = imread("8.jpg", -1);

	//Mat img_rgb = imread("N1.jpg", -1);
	//Mat img_rgb = imread("N2.jpg", -1);
	cv::Mat img_rgb = cv::imread(filePath, -1);


	//Mat img_rgb = imread("M1.jpg", -1);//wierd
	//Mat img_rgb = imread("M2.jpg", -1);
	//Mat img_rgb = imread("M3.jpg", -1);


	my_plate.setImg(img_rgb);
	vector<cv::Mat> plate_candidate;
	my_plate.getCandidatePlate(img_rgb, plate_candidate);
	my_plate.setPlateCandidate(plate_candidate);
	auto pd = plate_candidate.begin();
	bool havePlate = false;
	cv::Mat features = my_plate.getFeatures();;
	for (pd = plate_candidate.begin(); pd != plate_candidate.end(); pd++) {
		imshow("plate candidates", *pd);
		//此处增强程序鲁棒性，因为svm训练的时候用的是jpg文件，所以这里进行格式转换
		imwrite("plate_candidateeeee.jpg", *pd);
		cv::waitKey(1);
		cv::Mat plate_to_test = cv::imread("plate_candidateeeee.jpg", -1);
		my_plate.getLBPFeatures(plate_to_test, features);
		//my_plate.getLBPFeatures(*pd, features);
		//cout << features << endl;
		float response = svm_->predict(features);
		int result_plate = (int)response;
		if (result_plate == 1) {
			imshow("plate", *pd);
			//waitKey(1000);
			my_plate.setPlate(*pd);
			my_plate.setFeatures(features);
			havePlate = true;
			return havePlate;
		}
	}
	if (!havePlate) {
		cout << "error:can't fina a plate";
	}
	return havePlate;
}


/****************************************************************
input: Plate& my_plate, SvmTrain& my_svm
output: 1(plate), 0(not plate)
*****************************************************************/
void testSvm(Plate& my_plate, SvmTrain& my_svm) {
	my_svm.setSvmXml("svm.xml");
	cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::load(my_svm.getSvmXml());
	cv::Mat plateImg = cv::imread("plate_candi.jpg", -1);
	cv::Mat features = my_plate.getFeatures();
	my_plate.getLBPFeatures(plateImg, features);
	cout << features << endl << endl;
	float response = svm_->predict(features);
	int result_plate = (int)response;
	cout << "test result：" + to_string(result_plate);

}


/****************************************************************
input:
output: filtpath of a select picture
*****************************************************************/
cv::String selectPicture() {
	CString filePath; //保存打开文件的路径
	CString defaultDir = _T("F:\\MyProject\C++\\visual studio 2015\\carPlateRecognition_OOP\\carPlateRecognition_OOP"); //设置默认打开文件夹
	CString fileFilter = _T("文件(*.jpg;*.bmp)|*.jpg;*.bmp|All File (*.*)|*.*||"); //设置文件过滤
	CFileDialog *fileDlgP = new CFileDialog(true, defaultDir, _T(""), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, fileFilter, NULL);
	//CFileDialog fileDlgP(true, defaultDir, _T(""), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, fileFilter, NULL);
	//cv::String cvFilePath = "1"; return cvFilePath;
	cout << "Please select a picture to recognize: " << endl;
	//弹出选择文件对话框
	if (fileDlgP -> DoModal() == IDOK)
	{
		
		filePath = fileDlgP->GetPathName();//得到完整的文件名和目录名拓展名  
		CString filename = fileDlgP ->GetFileName();
		cv::String cvFilePath;
		cvFilePath = CStringA(filePath);
		delete fileDlgP;
		return cvFilePath;
	}
	
}