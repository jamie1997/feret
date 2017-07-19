#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;

int dist_cosine(cv::Mat ftest, cv::Mat ftrain)

{
	//ftest = ftest.t();

	//ftrain = ftrain.t();

	int ftestCols = ftest.cols;

	int ftrainRows = ftrain.rows;

	Vector<double> sums;

	double testSum, trainSum, dotResult;

	double minSum, minIndex = 0;

	ftest.convertTo(ftest, CV_64FC1);

	ftrain.convertTo(ftrain, CV_64FC1);



	// 1. 求ftest的元素的平方和的开平方

	testSum = 0;



	for (int i = 0; i < ftestCols; ++i)

	{

		testSum += pow(ftest.at<double>(0, i), 2);

	}

	testSum = sqrt(testSum);



	// 2. 求train中各个向量



	for (int m = 0; m < ftrainRows; ++m)

	{

		trainSum = 0;

		for (int j = 0; j < ftrain.cols; ++j)

		{

			trainSum += pow(ftrain.at<double>(m, j), 2);

		}

		trainSum = sqrt(trainSum);

		//dotResult = cvDotProduct(ftest.ptr<double>(0), ftrain.ptr<double>(i));

		dotResult = ftest.row(0).dot(ftrain.row(m));//此处有错

		sums.push_back(1 - abs(dotResult / (trainSum * testSum)));

	}

	// 3. 求最小值的下标

	minSum = sums[0];

	for (int i = 0; i < ftrainRows; i++)

	{

		if (sums[i] < minSum)

		{
			minIndex = i;
			minSum = sums[i];
		}

	}
	return minIndex;
	cout << minIndex;

}


int main()
{
	IplImage* img = NULL;
	IplImage* change;
	vector<cv::Mat> InImgs;
	cv::Mat* bmtx;

	vector<string> InLabel;
	Mat Intrain;

	ifstream infile("_training_cd.txt");
	vector<string> sname;
	string str;
	char charArray[256];

	while (infile >> str)
		sname.push_back(str);
	int ssize = sname.size();
	char path[256];
	for (int i = 0; i < ssize; i++)
	{
		strncpy(charArray, sname[i].c_str(), sizeof(sname[i]));
		sprintf(path, "%s%s%s", "../CSU-Std-Old/", charArray, ".bmp");
		img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
		if (img == NULL)
			continue;
		change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
		cvConvertScale(img, change, 1.0 / 255, 0);
		bmtx = new cv::Mat(change);
		InImgs.push_back(*bmtx);
		InLabel.push_back(sname[i]);
	}
	cout << "trainsize:" << ssize << "InImgssize:" << InImgs.size();
	vector<int> NumFilters;

	NumFilters.push_back(8);

	NumFilters.push_back(8);

	vector<int> blockSize;

	blockSize.push_back(15);   //  height / 4

	blockSize.push_back(15);    //  width / 4



	PCANet pcaNet = {

		2,

		5,

		NumFilters,

		blockSize,

		0

	};
	cout << "\r\n ====== Trained PCA filters ======= \r\n" << endl;
	int64 e1 = cv::getTickCount();

	PCA_Train_Result* result = PCANet_train(InImgs, &pcaNet, true);
	Intrain = result->Features;

	int64 e2 = cv::getTickCount();

	double time = (e2 - e1) / cv::getTickFrequency();

	cout << " PCANet Filterget time: " << time << endl;

	cout << "\n ====== PCA Training ======= \n" << endl;

	IplImage* trnimg;

	IplImage *trnchange;

	cv::Mat* trnbmtx;

	vector<cv::Mat> trnImgs;

	vector<string> trnLabel;

	vector<string> names;

	Mat ftrain;

	ifstream infile2("gallery.txt");
	vector<string> gall;

	while (infile2 >> str)

	{

		gall.push_back(str);

	}

	int gallsize = gall.size();
	cout << "gallsize:" << gallsize;

	for (int i = 0; i < gallsize; i++)

	{

		strncpy(charArray, gall[i].c_str(), sizeof(gall[i]));

		sprintf(path, "%s%s%s", "../CSU-Std-Old/", charArray, ".bmp");//同上

		trnimg = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);

		if (trnimg == NULL)

			continue;

		trnchange = cvCreateImage(cvGetSize(trnimg), IPL_DEPTH_64F, trnimg->nChannels);

		cvConvertScale(trnimg, trnchange, 1.0 / 255, 0);

		trnbmtx = new cv::Mat(trnchange);

		trnImgs.push_back(*trnbmtx);

		trnLabel.push_back(gall[i]);
	}
	cout << "trnImgssize:" << trnImgs.size();

	e1 = cv::getTickCount();

	PCA_Train_Result* trnresult = PCANet_train(trnImgs, &pcaNet, true);

	ftrain = trnresult->Features;

	e2 = cv::getTickCount();

	time = (e2 - e1) / cv::getTickFrequency();

	cout << " PCANet Training time: " << time << endl;
	

	IplImage* tstimg;
	IplImage* tstchange;
	cv::Mat* tstbmtx;

	vector<cv::Mat> testImgs;

	vector<string> testLabel;

	cv::Mat ftest;

	cout << "=======PCANet Testing======" << endl;

	ifstream infile3("dup1.txt");

	vector<string> dup;

	while (infile3 >> str)

	{

		dup.push_back(str);

	}

	int dupsize1 = dup.size();

	for (int i = 0; i<dupsize1; i++)

	{

		strncpy(charArray, dup[i].c_str(), sizeof(dup[i]));

		sprintf(path, "%s%s%s", "../CSU-Std-Old/", charArray, ".bmp");//同上

		tstimg = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);

		if (tstimg == NULL)

			continue;

		tstchange = cvCreateImage(cvGetSize(tstimg), IPL_DEPTH_64F, tstimg->nChannels);

		cvConvertScale(tstimg, tstchange, 1.0 / 255, 0);

		tstbmtx = new cv::Mat(tstchange);

		testImgs.push_back(*tstbmtx);

		testLabel.push_back(dup[i]);

	}

	int dupsize = testImgs.size();
	int duplabelsize = testLabel.size();

	cout << "dupsize1:" << dupsize1 << "dupsize:" << dupsize;

	Hashing_Result* hashing_r;

	PCA_Out_Result *out;

	float correct = 0;

	e1 = cv::getTickCount();

	for (int i = 0; i < dupsize; i++)

	{

		out = new PCA_Out_Result;

		out->OutImgIdx.push_back(0);

		out->OutImg.push_back(testImgs[i]);

		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize,

			pcaNet.NumFilters[0], result->Filters[0], 2);

		for (int j = 1; j<pcaNet.NumFilters[1]; j++)

		    out->OutImgIdx.push_back(j);



		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize,

			pcaNet.NumFilters[1], result->Filters[1], 2);

		hashing_r = HashingHist(&pcaNet, out->OutImgIdx, out->OutImg);

		hashing_r->Features.convertTo(hashing_r->Features, CV_64FC1);

        ftest = hashing_r->Features;
		int ind = dist_cosine(ftest, ftrain);
		string xLabel = trnLabel[ind];


		cout << "    " <<xLabel<<"---"<<testLabel[i];
		if (!xLabel.compare(0, 5, testLabel[i], 0, 5))

			correct++;







		delete out;

	}



	e2 = cv::getTickCount();

	time = (e2 - e1) / cv::getTickFrequency();

	cout << " test time usage: " << time << endl;

	cout << "correct:" << correct << "dupsize:" << dupsize << endl;



	cout << "Accuracy: " << correct / (float)dupsize << endl;

	system("pause");

	return 0;
}