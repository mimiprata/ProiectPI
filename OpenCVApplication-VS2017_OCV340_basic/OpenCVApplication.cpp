#include "stdafx.h"
#include "pch.h"
#include "common.h"
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
using namespace std;
using namespace cv;




void detectAndDisplay(Mat frame);
int* histogramsCheecks = new int[32 * 256]();
int* histogramsBeard = new int[32 * 256]();
//for mouse selection of the cheecks and beard
struct SelectionState {
	Point startPt, endPt, mousePos;
	bool started = false, done = false;

	Rect toRect() {
		return Rect(
			min(this->startPt.x, this->mousePos.x),
			min(this->startPt.y, this->mousePos.y),
			abs(this->startPt.x - this->mousePos.x),
			abs(this->startPt.y - this->mousePos.y));
	}
};

void onMouse(int event, int x, int y, int, void *data) {
	SelectionState *state = (SelectionState*)data;

	switch (event) {
	case EVENT_LBUTTONDOWN:
		state->startPt.x = x;
		state->startPt.y = y;
		state->mousePos.x = x;
		state->mousePos.y = y;
		state->started = true;
		break;

	case EVENT_LBUTTONUP:
		state->endPt.x = x;
		state->endPt.y = y;
		state->done = true;
		break;

	case EVENT_MOUSEMOVE:
		state->mousePos.x = x;
		state->mousePos.y = y;
		break;
	}
}
//build a Rect type from the rectagle selected
Rect selectRect(Mat image, Scalar color = Scalar(255, 0, 0), int thickness = 2) {
	const string window = "rect";
	SelectionState state;
	namedWindow(window, WINDOW_NORMAL);
	setMouseCallback(window, onMouse, &state);

	while (!state.done) {
		waitKey(100);

		if (state.started) {
			Mat copy = image.clone();
			Rect selection = state.toRect();
			rectangle(copy, selection, color, thickness);
			imshow(window, copy);
		}
		else {
			imshow(window, image);
		}
	}

	return state.toRect();
}

	
CascadeClassifier face_cascade;
unsigned int total = 0;
Mat applyConvolution(Mat ucharSrc, Mat floatKernel) {
	Mat dst = ucharSrc.clone();
	int r = floatKernel.rows / 2;
	int c = floatKernel.cols / 2;
	for (int i = r; i < ucharSrc.rows - r; i++)
		for (int j = c; j < ucharSrc.cols - c; j++) {
			float aux = 0.0f;
			for (int ki = -r; ki <= r; ki++)
				for (int kj = -c; kj <= c; kj++)
					aux += floatKernel.at<float>(r + ki, c + kj) * ucharSrc.at<uchar>(i + ki, j + kj);
			if (aux < 0.0f)
				aux = 0;
			else if (aux > 255.0f)
				aux = 255;
			dst.at<uchar>(i, j) = (uchar)aux;
		}
	return dst;
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

cv::Mat mkKernel(int ks, double sig, double th, double lm, double ps)
{
	int hks = (ks - 1) / 2;
	double theta = th * CV_PI / 180;
	double psi = ps * CV_PI / 180;
	double del = 2.0 / (ks - 1);
	double lmbd = lm;
	double sigma = sig / ks;
	double x_theta;
	double y_theta;
	cv::Mat kernel(ks, ks, CV_32F);
	for (int y = -hks; y <= hks; y++)
	{
		for (int x = -hks; x <= hks; x++)
		{
			x_theta = x * del*cos(theta) + y * del*sin(theta);
			y_theta = -x * del*sin(theta) + y * del*cos(theta);
			kernel.at<float>(hks + y, hks + x) = (float)exp(-0.5*(pow(x_theta, 2) + pow(y_theta, 2)) / pow(sigma, 2))* cos(2 * CV_PI*x_theta / lmbd + psi);
		}
	}
	return kernel;
}

int kernel_size = 21;
int pos_sigma = 5;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 90;
cv::Mat src_f;



void ProcessCheeck(int, void *, Mat src_f)
{
	cv::Mat dest;
	double sig1 = 15, sig2=21;
	double lm1 = 0.5 + 10 / 100.0;
	double lm2 = 0.5 + 20 / 100.0;
	double lm3 = 0.5 + 0 / 100.0;
	double th[8] = {0,60,45,30,160,140,115,90};
	double ps = 45;
	//the 32 filters for Gabor function
	Mat kernels[32] = {};
	kernels[0] = mkKernel(kernel_size, sig1, th[0], lm2, ps);
	kernels[1] = mkKernel(kernel_size, sig1, th[1], lm2, ps);
	kernels[2] = mkKernel(kernel_size, sig1, th[2], lm2, ps);
	kernels[3] = mkKernel(kernel_size, sig1, th[3], lm2, ps);
	kernels[4] = mkKernel(kernel_size, sig1, th[4], lm2, ps);
	kernels[5] = mkKernel(kernel_size, sig1, th[5], lm2, ps);
	kernels[6] = mkKernel(kernel_size, sig1, th[6], lm2, ps);
	kernels[7] = mkKernel(kernel_size, sig1, th[7], lm2, ps);

	kernels[8] = mkKernel(kernel_size, sig1, th[0], lm1, ps);
	kernels[9] = mkKernel(kernel_size, sig1, th[1], lm1, ps);
	kernels[10] = mkKernel(kernel_size, sig1, th[2], lm1, ps);
	kernels[11] = mkKernel(kernel_size, sig1, th[3], lm1, ps);
	kernels[12] = mkKernel(kernel_size, sig1, th[4], lm1, ps);
	kernels[13] = mkKernel(kernel_size, sig1, th[5], lm1, ps);
	kernels[14] = mkKernel(kernel_size, sig1, th[6], lm1, ps);
	kernels[15] = mkKernel(kernel_size, sig1, th[7], lm1, ps);

	kernels[16] = mkKernel(kernel_size, sig2, th[0], lm1, ps);
	kernels[17] = mkKernel(kernel_size, sig2, th[1], lm1, ps);
	kernels[18] = mkKernel(kernel_size, sig2, th[2], lm1, ps);
	kernels[19] = mkKernel(kernel_size, sig2, th[3], lm1, ps);
	kernels[20] = mkKernel(kernel_size, sig2, th[4], lm1, ps);
	kernels[21] = mkKernel(kernel_size, sig2, th[5], lm1, ps);
	kernels[22] = mkKernel(kernel_size, sig2, th[6], lm1, ps);
	kernels[23] = mkKernel(kernel_size, sig2, th[7], lm1, ps);

	kernels[24] = mkKernel(kernel_size, sig2, th[0], lm3, ps);
	kernels[25] = mkKernel(kernel_size, sig2, th[1], lm3, ps);
	kernels[26] = mkKernel(kernel_size, sig2, th[2], lm3, ps);
	kernels[27] = mkKernel(kernel_size, sig2, th[3], lm3, ps);
	kernels[28] = mkKernel(kernel_size, sig2, th[4], lm3, ps);
	kernels[29] = mkKernel(kernel_size, sig2, th[5], lm3, ps);
	kernels[30] = mkKernel(kernel_size, sig2, th[6], lm3, ps);
	kernels[31] = mkKernel(kernel_size, sig2, th[7], lm3, ps);
	
	//compute histogram of Gabor
	int ** histo = (int**)malloc(32 * sizeof(int*));
	for(int i=0;i<32;i++)
		histo[i]=(int*)malloc(256*sizeof(int));
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 256; j++)
			histo[i][j] = 0;
	for (int i = 0; i < 32; i++)
	{

		
		//dest = applyConvolution(src_f, kernels[i]);
		cv::filter2D(src_f, dest, CV_32F, kernels[i]);

		Mat aux = Mat::zeros(dest.rows, dest.cols, CV_32F);
		//normalize
		float max = 0;
		float min = 1000000;
		float pixel;
		for (int i = 0; i < dest.rows; i++)
			for (int j = 0; j < dest.cols; j++)
			{
				pixel = dest.at<float>(i, j);
				if (pixel < min)
					min = pixel;
				if (pixel > max)
					max = pixel;
				aux.at<float>(i, j) = pixel;
			}

		Mat destUint
			= cv::Mat(dest.rows, dest.cols, CV_8UC1);
		aux.convertTo(destUint, CV_8UC1, 255.0 / (max - min), -min);
		

		cv::imshow("Process window", dest);
		//compute histogram
		int* hist = new int[256]();

		for (int i = 0; i < destUint.rows; i++) {
			for (int j = 0; j < destUint.cols; j++) {
				hist[destUint.at<uchar>(i, j)]++;
			}
		}
		histo[i] = hist;
		showHistogram("histogram", histo[i], 256, 200);
	}
	//concatenate the 32 histograms
	int k = 0;
	for(int i=0;i<32;i++)
		for (int j = 0; j < 256; j++)
		{
			histogramsCheecks[k] = histo[i][j];
			k++;
		}
	//l2 normalization
	int val_normalizata1 = 0;
	for (int i = 0; i < 256; i++) {
		val_normalizata1 += histogramsCheecks[i] * histogramsCheecks[i];
	}
	int val_normalizata = sqrt(val_normalizata1);
	for (int i = 0; i < 32 * 256; i++)
		histogramsCheecks[i] = histogramsCheecks[i] / val_normalizata;
	

}




void ProcessBeard(int, void *, Mat src_f)
{
	cv::Mat dest;
	double sig1 = 15, sig2 = 21;
	double lm1 = 0.5 + 10 / 100.0;
	double lm2 = 0.5 + 20 / 100.0;
	double lm3 = 0.5 + 0 / 100.0;
	double th[8] = { 0,60,45,30,160,140,115,90 };
	double ps = 45;

	Mat kernels[32] = {};
	kernels[0] = mkKernel(kernel_size, sig1, th[0], lm2, ps);
	kernels[1] = mkKernel(kernel_size, sig1, th[1], lm2, ps);
	kernels[2] = mkKernel(kernel_size, sig1, th[2], lm2, ps);
	kernels[3] = mkKernel(kernel_size, sig1, th[3], lm2, ps);
	kernels[4] = mkKernel(kernel_size, sig1, th[4], lm2, ps);
	kernels[5] = mkKernel(kernel_size, sig1, th[5], lm2, ps);
	kernels[6] = mkKernel(kernel_size, sig1, th[6], lm2, ps);
	kernels[7] = mkKernel(kernel_size, sig1, th[7], lm2, ps);

	kernels[8] = mkKernel(kernel_size, sig1, th[0], lm1, ps);
	kernels[9] = mkKernel(kernel_size, sig1, th[1], lm1, ps);
	kernels[10] = mkKernel(kernel_size, sig1, th[2], lm1, ps);
	kernels[11] = mkKernel(kernel_size, sig1, th[3], lm1, ps);
	kernels[12] = mkKernel(kernel_size, sig1, th[4], lm1, ps);
	kernels[13] = mkKernel(kernel_size, sig1, th[5], lm1, ps);
	kernels[14] = mkKernel(kernel_size, sig1, th[6], lm1, ps);
	kernels[15] = mkKernel(kernel_size, sig1, th[7], lm1, ps);

	kernels[16] = mkKernel(kernel_size, sig2, th[0], lm1, ps);
	kernels[17] = mkKernel(kernel_size, sig2, th[1], lm1, ps);
	kernels[18] = mkKernel(kernel_size, sig2, th[2], lm1, ps);
	kernels[19] = mkKernel(kernel_size, sig2, th[3], lm1, ps);
	kernels[20] = mkKernel(kernel_size, sig2, th[4], lm1, ps);
	kernels[21] = mkKernel(kernel_size, sig2, th[5], lm1, ps);
	kernels[22] = mkKernel(kernel_size, sig2, th[6], lm1, ps);
	kernels[23] = mkKernel(kernel_size, sig2, th[7], lm1, ps);

	kernels[24] = mkKernel(kernel_size, sig2, th[0], lm3, ps);
	kernels[25] = mkKernel(kernel_size, sig2, th[1], lm3, ps);
	kernels[26] = mkKernel(kernel_size, sig2, th[2], lm3, ps);
	kernels[27] = mkKernel(kernel_size, sig2, th[3], lm3, ps);
	kernels[28] = mkKernel(kernel_size, sig2, th[4], lm3, ps);
	kernels[29] = mkKernel(kernel_size, sig2, th[5], lm3, ps);
	kernels[30] = mkKernel(kernel_size, sig2, th[6], lm3, ps);
	kernels[31] = mkKernel(kernel_size, sig2, th[7], lm3, ps);

	//int* histograms = new int[32 * 256]();
	int ** histo = (int**)malloc(32 * sizeof(int*));
	for (int i = 0; i < 32; i++)
		histo[i] = (int*)malloc(256 * sizeof(int));
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 256; j++)
			histo[i][j] = 0;
	for (int i = 0; i < 32; i++)
	{

		//i 
		//0...0..255
		//1   256..511
		// 2   512..767
		//dest = applyConvolution(src_f, kernels[i]);
		cv::filter2D(src_f, dest, CV_32F, kernels[i]);

		Mat aux = Mat::zeros(dest.rows, dest.cols, CV_32F);

		float max = 0;
		float min = 1000000;
		float pixel;
		for (int i = 0; i < dest.rows; i++)
			for (int j = 0; j < dest.cols; j++)
			{
				pixel = dest.at<float>(i, j);
				if (pixel < min)
					min = pixel;
				if (pixel > max)
					max = pixel;
				aux.at<float>(i, j) = pixel;
			}

		Mat destUint
			= cv::Mat(dest.rows, dest.cols, CV_8UC1);
		aux.convertTo(destUint, CV_8UC1, 255.0 / (max - min), -min);


		cv::imshow("Process windowB", dest);
		
		int* hist = new int[256]();

		for (int i = 0; i < destUint.rows; i++) {
			for (int j = 0; j < destUint.cols; j++) {
				hist[destUint.at<uchar>(i, j)]++;
			}
		}
		histo[i] = hist;
		showHistogram("histogramB", histo[i], 256, 200);
	}
	int k = 0;
	for (int i = 0; i < 32; i++)
		for (int j = 0; j < 256; j++)
		{
			histogramsBeard[k] = histo[i][j];
			
			k++;
		}
	int val_normalizata1 = 0;
	for (int i = 0; i < 256*32; i++) {
		val_normalizata1 += histogramsBeard[i] * histogramsBeard[i];
	}
	int val_normalizata = sqrt(val_normalizata1);
	cout << ("%d\n", val_normalizata);
	for (int i = 0; i < 32 * 256; i++) {
		histogramsBeard[i] = histogramsBeard[i] / val_normalizata;
		
	}
	


}
int main(void)
{
	int camera_device = 0;
	String face_cascade_name = "haarcascade_frontalface_alt.xml";

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	}

	VideoCapture capture;


	Mat frame;
	

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		detectAndDisplay(src);
		
		cv::Mat src1;


		cv::cvtColor(src, src1, CV_BGR2GRAY);
		src1.convertTo(src_f, CV_32F, 1.0 / 255, 0);
		cv::imshow("Src", src_f);

		//for the correct execution you have to select 2 areas from the "rect" image, one of the cheecks and one of the beard
		Rect rect = selectRect(src_f);
		Rect rect1 = selectRect(src_f);
		//construct the image from cheecks
		Mat cheecks = src_f(rect);
		imshow("Cheecks", cheecks);
		//construct the image from beard
		Mat beard = src_f(rect1);
		imshow("Beard", beard);


		//the distance between the two histograms, computed with Chi-Square 
		int d1 = 0;
		/*for (int i = 0; i < 32 * 256; i++) {
			d1 += (histogramsCheecks[i] - histogramsBeard[i])*(histogramsCheecks[i] - histogramsBeard[i]) / histogramsCheecks[i];
		}
		cout << ("%d", d1);*/


		//the distance between the two histograms, computed with Intersection


		int d2 = 0;

		/*for (int i = 0; i < 32 * 256; i++) {
			d2 +=min(histogramsCheecks[i],histogramsBeard[i]);
		}*/


		//the distance between the two histograms, computed with Correlation 

		int d3 = 0;
		/*int N = 32 * 256;
		int s = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			s += histogramsCheecks[i];
		}
		s = s / 32 * 256;
		int t = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			t += histogramsCheecks[i];
		}
		t = t / 32 * 256;

		int d31 = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			d31 += (histogramsCheecks[i] - s)*(histogramsBeard[i] - t);
		}
		int d321 = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			d321 += (histogramsCheecks[i] - s)*(histogramsCheecks[i] - s);
		}
		int d322 = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			d322 += (histogramsBeard[i] - t)*(histogramsBeard[i] - t);
		}

		int d32 = sqrt(d321*d322);
		d3=d31/d32;

		*/
		//the distance between the two histograms, computed with Bhattacharyya
		int d4 = 0;
		
		/*
		int d41 = 0;
		for (int i = 0; i < 32 * 256; i++)
		{
			d41 += sqrt(histogramsCheecks[i] * histogramsBeard[i]);
		}

		d4 = sqrt(1 - (1 / srqt(s*t * 32 * 32 * 256 * 256)*d41));
		*/

		if (!kernel_size % 2)
		{
			kernel_size += 1;
		}
		//process the cheecks
		ProcessCheeck(0, 0,cheecks);
		//process the beard
		ProcessBeard(0, 0, beard);
		cv::waitKey(0);
		
	}
	return 0;
}


void detectAndDisplay(Mat frame)
{
	Mat frame_gray, frame_gray_eql;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray_eql);

	//-- Detect faces
	vector<Rect> faces;
	vector<int> nbr_detect;

	face_cascade.detectMultiScale(frame_gray_eql, faces, nbr_detect);
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		Mat faceROI = frame(faces[i]);

		//imshow("roi", faceROI);
		cout << "nbr of detections = " << nbr_detect[i] << endl;
		cout << "detected faces = " << faces.size() << endl;
	}

	total = total + faces.size();
	cout << "total faces detected = " << total << endl;

	imshow("Capture - Face detection", frame);
}