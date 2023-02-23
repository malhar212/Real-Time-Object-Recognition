/*
Malhar Mahant & Kruthika Gangaraju
SP23

Implements different object recognition functions.
*/
#include "objectRecognitionFunctions.h"

int objectrecognition::generateBinaryImage(cv::Mat& src, cv::Mat& dst, int threshold)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if ((short)src.at<uchar>(i, j) < threshold) // Try changing to 128
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return 0;
}

int objectrecognition::erosion(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations)
{
	src.copyTo(dst);
	while (iterations != 0) {
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
					dst.at<uchar>(i, j) = 0;
					continue;
				}
				if ((short) src.at<uchar>(i - 1, j) == 0 || (short) src.at<uchar>(i, j - 1) == 0 || (short) src.at<uchar>(i, j + 1)
					== 0 || (short) src.at<uchar>(i + 1, j) == 0) {
					dst.at<uchar>(i, j) = 0;
				}/*
				else {
					dst.at<uchar>(i, j) = 255;
				}*/
			}
		}
		iterations--;
	}
	return 0;
}

int objectrecognition::dilation(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations)
{
	src.copyTo(dst);
	while (iterations != 0) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (i == 0 || i == src.rows - 1 || j == 0 || j == src.cols - 1) {
					dst.at<uchar>(i, j) = 0;
					continue;
				}
				if ((short)src.at<uchar>(i - 1, j) == 255 || (short)src.at<uchar>(i, j - 1) == 255 || (short)src.at<uchar>(i, j + 1)
					== 255 || (short)src.at<uchar>(i + 1, j) == 255) {
					dst.at<uchar>(i, j) = 255;
				}/*
				else {
					dst.at<uchar>(i, j) = 0;
				}*/
			}
		}
		iterations--;
	}
	return 0;
}

int objectrecognition::segmentImage(cv::Mat& src, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids, std::vector<Vec3b>& colors)
{
	int connectivity = 4;
	int n = 1;
	int numlabels = cv::connectedComponentsWithStats(src, labels, stats, centroids, 8, CV_32S);
	//std::vector<Vec3b> intensity(numlabels);
	if (colors.size() < numlabels) {
		colors.clear();
		colors.push_back(Vec3b(0, 0, 0));
		//intensity[0] = Vec3b(0, 0, 0);
		int area = 1024;
		for (int i = 1; i < numlabels; i++)
		{
			colors.push_back(Vec3b(0, 0, 0));
			//intensity[i] = Vec3b(255, 255, 255);
			if (stats.at<int>(i, CC_STAT_AREA) > area)
			{
				/*area = stats.at<int>(i, CC_STAT_AREA);
			}
			else
			{*/
				colors.at(i) = Vec3b(rand() % 256, rand() % 256, rand() % 256);
				//intensity[i] = Vec3b(0, 0, 0);
			}
		}
	}
	return numlabels;
}

int objectrecognition::selectRegion(cv::Mat& src, cv::Mat& dst, int numlabels, int selected, cv::Mat& labels, cv::Mat& stats, std::vector<Vec3b>& colors) {
	Mat colored_img = Mat::zeros(src.size(), CV_8UC3);
	// Mat intensity_img = Mat::zeros(src.size(), CV_8UC3);
	try {
		for (int i = 0; i < colored_img.rows; i++)
		{
			//std::cout << "Row: " << i << std::endl;
			for (int j = 0; j < colored_img.cols; j++)
			{
				//std::cout << "Col: " << j << std::endl;
				if (selected == 0) {
					int label = labels.at<int>(i, j);
					//std::cout << "Label: " << colors.size() << std::endl;
					colored_img.at<Vec3b>(i, j) = colors[label];
				}
				else {
					int label = labels.at<int>(i, j);
					if (label == selected) {
						colored_img.at<Vec3b>(i, j) = colors[label];
					}
				}
				// intensity_img.at<Vec3b>(i, j) = intensity[label];
			}
		}
	}
	catch (Exception e) {
		std::cerr << e.what() << std::endl;
	}
	colored_img.copyTo(dst);
	return 0;
}
