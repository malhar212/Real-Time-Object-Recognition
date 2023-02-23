#pragma once
/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Utility functions for object recognition.
 */
#ifndef _OBJECTRECOGNITIONFUNCTIONS_H_
#define _OBJECTRECOGNITIONFUNCTIONS_H_
#include <opencv2/opencv.hpp>
using namespace cv;

namespace objectrecognition {
	int generateBinaryImage(cv::Mat& src, cv::Mat& dst, int threshold);
	int erosion(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations);
	int dilation(cv::Mat& src, cv::Mat& dst, int connectedness, int iterations);
	int segmentImage(cv::Mat& src, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids, std::vector<Vec3b>& colors);
	int selectRegion(cv::Mat& src, cv::Mat& dst, int numlabels, int selected, cv::Mat& labels, cv::Mat& stats, std::vector<Vec3b>& colors);
}
#endif