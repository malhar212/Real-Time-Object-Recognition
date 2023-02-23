/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Defines functions to generate different kinds of features from an image and find similar images using generated features.
*/
#pragma once
#ifndef _MATCHFUNCTIONS_H_
#define _MATCHFUNCTIONS_H_
#include <opencv2/opencv.hpp>
namespace matchfunctions {
	// Calculate features using 9x9 pixel center crop of a given image
	int calculateHuMoments(cv::Mat& src, cv::Mat& target, std::vector<float>& features);
	int findAllContours(cv::Mat& src, std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy);
	int generateFeatures(std::vector<Point>& contour, std::vector<float>& features);
	int generateAndSaveFeatures(std::string label, std::vector<Point>& contour);
	int nearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels);
	int kNearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels);
}
#endif
