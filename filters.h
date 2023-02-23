/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Utility functions to apply filters to an image.
 */
#ifndef _FILTERS_H_
#define _FILTERS_H_
#include <opencv2/opencv.hpp>
using namespace cv;

namespace customfilters {
	// Alternative method to implement greyscale
	int greyscale(cv::Mat& src, cv::Mat& dst);
	// Method to implement negative filter
	int negative(cv::Mat& src, cv::Mat& dst);
	// Method to implement separable Gaussian Filter
	int blur5x5(cv::Mat& src, cv::Mat& dst);
	// Implements SobelX filter
	int sobelX3x3(cv::Mat& src, cv::Mat& dst);
	// Implements SobelY filter
	int sobelY3x3(cv::Mat& src, cv::Mat& dst);
	// Implements sobel filter using OpenCVs inbuilt methods
	int sobel(cv::Mat& src, cv::Mat& dst);
	// Finds magnitude of edge using SobelX and SobelY filters
	int magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst);
	// Helper method to implement magnitude filter
	int magnitudeHelper(cv::Mat& src, cv::Mat& dst);
	// Method to implement blur quantize
	int blurQuantize(cv::Mat& src, cv::Mat& dst, int levels);
	// Method to implement cartoonization
	int cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold);
	// Method to simulate deuteranomaly (green blindness)
	int deuteranomaly(cv::Mat& src, cv::Mat& dst);
	// Function to apply set brightness to the image
	void brightness(int brightness, cv::Mat& dst);
	// Function to apply Gabor Filter to an image using given theta and frequency
	int gaborFilter(cv::Mat& src, cv::Mat& dst, double theta, double freq);
}
#endif
