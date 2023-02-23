/*
Malhar Mahant & Kruthika Gangaraju
SP23

Implements different kinds of filters.
*/
#include <iostream>
#include "filters.h"
using namespace std;
using namespace customfilters;

// Alternative method to implement greyscale
int customfilters::greyscale(cv::Mat& src, cv::Mat& dst) {
	// Convert destination to single channel
	dst = Mat(src.rows, src.cols, CV_8UC1);
	cv::Mat transform = Mat();
	// Transform source to HLS color space
	cv::cvtColor(src, transform, cv::COLOR_BGR2HLS);
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			Vec3b hls = transform.at<Vec3b>(j, i);
			// Create image using only L from HLS to obtain greyscale effect
			dst.at<uchar>(j, i) = (uchar)hls[1];
		}
	}
	return 0;
}

// Method to implement separable Gaussian Filter
int customfilters::blur5x5(cv::Mat& src, cv::Mat& dst) {
	cv::Mat kernelX = (Mat_<short>(1, 5) << 1, 2, 4, 2, 1);
	cv::Mat kernelY = (Mat_<short>(5, 1) << 1, 2, 4, 2, 1);
	// Applying horizontal filter
	for (int j = 0; j < src.rows; j++) {
		for (int i = 2; i < src.cols - 2; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -2; l < 3; l++) {
				int matIndex = 2;
				Vec3b bgr = src.at<Vec3b>(j, i + l);
				b = b + (bgr[0] * kernelX.at<short>(0, matIndex + l));
				g = g + (bgr[1] * kernelX.at<short>(0, matIndex + l));
				r = r + (bgr[2] * kernelX.at<short>(0, matIndex + l));
			}
			Vec3b* row = dst.ptr<Vec3b>(j);
			row[i][0] = b/10;
			row[i][1] = g/10;
			row[i][2] = r/10;
		}
	}

	// Applying vertical filter
	for (int j = 2; j < dst.rows - 2; j++) {
		for (int i = 0; i < dst.cols; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -2; l < 3; l++) {
				int matIndex = 2;
				Vec3b bgr = dst.at<Vec3b>(j + l, i);
				b = b + (bgr[0] * kernelY.at<short>(matIndex + l, 0));
				g = g + (bgr[1] * kernelY.at<short>(matIndex + l, 0));
				r = r + (bgr[2] * kernelY.at<short>(matIndex + l, 0));
			}
			Vec3b* bgr = dst.ptr<Vec3b>(j);
			bgr[i][0] = b / 10;
			bgr[i][1] = g / 10;
			bgr[i][2] = r / 10;
		}
	}
	return 0;
}

// Implements SobelX filter
int customfilters::sobelX3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat sx = cv::Mat(src.size(), CV_16SC3);
	cv::Mat kernel = (Mat_<short>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	cv::Mat kernelY = (Mat_<short>(3, 1) << 1, 2, 1);
	cv::Mat kernelX = (Mat_<short>(1, 3) << 1, 0, -1);
	// Apply Horizontal kernel
	for (int j = 0; j < src.rows; j++) {
		for (int i = 1; i < src.cols - 1; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -1; l < 2; l++) {
				int matIndex = 1;
				Vec3b bgr = src.at<Vec3b>(j, i + l);
				b = b + (bgr[0] * kernelX.at<short>(0, matIndex + l)); // Kernel is small at shouldn't take much time
				g = g + (bgr[1] * kernelX.at<short>(0, matIndex + l));
				r = r + (bgr[2] * kernelX.at<short>(0, matIndex + l));
			}
			Vec3s *rptr = sx.ptr<Vec3s>(j);
			rptr[i][0] = b;
			rptr[i][1] = g;
			rptr[i][2] = r;
		}
	}
	// Apply Vertical Kernel
	for (int j = 1; j < sx.rows - 1; j++) {
		for (int i = 1; i < sx.cols - 1; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -1; l < 2; l++) {
				int matIndex = 1;
				Vec3s bgr = sx.at<Vec3s>(j + l, i);
				b = b + (bgr[0] * kernelY.at<short>(matIndex + l, 0));
				g = g + (bgr[1] * kernelY.at<short>(matIndex + l, 0));
				r = r + (bgr[2] * kernelY.at<short>(matIndex + l, 0));
			}
			Vec3s* rptr = sx.ptr<Vec3s>(j);
			rptr[i][0] = b/4;
			rptr[i][1] = g/4;
			rptr[i][2] = r/4;
		}
	}

	//// Applying 3x3 filter
	//for (int j = 1; j < src.rows - 1; j++) {
	//	for (int i = 1; i < src.cols - 1; i++) {
	//		//uchar sum = 0;
	//		char b = 0;
	//		char g = 0;
	//		char r = 0;
	//		for (int k = -1; k < 2; k++) {
	//			for (int l = -1; l < 2; l++) {
	//				int matIndex = 1;
	//				Vec3b bgr = src.at<Vec3b>(j + k, i + l);
	//				b = b + (bgr[0] * kernel.at<short>(matIndex + k, matIndex + l));
	//				g = g + (bgr[1] * kernel.at<short>(matIndex + k, matIndex + l));
	//				r = r + (bgr[2] * kernel.at<short>(matIndex + k, matIndex + l));
	//			}
	//		}
	//		Vec3s *rptr = sx.ptr<Vec3s>(j);
	//		rptr[i][0] = b;
	//		rptr[i][1] = g;
	//		rptr[i][2] = r;
	//	}
	//}

	//convertScaleAbs(sx, dst);
	dst.create(sx.size(), sx.type());
	sx.copyTo(dst);
	return 0;
}


// Implements SobelY filter
int customfilters::sobelY3x3(cv::Mat& src, cv::Mat& dst) {
	cv::Mat sy = cv::Mat(src.size(), CV_16SC3);
	//cv::Mat kernel = (Mat_<short>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	cv::Mat kernelY = (Mat_<short>(3, 1) << 1, 0, -1);
	cv::Mat kernelX = (Mat_<short>(1, 3) << 1, 2, 1);
	// Apply vertical kernel
	for (int j = 1; j < src.rows - 1; j++) {
		for (int i = 1; i < src.cols - 1; i++) {
			Vec3s bgr = sy.at<Vec3s>(j, i);
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -1; l < 2; l++) {
				int matIndex = 1;
				Vec3b bgr = src.at<Vec3b>(j + l, i);
				b = b + (bgr[0] * kernelY.at<short>(matIndex + l, 0));
				g = g + (bgr[1] * kernelY.at<short>(matIndex + l, 0));
				r = r + (bgr[2] * kernelY.at<short>(matIndex + l, 0));
			}
			Vec3s* rptr = sy.ptr<Vec3s>(j);
			rptr[i][0] = b;
			rptr[i][1] = g;
			rptr[i][2] = r;
		}
	}
	// Apply horizontal kernel
	for (int j = 0; j < sy.rows; j++) {
		for (int i = 1; i < sy.cols - 1; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			for (int l = -1; l < 2; l++) {
				int matIndex = 1;
				Vec3s bgr = sy.at<Vec3s>(j, i + l);
				b = b + (bgr[0] * kernelX.at<short>(0, matIndex + l));
				g = g + (bgr[1] * kernelX.at<short>(0, matIndex + l));
				r = r + (bgr[2] * kernelX.at<short>(0, matIndex + l));
			}
			Vec3s* rptr = sy.ptr<Vec3s>(j);
			rptr[i][0] = b/4;
			rptr[i][1] = g/4;
			rptr[i][2] = r/4;
		}
	}

	// Applying filter
	//for (int j = 1; j < src.rows - 1; j++) {
	//	for (int i = 1; i < src.cols - 1; i++) {
	//		//uchar sum = 0;
	//		char b = 0;
	//		char g = 0;
	//		char r = 0;
	//		for (int k = -1; k < 2; k++) {
	//			for (int l = -1; l < 2; l++) {
	//				int matIndex = 1;
	//				Vec3b bgr = src.at<Vec3b>(j + k, i + l);
	//				b = b + (bgr[0] * kernel.at<short>(matIndex + k, matIndex + l));
	//				g = g + (bgr[1] * kernel.at<short>(matIndex + k, matIndex + l));
	//				r = r + (bgr[2] * kernel.at<short>(matIndex + k, matIndex + l));
	//			}
	//		}
	//		Vec3b* rptr = sy.ptr<Vec3b>(j);
	//		rptr[i][0] = b;
	//		rptr[i][1] = g;
	//		rptr[i][2] = r;
	//	}
	//}
	//convertScaleAbs(sy, dst);
	dst.create(sy.size(), sy.type());
	sy.copyTo(dst);
	return 0;
}

// Implements sobel filter using OpenCVs inbuilt methods
int customfilters::sobel(cv::Mat& src, cv::Mat& dst) {
	// Remove noise by blurring with a Gaussian filter
	Mat src_blur;
	GaussianBlur(src, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// Convert the image to grayscale
	Mat src_gray;
	cvtColor(src_blur, src_gray, COLOR_BGR2GRAY);
	// Generate x and y gradient
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src_gray, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	// Calculating orientation
	// converting back to CV_8U
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	return(0);
}

// Helper method to implement magnitude filter
int customfilters::magnitudeHelper(cv::Mat& src, cv::Mat& dst) {
	cv::Mat sx, sy;
	// Finding X Gradient
	customfilters::sobelX3x3(src, sx);
	// Finding Y Gradient
	customfilters::sobelY3x3(src, sy);
	customfilters::magnitude(sx, sy, dst);
	return 0;
}

// Finds magnitude of edge using SobelX and SobelY filters
int customfilters::magnitude(cv::Mat& sx, cv::Mat& sy, cv::Mat& dst) {
	cv::Mat temp = Mat(sx.rows, sx.cols, CV_16SC3);
	// Applying Euclidean distance for magnitude to each pixel
	for (int j = 0; j < sx.rows; j++) {
		for (int i = 0; i < sx.cols; i++) {
			short b = 0;
			short g = 0;
			short r = 0;
			Vec3s sxbgr = sx.at<Vec3s>(j, i);
			Vec3s sybgr = sy.at<Vec3s>(j, i);
			b = (sxbgr[0] * sxbgr[0]) + (sybgr[0] * sybgr[0]);
			g = (sxbgr[1] * sxbgr[1]) + (sybgr[1] * sybgr[1]);
			r = (sxbgr[2] * sxbgr[2]) + (sybgr[2] * sybgr[2]);
			b = sqrt(b);
			g = sqrt(g);
			r = sqrt(r);
			Vec3s* rptr = temp.ptr<Vec3s>(j);
			rptr[i][0] = b;
			rptr[i][1] = g;
			rptr[i][2] = r;
		}
	}
	dst.create(sy.size(), sy.type());
	temp.copyTo(dst);
	return 0;
}

// Method to implement blur quantize
int customfilters::blurQuantize(cv::Mat& src, cv::Mat& dst, int levels) {
	dst = Mat(src.rows, src.cols, src.type());
	// Blur image using gaussian blur
	customfilters::blur5x5(src, dst);
	int bucket = 255 / levels;
	// Quantize image according to given levels
	for (int j = 0; j < dst.rows; j++) {
		for (int i = 0; i < dst.cols; i++) {
			Vec3b bgr = dst.at<Vec3b>(j, i);
			uchar bt = bgr[0] / bucket;
			uchar bf = bt * bucket;
			uchar gt = bgr[1] / bucket;
			uchar gf = gt * bucket;
			uchar rt = bgr[2] / bucket;
			uchar rf = rt * bucket;
			Vec3b* rptr = dst.ptr<Vec3b>(j);
			rptr[i][0] = bf;
			rptr[i][1] = gf;
			rptr[i][2] = rf;
		}
	}
	return 0;
}

// Method to implement cartoonization
int customfilters::cartoon(cv::Mat& src, cv::Mat& dst, int levels, int magThreshold) {
	cv::Mat mag, blur;
	dst = Mat(src.rows, src.cols, src.type());
	// Find magnitude for edge detection
	customfilters::magnitudeHelper(src, mag);
	// Blur and Quantize image
	customfilters::blurQuantize(src, blur, levels);
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			Vec3s* magptr = mag.ptr<Vec3s>(j);
			Vec3b* blurptr = blur.ptr<Vec3b>(j);
			// Find threshold using average of all channels
			short threshold = (magptr[i][0] + magptr[i][1] + magptr[i][2]) / 3;
			Vec3b* rptr = dst.ptr<Vec3b>(j);
			if (threshold < magThreshold) {
				rptr[i][0] = blurptr[i][0];
				rptr[i][1] = blurptr[i][1];
				rptr[i][2] = blurptr[i][2];
			}
			else {
				rptr[i][0] = 0;
				rptr[i][1] = 0;
				rptr[i][2] = 0;
			}
		}
	}
	return 0;
}


// Method to implement negative filter
int customfilters::negative(cv::Mat& src, cv::Mat& dst) {
	// Convert destination to single channel
	src.copyTo(dst);
	for (int j = 0; j < src.rows; j++) {
		Vec3b *rptr = dst.ptr<Vec3b>(j);
		for (int i = 0; i < src.cols; i++) {
			// inverting the values
			rptr[i][0] = 255 - rptr[i][0];
			rptr[i][1] = 255 - rptr[i][1];
			rptr[i][2] = 255 - rptr[i][2];
		}
	}
	return 0;
}

// Method to simulate deuteranomaly (green blindness)
int customfilters::deuteranomaly(cv::Mat& src, cv::Mat& dst) {
	src.copyTo(dst);
	for (int j = 0; j < src.rows; j++) {
		Vec3b* rptr = dst.ptr<Vec3b>(j);
		for (int i = 0; i < src.cols; i++) {
			// Roughly simulate green blindness by adding difference between R and G back to G and subtracting that difference from R
			rptr[i][2] = rptr[i][2] - (rptr[i][2] - rptr[i][1]);
			rptr[i][1] = rptr[i][1] + (rptr[i][2] - rptr[i][1]);
		}
	}
	return 0;
}


// Helper method to avoid uchar overflow
uchar rangedAdd(uchar c, int b) {
	int max = 255;
	if (b < 0) {
		if (c >= abs(b)) {
			return c + b;
		}
		else {
			return 0;
		}
	}
	if (b >= 0 && 255 - c >= b) {
		return b + c;
	}
	else {
		return 255;
	}

}

// Function to apply set brightness to the image
void customfilters::brightness(int brightness, cv::Mat& dst) {
	cv::Mat temp;
	dst.copyTo(temp);
	// Add or subtract the brightness constant from each channel of each pixel in the image.
	for (int j = 0; j < temp.rows; j++) {
		if (temp.channels() == 3) {
			// Color images
			Vec3b* rptr = temp.ptr<Vec3b>(j);
			for (int i = 0; i < temp.cols; i++) {
				rptr[i][0] = rangedAdd(rptr[i][0], brightness);
				rptr[i][1] = rangedAdd(rptr[i][1], brightness);
				rptr[i][2] = rangedAdd(rptr[i][2], brightness);
			}
		}
		else if (temp.channels() == 1) {			
			// Single channel images
			for (int i = 0; i < temp.cols; i++) {
				uchar* rptr = temp.ptr<uchar>(j, i);
				rptr[0] = rangedAdd(rptr[0], brightness);
			}
		}
	}
	temp.copyTo(dst);
}

//int customfilters::binaryThresholding(cv::Mat& src, cv::Mat& dst) {
//	cv::Mat src_gray;
//	cv::cvtColor(src, src_gray, COLOR_BGR2GRAY);
//	cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
//	cv::Mat contrastAdjustedImage;
//	try {
//		clahe->apply(src_gray, contrastAdjustedImage);
//	}
//	catch (cv::Exception& e) {
//		cerr << e.what() << endl;
//	}
//	cv::threshold(contrastAdjustedImage, dst, 125, 255, THRESH_BINARY);
//	return(0);
//}

// Function to apply Gabor Filter to an image using given theta and frequency
int customfilters::gaborFilter(cv::Mat& src, cv::Mat& dst, double theta, double freq) {
	//// Remove noise by blurring with a Gaussian filter
	//Mat src_blur;
	//GaussianBlur(src, src_blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
	// Convert the image to grayscale
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	cv::Mat kernel = cv::getGaborKernel(cv::Size(3, 3), 1, theta, freq, 0.02, 0);
	cv::Mat gaborFilteredImage;
	cv::filter2D(src_gray, gaborFilteredImage, CV_32F, kernel);
	cv::convertScaleAbs(gaborFilteredImage, dst);
	return(0);
}

