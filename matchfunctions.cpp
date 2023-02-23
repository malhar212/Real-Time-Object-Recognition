/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  Implements functions to generate different kinds of features from an image and find similar images using generated features.
*/
#include <cstring>
#include <cmath>
#include <dirent.h>
#include "csv_util.h"
#include "filters.h"
#include "matchfunctions.h"

#pragma warning(disable : 4996)

// Feature files
char standardDeviationsFile[256] = "standardDeviations.csv";
char labelToFeaturesFile[256] = "labelToFeatures.csv";

/*
* Helper method to check if a file exists.
*/
bool checkIfFileExists(char* fpath) {
	FILE* fp;
	fp = fopen(fpath, "r");
	if (!fp) {
		return false;
	}
	if (fp) {
		fclose(fp);
		return true;
	}
}

int matchfunctions::calculateHuMoments(cv::Mat& src, cv::Mat& target, std::vector<float>& features)
{
	// Calculate Moments 
	// cv::Moments moments = cv::moments(src, false);
	
	return 0;
}

float calculateScaledEuclideanDistance(std::vector<float>& featuresA, std::vector<float>& featuresB) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> stdDevMeanData;
	read_image_data_csv(standardDeviationsFile, labels, stdDevMeanData, 0);
	std::vector<float> stdDevMeanDataVector = stdDevMeanData.at(0);
	float scaledEuclideanDistance = 0;
	for (int i = 0; i < featuresA.size(); i++) {
		float error = (featuresA.at(i) - featuresB.at(i))/ stdDevMeanDataVector.at(i * 2);
		scaledEuclideanDistance = scaledEuclideanDistance + (error * error);
	}
	return scaledEuclideanDistance;
}

int matchfunctions::findAllContours(cv::Mat& src, std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy) {
 
    // Find contours in the binary image
    cv::findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	return 0;
}

int matchfunctions::generateFeatures(std::vector<Point>& contour, std::vector<float>& features) {
	cv::Moments moments = cv::moments(contour);
	// Calculate Hu Moments 
	double huMoments[7];
	/*
	* The second moment: This is the variance of the image, which measures the spread or distribution of the gray-level intensity values around the centroid. The second moment provides information about the size and shape of the object in the image.

	The third moment: This measures the skewness of the image, which describes the asymmetry of the distribution of the gray-level intensity values around the centroid. The third moment provides information about the orientation of the object in the image.
	*/
	features.push_back(moments.nu30);
	features.push_back(moments.nu03);
	cv::HuMoments(moments, huMoments);
	// Log scale hu moments 
	for (int i = 0; i < 7; i++) {
		huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
		features.push_back(huMoments[i]);
	}
	return 0;
}

int matchfunctions::generateAndSaveFeatures(std::string label, std::vector<Point>& contour) {
	std::vector<float> features;
	matchfunctions::generateFeatures(contour, features);
	char* labelChar = new char[256];
	strcpy(labelChar, label.c_str());
	if (!checkIfFileExists(standardDeviationsFile)) {
		std::vector<float> stdDevMeanFeatures;
		stdDevMeanFeatures.resize((features.size() * 2) + 1, 0);
		append_image_data_csv(standardDeviationsFile, labelChar, stdDevMeanFeatures, 1);
	}
	std::vector<char*> labels;
	std::vector<std::vector<float>> stdDevMeanData;
	read_image_data_csv(standardDeviationsFile, labels, stdDevMeanData, 0);
	std::vector<float> stdDevMeanDataVector = stdDevMeanData.at(0);
	// Total number of features
	float n = stdDevMeanDataVector.at(stdDevMeanDataVector.size() - 1);
	for (int i = 0; i < features.size(); i++) {
		// Calculate new mean
		float oldMean = stdDevMeanDataVector.at((i * 2)+ 1);
		float newMean = (n * oldMean + features.at(i)) / (n + 1);
		// Calculate new std dev
		float oldStd = stdDevMeanDataVector.at(i * 2);
		float variation = (features.at(i) - newMean);
		float newStd = sqrtf(((n * oldStd * oldStd) + variation * variation) / (n + 1));
		stdDevMeanDataVector.at(i * 2) = newStd;
		stdDevMeanDataVector.at((i * 2) + 1) = newMean;
	}
	stdDevMeanDataVector.at(stdDevMeanDataVector.size() - 1) = n + 1;
	append_image_data_csv(standardDeviationsFile, labelChar, stdDevMeanDataVector, 1);
	append_image_data_csv(labelToFeaturesFile, labelChar, features, 0);
}

int matchfunctions::nearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> featuresData;
	outLabels.resize(contours.size(), "");
	// Read features from CSV
	if (read_image_data_csv(labelToFeaturesFile, labels, featuresData, 0) >= 0) {
		int minArea = 1024;
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > minArea) {
				std::vector<float> features;
				matchfunctions::generateFeatures(contours[i], features);
				// Map of filenames to errors
				std::multimap<float, char*> map;
				for (int j = 0; j < labels.size(); j++) {
					// Calculate distance for each image
					std::vector<float> labelFeatures = featuresData.at(j);
					float error = calculateScaledEuclideanDistance(features, labelFeatures);
					map.insert({ error, labels.at(j) });
				}
				outLabels.at(i) = map.begin()->second;
			}
		}	
		return(0);
	}
	return -1;
}

int matchfunctions::kNearestNeighbor(std::vector<std::vector<Point>>& contours, std::vector<Vec4i>& hierarchy, std::vector<std::string>& outLabels) {
	std::vector<char*> labels;
	std::vector<std::vector<float>> featuresData;
	outLabels.resize(contours.size(), "");
	// Read features from CSV
	if (read_image_data_csv(labelToFeaturesFile, labels, featuresData, 0) >= 0) {
		int minArea = 1024;
		for (size_t i = 0; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > minArea) {
				std::vector<float> features;
				matchfunctions::generateFeatures(contours[i], features);
				// Map of filenames to errors
				std::map<std::string, std::vector<float>> map;
				for (int j = 0; j < labels.size(); j++) {
					// Calculate distance for each image
					std::vector<float> labelFeatures = featuresData.at(j);
					float error = calculateScaledEuclideanDistance(features, labelFeatures);
					std::string label = labels.at(j);
					std::vector<float> distances;
					if (map.find(label) != map.end()) {
						distances = map.at(label);
					}
					distances.push_back(error);
					map.insert({ labels.at(j), distances });
				}

				// For each label find the sum of distance to k-nearest neighbors
				std::multimap<float, std::string> distanceToClassMap;
				std::map<char*, std::vector<float>>::iterator itr;
				for (auto itr = map.begin(); itr != map.end(); ++itr) {
					std::string label = itr->first;
					std::vector<float> distances = itr->second;
					sort(distances.begin(), distances.end());
					float distanceToClass = 0;
					int j = 0;
					for (auto it = distances.begin(); j < 5 && it != distances.end(); it++) {
						distanceToClass += *it;
						j++;
					}
					distanceToClassMap.insert({ distanceToClass, label });
				}

				// multimap are sorted by minimum key 
				outLabels.at(i) = distanceToClassMap.begin()->second;
			}
		}
		return(0);
	}
	return -1;
}