/*
  Malhar Mahant & Kruthika Gangaraju
  SP23

  The main application to start the GUI and interact with the application.
*/
#include <direct.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <windows.h>
#include <commdlg.h>
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "filters.h"
#include "objectRecognitionFunctions.h"
#include "matchfunctions.h"
#include "videoObjectDetection.h"
#pragma warning(disable : 4996)

// UI Components
cv::Rect button1;
cv::Rect button2;
cv::Rect button3;
cv::Rect button4;
cv::Rect button5;
cv::Rect button6;
cv::Rect button7;
cv::Rect button8;
cv::Mat canvas;
cv::Mat displayImage;
cv::Mat image;
cv::Mat labels = cv::Mat(), stats = cv::Mat(), centroids = cv::Mat();
cv::Moments M = cv::Moments();
cv::RotatedRect rect = cv::RotatedRect();
double theta = 0.0;
std::vector<std::vector<Point>> contours;
std::vector<Vec4i> hierarchy;
std::vector<Vec3b> colors;
std::string windowName = "Object Detection";

// Feature files
char standardDeviations[256] = "standardDeviations.csv";
char histogramVectors[256] = "histogramMatchVectors.csv";
char centerHistogramVectors[256] = "centerHistogramMatchVectors.csv";
char sobelHistogramVectors[256] = "sobelHistogramMatchVectors.csv";
//char binaryThresholdingHistogramVectors[256] = "binaryThresholdingHistogramMatchVectors.csv";
char hsvHistogramVectors[256] = "hsvHistogramMatchVectors.csv";
char gabor1HistogramVectors[256] = "gabor1HistogramMatchVectors.csv";
char gabor2HistogramVectors[256] = "gabor2HistogramMatchVectors.csv";
char gabor3HistogramVectors[256] = "gabor3HistogramMatchVectors.csv";
char gabor4HistogramVectors[256] = "gabor4HistogramMatchVectors.csv";
char gabor5HistogramVectors[256] = "gabor5HistogramMatchVectors.csv";
char gabor6HistogramVectors[256] = "gabor6HistogramMatchVectors.csv";

// Windows File Picker
OPENFILENAMEA ofn;
char filename[MAX_PATH];
char dirname[256];
char buffer[256];
FILE* fp;
DIR* dirp;
struct dirent* dp;

// Default number of matches to display
int numberOfMatches = 3;
// Number of labels in current image
int numLabels = 0;
int selectedRegion = 0;
// Current step
char key = 0;
char temp = key;

/*
* Helper method to build absolute path of file.
*/
char* buildPathByDirAndRelPath(char* dir, char* relPath) {
    char buf[256];
    strcpy(buf, dir);
    strcat(buf, "/");
    strcat(buf, relPath);
    return buf;
}

void nextSelectedRegion() {
    if (selectedRegion + 1 == numLabels) {
        selectedRegion = 0;
        return;
    }
    selectedRegion++;
}

void prevSelectedRegion() {
    if (selectedRegion == 0) {
        selectedRegion = numLabels;
        return;
    }
    selectedRegion--;
}

/*
* Function to display multiple images in an opencv window. Modified to show image file names with the images.
* Original code courtesy opencv documentation example.
*/
void ShowManyImages(std::string title, int nArgs, std::vector<char*> imagefiles, char* dir) {
    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying
    if (nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if (nArgs > 14) {
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }
    // Determine the size of the image,
    // and the number of rows/cols
    // from number of arguments
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size * w, 60 + size * h), CV_8UC3);

    //// Used to get the arguments passed
    //va_list args;
    //va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < imagefiles.size(); i++, m += (20 + size)) {
        // Get the image
        cv::Mat img = cv::imread(buildPathByDirAndRelPath(dirname, imagefiles.at(i)), cv::IMREAD_UNCHANGED);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if (img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y) ? x : y;

        // Find the scaling factor to resize the image
        scale = (float)((float)max / size);

        // Used to Align the images
        if (i % w == 0 && m != 20) {
            m = 20;
            n += 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        cv::Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
        try {
            // Adding captions to image names
            cv::Rect caption = Rect(m + ROI.height, n + ROI.width, ROI.width, ROI.height * 0.1);
            cv::putText(DispImage, imagefiles.at(i), cv::Point(m, n - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }
        catch (cv::Exception& e) {
            std::cerr << e.what() << std::endl;
        }
        cv::Mat temp;
        resize(img, temp, cv::Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }

    // Create a new window, and show the Single Big Image
    cv::namedWindow(title, 1);
    imshow(title, DispImage);
    cv::waitKey();
    cv::destroyWindow(title);
    // End the number of arguments
    /*va_end(args);*/
}

/*
* Helper method to redraw UI components
*/
void redrawUI() {
    button1 = cv::Rect(0, 0, displayImage.cols * 0.16, 50);
    button2 = cv::Rect(button1.width, 0, displayImage.cols * 0.12, 50);
    button3 = cv::Rect(button2.x + button2.width, 0, displayImage.cols * 0.12, 50);
    button4 = cv::Rect(button3.x + button3.width, 0, displayImage.cols * 0.12, 50);
    button5 = cv::Rect(button4.x + button4.width, 0, displayImage.cols * 0.12, 50);
    button6 = cv::Rect(button5.x + button5.width, 0, displayImage.cols * 0.12, 50);
    button7 = cv::Rect(button6.x + button6.width, 0, displayImage.cols * 0.12, 50);
    button8 = cv::Rect(button7.x + button7.width, 0, displayImage.cols * 0.12, 50);
    // The canvas
    if (displayImage.type() == CV_8UC1)
        canvas = cv::Mat::zeros(displayImage.rows + button1.height, displayImage.cols, CV_8UC1);
    else
        canvas = cv::Mat3b(displayImage.rows + button1.height, displayImage.cols, cv::Vec3b(0, 0, 0));

    // Draw the button
    canvas(button1) = cv::Vec3b(200, 200, 200);
    canvas(button2) = cv::Vec3b(200, 200, 200);
    canvas(button3) = cv::Vec3b(200, 200, 200);
    canvas(button4) = cv::Vec3b(200, 200, 200);
    canvas(button5) = cv::Vec3b(200, 200, 200);
    canvas(button6) = cv::Vec3b(200, 200, 200);
    canvas(button7) = cv::Vec3b(200, 200, 200);
    canvas(button8) = cv::Vec3b(200, 200, 200);
    cv::putText(canvas(button1), "Open Training File", cv::Point(button1.width * 0.25, button1.height * 0.6), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button2), "T1", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button3), "T2", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button4), "T3", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button5), "T4", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button6), "T5", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button7), "T6", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    cv::putText(canvas(button8), "T7", cv::Point(button1.width * 0.35, button1.height * 0.7), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));

    // Draw the image
    displayImage.copyTo(canvas(cv::Rect(0, button1.height, displayImage.cols, displayImage.rows)));
}

void initializeColor() {
    colors.push_back(Vec3b(0, 0, 0));
    for (int i = 1; i < 10; i++) {
        colors.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
    }
}

void generateBinarizedImage()
{
    cv::Mat blurredImage = cv::Mat(image.size(), image.type());
    customfilters::blur5x5(image, blurredImage);
    cv::Mat greyscaleImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);;
    cv::cvtColor(blurredImage, greyscaleImage, COLOR_BGR2GRAY);
    cv::Mat binarizedImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    objectrecognition::generateBinaryImage(greyscaleImage, binarizedImage, 128);
    binarizedImage.copyTo(displayImage);
}

void generateCleanBinarizedImage()
{
    cv::Mat erodedImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    cv::Mat dilatedImage = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    objectrecognition::erosion(displayImage, erodedImage, 8, 5);
    objectrecognition::dilation(erodedImage, dilatedImage, 4, 5);
    dilatedImage.copyTo(displayImage);
}

void generateSegmentedImage()
{
    cv::Mat segmentedImage = cv::Mat();
    // std::cout << "Num Labels bef " << (numLabels == 0) << std::endl;
    numLabels = objectrecognition::segmentImage(displayImage, labels, stats, centroids, colors);
    // std::cout << "Num Labels after" << (numLabels == 0) << std::endl;
    objectrecognition::selectRegion(displayImage, segmentedImage, numLabels, selectedRegion, labels, stats, colors);
    segmentedImage.copyTo(displayImage);
}

void viewAnnotatedImage(cv::Mat& target, std::vector<std::string> labels) {
    // Iterate over each contour and compute its moments
    int minArea = 1024;
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (cv::contourArea(contours[i]) > minArea) {
            M = cv::moments(contours[i]);

            // Compute the axis of least central moment
            double mu20 = M.mu20;
            double mu02 = M.mu02;
            double mu11 = M.mu11;
            theta = 0.5 * atan2(2 * mu11, mu20 - mu02);

            // Compute the oriented bounding box
            rect = cv::minAreaRect(contours[i]);
            cv::Point2f box[4];
            rect.points(box);
            for (int j = 0; j < 4; j++)
            {
                line(target, box[j], box[(j + 1) % 4], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }

            // Calculate the endpoints of the line along axis of least central moment
            cv::Point2f centroid(M.m10 / M.m00, M.m01 / M.m00);
            double cosTheta = std::cos(theta);
            double sinTheta = std::sin(theta);
            double eigenvalue = rect.size.height / 2;
            cv::Point2f endpoint1(centroid.x - cosTheta * eigenvalue, centroid.y - sinTheta * eigenvalue);
            cv::Point2f endpoint2(centroid.x + cosTheta * eigenvalue, centroid.y + sinTheta * eigenvalue);
            cv::line(target, endpoint1, endpoint2, cv::Scalar(255, 0, 0), 2);
            if (!labels.empty()) {
                std::string label = labels.at(i);
                if (!label.empty()) {
                    cv::putText(target, label, rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 150, 150), 2);
                }
            }
        }
    }
}

void generateAnnotatedImage()
{
    cv::Mat annotatedImage;
    image.copyTo(annotatedImage);
    matchfunctions::findAllContours(displayImage, contours, hierarchy);
    std::vector<std::string> labels;
    viewAnnotatedImage(annotatedImage, labels);
    annotatedImage.copyTo(displayImage);
}

void runNearestNeighbor()
{
    matchfunctions::findAllContours(displayImage, contours, hierarchy);
    std::vector<std::string> labels;
    matchfunctions::nearestNeighbor(contours, hierarchy, labels);
    cv::Mat annotatedImage;
    image.copyTo(annotatedImage);
    viewAnnotatedImage(annotatedImage, labels);
    annotatedImage.copyTo(displayImage);
}

void runKNearestNeighbors()
{
    matchfunctions::findAllContours(displayImage, contours, hierarchy);
    std::vector<std::string> labels;
    matchfunctions::kNearestNeighbor(contours, hierarchy, labels);
    cv::Mat annotatedImage;
    image.copyTo(annotatedImage);
    viewAnnotatedImage(annotatedImage, labels);
    annotatedImage.copyTo(displayImage);
}

/*
* Callback function to handle mouse clicks.
*/
void mouseClickEvent(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        // New File button clicked
        if (button1.contains(cv::Point(x, y)))
        {
            std::cout << "Clicked!" << std::endl;
            ZeroMemory(&filename, sizeof(filename));
            ZeroMemory(&ofn, sizeof(ofn));
            ofn.lStructSize = sizeof(ofn);
            ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
            ofn.lpstrFilter = "Image Files\0*.jpg;*.png;*.bmp;*.jfif\0Any File\0*.*\0";
            ofn.lpstrFile = filename;
            ofn.nMaxFile = MAX_PATH;
            ofn.lpstrTitle = "Select a File!";
            ofn.Flags = OFN_NOCHANGEDIR | OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

            if (GetOpenFileNameA(&ofn))
            {
                std::cout << "You chose the file \"" << filename << "\"\n";
                image = cv::imread(filename, cv::IMREAD_UNCHANGED);
                image.copyTo(displayImage);
                redrawUI();
            }
            else
            {
                std::cout << CommDlgExtendedError();
            }
            rectangle(canvas(button1), button1, cv::Scalar(0, 0, 255), 2);
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateAnnotatedImage();
            cv::Mat previewImage = cv::Mat(displayImage);
            char d = NULL;
            std::string label;
            while (true) {
                cv::putText(previewImage, "Enter the label for the object and hit 'Enter' or 'Esc' key. Please keep single object in image for training.", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::putText(previewImage, "Label: " + label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::imshow("Add Label", previewImage);
                d = cv::waitKey(0);
                if (d == 13) {
                    // Todo calculate and save features.
                    int minArea = 1024;
                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        if (cv::contourArea(contours[i]) > minArea) {
                            matchfunctions::generateAndSaveFeatures(label, contours[i]);
                            break;
                        }
                    }
                    cv::destroyWindow("Add Label");
                    break;
                }
                else if (d == 27) {
                    cv::destroyWindow("Add Label");
                    break;
                }
                else {
                    label += d;
                }
            }
            return;
        }

        // Task 1 button clicked
        if (button2.contains(cv::Point(x, y)) && !image.empty()) {
            generateBinarizedImage();
            key = '1';
        }
        // Task 2 button clicked
        if (button3.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            key = '2';
        }
        std::cout << "Key " << key << std::endl;
        // Task 3 button clicked
        if (button4.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateSegmentedImage();
            key = '3';
        }

        // Task 4 button clicked
        if (button5.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateAnnotatedImage();
            key = '4';
        }

        // Task 5 button clicked
        if (button6.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateAnnotatedImage();
            cv::Mat previewImage = cv::Mat(displayImage);
            char d = NULL;
            std::string label;
            while (true) {
                cv::putText(previewImage, "Enter the label for the object and hit 'Enter' or 'Esc' key. Please keep single object in image for training.", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::putText(previewImage, "Label: " + label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::imshow("Add Label", previewImage);
                d = cv::waitKey(0);
                if (d == 13) {
                    // Todo calculate and save features.
                    int minArea = 1024;
                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        if (cv::contourArea(contours[i]) > minArea) {
                            matchfunctions::generateAndSaveFeatures(label, contours[i]);
                            break;
                        }
                    }
                    cv::destroyWindow("Add Label");
                    break;
                }
                else if (d == 27) {
                    cv::destroyWindow("Add Label");
                    break;
                }
                else {
                    label += d;
                }
            }
        }

        // Task 6 button clicked
        if (button7.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            runNearestNeighbor();
            key = '6';
        }

        // Task 7 button clicked
        if (button8.contains(cv::Point(x, y))) {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            runKNearestNeighbors();
            key = '7';
        }
        temp = key;
        redrawUI();
    }
    // Button released
    if (event == cv::EVENT_LBUTTONUP)
    {
        rectangle(canvas, button1, cv::Scalar(200, 200, 200), 2);
    }

    cv::imshow(windowName, canvas);
}


/*
* Main method to start application.
*/
int main(int argc, char* argv[]) {

    cv::VideoCapture* capdev;

    // open the video device 2 (Phone camera as webcam)
    try {
        capdev = new cv::VideoCapture(1);
    }
    catch (Exception e) {
        std::cerr << e.what() << std::endl;
        capdev = new cv::VideoCapture(0);
    }
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }

    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    key = 0;
    initializeColor();
    // check for sufficient arguments
    //if (argc < 2) {
    //    printf("usage: %s <directory path>\n", argv[0]);
    //    exit(-1);
    //}

    // get the directory path
    /*strcpy(dirname, argv[1]);
    printf("Processing directory %s\n", dirname);*/

    // open the directory
    /*dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }*/

    // Call helper method to generate features
    //generateFeatures();
    
    int saveCount = 1;
    cv::VideoWriter videoWriter = cv::VideoWriter();
    // Create opencv window
    cv::namedWindow(windowName, 1);

    // Set up mouse callback function
    cv::setMouseCallback(windowName, mouseClickEvent);

    // Define blank image
    displayImage = cv::Mat3b(300, 600, cv::Vec3b(0, 0, 0));

    while (cv::getWindowProperty(windowName, 0) >= 0) {
        *capdev >> image; // get a new frame from the camera, treat as a stream
        if (image.empty()) {
            printf("frame is empty\n");
            break;
        }
        if (key == 0)
            image.copyTo(displayImage);
        if (key == '1') {
            generateBinarizedImage();
        }

        if (key == '2') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
        }

        if (key == '3') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateSegmentedImage();
        }

        if (key == '4') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateAnnotatedImage();
        }

        if (key == '6') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            runNearestNeighbor();
        }

        if (key == '7') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            runKNearestNeighbors();
        }

        // Video capturing logic
        if (videoWriter.isOpened()) {
            videoWriter.write(displayImage);
        }

        redrawUI();
        cv::imshow(windowName, canvas);
        temp = key;
        key = cv::waitKey(10);
        
        // Implements toggle functionality
        if (key == temp) {
            key = 0;
        }

        // Start training mode
        if (key == 'n') {
            generateBinarizedImage();
            generateCleanBinarizedImage();
            generateAnnotatedImage();
            cv::Mat previewImage = cv::Mat(displayImage);
            char d = NULL;
            std::string label;
            while (true) {
                cv::putText(previewImage, "Enter the label for the object and hit 'Enter' or 'Esc' key. Please keep single object in image for training.", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::putText(previewImage, "Label: " + label, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 50, 150), 2);
                cv::imshow("Add Label", previewImage);
                d = cv::waitKey(0);
                if (d == 13) {
                    // Todo calculate and save features.
                    int minArea = 1024;
                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        if (cv::contourArea(contours[i]) > minArea) {
                            matchfunctions::generateAndSaveFeatures(label, contours[i]);
                            break;
                        }
                    }
                    cv::destroyWindow("Add Label");
                    break;
                }
                else if (d == 27) {
                    cv::destroyWindow("Add Label");
                    break;
                }
                else {
                    label += d;
                }
            }
        }

        if (key == 'o') {
            nextSelectedRegion();
            /*cv::Mat segmentedImage, labels, stats, centroids;
            objectrecognition::selectRegion(cleanBinarizedImage, segmentedImage, numLabels, selectedRegion, labels, stats, colors);
            segmentedImage.copyTo(displayImage);*/
            redrawUI();
        }

        if (key == 'p') {
            prevSelectedRegion();
            /*cv::Mat segmentedImage, labels, stats, centroids;
            objectrecognition::selectRegion(cleanBinarizedImage, segmentedImage, numLabels, selectedRegion, labels, stats, colors);
            segmentedImage.copyTo(displayImage);*/
            redrawUI();
        }

        if (key == -1) {
            key = temp;
        }

        if (key == 'n') {
            key = '4';
        }

        // Start video capture
        if (key == 'v') {
            key = temp;
            if (!videoWriter.isOpened()) {
                std::string fileLocation = "Capture_" + std::to_string(saveCount);
                fileLocation = fileLocation + ".mp4";
                bool isColor = key != '1' && key != '2';
                videoWriter = cv::VideoWriter(fileLocation, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 30, Size(refS.width, refS.height), isColor);
                // videoWriter.open()
                saveCount++;
                std::cout << "Start capturing video: " << fileLocation << std::endl;
            }
            else {
                std::cout << "Ending video capture" << std::endl;
                videoWriter.release();
            }
        }

        // Save image
        if (key == 's') {
            key = temp;
            std::string fileLocation = "Capture_" + std::to_string(saveCount);
            fileLocation = fileLocation + ".jpg";
            bool saved = cv::imwrite(fileLocation, displayImage);
            if (saved) {
                cv::namedWindow("Preview Saved File", 1);
                Mat image = imread(fileLocation,
                    IMREAD_UNCHANGED);

                // Error handling
                if (image.empty()) {
                    std::cout << "Image File Not Found" << std::endl;
                    cv::destroyWindow("Save File");
                }
                cv::imshow("Preview Saved File", displayImage);
                saveCount++;
                cv::waitKey(0);
                cv::destroyWindow("Preview Saved File");
            }
        }
        
        // Quit
        if (key == 'q') {
            break;
        }

        // Avoid removal of filter by other key inputs
        if (key != 0 && key != '1' && key != '2' && key != '3' && key != '4' && key != '6' && key != '7' /*&& key != 'c' && key != 'n' && key != 'd'*/) {
            key = temp;
        }
    }
    printf("Terminating\n");
    return(0);
}