// Libraries
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include <opencv2/rgbd.hpp>     // OpenCV RGBD Contrib package
#include <opencv2/highgui/highgui_c.h> // OpenCV High-level GUI

// STD
#include <string>
#include <thread>
#include <atomic>
#include <queue>

// disparity
#include "DisparityMapMaker.h"

using namespace cv;


#define SCALE_FACTOR 1

int CustomDepthTest();

int RealsenseDepthTest();