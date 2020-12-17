#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/rgbd/kinfu.hpp>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"         // Include short list of convenience functions for rendering

#include <thread>
#include <queue>
#include <atomic>
#include <string>
#include <fstream>
#include <vector>

#include "StereoImagePreprocessor.h"
#include "DisparityMapMaker.h"

using namespace cv;

Ptr<kinfu::Params> createRealsenseParams();

// Assigns an RGB value for each point in the pointcloud, based on the depth value
void colorize_pointcloud(const Mat points, Mat& color);

// Handles all the OpenGL calls needed to display the point cloud
void draw_kinfu_pointcloud(glfw_state& app_state, Mat points, Mat normals);

void export_to_ply(Mat points, Mat normals);


int kinFuExample();

int customDepthMapKinFuExample();

int updatedDepthKinFuExample();