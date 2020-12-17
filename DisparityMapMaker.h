#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/ximgproc.hpp>
using namespace cv;

class DisparityMapMaker
{
private:
	Ptr<StereoBM> left_matcher;
	Ptr<StereoMatcher> right_matcher;
	Ptr<ximgproc::DisparityWLSFilter> wlsFilter;
	//Ptr<cuda::DisparityBilateralFilter> bilateralFilter;
	//cuda::GpuMat gpuleft, gpuright, left_gpudisp, right_gpudisp;
	Mat left_disp, right_disp, filtered;
	Mat outputDisparity;
public:
	/*
	@param numDisparities the disparity search range. For each pixel algorithm
		will find the best disparity from 0 (default minimum disparity) to numDisparities.
		The search range can then be shifted by changing the minimum disparity.
	@param blockSizehe linear size of the blocks compared by the algorithm.
		The size should be odd (as the block is centered at the current pixel).
	@param filterRadius radius of belief filter
	@param filterIter number of iteration for belief filter
	@param lower_disp lower value of disparity (lower values will have this value)
	@param upper_disp upper value of disparity (upper values will have this value)
	*/
	DisparityMapMaker(int numDisparities, int blockSize);
	/*
	Computer a disparite map for leand and right images. The result stored in the obj.
	*/
	void compute(Mat& left, Mat& right);
	/* Get stored disparity map*/
	Mat getDisparityMap();
};