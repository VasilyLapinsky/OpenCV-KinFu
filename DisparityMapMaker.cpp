#include "DisparityMapMaker.h"

DisparityMapMaker::DisparityMapMaker(int numDisparities, int blockSize)
{
	left_matcher = StereoBM::create(numDisparities, blockSize);
	right_matcher = ximgproc::createRightMatcher(left_matcher);
	wlsFilter = ximgproc::createDisparityWLSFilter(left_matcher);

	//left_matcher = cuda::createStereoBM(numDisparities, blockSize);
	//bilateralFilter = cuda::createDisparityBilateralFilter(numDisparities, 15, 3);
}

void DisparityMapMaker::compute(Mat& left, Mat& right)
{
	//gpuleft.upload(left);
	// compute disparity
	left_matcher->compute(left, right, left_disp);
	right_matcher->compute(right, left, right_disp);
	// download disparity
	//left_gpudisp.download(left_disp);
	// apply filter
	Mat left_disp_float, right_disp_float; 
	left_disp.convertTo(left_disp_float, CV_32FC1, 1 / 16.0);
	right_disp.convertTo(right_disp_float, CV_32FC1, 1 / 16.0);

	wlsFilter->filter(left_disp_float, left, filtered, right_disp_float);
	//cuda::GpuMat gpufiltered;
	//left_gpudisp.upload(filtered);
	//bilateralFilter->apply(left_gpudisp, gpuleft, gpufiltered);
	//gpufiltered.download(filtered);
}

Mat DisparityMapMaker::getDisparityMap()
{
	return filtered;
}