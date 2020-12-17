#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "reconstruct.h"
#include "depthtest.h"

int main()
{
	//CustomDepthTest();
	//RealsenseDepthTest();
	//kinFuExample();
	//updatedDepthKinFuExample();
	customDepthMapKinFuExample();
	return 0;
}