#include "depthtest.h"


class QueuedMat {
public:

	Mat img; // Standard cv::Mat

	QueuedMat() {}; // Default constructor

	// Destructor (called by queue::pop)
	~QueuedMat() {
		img.release();
	};

	// Copy constructor (called by queue::push)
	QueuedMat(const QueuedMat& src) {
		src.img.copyTo(img);
	};
};


void make_depth_histogram(const Mat &depth, Mat &normalized_depth, int coloringMethod) {
	normalized_depth = Mat(depth.size(), CV_8U);
	int width = depth.cols, height = depth.rows;

	static uint32_t histogram[0x10000];
	memset(histogram, 0, sizeof(histogram));

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			++histogram[depth.at<ushort>(i, j)];
		}
	}

	for (int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i - 1]; // Build a cumulative histogram for the indices in [1,0xFFFF]

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (uint16_t d = depth.at<ushort>(i, j)) {
				int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
				normalized_depth.at<uchar>(i, j) = static_cast<uchar>(f);
			}
			else {
				normalized_depth.at<uchar>(i, j) = 0;
			}
		}
	}

	// Apply the colormap:
	applyColorMap(normalized_depth, normalized_depth, coloringMethod);
}

int CustomDepthTest()
{

	try {

		//Create a depth cleaner instance
		rgbd::DepthCleaner* depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

		// A librealsense class for mapping raw depth into RGB (pretty visuals, yay)
		rs2::colorizer color_map;

		// Create a pipeline and configure it
		rs2::pipeline p;
		rs2::config cfg;
		float depth_scale = 1;
		cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
		cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
		auto profile = p.start(cfg);
		DisparityMapMaker disparityMapMaker(32, 21);
		float f = 383.748443603516f, Tx = 50.11365562f, diff = (319.515716552734 - 319.515716552734);
		Mat depth;
		auto w = 640;
		auto h = 480;

		// openCV window
		const auto window_name_source = "Source Depth";
		namedWindow(window_name_source, WINDOW_AUTOSIZE);

		const auto window_name_filter = "Filtered Depth";
		namedWindow(window_name_filter, WINDOW_AUTOSIZE);

		// Atomic boolean to allow thread safe way to stop the thread
		std::atomic_bool stopped(false);

		// Declaring two concurrent queues that will be used to push and pop frames from different threads
		std::queue<QueuedMat> filteredQueue;
		std::queue<QueuedMat> originalQueue;

		// The threaded processing thread function
		std::thread processing_thread([&]() {
			while (!stopped) {

				rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera

				//rs2::frame rsleft = data.get_infrared_frame(1);
				//rs2::frame rsright = data.get_infrared_frame(2);
				Mat left(Size(w, h), CV_8UC1, (void*)data.get_infrared_frame(1).get_data(), Mat::AUTO_STEP);
				Mat right(Size(w, h), CV_8UC1, (void*)data.get_infrared_frame(2).get_data(), Mat::AUTO_STEP);

				disparityMapMaker.compute(left, right);
				Mat disparity = disparityMapMaker.getDisparityMap();
				//imshow("Disparity", disparity);
				depth = Mat(disparity.rows, disparity.cols, CV_32F);
				// Calculate depth
				depth.forEach<float>([f, Tx, diff, disparity](float &p, const int * position) -> void {
					p = (-f * Tx) / (float(diff - disparity.at<float>(position[0], position[1])));
				});

				//Create queued mat containers
				QueuedMat depthQueueMat;
				QueuedMat cleanDepthQueueMat;

				// Create an openCV matrix from the raw depth (CV_16U holds a matrix of 16bit unsigned ints)
				Mat rawDepthMat;
				depth.convertTo(rawDepthMat, CV_16U);
				// Create an openCV matrix for the DepthCleaner instance to write the output to
				Mat cleanedDepth(Size(w, h), CV_16U);

				//Run the RGBD depth cleaner instance
				depthc->operator()(rawDepthMat, cleanedDepth);

				const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
				Mat temp, temp2;

				// Downsize for performance, use a smaller version of depth image (defined in the SCALE_FACTOR macro)
				Mat small_depthf;
				resize(cleanedDepth, small_depthf, Size(), SCALE_FACTOR, SCALE_FACTOR);

				// Inpaint only the masked "unknown" pixels
				inpaint(small_depthf, (small_depthf == noDepth), temp, 5.0, INPAINT_TELEA);

				// Upscale to original size and replace inpainted regions in original depth image
				resize(temp, temp2, cleanedDepth.size());
				temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal

				// Use the copy constructor to copy the cleaned mat if the isDepthCleaning is true
				cleanDepthQueueMat.img = cleanedDepth;

				//Use the copy constructor to fill the original depth coming in from the sensr(i.e visualized in RGB 8bit ints)
				depthQueueMat.img = rawDepthMat;

				//Push the mats to the queue
				originalQueue.push(depthQueueMat);
				filteredQueue.push(cleanDepthQueueMat);
			}
		});

		Mat filteredDequeuedMat(Size(1280, 720), CV_8UC1);
		Mat originalDequeuedMat(Size(1280, 720), CV_8UC1);

		//Main thread function
		while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source) && cvGetWindowHandle(window_name_filter)) {

			//If the frame queue is not empty pull a frame out and clean the queue
			while (!originalQueue.empty()) {
				originalQueue.front().img.copyTo(originalDequeuedMat);
				originalQueue.pop();
			}


			while (!filteredQueue.empty()) {
				filteredQueue.front().img.copyTo(filteredDequeuedMat);
				filteredQueue.pop();
			}

			Mat coloredCleanedDepth;
			Mat coloredOriginalDepth;

			make_depth_histogram(filteredDequeuedMat, coloredCleanedDepth, COLORMAP_JET);
			make_depth_histogram(originalDequeuedMat, coloredOriginalDepth, COLORMAP_JET);

			imshow(window_name_filter, coloredCleanedDepth);
			imshow(window_name_source, coloredOriginalDepth);
		}

		// Signal the processing thread to stop, and join
		stopped = true;
		processing_thread.join();

		return EXIT_SUCCESS;
	}
	catch (const rs2::error & e) {
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}

int RealsenseDepthTest()
{
	try {

		//Create a depth cleaner instance
		rgbd::DepthCleaner* depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

		// A librealsense class for mapping raw depth into RGB (pretty visuals, yay)
		rs2::colorizer color_map;

		// Declare RealSense pipeline, encapsulating the actual device and sensors
		rs2::pipeline pipe;

		//Create a configuration for configuring the pipeline with a non default profile
		rs2::config cfg;

		//Add desired streams to configuration
		cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

		// Start streaming with default recommended configuration
		pipe.start(cfg);

		// openCV window
		const auto window_name_source = "Source Depth";
		namedWindow(window_name_source, WINDOW_AUTOSIZE);

		const auto window_name_filter = "Filtered Depth";
		namedWindow(window_name_filter, WINDOW_AUTOSIZE);

		// Atomic boolean to allow thread safe way to stop the thread
		std::atomic_bool stopped(false);

		// Declaring two concurrent queues that will be used to push and pop frames from different threads
		std::queue<QueuedMat> filteredQueue;
		std::queue<QueuedMat> originalQueue;

		// The threaded processing thread function
		std::thread processing_thread([&]() {
			while (!stopped) {

				rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
				rs2::frame depth_frame = data.get_depth_frame(); //Take the depth frame from the frameset
				if (!depth_frame) // Should not happen but if the pipeline is configured differently
					return;       //  it might not provide depth and we don't want to crash

				//Save a reference
				rs2::frame filtered = depth_frame;

				// Query frame size (width and height)
				const int w = depth_frame.as<rs2::video_frame>().get_width();
				const int h = depth_frame.as<rs2::video_frame>().get_height();

				//Create queued mat containers
				QueuedMat depthQueueMat;
				QueuedMat cleanDepthQueueMat;

				// Create an openCV matrix from the raw depth (CV_16U holds a matrix of 16bit unsigned ints)
				Mat rawDepthMat(Size(w, h), CV_16U, (void*)depth_frame.get_data());

				// Create an openCV matrix for the DepthCleaner instance to write the output to
				Mat cleanedDepth(Size(w, h), CV_16U);

				//Run the RGBD depth cleaner instance
				depthc->operator()(rawDepthMat, cleanedDepth);

				const unsigned char noDepth = 0; // change to 255, if values no depth uses max value
				Mat temp, temp2;

				// Downsize for performance, use a smaller version of depth image (defined in the SCALE_FACTOR macro)
				Mat small_depthf;
				resize(cleanedDepth, small_depthf, Size(), SCALE_FACTOR, SCALE_FACTOR);

				// Inpaint only the masked "unknown" pixels
				inpaint(small_depthf, (small_depthf == noDepth), temp, 5.0, INPAINT_TELEA);

				// Upscale to original size and replace inpainted regions in original depth image
				resize(temp, temp2, cleanedDepth.size());
				temp2.copyTo(cleanedDepth, (cleanedDepth == noDepth));  // add to the original signal

				// Use the copy constructor to copy the cleaned mat if the isDepthCleaning is true
				cleanDepthQueueMat.img = cleanedDepth;

				//Use the copy constructor to fill the original depth coming in from the sensr(i.e visualized in RGB 8bit ints)
				depthQueueMat.img = rawDepthMat;

				//Push the mats to the queue
				originalQueue.push(depthQueueMat);
				filteredQueue.push(cleanDepthQueueMat);
			}
		});

		Mat filteredDequeuedMat(Size(1280, 720), CV_16UC1);
		Mat originalDequeuedMat(Size(1280, 720), CV_8UC3);

		//Main thread function
		while (waitKey(1) < 0 && cvGetWindowHandle(window_name_source) && cvGetWindowHandle(window_name_filter)) {

			//If the frame queue is not empty pull a frame out and clean the queue
			while (!originalQueue.empty()) {
				originalQueue.front().img.copyTo(originalDequeuedMat);
				originalQueue.pop();
			}


			while (!filteredQueue.empty()) {
				filteredQueue.front().img.copyTo(filteredDequeuedMat);
				filteredQueue.pop();
			}

			Mat coloredCleanedDepth;
			Mat coloredOriginalDepth;

			make_depth_histogram(filteredDequeuedMat, coloredCleanedDepth, COLORMAP_JET);
			make_depth_histogram(originalDequeuedMat, coloredOriginalDepth, COLORMAP_JET);

			imshow(window_name_filter, coloredCleanedDepth);
			imshow(window_name_source, coloredOriginalDepth);
		}

		// Signal the processing thread to stop, and join
		stopped = true;
		processing_thread.join();

		return EXIT_SUCCESS;
	}
	catch (const rs2::error & e) {
		std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}