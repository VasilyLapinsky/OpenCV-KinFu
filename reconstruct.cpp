#pragma once
#include "reconstruct.h"

static float max_dist = 2.5;
static float min_dist = 0.11;

// Parameters
Ptr<kinfu::Params> createRealsenseParams()
{
	kinfu::Params p;


	p.frameSize = Size(640, 480);

	p.volumeType = kinfu::VolumeType::TSDF;

	float fx, fy, cx, cy;
	fx = 383.748443603516f;
	fy = 383.748443603516f;
	cx = 319.515716552734f;
	cy = 239.890762329102f;
	p.intr = Matx33f(fx, 0, cx,
		0, fy, cy,
		0, 0, 1);

	// 5000 for the 16-bit PNG files
	// 1 for the 32-bit float images in the ROS bag files
	p.depthFactor = 1;

	// sigma_depth is scaled by depthFactor when calling bilateral filter
	p.bilateral_sigma_depth = 0.04f;  //meter
	p.bilateral_sigma_spatial = 4.5; //pixels
	p.bilateral_kernel_size = 7;     //pixels

	p.icpAngleThresh = (float)(30. * CV_PI / 180.); // radians
	p.icpDistThresh = 0.1f; // meters

	p.icpIterations = { 10, 5, 4 };
	p.pyramidLevels = (int)p.icpIterations.size();

	p.tsdf_min_camera_movement = 0.f; //meters, disabled

	p.volumeDims = Vec3i::all(512); //number of voxels

	float volSize = 3.f;
	p.voxelSize = volSize / 512.f; //meters

	// default pose of volume cube
	p.volumePose = Affine3f().translate(Vec3f(-volSize / 2.f, -volSize / 2.f, 0.5f));
	p.tsdf_trunc_dist = 7 * p.voxelSize; // about 0.04f in meters
	p.tsdf_max_weight = 64;   //frames

	p.raycast_step_factor = 0.25f;  //in voxel sizes
	// gradient delta factor is fixed at 1.0f and is not used
	//p.gradient_delta_factor = 0.5f; //in voxel sizes

	//p.lightPose = p.volume_pose.translation()/4; //meters
	p.lightPose = Vec3f::all(0.f); //meters

	// depth truncation is not used by default but can be useful in some scenes
	p.truncateThreshold = 0.f; //meters

	return makePtr<kinfu::Params>(p);
}

// Thread-safe queue for OpenCV's Mat objects
class mat_queue
{
public:
	void push(Mat& item)
	{
		std::lock_guard<std::mutex> lock(_mtx);
		queue.push(item);
	}
	int try_get_next_item(Mat& item)
	{
		std::lock_guard<std::mutex> lock(_mtx);
		if (queue.empty())
			return false;
		item = std::move(queue.front());
		queue.pop();
		return true;
	}
private:
	std::queue<Mat> queue;
	std::mutex _mtx;
};

// Assigns an RGB value for each point in the pointcloud, based on the depth value
void colorize_pointcloud(const Mat points, Mat& color)
{
	// Define a vector of 3 Mat arrays which will hold the channles of points
	std::vector<Mat> channels(points.channels());
	split(points, channels);
	// Get the depth channel which we'll use to colorize the pointcloud
	color = channels[2];

	// Convert the depth matrix to unsigned char values
	float min = min_dist;
	float max = max_dist;
	color.convertTo(color, CV_8UC1, 255 / (max - min), -255 * min / (max - min));
	// Get an rgb value for each point
	applyColorMap(color, color, COLORMAP_JET);
}

// Handles all the OpenGL calls needed to display the point cloud
void draw_kinfu_pointcloud(glfw_state& app_state, Mat points, Mat normals)
{
	// Define new matrix which will later hold the coloring of the pointcloud
	Mat color;
	colorize_pointcloud(points, color);

	// OpenGL commands that prep screen for the pointcloud
	glLoadIdentity();
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
	glClear(GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	gluPerspective(65, 1.3, 0.01f, 10.0f);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

	glTranslatef(0, 0, 1 + app_state.offset_y*0.05f);
	glRotated(app_state.pitch - 20, 1, 0, 0);
	glRotated(app_state.yaw + 5, 0, 1, 0);
	glTranslatef(0, 0, -0.5f);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glBegin(GL_POINTS);
	// this segment actually prints the pointcloud
	for (int i = 0; i < points.rows; i++)
	{
		// Get point coordinates from 'points' matrix
		float x = points.at<float>(i, 0);
		float y = points.at<float>(i, 1);
		float z = points.at<float>(i, 2);

		// Get point coordinates from 'normals' matrix
		float nx = normals.at<float>(i, 0);
		float ny = normals.at<float>(i, 1);
		float nz = normals.at<float>(i, 2);

		// Get the r, g, b values for the current point
		uchar r = color.at<uchar>(i, 0);
		uchar g = color.at<uchar>(i, 1);
		uchar b = color.at<uchar>(i, 2);

		// Register color and coordinates of the current point
		glColor3ub(r, g, b);
		glNormal3f(nx, ny, nz);
		glVertex3f(x, y, z);
	}
	// OpenGL cleanup
	glEnd();
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}


void export_to_ply(Mat points, Mat normals)
{
	// First generate a filename
	//const size_t buffer_size = 50;
	//char fname[buffer_size];
	time_t t = time(0);   // get time now
	//struct tm * now = localtime(&t);
	//strftime(fname, buffer_size, "%m%d%y %H%M%S.ply", now);
	std::string fname = "data/pointcloud" + std::to_string(t) + ".ply";
	std::cout << "exporting to" << fname << std::endl;

	// Get rgb values for points
	Mat color;
	colorize_pointcloud(points, color);

	// Write the ply file
	std::ofstream out(fname);
	out << "ply\n";
	out << "format binary_little_endian 1.0\n";
	out << "comment pointcloud saved from Realsense Viewer\n";
	out << "element vertex " << points.rows << "\n";
	out << "property float" << sizeof(float) * 8 << " x\n";
	out << "property float" << sizeof(float) * 8 << " y\n";
	out << "property float" << sizeof(float) * 8 << " z\n";

	out << "property float" << sizeof(float) * 8 << " nx\n";
	out << "property float" << sizeof(float) * 8 << " ny\n";
	out << "property float" << sizeof(float) * 8 << " nz\n";

	out << "property uchar red\n";
	out << "property uchar green\n";
	out << "property uchar blue\n";
	out << "end_header\n";
	out.close();

	out.open(fname, std::ios_base::app | std::ios_base::binary);
	for (int i = 0; i < points.rows; i++)
	{
		// write vertices
		out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 0))), sizeof(float));
		out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 1))), sizeof(float));
		out.write(reinterpret_cast<const char*>(&(points.at<float>(i, 2))), sizeof(float));

		// write normals
		out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 0))), sizeof(float));
		out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 1))), sizeof(float));
		out.write(reinterpret_cast<const char*>(&(normals.at<float>(i, 2))), sizeof(float));

		// write colors
		out.write(reinterpret_cast<const char*>(&(color.at<uchar>(i, 0))), sizeof(uint8_t));
		out.write(reinterpret_cast<const char*>(&(color.at<uchar>(i, 1))), sizeof(uint8_t));
		out.write(reinterpret_cast<const char*>(&(color.at<uchar>(i, 2))), sizeof(uint8_t));
	}
}


int kinFuExample()
{
	setUseOptimized(true);
	// Declare KinFu and params pointers
	Ptr<kinfu::KinFu> kf;
	Ptr<kinfu::Params> params = createRealsenseParams();

	// Create a pipeline and configure it
	rs2::pipeline p;
	rs2::config cfg;
	float depth_scale;
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
	auto profile = p.start(cfg);
	auto dev = profile.get_device();
	auto stream_depth = profile.get_stream(RS2_STREAM_DEPTH);

	// Get a new frame from the camera
	rs2::frameset data = p.wait_for_frames();
	auto d = data.get_depth_frame();

	for (rs2::sensor& sensor : dev.query_sensors())
	{
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			// Set some presets for better results
			dpt.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
			// Depth scale is needed for the kinfu set-up
			depth_scale = dpt.get_depth_scale();
			break;
		}
	}
	cout << depth_scale << endl;
	
	// Declare post-processing filters for better results
	auto decimation = rs2::decimation_filter();
	auto spatial = rs2::spatial_filter();
	auto temporal = rs2::temporal_filter();

	auto clipping_dist = max_dist / depth_scale; // convert clipping_dist to raw depth units

	// Use decimation once to get the final size of the frame
	d = decimation.process(d);
	auto w = d.get_width();
	auto h = d.get_height();
	Size size = Size(w, h);

	auto intrin = stream_depth.as<rs2::video_stream_profile>().get_intrinsics();

	// Configure kinfu's parameters
	params->frameSize = size;
	params->intr = Matx33f(intrin.fx, 0,		intrin.ppx,
							0,		intrin.fy,	intrin.ppy,
							0,		0,			1);
	params->depthFactor = 1 / depth_scale;
	params->icpAngleThresh = (float)(20. * CV_PI / 180.);
	params->icpDistThresh = 0.05f;
	Vec<float, 5> coeffs(intrin.coeffs);
	// Initialize KinFu object
	kf = kinfu::KinFu::create(params);

	bool after_reset = false;
	mat_queue points_queue;
	mat_queue normals_queue;

	window app(1280, 720, "RealSense KinectFusion Example");
	glfw_state app_state;
	register_glfw_callbacks(app, app_state);

	std::atomic_bool stopped(false);

	// This thread runs KinFu algorithm and calculates the pointcloud by fusing depth data from subsequent depth frames
	std::thread calc_cloud_thread([&]() {
		Mat _points;
		Mat _normals;
		int counter = 0;
		try {
			while (!stopped)
			{
				rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera

				auto d = data.get_depth_frame();
				// Use post processing to improve results
				d = decimation.process(d);
				d = spatial.process(d);
				d = temporal.process(d);

				// Set depth values higher than clipping_dist to 0, to avoid unnecessary noise in the pointcloud
				//#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
				uint16_t* p_depth_frame = reinterpret_cast<uint16_t*>(const_cast<void*>(d.get_data()));
#pragma omp parallel for schedule(dynamic)
				for (int y = 0; y < h; y++)
				{
					auto depth_pixel_index = y * w;
					for (int x = 0; x < w; x++, ++depth_pixel_index)
					{
						// Check if the depth value of the current pixel is greater than the threshold
						if (p_depth_frame[depth_pixel_index] > clipping_dist)
						{
							p_depth_frame[depth_pixel_index] = 0;
						}
					}
				}

				// Define matrices on the GPU for KinFu's use
				UMat points;
				UMat normals;
				// Copy frame from CPU to GPU
				Mat dist(h, w, CV_16UC1, (void*)d.get_data());
				Mat undist;
				undistort(dist, undist, params->intr, coeffs);
				UMat frame(h, w, CV_16UC1);
				undist.copyTo(frame);
				undist.release();
				
				// Run KinFu on the new frame(on GPU)
				if (!frame.empty() && !kf->update(frame))
				{
					kf->reset(); // If the algorithm failed, reset current state

					// To avoid calculating pointcloud before new frames were processed, set 'after_reset' to 'true'
					after_reset = true;
					std::cout << counter << "\n";
					counter = 0;
					points.release();
					normals.release();
					std::cout << "reset" << std::endl;
				}
				else
				{
					++counter;
				}

				// Get current pointcloud
				if (counter > 10)
				{
					counter = 0;
					kf->getCloud(points, normals);
				}

				if (!points.empty() && !normals.empty())
				{
					// copy points from GPU to CPU for rendering
					points.copyTo(_points);
					points.release();
					normals.copyTo(_normals);
					normals.release();
					// Save the pointcloud obtained before failure
					export_to_ply(_points, _normals);
					// Push to queue for rendering
					points_queue.push(_points);
					normals_queue.push(_normals);
				}
				after_reset = false;
			}
		}
		catch (const std::exception& e) // Save pointcloud in case an error occurs (for example, camera disconnects)
		{
			export_to_ply(_points, _normals);
		}
	});

	// Main thread handles rendering of the pointcloud
	Mat points;
	Mat normals;
	while (app)
	{
		// Get the current state of the pointcloud
		points_queue.try_get_next_item(points);
		normals_queue.try_get_next_item(normals);
		if (!points.empty() && !normals.empty()) // points or normals might not be ready on first iterations
			draw_kinfu_pointcloud(app_state, points, normals);
	}
	stopped = true;
	calc_cloud_thread.join();

	// Save the pointcloud upon closing the app
	export_to_ply(points, normals);

	return 0;
}

int customDepthMapKinFuExample()
{
	setUseOptimized(true);
	// Declare KinFu and params pointers
	Ptr<kinfu::KinFu> kf;
	Ptr<kinfu::Params> params = createRealsenseParams();

	// Create a pipeline and configure it
	rs2::pipeline p;
	rs2::config cfg;
	float depth_scale = 1;
	cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
	cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
	auto profile = p.start(cfg);
	
	auto clipping_dist = 2.5 / depth_scale; // convert clipping_dist to raw depth units

	auto w = 640;
	auto h = 480;
	Size size = Size(w, h);

	// Configure kinfu's parameters
	params->frameSize = size;
	params->intr = Matx33f(383.748443603516f, 0, 319.515716552734f,
							0, 383.748443603516f, 239.890762329102f,
							0, 0, 1);
	params->depthFactor = 1 / depth_scale;
	params->icpAngleThresh = (float)(20. * CV_PI / 180.);
	params->icpDistThresh = 0.05f;
	Vec<float, 5> coeffs(0,0,0,0,0);
	
	
	float f = 383.748443603516f, Tx = 0.05011365562f, diff = (319.515716552734 - 319.515716552734);
	DisparityMapMaker disparityMapMaker(32, 21);

	// Initialize KinFu object
	kf = kinfu::KinFu::create(params);
	//Create a depth cleaner instance
	//rgbd::DepthCleaner* depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);
	int SCALE_FACTOR = 1;
	bool after_reset = false;
	mat_queue points_queue;
	mat_queue normals_queue;

	window app(1280, 720, "RealSense KinectFusion Example");
	glfw_state app_state;
	register_glfw_callbacks(app, app_state);

	std::atomic_bool stopped(false);

	// This thread runs KinFu algorithm and calculates the pointcloud by fusing depth data from subsequent depth frames
	std::thread calc_cloud_thread([&]() {
		Mat _points;
		Mat _normals;
		Mat depth;
		int counter = 0;
		try {
			while (!stopped)
			{
				rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera
				
				rs2::frame rsleft = data.get_infrared_frame(1);
				rs2::frame rsright = data.get_infrared_frame(2);
				Mat left(params->frameSize, CV_8UC1, (void*)rsleft.get_data(), Mat::AUTO_STEP);
				Mat right(params->frameSize, CV_8UC1, (void*)rsright.get_data(), Mat::AUTO_STEP);
				
				disparityMapMaker.compute(left, right);
				Mat disparity = disparityMapMaker.getDisparityMap();
				//imshow("Disparity", disparity);
				depth = Mat(disparity.rows, disparity.cols, CV_32F);
				// Calculate depth
				depth.forEach<float>([f, Tx, diff, disparity](float &p, const int * position) -> void {
					p = (-f*Tx) / (float(diff - disparity.at<float>(position[0], position[1])));
				});
				
				// Set depth values higher than clipping_dist to 0, to avoid unnecessary noise in the pointcloud
				depth.forEach<float>([clipping_dist](float &p, const int * position) -> void {
					if (p > clipping_dist) p = 0;
				});
				
				// Define matrices on the GPU for KinFu's use
				UMat points;
				UMat normals;
				// Copy frame from CPU to GPU
				Mat undist;
				//depth.convertTo(undist, CV_16UC1);
				undistort(depth, undist, params->intr, coeffs);
				UMat frame(h, w, CV_16UC1);
				undist.copyTo(frame);
				undist.release();
				// Run KinFu on the new frame(on GPU)
				if (!frame.empty() && !kf->update(frame))
				{
					kf->reset(); // If the algorithm failed, reset current state

					// To avoid calculating pointcloud before new frames were processed, set 'after_reset' to 'true'
					after_reset = true;
					std::cout << counter << "\n";
					counter = 0;
					points.release();
					normals.release();
					std::cout << "reset" << std::endl;
				}
				else
				{
					++counter;
				}

				// Get current pointcloud
				if (counter > 29)
				{
					counter = 0;
					kf->getCloud(points, normals);
				}


				if (!points.empty() && !normals.empty())
				{
					// copy points from GPU to CPU for rendering
					points.copyTo(_points);
					points.release();
					normals.copyTo(_normals);
					normals.release();

					// Save the pointcloud obtained before failure
					export_to_ply(_points, _normals);

					// Push to queue for rendering
					points_queue.push(_points);
					normals_queue.push(_normals);
				}
				after_reset = false;
			}
		}
		catch (const std::exception& e) // Save pointcloud in case an error occurs (for example, camera disconnects)
		{
			cout << e.what() << endl;
			export_to_ply(_points, _normals);
		}
	});

	// Main thread handles rendering of the pointcloud
	Mat points;
	Mat normals;
	while (app)
	{
		// Get the current state of the pointcloud
		points_queue.try_get_next_item(points);
		normals_queue.try_get_next_item(normals);
		if (!points.empty() && !normals.empty()) // points or normals might not be ready on first iterations
			draw_kinfu_pointcloud(app_state, points, normals);
	}
	stopped = true;
	calc_cloud_thread.join();

	// Save the pointcloud upon closing the app
	export_to_ply(points, normals);

	return 0;
}

int updatedDepthKinFuExample()
{
	int SCALE_FACTOR = 1;

	setUseOptimized(true);
	// Declare KinFu and params pointers
	Ptr<kinfu::KinFu> kf;
	Ptr<kinfu::Params> params = createRealsenseParams();

	// Create a pipeline and configure it
	rs2::pipeline p;
	rs2::config cfg;
	float depth_scale;
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16);
	auto profile = p.start(cfg);
	auto dev = profile.get_device();
	auto stream_depth = profile.get_stream(RS2_STREAM_DEPTH);

	// Get a new frame from the camera
	rs2::frameset data = p.wait_for_frames();
	auto d = data.get_depth_frame();

	for (rs2::sensor& sensor : dev.query_sensors())
	{
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			// Set some presets for better results
			dpt.set_option(RS2_OPTION_VISUAL_PRESET, RS2_RS400_VISUAL_PRESET_HIGH_DENSITY);
			// Depth scale is needed for the kinfu set-up
			depth_scale = dpt.get_depth_scale();
			break;
		}
	}

	// Declare post-processing filters for better results
	auto decimation = rs2::decimation_filter();
	auto spatial = rs2::spatial_filter();
	auto temporal = rs2::temporal_filter();

	auto clipping_dist = max_dist / depth_scale; // convert clipping_dist to raw depth units

	// Use decimation once to get the final size of the frame
	d = decimation.process(d);
	auto w = d.get_width();
	auto h = d.get_height();
	Size size = Size(w, h);

	auto intrin = stream_depth.as<rs2::video_stream_profile>().get_intrinsics();

	// Configure kinfu's parameters
	params->frameSize = size;
	params->intr = Matx33f(intrin.fx, 0, intrin.ppx,
		0, intrin.fy, intrin.ppy,
		0, 0, 1);
	params->depthFactor = 1 / depth_scale;
	params->icpAngleThresh = (float)(20. * CV_PI / 180.);
	params->icpDistThresh = 0.05f;
	Vec<float, 5> coeffs(intrin.coeffs);
	// Initialize KinFu object
	kf = kinfu::KinFu::create(params);
	// depth filter
	rgbd::DepthCleaner* depthc = new rgbd::DepthCleaner(CV_16U, 7, rgbd::DepthCleaner::DEPTH_CLEANER_NIL);

	bool after_reset = false;
	mat_queue points_queue;
	mat_queue normals_queue;

	window app(1280, 720, "RealSense KinectFusion Example");
	glfw_state app_state;
	register_glfw_callbacks(app, app_state);

	std::atomic_bool stopped(false);

	// This thread runs KinFu algorithm and calculates the pointcloud by fusing depth data from subsequent depth frames
	std::thread calc_cloud_thread([&]() {
		Mat _points;
		Mat _normals;
		int counter = 0;
		try {
			while (!stopped)
			{
				rs2::frameset data = p.wait_for_frames(); // Wait for next set of frames from the camera

				auto depth_frame = data.get_depth_frame();
				// Use post processing to improve results
				depth_frame = decimation.process(depth_frame);
				depth_frame = spatial.process(depth_frame);
				depth_frame = temporal.process(depth_frame);

				rs2::frame filtered = depth_frame;
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
				

				// Set depth values higher than clipping_dist to 0, to avoid unnecessary noise in the pointcloud
				//#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
				cleanedDepth.forEach<short>([clipping_dist](short &p, const int * position) -> void {
					if (p > clipping_dist) p = 0;
				});

				// Define matrices on the GPU for KinFu's use
				UMat points;
				UMat normals;
				// Copy frame from CPU to GPU
				Mat dist;
				cleanedDepth.convertTo(dist, CV_16UC1);

				Mat undist;
				undistort(dist, undist, params->intr, coeffs);
				UMat frame(h, w, CV_16UC1);
				undist.copyTo(frame);
				undist.release();

				// Run KinFu on the new frame(on GPU)
				if (!frame.empty() && !kf->update(frame))
				{
					kf->reset(); // If the algorithm failed, reset current state

					// To avoid calculating pointcloud before new frames were processed, set 'after_reset' to 'true'
					after_reset = true;
					std::cout << counter << "\n";
					counter = 0;
					points.release();
					normals.release();
					std::cout << "reset" << std::endl;
				}
				else
				{
					++counter;
				}

				// Get current pointcloud
				if (counter > 9)
				{
					counter = 0;
					kf->getCloud(points, normals);
				}

				if (!points.empty() && !normals.empty())
				{
					// copy points from GPU to CPU for rendering
					points.copyTo(_points);
					points.release();
					normals.copyTo(_normals);
					normals.release();
					// Save the pointcloud obtained before failure
					export_to_ply(_points, _normals);
					// Push to queue for rendering
					points_queue.push(_points);
					normals_queue.push(_normals);
				}
				after_reset = false;
			}
		}
		catch (const std::exception& e) // Save pointcloud in case an error occurs (for example, camera disconnects)
		{
			export_to_ply(_points, _normals);
		}
	});

	// Main thread handles rendering of the pointcloud
	Mat points;
	Mat normals;
	while (app)
	{
		// Get the current state of the pointcloud
		points_queue.try_get_next_item(points);
		normals_queue.try_get_next_item(normals);
		if (!points.empty() && !normals.empty()) // points or normals might not be ready on first iterations
			draw_kinfu_pointcloud(app_state, points, normals);
	}
	stopped = true;
	calc_cloud_thread.join();

	// Save the pointcloud upon closing the app
	export_to_ply(points, normals);

	return 0;
}
