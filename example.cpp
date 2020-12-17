#include "example.hpp"

void set_viewport(const rect& r)
{
	glViewport((int)r.x, (int)r.y, (int)r.w, (int)r.h);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glOrtho(0, r.w, r.h, 0, -1, +1);
}

void draw_pointcloud(float width, float height, glfw_state& app_state, rs2::points& points)
{
	if (!points)
		return;

	// OpenGL commands that prep screen for the pointcloud
	glLoadIdentity();
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
	glClear(GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	gluPerspective(60, width / height, 0.01f, 10.0f);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

	glTranslatef(0, 0, +0.5f + app_state.offset_y*0.05f);
	glRotated(app_state.pitch, 1, 0, 0);
	glRotated(app_state.yaw, 0, 1, 0);
	glTranslatef(0, 0, -0.5f);

	glPointSize(width / 640);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());
	float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
	glBegin(GL_POINTS);


	/* this segment actually prints the pointcloud */
	auto vertices = points.get_vertices();              // get vertices
	auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
	for (int i = 0; i < points.size(); i++)
	{
		if (vertices[i].z)
		{
			// upload the point and texture coordinates only for points we have depth data for
			glVertex3fv(vertices[i]);
			glTexCoord2fv(tex_coords[i]);
		}
	}

	// OpenGL cleanup
	glEnd();
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void quat2mat(rs2_quaternion& q, GLfloat H[16])  // to column-major matrix
{
	H[0] = 1 - 2 * q.y*q.y - 2 * q.z*q.z; H[4] = 2 * q.x*q.y - 2 * q.z*q.w;     H[8] = 2 * q.x*q.z + 2 * q.y*q.w;     H[12] = 0.0f;
	H[1] = 2 * q.x*q.y + 2 * q.z*q.w;     H[5] = 1 - 2 * q.x*q.x - 2 * q.z*q.z; H[9] = 2 * q.y*q.z - 2 * q.x*q.w;     H[13] = 0.0f;
	H[2] = 2 * q.x*q.z - 2 * q.y*q.w;     H[6] = 2 * q.y*q.z + 2 * q.x*q.w;     H[10] = 1 - 2 * q.x*q.x - 2 * q.y*q.y; H[14] = 0.0f;
	H[3] = 0.0f;                      H[7] = 0.0f;                      H[11] = 0.0f;                      H[15] = 1.0f;
}

// Handles all the OpenGL calls needed to display the point cloud w.r.t. static reference frame
void draw_pointcloud_wrt_world(float width, float height, glfw_state& app_state, rs2::points& points, rs2_pose& pose, float H_t265_d400[16], std::vector<rs2_vector>& trajectory)
{
	if (!points)
		return;

	// OpenGL commands that prep screen for the pointcloud
	glLoadIdentity();
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
	glClear(GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	gluPerspective(60, width / height, 0.01f, 10.0f);


	// viewing matrix
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	// rotated from depth to world frame: z => -z, y => -y
	glTranslatef(0, 0, -0.75f - app_state.offset_y*0.05f);
	glRotated(app_state.pitch, 1, 0, 0);
	glRotated(app_state.yaw, 0, -1, 0);
	glTranslatef(0, 0, 0.5f);

	// draw trajectory
	glEnable(GL_DEPTH_TEST);
	glLineWidth(2.0f);
	glBegin(GL_LINE_STRIP);
	for (auto&& v : trajectory)
	{
		glColor3f(0.0f, 1.0f, 0.0f);
		glVertex3f(v.x, v.y, v.z);
	}
	glEnd();
	glLineWidth(0.5f);
	glColor3f(1.0f, 1.0f, 1.0f);

	// T265 pose
	GLfloat H_world_t265[16];
	quat2mat(pose.rotation, H_world_t265);
	H_world_t265[12] = pose.translation.x;
	H_world_t265[13] = pose.translation.y;
	H_world_t265[14] = pose.translation.z;

	glMultMatrixf(H_world_t265);

	// T265 to D4xx extrinsics
	glMultMatrixf(H_t265_d400);


	glPointSize(width / 640);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, app_state.tex.get_gl_handle());
	float tex_border_color[] = { 0.8f, 0.8f, 0.8f, 0.8f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_border_color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE
	glBegin(GL_POINTS);

	/* this segment actually prints the pointcloud */
	auto vertices = points.get_vertices();              // get vertices
	auto tex_coords = points.get_texture_coordinates(); // and texture coordinates
	for (int i = 0; i < points.size(); i++)
	{
		if (vertices[i].z)
		{
			// upload the point and texture coordinates only for points we have depth data for
			glVertex3fv(vertices[i]);
			glTexCoord2fv(tex_coords[i]);
		}
	}

	// OpenGL cleanup
	glEnd();
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

// Registers the state variable and callbacks to allow mouse control of the pointcloud
void register_glfw_callbacks(window& app, glfw_state& app_state)
{
    app.on_left_mouse = [&](bool pressed)
    {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset)
    {
        app_state.offset_x -= static_cast<float>(xoffset);
        app_state.offset_y -= static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y)
    {
        if (app_state.ml)
        {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key)
    {
        if (key == 32) // Escape
        {
            app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}