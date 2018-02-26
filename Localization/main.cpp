#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// BGR
Scalar color_green(0,255,0);
Scalar color_red(0,0,255);

// frame size
double scalar = 2;
int width = 320 * scalar, height = 240 * scalar;

int thresh = 20; // max color difference between current and previous frame to count as movement
int blurFactor = 17;

bool replay = true;
int main(int argc, char** argv){
    while(replay) {
        string vidName = "bounceBall.mp4";;
        string path = "../../" + vidName;

        VideoCapture cam(path);
        Mat orig_frame;
        Mat current_frame;
        Mat last_frame;
        Mat diff;
        Mat movement;



        bool running = true;
        bool firstFrame = true;

        vector<Point> points;
        vector<cv::Rect> bounding_rect;

        while (running) {
            // get newest frame

            if (!cam.read(current_frame)) {
                running = false;
                continue;
            }
            resize(current_frame, current_frame, cv::Size(width, height));
            orig_frame = current_frame.clone();
            cvtColor(current_frame, current_frame, CV_RGB2GRAY);
            if (firstFrame) {
                last_frame = current_frame;
                firstFrame = false;
                movement = Mat(last_frame.rows, last_frame.cols, CV_8UC3, Scalar(100, 100, 100));
                continue;
            }

            GaussianBlur(current_frame, current_frame, Size(blurFactor, blurFactor), 0.0, 0);
            absdiff(current_frame, last_frame, diff);// Absolute differences between the 2 images
            threshold(diff, diff, thresh, 255, CV_THRESH_BINARY);

            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            Mat edges;
            /// Detect edges using canny
            Canny(diff, edges, 100, 100 * 2, 3);
            /// Find contours
            findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            double largest_area = 0;
            int largest_contour_index = 0;
            for (int i = 0; i < contours.size(); i++) {
                double a = contourArea(contours[i], false);  //  Find the area of contour
                if (a > largest_area) {
                    largest_area = a;
                    largest_contour_index = i;                //Store the index of largest contour
                }
            }


            ///////////////////////////////////
            //          Draw to/show Mat
            ///////////////////////////////////

            last_frame = current_frame;

            if (contours.size() > 0)
                bounding_rect.push_back(boundingRect(
                        contours[largest_contour_index])); // Find the bounding rectangle for biggest contour

            drawContours(orig_frame, contours, largest_contour_index, color_green, CV_FILLED, 8,
                         hierarchy); // Draw the largest contour using previously stored index.

            imshow("Threshold Frame", diff);
            imshow("Current Frame", orig_frame);
            for (int i = 1; i < bounding_rect.size(); i++) {
                int cx1 = bounding_rect[i - 1].x + bounding_rect[i - 1].width / 2;
                int cy1 = bounding_rect[i - 1].y + bounding_rect[i - 1].height / 2;
                int cx2 = bounding_rect[i].x + bounding_rect[i].width / 2;
                int cy2 = bounding_rect[i].y + bounding_rect[i].height / 2;
                line(movement, Point(cx1, cy1), Point(cx2, cy2), color_green, 1);
            }
            for (int i = 1; i < bounding_rect.size(); i++) {
                int cx1 = bounding_rect[i - 1].x + bounding_rect[i - 1].width / 2;
                int cy1 = bounding_rect[i - 1].y + bounding_rect[i - 1].height / 2;
                circle(movement, Point(cx1, cy1), 1, color_red, 2);
            }

            imshow("Movement", movement);

            // pause on current frame for 1 ms
            waitKey(1);

        }
        if (waitKey() != 'r'){
            break;
        }

    }
    return 0;
}