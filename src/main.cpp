#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 3 )
   { return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
   { return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 5000;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute( img_1, keypoints_1, descriptors_1 );
  extractor.compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors with a brute force matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

  //-- Show detected matches
  imshow("Matches", img_matches );

  waitKey(0);

  return 0;
  }

 /** @function readme */
 void readme()
 { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }


/*
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> //Thanks to Alessandro
#include <iostream>

int main(int argc, const char* argv[])
{

    cvNamedWindow("win");

    CvCapture* capture = cvCreateCameraCapture(0);
    IplImage* frame;
    cv::Mat input;

    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    // Add results to image and save.
    cv::Mat output;

    long id = 0;
    std::string name;
    char t[256];
    std::string s;

    while(1) {
        frame = cvQueryFrame(capture);
        if(!frame) break;
        cvShowImage("win", frame);

        input = cv::Mat(frame, 0);//cv::imread("input.jpg", 0); //Load as grayscale
        detector.detect(input, keypoints);
        cv::drawKeypoints(input, keypoints, output);


        sprintf(t, "%ld", id);
        s = t;
        name = "sift_result_" + s + ".jpg";
        std::cout << name << std::endl;
        cv::imwrite(name, output);
        id++;


        char c = cvWaitKey(50);
        if(c==27) break;
    }

    cvReleaseCapture(&capture);
    cvDestroyWindow("win");

    /*
    const cv::Mat input = cv::imread("input.jpg", 0); //Load as grayscale

    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("sift_result.jpg", output);

    */
/*
    return 0;
}*/
