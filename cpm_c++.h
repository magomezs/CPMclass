#ifndef CPM_C_H
#define CPM_C_H

#include <caffe/net.hpp>
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>

using namespace caffe;
using namespace cv;
using namespace std;


Mat intMat2floatMat(Mat*);
Mat padRightDownCorner(Mat*);



class CDM{
public:
   CDM(){};
   CDM(const string&, const string&);
   Mat featureComputation(Mat img);

private:
   void WrapInputLayer(std::vector<cv::Mat>* input_channels);
   void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
   caffe::shared_ptr<Net<float> > net_;
   cv::Size input_geometry_;
   int num_channels_;
};


class CPM {
public:
   vector<Point> sizes;
   CPM(){};
   CPM(const string&, const string&);
   std::vector<Mat> featureComputation(const cv::Mat& img, const cv::Mat& map);

private:
   void WrapInputLayer(std::vector<cv::Mat>* input_channels);
   void Preprocess(const cv::Mat& img, const cv::Mat& map, std::vector<cv::Mat>* input_channels);
   caffe::shared_ptr<Net<float> > net_;
   cv::Size input_geometry_;
   int num_channels_;
 };


class CPMclass{
public:
    CDM cdm;
    CPM cpm;
    CPMclass(){};
    CPMclass(const string&, const string&, const string&, const string&);
    vector<vector<Point>> detect(Mat* frame);
    void showSkeletons(string, vector<vector<Point>>*, Mat* frame, int time);
};


#endif // CPM_C_H
