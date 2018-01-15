#include "cpm_c++.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cxcore.h>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;

Mat padRightDownCorner(Mat* img){
    int h = img->rows;
    int w = img->cols;

    int h_new= (h%8==0)? h: h+8-(h%8);
    int w_new= (w%8==0)? w: w+8-(w%8);

    Mat img_padded=Mat(h_new,w_new,img->type(), Scalar(0.5, 0.5, 0.5));
    img->copyTo(img_padded(Rect(0,0,w,h)));

    return img_padded;
}


Mat intMat2floatMat(Mat* input){
    Mat float_input= Mat(input->rows, input->cols, CV_32FC3);
    for (int i=0; i<float_input.rows; i++)
    {
        for (int j=0; j<float_input.cols; j++)
        {
            float_input.at<Vec3f>(i,j)[0] = (float(uint(input->at<Vec3b>(i,j)[0]))/(float)255)<0.0 ? 0.0 :float(uint(input->at<Vec3b>(i,j)[0]))/(float)255;
            float_input.at<Vec3f>(i,j)[1] = (float(uint(input->at<Vec3b>(i,j)[1]))/(float)255)<0.0 ? 0.0 :float(uint(input->at<Vec3b>(i,j)[1]))/(float)255;
            float_input.at<Vec3f>(i,j)[2] = (float(uint(input->at<Vec3b>(i,j)[2]))/(float)255)<0.0 ? 0.0 :float(uint(input->at<Vec3b>(i,j)[2]))/(float)255;
            float_input.at<Vec3f>(i,j)[0] = (float(uint(input->at<Vec3b>(i,j)[0]))/(float)255)>1.0 ? 1.0 :float(uint(input->at<Vec3b>(i,j)[0]))/(float)255;
            float_input.at<Vec3f>(i,j)[1] = (float(uint(input->at<Vec3b>(i,j)[1]))/(float)255)>1.0 ? 1.0 :float(uint(input->at<Vec3b>(i,j)[1]))/(float)255;
            float_input.at<Vec3f>(i,j)[2] = (float(uint(input->at<Vec3b>(i,j)[2]))/(float)255)>1.0 ? 1.0 :float(uint(input->at<Vec3b>(i,j)[2]))/(float)255;
        }
    }
    return float_input;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CDM::CDM(const string& model_file, const string& trained_file) {
  Caffe::set_mode(Caffe::GPU);

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer_a = net_->input_blobs()[0];

  num_channels_ = input_layer_a->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer_a->width(), input_layer_a->height());
}

Mat CDM::featureComputation(Mat img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();


  /* Copy the output layer to a Mat*/
  Blob<float>* output_layer = net_->output_blobs()[0];
  int width=output_layer->shape()[3];
  int height = output_layer->shape()[2];
  Mat people_map(height, width, CV_32FC1);

  const float* begin = output_layer->cpu_data();
  const float* end = begin + width*height;//output_layer->channels();
  for (int a = 0; a < output_layer->channels(); ++a) {
      std::vector<float> result=std::vector<float>(begin, end);
      for (int i=0; i<height; i++)
      {
          for (int j=0; j<width; j++)
          {
            people_map.at<float>(i,j)=result[width*i+j];
          }
      }
      begin=end;
      end=begin+width*height;
  }


  return people_map;

}

/* Wrap the input layer of the network*/
void CDM::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void CDM::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {

  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;


  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::split(sample_float, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CPM::CPM(const string& model_file, const string& trained_file) {
  Caffe::set_mode(Caffe::CPU);

  // Load the network.
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer_a = net_->input_blobs()[0];

  num_channels_ = input_layer_a->channels();
  CHECK(num_channels_ == 4)
    << "Input layer should have 1 or 3 channels."<<num_channels_;
  input_geometry_ = cv::Size(input_layer_a->width(), input_layer_a->height());

  net_->ForwardPrefilled();
}

void CPM::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
}

void CPM::Preprocess(const cv::Mat& img, const cv::Mat& map, std::vector<cv::Mat>* input_channels) {
    // Convert the input image to the input image format of the network.
    Mat channel[3];
    split(img, channel);

    cv::Mat sample(img.cols, img.rows, CV_32FC4);
    cv::Mat prueba(img.cols, img.rows, CV_32FC1);
    for (int i=0; i<img.rows; i++)
    {
        for (int j=0; j<img.cols; j++)
        {
          sample.at<Vec4f>(i,j)[0] = channel[0].at<float>(i,j)-0.5;
          sample.at<Vec4f>(i,j)[1] = channel[1].at<float>(i,j)-0.5;
          sample.at<Vec4f>(i,j)[2] = channel[1].at<float>(i,j)-0.5;
          sample.at<Vec4f>(i,j)[3] = map.at<float>(i,j);
          prueba.at<float>(i,j)=sample.at<Vec4f>(i,j)[0];
        }
    }
    double min, max;
    cv::Point min_loc, max_loc;

    cv::minMaxLoc(prueba, &min, &max, &min_loc, &max_loc);

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
      cv::resize(sample, sample_resized, input_geometry_);
    else
       sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 4)
      sample_resized.convertTo(sample_float, CV_32FC4);
    else
      sample_resized.convertTo(sample_float, CV_32FC1);

    cv::split(sample_float, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)== net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}


std::vector<Mat> CPM::featureComputation(const cv::Mat& img, const cv::Mat& map) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);

    // Forward dimension change to all layers.
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, map, &input_channels);

    net_->ForwardPrefilled();

    // Copy the output layer to a Mat
    Blob<float>* output_layer = net_->output_blobs()[0];
    int width=output_layer->shape()[3];
    int height = output_layer->shape()[2];
    vector<Mat> part_maps;

    const float* begin = output_layer->cpu_data();
    const float* end = begin + width*height;
    for (int a = 0; a < output_layer->channels(); ++a) {
        std::vector<float> result=std::vector<float>(begin, end);
        cv::Mat channel(height, width, CV_32FC1);
        for (int i=0; i<height; i++)
        {
            for (int j=0; j<width; j++)
            {
              channel.at<float>(i,j)=result[width*i+j];
            }
        }

        part_maps.push_back(channel);
        begin=end;
        end=begin+width*height;
    }
    return part_maps;
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CPMclass::CPMclass(const string& model_cdm_file, const string& trained_cdm_file, const string& model_cpm_file, const string& trained_cpm_file){
    cdm=CDM(model_cdm_file, trained_cdm_file);
    cpm= CPM(model_cpm_file, trained_cpm_file);
}

vector<vector<Point>> CPMclass::detect(Mat* frame){
       vector<vector<Point>> skeletons;
       int sigma = 21;
       int boxsize = 368;

       //gaussian map
       Mat gaussian = Mat::zeros(boxsize, boxsize, CV_32FC1);
       for(int x=0; x<boxsize; x++){
           for(int y=0; y<boxsize; y++){
               float dist_sq = (x - boxsize/2) * (x - boxsize/2) + (y - boxsize/2) * (y - boxsize/2);
               float exponent = dist_sq / 2.0 / sigma / sigma;
               gaussian.at<float>(x, y) = exp(-exponent);
           }

       }

       float scale=1;
       float factor=scale*(float)boxsize/(float)frame->rows;//factor to resize the input

       Mat input;
       frame->copyTo(input);
       Mat float_input=Mat(frame->rows, frame->cols, CV_32FC3);
       Mat input_image_resize, result, result_resize;

       //input to float
       float_input=intMat2floatMat(&input);

       //resize
       resize(float_input, input_image_resize, Size(0,0), factor, factor, INTER_CUBIC);

       //padded (x8)
       Mat padded_img=padRightDownCorner(&input_image_resize);

       //get gaussian
       result= cdm.featureComputation(padded_img);//result is 8 times smaller than padded_img

       //result resize, same size than padded_img
       resize(result, result_resize, Size(0,0), (float)padded_img.cols/(float)result.cols, (float)padded_img.rows/(float)result.rows, INTER_CUBIC);

       //Positions estimation
       Mat locations;
       vector<Vec3f> circles;
       threshold(result_resize, locations, 0.5, 1, 1);//threshold(result_resize, locations, threshold_param, 1, 1);
       locations.convertTo(locations, CV_8UC3);
       HoughCircles(locations, circles, CV_HOUGH_GRADIENT, 1, 15, 1, 10, 0, 200 );// Apply the Hough Transform to find the circles       

       // Get joints
       for( size_t i = 0; i < circles.size(); i++ ){
             vector<Point> skeleton;
             Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
             Mat color= Mat(boxsize, boxsize, CV_32FC3, Scalar(0,0,0));
             for(int x=0; x<boxsize; x++){
                 for(int y=0; y<boxsize; y++){
                     int u=x+center.x-cvRound((float)boxsize/2.0);
                     int v=y+center.y-cvRound((float)boxsize/2.0);
                     if(u>=0 && v>=0 && u<padded_img.cols && v<padded_img.rows){
                         color.at<Vec3f>(y,x)[0] = (float(padded_img.at<Vec3f>(v,u)[0]));
                         color.at<Vec3f>(y,x)[1] = (float(padded_img.at<Vec3f>(v,u)[1]));
                         color.at<Vec3f>(y,x)[2] = (float(padded_img.at<Vec3f>(v,u)[2]));
                     }
                 }
             }

             vector<Mat> results;
             results= cpm.featureComputation(color, gaussian);

             std::vector<cv::Point> predictions, predictions_in_frame;
             for(int i=0; i<results.size(); i++)
             {
                    Mat part;
                    resize(results[i], part, Size(0,0), 8, 8, INTER_CUBIC);
                    double min, max;
                    cv::Point min_loc, max_loc, position;
                    cv::minMaxLoc(part, &min, &max, &min_loc, &max_loc);
                    predictions.push_back(max_loc);
                    position.x=cvRound((float)(cvRound(max_loc.x)+cvRound((float)center.x)-cvRound((float)boxsize/2.0))/factor);
                    position.y=cvRound((float)(cvRound(max_loc.y)+cvRound((float)center.y)-cvRound((float)boxsize/2.0))/factor);
                    predictions_in_frame.push_back(position);
             }

             //extract joints
             for(int j=0; j<predictions.size(); j++){
                 skeleton.push_back(predictions_in_frame[j]);
             }

             skeletons.push_back(skeleton);
       }

       return skeletons;
}

void CPMclass::showSkeletons(string windowname, vector<vector<Point>>* skeletons, Mat* frame, int time){

    Mat img=*frame;
    Mat img_out;
    img.copyTo(img_out);
    static Scalar red=Scalar(0,0,255);
    static Scalar pink=Scalar(255,0,255);
    static Scalar purple=Scalar(255,0,128);
    static Scalar orange=Scalar(0,128,255);
    static Scalar yellow=Scalar(0,255,255);
    static Scalar cian=Scalar(255,255,0);
    static Scalar blue=Scalar(255,0,0);
    static Scalar lime=Scalar(0,255,0);
    static Scalar green=Scalar(0, 128, 0);


    for (int i=0; i<int(skeletons->size()); i++){
        vector<Point> p=skeletons->at(i);
        line(img_out, p[0], p[1], red, 2,8);
        line(img_out, p[2], p[3], pink, 2,8);
        line(img_out, p[3], p[4], purple, 2,8);
        line(img_out, p[5], p[6], orange, 2,8);
        line(img_out, p[6], p[7], yellow, 2,8);
        line(img_out, p[8], p[9], cian, 2,8);
        line(img_out, p[9], p[10], blue, 2,8);
        line(img_out, p[11], p[12], lime, 2,8);
        line(img_out, p[12], p[13], green, 2,8);
    }

    imshow(windowname, img_out);
    waitKey(time);

}



