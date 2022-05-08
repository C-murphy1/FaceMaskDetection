#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace cv;
using namespace std;
using namespace dnn;

int main(int argc, char* argv[] ){
    vector<string> class_names;
    class_names.push_back("No Mask");
    class_names.push_back("Wearing a Mask");

    Mat input = imread(argv[1]); //Inputs target image
    Mat input_gray;
    Mat input_resized;
    auto model = readNetFromTensorflow("model/saved_model.pb");//imports model

    cvtColor(input, input_gray, COLOR_BGR2GRAY); //converts image to greyscale 
    resize(input_gray, input_resized, Size(180,180),INTER_LINEAR); //Resizes image to match training data

    model.setInput(input_resized);
    Mat outputs = model.forward();

    Point classIdPoint;
    double final_prob;

    minMaxLoc(outputs.reshape(1, 1), 0, &final_prob, 0, &classIdPoint);
    int label_id = classIdPoint.x;

    // Print predicted class.
    string out_text = format("%s, %.3f", (class_names[label_id].c_str()), final_prob);
    cout << out_text;

}