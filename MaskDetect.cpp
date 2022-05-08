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
    class_names.push_back("Mask");
    class_names.push_back("No Mask");

    Mat input = imread(argv[1]); //Inputs target image
    Mat input_gray;
    Mat input_resized;


    auto model = readNetFromONNX("model/model.onnx");//imports model


    cvtColor(input, input_gray, COLOR_BGR2GRAY); //converts image to greyscale 
    resize(input_gray, input_resized, Size(180,180),INTER_LINEAR); //Resizes image to match training data

    model.setInput(input_resized); //Sets the formatted input image as the model input
    Mat outputs = model.forward(); //Gives the imgage to the model to get a result

    Point classType;
    double prob;

    minMaxLoc(outputs.reshape(1, 1), 0, &prob, 0, &classType);
    int label = classType.x;

    cout << class_names[label] << " with probability " << prob << "% " << "\n";
}