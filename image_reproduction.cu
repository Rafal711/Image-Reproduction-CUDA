#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>
#include <math.h>

//crossover operation functions
//running on max 1024 threads since that is a max chromosome length
//function running in one block (?)

//single single-point crossover operation
__device__ void single_point(const uint8_t* parent_one, const uint8_t* parent_two,
                             uint8_t* offspring_one, uint8_t* offspring_two, const int* crossover_point) {

    int id = threadIdx.x; //get the ID of a thread
    //int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id <= *crossover_point) {
        offspring_one[id] = parent_one[id];
        offspring_two[id] = parent_two[id];
    }
    else {
        offspring_one[id] = parent_two[id];
        offspring_two[id] = parent_one[id];
    }

}


//single two-point crossover operation
__device__ void two_point(const uint8_t* parent_one, const uint8_t* parent_two,
                          uint8_t* offspring_one, uint8_t* offspring_two,
                          const int* first_crossover_point, const int* second_crossover_point) {

    int id = threadIdx.x; //get the ID of a thread
    //int id = blockDim.x * blockIdx.x + threadIdx.x;


    if (id <= *first_crossover_point || id > *second_crossover_point) {
        offspring_one[id] = parent_one[id];
        offspring_two[id] = parent_two[id];
    }
    else {
        offspring_one[id] = parent_two[id];
        offspring_two[id] = parent_one[id];
    }

}


//single uniform crossover operation
__device__ void uniform(const uint8_t* parent_one, const uint8_t* parent_two,
                        uint8_t* offspring_one, uint8_t* offspring_two,
                        const bool* crossover_mask) {

    int id = threadIdx.x; //get the ID of a thread
    //int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (crossover_mask[id]) {
        offspring_one[id] = parent_two[id];
        offspring_two[id] = parent_one[id];
    }
    else {
        offspring_one[id] = parent_one[id];
        offspring_two[id] = parent_two[id];
    }
}


__device__ void perform_crossover() {
    //TODO
}



cv::Mat* imgRGB2chromosome(cv::Mat image) {
    cv::Mat* reshaped_img = new cv::Mat();
    *reshaped_img = image.reshape(1, 1);
    return reshaped_img;
}

void reproduce_image(cv::Mat original_image) {
    cv::Mat* chromosome_base = imgRGB2chromosome(original_image);

    std::cout << "Original size: " << original_image.size << std::endl;
    std::cout << "Reshaped size: " << chromosome_base->size << std::endl;

}

void run_program() {
    std::string image_path = "tangerines.jpg";
    cv::Mat img = cv::imread(image_path);

    int number_of_iterations = 10000;

    // GA reproduction parameters
    int number_of_parents = 4;
    int number_of_offsprings = 8;

    // mutation parameters
    float mutation_percentage = 0.01;

    // termination condition
    float epsilon = pow(10, -12);
    int terminate_after = 500;

    reproduce_image(img);



    //cv::namedWindow("First OpenCV Application", cv::WINDOW_AUTOSIZE);
    //cv::imshow("First OpenCV Application", img);
    //cv::moveWindow("First OpenCV Application", 0, 45);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
 
}


int main()
{
    run_program();
    return 0;
}