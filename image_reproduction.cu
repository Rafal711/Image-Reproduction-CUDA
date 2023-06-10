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

#define N 1024 //max number of threads in one block


//CROSSOVER

enum Crossover_method {
    SINGLE_POINT,
    TWO_POINT,
    UNIFORM
};


//crossover operation functions
//running as CHILD kernels - 1D grid, block -> crossover of one pair, thread in a block -> gene in a chromosome 

//single single-point crossover operation
__global__ void single_point(const uint8_t* parents[][2], uint8_t* offsprings[], const int* crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = blockIdx.x;    // index of a pair of parents

    int offspring_idx = 2 * pair_idx;

    if (gene_idx <= *crossover_point) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][0][gene_idx];
        offsprings[offspring_idx+1][gene_idx] = parents[pair_idx][1][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][1][gene_idx];
        offsprings[offspring_idx+1][gene_idx] = parents[pair_idx][0][gene_idx];
    }

}


//single two-point crossover operation
__global__ void two_point(const uint8_t* parents[][2], uint8_t* offsprings[],
                          const int* first_crossover_point, const int* second_crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = blockIdx.x;    // index of a pair of parents

    int offspring_idx = 2 * pair_idx;

    if (gene_idx <= *first_crossover_point || gene_idx > *second_crossover_point) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][0][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][1][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][1][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][0][gene_idx];
    }
}


//single uniform crossover operation
__global__ void uniform(const uint8_t* parents[][2], uint8_t* offsprings[], const bool* crossover_mask) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = blockIdx.x;    // index of a pair of parents

    int offspring_idx = 2 * pair_idx;

    if (crossover_mask[gene_idx]) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][1][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][0][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][0][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][1][gene_idx];
    }
}


//crossover over a given mating pool (called for every thread (algorithm) in reproduce_image())
__device__ void perform_crossover(const uint8_t* mating_pool[], uint8_t* offsprings[],
                                  const Crossover_method method, const int number_of_offsprings,
                                  const int mating_pool_length) {

    int number_of_crossovers = number_of_offsprings / 2;

    //allocating memory for a 2D array of parent chromosomes
    uint8_t*** parents = new uint8_t**[number_of_crossovers];

    for (int i=0; i < number_of_crossovers; i++) {
        parents[i] = new uint8_t*[2];
    }

    //TODO: choosing random pairs of parents and launching child kernels

    
    // launching crossover as child kernels 
    if (method == Crossover_method::SINGLE_POINT) {
        //single_point << <x, y >> > ();
    }
    else if (method == Crossover_method::TWO_POINT) {
        //two_point << <x, y >> > ();
    }
    else {
        //uniform << <x, y >> > ();
    }
    
    //freeing the memory
    for (int i = 0; i < number_of_crossovers; i++) {
        delete[] parents[i];
    }
    delete[] parents;
}

//


// main loop of the algorithm (kernel - 1 block with N threads, where 1 thread is 1 algorithm)
__global__ void reproduce_image(uint8_t base_chromosome) {


}


void imgRGB2chromosome(cv::Mat image, uint8_t* chromosome) {

    // TODO: convert cv::Mat image to chromosome

    /*cv::Mat* reshaped_img = new cv::Mat();
    *reshaped_img = image.reshape(1, 1);
    return reshaped_img;*/
}


__host__ void run_program() {
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

    //imgRGB2chromosome(...)
    
    // kernel launch
    //reproduce_image(...);


    ////Display the result image
    //cv::namedWindow("Result image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Result image", img);
    //cv::moveWindow("Result image", 0, 45);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
 
}


int main()
{
    run_program();
    return 0;
}