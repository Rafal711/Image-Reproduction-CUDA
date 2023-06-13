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

#define N 1024 // max number of threads in one block

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

///////////////////////////////////////////////////////////////////////////////////////
// CROSSOVER

enum Crossover_method {
    SINGLE_POINT,
    TWO_POINT,
    UNIFORM
};


// crossover operation functions
// running as CHILD kernels - 1D grid, block -> crossover of one pair, thread in a block -> gene in a chromosome 

// single single-point crossover operation
__global__ void single_point(const uint8_t* parents[], uint8_t* offsprings[], const int* crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = 2 * blockIdx.x;    // index of a pair of parents

    int offspring_idx = pair_idx;

    if (gene_idx <= *crossover_point) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][gene_idx];
        offsprings[offspring_idx+1][gene_idx] = parents[pair_idx+1][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx+1][gene_idx];
        offsprings[offspring_idx+1][gene_idx] = parents[pair_idx][gene_idx];
    }

}


// single two-point crossover operation
__global__ void two_point(const uint8_t* parents[], uint8_t* offsprings[],
                          const int* first_crossover_point, const int* second_crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = 2 * blockIdx.x;    // index of a pair of parents

    int offspring_idx = pair_idx;

    if (gene_idx <= *first_crossover_point || gene_idx > *second_crossover_point) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx+1][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx+1][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][gene_idx];
    }
}


// single uniform crossover operation
__global__ void uniform(const uint8_t* parents[], uint8_t* offsprings[], const bool* crossover_mask) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome
    int pair_idx = 2* blockIdx.x;    // index of a pair of parents
    int offspring_idx = pair_idx;

    if (crossover_mask[gene_idx]) {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx+1][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx][gene_idx];
    }
    else {
        offsprings[offspring_idx][gene_idx] = parents[pair_idx][gene_idx];
        offsprings[offspring_idx + 1][gene_idx] = parents[pair_idx+1][gene_idx];
    }
}


__device__ void generate_random_unique_array(curandState* state, const int state_idx, int* tab, const int length, const int max) {
    for (int i = 0; i < length; i++) {
        bool unique = 0;
        while (!unique) {
            unique = 1;
            float random = curand_uniform(&state[state_idx]);
            random *= (max + 0.999999);
            tab[i] = (int)truncf(random);
            for (int j = 0; j < i; j++) {
                if (tab[i] == tab[j]) {
                    unique = 0;
                    break;
                }
            }
        }
    } 
}


// crossover over a given mating pool (called for every thread (algorithm) in reproduce_image())
__device__ void perform_crossover(const uint8_t** mating_pool[], uint8_t** offsprings[],
                                  const Crossover_method method, const int* combinations,
                                  const int mating_pool_length, const int number_of_offsprings, 
                                  curandState* state) {

    int algorithm_idx = threadIdx.x;
    const int number_of_crossovers = number_of_offsprings / 2;

    // allocating memory for a 2D array of parent chromosomes (every 2 make a pair)
    const uint8_t** parents = new uint8_t*[number_of_crossovers*2];

    // generate array of random integers (pair idx in combinations array), no repetitions
    int* rand_int = new int[number_of_crossovers];
    generate_random_unique_array(state, algorithm_idx, rand_int, number_of_crossovers, mating_pool_length*(mating_pool_length-1)/2);

    // pick pairs based on randomly generated array
    for (int i = 0; i < number_of_crossovers; i++) {
        parents[2 * i] = mating_pool[algorithm_idx][combinations[2 * rand_int[i]]];
        parents[2 * i + 1] = mating_pool[algorithm_idx][combinations[2 * rand_int[i] + 1]];
    }
    
    // launching crossover as child kernels
    float random;
    if (method == Crossover_method::SINGLE_POINT) {
        random = curand_uniform(&state[algorithm_idx]) * (1024 + 0.999999);
        int crossover_point = (int)truncf(random);
        single_point << <number_of_crossovers, 1024 >> > (parents, offsprings[algorithm_idx], &crossover_point);
    }
    else if (method == Crossover_method::TWO_POINT) {
        random = curand_uniform(&state[algorithm_idx]) * (1024 + 0.999999);
        int first_crossover_point = (int)truncf(random);
        random = curand_uniform(&state[algorithm_idx]) * (1024 + 0.999999);
        int second_crossover_point = (int)truncf(random);
        two_point << <number_of_crossovers, 1024 >> > (parents, offsprings[algorithm_idx], &first_crossover_point, &second_crossover_point);
    }
    else {
        bool crossover_mask[1024];
        //generate nr of genes to crossover
        for (int i = 0; i < 1024; i++) {
            random = curand_uniform(&state[algorithm_idx]) * (1 + 0.999999);
            crossover_mask[i] = (int)truncf(random);
        }
        uniform << <number_of_crossovers, 1024>> > (parents, offsprings[algorithm_idx], crossover_mask);
    }
    
    // freeing the memory
    delete[] parents;
    delete[] rand_int;
}

///////////////////////////////////////////////////////////////////////////////////////


// main loop of the algorithm (kernel - 1 block with N threads, where 1 thread is 1 algorithm)
__global__ void reproduce_image(uint8_t base_chromosome) {


}


__global__ void setup_curand(curandState* state) {
    int idx = threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}


void cvMatToGrayArray(const cv::Mat& image, uint8_t* array) {
    cv::Mat greyMat;
    cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);

    for (int i = 0; i < greyMat.rows * greyMat.cols; i++) {
        array[i] = *(greyMat.begin<uint8_t>() + i);
    }
}

cv::Mat grayArrayToCvMat(const cv::Mat& image, const uint8_t* array) {
    cv::Mat reshaped = cv::Mat(image.rows * image.cols, 1, CV_8UC1, (unsigned*)array).reshape(0, image.rows);
    return reshaped;
}

__host__ void generate_all_idx_combinations(int* pairs, int max_idx) {

    int counter = 0;
    for (int i = 0; i < max_idx; i++) {
        for (int j = i; j < max_idx; j++) {
            pairs[2 * counter] = i;
            pairs[2 * counter + 1] = j;
        }
    }
}

__global__ void setup_kernel_multi_blocks(curandState* state, uint16_t seed_offset)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed_offset + idx, idx, 0, &state[idx]);
}

__global__ void populationInit_multi_blocks(uint8_t* population, curandState* state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int range{ 255 };
    population[idx] = (uint8_t)(ceil((curand_uniform(&(state[idx])) * (range + 1))) - 1);;
}

__host__ void run_program() {
    std::string image_path = "tangerines.jpg";
    cv::Mat image = cv::imread(image_path);
    uint8_t* grayArray = new uint8_t[image.rows * image.cols];
    cvMatToGrayArray(image, grayArray);

    uint16_t number_of_iterations = 10000;

    // GA reproduction parameters
    uint16_t number_of_parents = 4;
    uint16_t number_of_offsprings = 8;

    // mutation parameters
    float mutation_percentage = 0.01;

    // termination condition
    float epsilon = 1.0E-12F;
    uint16_t terminate_after = 500;

    uint16_t nr_of_parallel_algorithms = 3;
    

    uint16_t chromosome_size = 5;
    uint16_t population_size = 4;
    uint16_t number_of_threads = chromosome_size;
    uint16_t number_of_blocks = population_size * nr_of_parallel_algorithms;
    uint16_t multiple_population_size = population_size * nr_of_parallel_algorithms * chromosome_size;


    // HOST concatenated population allocation
    uint8_t* h_mpopulation = new uint8_t[multiple_population_size];

    // DEVICE concatenated population allocation
    uint8_t* d_mpopulation;
    CUDA_CALL(cudaMalloc(&d_mpopulation, sizeof(uint8_t) * multiple_population_size));

    // DEVICE curandState initialization
    curandState* devStates;
    CUDA_CALL(cudaMalloc((void**)&devStates, multiple_population_size * sizeof(curandState)));

    // ------------------------------------------ genetic algorithm start ------------------------------------------
    setup_kernel_multi_blocks <<< number_of_blocks, number_of_threads >>> (devStates, 7);
    populationInit_multi_blocks <<< number_of_blocks, number_of_threads >>> (d_mpopulation, devStates);








    // ------------------------------------------ genetic algorithm end ------------------------------------------

    CUDA_CALL(cudaMemcpy(h_mpopulation, d_mpopulation, sizeof(uint8_t) * multiple_population_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // show results
    for (std::size_t i = 0; i < multiple_population_size; ++i) {
        std::cout << static_cast<int>(h_mpopulation[i]) << " ";
    }

    // free all device memory
    cudaFree(d_mpopulation);
    // free all host memory
    delete[] h_mpopulation;







    // for random number generation on GPU
    curandState* d_state;
    std::cout << d_state->d << std::endl;
    cudaMalloc(&d_state, sizeof(curandState));
    setup_curand << <1, nr_of_parallel_algorithms >> > (d_state);

    
    //cudaDeviceSynchronize();
    size_t combinations_nr = number_of_parents * (number_of_parents - 1) / 2;
    int* combinations = new int[combinations_nr];
    generate_all_idx_combinations(combinations, number_of_parents - 1);
    cudaMalloc(&combinations, combinations_nr * sizeof(int));
    //perform_crossover << < >> > ();



    // imgRGB2chromosome(...)
    
    // kernel launch
    // reproduce_image(...);

    delete[] combinations;
    cudaFree(&d_state);
    ////Display the result image
    //cv::Mat resultImage = grayArrayToCvMat(image, grayArray);
    //cv::imwrite("Result.jpg", image);
    //cv::namedWindow("Result image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Result image", resultImage);
    //cv::moveWindow("Result image", 0, 45);
    //cv::waitKey(0);
    //cv::destroyAllWindows();
 
}


int main()
{
    run_program();
    return 0;
}
