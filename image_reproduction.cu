#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

#include <iostream>
#include <string>
#include <vector>

#include <stdio.h>
#include <math.h>

#define N 1024 // max number of threads in one block

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

#define CUDA_CALL_V(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return;}} while(0)



__device__ void generate_random_unique_array(curandState * state, uint16_t * tab,
    const int length, const int min, const int max) {

    int state_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < length; i++) {
        bool unique = 0;
        while (!unique) {
            unique = 1;
            float random = curand_uniform(&state[state_idx]);
            random *= (max - min + 0.999999);
            tab[i] = (uint16_t)truncf(random);
            for (int j = 0; j < i; j++) {
                if (tab[i] == tab[j]) {
                    unique = 0;
                    break;
                }
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////
// CROSSOVER

enum Crossover_method {
    SINGLE_POINT,
    TWO_POINT,
    UNIFORM
};

// single single-point crossover operation
__device__ void single_point(const uint8_t* parentA, const uint8_t* parentB,
                             uint8_t* offspringA, uint8_t* offspringB,
                             const uint16_t crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome

    if (gene_idx <= crossover_point) {
        offspringA[gene_idx] = parentA[gene_idx];
        offspringB[gene_idx] = parentB[gene_idx];
    }
    else {
        offspringA[gene_idx] = parentB[gene_idx];
        offspringB[gene_idx] = parentA[gene_idx];
    }

}

// single two-point crossover operation
__device__ void two_point(const uint8_t* parentA, const uint8_t* parentB,
                          uint8_t* offspringA, uint8_t* offspringB,
                          const int first_crossover_point, const int second_crossover_point) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome

    if (gene_idx <= first_crossover_point || gene_idx > second_crossover_point) {
        offspringA[gene_idx] = parentA[gene_idx];
        offspringB[gene_idx] = parentB[gene_idx];
    }
    else {
        offspringA[gene_idx] = parentB[gene_idx];
        offspringB[gene_idx] = parentA[gene_idx];
    }
}

// single uniform crossover operation
__device__ void uniform(const uint8_t* parentA, const uint8_t* parentB,
                        uint8_t* offspringA, uint8_t* offspringB,
                        const bool crossover_mask) {

    int gene_idx = threadIdx.x;   // index of a gene in a parent chromosome

    if (crossover_mask) {
        offspringA[gene_idx] = parentB[gene_idx];
        offspringB[gene_idx] = parentA[gene_idx];
    }
    else {
        offspringA[gene_idx] = parentA[gene_idx];
        offspringB[gene_idx] = parentB[gene_idx];
    }
}

// choosing parent pairs and calling a chosen crossover method 
__global__ void perform_crossover(uint8_t* selected_pool, uint8_t* offsprings,
                                  const uint16_t number_of_offsprings, const Crossover_method method,
                                  const uint16_t pool_size, const uint16_t chromosome_length,
                                  curandState* state, const uint16_t* combinations, uint16_t* random_ids) {

    // calculating indices
    int algorithm_idx = blockIdx.x;
    int block_population_start_idx = algorithm_idx * pool_size * chromosome_length;
    int gene_idx = threadIdx.x + block_population_start_idx;

    int block_offsprings_start = algorithm_idx * number_of_offsprings * chromosome_length;

    // nr of crossovers (nr of pairs to pick)
    const int number_of_crossovers = number_of_offsprings / 2;

    // generate array of random numbers (indices of pairs in combinations array), no repetitions
    uint16_t max = pool_size * (pool_size - 1) / 2 - 1; //nr of possible combinations (-1 for indexing)
    if (threadIdx.x == 0) {
        //generate only once for the entire block
        generate_random_unique_array(state, random_ids + block_population_start_idx, number_of_crossovers, 0, max);
    }
    __syncthreads(); //synchronize so all the threads see the generated values
   
    //iterating over parent pairs
    for (int pair_id = 0; pair_id < number_of_crossovers; pair_id++) {

        //get the indices of parents (based one previously randomly generated values)
        uint16_t parentA_idx = combinations[2 * random_ids[block_population_start_idx + pair_id]];
        uint16_t parentB_idx = combinations[2 * random_ids[block_population_start_idx + pair_id] + 1];

        //pick parents from the mating pool 
        uint8_t* parentA = selected_pool + chromosome_length * parentA_idx + block_population_start_idx;
        uint8_t* parentB = selected_pool + chromosome_length * parentB_idx + block_population_start_idx;

        //pick offsprings 
        uint8_t* offspringA = offsprings + chromosome_length * 2 * pair_id + block_offsprings_start;
        uint8_t* offspringB = offsprings + chromosome_length * (2 * pair_id + 1) + block_offsprings_start;


        if (method == Crossover_method::SINGLE_POINT) {
            __shared__ uint16_t crossover_point;
            if (threadIdx.x == 0) {
                float random = curand_uniform(&state[algorithm_idx]) * (chromosome_length - 1 + 0.999999);
                crossover_point = (int)truncf(random);
            }
            __syncthreads();
            single_point(parentA, parentB, offspringA, offspringB, crossover_point);
        }
        else if (method == Crossover_method::TWO_POINT) {
            __shared__ uint16_t crossover_points[2];
            if (threadIdx.x == 0) {
                generate_random_unique_array(state, crossover_points, 2, 0, chromosome_length-1);
            }
            __syncthreads();
            if (crossover_points[0] < crossover_points[1]) {
                two_point(parentA, parentB, offspringA, offspringB, crossover_points[0], crossover_points[1]);
            }
            else {
                two_point(parentA, parentB, offspringA, offspringB, crossover_points[1], crossover_points[0]);
            }
        }
        else {
            bool crossover_mask;
            if (threadIdx.x < chromosome_length) {
                crossover_mask = (bool)truncf(curand_uniform(&state[blockIdx.x * blockDim.x + threadIdx.x]) * (1 + 0.999999));
            }
            uniform(parentA, parentB, offspringA, offspringB, crossover_mask);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////
// MUTATION

__global__ void replacement(uint8_t* population, const float mutationPercentage, const uint16_t chromosomeLength, curandState* state) {
    
    int chromosomeId = blockIdx.x;
    const int range{ 255 };
    uint16_t numOfGenesToMutation = (uint16_t)(ceil(mutationPercentage * chromosomeLength));
    extern __shared__ uint16_t idToReplace[];
    extern __shared__ uint8_t valueTiReplace[];
    //uint16_t* idToReplace = new uint16_t[numOfGenesToMutation];
    //uint8_t* valueTiReplace = new uint8_t[numOfGenesToMutation];
    generate_random_unique_array(&state[chromosomeId], idToReplace, numOfGenesToMutation, chromosomeId * chromosomeLength, chromosomeId * chromosomeLength + chromosomeLength - 1);
    generate_random_unique_array(&state[chromosomeId], idToReplace, numOfGenesToMutation, 0, range);
    __syncthreads();

    for (int i = 0; i < numOfGenesToMutation; i++) {
        population[idToReplace[i]] = valueTiReplace[i];
    }

    //delete[] idToReplace;
    //delete[] valueTiReplace;
}

__global__ void randomSwap(uint8_t* population, const float mutationPercentage, const uint16_t chromosomeLength, curandState* state) {
    
    int chromosomeId = blockIdx.x;
    uint16_t numOfGenesToMutation = (uint16_t)(ceil(mutationPercentage * chromosomeLength)) * 2;
    uint16_t tempGene;
    extern __shared__ uint16_t idToSwap[];
    //uint16_t* idToSwap = new uint16_t[numOfGenesToMutation];
    generate_random_unique_array(&state[chromosomeId], idToSwap, numOfGenesToMutation, chromosomeId * chromosomeLength, chromosomeId * chromosomeLength + chromosomeLength - 1);
    __syncthreads();

    for (int i = 0; i < numOfGenesToMutation; i++) {
        tempGene = population[idToSwap[i]];
        population[idToSwap[i]] = population[idToSwap[numOfGenesToMutation - 1 - i]];
        population[idToSwap[numOfGenesToMutation - 1 - i]] = tempGene;
    }
    
    //delete[] idToSwap;
}

__global__ void adjecentSwap(uint8_t* population, const float mutationPercentage, const uint16_t chromosomeLength, curandState* state) {
    int chromosomeId = blockIdx.x;
    uint16_t numOfGenesToMutation = (uint16_t)(ceil(mutationPercentage * chromosomeLength)) * 2;
    uint16_t swapGene;
    extern __shared__ uint16_t idToSwap[];
    //uint16_t* idToSwap = new uint16_t[numOfGenesToMutation];
    generate_random_unique_array(&state[chromosomeId], idToSwap, numOfGenesToMutation, chromosomeId * chromosomeLength + 1, chromosomeId * chromosomeLength + chromosomeLength - 1);
    __syncthreads();

    for (int i = 0; i < numOfGenesToMutation; i++) {
        swapGene = population[idToSwap[i] - 1];
        population[idToSwap[i] - 1] = population[idToSwap[i]];
        population[idToSwap[i]] = swapGene;
    }

    //delete[] idToSwap;
}

void performMutation(uint8_t* population, const float mutationPercentage, const uint16_t chromosomeLength, uint16_t number_of_blocks, void(*mutationFuncPtr)(uint8_t*, const float, const uint16_t, curandState*), curandState* state) {

    mutationFuncPtr <<< number_of_blocks, 1 >>> (population, mutationPercentage, chromosomeLength, state);
}

///////////////////////////////////////////////////////////////////////////////////////

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


__host__ void generate_all_idx_combinations(uint16_t* pairs, int max_idx) {

    int counter = 0;
    for (uint16_t i = 0; i <= max_idx; i++) {
        for (uint16_t j = i + 1; j <= max_idx; j++) {
            pairs[2 * counter] = i;
            pairs[2 * counter + 1] = j;
            counter++;
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

__global__ void sum_individual_chromosomes(uint8_t* population, float* chromosomes_sums) {
    __shared__ float sum_result;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sum_result = 0;
    atomicAdd(&sum_result, static_cast<float>(population[idx]));
    __syncthreads();
    chromosomes_sums[blockIdx.x] = sum_result;
}

__global__ void sum_individual_chromosomes(float* population, float* chromosomes_sums) {
    __shared__ float sum_result;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    sum_result = 0;
    atomicAdd(&sum_result, population[idx]);
    __syncthreads();
    chromosomes_sums[blockIdx.x] = sum_result;
}

__global__ void abs_subtract_individual_chromosomes(uint8_t* base_chromosome, uint8_t* population, float* chromosomes_differences) {
    extern __shared__ float subtraction_result[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    subtraction_result[threadIdx.x] = static_cast<float>(base_chromosome[threadIdx.x] - population[idx]);
    __syncthreads();
    chromosomes_differences[idx] = abs(subtraction_result[threadIdx.x]);
}

__global__ void subtract_individual_chromosomes(float* base_chromosome, float* population, float* chromosomes_differences) {
    extern __shared__ float subtraction_result[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    subtraction_result[threadIdx.x] = static_cast<float>(base_chromosome[threadIdx.x] - population[idx]);
    __syncthreads();
    chromosomes_differences[idx] = subtraction_result[threadIdx.x];
}

__global__ void divide_vector_element_wise(float* cuda_vector, float denominator) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    cuda_vector[idx] = cuda_vector[idx] / denominator;
    __syncthreads();
}

void fitness_function(
    uint8_t* base_chromosome,
    uint16_t chromosome_size,
    uint8_t* population,
    uint16_t multiple_population_size,
    uint16_t number_of_blocks,
    uint16_t number_of_threads,
    float* population_fitness)
{

    float* d_partial_results; float* d_base_chromosome_sum;
    CUDA_CALL_V(cudaMalloc(&d_partial_results, sizeof(float) * multiple_population_size));
    CUDA_CALL_V(cudaMalloc(&d_base_chromosome_sum, sizeof(float) * number_of_blocks));

    abs_subtract_individual_chromosomes <<< number_of_blocks, number_of_threads, sizeof(float)* number_of_threads >>> (base_chromosome, population, d_partial_results);

    sum_individual_chromosomes <<< number_of_blocks, number_of_threads >>> (d_partial_results, population_fitness);

    divide_vector_element_wise <<< number_of_blocks, 1 >>> (population_fitness, chromosome_size);

    sum_individual_chromosomes <<< 1, number_of_threads >>> (base_chromosome, d_base_chromosome_sum);

    subtract_individual_chromosomes <<< number_of_blocks, 1, sizeof(float) >>> (d_base_chromosome_sum, population_fitness, population_fitness);

    cudaFree(d_partial_results);
    cudaFree(d_base_chromosome_sum);
}

__global__ void sort_population_by_indicies(uint8_t* population, int* indicies) {
    extern __shared__ uint8_t temp_chromosome[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int targetIdx = indicies[idx];
    temp_chromosome[threadIdx.x] = population[targetIdx];
    __syncthreads();
    population[idx] = temp_chromosome[threadIdx.x];
}

void sort_population_by_fitness(
    uint8_t* population,
    float* population_fitness,
    uint16_t population_fit_size,
    uint16_t population_size,
    uint16_t number_of_blocks,
    uint16_t number_of_threads,
    uint16_t multiple_population_size)
{
    auto population_fit_th = thrust::device_ptr<float>(population_fitness);

    thrust::device_vector<int> indicies(population_fit_size);
    for (int i = 0; i < population_fit_size; ++i) {
        indicies[i] = i;
    }

    for (int i = 0; i < population_fit_size; ++i) {
        auto fit_it = population_fit_th + i * population_size;
        auto idx_it = indicies.begin() + i * population_size;
        thrust::stable_sort_by_key(thrust::device, fit_it, fit_it + population_size, idx_it, thrust::greater<float>());
    }

    thrust::device_vector<int> extended_indicies(multiple_population_size);
    for (std::size_t i = 0; i < population_fit_size; ++i) {
        for (std::size_t j = 0; j < number_of_threads; ++j) {
            extended_indicies[i * number_of_threads + j] = indicies[i] * number_of_threads + j;
        }
    }

    auto* indicies_dptr = thrust::raw_pointer_cast(&extended_indicies[0]);
    sort_population_by_indicies <<< number_of_blocks, number_of_threads, sizeof(uint8_t)* number_of_threads >>> (population, indicies_dptr);
}

void ranking(
    uint8_t* population,
    float* population_fitness,
    uint8_t* mating_pool,
    uint16_t number_of_parents,
    uint16_t number_of_algorithms,
    uint16_t chromosome_size,
    uint16_t population_size,
    uint16_t number_of_blocks,
    uint16_t multiple_population_size)
{
    sort_population_by_fitness(population, population_fitness, number_of_blocks, population_size, number_of_blocks, chromosome_size, multiple_population_size);

    auto population_th = thrust::device_ptr<uint8_t>(population);
    auto mating_pool_th = thrust::device_ptr<uint8_t>(mating_pool);

    for (std::size_t i = 0; i < number_of_algorithms; ++i) {
        auto source_start = population_th + i * population_size * chromosome_size;
        int offset_tgt = i * number_of_parents * chromosome_size;
        int copy_range = number_of_parents * chromosome_size;
        thrust::copy(thrust::device, source_start, source_start + copy_range, mating_pool_th + offset_tgt);
    }
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
    Crossover_method crossover_method = Crossover_method::UNIFORM;

    // mutation parameters
    float mutation_percentage = 0.01;
    void(*mutationMethodPtr)(uint8_t*, const float, const uint16_t, curandState*) = replacement;

    // termination condition
    float epsilon = 1.0E-12F;
    uint16_t terminate_after = 500;

    uint16_t nr_of_parallel_algorithms = 3;
    

    uint16_t chromosome_size = 5;
    uint16_t population_size = 4;
    uint16_t number_of_threads = chromosome_size;
    uint16_t number_of_blocks = population_size * nr_of_parallel_algorithms;
    uint16_t multiple_population_size = population_size * nr_of_parallel_algorithms * chromosome_size;
    uint16_t number_of_crossovers = number_of_offsprings / 2;
    uint16_t number_of_combinations = number_of_parents * (number_of_parents - 1) / 2;

    // HOST variables allocation
    uint8_t* h_mpopulation = new uint8_t[multiple_population_size]; // host population
    float* h_mpopulation_fitness = new float[number_of_blocks]; // fitness for host population
    uint8_t* h_bchromosome = new uint8_t[chromosome_size]; // base chromosome
    uint16_t* h_combinations = new uint16_t[2 * number_of_combinations]; //all possible parents combinations
    uint8_t* h_mating_pool = new uint8_t[number_of_parents * chromosome_size * nr_of_parallel_algorithms];

    //fill the combinations array
    generate_all_idx_combinations(h_combinations, number_of_parents - 1);

    // DEVICE variables allocation
    uint8_t* d_mpopulation;
    CUDA_CALL(cudaMalloc(&d_mpopulation, sizeof(uint8_t) * multiple_population_size));

    uint8_t* d_bchromosome;
    CUDA_CALL(cudaMalloc(&d_bchromosome, sizeof(uint8_t) * chromosome_size));

    float* d_mpopulation_fitness;
    CUDA_CALL(cudaMalloc(&d_mpopulation_fitness, sizeof(float) * number_of_blocks));

    uint8_t* d_mating_pool;
    CUDA_CALL(cudaMalloc(&d_mating_pool, sizeof(uint8_t) * number_of_parents * chromosome_size * nr_of_parallel_algorithms));

    uint16_t* d_combinations;
    CUDA_CALL(cudaMalloc(&d_combinations, 2 * number_of_combinations * sizeof(uint16_t)));

    uint16_t* d_random_pair_ids;
    CUDA_CALL(cudaMalloc(&d_random_pair_ids, nr_of_parallel_algorithms * number_of_crossovers * sizeof(uint16_t)));

    // Host -> Device
    CUDA_CALL(cudaMemcpy(d_mpopulation, h_mpopulation, sizeof(uint8_t) * multiple_population_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bchromosome, h_bchromosome, sizeof(uint8_t) * chromosome_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_combinations, h_combinations, 2 * number_of_combinations * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // DEVICE curandState initialization
    curandState* devStates;
    CUDA_CALL(cudaMalloc((void**)&devStates, multiple_population_size * sizeof(curandState)));

    // ------------------------------------------ genetic algorithm start ------------------------------------------
    setup_kernel_multi_blocks <<< number_of_blocks, number_of_threads >>> (devStates, 7);
    populationInit_multi_blocks <<< number_of_blocks, number_of_threads >>> (d_mpopulation, devStates);


    //-----------loop start----------

    //fitness_function(d_bchromosome, chromosome_size, d_mpopulation, multiple_population_size, number_of_blocks, number_of_threads, d_mpopulation_fitness);

    //ranking(d_mpopulation, d_mpopulation_fitness, d_mating_pool, number_of_parents, nr_of_parallel_algorithms, chromosome_size, population_size, number_of_blocks, multiple_population_size);

    //perform_crossover <<<nr_of_parallel_algorithms, number_of_threads >>> (<POPULATION_AFTER_SELECTION>, d_mpopulation, number_of_offsprings,
    //                                                                        crossover_method, number_of_parents, chromosome_size, devStates,
    //                                                                        d_combinations, d_random_pair_ids);

    //performMutation(d_mpopulation, mutation_percentage, chromosome_size, number_of_blocks, mutationMethodPtr, devStates);

    //-----------loop end----------

    // ------------------------------------------ genetic algorithm end ------------------------------------------

    // DEVICE -> HOST

    CUDA_CALL(cudaMemcpy(h_mpopulation, d_mpopulation, sizeof(uint8_t) * multiple_population_size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(h_mpopulation_fitness, d_mpopulation_fitness, sizeof(float) * number_of_blocks, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_mating_pool, d_mating_pool, sizeof(uint8_t) * number_of_parents * chromosome_size * nr_of_parallel_algorithms, cudaMemcpyDeviceToHost));

    // show results
    for (std::size_t i = 0; i < multiple_population_size; ++i) {
        std::cout << static_cast<int>(h_mpopulation[i]) << " ";
    }

    // free all device memory
    cudaFree(d_mpopulation);
    cudaFree(d_mpopulation_fitness);
    cudaFree(d_bchromosome);
    cudaFree(&devStates);
    cudaFree(d_combinations);
    cudaFree(d_random_pair_ids);
    cudaFree(d_mating_pool);
    // free all host memory
    delete[] h_mpopulation;
    delete[] h_mpopulation_fitness;
    delete[] h_bchromosome;
    delete[] h_combinations;
    delete[] h_mating_pool;

    ////Display the result image
    //cv::Mat resultImage = grayArrayToCvMat(image, grayArray);
    //delete[] grayArray;
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
