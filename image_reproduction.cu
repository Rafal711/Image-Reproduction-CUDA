#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

cv::Mat* imgRGB2chromosome(cv::Mat image) {
    cv::Mat* reshaped_img = new cv::Mat();
    *reshaped_img = image.reshape(1, 1);
    return reshaped_img;
}

cv::Mat* reproduce_image(cv::Mat original_image) {
    cv::Mat* chromosome_base = imgRGB2chromosome(original_image);

    std::cout << "Original size: " << original_image.size << std::endl;
    std::cout << "Reshaped size: " << chromosome_base->size << std::endl;

    cv::Mat result;
    return &result;

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

    cv::Mat* result_image = reproduce_image(img);


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