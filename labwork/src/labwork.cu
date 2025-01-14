#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <math.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    omp_set_num_threads(15);
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device name: %s, device core clock %d, id: %d\n",prop.name,prop.clockRate,i);
        // something more here
    }

}

__global__ void rgb2grayCUDA(uchar3 *input, uchar3 *output) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
        output[tid].z = output[tid].y = output[tid].x;
    }

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height * 3;  // let's do it 100 times, otherwise it's too fast!
    uchar3 *devInput;
    uchar3 *devGray;
    outputImage = static_cast<char *>(malloc(pixelCount));
    // Allocate CUDA memory
    // Copy CUDA Memory from CPU to GPU
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 512;
    int numBlock = pixelCount / (blockSize*3);
    rgb2grayCUDA<<<numBlock, blockSize>>>(devInput, devGray);
    
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount, cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

__global__ void rgb2grayCUDABlock(uchar3 *input, uchar3 *output) {
        int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
        int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
        int tid = tid_x + blockDim.x * gridDim.x * tid_y;
        output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
        output[tid].z = output[tid].y = output[tid].x;
    }

void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height * 3; 
    uchar3 *devInput;
    uchar3 *devGray;
    outputImage = static_cast<char *>(malloc(pixelCount));
    // Allocate CUDA memory
    // Copy CUDA Memory from CPU to GPU
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3(ceil(1.0*inputImage->width/32),ceil(1.0*inputImage->height)/32);
    rgb2grayCUDABlock<<<gridSize, blockSize>>>(devInput, devGray);
    
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount, cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork5_CPU() {
    outputImage = static_cast<char *>(malloc(inputImage->width * inputImage->height*3));
    int gaussian[7][7] = {{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,13,59,97,59,13,1},{2,22,97,159,97,22,2},{1,13,59,97,59,13,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};
    int total = 1003;
    for (int row = 3; row < inputImage->height-3; row++) {
        for (int col = 3; col < inputImage->width-3; col++) {
            int sumR = 0,sumG = 0,sumB = 0;
            for(int i = -3; i<=3 ; ++i)
                for(int j = -3; j<=3 ; ++j){
                    int pos = (row + i) * inputImage->width + (col + j);
                    sumR += inputImage->buffer[pos*3] * gaussian[i+3][j+3];
                    sumG += inputImage->buffer[pos*3 + 1] * gaussian[i+3][j+3];
                    sumB += inputImage->buffer[pos*3 + 2] * gaussian[i+3][j+3];
                }
            int current_pos = row * inputImage->width + col;
            outputImage[current_pos * 3] = sumR/total;
            outputImage[current_pos * 3 + 1] = sumG/total;
            outputImage[current_pos * 3 + 2] = sumB/total;
        }
    }
}

__global__ void GaussianCUDABlock(uchar3 *input, uchar3 *output) {
    int gaussian[7][7] = {{0,0,1,2,1,0,0},{0,3,13,22,13,3,0},{1,13,59,97,59,13,1},{2,22,97,159,97,22,2},{1,13,59,97,59,13,1},{0,3,13,22,13,3,0},{0,0,1,2,1,0,0}};
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tid_x + blockDim.x * gridDim.x * tid_y;
    __shared__ int shared_gaussian[7][7];
    if (tid_x < 7 && tid_y < 7)
        shared_gaussian[tid_x][tid_y] = gaussian[tid_x][tid_y];
    __syncthreads();
    int total = 1003;
    int sumR = 0,sumG =0, sumB = 0;
    for(int i = -3; i<=3 ; ++i)
        for(int j = -3; j<=3 ; ++j){
            int cell_tid = tid + i * blockDim.x * gridDim.x + j ;
            sumR += input[cell_tid].x * shared_gaussian[i+3][j+3];
            sumG += input[cell_tid].y * shared_gaussian[i+3][j+3];
            sumB += input[cell_tid].z * shared_gaussian[i+3][j+3];
        }
    output[tid].x = sumR/total;
    output[tid].y = sumG/total;
    output[tid].z = sumB/total;
}
void Labwork::labwork5_GPU() {
    int pixelCount = inputImage->width * inputImage->height * 3; 
    uchar3 *devInput;
    uchar3 *devGray;
    outputImage = static_cast<char *>(malloc(pixelCount));
    // Allocate CUDA memory
    // Copy CUDA Memory from CPU to GPU
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3(ceil(1.0*inputImage->width/32),ceil(1.0*inputImage->height)/32);
    GaussianCUDABlock<<<gridSize, blockSize>>>(devInput, devGray);
    
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount, cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}
__device__ int binary(int a){
    return a<155 ? 0:255;
}
__global__ void graybinaryCUDABlock(uchar3 *input, uchar3 *output) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = tid_x + blockDim.x * gridDim.x * tid_y;
    output[tid].x = binary(input[tid].x);
    output[tid].z = output[tid].y = output[tid].x;
}
void Labwork::labwork6_GPU() {
    int pixelCount = inputImage->width * inputImage->height * 3; 
    uchar3 *devInput;
    uchar3 *devGray;
    outputImage = static_cast<char *>(malloc(pixelCount));
    // Allocate CUDA memory
    // Copy CUDA Memory from CPU to GPU
    cudaMalloc(&devInput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devGray, pixelCount * sizeof(uchar3));
    cudaMemcpy(devInput, inputImage->buffer, pixelCount, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3(ceil(1.0*inputImage->width/32),ceil(1.0*inputImage->height)/32);
    graybinaryCUDABlock<<<gridSize, blockSize>>>(devInput, devGray);
    
    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devGray, pixelCount, cudaMemcpyDeviceToHost);
    // Cleaning
    cudaFree(devInput);
    cudaFree(devGray);
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























