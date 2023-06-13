#include <iostream>
#include <string>
#include <random>
#include <string.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <assert.h>
#include <algorithm>

// Constants
const int threadCount = 6;
const int maxEpochs = 500;
const bool white = 0;
const bool black = 1;

// Hyperparameters
const double beta1 = 0.9;
const double beta2 = 0.999;
const double evalWeight = 0.9;
const int batchSize = 16384;

double lr = 0.001;
const int lrDecayRate = 5;
const double lrDecay = 0.875;

// Network sizes
const int kingBucketCount = 6;
const int singleBucketSize = 768;
const int inputHalf = singleBucketSize * kingBucketCount;
const int hiddenHalf = 384;

const int bucketId[64] = {
    0, 0, 1, 1, 2, 2, 3, 3,
    0, 0, 1, 1, 2, 2, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5
};

// Weights and bias
double W1[inputHalf * hiddenHalf];
double B1[hiddenHalf];
double W2[hiddenHalf * 2];
double B2;

// Math stuffs (note that we scale down our evaluation by 400. This is very
// useful since now plain sigmoid(eval) will give us WDL

const double evalScale = 400; 
const double Q1 = 255;
const double Q2 = 64;

// Gradients (each thread calculates gradients and we combine)
struct gradientStorage{
    double W1[inputHalf * hiddenHalf];
    double B1[hiddenHalf];
    double W2[hiddenHalf * 2];
    double B2;
};
gradientStorage tGrad[threadCount];
double errorAccumulation[threadCount];

// Input output stuff
const int iterationReport = 200;
std::string batchData[batchSize];
int sessionId;
int exportNumber;

inline long long getTime(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline void initSessionId(){
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    sessionId = rng() >> 1;
}

inline void importCheckpoint(std::string path){
    // Note that we may only import checkpoints (networks with double weights)
    std::ifstream inputFile(path, std::ios::binary);
    inputFile.read(reinterpret_cast<char*>(&W1), sizeof(W1));
    inputFile.read(reinterpret_cast<char*>(&B1), sizeof(B1));
    inputFile.read(reinterpret_cast<char*>(&W2), sizeof(W2));
    inputFile.read(reinterpret_cast<char*>(&B2), sizeof(B2));
}

inline void exportNetworkCheckpoint(){
    std::ofstream outFile("weights_" + std::to_string(sessionId) + "_" + std::to_string(exportNumber) + ".bin", std::ios::out | std::ios::binary);
    exportNumber++;

    outFile.write(reinterpret_cast<char*>(&W1), sizeof(W1));
    outFile.write(reinterpret_cast<char*>(&B1), sizeof(B1));
    outFile.write(reinterpret_cast<char*>(&W2), sizeof(W2));
    outFile.write(reinterpret_cast<char*>(&B2), sizeof(B2));
    outFile.close();
}

inline void exportNetworkQuantized(){
    std::ofstream outFile("weights_" + std::to_string(sessionId) + "_" + std::to_string(exportNumber) + "Q.bin", std::ios::out | std::ios::binary);
    exportNumber++;

    int16_t qW1[inputHalf * hiddenHalf];
    int16_t qB1[hiddenHalf];
    int16_t qW2[hiddenHalf * 2];
    int16_t qB2;
    
    // We need to convert these doubles to int16. Let's multiply W1 and B1 by Q1,
    // W2 by Q2, and B2 by Q1 * Q2 so that we can factor Q1 * Q2 out of everyone.
    // Then to get back the original value, we divide by Q1 * Q2

    for (int i = 0; i < inputHalf * hiddenHalf; i++){
        qW1[i] = round(W1[i] * Q1);
    }
    for (int i = 0; i < hiddenHalf; i++){
        qB1[i] = round(B1[i] * Q1);
    }
    for (int i = 0; i < hiddenHalf * 2; i++){
        qW2[i] = round(W2[i] * Q2);
    }
    qB2 = round(B2 * Q1 * Q2);

    outFile.write(reinterpret_cast<char*>(&qW1), sizeof(qW1));
    outFile.write(reinterpret_cast<char*>(&qB1), sizeof(qB1));
    outFile.write(reinterpret_cast<char*>(&qW2), sizeof(qW2));
    outFile.write(reinterpret_cast<char*>(&qB2), sizeof(qB2));
    outFile.close();
}

inline void initWithRandomWeights(){
    // Normal distribution constructor is (mean, standard deviation)
    // HE initialization is (mean = 0, std = sqrt(2 / indegree))
    // On average, ~20 input neurons are set to 1 so we say that is our indegree

    std::default_random_engine generator(69);
    std::normal_distribution<double> random1(0, sqrt(2.0 / 20));
    std::normal_distribution<double> random2(0, sqrt(2.0 / (hiddenHalf * 2)));
    std::normal_distribution<double> random3(0, 0.01);

    for (int i = 0; i < inputHalf * hiddenHalf; i++){
        W1[i] = random1(generator);
    }
    for (int i = 0; i < hiddenHalf * 2; i++){
        W2[i] = random2(generator);
    }
    for (int i = 0; i < hiddenHalf; i++){
        B1[i] = random3(generator);
    }
    B2 = random3(generator);
}

inline int inputIndex(bool perspective, bool col, int piece, int sq, int kb){
    return (col == perspective ? 0 : 384) 
           + 64 * piece
           + (perspective == white ? sq : (sq ^ 56))
           + kb * singleBucketSize;
}

inline double crelu(double value){
    return std::clamp(value, 0.0, 1.0);
}

inline double dCrelu(double value){
    return (value > 0 and value < 1);
}

inline double sigmoid(double value){
    return 1 / (1 + exp(-value));
}

inline double dSigmoid(double value){
    return sigmoid(value) * (1.0 - sigmoid(value));
}

inline double error(double output, double eval, double wdl){
    double expected = evalWeight * sigmoid(eval) + (1 - evalWeight) * wdl;
    return pow(sigmoid(output) - expected, 2);
}

inline double dError(double output, double eval, double wdl){
    double expected = evalWeight * sigmoid(eval) + (1 - evalWeight) * wdl;
    return 2 * (sigmoid(output) - expected) * dSigmoid(output);
}

inline void extractInputFeatures(std::string fen, std::vector<int> (&featureIdx)[2], double &eval, double &wdl){
    // Fen is in the form: (<fen> | <eval relative to white> | <wdl relative to white>)
    // Pay attention to the fact that it is relative to white

    int whiteBucket = -1;
    int blackBucket = -1;
    std::vector<std::tuple<int, bool, int>> pieces;

    for (int i = 0, sq = 56;; i++){
        if (fen[i] == ' '){
            break;
        }
        else if (fen[i] == '/'){
            sq -= 16;
        }
        else if (isdigit(fen[i])){
            sq += fen[i] - '0';
        }
        else{
            bool col = islower(fen[i]);
            int piece = -1;

            if (tolower(fen[i]) == 'p'){
                piece = 0;
            }
            else if (tolower(fen[i]) == 'n'){
                piece = 1;
            }
            else if (tolower(fen[i]) == 'b'){
                 piece = 2;
            }
            else if (tolower(fen[i]) == 'r'){
                piece = 3;
            }
            else if (tolower(fen[i]) == 'q'){
                piece = 4;
            }
            else if (tolower(fen[i]) == 'k'){
                piece = 5;
            }
            else{ 
                assert(false);
            }
            
            // If it's a king, we update the buckets (flip king position for black)
            if (piece == 5){
                if (col == white){
                    whiteBucket = bucketId[sq];
                }
                else{ 
                    blackBucket = bucketId[sq ^ 56];
                }
            }
            // Add pieces to list
            pieces.push_back({piece, col, sq});
            sq++;
        }
    }
    for (auto [piece, col, sq] : pieces){
        featureIdx[white].push_back(inputIndex(white, col, piece, sq, whiteBucket));
        featureIdx[black].push_back(inputIndex(black, col, piece, sq, blackBucket));
    }

    // Note that we scale eval down by ~400 and we train the NN to output an already
    // scaled down eval. The important property is that taking a plain sigmoid
    // of the scaled down eval will directly get us WDL

    eval = std::stod(fen.substr(fen.find("|") + 2, fen.size() - fen.find("|") - 8));
    wdl = std::stod(fen.substr(fen.size() - 3));
    eval /= evalScale;

    // Force stm to be the top accumulator by swapping the 2 sides if it is
    // black to move (since white is always the top accumulator) 

    if (fen[fen.find(" ") + 1] == 'b'){
        std::swap(featureIdx[white], featureIdx[black]);

        // Make eval and wdl relative to stm
        eval *= -1;
        wdl = 1 - wdl;
    }
}

inline double forwardProp(std::vector<int> (&featureIdx)[2], double (&accumulator)[hiddenHalf * 2]){
    // Remember the B2 bias when we forward prop
    double output = B2;

    // Initialize with B1
    for (int i = 0; i < hiddenHalf; i++){
        accumulator[i] = accumulator[i + hiddenHalf] = B1[i];
    }

    // This part is the bottleneck and should be as optimized as possible
    for (int idx : featureIdx[0]){
        int start = idx * hiddenHalf;
        for (int i = 0; i < hiddenHalf; i++){
            accumulator[i] += W1[start + i];
        }
    }
    for (int idx : featureIdx[1]){
        int start = idx * hiddenHalf;
        for (int i = 0; i < hiddenHalf; i++){
            accumulator[i + hiddenHalf] += W1[start + i];
        }
    }

    // Make the inference
    for (int i = 0; i < hiddenHalf * 2; i++){
        output += crelu(accumulator[i]) * W2[i];
    }
    return output;
}

inline void backProp(gradientStorage &grad, std::vector<int> (&featureIdx)[2], double (&accumulator)[hiddenHalf * 2], double eval, double wdl, double output){
    // Other gradient variables
    double gradZ2;
    double gradA1[hiddenHalf * 2];
    double gradZ1[hiddenHalf * 2];

    // Calculate dZ2
    gradZ2 = dError(output, eval, wdl);

    // Calculate dB2
    grad.B2 += gradZ2;
    
    // Calculate dW2, dA1, dZ1
    for (int i = 0; i < hiddenHalf * 2; i++){
        grad.W2[i] += gradZ2 * crelu(accumulator[i]);
        gradA1[i] = W2[i] * gradZ2;
        gradZ1[i] = gradA1[i] * dCrelu(accumulator[i]);
    }
    
    // Calculate dB1 (note that this single bias has influence on 2 nodes)
    for (int i = 0; i < hiddenHalf; i++){
        grad.B1[i] += gradZ1[i] + gradZ1[i + hiddenHalf];
    }

    // Calculate dW1 (note that a single weight can influence 2 nodes)
    // This part is the bottleneck and should be as optimized as possible

    for (int idx : featureIdx[0]){
        int start = idx * hiddenHalf;
        for (int i = 0; i < hiddenHalf; i++){
            grad.W1[start + i] += gradZ1[i];
        }
    }
    for (int idx : featureIdx[1]){
        int start = idx * hiddenHalf;
        for (int i = 0; i < hiddenHalf; i++){
            grad.W1[start + i] += gradZ1[i + hiddenHalf];
        }
    }
}

inline void proccessBatch(int td){
    for (int batchIdx = td; batchIdx < batchSize; batchIdx += threadCount){
        std::string fen = batchData[batchIdx];
        std::vector<int> featureIdx[2];
        double accumulator[hiddenHalf * 2];
        double eval;
        double wdl;

        // Initialize
        extractInputFeatures(fen, featureIdx, eval, wdl);

        // Forward prop
        double output = forwardProp(featureIdx, accumulator);
        errorAccumulation[td] += error(output, eval, wdl);

        // Back prop
        backProp(tGrad[td], featureIdx, accumulator, eval, wdl, output);
    }
}

inline void adamUpdate(double &target, double &momentum, double &rms, double grad){
    // Apply update
    momentum = momentum * beta1 + (1.0 - beta1) * grad;
    rms = rms * beta2 + (1.0 - beta2) * grad * grad;

    // Apply gradient
    target -= lr * momentum / (sqrt(rms) + 1e-8);    
}

inline void train(std::string dataPath, std::string reportLossPath){
    // Intialize
    std::ofstream outputLoss(reportLossPath);
    gradientStorage momentum = {};
    gradientStorage rms = {};

    long long totalIterations = 0;
    long long startTime = getTime();
    
    for (int epoch = 1; epoch <= maxEpochs; epoch++){
        std::ifstream inputFile(dataPath);
        double epochErrorAccum = 0;
        long long batchIterations = 0;
        long long epochStartTime = getTime();

        while (true){
            // We first read the raw data and stop if there is not enough for our batch or there's a new line
            bool stop = false;

            for (int i = 0; i < batchSize; i++){
                if (!std::getline(inputFile, batchData[i]) 
                    or batchData[i].empty() 
                    or !(std::count(batchData[i].begin(), batchData[i].end(), '|') == 2 and isdigit(batchData[i].back())))
                {
                    stop = true;
                    break;
                }
            }
            if (stop){
                break;
            }

            // Update iterations
            batchIterations++;
            totalIterations++;

            // Reset total error and gradients
            for (int td = 0; td < threadCount; td++){
                errorAccumulation[td] = 0;
                tGrad[td] = {};
            }

            // Now we go to our multithreading step where we split the data into chunks,
            // extract features from the data and perform forward and back prop.

            std::thread threads[threadCount];
            
            for (int td = 0; td < threadCount; td++){
                threads[td] = std::thread(proccessBatch, td);
            }
            for (int td = 0; td < threadCount; td++){
                threads[td].join();
            }
            
            // Accumulate error
            double batchError = 0;
            for (int td = 0; td < threadCount; td++){
                batchError += errorAccumulation[td];
            }
            epochErrorAccum += batchError;

            // Apply gradients to W1
            for (int i = 0; i < inputHalf * hiddenHalf; i++){
                double grad = 0;
                for (int td = 0; td < threadCount; td++){
                    grad += tGrad[td].W1[i];
                }
                adamUpdate(W1[i], momentum.W1[i], rms.W1[i], grad);
            }

            // Apply gradients to B1
            for (int i = 0; i < hiddenHalf; i++){
                double grad = 0;
                for (int td = 0; td < threadCount; td++){
                    grad += tGrad[td].B1[i];
                }
                adamUpdate(B1[i], momentum.B1[i], rms.B1[i], grad);
            }

            // Apply gradients to W2
            for (int i = 0; i < hiddenHalf * 2; i++){
                double grad = 0;
                for (int td = 0; td < threadCount; td++){
                    grad += tGrad[td].W2[i];
                }
                adamUpdate(W2[i], momentum.W2[i], rms.W2[i], grad);
            }

            // Apply gradients to B2
            {
                double grad = 0;
                for (int td = 0; td < threadCount; td++){
                    grad += tGrad[td].B2;
                }
                adamUpdate(B2, momentum.B2, rms.B2, grad);
            }

            // Print info
            if (batchIterations % iterationReport == 0){
                std::cout<<"Batch: "                <<batchIterations
                         <<", Current Batch Error: "<<batchError / batchSize
                         <<", Epoch Average: "      <<epochErrorAccum / (batchSize * batchIterations)
                         <<", Time: "               <<(getTime() - startTime) / 1000.0
                         <<", Fens/s: "             <<(batchIterations * batchSize) / ((getTime() - epochStartTime) / 1000.0)
                         <<std::endl;
            }
        }
        inputFile.close();

        // Print info after each epoch
        std::cout<<"Finished Epoch: "<<epoch<<std::endl;
        outputLoss<<totalIterations<<" "<<epochErrorAccum / (batchSize * batchIterations)<<std::endl;

        // Export networks after each epoch
        exportNetworkCheckpoint();
        exportNetworkQuantized();

        // Decay learning rate
        if (epoch % lrDecayRate == 0){
            std::cout<<"Learning rate decayed from "<<lr<<" to "<<lr * lrDecay<<std::endl;
            lr *= lrDecay;
        }
    }
}

int main(){
    
    
    // Batch Iteration: 200, Average Batch Error: 0.0118741, Total Average: 0.0136062, Time: 10.877, Fens/s: 301370
    // Batch Iteration: 5000, Average Batch Error: 0.00769513, Total Average: 0.0092542, Time: 266.131, Fens/s: 307823

    initSessionId();
    initWithRandomWeights();
    train("C:\\Users\\allen\\Documents\\filteredData.plain", "loss.txt");
}