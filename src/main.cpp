#include "binpack/nnue_training_data_formats.h"
#include <iostream>
#include <string>
#include <random>
#include <string.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <assert.h>
#include <algorithm>
#include <numeric>

// Constants
const int THREAD_COUNT = 6;
const int DATA_LOAD_SZ = (1 << 20);
const int BATCH_SIZE = 16384;
const int EPOCH_SIZE = 98304000;
const int MAX_EPOCHS = 100000;
const int FEN_SKIP = 20; // (1 / FEN_SKIP) chance of using fen

// Adam
const double BETA_1 = 0.9;
const double BETA_2 = 0.999;

// Eval weight in loss and eval scale
const double EVAL_WEIGHT = 0.9;
const double EVAL_SCALE = 400; 

// Learning rate and cosine annealing
const double LR_BASE = 0.001;
const double LR_DECAY = 0.99692;
const int LR_TRANSITION = 350;

const double COSINE_MIN = 0.0001;
const double COSINE_MAX = 0.0004;
const double COSINE_INTERVAL_MULTIPLIER = 1.5;
const int COSINE_INTERVAL_BASE = 20;

// Network stuff
const int KING_BUCKET_COUNT = 10;
const int OUTPUT_WEIGHT_BUCKET_COUNT = 8;
const int SINGLE_BUCKET_SIZE = 768;
const int INPUT_HALF = SINGLE_BUCKET_SIZE * KING_BUCKET_COUNT;
const int HIDDEN_HALF = 512;

const int BUCKET_ID[64] = {
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9
};

// Quantization
const double Q1 = 256;
const double Q2 = 256;

// The model itself
struct NeuralNetwork{
    double W1[INPUT_HALF * HIDDEN_HALF];
    double B1[HIDDEN_HALF];
    double W2[OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2];
    double B2[OUTPUT_WEIGHT_BUCKET_COUNT];
};
NeuralNetwork model;
NeuralNetwork momentum;
NeuralNetwork rms;
NeuralNetwork threadGrad[THREAD_COUNT];

// Error
double errorAccumulation[THREAD_COUNT];

// Export 
const int iterationReport = 200;
int sessionId;
int exportCheckpointNumber;
int exportNetworkNumber;

// Data struct for storing sets of features
struct Data{
    int stmSz;
    int stmFeatures[32];
    
    int enemySz;
    int enemyFeatures[32];
    
    double score;
    double wdl;

    void reset(){
        stmSz = 0;
        enemySz = 0;
    }

    void addFeature(bool isTop, int feature){
        if (isTop){
            stmFeatures[stmSz++] = feature;
        }
        else{
            enemyFeatures[enemySz++] = feature;
        }
    }

    void setScore(double score_){
        score = score_;
    }

    void setWDL(double wdl_){
        wdl = wdl_;
    }

    int calculateOutputBucket(){
        return (stmSz - 2) / 4;
    }
};

// Data loader
namespace DataLoader{
    Data currentData[DATA_LOAD_SZ];
    Data nextData[DATA_LOAD_SZ];
    std::mt19937 rng(69);

    int permuteShuffle[DATA_LOAD_SZ];
    int position = 0;

    binpack::CompressedTrainingDataEntryReader reader("");
    std::string inp;
    
    std::thread readingThread;

    int featureIndex(chess::Color perspective, chess::Color col, chess::PieceType piece, chess::Square sq, chess::Square ksq){
        // It is expected that white = 0, black = 1, pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5
        return (col == perspective ? 0 : 384) 
            + 64 * static_cast<int>(piece)
            + (perspective == chess::Color::White ? static_cast<int>(sq) : (static_cast<int>(sq) ^ 56))
            + BUCKET_ID[static_cast<int>(ksq) ^ (perspective == chess::Color::Black ? 56 : 0)] * SINGLE_BUCKET_SIZE;
    }

    void loadNext(){
        int counter = 0;
        std::string arr[DATA_LOAD_SZ];

        while (counter < DATA_LOAD_SZ){
            // If we finished, go back to the beginning
            if (!reader.hasNext()){
                reader = binpack::CompressedTrainingDataEntryReader(inp);
            }

            // Get info
            auto entry = reader.next();
            auto &pos = entry.pos;
            auto pieces = pos.piecesBB();

            // Skip data if we filtered it out
            if (entry.score == 32002){
                continue;
            }

            // Probability of using is: (1 / FEN_SKIP)
            if (rng() % FEN_SKIP != 0){
                continue;
            }

            // Read and prepare entry
            auto &d = nextData[permuteShuffle[counter]];
            d.reset();
            
            for (auto persp : {chess::Color::White, chess::Color::Black}){
                auto ksq = pos.kingSquare(persp);

                for (auto sq : pieces){
                    auto p = pos.pieceAt(sq);
                    int feature = featureIndex(persp, p.color(), p.type(), sq, ksq);

                    // The top accumulator is for side to move
                    d.addFeature(persp == pos.sideToMove(), feature);
                }
            }

            // Note that score and result (-1/0/1) should be relative to the player to move and score should be on a centipawn scale
            d.setScore(entry.score / EVAL_SCALE);
            d.setWDL(entry.result == -1 ? 0.0 : entry.result == 0 ? 0.5 : 1.0);

            // Increment counter
            counter++;
        }
    }

    void advanceDataLoader(){
        position += BATCH_SIZE;

        if (position == DATA_LOAD_SZ){
            // Join thread that's reading nextData
            if (readingThread.joinable()){
                readingThread.join();
            }

            // Bring next data to current position
            std::swap(currentData, nextData);
            position = 0;

            // Begin a new thread to read nextData
            readingThread = std::thread(DataLoader::loadNext);
        }
    }

    void init(std::string inp_){
        inp = inp_;
        reader = binpack::CompressedTrainingDataEntryReader(inp);

        std::default_random_engine generator(69);
        std::iota(permuteShuffle, permuteShuffle + DATA_LOAD_SZ, 0);
        std::shuffle(permuteShuffle, permuteShuffle + DATA_LOAD_SZ, generator);

        loadNext();
        std::swap(currentData, nextData);
        loadNext();
    }
}

inline long long getTime(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
    double expected = EVAL_WEIGHT * sigmoid(eval) + (1 - EVAL_WEIGHT) * wdl;
    return pow(sigmoid(output) - expected, 2);
}

inline double dError(double output, double eval, double wdl){
    double expected = EVAL_WEIGHT * sigmoid(eval) + (1 - EVAL_WEIGHT) * wdl;
    return 2 * (sigmoid(output) - expected) * dSigmoid(output);
}

inline void adamUpdate(double grad, double lr, double &targetValue, double &momentumValue, double &rmsValue){
    // Apply update
    momentumValue = momentumValue * BETA_1 + (1.0 - BETA_1) * grad;
    rmsValue = rmsValue * BETA_2 + (1.0 - BETA_2) * grad * grad;

    // Apply gradient
    targetValue -= lr * momentumValue / (static_cast<double>(sqrt(rmsValue)) + 1e-8);    
}

void initSessionId(){
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    sessionId = rng() >> 1;
}

 void importCheckpoint(std::string path){
    // Note that we may only import checkpoints (networks with double weights)
    std::ifstream inputFile(path, std::ios::binary);
    inputFile.read(reinterpret_cast<char*>(&model.W1), sizeof(model.W1));
    inputFile.read(reinterpret_cast<char*>(&model.B1), sizeof(model.B1));
    inputFile.read(reinterpret_cast<char*>(&model.W2), sizeof(model.W2));
    inputFile.read(reinterpret_cast<char*>(&model.B2), sizeof(model.B2));
}

void exportNetworkCheckpoint(){
    std::ofstream outFile("checkpoint_" + std::to_string(sessionId) + "_" + std::to_string(exportCheckpointNumber) + ".bin", std::ios::out | std::ios::binary);
    exportCheckpointNumber++;

    outFile.write(reinterpret_cast<char*>(&model.W1), sizeof(model.W1));
    outFile.write(reinterpret_cast<char*>(&model.B1), sizeof(model.B1));
    outFile.write(reinterpret_cast<char*>(&model.W2), sizeof(model.W2));
    outFile.write(reinterpret_cast<char*>(&model.B2), sizeof(model.B2));
    outFile.close();
}

inline void exportNetworkQuantized(){
    std::ofstream outFile("nnueweights_" + std::to_string(sessionId) + "_" + std::to_string(exportNetworkNumber) + ".bin", std::ios::out | std::ios::binary);
    exportNetworkNumber++;

    int16_t qW1[INPUT_HALF * HIDDEN_HALF];
    int16_t qB1[HIDDEN_HALF];
    int16_t qW2[OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2];
    int16_t qB2[OUTPUT_WEIGHT_BUCKET_COUNT];
    
    // We need to convert these doubles to int16. Let's multiply W1 and B1 by Q1,
    // W2 by Q2, and B2 by Q1 * Q2 so that we can factor Q1 * Q2 out of everyone.
    // Then to get back the original value, we divide by Q1 * Q2

    for (int i = 0; i < INPUT_HALF * HIDDEN_HALF; i++){
        qW1[i] = round(model.W1[i] * Q1);
    }
    for (int i = 0; i < HIDDEN_HALF; i++){
        qB1[i] = round(model.B1[i] * Q1);
    }
    for (int i = 0; i < OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2; i++){
        qW2[i] = round(model.W2[i] * Q2);
    }
    for (int i = 0; i < OUTPUT_WEIGHT_BUCKET_COUNT; i++){
        qB2[i] = round(model.B2[i] * Q1 * Q2);
    }

    outFile.write(reinterpret_cast<char*>(&qW1), sizeof(qW1));
    outFile.write(reinterpret_cast<char*>(&qB1), sizeof(qB1));
    outFile.write(reinterpret_cast<char*>(&qW2), sizeof(qW2));
    outFile.write(reinterpret_cast<char*>(&qB2), sizeof(qB2));
    outFile.close();
}

void initWithRandomWeights(){
    // Normal distribution constructor is (mean, standard deviation)
    // HE initialization is (mean = 0, std = sqrt(2 / indegree))
    // On average, ~20 input neurons are set to 1 so we say that is our indegree

    std::default_random_engine generator(69);
    std::normal_distribution<double> random1(0, sqrt(2.0 / 20));
    std::normal_distribution<double> random2(0, sqrt(2.0 / (HIDDEN_HALF * 2)));
    std::normal_distribution<double> random3(0, 0.01);

    for (int i = 0; i < INPUT_HALF * HIDDEN_HALF; i++){
        model.W1[i] = random1(generator);
    }
    for (int i = 0; i < OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2; i++){
        model.W2[i] = random2(generator);
    }
    for (int i = 0; i < HIDDEN_HALF; i++){
        model.B1[i] = random3(generator);
    }
    for (int i = 0; i < OUTPUT_WEIGHT_BUCKET_COUNT; i++){
        model.B2[i] = random3(generator);
    }
}

void proccessData(int curThread, Data *dat){
    NeuralNetwork &grad = threadGrad[curThread];

    for (int batchIdx = curThread; batchIdx < BATCH_SIZE; batchIdx += THREAD_COUNT){

        /********************************************STEP 1********************************************/

        Data &entry = dat[batchIdx];

        double accumulator[HIDDEN_HALF * 2];
        double output = 0;

        double gradZ2;
        double gradA1[HIDDEN_HALF * 2];
        double gradZ1[HIDDEN_HALF * 2];

        int outputWeightBucket = entry.calculateOutputBucket();
        int outputWeightsIndex = outputWeightBucket * HIDDEN_HALF * 2;

        /********************************************STEP 2********************************************/
        
        // Initialize with B1
        for (int i = 0; i < HIDDEN_HALF; i++){
            accumulator[i] = accumulator[i + HIDDEN_HALF] = model.B1[i];
        }

        // Calculate accumulator
        for (int f = 0; f < entry.stmSz; f++){
            int start = entry.stmFeatures[f] * HIDDEN_HALF;

            for (int i = 0; i < HIDDEN_HALF; i++){
                accumulator[i] += model.W1[start + i];
            }
        }
        for (int f = 0; f < entry.enemySz; f++){
            int start = entry.enemyFeatures[f] * HIDDEN_HALF;

            for (int i = 0; i < HIDDEN_HALF; i++){
                accumulator[i + HIDDEN_HALF] += model.W1[start + i];
            }
        }

        // Make the inference
        for (int i = 0; i < HIDDEN_HALF * 2; i++){
            output += crelu(accumulator[i]) * model.W2[i + outputWeightsIndex];
        }
        output += model.B2[outputWeightBucket];

        // Get error
        errorAccumulation[curThread] += error(output, entry.score, entry.wdl);

        /********************************************STEP 3********************************************/

        // Calculate dZ2
        gradZ2 = dError(output, entry.score, entry.wdl);

        // Calculate dB2
        grad.B2[outputWeightBucket] += gradZ2;
        
        // Calculate dW2, dA1, dZ1
        for (int i = 0; i < HIDDEN_HALF * 2; i++){
            grad.W2[i + outputWeightsIndex] += gradZ2 * crelu(accumulator[i]);
            gradA1[i] = model.W2[i + outputWeightsIndex] * gradZ2;
            gradZ1[i] = gradA1[i] * dCrelu(accumulator[i]);
        }
        
        // Calculate dB1 (note that this single bias has influence on 2 nodes)
        for (int i = 0; i < HIDDEN_HALF; i++){
            grad.B1[i] += gradZ1[i] + gradZ1[i + HIDDEN_HALF];
        }

        // Calculate dW1 (note that a single weight can influence 2 nodes)
        for (int f = 0; f < entry.stmSz; f++){
            int start = entry.stmFeatures[f] * HIDDEN_HALF;
            for (int i = 0; i < HIDDEN_HALF; i++){
                grad.W1[start + i] += gradZ1[i];
            }
        }
        for (int f = 0; f < entry.enemySz; f++){
            int start = entry.enemyFeatures[f] * HIDDEN_HALF;
            for (int i = 0; i < HIDDEN_HALF; i++){
                grad.W1[start + i] += gradZ1[i + HIDDEN_HALF];
            }
        }
    }
}

void applyGradients(int curThread, double lr){
    // Apply gradients to W1
    for (int i = curThread; i < INPUT_HALF * HIDDEN_HALF; i += THREAD_COUNT){
        double grad = 0;
        for (int td = 0; td < THREAD_COUNT; td++){
            grad += threadGrad[td].W1[i];
        }
        adamUpdate(grad, lr, model.W1[i], momentum.W1[i], rms.W1[i]);
    }

    // Apply gradients to B1
    for (int i = curThread; i < HIDDEN_HALF; i += THREAD_COUNT){
        double grad = 0;
        for (int td = 0; td < THREAD_COUNT; td++){
            grad += threadGrad[td].B1[i];
        }
        adamUpdate(grad, lr, model.B1[i], momentum.B1[i], rms.B1[i]);
    }

    // Apply gradients to W2
    for (int i = curThread; i < OUTPUT_WEIGHT_BUCKET_COUNT * HIDDEN_HALF * 2; i += THREAD_COUNT){
        double grad = 0;
        for (int td = 0; td < THREAD_COUNT; td++){
            grad += threadGrad[td].W2[i];
        }
        adamUpdate(grad, lr, model.W2[i], momentum.W2[i], rms.W2[i]);
    }

    // Apply gradients to B2
    for (int i = curThread; i < OUTPUT_WEIGHT_BUCKET_COUNT; i += THREAD_COUNT){
        double grad = 0;
        for (int td = 0; td < THREAD_COUNT; td++){
            grad += threadGrad[td].B2[i];
        }
        adamUpdate(grad, lr, model.B2[i], momentum.B2[i], rms.B2[i]);
    }
}

void train(std::string inputPath, std::string reportLossPath){
    // Intialize
    std::ofstream outputLoss(reportLossPath);
    DataLoader::init(inputPath);

    long long startTime = getTime();
    double lr = LR_BASE;

    int cosineInterval = COSINE_INTERVAL_BASE;
    int cosineCounter = -1;

    for (int epoch = 1; epoch <= MAX_EPOCHS; epoch++){
        double epochError = 0;
        double iterationReportError = 0;

        long long batchIterations = 0;
        long long positionsSeen = 0;
        long long epochStartTime = getTime();
        
        // Print starting epoch and learning rate
        if (epoch <= LR_TRANSITION){
            std::cout << "Starting Epoch: "  << epoch
                      << ", Learning Rate: " << lr
                      << std::endl;
        }
        else{
            std::cout << "Starting Epoch: "    << epoch
                      << ", Learning Rate: "   << lr
                      << ", Cosine Counter: "  << cosineCounter
                      << ", Cosine Interval: " << cosineInterval
                      << std::endl;
        }
        
        // Begin epoch
        while (positionsSeen < EPOCH_SIZE){
            // Initialize
            std::thread threads[THREAD_COUNT];
            Data *dat = DataLoader::currentData + DataLoader::position;
            double batchError = 0;

            batchIterations++;
            positionsSeen += BATCH_SIZE;

            for (int td = 0; td < THREAD_COUNT; td++){
                errorAccumulation[td] = 0;
                threadGrad[td] = {};
            }

            // Proccess data in a multithreaded way
            for (int td = 0; td < THREAD_COUNT; td++){
                threads[td] = std::thread(proccessData, td, dat);
            }
            for (int td = 0; td < THREAD_COUNT; td++){
                threads[td].join();
            }
            
            // Accumulate error
            for (int td = 0; td < THREAD_COUNT; td++){
                batchError += errorAccumulation[td];
            }
            epochError += batchError;
            iterationReportError += batchError;

            // Apply gradients with ADAM and multithreading
            for (int td = 0; td < THREAD_COUNT; td++){
                threads[td] = std::thread(applyGradients, td, lr);
            }
            for (int td = 0; td < THREAD_COUNT; td++){
                threads[td].join();
            }

            // Advance data loader index
            DataLoader::advanceDataLoader();

            // Report info after certain number of iterations
            if (batchIterations % iterationReport == 0){
                std::cout << "Batch: "                    << batchIterations
                          << ", Iteration Report Error: " << iterationReportError / (static_cast<long long>(BATCH_SIZE) * iterationReport)
                          << ", Epoch Average: "          << epochError / (BATCH_SIZE * batchIterations)
                          << ", Time: "                   << (getTime() - startTime) / 1000.0
                          << ", Fens/s: "                 << (batchIterations * BATCH_SIZE) / ((getTime() - epochStartTime) / 1000.0)
                          << std::endl;
                
                iterationReportError = 0;
            }
        }
        
        // Print average error after each epoch
        std::cout << "Finished Epoch: " << epoch
                  << ", Error: "        << epochError / (BATCH_SIZE * batchIterations)
                  << std::endl;

        outputLoss << epoch << " " << epochError / (BATCH_SIZE * batchIterations) << std::endl;

        // Export networks
        exportNetworkCheckpoint();
        exportNetworkQuantized();
        
        // Exponential learning rate decay (do at the end of [1...LR_TRANSITION) epochs)
        if (epoch < LR_TRANSITION){
            lr *= LR_DECAY;
        }
        // Cosine annealing
        else{
            if (cosineCounter == cosineInterval){
                cosineCounter = 0;
                cosineInterval *= COSINE_INTERVAL_MULTIPLIER;
            }
            else{
                cosineCounter++;
            }
            // Adjust learning rate
            lr = COSINE_MIN + 0.5 * (COSINE_MAX - COSINE_MIN) * (1.0 + static_cast<double>(cos(static_cast<double>(cosineCounter) / static_cast<double>(cosineInterval) * 3.14159265)));
        }
    }
}

int main(){ 
    initSessionId();
    initWithRandomWeights();
    train("lcfishFiltered.binpack", "loss.txt");
}