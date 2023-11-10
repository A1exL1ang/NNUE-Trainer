# NNUE-Trainer
This is the C++ CPU trainer for Superultra's (my chess engine) NNUE. The architecture of the most recent network is (768x10-->512)x2-->1. The network has perspective, 10 king buckets, and 8 output weight buckets based on the number of pieces remaining. 

The trainer supports multithreading, fen skipping, and uses the ADAM optimization. It reads data directly off binpacks thanks to the Stockfish binpack reading code. Note that the trainer assumes scores are on a centipawn scale and that the data is prefiltered (a score of 32002 indicates that the position is skipped). Furthermore, the score and result (-1/0/1) is relative to the side to move.
