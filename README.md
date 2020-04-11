# Parallel Convolutional Neural Network (PCNN)
PCNN is a light-weighted open-source C/C++ software framework for Convolutional Neural Network (CNN).
This software program can be used to solve classification/regression problems using deep CNNs.
PCNN is designed to run on CPU-based distributed-memory platforms, using MPI for inter-process communications.
The training is parallelized using data parallelism.

## Source Tree Structure
 + src: PCNN source code. To build an executable, refer to the src/README.md.
 + use_cases
   + classification: documents about how to use PCNN for classification problems (CIFAR10 and ImageNet)
   + regression: documents about how to use PCNN for regression problems (DIV2K image super-resolution)

## Questions/Comments
 + Sunwoo Lee <slz839@eecs.northwestern.edu>
 + Qiao Kang <qkt561@eecs.northwestern.edu>
 + Wei-keng Liao <wkliao@eecs.northwestern.edu>

## Publications
* Sunwoo Lee, Qiao Kang, Sandeep Madireddy, Prasanna Balaprakash, Ankit Agrawal, Alok Choudhary, Richard Archibald, and Wei-keng Liao. Improving Scalability of Parallel CNN Training by Adjusting Mini-Batch Size at Run-Time. In IEEE International Conference on Big Data, December 2019 [pdf](http://cucis.eecs.northwestern.edu/publications/pdf/LKM19.pdf)
* Sunwoo Lee, Ankit Agrawal, Prasanna Balaprakash, Alok Choudhary, and Wei-keng Liao. Communication-Efficient Parallelization Strategy for Deep Convolutional Neural Network Training. In the Workshop on Machine Learning in HPC Environments, held in conjunction with the International Conference for High Performance Computing, Networking, Storage and Analysis, November 2018 [pdf](http://cucis.eecs.northwestern.edu/publications/pdf/LAB18.pdf)
* Sunwoo Lee, Dipendra Jha, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao. Parallel Deep Convolutional Neural Network Training by Exploiting the Overlapping of Computation and Communication (best paper finalist). In the 24th International Conference on High-Performance Computing, Data, and Analytics, December 2017 [pdf](http://cucis.eecs.northwestern.edu/publications/pdf/LJA17.pdf)

## Project Funding Support
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program,
[RAPIDS Institute](https://rapids.lbl.gov).
