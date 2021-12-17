# PCNN: Parallel Convolutional Neural Network
PCNN is an open-source C/C++ software implementation of Convolutional Neural Network (CNN).
It can be used to train and deploy a deep CNN model for general classification/regression problems.
The parallelization strategy used in PCNN is data parallelism and developed to run on
CPU-based distributed-memory parallel computers, using MPI for inter-process communications.
It has been evaluated on [Cori](https://www.nersc.gov/systems/cori/), the Cray XC40
supercomputer at [NERSC](https://www.nersc.gov). 

This work aims to provide a good scalability of parallel CNN training while achieving the same accuracy as that of sequential training.
PCNN exploits the overlap of computation and communication to improve the scalability.
In order to maximize the degree of overlap, the gradients are averaged across all the processes using communication-efficient gradient averaging algorithm proposed in [[2](#ref2)].

## Source Tree Structure
 + [./src](src): The folder contains the source codes.
   To build an executable, refer to the src/README.md.
 + Use cases
   + [./use_cases/classification](use_cases/classification): A use case of running PCNN for classification problems using CIFAR10 and ImageNet data.
   + [./use_cases/regression](use_cases/regression): A use case of running  PCNN for regression problems using DIV2K image super-resolution data.

## Questions/Comments
 + Sunwoo Lee <<slz839@eecs.northwestern.edu>>
 + Qiao Kang <<qkt561@eecs.northwestern.edu>>
 + Wei-keng Liao <<wkliao@northwestern.edu>>

## Publications
1. Sunwoo Lee, Qiao Kang, Reda Al-Bahrani, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao. [Improving Scalability of Parallel CNN Training by Adaptively Adjusting Parameter Update Frequency](https://www.sciencedirect.com/science/article/pii/S0743731521001830). Parallel and Distributed Computing (JPDC), 159:10–23, January 2022.
2. Sunwoo Lee, Qiao Kang, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao, [Communication-Efficient Local Stochastic Gradient Descent for Scalable Deep Learning](https://ieeexplore.ieee.org/document/9378178). In IEEE International Conference on Big Data, December 2020.
3. Sunwoo Lee, Qiao Kang, Sandeep Madireddy, Prasanna Balaprakash, Ankit Agrawal, Alok Choudhary, Richard Archibald, and Wei-keng Liao. [Improving Scalability of Parallel CNN Training by Adjusting Mini-Batch Size at Run-Time](https://ieeexplore.ieee.org/document/9006550). In IEEE International Conference on Big Data, December 2019.
4. Sunwoo Lee, Ankit Agrawal, Prasanna Balaprakash, Alok Choudhary, and Wei-keng Liao. [Communication-Efficient Parallelization Strategy for Deep Convolutional Neural Network Training](https://ieeexplore.ieee.org/document/8638635). In the Workshop on Machine Learning in HPC Environments, held in conjunction with the International Conference for High Performance Computing, Networking, Storage and Analysis, November 2018.
5. Sunwoo Lee, Dipendra Jha, Ankit Agrawal, Alok Choudhary, and Wei-keng Liao. [Parallel Deep Convolutional Neural Network Training by Exploiting the Overlapping of Computation and Communication](https://ieeexplore.ieee.org/document/8287749) (best paper finalist). In the 24th International Conference on High-Performance Computing, Data, and Analytics, December 2017.

## Project Funding Support
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program,
[RAPIDS Institute](https://rapids.lbl.gov).
