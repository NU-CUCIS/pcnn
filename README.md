# Parallel Convolutional Neural Network (PCNN)
PCNN is a light-weighted open source software framework for parallel CNN training on distributed-memory platforms.
PCNN is implemented in C/C++ language.
For parallel training, MPI is used to implement data parallelism and each MPI process adopts OpenMP to utilize multiple compute cores.

## Repository Structure
 + src: PCNN source code
 + examples
   + cifar10: documents for a case study of CIFAR10 classification
   + imagenet: documents for a case study of ImageNet classificaiton
   + div2k: documents for a case study of DIV2K super-resolution (image regression)
 
## Authors
 + Sunwoo Lee <slz839@eecs.northwestern.edu>
 + Qiao Kang <qkt561@eecs.northwestern.edu>
 + Wei-keng Liao <wkliao@eecs.northwestern.edu>

## Project Funding Support
This material is based upon work supported by the U.S. Department of Energy,
Office of Science, Office of Advanced Scientific Computing Research, Scientific
Discovery through Advanced Computing ([SciDAC](https://www.scidac.gov)) program,
[RAPIDS Institute](https://rapids.lbl.gov).
