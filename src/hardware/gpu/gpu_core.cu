/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


int* get_block_dim(int N, int blockSize){
  int* res = new int[2];
  int blocks = (N + blockSize - 1) / blockSize;
  if (N<blockSize) { blockSize = N; }

   res[0] = blocks;
   res[1] = blockSize;
  return res;
}


void gpu_copy_to_gpu(float *nptr,Tensor *A){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(A->ptr,nptr,A->size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}


void gpu_copy_from_gpu(Tensor *A,float *nptr){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(nptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}


void gpu_copy_gpu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}


void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc){
  int device=A->gpu_device;
  cudaSetDevice(device);

  int at=A->size/A->shape[0];
  int bt=B->size/B->shape[0];

  int t=1;
  for(int i=2;i<B->ndim;i++)
    t*=B->shape[i];

  int tot=B->shape[0]*(bend-1)*B->shape[1]*t;
  int r,c;

  while (aend-aini>0) {

      if ((aend-aini)>MAX_TPB) r=MAX_TPB;
      else r=(aend-aini);
      c=t;

      dim3 dimGrid(A->shape[0],c);
      dim3 dimBlock(r);

      fill<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,t,aini,at,bini,bt,tot,inc);
      aini+=MAX_TPB;
      bini+=MAX_TPB;

  }

    //check_cuda(cudaDeviceSynchronize(),"fill");

}


void gpu_mask(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  mask<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
  check_cuda(cudaDeviceSynchronize(),"mask");

}


void gpu_fill_(Tensor *A, float v) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    fill_<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
    check_cuda(cudaDeviceSynchronize(),"set");
}


void gpu_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses== nullptr){
        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    select<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "select");
}

void gpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses== nullptr){
        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), A->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->addresses, A->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(A);  // A is the small
    select_back<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "select_back");
}


void gpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses== nullptr){
        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    set_select<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "set_select");
}


void gpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses== nullptr){
        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(A);  // B is the small
    set_select_back<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "set_select_back");
}