#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/tensor_cuda.h"
#include "../../hardware/gpu/tensor_cuda_op.h"
#endif


// ReLU
void ReLu(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::ReLu");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_relu(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_relu(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// RELU Derivative, always increment over parent delta
void D_ReLu(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ReLu");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

    PD->tsem->lock();
    if (D->isCPU()) {
        cpu_d_relu(D, I, PD);
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_relu(D,I,PD);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}

// SOFTMAX
void Softmax(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softmax");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::Softmax");
    if (A->ndim != 2) msg("Softmax only over 2D Tensor (batch x logits)", "Tensor::Softmax");

    B->tsem->lock();

    if (A->isCPU()) {
        cpu_softmax(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_softmax(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// SOFTMAX DERIVATIVE
void D_Softmax(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Softmax");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Softmax");
    if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");

    if (D->isCPU()) {
       cpu_d_softmax(D, I, PD);
    }
#ifdef cGPU
    else if (D->isGPU())
      {

        Tensor *aux=new Tensor(D->getShape(),D->device);
        aux->set(1.0);
        Tensor::add(1.0,aux,-1.0,I,aux,0);
        Tensor::el_mult(I,aux,aux,0);
        Tensor::el_mult(D,aux,PD,1);

        delete aux;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}