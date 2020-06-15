#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_D_EXP
void k_d_exp(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i] * I[i];
}
#endif

}
