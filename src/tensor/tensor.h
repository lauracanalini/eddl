
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#ifndef EDDLL_TENSOR_H
#define EDDLL_TENSOR_H

#include <stdio.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>

#define DEV_CPU 0

#define DEV_GPU 1000
#define DEV_GPU_0 1000
#define DEV_GPU_1 1001
#define DEV_GPU_2 1002
#define DEV_GPU_3 1003
#define DEV_GPU_4 1004
#define DEV_GPU_5 1005
#define DEV_GPU_6 1006
#define DEV_GPU_7 1007
#define DEV_GPU_8 1008

#define DEV_FPGA 2000
#define DEV_FPGA_0 2000
#define DEV_FPGA_1 2001
#define DEV_FPGA_2 2002
#define DEV_FPGA_3 2003
#define DEV_FPGA_4 2004
#define DEV_FPGA_5 2005
#define DEV_FPGA_6 2006
#define DEV_FPGA_7 2007
#define DEV_FPGA_8 2008

#define MAX_GPUS 8

using namespace std;
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXRMf;

typedef vector<int> tshape;

void msg(string s);

void msg(string s, string s2);

class Tensor;

class ConvolDescriptor {
public:
    vector<int> ksize;
    vector<int> stride;
    vector<int> pad;

    int nk, kr, kc, kz;
    int sr, sc;
    int ir, ic, iz;
    int r, c, z;
    int padr, padc;
    //float *ptr;


    Tensor *I; // Input map
    Tensor *ID;// Delta input map
    Tensor *K; // filters
    Tensor *bias; // bias
    Tensor *gK;// gradient filters
    Tensor *gbias;// gradient bias
    Tensor *D; // Delta
    Tensor *O; // Outputmap

    // CPU implementation
    float *ptrI;
    Eigen::MatrixXf matI; // input
    Eigen::MatrixXf matK; // kernels
    Eigen::MatrixXf matO; // output
    Eigen::MatrixXf matD; // Delta
    Eigen::MatrixXf matgK; // gradient kernels

    //...
    ConvolDescriptor();

    ConvolDescriptor(int filters, const vector<int> &ks, const vector<int> &st, string p);

    ConvolDescriptor(const initializer_list<int> &ks, const initializer_list<int> &st, const initializer_list<int> &p);

    void build(Tensor *A);
    void resize(Tensor *A);
};

class PoolDescriptor : public ConvolDescriptor {
public:

    Tensor *indX, *indY; // indexes

    //...
    PoolDescriptor(const initializer_list<int> &ks, const initializer_list<int> &st, string p);

    PoolDescriptor(const initializer_list<int> &ks, const initializer_list<int> &st, const initializer_list<int> &p);

    PoolDescriptor(const vector<int> &ks, const vector<int> &st, string p);

    void build(Tensor *A);
    void resize(Tensor *A);
};


class Tensor {

public:
    int device;
    int ndim;
    int size;
    vector<int> shape;
    vector<int> stride;

    float *ptr;

    // CPU
    Eigen::MatrixXf *ptr2;
    Eigen::MatrixXf mat;

    // GPU
    int gpu_device;

    //FPGA

    // Multithreading. Tensor semaphore
    mutex *tsem;

    // Constructors
    Tensor();
    Tensor(const initializer_list<int> &init, int dev=DEV_CPU);
    explicit Tensor(vector<int> shape, int dev=DEV_CPU);
    Tensor(vector<int> shape, Tensor *T);
    explicit Tensor(string fname, int bin=1);
    Tensor(vector<int> shape, float *fptr, int dev=DEV_CPU);

    ~Tensor();

    vector<int> getShape();

    void info();

    Tensor *share();

    void print();

    void save(string s);

    // data
    void point2data(const vector<int>& shape, float *ptr);
    void copydata(const vector<int>& s, float *newptr);

    // devices
    int isCPU();

    int isGPU();

    int isFPGA();

    ////////////////////////////////


    // math
    void set(float v);

    void mult(float v);

    void div(float v);

    void sum(float v);

    void sub(float v);


    void set_abs();

    void set_log();

    void set_log2();

    void set_log10();

    void set_exp();

    void set_sqrt();

    void set_sqr();

    float total_sum();

    float total_abs();

    //rand
    void rand_uniform(float v);

    void rand_suniform(float v);

    void rand_gaussian(float m, float s);

    void rand_binary(float v);


    ///////// static metods
    static int eqsize(Tensor *A, Tensor *B);

    static void transpose(Tensor *A, Tensor *B, vector<int> dims);

    static void copy(Tensor *A, Tensor *B);

    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

    static void sign(Tensor *A, Tensor *B);

    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);

    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);

    static void inc(Tensor *A, Tensor *B);

    static void sum(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);

    static void sum(Tensor *A, Tensor *B, Tensor *C);

    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);

    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

    static void reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);

    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);

    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);


///////7
    static void reduceTosum(Tensor *A, Tensor *B, int axis);

    static void reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C, int incB);
    static void delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C,int incB);

//////
    static void cent(Tensor *A, Tensor *B, Tensor *C);

    static int accuracy(Tensor *A, Tensor *B);

/////
    static void ReLu(Tensor *A, Tensor *B);

    static void Softmax(Tensor *A, Tensor *B);

    static void D_ReLu(Tensor *D, Tensor *I, Tensor *PD);

    static void D_Softmax(Tensor *D, Tensor *I, Tensor *PD);


    ///////
    static void Conv2D(ConvolDescriptor *D);

    static void Conv2D_grad(ConvolDescriptor *D);

    static void Conv2D_back(ConvolDescriptor *D);


    static void MPool2D(PoolDescriptor *D);

    static void MPool2D_back(PoolDescriptor *D);

};


#endif //EDDLL_TENSOR_H





















/////////////
