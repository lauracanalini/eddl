
/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_NET_H
#define EDDL_NET_H

#include <string>
#include <vector>

#include "../layers/layer.h"
#include "../optimizers/optim.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"
#include "compserv.h"

using namespace std;

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;
typedef vector<vtensor> Mtensor;
typedef vector<string> vstring;
typedef vector<float> verr;
typedef vector<int> vind;
typedef vector<Loss *> vloss;
typedef vector<Metric *> vmetrics;


/////////////////////////////////////////
int isIn(Layer *l, vlayer vl, int &ind);
int isInorig(Layer *l, vlayer vl, int &ind);

#define MAX_THREADS 1024

class Net {
private:
    void build(Optimizer *opt, vloss lo, vmetrics me);

    void set_compserv(CompServ *cs);

public:
    string name;
    int dev;
    int batch_size;
    int tr_batches;
    int inferenced_samples;
    int trmode;

    vector<int> devsel;
    CompServ *cs;

    vlayer layers;
    vlayer lin;
    vlayer lout;
    vlayer vfts;
    vlayer vbts;

    vloss losses;
    vmetrics metrics;
    verr fiterr;
    verr total_loss;
    verr total_metric;
    FILE *flog_tr;
    FILE *flog_ts;

    Optimizer *optimizer;
    vector<Net *> snets;

    vtensor Xs[MAX_THREADS];
    vtensor Ys[MAX_THREADS];

    Net(vlayer in, vlayer out);
    ~Net();

    void build(Optimizer *opt, vloss lo, vmetrics me, CompServ *cs);
    void toGPU(vector<int> &g,int lsb);
    void toCPU(int t);

    void fts();
    void bts();
    void split(int c, int todev);
    int inNet(Layer *l);
    void walk(Layer *l);
    void walk_back(Layer *l);


    void resize(int batch);

    string summary();
    void plot(string fname,string mode);

    void setmode(int m);
    void sync_weights();

    void save(const string& filename, const string& format);
    void load(const string& filename, const string& format);
    void setlogfile(string fname);


    //Func
    void do_initialize();
    void do_reset();
    void do_reset_grads();
    void do_forward();
    void do_delta();
    void do_compute_loss();
    void do_backward();
    void do_applygrads();


    // API
    void run_snets(void *(*F)(void *t));
    void forward(vector<Layer *> in);
    void forward(vector<Tensor*> in);
    void forward();
    void reset_loss();
    void print_loss(int b);
    void backward(vector<Tensor *> target);
    void backward(Layer* (*f)(Layer *),Layer *out);
    void backward();
    void delta();
    void reset();
    void reset_grads();
    void update();
    void compute_loss();
    void clamp(float min,float max);


    void fit(vtensor tin, vtensor tout, int batch_size, int epochs);
    void train_batch(vtensor X, vtensor Y, vind sind, int eval = 0);
    void evaluate(vtensor tin, vtensor tout);
    void predict(vtensor tin, vtensor tout);





};

void collectTensor(Layer *l,string tname="output");
void distributeTensor(Layer *l,string tname="output");

#endif  //EDDL_NET_H
