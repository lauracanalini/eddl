// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "eddl.h"
#include "eddl.h"

int main(int argc, char **argv) {

    Tensor *A=new Tensor({4,2,3,7});
    Tensor *B=new Tensor({4,3});
    Tensor *C=new Tensor({4,3});

    vector<int> axis;
    axis.push_back(1);
    axis.push_back(3);

    A->info();
    A->set(1.0);
    A->rand_uniform(1.0);
    A->print();


    cout<<"Mean\n";
    Tensor::reduce(A,B,axis,"mean",false,NULL,0);

    B->info();
    B->print();

 /////
    cout<<"Max\n";
    Tensor::reduce(A,B,axis,"max",false,C,0);

    B->info();
    B->print();
    C->info();
    C->print();
    cout<<"Delta max\n";
    Tensor::delta_reduce(B,A,axis,"max",false,C,0);

    A->print();

/////
    cout<<"Sum\n";
    Tensor::reduce(A,B,axis,"sum",false,NULL,0);
    B->info();
    B->print();

    cout<<"==================\n";
    cout<<"keepdims true\n";
    cout<<"==================\n";
    A=new Tensor({4,2,3});
    B=new Tensor({4,2,3});
    C=new Tensor({4,2,3});

    vector<int> axis2;
    axis2.push_back(1);

    A->info();
    A->set(1.0);
    A->print();

    cout<<"Mean\n";
    Tensor::reduce(A,B,axis2,"mean",true,NULL,0);

    B->info();
    B->print();

    /////
    cout<<"Max\n";
    Tensor::reduce(A,B,axis2,"max",true,C,0);

    B->info();
    B->print();
    C->info();
    C->print();

    cout<<"Delta max\n";
    Tensor::delta_reduce(B,A,axis2,"max",true,C,0);
    A->print();

    /////
    cout<<"Sum\n";
    Tensor::reduce(A,B,axis2,"sum",true,NULL,0);
    B->info();
    B->print();

    cout<<"==================\n";
    cout<<"EDDL Layers\n";
    cout<<"==================\n";

    tensor t = eddl.T({1,10,10,4});
    t->data->set(1.0);
    t->data->ptr[0]=10;

    cout<<"\nMean\n";
    layer m=eddl.Mean(t,{1,2});
    m->forward();
    m->output->info();
    m->output->print();

    cout<<"\nVar\n";
    //t->data->print();
    layer v=eddl.Var(t,{1,2});

    v->forward();
    v->output->info();
    v->output->print();



}


///////////