/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "loss.h"

using namespace std;

LDice::LDice() : Loss("dice") {}

// Dice derivative: DeltaDi = 2 * (Ti^2 + Ti + 1) / (Ti + Yi + 1)
void LDice::delta(Tensor *T, Tensor *Y, Tensor *D)
{
    Tensor one(T->getShape(), T->device);
    Tensor Ti1(T->getShape(), T->device);
    Tensor den(T->getShape(), T->device);
    Tensor res(T->getShape(), T->device);

    one.fill_(1.0);

    // (Ti + 1)
    Tensor::add(1, &one, 1, T, &Ti1, 0);
    // (Ti + Yi + 1)
    Tensor::add(1, &Ti1, 1, Y, &den, 0);
    // Ti^2
    Tensor::el_mult(T, T, D, 0);
    // (Ti^2 + Ti + 1)
    Tensor::add(1, D, 1, &Ti1, D, 0);
    // (Ti^2 + Ti + 1) / (Ti + Yi + 1)
    Tensor::el_div(D, &den, D, 0);

    D->mult_(-2.);
}

// Dice: Di = 2 * (Ti * Yi + 1.) / (Ti + Yi + 1.)
float LDice::value(Tensor *T, Tensor *Y)
{
    float f;
    Tensor aux1(T->getShape(), T->device);

#pragma omp parallel for
    for (int i = 0; i < T->size; i++) {
        aux1.ptr[i] = (2 * T->ptr[i] * Y->ptr[i] + 1.) / (T->ptr[i] + Y->ptr[i] + 1.);
    }

    f = 1 - aux1.sum();
    return f;
}