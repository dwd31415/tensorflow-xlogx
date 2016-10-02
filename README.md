# tensorflow-xlogx
This package implements a C++/Cuda OP for evaluating this \ell function:

<img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/formula_ell.png" width="350">

This extends the usual x*log(x) by defining it for x=0, the inserted value 0 is the limit of said function for x -> 0:

<img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/limit.png" width="230">
