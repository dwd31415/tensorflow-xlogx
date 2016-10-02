# tensorflow-xlogx
This package implements a C++/Cuda OP for evaluating this \ell function:
![\ell function](https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/formula_ell.png )

This extends the usual x*log(x) by defining it for x=0, the inserted value 0 is the limit of said function for x -> 0:
