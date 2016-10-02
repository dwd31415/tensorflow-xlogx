# tensorflow-xlogx
This package implements a C++/Cuda OP for evaluating this &#8467; function:

<img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/formula_ell.png" width="300">

This extends the usual x*log(x) by defining it for x=0, the inserted value of 0 is the limit of said function for x -> 0:

 <img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/limit.png" width="210">

### Installation 
Just clone this repo and run 
```
python3 setup.py install
```
