# tensorflow-xlogx
This package implements a C++/Cuda OP for evaluating this &#8467; function(called x\*log(x) for reasons of simplicity in the rest of the README file):

<img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/formula_ell.png" width="300">

This extends the usual x\*log(x) by defining it for x=0, the inserted value of 0 is the limit of said function for x -> 0:

 <img src="https://raw.githubusercontent.com/dwd31415/tensorflow-xlogx/master/formulas/limit.png" width="210">

### Installation 
Just clone this repo and run 
```
python3 setup.py install
```
TensorFlow already has to be installed and should be up to date. 
### Configuration 
By changing the content of src/settings.h and reinstalling you can configure this package.
#### THROW_ERROR_FOR_NEGATIVE_LOGS_ON_CPU
This is a simple on/off switch if you set it to 1, the xlogx function will not return NaN for negtive inputs any more, but instead throw an error, this behaviour is only implemented on the GPU right now.

*Default: 0*
#### FIX_DERIVATIVE_FROM_0_TO_EPSILON
The limit of log(x) + 1(the deriavtive of x\*log(x)) for x->0 is not a real number, but -&#8734;(an extended real number), therefore we have to choose some value for it, this option allows you to extend this and make the derivative of x\*log(x) return a fixed value for all 0	&#8804; x	&#8804; &#1013;. This allows you to make the derivative a continuous function.

*Default: 0*
#### EPSILON
The &#1013; mentioned in FIX_DERIVATIVE_FROM_0_TO_EPSILON.

*Default: 1e-3*
#### FIXED_DERIVATIVE_VALUE 
The value for the derivative of x\*log(x) for x=0 or 0	&#8804; x	&#8804; &#1013;(see FIX_DERIVATIVE_FROM_0_TO_EPSILON).

*Default: -1.0*

### Use Case
Implementing the &#8467; function in tensorflow is to my knowledge only possible with some kind of switch, if raw performance is paramount this is very inefficent, by implementing &#8467; in pure C++ and CUDA C this package is a way around the need for symbolic switches in your TensorFlow graph. 
