# UserfulDecorators
Decorator for Solving the problem of size mismatch during prediction of the CNN model with down sampling

# Function
When there is a structure of down-sampling and then up-sampling in the network, when the input size cannot be divisible by the down-sampling multiple, a size mismatch error will be reported. In order to solve this problem, we can make the input size meet the requirements of the network by performing appropriate padding on the input size, and tailor the output accordingly.

