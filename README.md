# Neural Network
Installing
-----------
```
npm i @death_raider/neural-network
```
About
-----
This is an easy to use Neural Network package with SGD using backpropagation as a gradient computing technique.

Creating the model
------------------
```js
const NeuralNetwork = require('./Neural-Network.js').NeuralNetwork
//creates ANN with 2 input nodes, 1 hidden layers with 2 hidden nodes and 1 output node
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2],
  output_nodes :1,
  weight_bias_initilization_range : [-1,1]
});
```
Parameters like the activations for hidden layer and output layers are set as leaky relu and sigmoid respectively but can changed
```js
//format for activation function = [ function ,  derivative of function ]
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)] //sets activation for hidden layers as sigmoid function
```
Training, Testing and Using
---------------------------
For this example we'll be testing it on the XOR function.

There are 2 ways we can go about training:

1) Inbuilt Function
```js
function xor(){
  let inp = [Math.floor(Math.random()*2),Math.floor(Math.random()*2)]; //random inputs 0 or 1 per cell
  let out = (inp.reduce((a,b)=>a+b)%2 == 0)?[0]:[1]; //if even number of 1's in input then 0 else 1 as output
  return [inp,out]; //train or validation functions should have [input,output] format
}
network.train({
  TotalTrain : 1e+6, //total data for training (not epochs)
  batch_train : 1, //batch size for training
  trainFunc : xor, //training function to get data
  TotalVal : 1000, //total data for validation (not epochs)
  batch_val : 1, //batch size for validation
  validationFunc : xor, //validation function to get data
  learning_rate : 0.1, //learning rate (default = 0.0000001)
  momentum : 0.9 // momentum for SGD
});
```
The `trainFunc` and `validationFunc` recieve an input of the batch iteration and the current epoch which can be used in the functions.

_`NOTE: The validationFunc is called AFTER the training is done`_

Now to see the avg. test loss:
```js
console.log("Average Validation Loss ->",network.Loss.Validation_Loss.reduce((a,b)=>a+b)/network.Loss.Validation_Loss.length);
// Result after running it a few times
// Average Validation Loss -> 0.00004760326022482792
// Average Validation Loss -> 0.000024864418333478723
// Average Validation Loss -> 0.000026908106414283446
```
2) Iterative
```js
for(let i = 0; i < 10000; i++){
  let [inputs,outputs] = xor()
  let dnn = network.trainIteration({
    input : inputs,
    desired : outputs,
  })
  network.update(dnn.Updates.updatedWeights,dnn.Updates.updatedBias,0.1)
  console.log(dnn.Cost,dnn.layers); //optional to view the loss and the hidden layers
}
// output after 10k iterations
// 0.00022788194782669534 [
//   [ 1, 1 ],
//   [ 0.6856085043616054, -0.6833685003507397 ],
//   [ 0.021348627488749498 ]
// ]
```
This iterative method can be used for visulizations, dynamic learning rate, etc...

To use the network:
```js
// network.use(inputs)  --> returns the hidden node values as well
let output = [ //truth table for xor gate
  network.use([0,0]),
  network.use([0,1]),
  network.use([1,0]),
  network.use([1,1])
]
```

To get the gradients w.r.t the inputs (Questionable correct me if wrong values)
```js
console.log( network.getInputGradients() );
```

Saving and Loading Models
-------------------------
This package allows to save the hyperparameters(weights and bias) in a file(s) and then unpack them, allowing us to use pretrained models.
Saving the model couldnt be further from simplicity:
```js
network.save(path)
```
Loading the model requires a bit more work as it is asynchronous:
```js
const NeuralNetwork = require('./Neural-Network.js')
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2],
  output_nodes :1,
  weight_bias_initilization_range : [-1,1]
});
(async () =>{
  await network.load(path) //make sure network is of correct structure
  let output = [  
    network.use([0,0]),  // --> returns the hidden node values as well
    network.use([0,1]),  // --> returns the hidden node values as well
    network.use([1,0]),  // --> returns the hidden node values as well
    network.use([1,1])   // --> returns the hidden node values as well
  ]
})()
```

# Linear Algebra
This class is not the most optimized as it can be, but the implementation of certain functions are based on traditional methods to solving them. Those functions will be marked with the * symbol.

Base function
--------------
The base function (basefunc) is a recursive function that takes in 3 parameters a, b, and Opt where a is an array and b is an object and opt is a function. The basefunc goes over all elements of a and also b if b is an array and then passes those elements to the opt function defined by the user. opt will take in 2 parameters and the return can be any object.
```js
linearA = new LinearAlgebra
let a = [
    [1,2,3,4],
    [5,6,7,8]
]
let b = [
    [8,7,6,5],
    [4,3,2,1]
]
function foo(p,q){
    return p*q
}
console.log(linearA.basefunc(a,b,foo))
// [ [ 8, 14, 18, 20 ], [ 20, 18, 14, 8 ] ]
```
Matrix Manipulation
--------------------
<ol>
    <li>Transpose(.transpose(matrix))</li>
        It gives the transpose of the matrix (only depth 2).<br /><br />
    <li>Scalar Matrix Product(.scalarMatrixProduct(scalar,matrix))</li>
        It gives a matrix which has been multiplied a scalar. Matrix can be of any depth.<br /><br />
    <li>Scalar Vector Product(.scalarVectorProduct(scalar,vector))</li>
        It gives a vector(array) which has been multipied by a scalar.<br /><br />
    <li>Vector Dot Product(.vectorDotProduct(vec1,vec2))</li>
        It gives the dot product for vectors.<br /><br />
    <li>Matrix vector product(.MatrixvectorProduct(matrix,vector))</li>
        It gives the product of a matrix and a vector.<br /><br />
    <li>Matrix Product(.matrixProduct(matrix1,matrix2))</li>
        It gives the product between 2 matrices.<br /><br />
    <li>Kronecker Product(.kroneckerProduct(matrix1,matrix2))</li>
        It gives the kronecker product of 2 matrices.<br /><br />
    <li>Flip(.flip(matrix))</li>
        It flips the matrix by 180 degrees.<br /><br />
    <li>Minor*(.minor(matrix,i,j))</li>
        It calculates the minor of a matrix given the index of an element.<br /><br />
    <li>Determinant*(.determinant(matrix))</li>
        It calculates the determinant of a matrix using minors.<br /><br />
    <li>Invert Matrix*(.invertMatrix(matrix))</li>
        It inverts the matrix using the cofactors.<br /><br />
    <li>Vectorize(.vectorize(matrix))</li>
        Vectorizes the matrix by stacking the columns.<br /><br />
    <li>im2row & im2col*(.im2row(matrix,[shape_x,shape_y]) / .im2col(matrix,[shape_x,shape_y]))</li>
        Gives the im2row and im2col expansion using a recursive method.<br /><br />
    <li>Reconstruct Matrix(.reconstructMatrix(array,{x:x,y:y,z:z}))</li>
        It gives the matrix of the specificed dimension from a flat array.<br /><br />
    <li>Normalize(.normalize(matrix,lower_limit,upper_limit))</li>
        It gives the normalized version of the matrix between the specified limits.<br /><br />
    <li>Weighted Sum(.weightedSum(weight,matrix1,matrix2,matrix3,...))</li>
        Takes the element from Matrix1 and adds to the element of Matrix2 * weight and then the result is added to the element of Matrix3 * weight and repeated for all given matrices.
</ol>
#Convolution
This class can compute the convolution of an 3 dimensional array with a filter of 4 dimensions using the im2row operator, more details can be found <a href="https://cs.nju.edu.cn/wujx/paper/CNN.pdf">here</a>. Aside from convolution, It also provides the Input gradients and updates the filter based on the previous gradients and a learning rate.

Convolution
-----------
<h3>.convolution(input, filters, reshape, activation)</h3>
Input is a 3 dimensional input of shape CxHxW <br />
Filters is a 4 dimensional input of shape DxCxH'xW' <br />
Reshape is bool. If true then it is reshaped into DxH"xW" and the activations function is applied to all elements else the output is of shape H"W"xD and the columns are stacked of the output to get the H"W" <br />

```js
const {Convolution, LinearAlgebra} = require('./Neural-Network.js')
const conv = new Convolution
let input = [[
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0]
]] // shape ->  1x6x6
let filter = [
    [[
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]],
    [[
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ]]
] // shape -> 2x1x3x3
output = conv.convolution(input,filter,true,(x)=>x)
console.log(output)
// [
//   [
//       [ 1, 3, 3, 1 ],
//       [ 2, 3, 3, 2 ],
//       [ 2, 3, 3, 2 ],
//       [ 1, 3, 3, 1 ]
//   ],
//   [
//       [ 1, 2, 2, 1 ],
//       [ 3, 3, 3, 3 ],
//       [ 3, 3, 3, 3 ],
//       [ 1, 2, 2, 1 ]
//   ]
// ]
```
<h3>.layerGrads(PreviousGradients)</h3>
PreviousGradients has shape H"W"xD which is same has shape of output if reshape is false in the above convolution.<br />
Returns a matrix of shape CxHxW <br />

```js
let fake_grads = [
    [0,0],[1,0],[0,1],[1,1],[0,0],[1,0],[0,1],[1,0],
    [0,0],[1,1],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]
]
let next_layer_grads = conv.layerGrads(fake_grads)
console.log(next_layer_grads)
// [
//   [
//     [ 0, 0, 0, 0, 0, 0 ],
//     [ 0, 1, 1, 2, 2, 1 ],
//     [ 0, 1, 2, 2, 2, 0 ],
//     [ 1, 4, 5, 5, 4, 1 ],
//     [ 1, 2, 2, 1, 1, 0 ],
//     [ 0, 1, 1, 1, 1, 0 ]
//   ]
// ]
```
If u have PreviousGradients of shape DxH"xW" then you can do this to convert into that format using the LinearAlgebra class
```js
let fake_grads = [
    [
        [0,1,1,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,1,1,0]
    ],
    [
        [0,0,0,0],
        [1,1,1,1],
        [1,1,1,1],
        [0,0,0,0]
    ]
]
const La = new LinearAlgebra
fake_grads = La.vectorize(fake_grads)
fake_grads = La.reconstructMatrix(fake_grads,{x:4*4,y:2,z:1}).flat(1)
fake_grads = La.transpose(fake_grads)
let next_layer_grads = conv.layerGrads(fake_grads)
console.log(next_layer_grads)
// [
//   [
//     [ 0, 0, 1, 1, 0, 0 ],
//     [ 0, 0, 2, 2, 0, 0 ],
//     [ 1, 2, 6, 6, 2, 1 ],
//     [ 1, 2, 6, 6, 2, 1 ],
//     [ 0, 0, 2, 2, 0, 0 ],
//     [ 0, 0, 1, 1, 0, 0 ]
//   ]
// ]
```
<h3>.filterGrads(PreviousGradients, learning_rate)</h3>
PreviousGradients are of the same shape as needed in layer gradient. It updates the filters so nothing else is needed

```js
conv.filterGrads(fake_grads,0.1)
```
<h3>.saveFilters(folder)</h3>
Saves the filters in text format in Filters.txt in the specified folder

```js
conv,saveFilters("path")
```
# Max Pool
Does a max pool on a matrix using the im2row method.<br />
<h3>.pool(inout, size, stride, reshape)</h3>
Input is of shape CxHxW, size and stride are both integers, and reshape is bool.<br />
if reshape is true then output is a matrix, otherwise output will be the vectorized matrix.

```js
const {MaxPool} = require('./Neural-Network.js')
const mxpool = new MaxPool
let input = [[
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0]
]] // shape ->  1x6x6

let output = mxpool.pool(input)//other arguments default to 2,2,and true
console.log(output)
// [
//     [
//         [ 0, 1, 0 ],
//         [ 1, 1, 1 ],
//         [ 0, 1, 0 ]
//     ]
// ]
```
<h3>.layerGrads(PreviousGradients)</h3>
PreviousGradients are of the same shape as needed in the convolution class. The output of the function is the layer gradient of the same format.

```js
let fake_grads = [
    [ 0 ], [ 1 ],
    [ 0 ], [ 1 ],
    [ 5 ], [ 1 ],
    [ 0 ], [ 1 ],
    [ 0 ]
]
let input_grads = mxpool.layerGrads(fake_grads)
console.log(input_grads);
// [
//   [ 0 ], [ 0 ], [ 1 ], [ 0 ], [ 0 ],
//   [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ],
//   [ 0 ], [ 0 ], [ 1 ], [ 0 ], [ 5 ],
//   [ 0 ], [ 1 ], [ 0 ], [ 0 ], [ 0 ],
//   [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ],
//   [ 0 ], [ 1 ], [ 0 ], [ 0 ], [ 0 ],
//   [ 0 ], [ 0 ], [ 0 ], [ 0 ], [ 0 ],
//   [ 0 ]
// ]
```
<h3>.savePool(foler)</h3>
Like the .saveFilters function, this function also saves the pooling details needed for the layerGrads function but its not necessary to save these details and it wont have an effect on the learning of the network but is there just to see how the pooling is being done

```js
mxpool.savePool("path")
```

#Application of CNN
--------------------
In the Application.js file, I have created a simple CNN for mnist number recognition but there are more modules needed to install first. After training finishes then u can start running plot.py to see how the loss and accuracy changed

```cmd
npm install mnist cli-progress
pip install numpy matplotlib
```

#Future Updates
--------------
1) Convolution and other image processing functions    ✔️done
2) Convolutional Neural Network (CNN)    ✔️ done
3) Visualization of Neural Network     ❌ pending (next)
4) Recurrent Neural Network (RNN)     ❌ pending
5) Long Short Term Memory (LSTM)    ❌ pending
