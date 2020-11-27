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
const NeuralNetwork = require('@death_raider/neural-network').NeuralNetwork
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
  learning_rate : 0.1 //learning rate (default = 0.0000001)
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
    learning_rate : 0.5
  })
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
const NeuralNetwork = require('./Neural Network/Neural-Network.js')
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

# Image Processing
Some basic image processing and augmentation can help increase the dataset size while training networks and helps in better generalizations.

Strarting up
------------
We require the package to use
```js
const ImageProcessing = require('@death_raider/neural-network').ImageProcessing
let augmentation = new ImageProcessing()
```

Creating Matricies
------------------
 We can create a C x H x W matrix using this function where C is the number of H x W matricies. The matrix will be between -1 and 1
```js
let matrix = augmentation.createMatrix(3,3,3);
console.log("matrix",matrix);
/*
matrix [
  [
    [ -0.678372439890254, 0.6500779334590403, -0.4565339688601684 ],
    [ -0.8359020914470849, -0.8694483636477246, 0.16924704621900544 ],
    [ -0.5988398284917427, -0.4382417489861301, -0.32707306499371747 ]
  ],
  [
    [ -0.8922996259757201, -0.826901601847978, -0.7183706165039161 ],
    [ 0.7109725262462825, 0.18541035867637534, 0.23335619400348895 ],
    [ -0.732994280493128, -0.4293017471231919, -0.30308384999392013 ]
  ],
  [
    [ 0.4725701029130227, -0.4855206864171042, 0.2166882512222812 ],
    [ 0.9341185521019892, 0.8573285355870115, -0.7320992421323922 ],
    [ -0.043161461538433255, -0.37925479628203806, 0.8177539982172051 ]
  ]
]
*/
```
Convolution
-----------
We can convolve an image matrix in 2 ways:
1) H x W convolution
```js
let matrix = [[1,1,1],[1,1,1],[1,1,1]]
let conv = augmentation.Convolution({
  matrix: matrix, //matrix type H x W
  filter: [  // filter type  H x W
    [1,1],
    [1,1]
  ],
  bias: 0, // bias
  step: {x:1,y:1}, // stride to move the filter
  padding: 0, // amount to add the input matrix with 0
  type: "conv", // can be "conv" or "max_pool"
  activation: "relu" // can be "linear","relu" or "sigmoid"
})
console.log("single conv->",conv);
// single conv-> [ [ 4, 4 ], [ 4, 4 ] ]
```

2) C x H x W convolution
This is for multi channel convolution with varing feature maps.
In this example we have a 2 x 3 x 3 input image and we convlve it with a 2 x 2 x 2 filter to get a 2 x 2 x 2 x 2 so each channel of the input
got convolved with the 2 filters.
```js
let matrix = [
  [[1,1,1],[1,1,1],[1,1,1]], // channel 1
  [[0,0,0],[1,1,1],[0,0,0]]  // channel 2
]
let convMultiChannel = augmentation.convolutionLayers({
  matrix: matrix, //matrix type C x H x W
  kernal: [  //kernal type C x H x W
    [
      [0,1],
      [0,0]
    ],
    [
      [0,0],
      [1,0]
    ]
  ],
  featureMaps: 2, // between 0 and kernal.length
  stride:{x:1,y:1},
  padding:0,
  bias:0,
  type:"conv",
  activation: "relu"
})
console.log("multi conv",convMultiChannel) //output (channels x convFeature maps x H_new x W_new)
convMultiChannel.map(e=>console.log(e))
/*
multi conv [
  [ [ [Array], [Array] ], [ [Array], [Array] ] ],
  [ [ [Array], [Array] ], [ [Array], [Array] ] ]
]
[ [ [ 1, 1 ], [ 1, 1 ] ], [ [ 1, 1 ], [ 1, 1 ] ] ]
[ [ [ 0, 0 ], [ 1, 1 ] ], [ [ 1, 1 ], [ 0, 0 ] ] ]
*/
```

Future Updates
--------------
1) Convolution and other image processing functions    ✔️done
2) Convolutional Neural Network (CNN)    ❌ pending (next)
3) Visulization of Neural Network     ❌ pending (next)
4) Recurrent Neural Network (RNN)     ❌ pending
5) Long Short Term Memory (LSTM)    ❌ pending
6) Proper documentation    ❌ pending
