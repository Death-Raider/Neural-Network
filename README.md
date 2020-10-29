# Neural Network
An easy to use Neural Network package with SGD using backpropergation as a gradient computing technique.

Creating the model
------------------
```js
const NeuralNetwork = require('./Neural Network/Neural-Network.js')
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
For this example we'll be testing it on the XOR function
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
To use the network:
```js
// network.use(inputs)
let output = [ //truth table for xor gate
  network.use([0,0]),
  network.use([0,1]),
  network.use([1,0]),
  network.use([1,1])
]
console.log("truth table ->",output)
// truth table -> [
//   [ 0.006628016503533872 ],
//   [ 0.9924992698592481 ],
//   [ 0.9925101128583149 ],
//   [ 0.006998581457622981 ]
// ]
```
Saving and Loading Models
-------------------------
