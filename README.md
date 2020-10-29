# Neural-Network
An easy to use Neural Network package with SGD using backpropergation as a gradient computing technique.

Creating the model
------------------
```js
const NeuralNetwork = require('.Neural Network/Neural-Network.js')
//creates ANN with 2 inputs, 2 hidden layers with 2 hidden nodes each and 2 output nodes
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2,2],
  output_nodes :2,
  weight_bias_initilization_range : [-1,1] 
});

```
