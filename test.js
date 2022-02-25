const NeuralNetwork = require('@death_raider/neural-network').NeuralNetwork
//creates ANN with 2 input nodes, 1 hidden layers with 2 hidden nodes and 1 output node
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2],
  output_nodes :1,
  weight_bias_initilization_range : [-1,1]
});
//format for activation function = [ function ,  derivative of function ]
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)] //sets activation for hidden layers as sigmoid function
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

console.log("Average Validation Loss ->",network.Loss.Validation_Loss.reduce((a,b)=>a+b)/network.Loss.Validation_Loss.length);

for(let i = 0; i < 10000; i++){
  let [inputs,outputs] = xor()
  let dnn = network.trainIteration({
    input : inputs,
    desired : outputs,
    learning_rate : 0.5
  })
  console.log(dnn.Cost,dnn.layers); //optional to view the loss and the hidden layers
}

let output = [ //truth table for xor gate
  network.use([0,0]),
  network.use([0,1]),
  network.use([1,0]),
  network.use([1,1])
]

console.log( network.getInputGradients() );

network.save("path")

(async () =>{
  await network.load("path") //make sure network is of correct structure
  let output = [
    network.use([0,0]),  // --> returns the hidden node values as well
    network.use([0,1]),  // --> returns the hidden node values as well
    network.use([1,0]),  // --> returns the hidden node values as well
    network.use([1,1])   // --> returns the hidden node values as well
  ]
})()
const ImageProcessing = require('@death_raider/neural-network').ImageProcessing
let augmentation = new ImageProcessing()
let matrix = augmentation.createMatrix(3,3,3);
console.log("matrix",matrix);
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
let matrix = [
  [1,2,3,4],
  [5,6,7,8],
  [9,10,11,12],
  [13,14,15,16]
]
let flipped = augmentation.Flip(matrix) // matrix type H x W
console.log("flipped",flipped);
let matrix = [
  [[1,2],[3,4]], // channel 1
  [[0.5,0.5],[0.5,0.5]], // channel 2
]
augmentation.Normalize(matrix)  // needs type (C x H x W)
console.log("normal matrix",matrix);
augmentation.Normalize(matrix)
console.log("normal matrix",matrix);

matrix = [
  [[-5,-3],[6,7]] // chanel 1
]
augmentation.Normalize(matrix)
console.log("normal matrix",matrix);
matrix =[
  [ // channel 1
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
  ],
  [ // channel 2
    [-1,-2,-3,-4],
    [-5,-6,-7,-8],
    [-9,-10,-11,-12],
    [-13,-14,-15,-16]
  ]
]

let flat = augmentation.flattenImage(matrix) // needs type (C x H x W)
console.log(flat);
let flatArr = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
let reconstruct = augmentation.reconstructMatrix(flatArr,{z:4,y:2,x:2})
console.log(reconstruct);
