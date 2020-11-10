//taking element properties from canvas
const c = document.getElementById("canvas");
const ctx = c.getContext("2d");
const radius = 10;
const buffer = 5;
let hiddenLayerCountHTML;
let hiddenNodeCountHTML;
let outputCountHTML;
let inputCountHTML;
let spaceLayer;
let nodeCount = 0;
let layer = 0;
let Batch = 32;

function plotNN(){
  clear(ctx,c);

  spaceLayer = parseFloat(document.getElementById("layerspace").value);
  inputCountHTML = parseInt(document.getElementById("inputCount").value);
  outputCountHTML = parseInt(document.getElementById("outputCount").value);
  hiddenNodeCountHTML = parseInt(document.getElementById("hiddenNodeCount").value);
  hiddenLayerCountHTML = parseInt(document.getElementById("hiddenLayerCount").value);
  epochsHTML = parseInt(document.getElementById("training").value);

  //addes the nodes for the input layer
  for(let i = 0 ;i < inputCountHTML; i++) addNode();
  //new layer begins
  addLayer();
  //adds the nodes for all of the hidden layes
  for(let i = 0; i < hiddenLayerCountHTML; i++){
    for(let j = 0; j < hiddenNodeCountHTML; j++) addNode();
    addLayer();
  }
  //addes the nodes for the output layer
  for(let i = 0; i < outputCountHTML; i++) addNode();
  //draws all the weights
  addWeights();
  let hidden = new Array(hiddenLayerCountHTML).fill(hiddenNodeCountHTML)
  let network = new NeuralNetwork({
    input_nodes : inputCountHTML,
    layer_count : hidden,
    output_nodes : outputCountHTML,
    weight_bias_initilization_range : [-1,1]
  });
  let funcStr = eval(document.getElementById("trainFunc").value);
  network.train({
    TotalTrain : epochsHTML,
    trainFunc : funcStr,
    learning_rate : 0.1,
    TotalVal : 100*Batch,
    validationFunc : funcStr,
    batch_train: Batch
  });
  //variable color introduced will contain the hex values of the colors of the weights
  var color;
  //it gets the color and then colors that specific weight accoring to their value
  for(let j = 0; j < network.Weights.length; j++){
    for(let k = 0; k < network.Weights[j].length; k++){
      for(let p = 0; p < network.Weights[j][k].length; p++){
        color = addColor(network.Weights[j][k][p]);
        individualWeightAccessAndOverlap(color,p,k,j);
      }
    }
  }

  let outputs = network.Nodes[hiddenLayerCountHTML + 1];
  for(let i = 0; i < 100; i++ ){
    let myobj1 = document.getElementById(`${i}`);
    let myobj2 = document.getElementById(`space ${i}`);
    if(myobj1 !== null) myobj1.remove();
    if(myobj2 !== null) myobj2.remove();
  }
  if(document.getElementsByClassName("disabled-values").length === 1){ //that one is for the error one
    for(let i = 0; i < outputs.length; i++){
      let outputBox = document.createElement("input");
      outputBox.size = 30;
      outputBox.value = `Output_${i} : ${outputs[i]}`;
      outputBox.className = "disabled-values";
      outputBox.id = `${i}`;
      outputBox.disabled = true;
      let space = document.createElement("BR");
      space.id = `space ${i}`;
      document.getElementById('inputsNetwork').append(outputBox);
      document.getElementById('inputsNetwork').append(space);
    }
  }
  document.getElementById("Error").value = network.Loss.Validation_Loss.reduce((a,b)=>a+b)/network.Loss.Validation_Loss.length;
  console.log(network.Weights);
  //resets the layer count and also the node cound so the fucntion can be called again
  layer = 0;
  nodeCount = 0;
  clear(ctx_error,c_error)
  plotaxis(ctx_error,c_error,network.Loss.Train_Loss)
}
