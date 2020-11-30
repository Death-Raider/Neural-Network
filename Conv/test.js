const {NeuralNetwork,ImageProcessing} = require("@death_raider/neural-network")

class ConvolutionNeuralNetwork {
  constructor({conv_layers = [],feature_maps = [],strides = [],ImageShape={x:1,y:1},layer_nodes=[],output_nodes=0}={}){
    this.network = conv_layers;
    this.strides = strides;
    feature_maps.map(e=>{
      if(e!==1)throw "Err: Currently only 1 feature map supported per layer"
    })
    conv_layers.map(e=>{
      if(e[0] === "max_pool")throw "Err: Currently only supports convolution"
    })
    this.augmentation = new ImageProcessing();
    this.featureMapCount = feature_maps;
    this.Filters = [];
    this.ConvLayers;
    this.learning_rate_convolution;
    this.FullyConnected;
    this.FCshape;
    this.LayerGrad;
    this.FilterGrad = [];
    this.Bias = [];
    this.inputShape = ImageShape;
    for(let i = 0; i < this.network.length; i++){
      let layer = this.network[i];
      this.Filters[i] = this.augmentation.createMatrix(this.featureMapCount[i],layer[layer.length-1],layer[layer.length-1]);
      this.Bias[i] = Math.random()*0.02 - 0.01
    }
    let outputSize = this.calcSize()
    this.FCnetwork = new NeuralNetwork({
      input_nodes : outputSize,
      layer_count : layer_nodes,
      output_nodes : output_nodes,
      weight_bias_initilization_range : [-1,1]
    });
    // this.FCnetwork.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
  }
  forwordPass({image,desired_outputs,learning_rate_convolution}){
    if(image.length !== 1) throw "Err: Currently only type (1 x H x W) input supported"
    this.ConvLayers = {Layer_0 : image}
    this.learning_rate_convolution = learning_rate_convolution;
    //parsing input given and applying filters (aka feed forward of CNN)
    let biasMax = Math.max(Math.abs(...this.Bias))
    for(let i = 0; i < this.network.length; i++){
      let layer = this.network[i];
      this.ConvLayers[`Layer_${i+1}`] = this.augmentation.convolutionLayers({
        matrix : this.ConvLayers[`Layer_${i}`],
        kernal : this.Filters[i],
        featureMaps : this.featureMapCount[i],
        stride : this.strides[i],
        padding : 0,
        bias : this.Bias[i],
        type : layer[0],
        activation : (layer[0] !== "max_pool")?layer[1]:"relu"
      }).flat()
    }
    //reshaping inputs for neural network training
    let last_layer_conv = Object.keys(this.ConvLayers)[Object.keys(this.ConvLayers).length - 1];
    [this.FullyConnected,this.FCshape] = this.augmentation.flattenImage((this.network[this.network.length - 1][0] === "max_pool")?this.ConvLayers[last_layer_conv][0]:this.ConvLayers[last_layer_conv])
    //train once
    let dnn = this.FCnetwork.trainIteration({
      input : this.FullyConnected,
      desired : desired_outputs,
      learning_rate : 0.05
    })
    console.log(dnn.Cost);
    return dnn
  }
  backwordPass(){
    //reshaping inputs to matrix for convolutional training
    this.LayerGrad = new Array(Object.keys(this.ConvLayers).length).fill(0).map(e=>0)
    this.FilterGrad = new Array(this.Filters.length).fill(0).map(e=>0)
    this.LayerGrad[this.LayerGrad.length-1] = this.FCnetwork.getInputGradients()
    this.LayerGrad[this.LayerGrad.length-1] = this.augmentation.reconstructMatrix(this.LayerGrad[this.LayerGrad.length-1],this.FCshape).flat()
    //backword pass for CNN
    for(let i = this.Filters.length-1; i > -1 ; i--){
      let flippedFilter = this.augmentation.Flip(this.Filters[i][0]);
      this.LayerGrad[i] = this.augmentation.Convolution({
        matrix : this.LayerGrad[i+1],
        filter : flippedFilter,
        step : {x:1,y:1},
        padding : flippedFilter.length-1,
        type : "conv",
        activation : "linear"
      });
      this.FilterGrad[i] = this.augmentation.Convolution({
        matrix : this.ConvLayers[`Layer_${i}`][0],
        filter : this.LayerGrad[i+1],
        step : {x:1,y:1},
        padding : 0,
        type : "conv",
        activation : "linear"
      });

      this.Bias[i] -= this.learning_rate_convolution*this.LayerGrad[i+1].flat().reduce((a,b)=>a+b)
    }
    this.FilterGrad = this.FilterGrad.map(e=>[e])
    this.Filters = this.addMatrix(this.Filters,this.FilterGrad,this.learning_rate_convolution)
  }
  addMatrix(M1,M2,constant){ // M1 to be updated from M2
    for(let z in M1){
      for(let y in M1[z]){
        for(let x in M1[z][y]){
          for(let w in M1[z][y][x]){
            M1[z][y][x][w] += -constant*M2[z][y][x][w];
          }
        }
      }
    }
    return M1
  }
  calcSize(){
    let sumFilterLen = 0;
    for(let i = 0; i < this.Filters.length;i++){
      let layer = this.network[i];
      sumFilterLen += layer[layer.length-1];
    }
    let num = this.Filters.length - sumFilterLen
    return (this.inputShape.x + num)*(this.inputShape.y + num);
  }
}
const fs = require('fs');
const spawn = require('child_process').spawn;

try{
  (async()=>{
    let x = new ConvolutionNeuralNetwork({
      conv_layers :  [["conv","relu",4],["conv","relu",3]], //[type,activation,filterSize]
      feature_maps : [1,1], // number of feature maps
      strides :      [{x:1,y:1},{x:1,y:1},{x:1,y:1}], //strides for the filters
      layer_nodes : [], //for fully connected part
      output_nodes : 10, //for fully connected part
      ImageShape : {x:8,y:12} //input image shape
    });
    let y = new NeuralNetwork({
      input_nodes : 96,
      layer_count : [7,7],
      output_nodes : [10],
      weight_bias_initilization_range : [-0.1,0.1]
    });
    let lossC = []
    let lossN = []

    for(let k = 0; k < 10000; k++){
      let rand = Math.floor(Math.random()*10)//random class (0-9)
      let rand2 = Math.floor(Math.random()*500)
      let out = new Array(10).fill(0).map(e=>0)
      out[rand] = 1
      inp = [(await x.augmentation.processImage(`Train_Images/${rand}'s/output/${rand2}(${rand}).bmp`))[0]]

      let dnn = x.forwordPass({
        image : inp,
        desired_outputs : out,
        learning_rate_convolution : 0.03
      })
      x.backwordPass()

      let ann = y.trainIteration({
        input : inp[0].flat(),
        desired : out,
        learning_rate : 0.05
      })

      let filterCreate = new Promise(function(resolve, reject) {
        callFile("python","filters.py",JSON.stringify(x.Filters.flat()),resolve,reject)
      });
      let imageCreate = new Promise(function(resolve, reject) {
        callFile("python","image.py",JSON.stringify(x.ConvLayers),resolve,reject)
      });
      await filterCreate;
      await imageCreate;

      console.log("iteration ->",k, "label",rand,out);
      lossC.push(dnn.Cost)
      lossN.push(ann.Cost)

      fs.writeFileSync("costC.txt",JSON.stringify(lossC))
      fs.writeFileSync("costN.txt",JSON.stringify(lossN))
    }
    console.log(x);
    console.log(x.ConvLayers);
    console.log(x.FCnetwork);

  })()
}catch(e){console.log(e)}

function callFile(open,filePath,sendData = 'no Data',resolve,reject){
  const test = spawn(open,[filePath]);
  let x;

  test.stdin.write(sendData);
  test.stdin.end();

  test.stdout.on('data',(data) =>{
    x = data.toString('utf8');
  });
  test.stdout.on('end',()=>{
    resolve(x)
  });
  test.stderr.on('error',(err)=>{
    console.log(err);
    reject(err)
  });
}
