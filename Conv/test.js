class NeuralNetwork {
  constructor({input_nodes,layer_count,output_nodes,weight_bias_initilization_range=[-0.001,0.001] } = {}){
    if(input_nodes === undefined || layer_count === undefined || output_nodes === undefined) throw "Error: structural values not given"
    let parameters=createParameters(input_nodes,layer_count,output_nodes,weight_bias_initilization_range[0],weight_bias_initilization_range[1]);
    const copyRadar3D = (y,z = []) =>{for (let _a in y) for (let _b in y[_a]) {if(!z[_a]) z[_a] = []; z[_a][_b] = y[_a][_b].slice();};return z}
    const copyRadar2D = (y,z = []) =>{for (let _a in y){if(!z[_a]) z[_a] = []; z[_a] = y[_a].slice();};return z}
    this.HiddenLayerCount=layer_count;
    this.Weights = copyRadar3D(parameters[0])
    this.WeightUpdates = copyRadar3D(parameters[0])
    this.Bias = copyRadar2D(parameters[1])
    this.BiasUpdates = copyRadar2D(parameters[1])
    this.Activation = {
      hidden:[(x)=>(x>0)?x:x*0.1,(x)=>(x>0)?1:0.1],
      output:[(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
    }
    function createParameters(input,LayerCount,output,a,b){
      let MatrixW=[], MatrixB=[];
      for(let j = 0; j < LayerCount.length + 1; j++){
        MatrixW[j]=[]; MatrixB[j]=[];
        for(let i = 0; i < ((j == LayerCount.length)?output:LayerCount[j]);i++){
          MatrixW[j][i]=[];
          MatrixB[j][i]=Math.random()*(b-a)+a;
          for(let k = 0; k < ((j == 0)?input:LayerCount[j-1]); k++) MatrixW[j][i][k]=Math.random()*(b-a)+a;
        }
      }
      return [MatrixW,MatrixB];
    }
  }
  GetLayerValues(InputVector,Activation){//Forword Pass
    let activated = [];
    activated[0]=InputVector.slice();
    for(let i = 0; i < this.Weights.length; i++){
      activated[i+1]=[];
      for(let j = 0,sum = 0; j < this.Weights[i].length; sum = 0,j++){
        for(let k = 0; k < this.Weights[i][j].length; k++) sum+=this.Weights[i][j][k]*activated[i][k];
        activated[i+1].push(Activation[(i == this.Weights.length-1)?1:0](sum+this.Bias[i][j]) );
      }
    }
    return activated;
  }
  changes(Desired,Output,DerivativeActivation){//backword pass
    let cost=0;
    for(let i = 0; i < Desired.length; i++) cost+=0.5*Math.pow(Output[i]-Desired[i],2);
    for(let i = 0; i < this.Nodes[this.HiddenLayerCount.length + 1].length; i++){
      this.BiasUpdates[this.Weights.length-1][i]=(this.Nodes[this.HiddenLayerCount.length+1][i]-Desired[i])*DerivativeActivation[1](this.Nodes[this.HiddenLayerCount.length+1][i]);
      for(let j = 0; j < this.Nodes[this.HiddenLayerCount.length].length; j++) this.WeightUpdates[this.Weights.length-1][i][j]=(this.BiasUpdates[this.Weights.length-1][i]*this.Nodes[this.HiddenLayerCount.length][j]);
    }
    for(let j = this.Weights.length - 2; j > -1; j--){//iterates of all layers except the last one
      for(let k = 0,sum = 0; k < this.Weights[j].length; k++,sum = 0){
        for(let m = 0; m < this.Weights[j+1].length; m++) sum+=this.Weights[j+1][m][k]*this.WeightUpdates[j+1][m][k];
        for(let p = 0; p < this.Weights[j][k].length; p++) this.WeightUpdates[j][k][p]=(sum*(DerivativeActivation[0](this.Nodes[j+1][k]))*this.Nodes[j][p])/((this.Nodes[j+1][k]==0)?1:this.Nodes[j+1][k]);
        this.BiasUpdates[j][k]=(sum*(DerivativeActivation[0](this.Nodes[j+1][k])))/((this.Nodes[j+1][k]==0)?1:this.Nodes[j+1][k]);
      }
    }
    return {updatedWeights:this.WeightUpdates,updatedBias:this.BiasUpdates,Cost:cost};
  }
  getInputGradients(grad = []){
    for(let k = 0,sum = 0; k < this.Nodes[0].length; k++,sum = 0){
      for(let m = 0; m < this.Weights[0].length; m++) sum += this.Weights[0][m][k]*this.WeightUpdates[0][m][k];
      grad[k] = sum/((this.Nodes[0][k]===0)?1:this.Nodes[0][k]);
    }
    return grad
  }
  update(secondTensor,secondMatrixBias,rate){//Readjustment of weights and bias
    for(let i = 0; i < secondTensor.length; i++){
      for(let j = 0; j < secondTensor[i].length; j++){
        for(let k = 0; k < secondTensor[i][j].length; k++) this.Weights[i][j][k]-=rate*secondTensor[i][j][k];
        this.Bias[i][j]-=rate*secondMatrixBias[i][j];
      }
    }
  }
  train({TotalTrain=0,trainFunc=()=>{},TotalVal=0,validationFunc=()=>{},learning_rate=0.0005,batch_train = 1,batch_val = 1} = {}){
    let cost=[], cost_val=[], changing = [];
    for(let i = 0; i < parseInt((TotalTrain/batch_train+TotalVal/batch_val)); i++){
      let batch = (i < parseInt(TotalTrain/batch_train))?batch_train:batch_val;
      let sumCost = 0;
      for(let b = 0; b < batch ; b++){
        let [input,desir] = (i <= parseInt(TotalTrain/batch_train))?trainFunc(b,i):validationFunc(b,i);
        this.Nodes=this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]]);
        changing[b]=this.changes(desir,this.Nodes[this.Nodes.length-1],[this.Activation.hidden[1],this.Activation.output[1]]);
        sumCost += changing[b].Cost/batch
      }
      if(i < parseInt(TotalTrain/batch_train)){
        cost.push(sumCost);
        for(let x = 0; x < changing[0].updatedWeights.length; x++){
          for(let y = 0,sumBias = 0; y < changing[0].updatedWeights[x].length; y++,sumBias = 0){
            for(let z = 0,sumWeight = 0; z < changing[0].updatedWeights[x][y].length; z++,sumWeight = 0){
              for(let b = 0; b < batch; b++)sumWeight += changing[b].updatedWeights[x][y][z]/batch
              this.WeightUpdates[x][y][z] = sumWeight;
            }
            for(let b = 0; b < batch; b++)sumBias += changing[b].updatedBias[x][y]/batch
            this.BiasUpdates[x][y] = sumBias;
          }
        }
        this.update(this.WeightUpdates,this.BiasUpdates,learning_rate);
      }else{cost_val.push(sumCost)}
    }
    this.Loss = {Train_Loss:cost,Validation_Loss:cost_val};
  }
  trainIteration({input,desired,learning_rate=0.0001}={}){
    this.Nodes = this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]]);
    let changing = this.changes(desired,this.Nodes[this.Nodes.length-1],[this.Activation.hidden[1],this.Activation.output[1]]);
    this.update(this.WeightUpdates,this.BiasUpdates,learning_rate);
    return {Cost:changing.Cost,layers:this.Nodes}
  }
  use(input){return this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]])}
  save(folder){
    const fs = require('fs')
    if(!fs.existsSync(folder)) fs.mkdirSync(folder);
    writeData(this.Weights,folder,"Weights",'W')
    writeData(this.Bias,folder,"Bias",'B')
    function writeData(arr,folder,slice,initilizer){
      if(!fs.existsSync(`${folder}/${slice}`)) fs.mkdirSync(`${folder}/${slice}`);
      for(let i = 0; i < arr.length; i++){
        let stream = fs.createWriteStream(`${folder}/${slice}/${i}.txt`);
        if(initilizer === 'B'){
          stream.write(JSON.stringify(arr[i]))
          stream.write("\n")
        }
        if(initilizer === 'W'){
          for(let j = 0; j < arr[i].length; j++){
            stream.write(JSON.stringify(arr[i][j]))
            stream.write("\n")
          }
        }
        stream.end();
      }
      console.log(`done saving ${(initilizer === 'W')?"Weights":"Bias"}`);
    }
  }
  async load(path){
    function getLines(folder,res){
      const readline  = require('readline');
      const fs = require('fs');
      let sub_dir = fs.readdirSync(folder)
      let readInterface, lines=[];
      for(let s in sub_dir){
        let files = fs.readdirSync(`${folder}/${sub_dir[s]}`)
        lines[s] = []
        for(let f in files){
          lines[s][f] = [];
          readInterface = readline.createInterface({input: fs.createReadStream(`${folder}/${sub_dir[s]}/${files[f]}`)});
          readInterface.on('line', function(line){lines[s][f].push(JSON.parse(line))});
        }
      }
      readInterface.on('close', ()=>{res(lines)});
    }
    let model = new Promise(function(resolve, reject){getLines(path,resolve)});
    model = await model;
    this.Bias = model[0].flat();
    this.Weights = model[1];
  }
}
class ConvolutionNeuralNetwork {
  constructor({conv_layers = [],feature_maps = [],strides = []}={}){
    this.network = conv_layers;
    this.strides = strides;
    feature_maps.map(e=>{
      if(e!==1)throw "Err: Currently only 1 feature map supported per layer"
    })
    conv_layers.map(e=>{
      if(e[0] === "max_pool")throw "Err: Currently only supports convolution"
    })
    this.featureMapCount = feature_maps;
    this.Filters = [];
    this.ConvLayers;
    this.learning_rate;
    this.FullyConnected;
    this.FCshape;
    this.LayerGrad;
    this.FilterGrad = [];
    this.Bias = [];
    for(let i = 0; i < this.network.length; i++){
      let layer = this.network[i];
      this.Filters[i] = this.createMatrix(layer[layer.length-1],layer[layer.length-1],this.featureMapCount[i]);
      this.Bias[i] = Math.random()*0.02 - 0.01
    }
  }
  Convolution({matrix,filter,bias = 0,step = {x:1,y:1},padding = 0,type="conv",activation = "linear"} = {}){
    let activationBank = {
      relu:[(x)=>(x>0)?x:0,(x)=>(x>0)?1:0],
      sigmoid:[(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)],
      linear:[(x)=>x,(x)=>1]
    };
    let Activation = activationBank[activation.toLowerCase()] //linear activation and derivative
    //dot product of filter and the selected image section
    const filterDot = (x,y,M,F) => {
      let sum = 0;
      for(let f1 = 0; f1 < F.length; f1++){
        for(let f2 = 0; f2 < F[f1].length; f2++){
          sum += F[f1][f2]*M[f1+y*step.y][f2+x*step.x];
        }
      }
      return sum;
    }
    // apply padding to matrix
    const padMatrix = (matrix,pad,val) => {
      let padded_matrix = new Array(matrix.length + 2*pad).fill(0).map(e=>new Array(matrix[0].length + 2*pad).fill(val));
      for(let i = pad; i < matrix.length + pad; i++)
        for(let j = pad; j < matrix[i-pad].length + pad; j++) padded_matrix[i][j] = matrix[i-pad][j-pad];
      return padded_matrix;
    }
    //special case for max_pool
    if(type==="max_pool"){
      filter = [[0,0],[0,0]];
      bias = 0;
      step = {x:2,y:2}
    }
    if(matrix === undefined || filter === undefined) throw "Err: Input matrix or filter not specified"
    //adds padding to matrix
    matrix = padMatrix(matrix,padding,0)
    let mask = []; //use to create a mask when max_pooling for gradient transfer
    let outputSize = {
      y : 1+(matrix.length - filter.length)/step.y,
      x : 1+(matrix[0].length - filter[0].length)/step.x
    }
    //checking if convolution is possible
    if(outputSize.y-Math.floor(outputSize.y) !== 0 || outputSize.x-Math.floor(outputSize.x) !== 0 ) throw "Err: size not compatible with " + type;
    let output = new Array(outputSize.y).fill(0).map(e=>new Array(outputSize.x).fill(0))//convoluted output
    //types of convolutions
    for(let y = 0; y < outputSize.y; y++){
      mask[y] = []
      for(let x = 0; x < outputSize.x; x++){
        if(type==="conv") output[y][x] = Activation[0](filterDot(x,y,matrix,filter) + bias);
        else if(type==="max_pool"){
          mask[y][x] = {value:0,position:{x:0,y:0}};
          for(let f1 = 0; f1 < 2; f1++){
            for(let f2 = 0; f2 < 2; f2++){
              if(matrix[f1+y*step.y][f2+x*step.x]>=mask[y][x].value){
                mask[y][x].value = matrix[f1+y*step.y][f2+x*step.x];
                mask[y][x].position.x = f2+x*step.x;
                mask[y][x].position.y = f1+y*step.y;
              }
            }
          }
          output[y][x] = mask[y][x].value + bias
        }
        else throw "Err: unidentified type"
      }
    }
    return (type==="max_pool")?[output,mask.flat()]:output
  }
  //func for CNN
  convolutionLayers({matrix,kernal,featureMaps,stride,padding=0,bias=0,type,activation} = {}){
    let conv1 = []
    for(let j = 0; j < matrix.length; j++){
      conv1[j] = []
      for(let i = 0; i < featureMaps; i++){
        conv1[j][i] = this.Convolution({
          matrix : matrix[j],
          filter : kernal[i],
          step : stride,
          padding : padding,
          type : type,
          bias : bias,
          activation : activation
        })
      }
    }
    return conv1.flat()
  }
  forwordPass({image,layer_nodes,output_nodes,desired_outputs,learning_rate}){
    if(image.length !== 1) throw "Err: Currently only type (1 x H x W) input supported"
    this.ConvLayers = {Layer_0 : image}
    this.learning_rate = learning_rate;
    //parsing input given and applying filters (aka feed forward of CNN)
    let biasMax = Math.max(Math.abs(...this.Bias))
    for(let i = 0; i < this.network.length; i++){
      let layer = this.network[i];
      this.ConvLayers[`Layer_${i+1}`] = this.convolutionLayers({
        matrix : this.ConvLayers[`Layer_${i}`],
        kernal : this.Filters[i],
        featureMaps : this.featureMapCount[i],
        stride : this.strides[i],
        padding : 0,
        bias : this.Bias[i],
        type : layer[0],
        activation : (layer[0] !== "max_pool")?layer[1]:"relu"
      })
    }
    //reshaping inputs for neural network training
    let last_layer_conv = Object.keys(this.ConvLayers)[Object.keys(this.ConvLayers).length - 1];
    [this.FullyConnected,this.FCshape] = this.flattenFeatureMaps((this.network[this.network.length - 1][0] === "max_pool")?this.ConvLayers[last_layer_conv][0]:this.ConvLayers[last_layer_conv])
    let max = Math.max(...this.FullyConnected)
    console.log("max  ", max);
    if(max > 1) for(let i in this.FullyConnected)this.FullyConnected[i] /= max
    //creating neural network for fully connected parts
    this.FCnetwork = new NeuralNetwork({
      input_nodes : this.FullyConnected.length,
      layer_count : layer_nodes,
      output_nodes : output_nodes,
      weight_bias_initilization_range : [-0.1,0.1]
    });
    this.FCnetwork.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
    //train once
    let dnn = this.FCnetwork.trainIteration({
      input : this.FullyConnected,
      desired : desired_outputs,
      learning_rate : 0.5
    })
    console.log(dnn.Cost, dnn.layers);
    return dnn
  }
  backwordPass(){
    //reshaping inputs to matrix for convolutional training
    this.LayerGrad = new Array(Object.keys(this.ConvLayers).length).fill(0).map(e=>0)
    this.FilterGrad = new Array(this.Filters.length).fill(0).map(e=>0)

    this.LayerGrad[this.LayerGrad.length-1] = this.FCnetwork.getInputGradients()
    this.LayerGrad[this.LayerGrad.length-1] = this.reconstructMatrix(this.LayerGrad[this.LayerGrad.length-1],this.FCshape)
    //backword pass for CNN
    for(let i = this.Filters.length-1; i > -1 ; i--){
      this.LayerGrad[i] = this.Convolution({
        matrix : this.LayerGrad[i+1],
        filter : this.Rotate(this.Filters[i][0]),
        step : {x:1,y:1},
        padding : this.Filters[i][0].length-1,
        type : "conv",
        activation : "linear"
      });
      this.Rotate(this.Filters[i][0])//rotates again so its "normal"
      this.FilterGrad[i] = this.Convolution({
        matrix : this.ConvLayers[`Layer_${i}`][0],
        filter : this.LayerGrad[i+1],
        step : {x:1,y:1},
        padding : 0,
        type : "conv",
        activation : "linear"
      });
      this.Bias[i] -= this.learning_rate*this.LayerGrad[i+1].flat().reduce((a,b)=>a+b)
    }
    this.FilterGrad = this.FilterGrad.map(e=>[e])
    this.Filters = this.addMatrix(this.Filters,this.FilterGrad,this.learning_rate)
  }
  //matrix manupliation
  flattenFeatureMaps(featureMapMatrix){
    let connected = [];
    let shape = {y:featureMapMatrix[0].length,x:featureMapMatrix[0][0].length}
    for(let featurePlane of featureMapMatrix) connected.push(featurePlane.flat())
    return [connected.flat(),shape]
  }
  async processImage(path){
    const pixels = require('image-pixels')
    let {data, width, height} = await pixels(path)

    let local_newImg_R = [], local_newImg_G = [], local_newImg_B = [],
    newImg_R = [], newImg_G = [], newImg_B = [];
    for(let i = 0; i < data.length/4; i++){
      local_newImg_R[i] = data[4*i]/255;
      local_newImg_G[i] = data[4*i+1]/255;
      local_newImg_B[i] = data[4*i+2]/255;
    }
    //formatting into correct form
    for(let i = 0; i < local_newImg_R.length/width; i++){
      newImg_R[i] = [];
      newImg_G[i] = [];
      newImg_B[i] = [];
      for(let j = 0; j < width; j++){
        newImg_R[i][j] = local_newImg_R[j + width*i]
        newImg_G[i][j] = local_newImg_G[j + width*i]
        newImg_B[i][j] = local_newImg_B[j + width*i]
      }
    }
    return [newImg_R,newImg_G,newImg_B]
  }
  createMatrix(x,y,z){
    let M = []
    for(let i = 0; i < z; i++){
      M[i] = []
      for(let j = 0; j < y; j++){
        M[i][j] = []
        for(let k = 0; k < x; k++){
          M[i][j][k] = Math.random()*2 - 1
        }
      }
    }
    return M;
  }
  reconstructMatrix(flatArr,m,Matrix=[]){
    for(let i = 0; i < m.y; i++){
      Matrix[i] = []
      for(let j = 0; j < m.x; j++){
        Matrix[i][j] = flatArr[j + m.x*i]
      }
    }
    return Matrix
  }
  Rotate(matrix){ // rotates matrix by 180
    for(let i = 0; i < Math.floor(matrix.length/2); i++){
      let a = matrix[i].slice()
      let b = matrix[matrix.length - 1 - i].slice()
      matrix[i] = b;
      matrix[matrix.length - 1 - i] = a;
    }
    return matrix;
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
  Normalization(Matrix){
    let x = Math.max(Math.abs(...Matrix.flat(Infinity)))
    if(x !== 0){
      for(let i = 0; i < Matrix.length; i++){
        for(let j = 0; j < Matrix[i].length; j++){
          for(let k = 0; k < Matrix[i][j].length; k++){
            Matrix[i][j][k] /= x
          }
        }
      }
    }
  }
}
const fs = require('fs');
const spawn = require('child_process').spawn;

try{
  (async()=>{
    let x = new ConvolutionNeuralNetwork({
      conv_layers :  [["conv","sigmoid",5],["conv","sigmoid",3]],
      feature_maps : [1,1],
      strides :      [{x:1,y:1},{x:1,y:1},{x:1,y:1}]
    });
    let dnn = []
    let image1 = await x.processImage("Train_Images/0's/0.png")
    for(let k = 0; k < 200; k++){
      let rand = (Math.random()*2-1 > 0)
      dnn[k] = x.forwordPass({
        image : [image1[0]],
        layer_nodes : [],
        output_nodes : 1,
        desired_outputs : [0],
        learning_rate : 1e-1
      })
      x.backwordPass()
      let filterCreate = new Promise(function(resolve, reject) {
        callFile("python","filters.py",JSON.stringify(x.Filters.flat()),resolve,reject)
      });
      let imageCreate = new Promise(function(resolve, reject) {
        callFile("python","image.py",JSON.stringify(x.ConvLayers),resolve,reject)
      });
      await filterCreate;
      await imageCreate;
      console.log("iteration ->",k);
      for(let i in x.ConvLayers){ if(i !== "Layer_0") x.Normalization(x.ConvLayers[i]);}
    }
    console.log(x.FCnetwork);
    console.log(x);
    console.log(x.FilterGrad[0][0]);
    let a = []
    dnn.map(e=>{a.push(e.Cost)})
    fs.writeFileSync("cost.txt",JSON.stringify(a))

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
