// TODO: make more transparent
class NeuralNetwork {
  constructor({input_nodes,layer_count,output_nodes} = {}){
    if(input_nodes === undefined || layer_count === undefined || output_nodes === undefined) throw "Error: structural values not given"
    let parameters=createParameters(input_nodes,layer_count,output_nodes,-0.001,0.001);
    const copyRadar3D = (y,z = []) =>{for (let _a in y) for (let _b in y[_a]) {if(!z[_a]) z[_a] = []; z[_a][_b] = y[_a][_b].slice();};return z}
    const copyRadar2D = (y,z = []) =>{for (let _a in y){if(!z[_a]) z[_a] = []; z[_a] = y[_a].slice();};return z}
    this.HiddenLayerCount=layer_count;
    this.Weights = copyRadar3D(parameters[0])
    this.WeightUpdates = copyRadar3D(parameters[0])
    this.Bias = copyRadar2D(parameters[1])
    this.BiasUpdates = copyRadar2D(parameters[1])
    this.Activation = { //user can change this
      hidden:[(x)=>(x>0)?x:x*0.1,(x)=>(x>0)?1:0.1],
      output:[(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
    }
    function createParameters(input,LayerCount,output,a,b){//a->min witght, b->max weight
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
  changes(Desired,Output,DerivativeActivation){
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
        let [input,desir] = (i <= parseInt(TotalTrain/batch_train))?trainFunc():validationFunc();
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
  use(input){return this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]])[2]}
  save(folder){
    const fs = require('fs')
    if(!fs.existsSync(folder)) fs.mkdirSync(folder);
    writeData(this.Weights,folder,"Weights",'W')
    writeData(this.Bias,folder,"Bias",'B')
    function writeData(arr,folder,slice,initilizer){
      if(!fs.existsSync(`${folder}/${slice}`)) fs.mkdirSync(`${folder}/${slice}`);
      for(let i = 0; i < arr.length; i++){
        let stream = fs.createWriteStream(`${folder}/${slice}/${i}.txt`);
        //bias
        if(initilizer === 'B'){
          stream.write(JSON.stringify(arr[i]))
          stream.write("\n")
        }
        //weights
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
  async load(folder){
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
    let model = new Promise(function(resolve, reject){getLines('model',resolve)});
    model = await model;
    this.Bias = model[0].flat();
    this.Weights = model[1];
  }
}
