class LinearAlgebra {
  constructor(){}
  basefunc(a,b,opt){ return a instanceof Array ? a.map((c, i) => this.basefunc(a[i], Array.isArray(b)?b[i]:b, opt)) : opt(a,b)}; // base function for any depth code
  transpose(m){ return m[0].map((e,i) => m.map(row => row[i])); } //only depth 2
  scalarMatrixProduct(s,m){return this.basefunc(m,s,(arr,scalar)=>arr*scalar)}; //max any depth
  scalarVectorProduct(s,v){return v1.map(e=>e*s)};//only depth 1
  vectorDotProduct(v1,v2){return v1.map((e,i,a)=>e*v2[i]).reduce((a,b)=>a+b)}; //only both depth 1
  MatrixvectorProduct(m,v){return v.map((e,i)=>this.scalarMatrixProduct(e,this.transpose(m)[i])).reduce((a,b)=>a.map( (x, i)=> x + b[i] ))}; //only depth 2 and 1
  matrixProduct(m1,m2){return m1.map(row => m2[0].map((_,i)=>this.vectorDotProduct( row, m2.map(e=>e[i]) )) )}
  kroneckerProduct(a,b,r=[],t=[]) {return a.map(a=>b.map(b=>a.map(y=>b.map(x=>r.push(y*x)),t.push(r=[]))))&&t}
  flip(matrix){
    let reversed=(a)=>a.slice(0).reverse()
    return reversed(matrix).map(reversed)
  }
  minor(m,i=0,j=0,s=m.length-1){return Array(s).fill(0).map((e,p)=>{
      let l = m[p+(p>=i?1:0)].slice()
      l.splice(j,1)
      return l
    })}
  determinant(m,s=m.length){ // matrix (nxn) and its order n > 1
    if(s == 2){
      return m[0][0]*m[1][1] - m[0][1]*m[1][0] //determinant of 2x2 matrix
    }else{
      let sum = 0
      for(let i = 0; i < s; i++)
        sum += (-1)**(i)*m[0][i]*this.determinant(this.minor(m,0,i),s-1)
      return sum
    }
  };
  invertMatrix(m,s=m.length){ // any nxn matrix
    let cofactorMatrix = Array(s).fill(0).map(e=>Array(s));
    let det = 0;
    for(let i = 0; i < s; i++){
      for(let j = 0; j < s; j++)
        cofactorMatrix[j][i] = (-1)**(i+j)*this.determinant(this.minor(m,i,j),s-1); // transpose + values
      det += m[i][0]*cofactorMatrix[0][i];
    }
    if(!det){
      console.log("matrix not invertiable det =",det);
      return false
    }
    let invert =  this.scalarMatrixProduct(1/det,cofactorMatrix)
    return invert
  }
  weightedSum(k=1,...M){return M.reduce((a,b)=>this.basefunc(a,b,(x,y)=>x+k*y))} ;// same but any depth
  normalize(m,a=-1,b=1){
      let min_max = {min:Math.min(...m.flat(Infinity)),max:Math.max(...m.flat(Infinity))}
      if(min_max.max === min_max.min){
          if(min_max.max == 0) return m
          return this.basefunc(m,min_max.max,(x,y)=>x/y)
      }
      return this.basefunc(m,min_max,(x,y)=>(b-a)*(x-y.min)/(y.max-y.min)+a)
  } ;// any depth of matrix
  vectorize(m){return Array.isArray(m[0][0])?m.flatMap(e=>this.vectorize(e)):this.transpose(m).flat(2)}; // any depth
  im2row(m,a,q=1,t=[],i=0,j=0,k=0,r){
    if(Array.isArray(m[0][0])){
      m.map((e,f)=>this.im2row(e,a,q,t[f]=[],i,j,f))
      for(let index = 1; index < t.length; index++) t[0] = t[0].map((e,i)=>e.concat(t[index][i]))
      return t[0]
    }else{
      t.push(r=[])
      for(let x = 0; x < a[1]; x++){
        for(let y = 0; y < a[0]; y++)
          r.push(m[y+i][x+j])
      }
      return ( i < m.length-a[0] ? this.im2row(m,a,q,t,i+=q,j,k) : j < m[0].length-a[1] ? this.im2row(m,a,q,t,0,j+=q,k) : t )
    }
  };
  im2col(m,a){return this.transpose(this.im2row(m,a))};
  reconstructMatrix(flatArr,m,Matrix=[]){
    for(let z = 0; z < m.z; z++){
      Matrix[z] = []
      for(let i = 0; i < m.y; i++){
        Matrix[z][i] = []
        for(let j = 0; j < m.x; j++){
          Matrix[z][i][j] = flatArr[j + m.x*i + m.y*m.x*z]
        }
      }
    }
    return Matrix
  }
}
class NeuralNetwork extends LinearAlgebra{
  constructor({input_nodes,layer_count,output_nodes,weight_bias_initilization_range=[-0.001,0.001] } = {}){
    super()
    if(input_nodes === undefined || layer_count === undefined || output_nodes === undefined) throw "Error: structural values not given"
    let parameters=createParameters(input_nodes,layer_count,output_nodes,weight_bias_initilization_range[0],weight_bias_initilization_range[1]);
    this.copyRadar3D = (y,z = []) =>{for (let _a in y) for (let _b in y[_a]) {if(!z[_a]) z[_a] = []; z[_a][_b] = y[_a][_b].slice();};return z}
    this.copyRadar2D = (y,z = []) =>{for (let _a in y){if(!z[_a]) z[_a] = []; z[_a] = y[_a].slice();};return z}
    this.HiddenLayerCount=layer_count;
    this.Weights = this.copyRadar3D(parameters[0])
    this.WeightUpdates = this.copyRadar3D(parameters[0])
    this.Bias = this.copyRadar2D(parameters[1])
    this.BiasUpdates = this.copyRadar2D(parameters[1])
    this.previousGrads = {
        Weights: this.copyRadar3D(parameters[0]),
        Bias: this.copyRadar2D(parameters[1])
    }
    this.Activation = {
      hidden:[(x)=>(x>0)?x:x*0.1,(x)=>(x>0)?1:0.1],
      output:[(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
    }
    this.loss_func = {
        out: (X,Y,cost = 0)=>{
            for(let m = 0; m < Y.length; m++) cost+=0.5*Math.pow(X[m]-Y[m],2);
            return cost
        },
        derivative: (X,Y,i)=>{
            return (X-Y)*this.Activation.output[1](X) // dY = 0
        }
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
    let cost = this.loss_func.out(Output,Desired)

    for(let i = 0; i < this.Nodes[this.HiddenLayerCount.length + 1].length; i++){
      this.BiasUpdates[this.Weights.length-1][i] = this.loss_func.derivative(this.Nodes[this.HiddenLayerCount.length+1][i] , Desired[i],i) // output node of model and desired nodes
      for(let j = 0; j < this.Nodes[this.HiddenLayerCount.length].length; j++) this.WeightUpdates[this.Weights.length-1][i][j] = (this.BiasUpdates[this.Weights.length-1][i]*this.Nodes[this.HiddenLayerCount.length][j]);
    }
    for(let j = this.Weights.length - 2; j > -1; j--){//iterates of all layers except the last one
      for(let k = 0,sum = 0; k < this.Weights[j].length; k++,sum = 0){
        for(let m = 0; m < this.Weights[j+1].length; m++) sum += this.Weights[j+1][m][k]*this.WeightUpdates[j+1][m][k];
        this.BiasUpdates[j][k]= (sum*(DerivativeActivation[0](this.Nodes[j+1][k])))/((this.Nodes[j+1][k]==0)?1:this.Nodes[j+1][k]);
        for(let p = 0; p < this.Weights[j][k].length; p++) this.WeightUpdates[j][k][p] = this.BiasUpdates[j][k] * this.Nodes[j][p];
      }
    }
    return {updatedWeights:this.WeightUpdates,updatedBias:this.BiasUpdates,Cost:cost};
  }
  getInputGradients(grad = []){
    for(let k = 0,sum = 0; k < this.Nodes[0].length; k++,sum = 0){
      for(let m = 0; m < this.Weights[0].length; m++)
        sum += this.Weights[0][m][k]*this.WeightUpdates[0][m][k]
      grad[k] = sum/((this.Nodes[0][k]===0)?1:this.Nodes[0][k]);
    }
    return grad
  }
  update(secondTensor,secondMatrixBias,rate){//Readjustment of weights and bias
    for(let i = 0; i < secondTensor.length; i++){
      for(let j = 0; j < secondTensor[i].length; j++){
        for(let k = 0; k < secondTensor[i][j].length; k++) this.Weights[i][j][k] -= rate*secondTensor[i][j][k];
        this.Bias[i][j]-=rate*secondMatrixBias[i][j];
      }
    }
  }
  train({TotalTrain=0,trainFunc=()=>{},TotalVal=0,validationFunc=()=>{},learning_rate=0.0005,batch_train=1,batch_val=1,momentum=0}={}){
    let cost=[], cost_val=[], changing = [];
    let Parameters = {W:[],B:[]}
    for(let i = 0; i < Math.floor((TotalTrain/batch_train+TotalVal/batch_val)); i++){
      let batch = (i < Math.floor(TotalTrain/batch_train))?batch_train:batch_val;
      let sumCost = 0;
      for(let b = 0; b < batch ; b++){
        let [input,desir] = (i <= Math.floor(TotalTrain/batch_train))?trainFunc(b,i):validationFunc(b,i);
        this.Nodes=this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]]);
        changing[b] = this.changes(desir,this.Nodes[this.Nodes.length-1],[this.Activation.hidden[1],this.Activation.output[1]]);
        sumCost += changing[b].Cost/batch
      }
      if(i < Math.floor(TotalTrain/batch_train)){
        cost.push(sumCost);
        Parameters.W.push(this.copyRadar3D(this.Weights))
        Parameters.B.push(this.copyRadar2D(this.Bias))
        for(let x = 0; x < changing[0].updatedWeights.length; x++){
          for(let y = 0,sumBias = 0; y < changing[0].updatedWeights[x].length; y++,sumBias = 0){
            for(let z = 0,sumWeight = 0; z < changing[0].updatedWeights[x][y].length; z++,sumWeight = 0){
              for(let b = 0; b < batch; b++){
                  sumWeight += changing[b].updatedWeights[x][y][z]
              }
              sumWeight = this.previousGrads.Weights[x][y][z]*momentum +  (1-momentum)*sumWeight
              this.WeightUpdates[x][y][z] = sumWeight;
            }
            for(let b = 0; b < batch; b++){
                sumBias += changing[b].updatedBias[x][y]
            }
            sumBias = this.previousGrads.Bias[x][y]*momentum + (1-momentum)*sumBias
            this.BiasUpdates[x][y] = sumBias;
          }
        }
        this.update(this.WeightUpdates,this.BiasUpdates,learning_rate);
        this.previousGrads.Weights = this.copyRadar3D(this.WeightUpdates)
        this.previousGrads.Bias = this.copyRadar2D(this.BiasUpdates)
      }else{cost_val.push(sumCost)}
    }
    this.Loss = {Train_Loss:cost,Validation_Loss:cost_val, params: Parameters};

  }
  trainIteration({input,desired}={}){
    this.Nodes = this.GetLayerValues(input,[this.Activation.hidden[0],this.Activation.output[0]]);
    let changing = this.changes(desired,this.Nodes[this.Nodes.length-1],[this.Activation.hidden[1],this.Activation.output[1]]);
    return {Cost:changing.Cost,Layers:this.Nodes, Updates: changing}
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

    const readline  = require('readline');
    const fs = require('fs');
    let sub_dir = fs.readdirSync(path)
    let readInterface, lines=[];

    for(let s in sub_dir){
        let files = fs.readdirSync(`${path}/${sub_dir[s]}`)
        lines[s] = []
        for(let f in files){
          lines[s][f] = [];
          let readLinePromise = new Promise((res,rej)=>{
              let promiseArr = []
              readInterface = readline.createInterface({
                  input: fs.createReadStream(`${path}/${sub_dir[s]}/${files[f]}`)
              });
              readInterface.on('line', function(line){
                  promiseArr.push(JSON.parse(line))
              });
              readInterface.on('close', ()=>{res(promiseArr)});
          })
          lines[s][f].push(await readLinePromise)
        }
    }
    let model = lines
    // model = await model;
    this.Bias = model[0].flat(2);
    this.Weights = model[1].flat(1);
  }
}

class Convolution extends LinearAlgebra{
  constructor(){
    super();
    this.x_shape = []
    this.f_shape = []
    this.y_shape = []
    this.F = []
    this.phi = []
  }
  convolution(x,f=this.F,reshape=true,activation=(x)=>(x>0)?x:0){
    if(this.x_shape.length === this.f_shape.length){// only true when both are length 0 aka when first defined
      this.x_shape = [x[0].length, x[0][0].length,x.length]
      this.f_shape = [f[0][0].length,f[0][0][0].length,f[0].length,f.length]
    }
    this.y_shape = [this.x_shape[0]-this.f_shape[0]+1, this.x_shape[1]-this.f_shape[1]+1 , this.f_shape[3]]
    this.phi = super.im2row(x,[this.f_shape[0], this.f_shape[1]])
    if(this.F.length === 0){
      for(let d = 0; d < this.f_shape[3]; d++) this.F.push(super.vectorize(f[d]))
    }
    let F = super.transpose(this.F)
    let y = super.matrixProduct(this.phi,F)
    if(reshape){
      y = super.vectorize(y).map(e=>activation(e))
      return super.reconstructMatrix(y, {x:this.y_shape[0],y:this.y_shape[1],z:this.y_shape[2]}).map(e=>super.transpose(e))
    }
    return y
  }
  filterGrads(prevGrads,lr = 0.01){
    let grads_f = super.matrixProduct(super.transpose(prevGrads), this.phi)
    this.F = super.weightedSum(-lr,this.F,grads_f)
    return grads_f
  }
  layerGrads(prevGrads){
    //m -> array to store layer gradients of shape (H_l W_l x D_l)
    let m = Array(this.x_shape[2]).fill(0).map( e=>Array(this.x_shape[0]).fill(0).map( q=>Array(this.x_shape[1]) ) )
    let mu = super.matrixProduct(prevGrads, this.F)
    const m_inv = (i_l, j_l, d_l) =>{ //mapping from x (i_l,j_l,d_l) to phi (p,q)
      let set = []
      for(let w = 0; w < this.f_shape[0]*this.f_shape[1]*this.f_shape[2]; w++){
        let i = w%this.f_shape[0]
        let j = Math.floor(w/this.f_shape[0])%this.f_shape[1]
        let y_i = i_l - i
        let y_j = j_l - j
        let p = y_i + this.y_shape[0]*y_j
        let q = i + j*this.f_shape[0] + this.f_shape[0]*this.f_shape[1]*d_l
        if(y_i > -1 && y_i < this.y_shape[0] && y_j < this.y_shape[1] && y_j > -1 && p < this.y_shape[0]*this.y_shape[1] && q < this.f_shape[0]*this.f_shape[1]*m.length)
          if(set.filter(e=>e[0]==p&&e[1]==q).length == 0) set.push([p,q]) //checking for duplicates
      }
      return set
    }
    for(let o = 0; o < m.length*m[0].length*m[0][0].length; o++){ // goes through all indices of m
      let i_l = o%m[0].length
      let j_l = Math.floor(o/m[0].length)%m[0][0].length
      let d_l = Math.floor(o/(m[0].length*m[0][0].length))
      let indices = m_inv(i_l,j_l,d_l)
      m[d_l][i_l][j_l] = 0
      for(let k = 0; k < indices.length; k++) m[d_l][i_l][j_l] += mu[indices[k][0]][indices[k][1]]
    }
    return m
  }
  saveFilters(folder){
      const fs = require('fs')
      if(!fs.existsSync(folder)) fs.mkdirSync(folder);
      let stream = fs.createWriteStream(`${folder}/Filter.txt`);
      let data = {
          X_shape:this.x_shape,
          Y_shape:this.y_shape,
          F_shape:this.f_shape,
          Filter:this.F,
      }
      stream.write(JSON.stringify(data))
      stream.end()
  }
}
class MaxPool extends LinearAlgebra{
  constructor(){
    super();
    this.shape = []
    this.location = []
  }
  pool(inp,size = 2,stride=2,reshape=true){
    let M = inp.map((m,index)=>{
      if(m.length%stride != 0 || m[0].length%stride != 0 || size > m.length || size > m[0].length){
        console.log("pooling cant be done with set stride");
        return false;
      };
      this.shape = [inp[0].length, inp[0][0].length, inp.length];
      let H = m.length/stride;
      let W = m[0].length/stride;
      let i=0,j=0;
      this.location[index] = [];
      let phi = super.im2row(m,[size,size],stride).map((e,l)=>{
        let max = Math.max(...e);
        let q = e.indexOf(max);
        let fi = q%size+i , fj = Math.floor(q/size)+j;
        i=(i<(m.length-size))? i+stride:0;
        (j < (m[0].length-size) && i==0)?j+=stride:0;
        this.location[index].push([fi,fj]);
        return max;
      });
      return super.reconstructMatrix(phi, {x:W,y:H,z:1}).map(e=>super.transpose(e)).flat();
    });
    return reshape?M:super.vectorize(M);
  }
  layerGrads(prevGrads){
    let m = Array(this.shape[1]*this.shape[0]).fill(0).map(e=>Array(this.shape[2]).fill(0).map(e=>0))
    this.location.forEach((e,i)=>{
      prevGrads.forEach((p,j)=>{
        let index = e[j][0]+e[j][1]*this.shape[1]
        m[index][i] = p[i]
      })
    })
    return m
  }
  savePool(folder){
      const fs = require('fs')
      if(!fs.existsSync(folder)) fs.mkdirSync(folder);
      let stream = fs.createWriteStream(`${folder}/Pool.txt`);
      let data = {
          Loc:this.location,
          Shape:this.shape
      }
      stream.write(JSON.stringify(data))
      stream.end()

  }
}
module.exports = {NeuralNetwork,LinearAlgebra,Convolution,MaxPool};
