//BREAD AND BUTTER OF EVERYTHING
function Convolution({matrix,filter,bias = 0,step = {x:1,y:1},padding = 0,type="conv",activation = "linear"} = {}){
  let activationBank = {
    relu:[(x)=>(x>0)?x:x*0.1,(x)=>(x>0)?1:0.1],
    sigmoid:[(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)],
    linear:[(x)=>x,(x)=>1]
  };
  let Activation = activationBank[activation] //linear activation and derivative
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
  console.log("size ->",outputSize);
  //checking if convolution is possible
  if(outputSize.y-Math.floor(outputSize.y) !== 0 || outputSize.x-Math.floor(outputSize.x) !== 0 ) throw "Err: size not compatible with " + type;
  let output = new Array(outputSize.y).fill(0).map(e=>new Array(outputSize.x).fill(0))//convoluted output
  //types of convolutions
  for(let y = 0; y < outputSize.y; y++){
    mask[y] = []
    for(let x = 0; x < outputSize.x; x++){
      if(type==="conv") output[y][x] = Activation[0](filterDot(x,y,matrix,filter) + bias);
      else if(type==="full_conv")output[y][x] = Activation[0](filterDot(outputSize.x-(x+1),outputSize.y-(y+1),matrix,filter) + bias);
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
function createMatrix(x,y,z,M = []){

  for(let i = 0; i < z; i++){
    M[i] = []
    for(let j = 0; j < y; j++){
      M[i][j] = []
      for(let k = 0; k < x; k++){
        M[i][j][k] = Math.random()*1-0.5
      }
    }
  }
  return M;
}
function convolutionLayers({matrix,kernal,featureMaps,stride,padding=0,type,activation} = {}){
  let conv1 = []
  for(let j = 0; j < matrix.length; j++){
    conv1[j] = []
    for(let i = 0; i < featureMaps; i++){
      conv1[j][i] = Convolution({
        matrix : matrix[j],
        filter : kernal[i],
        step : stride,
        padding : padding,
        type : type,
        activation : activation
      })
    }
  }
  return conv1.flat()
}
function flattenFeatureMaps(featureMapMatrix){
  let connected = [];
  for(featurePlane of featureMapMatrix) connected.push(featurePlane.flat())
  return connected.flat()
}
function backpass(){

}
//image processing
async function processImage(path){
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
function Rotate(matrix){ // rotates matrix by 180
  for(i = 0; i < Math.floor(matrix.length/2); i++){
    let a = matrix[i].slice()
    let b = matrix[matrix.length - 1 - i].slice()
    matrix[i] = b;
    matrix[matrix.length - 1 - i] = a;
  }
  return matrix;
}
function save(y,x){
  const fs = require('fs')
  for(let i in x)
  fs.writeFileSync(`images/${y}(${i}).txt`,JSON.stringify(x[i]))
}

(async ()=>{
  let image = createMatrix(28,28,1) // simulates a 10x10 RGB image

  let input = {
    conv_layers :  [["conv","sigmoid",3],["conv","sigmoid",3],["conv","sigmoid",3],["conv","relu",2]],
    feature_maps : [       1         ,     1        ,        1        ,      1    ],
    strides :      [    {x:1,y:1}     ,    {x:1,y:1}    ,      {x:1,y:1}    ,   {x:2,y:2} ]
  }

  // let imageRGB = await processImage("C:/Users/Darsh/Desktop/oreo1.jpg")
  let ConvLayers = {Layer_0 : image}
  let Filters = [];

  for(let i = 0; i < input["conv_layers"].length; i++){
    let layer = input.conv_layers[i]
    Filters[i] = createMatrix(layer[layer.length-1],layer[layer.length-1],input["feature_maps"][i]);
    console.log(Filters[i].length,layer);
    ConvLayers[`Layer_${i+1}`] = convolutionLayers({
      matrix : ConvLayers[`Layer_${i}`],
      kernal : filterRGB,
      featureMaps : input["feature_maps"][i],
      stride : input["strides"][i],
      padding : 0,
      type : layer[0],
      activation : (layer[0] !== "max_pool")?layer[1]:"relu"
    })
  }

  console.log(flattenFeatureMaps((input.conv_layers[input.conv_layers.length - 1][0] === "max_pool")?ConvLayers.Layer_4[0]:ConvLayers.Layer_4));

  train()

})()
