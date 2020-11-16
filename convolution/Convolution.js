function Convolution({matrix,filter,bias = 0,step = {x:1,y:1},padding = 0,type="conv"} = {}){
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
  //special case for max-pool
  if(type==="max-pool"){
    filter = [[0,0],[0,0]];
    bias = 0;
    step = {x:2,y:2}
  }
  if(matrix === undefined || filter === undefined) throw "Err: Input matrix or filter not specified"
  //adds padding to matrix
  matrix = padMatrix(matrix,padding,0)
  let mask = []; //use to create a mask when max-pooling for gradient transfer
  let output = []; //convoluted output
  let outputSize = {
    y : 1+(matrix.length - filter.length)/step.y,
    x : 1+(matrix[0].length - filter[0].length)/step.x
  }
  //checking if convolution is possible
  if(outputSize.y-Math.floor(outputSize.y) !== 0 || outputSize.x-Math.floor(outputSize.x) !== 0 ) throw "Err: size not compatible";
  //types of convolutions
  if(type==="full"){
    //sliding one matrix over another from right to left, bottom to top
    for(let y = outputSize.y-1; y > -1 ; y--){
      output[y] = [];
      for(let x = outputSize.x-1; x > -1 ; x--) output[y][x] = Activation[0](filterDot(x,y,matrix,filter) + bias);
    }
  }else if(type==="conv"){
    //sliding one matrix over another from left to right, top to bottom
    for(let y = 0; y < outputSize.y; y++){
      output[y] = [];
      for(let x = 0; x < outputSize.x; x++) output[y][x] = Activation[0](filterDot(x,y,matrix,filter) + bias);
    }
  }
  else if(type==="max-pool"){
    //max pooling and creating mask
    for(let y = 0; y < outputSize.y; y++){
      output[y] = [];
      mask[y] = []
      for(let x = 0; x < outputSize.x; x++){
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
    }
  }
  else throw "Err: unidentified type"
  return (type==="max-pool")?[output,mask.flat()]:output
}
//func for backprop, use coming later
function Rotate(matrix){ // rotates matrix by 180
  for(i = 0; i < Math.floor(matrix.length/2); i++){
    let a = matrix[i].slice()
    let b = matrix[matrix.length - 1 - i].slice()
    matrix[i] = b;
    matrix[matrix.length - 1 - i] = a;
  }
  return matrix
}

let kernal = [
  [0,1,0],
  [1,-5,1],
  [0,1,0]
];
let stride = {x:1,y:1};
//input image (pic of 2)
let image = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 1, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 1, 1, 1, 1, 0, 0, 0],
  [0, 1, 0, 1, 1, 0, 0, 0],
  [0, 1, 1, 1, 1, 1, 0, 1],
  [0, 0, 0, 0, 0, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0]
]
//activation for Convolution
let Activation = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)] //sigmoid
//feature maps
let convo1 = Convolution({
  matrix : image,
  filter : kernal,
  step : stride,
  padding : 0,
  type : "conv"
})
let convo2 = Convolution({
  matrix : convo1,
  filter : kernal,
  step : stride,
  padding : 0,
  type : "conv"
})
save("image",image);
save("convo1",convo1);
save("convo2",convo2);

function save(y,x){
  const fs = require('fs')
  fs.writeFileSync("images/"+y+".txt",JSON.stringify(x))
  console.log("saved as " + y + ".txt in images");
}
