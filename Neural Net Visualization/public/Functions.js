function clear(m,c){
  m.beginPath();
  m.fillStyle = "black";
  m.fillRect(0,0,c.width,c.height);
}
function addNode(){
  let n = nodeCount++;
  // makes a circle at center
  ctx.beginPath();
  ctx.arc(radius + buffer + spaceLayer*layer, radius + buffer + 30*n, radius, 0, 2*Math.PI); // .arc(x,y,r,start angle,end angle)
  ctx.strokeStyle = "white";
  ctx.stroke();
}
function addLayer(){
  layer++;
  nodeCount = 0;
}
function Rgb(r,g,b){
  r = r.toString(16);
  g = g.toString(16);
  b = b.toString(16);
  if (r.length == 1)
    r = "0" + r;
  if (g.length == 1)
    g = "0" + g;
  if (b.length == 1)
    b = "0" + b;

  return "#" + r + g + b;
}
function addWeights(){
  inputCountHTML = parseFloat(document.getElementById("inputCount").value);
  outputCountHTML = parseFloat(document.getElementById("outputCount").value);
  hiddenNodeCountHTML = parseFloat(document.getElementById("hiddenNodeCount").value);
  hiddenLayerCountHTML = parseFloat(document.getElementById("hiddenLayerCount").value);
  spaceLayer = parseFloat(document.getElementById("layerspace").value);

  ctx.beginPath();
  for(let k = 0; k < hiddenNodeCountHTML; k++){
    for(let j = 1; j < layer - 1; j++){
      for(let p = 0; p <hiddenNodeCountHTML; p++){
        ctx.moveTo(buffer+spaceLayer*j + 2*radius,buffer+(1+(3*k))*radius);
        ctx.lineTo(buffer+spaceLayer*(j+1),buffer+(1+(3*p))*radius);
      }
    }
  }
  for(let k = 0; k < inputCountHTML; k++){
    for(let j = 0; j < 1; j++){
      for(let p = 0; p <hiddenNodeCountHTML; p++){
        ctx.moveTo(buffer+spaceLayer*j + 2*radius,buffer+(1+(3*k))*radius);
        ctx.lineTo(buffer+spaceLayer*(j+1),buffer+(1+(3*p))*radius);
      }
    }
  }
  for(let k = 0; k < hiddenNodeCountHTML; k++){
    for(let j = layer-1; j < layer ; j++){
      for(let p = 0; p < outputCountHTML; p++){
        ctx.moveTo(buffer+spaceLayer*j + 2*radius,buffer+(1+(3*k))*radius);
        ctx.lineTo(buffer+spaceLayer*(j+1),buffer+(1+(3*p))*radius);
      }
    }
  }
  ctx.stroke();
}
function individualWeightAccessAndOverlap(w,k,p,j){ //individualWeightAccess(color,"k"th node,"p"th node,"j"th layer)
  ctx.beginPath();
  ctx.moveTo(buffer+spaceLayer*j + 2*radius,buffer+(1+(3*k))*radius);
  ctx.lineTo(buffer+spaceLayer*(j+1),buffer+(1+(3*p))*radius);
  ctx.strokeStyle = w;
  ctx.stroke();
}
function addColor(value){//a-> opacity
  let color;
  if(value < 0){
    let g = parseInt((value+1)*255)
    color = Rgb(g,g,255);
  }
  if(value == 0){
    color = Rgb(255,255,255)
  }
  if(value > 0){
    let g = parseInt(value*255)
    color = Rgb(255,255-g,255-g);
  }
  //handles opacity (a)
  a = Math.round(Math.abs(2*value/3) * 100) / 100;
  let alpha = Math.round(a * 255);
  let hexAlpha = (alpha + 0x10000).toString(16).substr(-2).toUpperCase();
  color = color + hexAlpha;
  console.log(value,color,hexAlpha);
  return color;
}
