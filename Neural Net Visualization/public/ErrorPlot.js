const c_error = document.getElementById("errorGraph");
const ctx_error = c_error.getContext("2d");
let spacing = 20;
let tickmarkCount = c_error.width/spacing

function plotaxis(m,n,Loss){
  epochsHTML = parseInt(document.getElementById("training").value);
  m.beginPath();
  //X-axis
  m.moveTo(0,n.height-20)
  m.lineTo(n.width,n.height-20)
  m.strokeStyle = "white";
  //Y-axis
  m.moveTo(20,0)
  m.lineTo(20,n.height)
  m.strokeStyle = "white";
  m.stroke();
  //tickmarks

  if(tickmarkCount*epochsHTML/Batch > 300)spacing = 30
  tickmarkCount = n.width/spacing
  //X-AXIS
  for(let i = 0; i < tickmarkCount; i++){
    m.beginPath();
    m.font = "8px Arial";
    m.fillStyle = "white";
    // +x
    m.fillRect(spacing*(i)+20,n.height-20,1,2); //.fillRect(x,y,breadth,length)
    m.fillText((epochsHTML*i/(tickmarkCount*Batch)).toFixed(1),spacing*(i)+20,n.height-20+10);
    // -x
    m.fillRect(-spacing*(i)+20,n.height-20,1,2);
    m.fillText((-epochsHTML*i/(tickmarkCount*Batch)).toFixed(1),-spacing*(i)+20,n.height-20+10);
    m.stroke();
  }
  //Y-AXIS
  for(let i = 0; i < n.height/20; i++) {
    m.beginPath();
    m.font = "8px Arial";
    m.fillStyle = "white";
    // +y
    m.fillRect(20,n.height-20-20*(i),-2,1);
    m.fillText(i/4,20,n.height-20-20*(i));
    // -y
    m.fillRect(20,n.height-20+20*(i),-2,1);
    m.fillText(-i/4,20,n.height-20+20*(i));
    m.stroke();
  }
  //PLOT LOSS
  for(let i = 0; i < tickmarkCount-1; i++){
    let position1 = Math.floor(epochsHTML*i/(tickmarkCount*Batch))
    let position2 = Math.floor(epochsHTML*(i+1)/(tickmarkCount*Batch))
    m.beginPath();
    m.moveTo(spacing*(i)+20,n.height-20-20*(Loss[position1]*4))
    m.lineTo(spacing*(i+1)+20,n.height-20-20*(Loss[position2]*4))
    m.strokeStyle = "White";
    m.stroke();
  }
}
