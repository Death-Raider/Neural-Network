const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const mnist = require('mnist')
const cliProgress = require('cli-progress');

function createMatrix(z,y,x,value){
    let M = []
    for(let i = 0; i < z; i++){
        M[i] = []
        for(let j = 0; j < y; j++){
            M[i][j] = []
            for(let k = 0; k < x; k++){
                M[i][j][k] = value()
            }
        }
    }
    return M
}

let La = new LinearAlgebra;
let conv = new Convolution;
let conv2 = new Convolution;
let mxPool1 = new MaxPool;

let set = mnist.set(8000, 2000)
//Creating filters
let filter_count_1 = 4
let f = []
for(let i = 0; i<filter_count_1; i++) f.push(createMatrix(1,5,5,()=>Math.random()))

let filter_count_2 = 10
let f2 = []
for(let i= 0; i<filter_count_2; i++) f2.push(createMatrix(filter_count_1,5,5,()=>Math.random()))

let BATCH_SIZE = 1
let EPOCH = 1

//initilizing network
let network = new NeuralNetwork({
  input_nodes : 8*8*filter_count_2,
  layer_count : [100],
  output_nodes :10,
  weight_bias_initilization_range : [-1,1]
});
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
//Required parameters
let acc = {t:0,f:0}
let BATCH_Stack = {
    Filters:[],
    FullyConnected:[]
}
let Prev = {
    W:[],
    B:[]
}
Train()
function forward_pass(input,desired){
    let y1 = conv.convolution(input,f,true)
    if(y1.flat(Infinity).filter(e=>e===0).length != y1.flat(Infinity).length)
        y1 = La.normalize(y1,-1,1)
    //Max Pool
    let y2 = mxPool1.pool(y1)
    //conv2
    let y3 = conv2.convolution(y2,f2,true)
    if(y3.flat(Infinity).filter(e=>e===0).length != y3.flat(Infinity).length)
        y3 = La.normalize(y3,0,1)
    //feed to network and get cost
    let out = network.trainIteration({
        input:La.vectorize(y3),
        desired:desired,
    });
    let pred = out.Layers[out.Layers.length-1]

    return [y1,y2,y3,out, pred]
}
function backword_pass(){
    //getting gradients from network and reshape into proper format
    let grads_y3 = network.getInputGradients()
    grads_y3 = La.reconstructMatrix(grads_y3,{x:8*8,y:filter_count_2,z:1}).flat(1)
    grads_y3 = La.transpose(grads_y3)
    //sending grads for conv
    let grads_y2 = conv2.layerGrads(grads_y3)
    grads_y2 = La.vectorize(grads_y2)
    grads_y2 = La.reconstructMatrix(grads_y2,{x:12*12,y:filter_count_1,z:1}).flat(1)
    grads_y2 = La.transpose(grads_y2)
    //sending grads to pool
    let grads_y1 = mxPool1.layerGrads(grads_y2)
    //sendin grads to conv
    let grads_x = conv.layerGrads(grads_y1)
    //no point in sending grads to input layer but still doing it
    grads_x = La.vectorize(grads_x)
    grads_x = La.reconstructMatrix(grads_x,{x:28*28,y:1,z:1}).flat(1)
    grads_x = La.transpose(grads_x)

    return {grads_y3,grads_y2,grads_y1,grads_x}
}
function Train(){
    momentum = 0.9
    for(let epoch = 0; epoch < EPOCH; epoch++){
        //setting up the the progress bar
        let bar1 = new cliProgress.SingleBar({
            format: 'Epoch:{epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc}% | Loss: {loss}'
        }, cliProgress.Presets.shades_classic);
        bar1.start(set.training.length, 0,{
            epoch:epoch,
            acc:acc.t*BATCH_SIZE/10,
            loss:0
        })
        acc = {t:0,f:0} // reset accuracy every epoch
        let cost = 0
        for(let step = 0; step < set.training.length; step++){

            let x = La.normalize(set.training[step].input,-1,1)
            x = La.reconstructMatrix(x,{x:28,y:28,z:1})
            let desired = set.training[step].output

            let [y1,y2,y3,out, pred] = forward_pass(x,desired)
            cost = out.Cost.toFixed(3)

            if(step%1000 == 0){
                // every 1000 steps reset accuracy and fill logs
                acc = {t:0,f:0}
            }

            if(step%BATCH_SIZE == 0){
                if(step != 0){// every bacth if step is not 0
                    //getting initial changes
                    let WeightUpdate =  BATCH_Stack.FullyConnected[0].WeightUpdate
                    let BiasUpdate =  BATCH_Stack.FullyConnected[0].BiasUpdate
                    let F1 = BATCH_Stack.Filters[0].grads_y1
                    let F2 = BATCH_Stack.Filters[0].grads_y3
                    for(let i = 1; i < BATCH_SIZE; i++){// adding the other weights of the batch together
                        WeightUpdate = La.basefunc(WeightUpdate,BATCH_Stack.FullyConnected[i].WeightUpdate,(x,y)=>x+y)
                        BiasUpdate = La.basefunc(BiasUpdate,BATCH_Stack.FullyConnected[i].BiasUpdate,(x,y)=>x+y)
                        F1 = La.basefunc(F1,BATCH_Stack.Filters[i].grads_y1,(x,y)=>x+y)
                        F2 = La.basefunc(F2,BATCH_Stack.Filters[i].grads_y3,(x,y)=>x+y)
                    }
                    //Applying momentum
                    WeightUpdate = La.basefunc(BATCH_Stack.FullyConnected[BATCH_SIZE-1].WeightUpdate,WeightUpdate,(a,b)=>(a*momentum + b*(1-momentum)))
                    BiasUpdate = La.basefunc(BATCH_Stack.FullyConnected[BATCH_SIZE-1].BiasUpdate,BiasUpdate,(a,b)=>(a*momentum + b*(1-momentum)))
                    F1 = La.basefunc(BATCH_Stack.Filters[BATCH_SIZE-1].grads_y1,F1,(a,b)=>(a*momentum + b*(1-momentum)))
                    F2 = La.basefunc(BATCH_Stack.Filters[BATCH_SIZE-1].grads_y3,F2,(a,b)=>(a*momentum + b*(1-momentum)))
                    //updating
                    network.update(WeightUpdate,BiasUpdate,0.1);
                    conv2.filterGrads(F2,1e-4)
                    conv2.F = La.normalize(conv2.F,-1,1)
                    conv.filterGrads(F1,1e-3)
                    conv.F = La.normalize(conv.F,-1,1)
                    //update Accuracy
                    if(pred.indexOf(Math.max(...pred))==desired.indexOf(Math.max(...desired))){
                        acc.t += 1
                    }else{
                        acc.f += 1
                    }
                }
                //Reinitilizing Batch
                BATCH_Stack.Filters = []
                BATCH_Stack.FullyConnected = []
                //move pregress bar by batch size and show the following
                bar1.increment(BATCH_SIZE,{epoch:epoch,loss:out.Cost.toFixed(3),acc:acc.t*BATCH_SIZE/10})
            }
            //update batch
            BATCH_Stack.Filters.push(backword_pass())
            BATCH_Stack.FullyConnected.push({WeightUpdate:network.WeightUpdates,BiasUpdate:network.BiasUpdates})
        }
        //stop bar at end of epoch
        bar1.stop()
        // print out details
        console.log(
            "cost:",cost,
            "epoch",epoch,
            "acc:",acc.t*BATCH_SIZE/10,"%",acc.f*BATCH_SIZE/10,"%"
        );
        //Save network
        // network.save("Net1")
        // conv.saveFilters("Conv")
        // conv2.saveFilters("Conv2")
        // mxPool1.savePool("Pool")
    }
}
