const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const mnist = require('mnist')
const fs = require('fs')
const cliProgress = require('cli-progress');
const LA = new LinearAlgebra

function createMatrix(z,y,x,value){
	let M = []
	for(let i = 0; i < z; i++){
		M[i] = []
		for(let j = 0; j < y; j++){
			M[i][j] = []
			for(let k = 0; k < x; k++){
				M[i][j][k] = value(i,j,k)
			}
		}
	}
	return M
}
function shuffle(array){
    indicies = []
    for(let i = 0; i <= 1000; i++){
        indicies[i] = Math.floor(array.length*Math.random())
        if (Math.floor(Math.random()*2)){
            i1 = Math.floor(indicies.length*Math.random())
            i2 = Math.floor(indicies.length*Math.random())
            t1 = array[indicies[i2]]
            array[indicies[i2]] = array[indicies[i1]]
            array[indicies[i1]] = t1  
        }
    }
    return array
}
const print=(...x)=>console.log(...x)

main()
function main(){
	const input_shape = [28,28,1]
	const {model,filters,filterShape} = createModel(input_shape)
	model_architecture(model,filters,filterShape)
	const learningRates = [1e-4,1e-5,3e-3] // conv1 conv2 network
	const EPOCH = 4
	const BATCH_SIZE = 5
	const Set = mnist.set(500,3)
	const trainingSet = Set.training
	const testSet = Set.test
	
	for(let epoch = 0; epoch < EPOCH; epoch++){
		let acc = {t:0,f:0}, gradientArray = [];
		let bar1 = new cliProgress.SingleBar({
			format: 'Epoch:{epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc}% | Loss: {loss}'
		}, cliProgress.Presets.shades_classic);
		bar1.start(trainingSet.length, 0,{
			epoch:epoch,
			acc:0,
			loss:0
		})
		for(let step = 0; step < trainingSet.length; step++){
			let input = trainingSet[step].input
			input = LA.normalize(input,-1,1)
			input = LA.reconstructMatrix(input,{x:28,y:28,z:1})
			let desired = trainingSet[step].output
			let output=forwardPass(model,input)
			gradientArray.push(backwordPass(model,desired,filters))
			if(step%BATCH_SIZE == 0){
				gradientArray = gradientArray.reduce((a,b)=>LA.basefunc(a,b,(i,j)=>(0.95)*i+(0.05)*j))
				updateModel(model,gradientArray,learningRates)
				bar1.increment(BATCH_SIZE,{
					epoch:epoch,
					loss:(model[model.length-1].cost).toFixed(3),
					acc:(100*acc.t/(acc.t+acc.f)).toFixed(3)
				})
				gradientArray = []
				if(output[output.length-1].indexOf(Math.max(...output[output.length-1])) == desired.indexOf(Math.max(...desired))){
					acc.t += 1
				}else{
					acc.f += 1
				}
			}
		}
		bar1.stop()
		print(
			"cost:",(model[model.length-1].cost).toFixed(3),
			"epoch:",epoch,
			"acc:",`${100*acc.t/(acc.t+acc.f)}%`
		);
		print(acc)
		saveModel(model,"Network")
	}
}
function createModel(input_shape){

	const conv1 = new Convolution
	const conv2 = new Convolution
	const maxPool1 = new MaxPool  
	const maxPool2 = new MaxPool  

	let [fx,f1,f2] = [input_shape[2],8,8]; // filter count
	let [fs1,fs2] = [5,3]; // filter sizes
	let fw = 0.1 //max initilization weight for filter

	conv1.x_shape = input_shape // y,x,z
	conv1.f_shape = [fs1,fs1,fx,f1] // y,x,z,f1
	conv1.y_shape = [
		conv1.x_shape[0]-conv1.f_shape[0]+1,
		conv1.x_shape[1]-conv1.f_shape[1]+1,
		f1
	]
	for(let i = 0; i<f1; i++)
		conv1.F.push(LA.vectorize(createMatrix(fx,fs1,fs1,()=>fw*Math.random())))

	maxPool1.shape = conv1.y_shape  // y,x,f1

	conv2.x_shape = [
		(maxPool1.shape[0] - 2)/2 + 1,
		(maxPool1.shape[1] - 2)/2 + 1,
		f1
	]
	conv2.f_shape = [fs2,fs2,f1,f2]
	conv2.y_shape = [
		conv2.x_shape[0]-conv2.f_shape[0]+1,
		conv2.x_shape[1]-conv2.f_shape[1]+1,
		f2
	]
	for(let i = 0; i<f2; i++)
		conv2.F.push(LA.vectorize(createMatrix(f1,fs2,fs2,()=>fw*Math.random())))

	maxPool2.shape = conv2.y_shape
	output_shape = ((maxPool2.shape[0] - 2)/2 + 1)*((maxPool2.shape[1] - 2)/2 + 1)*f2

	const network = new NeuralNetwork({
		input_nodes: output_shape,
		layer_count:[120,60],
		output_nodes:10,
		weight_bias_initilization_range:[-0.1,0.1]
	})

	function softmax(X){
		let max = Math.max(...X)
		let exps = X.map((e,i)=>(Math.exp(e-max)))
		let sum = exps.reduce((a,b)=>a+b)
		return exps.map((e,i)=>e/sum)    
	}
	function softmax_der(X,Y,i){
		let del = X[i]*(1-X[i])
		return del
	}
	function categorial_cross_entropy(X,Y){
		let cost = 0
		for(let i = 0; i < X.length; i++){
			cost -= Y[i]*Math.log(X[i]) 
		}
		return cost
	}
	function der_categorial_cross_entropy(X,Y,i){
		return X[i]-Y[i]
	}
	network.Activation.output = [softmax,softmax_der]
	network.loss_func.out = categorial_cross_entropy
	network.loss_func.derivative = der_categorial_cross_entropy

	const model = [conv1,maxPool1,conv2,maxPool2,network]
	return {
		model:model,
		filters: [fx,f1,f2],
		filterShape: [fs1,fs2]
	}
}
function model_architecture(model,filters,filterShape){
	let model_str = "Layer Name    Input Shape  Output Shape\n"
	for(let layer of model){
		let name = layer.constructor.name;
		switch(name){
			case "Convolution":
				model_str += name + "    " + layer.x_shape + "      " + layer.y_shape + "\n"
				break;
			case "MaxPool":
				model_str += name + "        " + layer.shape + "      " + layer.shape.map((e,i)=>(i!=2)?e/2:e) + "\n"
				break;
			case "NeuralNetwork":
				let val = `${layer.Weights[0][0].length} `
				for(let i = 0; i<layer.Bias.length;i++){
					val += layer.Bias[i].length + " "
				}
				model_str += name + "     " + val + "\n"
				break;
		}
	}
	print(model_str)
	return 0
}
function forwardPass(model,input){
	let y = input
	for(let layer of model){
		switch (layer.constructor.name){
			case "Convolution":
				y = layer.convolution(y)
				break;
			case "MaxPool":
				y = layer.pool(y)
				break;
			case "NeuralNetwork":
				y= LA.vectorize(y)
				y = layer.GetLayerValues(y)
				layer.Nodes = y
				break;
			default:
				throw console.error("unidentified layer",layer.constructor.name);
		}
	}
	return y
}
function backwordPass(model,desired,filters=[]){
	let y = new Array(model.length).fill(0)
	let f = filters.length-1
	for(let i = model.length-1; i > -1 ; i--){
		layer = model[i]
		switch (layer.constructor.name) {
			case "Convolution":
				f -= 1
				y[i] = layer.layerGrads(y[i+1])
				y[i] = LA.vectorize(y[i])
				y[i] = LA.reconstructMatrix(y[i],{x:y[i].length/filters[f],y:filters[f],z:1}).flat(1)
				y[i] = LA.transpose(y[i])
				break;
			case "MaxPool":
				y[i] = layer.layerGrads(y[i+1])
				break;
			case "NeuralNetwork":
				layer.changes(desired,layer.Nodes[layer.Nodes.length-1])
				y[i] = layer.getInputGradients()
				y[i] = LA.reconstructMatrix(y[i],{x:y[i].length/filters[f],y:filters[f],z:1}).flat(1)
				y[i] = LA.transpose(y[i])
				break;
			default:
				throw console.error("Unidentified model layer");
		}
	}
	return y
}
function updateModel(model,gradientArray,learningRates){
	let k = 0, i = 1
	for(let layer of model){
		switch (layer.constructor.name){
			case "Convolution":
				layer.filterGrads(gradientArray[i],learningRates[k])
				k+=1
				break;
			case "MaxPool":
				break;
			case "NeuralNetwork":
				layer.update(layer.WeightUpdates,layer.BiasUpdates,learningRates[k]);
				k+=1
				break;
			default:
			   throw console.error("Unidentified model layer");
		}
		i+= 1
	}
}
function saveModel(model,folder){
	for(let layer of model){
		switch (layer.constructor.name){
			case "Convolution":
				layer.saveFilters(folder)
				break;
			case "MaxPool":
				layer.savePool(folder)
				break;
			case "NeuralNetwork":
				layer.save(folder)
				break;
			default:
			   throw console.error("Unidentified model layer");
		}

	}	
}