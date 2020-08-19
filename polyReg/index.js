let x_vals = [];
let y_vals = [];

let a, b, c, d, e;
let dragging = false;

let cur = 0;

let order = 1;
let epoch = 1000;

const learningRate = 0.2;

function changeOrder(){
    order = document.getElementById("order").value;
    a = b = c = d = e = 0;
    tf.tidy(()=> {
        a = tf.variable(tf.scalar(random(-1, 1)));
        b = tf.variable(tf.scalar(random(-1, 1)));
        if(order >= 2)
            c = tf.variable(tf.scalar(random(-1, 1)));
        if(order >= 3)
            d = tf.variable(tf.scalar(random(-1, 1)));
        if(order >= 4)
            e = tf.variable(tf.scalar(random(-1, 1))); 
    });
}

function changeEpoch(){ 
    epoch = document.getElementById("epoch").value;
    cur = 0;
    //console.log(epoch);
}

function readDirectory(e) {
    var content = "";
    for(let i = 0; i < e.target.files.length; i++){
        content += e.target.files[i].name + "\n ";
    }
    displayContents(content);
}
                
function readSingleFile(e) {
    //console.log(e);
    var file = e.target.files[0];
    if (!file) {
        return;
    }
    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target.result;
        // Display file content
        displayContents(contents);
        i = 0;
    };
    reader.readAsText(file);
}
            
function displayContents(contents) {
    //var element = document.getElementById('file-content');
    var data = JSON.parse(contents);
    x_vals = data.x;
    y_vals = data.y;
    //for(var i = 0; i < x.length; i++)
    //    console.log("("+x_vals[i]+" , "+y_vals[i]+")");
    //element.innerHTML = contents;
}

setTimeout(()=>{
    document.getElementById('file-input').addEventListener('change', readSingleFile, false);     
    //document.getElementById('directory-input').addEventListener('change', readDirectory, false);
},500);

const optimizer = tf.train.adam(learningRate);

function dis() {
    console.log("a : " + a);
    console.log("b : " + b);
    console.log("c : " + c);
    console.log("d : " + d);
    console.log("e : " + e);

    var x = document.getElementById("x").value;
    var y;
    tf.tidy(() => {
      y = predict([x]).dataSync();
    });
    console.log(x);
    console.log(y[0]);
    document.querySelector("#file-content").textContent = y[0];
}

function setup() {
  var can = createCanvas(400, 400);
  can.parent('canvasid');
  a = tf.variable(tf.scalar(random(-1, 1)));
  b = tf.variable(tf.scalar(random(-1, 1)));
  if(order >= 2)
      c = tf.variable(tf.scalar(random(-1, 1)));
  if(order >= 3)
      d = tf.variable(tf.scalar(random(-1, 1)));
  if(order >= 4)
      e = tf.variable(tf.scalar(random(-1, 1))); 
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(x) {
  let xs = tf.tensor1d(x);
  // y = ax^4 + bx^3 + cx^2 + dx + e
  let ys ;
  if(order == 4 ) {
    ys = xs
    .pow(tf.scalar(4)).mul(a)
    .add(xs.pow(tf.scalar(3)).mul(b))
    .add(xs.square().mul(c))
    .add(xs.mul(d))
    .add(e); 
  } else if (order == 3 ) {
    ys = xs
    .pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d)
  } else if (order == 2 ) {
    ys = xs
    .square().mul(a)
    .add(xs.mul(b))
    .add(c)
  } else if (order == 1 ) {
    ys = xs.mul(a).add(b);
  }
  return ys;
}
/*
function mousePressedCan() {
  dragging = true;
}

function mouseReleasedCan() {
  dragging = false;
}
*/
function draw() {
/*
  if (dragging) {
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  } else {
*/  
    if(cur < epoch){
        tf.tidy(() => {
            if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
            }
            cur++;
        });
        //console.log("training");
    } else {
        //console.log("idle");
    }
//  }

  background(0);

  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], -1, 1, height, 0);
    point(px, py);
  }

  const curveX = [];
  for (let x = -1; x <= 1; x += 0.05) {
    curveX.push(x);
  }

  const ys = tf.tidy(() => predict(curveX));
  let curveY = ys.dataSync();
  ys.dispose();

  beginShape();
  noFill();
  stroke(255);
  strokeWeight(2);
  for (let i = 0; i < curveX.length; i++) {
    let x = map(curveX[i], -1, 1, 0, width);
    let y = map(curveY[i], -1, 1, height, 0);
    vertex(x, y);
  }
  endShape();

  //console.log(tf.memory().numTensors);
}     