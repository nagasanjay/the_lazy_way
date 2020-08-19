let baseRecognizer;
let transferRecognizer;

function collect(label) {
    if (label == null) {
        return;
    }
    transferRecognizer.collectExample(label);
    document.querySelector('#console').textContent =
       `${transferRecognizer.countExamples()}`;
    console.log(transferRecognizer.countExamples());
}

function train() {
    transferRecognizer.train({
        epochs: 25,
        callback: {
          onEpochEnd: async (epoch, logs) => {
            document.querySelector('#console').textContent = 
            `Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`;
          }
        }
    });
}

function listen() {
    var x = transferRecognizer.listen(result => {
        console.log("here");
        const words = transferRecognizer.wordLabels();
        console.log(words.length);
        var max = 0;
        var score = 0;
        document.querySelector('#console').textContent = ``;
        for (let i = 0; i < words.length; ++i) {
            console.log('score for word '+words[i] + '=' + result.scores[i]);
            document.querySelector('#console').textContent += 
            `\n\nscore for word ${words[i]} = ${result.scores[i]}`;
            if(score < result.scores[i]){
                score = result.scores[i];
                max = i;
            }
        }
        document.querySelector('#output').textContent = `The word you said is ${words[max]}`;
    }, {probabilityThreshold: 0.2});
    setTimeout(() => transferRecognizer.stopListening(), 10e2);
    console.log(x);
}

async function app() {
    baseRecognizer = speechCommands.create('BROWSER_FFT');
    await baseRecognizer.ensureModelLoaded();
    console.log("loaded");
    transferRecognizer = baseRecognizer.createTransfer('name');
}

app();