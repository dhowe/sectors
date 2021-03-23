//require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const fs = require('fs');
const np = require('jsnumpy');
const rita = require('rita');
const use = require('@tensorflow-models/universal-sentence-encoder');
const similarity = require('compute-cosine-similarity');

function cosine(u, v) {
  return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
}

let model, cvec, refsent = "My favorite food is strawberry ice cream.";
let data = fs.readFileSync('./data/dracula.txt', { encoding: 'utf8' });
let sentences = rita.sentences(data);//.slice(0, 1000);
console.log(`Found ${sentences.length} sentences`);

use.load().then(m => {
  (model = m).embed([refsent]).then(e => cvec = e.arraySync()[0]);
}).then(() => {
  console.time("embed");
  //cembed = await model.embed(comparison);
  //console.log(cembed);
  model.embed(sentences).then(embeddings => {
    console.timeEnd("embed");
    let vectors = embeddings.arraySync();
    let all = vectors.map((v, i) => {
      let cs = similarity(v, cvec);
      return { sentence: sentences[i], distance: 1 - cs };
      //console.log(i+") "+cs, sentences[i]);
    });
    console.log(all.sort((a, b) => a.distance - b.distance).map(s => s.sentence).slice(0, 10));

    // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
    // So in this example `embeddings` has the shape [2, 512].
    //console.log(Object.keys(embeddings));;
    //console.log(embeddings.arraySync()[0]);
    //embeddings.print(true /* verbose */);


  });
});


return;

use.load().then(model => {
  console.time("embed");
  //cembed = await model.embed(comparison);
  //console.log(cembed);
  model.embed(sentences).then(embeddings => {
    let vectors = embeddings.arraySync();
    vectors.forEach(v => { });
    // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
    // So in this example `embeddings` has the shape [2, 512].
    //console.log(Object.keys(embeddings));;
    //console.log(embeddings.arraySync()[0]);
    //embeddings.print(true /* verbose */);

    console.timeEnd("embed");
  });
});
