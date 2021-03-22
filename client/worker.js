import { solve_with_api } from './wasm.js'

onmessage = function(input) {
    var result = solve_with_api(input.data[0], input.data[1]);
    //console.log("result: " + result);
    postMessage(result);
  }