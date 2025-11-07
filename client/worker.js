import { solve_with_api } from './wasm.js'

onmessage = function(input) {
    // input.data expected [boardStr, algo_type, width, height]
    var result = solve_with_api(input.data[0], input.data[1], input.data[2], input.data[3]);
    //console.log("result: " + result);
    postMessage(result);
  }