
// Here we are importing the default export from our
// Outputted wasm-bindgen ES Module. As well as importing
// the named exports that are individual wrapper functions
// to facilitate handle data passing between JS and Wasm.
import wasmInit, {
    gomoku_solve,
  } from "./pkg/gomoku.js";

// Initialize WASM
wasmInit("./pkg/gomoku_bg.wasm").then(() => {
  console.log("WASM loaded successfully in main thread");
});

export function solve_with_api(input, algo_type, width, height) {
  // Call our exported function
  const result = gomoku_solve(input, algo_type, width, height);

  // Log the result to the console
  //console.log(result);
  return result;
}
