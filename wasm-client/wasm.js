
// Here we are importing the default export from our
// Outputted wasm-bindgen ES Module. As well as importing
// the named exports that are individual wrapper functions
// to facilitate handle data passing between JS and Wasm.
import wasmInit, {
    add_str,
    gomoku_solve,
  } from "./pkg/wasm_demo.js";
  
  wasmInit("./pkg/wasm_demo_bg.wasm");

  export function solve_now(input, player) {
    // Instantiate our wasm module

    // Call our exported function
    const result = gomoku_solve(input, player);

    // Log the result to the console
    console.log(result);
    return result;
  }

