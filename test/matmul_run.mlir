// Wrap everything in a single module
!t_sq = tensor<2048x2048xf64>
module {
    func.func private @timestamp() -> i64
    func.func private @print_gflops(f64, f64)
    func.func private @print_f64(f64)

    // CHECK-LABEL: func @main
    // CHECK-NOT:   memref.alloc
    // CHECK:       %C_result = linalg.matmul
    func.func @main() {
        // Define tensor type alias

        // 1. Create the tensor initialized with 1.0
        %cst_tensor_val = arith.constant dense<1.0> : !t_sq

        // 2. Time the linalg.matmul operation
        %start_time = func.call @timestamp() : () -> i64
        
        // Use %cst_tensor_val directly for A, B, and C_init
        %C_result = linalg.matmul
          ins(%cst_tensor_val, %cst_tensor_val : !t_sq, !t_sq)
         outs(%cst_tensor_val : !t_sq) -> !t_sq

        %end_time = func.call @timestamp() : () -> i64

        // 3. Calculate GFLOP/s
        %size = arith.constant 2048.0 : f64
        %two = arith.constant 2.0 : f64
        %size_squared = arith.mulf %size, %size : f64      // N*N
        %size_cubed = arith.mulf %size_squared, %size : f64 // N*N*N
        %total_flops = arith.mulf %size_cubed, %two : f64   // (N^3) * 2

        %billion = arith.constant 1000000000.0 : f64

        // Calculate elapsed time in nanoseconds
        %elapsed_ns_i = arith.subi %end_time, %start_time : i64
        // Convert elapsed time to a double
        %elapsed_ns_f = arith.uitofp %elapsed_ns_i : i64 to f64

        // Calculate elapsed time in seconds
        %elapsed_s = arith.divf %elapsed_ns_f, %billion : f64
        
        // Calculate GFLOP/s
        %gflops = arith.divf %total_flops, %elapsed_ns_f : f64

        // 4. Print the final result
        call @print_gflops(%gflops, %elapsed_s) : (f64, f64) -> ()

		%c0 = arith.constant 0 : index
        %first_element = tensor.extract %C_result[%c0, %c0] : !t_sq
        call @print_f64(%first_element) : (f64) -> ()

	return
    }
} // <--- Close the module
