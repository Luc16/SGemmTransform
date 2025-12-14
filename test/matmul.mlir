// Tipos de tensor
!a_t = tensor<192x256xf32>   // A: M=192, K=256
!b_t = tensor<256x64xf32>    // B: K=256, N=64
!c_t = tensor<192x64xf32>    // C: M=192, N=64

module {
  func.func @matmul(%A: !a_t, %B: !b_t, %Cinit: !c_t) -> !c_t {
    %C = linalg.matmul
      ins(%A, %B : !a_t, !b_t)
      outs(%Cinit : !c_t) -> !c_t
    return %C : !c_t
  }
}
