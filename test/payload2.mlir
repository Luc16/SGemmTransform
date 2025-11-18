// Tipos de tensor
!a_t = tensor<2048x2048xf32>   // A: M=2048, K=2048
!b_t = tensor<2048x2048xf32>   // B: K=2048, N=2048
!c_t = tensor<2048x2048xf32>   // C: M=2048, N=2048

module {
  func.func @matmul(%A: !a_t, %B: !b_t, %Cinit: !c_t) -> !c_t {
    %C = linalg.matmul
      ins(%A, %B : !a_t, !b_t)
      outs(%Cinit : !c_t) -> !c_t
    return %C : !c_t
  }
}
