# mkdir build && cd build
cd build
cmake -G Ninja .. \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
   -DMLIR_DIR=$HOME/work/llvm-project/build/lib/cmake/mlir \
   -DLLVM_DIR=$HOME/work/llvm-project/build/lib/cmake/llvm
cmake --build . --target transform-opt -j$(nproc)
cd ..

# [ ! -e compile_commands.json ] && ln -s build/compile_commands.json .
