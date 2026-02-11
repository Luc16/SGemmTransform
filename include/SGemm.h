//===-- SGemm.h - Transform dialect Extension SGemm -------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the SGemm.
//
//===----------------------------------------------------------------------===//

#ifndef SGEMM_H
#define SGEMM_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Transform/IR/TransformAttrs.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>

namespace mlir {
class CallOpInterface;
class RewriterBase;

namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg

namespace transform {
class AnyOpType;
class AnyValueType;
class OperationType;
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;
} // namespace mlir

// To debug info, use: transform-opt your_parameters -debug-only=SGemm
#define DEBUG_TYPE "sgemm"
#define DBGS() (llvm::dbgs() << "\n[" DEBUG_TYPE "] ")

#define GET_OP_CLASSES
#include "SGemm.h.inc"

// Registers our Transform dialect extension.
void registerSGemm(::mlir::DialectRegistry &registry);

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

namespace {

using namespace mlir;

/// Create a named location from a name and another location.
static inline Location CreateNameLoc(const Twine &name, Location loc) {
#ifndef NDEBUG
  return NameLoc::get(StringAttr::get(loc.getContext(), name), loc);
#else
  return loc;
#endif
}

/// Set named location to an existing value
static inline Value SetNameLoc(Value value, const Twine &name) {
#ifndef NDEBUG
  value.setLoc(CreateNameLoc(name, value.getLoc()));
#endif
  return value;
}

/// Set named location to an existing op with a single result
static inline Operation *SetNameLoc(Operation *op, const Twine &name) {
#ifndef NDEBUG
  assert(op->getNumResults() == 1 && "Operation must have a single result");
  SetNameLoc(op->getResult(0), name);
#endif
  return op;
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  Type ety = x.getType();
  if (isa<IntegerType>(ety))
    return builder.create<arith::AddIOp>(loc, x, y);
  else if (ety.isF16() || ety.isBF16() || ety.isF32() || ety.isF64())
    return builder.create<arith::AddFOp>(loc, x, y);
  return nullptr;
}

static Value createMul(Location loc, Value x, Value y, OpBuilder &builder) {
  Type ety = x.getType();
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, ety, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, ety, /*isUnsignedCast=*/false);
  if (isa<IntegerType>(ety))
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  else if (ety.isF16() || ety.isBF16() || ety.isF32() || ety.isF64())
    return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// Inline functions
//===----------------------------------------------------------------------===//

static inline mlir::Location operator<<(mlir::Location loc,
                                        const mlir::Twine &name) {
  return CreateNameLoc(name, loc);
}

static inline mlir::Value operator<<(mlir::Value value,
                                     const mlir::Twine &name) {
  return SetNameLoc(value, name);
}

static inline mlir::Operation *operator<<(mlir::Operation *op,
                                          const mlir::Twine &name) {
  return SetNameLoc(op, name);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

typedef struct {
  uint32_t l1_size;
  uint32_t l2_size;
  uint32_t l3_size;
  uint32_t vec_len; // In bits
  uint32_t num_vecs; // Number of vector registers  
  uint32_t is_risc;
} ArchInfo;

typedef struct {
  uint8_t nrows;
  uint8_t ncols;
  uint8_t unroll;
  uint16_t noutput;
} mKInfo;

/// Pack Mc, Kc, Nc (outer tiles) and mr, nr (inner tiles) from mKInfo.
/// We mirror mK.nrows as "mr" and mK.ncols as "nr".
struct GemmTileSizes {
  // Outer tiles
  int64_t Mc = 0, Kc = 0, Nc = 0;
  // Inner tiles (micro-kernel)
  int64_t mr = 0, nr = 0;
};

struct GemmOriginalSizes {
	int64_t M = 0, K = 0, N = 0;
};

enum class PadNeed { None, Needed, Maybe };

struct MaybePaddedABC {
  Value A, B, C;
  bool didPadA = false;
  bool didPadB = false;
  bool didPadC = false;
};

struct PackedAB {
  Value aPack;
  Value bPack;
  RankedTensorType aPackTy;
  RankedTensorType bPackTy;
};

#endif // SGEMM_H
