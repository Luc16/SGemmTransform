//===-- SGemm.cpp - transform dialect Extension SGemm ------------------===//
//----------------------------------------------------------------------------//

#include "SGemm.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::linalg;
using namespace mlir::transform;

#define GET_OP_CLASSES
#include "SGemm.cpp.inc"

/// Define the SGemm transform dialect. This uses the CRTP idiom to identify extensions.
class SGemm
    : public transform::TransformDialectExtension<SGemm> {
public:
  // The TypeID of this extension.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SGemm)

  // The extension must derive the base constructor.
  using Base::Base;

  // This function initializes the extension, similarly to `initialize` in
  // dialect definitions. List individual operations and dependent dialects
  // here.
  void init();
};

void SGemm::init() {
  // As an transform extension dialect, we must declare all dependent dialects.
  // These dialects will be loaded along with the extension and, therefore,
  // along with the Transform dialect. The dependent dialects contain the
  // attributes or types used by transform operations.
  declareDependentDialect<linalg::LinalgDialect>();

  // When transformations are applied, they may produce new operations from
  // previously unloaded dialects. Typically, a pass would need to declare
  // itself dependent on the dialects containing such new operations. To avoid
  // confusion with the dialects the extension itself depends on, the Transform
  // dialects differentiates between:
  //   - dependent dialects, which are used by the transform operations, and
  //   - generated dialects, which contain the entities (attributes, operations,
  //     types) that may be produced by applying the transformation even when
  //     not present in the original payload IR.
  declareGeneratedDialect<affine::AffineDialect>();
  declareGeneratedDialect<arith::ArithDialect>();
  declareGeneratedDialect<index::IndexDialect>();
  declareGeneratedDialect<scf::SCFDialect>();
  declareGeneratedDialect<tensor::TensorDialect>();
  declareGeneratedDialect<LLVM::LLVMDialect>();

  // Finally, we register the additional transform operations with the dialect.
  // List all operations generated from ODS. This call will perform additional
  // checks that the operations implement the transform and memory effect
  // interfaces required by the dialect interpreter and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "SGemm.cpp.inc"
      >();
}

// ====--------------------------------------------------------------------------
// Utilities for zero constants, arithmetic on index values, and padding tensors.
// ====--------------------------------------------------------------------------

/// Create a zero constant compatible with `elemTy`.
/// Supports float, integer, and index element types.
/// Falls back to bitcast from i32 when needed.
static Value buildZeroLike(OpBuilder &b, Location loc, Type elemTy) {
  if (isa<FloatType>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getFloatAttr(elemTy, 0.0));

  if (auto it = dyn_cast<IntegerType>(elemTy))
    return b.create<arith::ConstantOp>(loc, b.getIntegerAttr(elemTy, 0));

  if (isa<IndexType>(elemTy))
    return b.create<arith::ConstantIndexOp>(loc, 0);

  // Fallback: materialize i32 0 and bitcast to the requested element type.
  auto i32Zero = b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(0));
  return b.create<arith::BitcastOp>(loc, elemTy, i32Zero);
}

/// Compute ceilDiv(x, m) for index-typed `x` and constant `m > 0`.
/// This is safe for dynamic shapes because it uses index ops.
static Value buildCeilDiv(OpBuilder &b, Location loc, Value x, int64_t m) {
  assert(m > 0 && "ceilDiv divisor must be positive");
  Value mC = b.create<arith::ConstantIndexOp>(loc, m);
  Value oneLess = b.create<arith::ConstantIndexOp>(loc, m - 1);
  Value num = b.create<arith::AddIOp>(loc, x, oneLess);
  return b.create<arith::DivUIOp>(loc, num, mC);
}

/// Multiply index-typed `a` by a positive constant `cst`.
static Value buildMul(OpBuilder &b, Location loc, Value a, int64_t cst) {
  assert(cst >= 0 && "mul constant must be non-negative");
  return b.create<arith::MulIOp>(loc, a, b.create<arith::ConstantIndexOp>(loc, cst));
}

/// Get `tensor.dim` as index value for ranked tensors.
static Value dimAsIndex(OpBuilder &b, Location loc, Value tensor, int64_t d) {
  return b.create<tensor::DimOp>(loc, tensor, d);
}

/// Pad `tensor` so that selected dimensions become multiples of the given `multiples`.
/// - `dimsToPad`: dimensions to be padded (e.g., {0, 1}).
/// - `multiples`: target multiples per dimension (e.g., {MR, KC}).
/// The function generates `tensor.pad` with low=0 and computed high padding, yielding
/// zero in the padding region. Returns the padded tensor value on success.
static FailureOr<Value> padToMultiples(RewriterBase &rewriter, Location loc,
                                       Value tensor,
                                       ArrayRef<int64_t> dimsToPad,
                                       ArrayRef<int64_t> multiples) {

  auto rtt = dyn_cast<RankedTensorType>(tensor.getType());
  if (!rtt || dimsToPad.size() != multiples.size())
    return failure();

  // Prepare low and high paddings for all dimensions.
  SmallVector<OpFoldResult> lowPads(rtt.getRank());
  SmallVector<OpFoldResult> highPads(rtt.getRank());
  for (int64_t d = 0, rank = rtt.getRank(); d < rank; ++d) {
    // Low padding is always 0 for our use case.
    lowPads[d] = rewriter.getIndexAttr(0);

    // Determine if this dimension must be padded to a multiple.
    int64_t listIdx = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(dimsToPad.size()); ++i)
      if (dimsToPad[i] == d) { listIdx = i; break; }

    if (listIdx < 0) {
      // No padding on this dimension.
      highPads[d] = rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult();
      continue;
    }

    // Compute high pad amount per dimension.
    // target = ceilDiv(dim, multiple) * multiple; high = target - dim.
    const int64_t mult = multiples[listIdx];
    int64_t dimStatic = rtt.getDimSize(d);

    if (!ShapedType::isDynamic(dimStatic)) {
      // Fully static: compute high pad as a constant.
      int64_t blocks = (dimStatic + mult - 1) / mult;
      int64_t targetStatic = blocks * mult;
      int64_t highStatic = targetStatic - dimStatic;
      highPads[d] = rewriter.getIndexAttr(highStatic);
    } else {
      // Dynamic: fall back to SSA computations.
      Value dimV = dimAsIndex(rewriter, loc, tensor, d);
      Value ceilDV = buildCeilDiv(rewriter, loc, dimV, mult);
      Value target = buildMul(rewriter, loc, ceilDV, mult);
      highPads[d] = rewriter.create<arith::SubIOp>(loc, target, dimV).getResult();
    }
  }

  // Before creating tensor::PadOp, fast-path when no padding will be added.
  // If all computed high pads are constant 0, return the original tensor.
  bool allConstZero = llvm::all_of(highPads, [](OpFoldResult ofr) {
    return mlir::isZeroInteger(ofr);
  });
  if (allConstZero)
    return tensor; // no-op: already multiple-aligned

  // Compute result type (keeps shapes static when inputs/multiples are static).
  SmallVector<int64_t> outShape(rtt.getShape().begin(), rtt.getShape().end());
  for (auto it : llvm::enumerate(dimsToPad)) {
    int64_t d = it.value();
    int64_t mult = multiples[it.index()];
    int64_t sz = rtt.getDimSize(d);
    if (!ShapedType::isDynamic(sz)) {
      int64_t blocks = (sz + mult - 1) / mult;
      outShape[d] = blocks * mult;         // <- static padded size
    } else {
      outShape[d] = ShapedType::kDynamic;  // keep dynamic only if truly dynamic
    }
  }
    auto resType = RankedTensorType::get(outShape, rtt.getElementType());

  // Build the pad op with an explicit (preferably static) result type.
  auto padOp = rewriter.create<tensor::PadOp>(
      loc,
      /*resultType=*/resType,
      tensor,
      /*low=*/lowPads,
      /*high=*/highPads,
      /*nofold=*/false /* allow canonicalizations when safe */);

  // Create the pad region block and add `rank` index block arguments.
  {
    OpBuilder::InsertionGuard guard(rewriter);
    Block *body = rewriter.createBlock(&padOp.getRegion());
    int64_t rank = rtt.getRank();
    for (int64_t d = 0; d < rank; ++d)
      body->addArgument(rewriter.getIndexType(), loc);

    // Yield zero as the padding value.
    Value zero = buildZeroLike(rewriter, loc, rtt.getElementType());
    rewriter.create<tensor::YieldOp>(loc, zero);
  }

  return padOp.getResult();

}

// Return whether a ranked tensor type needs padding to become multiples of `multiples`.
// - None:   all static dims are already multiples -> no padding needed
// - Needed: at least one static dim is not a multiple -> padding definitely needed
// - Maybe:  some dims are dynamic -> may need runtime padding, keep the path enabled
static PadNeed needsPaddingToMultiples(RankedTensorType rtt,
                                       ArrayRef<int64_t> multiples) {

  assert(rtt && "expected ranked tensor type");
  assert(static_cast<size_t>(rtt.getRank()) == multiples.size() &&
         "rank/multiples size mismatch");
  bool sawDynamic = false;
  for (int64_t d = 0; d < rtt.getRank(); ++d) {
    int64_t sz = rtt.getDimSize(d);
    int64_t m  = multiples[d];
    if (ShapedType::isDynamic(sz)) {
      sawDynamic = true;
      continue;
    }
    if (m > 1 && (sz % m) != 0) return PadNeed::Needed;
  }
  return sawDynamic ? PadNeed::Maybe : PadNeed::None;
}


/// Extract A/B/C from a linalg.generic that models Matmul (A@0, B@1, C init@0),
/// set a safe insertion point, and invoke padToMultiples on each tensor.
/// IMPORTANT: At this stage we do not replace the operands of the generic yet.
/// We only compute and return the padded tensors so the caller can later decide
/// how to rewire the IR (e.g., cloning the op or stitching slices).
static FailureOr<MaybePaddedABC>
preparePaddingForGemmLike(RewriterBase &rewriter,
                          linalg::GenericOp gemmLike,
                          const GemmTileSizes &ts) {

  if (!gemmLike) return failure();
  Location loc = gemmLike.getLoc();

  // Expect (A,B) as DPS inputs and (C) as the sole init.
  if (gemmLike.getNumDpsInputs() < 2 || gemmLike.getNumDpsInits() < 1)
    return failure();

  Value A = gemmLike.getDpsInputs()[0]; // [M,K]
  Value B = gemmLike.getDpsInputs()[1]; // [K,N]
  Value C = gemmLike.getDpsInits()[0];  // [M,N]

  // If all three are statically-known multiples, bail out early.
  auto rttA = dyn_cast<RankedTensorType>(A.getType());
  auto rttB = dyn_cast<RankedTensorType>(B.getType());
  auto rttC = dyn_cast<RankedTensorType>(C.getType());
  if (!rttA || !rttB || !rttC) return failure(); // expect ranked tensors

  PadNeed needA = needsPaddingToMultiples(rttA, {ts.mr, ts.Kc});
  PadNeed needB = needsPaddingToMultiples(rttB, {ts.Kc, ts.nr});
  PadNeed needC = needsPaddingToMultiples(rttC, {ts.mr, ts.nr});

  if (needA == PadNeed::None && needB == PadNeed::None && needC == PadNeed::None) {
    // Nothing to do: keep the IR pristine (no prelude ops).
    return MaybePaddedABC{A, B, C, /*didPadA=*/false, /*didPadB=*/false, /*didPadC=*/false};
  }

  // From here on we may create pad ops (Needed/Maybe).
  // Choose an insertion point right before the gemmLike.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(gemmLike);

  MaybePaddedABC out{A, B, C, false, false, false};

  if (needA != PadNeed::None) {
    // Pad A on (M,K) to multiples of (mr, Kc)
    auto aPadOr = padToMultiples(rewriter, loc, A,
                                 /*dimsToPad=*/ArrayRef<int64_t>{0, 1},
                                 /*multiples=*/ArrayRef<int64_t>{ts.mr, ts.Kc});
    if (mlir::succeeded(aPadOr)) {
      Value aPad = *aPadOr;
      out.A = aPad;
      out.didPadA = true;
    }
  }

  if (needB != PadNeed::None) {
    // Pad B on (K, N) to multiples of (Kc, nr).
    // Rationale: the reduction dimension must align to Kc; output columns align to nr.
    auto bPadOr = padToMultiples(rewriter, loc, B,
                                 /*dimsToPad=*/ArrayRef<int64_t>{0, 1},
                                 /*multiples=*/ArrayRef<int64_t>{ts.Kc, ts.nr});
    if (mlir::succeeded(bPadOr)) {
      Value bPad = *bPadOr;
      out.B = bPad;
      out.didPadB = true;
    }
  }

  if (needC != PadNeed::None) {
    // Pad C on (M, N) to multiples of (mr, nr).
    // Rationale: the output micro-tile is mr x nr, keep tails aligned to the micro-kernel.
    auto cPadOr = padToMultiples(rewriter, loc, C,
                                 /*dimsToPad=*/ArrayRef<int64_t>{0, 1},
                                 /*multiples=*/ArrayRef<int64_t>{ts.mr, ts.nr});
    if (mlir::succeeded(cPadOr)) {
      auto cPad = *cPadOr;
      out.C = cPad;
      out.didPadC = true;
    }
  }

  return out;
}

//===----------------------------------------------------------------------===//
// Helpers for 'tensor.empty' with dynamic dims
//===----------------------------------------------------------------------===//

/// Converts an OpFoldResult to a Value, materializing an index constant when it's an Attribute.
static Value valueFromOFR(OpBuilder &rewriter, Location loc, OpFoldResult ofr) {
  if (auto v = dyn_cast<Value>(ofr))
    return v;
  // Attribute branch: expect IntegerAttr for index.
  auto attr = dyn_cast<Attribute>(ofr);
  auto ia   = mlir::cast<IntegerAttr>(attr);
  return rewriter.create<arith::ConstantIndexOp>(loc, ia.getInt());

}

/// Creates a tensor.empty of type `resultTy`.
/// For every dynamic dimension ('?') in `resultTy`, the caller must supply one
/// size Value in `dynSizesInRankOrder`, in increasing dimension order.
static Value buildEmptyForType(OpBuilder &rewriter, Location loc,
                               RankedTensorType resultTy,
                               ArrayRef<Value> dynSizesInRankOrder) {
  // Collect the static shape and the dynamic size operands in rank order.
  SmallVector<int64_t> staticShape(resultTy.getShape().begin(),
                                   resultTy.getShape().end());

  SmallVector<Value> dynSizes;
  dynSizes.reserve(dynSizesInRankOrder.size());
  for (int64_t d = 0, idx = 0; d < (int64_t)staticShape.size(); ++d) {
    if (ShapedType::isDynamic(staticShape[d])) {
      assert(idx < (int64_t)dynSizesInRankOrder.size() &&
             "missing dynamic size operand for dynamic dim");
      dynSizes.push_back(dynSizesInRankOrder[idx++]);
    }
  }

  // Build the EmptyOp using the (staticShape, elementType, dynamicSizes, encoding) order.
  return rewriter.create<tensor::EmptyOp>(
      loc,
      /*staticShape=*/ArrayRef<int64_t>(staticShape),
      /*elementType=*/resultTy.getElementType(),
      /*dynamicSizes=*/ValueRange(dynSizes),
      /*encoding=*/resultTy.getEncoding());
}

/// Extract one size Value for a given dimension from a tensor value.
/// If the producer is a tensor.extract_slice, reuse its mixedSizes[i];
/// otherwise, fall back to tensor.dim (which is fine at tile granularity).
static Value getDimValueFor(OpBuilder &rewriter, Location loc, Value tensor, int64_t dim) {
  if (Operation *def = tensor.getDefiningOp())
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def)) {
      OpFoldResult sz = slice.getMixedSizes()[dim];
      if (auto cst = dyn_cast<Attribute>(sz)) {
        // Promote static integer to an index constant.
        auto ival = cast<IntegerAttr>(cst).getInt();
        return rewriter.create<arith::ConstantIndexOp>(loc, ival);
      }
      return valueFromOFR(rewriter, loc, sz);
    }
  // Generic fallback.
  return rewriter.create<tensor::DimOp>(loc, tensor, dim);
}

//===----------------------------------------------------------------------===//
// Packing helpers (row-panel for A, col-panel for B)
//===----------------------------------------------------------------------===//

/// Return the ExtractSliceOp that ultimately produces `v`, or null if none.
static tensor::ExtractSliceOp getSliceProducerOrNull(Value v) {
  Value cur = v;
  while (Operation *def = cur.getDefiningOp()) {
    if (auto slice = dyn_cast<tensor::ExtractSliceOp>(def))
      return slice;
    // Skip through trivial wrappers.
    continue;
  }
  return nullptr;
}

/// Conservatively check whether `slice` is invariant in the current `loop`.
static bool isExtractSliceInvariantToLoop(tensor::ExtractSliceOp slice,
                                          scf::ForOp loop) {
  if (!loop) return true;
  auto *loopBody = loop.getBody();
  Value iv = loop.getInductionVar();
  for (Value v : slice->getOperands()) {
    if (!v) continue;
    if (v == iv) return false;
    Operation *def = v.getDefiningOp();
    if (!def) {
      // if it belongs to the loop body, be conservative.
      if (auto barg = dyn_cast<BlockArgument>(v))
        if (barg.getOwner() == loopBody) return false;
      continue;
    }
    if (loop->isAncestor(def)) return false;
  }
  return true;
}

/// Compute packed type for A: from [Mc, Kc] to [Mc/mr, Kc, mr].
static RankedTensorType
computeAPackedType(RankedTensorType aType, int64_t mr) {
  auto shape = aType.getShape();
  int64_t Mc = shape[0], Kc = shape[1];
  if (!ShapedType::isDynamic(Mc))
    assert(Mc % mr == 0 && "Mc must be a multiple of mr for simple packing");

  auto divOrDyn = [&](int64_t d, int64_t c) {
    return ShapedType::isDynamic(d) ? ShapedType::kDynamic : (d / c);
  };

  SmallVector<int64_t, 3> newShape = {divOrDyn(Mc, mr), Kc, mr};
  return RankedTensorType::get(newShape, aType.getElementType());
}

/// Compute packed type for B: from [Kc, Nc] to [Kc, Nc/nr, nr].
static RankedTensorType
computeBPackedType(RankedTensorType bType, int64_t nr) {
  auto shape = bType.getShape();
  int64_t Kc = shape[0], Nc = shape[1];
  if (!ShapedType::isDynamic(Nc))
    assert(Nc % nr == 0 && "Nc must be a multiple of nr for simple packing");

  auto divOrDyn = [&](int64_t d, int64_t c) {
    return ShapedType::isDynamic(d) ? ShapedType::kDynamic : (d / c);
  };

  SmallVector<int64_t, 3> newShape = {Kc, divOrDyn(Nc, nr), nr};
  return RankedTensorType::get(newShape, bType.getElementType());
}

/// Build affine maps for A-pack:  in:  (i,k)  -> out: (io, k, ii)
/// with io=floor(i/mr), ii=mod(i,mr)
static void buildAPackMaps(MLIRContext *ctx, int64_t mr,
                           AffineMap &inMap, AffineMap &outMap) {
  // dims: (i,k)
  AffineExpr i, k;
  bindDims(ctx, i, k);
  AffineExpr io = i.floorDiv(mr);
  AffineExpr ii = i % mr;
  inMap  = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{i, k}, ctx);
  outMap = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{io, k, ii}, ctx);
}

/// Build affine maps for B-pack:  in:  (k,j)  -> out: (k, jo, ji)
/// with jo=floor(j/nr), ji=mod(j,nr)
static void buildBPackMaps(MLIRContext *ctx, int64_t nr,
                           AffineMap &inMap, AffineMap &outMap) {
  // dims: (k,j)
  AffineExpr k, j;
  bindDims(ctx, k, j);
  AffineExpr jo = j.floorDiv(nr);
  AffineExpr ji = j % nr;
  inMap  = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{k, j}, ctx);
  outMap = AffineMap::get(/*dims=*/2, /*symbols=*/0, ArrayRef<AffineExpr>{k, jo, ji}, ctx);
}

/// Build A_pack at a precise insertion point. If `afterOp` != nullptr,
/// insert right AFTER it; otherwise insert BEFORE `beforeOp`.
/// in  : A tile type [Mc, Kc]
/// out : A_pack type [Mc/mr, Kc, mr]
static FailureOr<std::pair<Value, RankedTensorType>>
buildAPackAt(RewriterBase &rewriter, Location loc, Value A, int64_t mr,
             Operation *afterOp, Operation *beforeOp) {

  OpBuilder::InsertionGuard g(rewriter);

  if (afterOp) {
    rewriter.setInsertionPointAfter(afterOp);
  } else if (beforeOp) {
    rewriter.setInsertionPoint(beforeOp);
  } else {
    return failure();
  }

  // aType is the rank-2 tile of A: tensor<?xKc> in the tail case.
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  if (!aType || aType.getRank() != 2) return failure();

  // Compute the packed type: [Mc/mr, Kc, mr] with dynamic dims propagated.
  auto aPackTy = computeAPackedType(aType, mr);
  AffineMap aInMap, aOutMap;
  buildAPackMaps(rewriter.getContext(), mr, aInMap, aOutMap);

  // Build dynamic size operands for dynamic dims of aPackTy, in rank order.
  // aType is the tile type of A (rank-2). `A` is the tile value (extract_slice result).
  SmallVector<Value> dynSizesForAPack;
  if (ShapedType::isDynamic(aPackTy.getDimSize(0))) {
    // dim 0 of A_pack is Mc/mr. Get Mc from A tile dim[0] and divide by mr.
    Value mc   = getDimValueFor(rewriter, loc, /*aLocal=*/A, /*dim=*/0);
    Value cMr  = rewriter.create<arith::ConstantIndexOp>(loc, mr);
    Value mcDr = rewriter.create<arith::DivSIOp>(loc, mc, cMr);
    dynSizesForAPack.push_back(mcDr);
  }
  if (ShapedType::isDynamic(aPackTy.getDimSize(1))) {
    // dim 1 of A_pack is Kc (tile K). Reuse tile K from A tile dim[1].
    Value kc = getDimValueFor(rewriter, loc, /*aLocal=*/A, /*dim=*/1);
    dynSizesForAPack.push_back(kc);
  }

  // Create empty for A_pack providing dynamic sizes only for '?' dims.
  Value aEmpty = buildEmptyForType(rewriter, loc, aPackTy, dynSizesForAPack);

  auto aPack = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{aPackTy}, ValueRange{A}, ValueRange{aEmpty},
      ArrayRef<AffineMap>{aInMap, aOutMap},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        b.create<linalg::YieldOp>(nloc, args[0]);
      });
  aPack->setAttrs({{"APacking", rewriter.getUnitAttr()}});
  return std::make_pair(aPack.getResult(0), aPackTy);
}

// Build B_pack at the current insertion point.
// in  : B tile type [Kc, Nc]
// out : B_pack type [Kc, Nc/nr, nr]
static FailureOr<std::pair<Value, RankedTensorType>>
buildBPackAt(RewriterBase &rewriter, Location loc, Value B, int64_t nr) {

  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!bType || bType.getRank() != 2) return failure();
  auto bPackTy = computeBPackedType(bType, nr);

  AffineMap bInMap, bOutMap;
  buildBPackMaps(rewriter.getContext(), nr, bInMap, bOutMap);

  SmallVector<Value> dynSizesForBPack;
  if (ShapedType::isDynamic(bPackTy.getDimSize(0))) {
    // dim 0 of B_pack is Kc. Reuse tile K from B tile dim[0].
    Value kc = getDimValueFor(rewriter, loc, /*bLocal=*/B, /*dim=*/0);
    dynSizesForBPack.push_back(kc);
  }
  if (ShapedType::isDynamic(bPackTy.getDimSize(1))) {
    // dim 1 of B_pack is Nc/nr. Compute Nc from B tile dim[1] and divide by nr.
    Value nc  = getDimValueFor(rewriter, loc, /*bLocal=*/B, /*dim=*/1);
    Value cNr = rewriter.create<arith::ConstantIndexOp>(loc, nr);
    Value ncDr= rewriter.create<arith::DivSIOp>(loc, nc, cNr);
    dynSizesForBPack.push_back(ncDr);
  }

  // Create empty for B_pack providing dynamic sizes only for '?' dims.
  Value bEmpty = buildEmptyForType(rewriter, loc, bPackTy, dynSizesForBPack);

  auto bPack = rewriter.create<linalg::GenericOp>(
      loc, TypeRange{bPackTy}, ValueRange{B}, ValueRange{bEmpty},
      ArrayRef<AffineMap>{bInMap, bOutMap},
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                       utils::IteratorType::parallel},
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        b.create<linalg::YieldOp>(nloc, args[0]);
      });
  bPack->setAttrs({{"BPacking", rewriter.getUnitAttr()}});
  return std::make_pair(bPack.getResult(0), bPackTy);
}

/// Rewrite ukernel (linalg.generic) to read from A_pack / B_pack.
/// Keeps the same output (C tile) and body (mul+add), only remaps inputs.
static FailureOr<linalg::GenericOp>
rewriteUkernelToUsePackedAB(RewriterBase &rewriter, linalg::GenericOp ukernel,
                            Value aPack, Value bPack) {

  MLIRContext *ctx = rewriter.getContext();
  Location loc = ukernel.getLoc();

  // Old operands
  Value oldA = ukernel.getDpsInputs()[0];
  Value oldB = ukernel.getDpsInputs()[1];
  Value oldC = ukernel.getDpsInits()[0];

  // We will keep the same iterator types and the same result tensor type as `oldC`.
  // Only the input operands A/B are swapped to read from packed tensors via new indexing maps.
  SmallVector<utils::IteratorType> iters =
      llvm::to_vector(ukernel.getIteratorTypesArray());
  auto oldMaps = ukernel.getIndexingMapsArray();
  if (oldMaps.size() != 3)
    return failure();

  // Determine loop role positions from iterator types:
  // exactly one reduction (K), and two parallels (M,N) in any order.
  int redPos = -1, par0 = -1, par1 = -1;
  for (int p = 0; p < static_cast<int>(iters.size()); ++p) {
    if (iters[p] == utils::IteratorType::reduction) {
      redPos = p;
    } else {
      (par0 < 0) ? par0 = p : par1 = p;
    }
  }
  if (redPos < 0 || par0 < 0 || par1 < 0)
    return failure();

  // Decide which parallel loop indexes rows (M) and which indexes cols (N)
  // by inspecting the C map. We expect C's map to be rank-2 like (M,N).
  AffineMap cOld = oldMaps[2];
  if (cOld.getNumResults() != 2 || cOld.getNumDims() != static_cast<unsigned>(iters.size()))
    return failure();

  // Extract which dim goes to C's first and second result component.
  auto res0 = dyn_cast<AffineDimExpr>(cOld.getResult(0));
  auto res1 = dyn_cast<AffineDimExpr>(cOld.getResult(1));
  if (!res0 || !res1) return failure();
  int c0 = res0.getPosition();
  int c1 = res1.getPosition();

  // Map those dims to "M" and "N".
  int mPos = c0; // loop dim feeding the first C index
  int nPos = c1; // loop dim feeding the second C index
  if (iters[mPos] != utils::IteratorType::parallel ||
      iters[nPos] != utils::IteratorType::parallel)
    return failure(); // sanity: both must be parallel

  // Query mr/nr from the packed tensor types.
  auto aPackTy = cast<RankedTensorType>(aPack.getType()); // [..., Kc, mr]
  auto bPackTy = cast<RankedTensorType>(bPack.getType()); // [Kc, ..., nr]
  const int64_t mr = aPackTy.getShape().back();
  const int64_t nr = bPackTy.getShape().back();

  // Build new indexing maps (dims = number of ukernel loops).
  // A_pack index: ( floor(M/mr), K, M % mr )
  // B_pack index: ( K, floor(N/nr), N % nr )
  // C index    : ( M, N )
  SmallVector<AffineExpr> dims;
  dims.reserve(iters.size());
  for (unsigned d = 0; d < iters.size(); ++d)
    dims.push_back(getAffineDimExpr(d, ctx));

  AffineExpr M  = dims[mPos];
  AffineExpr N  = dims[nPos];
  AffineExpr K  = dims[redPos];

  AffineExpr io = M.floorDiv(mr);
  AffineExpr ii = M % mr;
  AffineExpr jo = N.floorDiv(nr);
  AffineExpr ji = N % nr;

  AffineMap aPackMap = AffineMap::get(/*dims=*/iters.size(), /*symbols=*/0,
                                      ArrayRef<AffineExpr>{io, K, ii}, ctx);
  AffineMap bPackMap = AffineMap::get(/*dims=*/iters.size(), /*symbols=*/0,
                                      ArrayRef<AffineExpr>{K, jo, ji}, ctx);
  AffineMap cMap     = AffineMap::get(/*dims=*/iters.size(), /*symbols=*/0,
                                      ArrayRef<AffineExpr>{M, N}, ctx);

  // The result tensor type stays exactly the same as oldC's type.
  auto outTy = cast<RankedTensorType>(oldC.getType());
  Value outInit = oldC;

  // Materialize the remapped ukernel that consumes A_pack/B_pack and writes to the same C tile.
  rewriter.setInsertionPoint(ukernel);
  auto remapped = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/TypeRange{outTy},
      /*inputs=*/ValueRange{aPack, bPack},
      /*outputs=*/ValueRange{outInit},
      /*indexingMaps=*/ArrayRef<AffineMap>{aPackMap, bPackMap, cMap},
      /*iteratorTypes=*/iters,
      [&](OpBuilder &b, Location nloc, ValueRange args) {
        // args: A(M,K), B(K,N), Acc(M,N)
        Value mul = createMul(nloc, args[0], args[1], b);
        Value sum = createAdd(nloc, mul, args[2], b);
        b.create<linalg::YieldOp>(nloc, sum);
      });

  // Replace the original ukernel with the remapped one.
  rewriter.replaceOp(ukernel, remapped->getResults());
  remapped->setAttrs({{"microkernel", rewriter.getUnitAttr()}});
  return remapped;

}

/// Factor the code that, given a tiled ukernel, builds A_pack/B_pack at the
/// right insertion points and rewrites the ukernel to consume the packed
/// operands. Updates localResults[0] with the new ukernel.
/// Returns success if packing+rewrite was applied.
static LogicalResult packAndRetargetUkernel(RewriterBase &rewriter,
                                            linalg::GenericOp ukernel,
                                            const mKInfo &mK,
                                            SmallVector<Operation*, 6> &localResults) {

  if (!ukernel) return failure();
  Location loc = ukernel.getLoc();
  int64_t mr = mK.nrows;
  int64_t nr = mK.ncols;

  // Grab original A/B from the ukernel
  Value A = ukernel.getDpsInputs()[0];
  Value B = ukernel.getDpsInputs()[1];

  // Try to hoist A's slice outside the innermost N-loop and decide IPs.
  Operation *afterA = nullptr;
  Operation *beforeA = ukernel.getOperation();

  // Expect localResults[5] to be the innermost scf.for (as produced by tiling).
  if (localResults.size() > 5) {
    if (auto innerFor = dyn_cast_or_null<scf::ForOp>(localResults[5])) {
      if (auto aSlice = getSliceProducerOrNull(A)) {
        if (isExtractSliceInvariantToLoop(aSlice, innerFor)) {
          // Move the slice that feeds A outside the inner loop.
          aSlice->moveBefore(innerFor);
          // Insert A_pack right AFTER this slice
          afterA = aSlice.getOperation();
        }
      }
    }
  }

  // Build A_pack at the chosen insertion point
  auto aPackOr = buildAPackAt(rewriter, loc, A, mr, afterA, beforeA);
  if (failed(aPackOr)) return failure();
  Value aPack = aPackOr->first;

  // Build B_pack right before ukernel
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(ukernel);
  auto bPackOr = buildBPackAt(rewriter, loc, B, nr);
  if (failed(bPackOr)) return failure();
  Value bPack = bPackOr->first;

  // Rewrite ukernel to read from packed A/B and keep C intact
  auto newUkernelOr = rewriteUkernelToUsePackedAB(rewriter, ukernel, aPack, bPack);
  if (failed(newUkernelOr)) return failure();

  linalg::GenericOp newUkernel = *newUkernelOr;
  newUkernel->setAttrs({{"microkernel", rewriter.getUnitAttr()}});
  if (!localResults.empty())
    localResults[0] = newUkernel.getOperation();

  return success();
}

//===----------------------------------------------------------------------===//
// Tiling Helpers
//===----------------------------------------------------------------------===//

/// Selects block (tiling) sizes for a GEMM at two levels  
static GemmTileSizes computeGemmTiles(const mKInfo &mK, const ArchInfo &arch) {

  GemmTileSizes ts;
  // Map SConv-style knobs into GEMM intuitively:
  //   - mr <- nrows (rows of micro-kernel)
  //   - nr <- ncols (cols of micro-kernel)
  //   - Choose outer tiles as multiples of mr/nr and a modest Kc.
  ts.mr = std::max<int64_t>(mK.nrows, 4); // e.g., 4..16
  ts.nr = std::max<int64_t>(mK.ncols, 8); // e.g., 8..32

  // Outer: pick Mc/Nc as a few microkernels; Kc as a reduction chunk.
  ts.Mc = ts.mr * 8;         // 8 microkernels stacked on M
  ts.Nc = ts.nr * 4;         // 4 microkernels stacked on N
  ts.Kc = 128;               // conservative reduction chunk (tune by arch)

  // We can refine later with arch.l2_size (VTCM-size ?).
  (void)arch;
  return ts;
}

/// Apply 2-level tiling to a linalg op with iterators [i (M) : parallel, k (K) : reduction, j (N) : parallel].
/// Produces: outer tiling (Mc,Kc,Nc) with interchange {0,2,1} (loops: i, j, k),
/// then, apply inner tiling (mr,0,nr) on the inner tiled op.
static LogicalResult
applyTileToGemm(RewriterBase &rewriter, Operation *transformOp, Operation *target,
                const mKInfo &mK, const ArchInfo &arch,
                SmallVector<Operation*, 6> &outResults) {

  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
  if (!tilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported");

  // Outer level: (Mc, Kc, Nc) over (i, k, j) with interchange {0,2,1} -> loops order (i, j, k).
  GemmTileSizes ts = computeGemmTiles(mK, arch);

  SmallVector<int64_t, 2> outerTileSz = {ts.Mc, ts.Kc};
  SmallVector<OpFoldResult> outerTileOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), outerTileSz);
  SmallVector<int64_t, 2> outerInterchange = {0, 1}; // keep (i,k,j) order

  scf::SCFTilingOptions outerOpts;
  outerOpts.setTileSizes(outerTileOfr).setInterchange(outerInterchange);
  outerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> outerRes = scf::tileUsingSCF(rewriter, tilingInterfaceOp, outerOpts);
  if (failed(outerRes))
    return transformOp->emitError("First level tiling for GEMM failed.");

  // Outer replace: use loop results when loops exist.
  if (!outerRes->loops.empty()) {
    // The scf.for returns the assembled tensor via yield -> results().
    rewriter.replaceOp(tilingInterfaceOp, outerRes->loops.front()->getResults());
  } else if (!outerRes->tiledOps.empty()) {
    // Degenerate case (no loops): fall back to the tiled op result.
    rewriter.replaceOp(tilingInterfaceOp, outerRes->tiledOps.front()->getResults());
  } else {
    // No tiled op produced: a safe fallback, just erase.
    rewriter.eraseOp(tilingInterfaceOp);
  }

  Operation *innerOp = outerRes->tiledOps.front();
  Location loc = innerOp->getLoc();
  int64_t mr = mK.nrows;
  int64_t nr = mK.ncols;

	  // Grab original A/B from the inner tiled ops
  Value A = innerOp->getOperand(0);

  // Try to hoist A's slice outside the innermost N-loop and decide IPs.
  Operation *afterA = nullptr;
  Operation *beforeA = innerOp;

  // Expect localResults[5] to be the innermost scf.for (as produced by tiling).
  if (outerRes->loops.size() > 2) {
    if (auto innerFor = dyn_cast_or_null<scf::ForOp>(outerRes->loops[1].getOperation())) {
      if (auto aSlice = getSliceProducerOrNull(A)) {
        if (isExtractSliceInvariantToLoop(aSlice, innerFor)) {
          // Move the slice that feeds A outside the inner loop.
          aSlice->moveBefore(innerFor);
          // Insert A_pack right AFTER this slice
          afterA = aSlice.getOperation();
        }
      }
    }
  }

  // Build A_pack at the chosen insertion point
  auto aPackOr = buildAPackAt(rewriter, loc, A, mr, afterA, beforeA);
  if (failed(aPackOr)) return failure();
  Value aPack = aPackOr->first;


  Operation *outer2Op = outerRes->tiledOps.front();

  auto outer2TilingInterfaceOp = dyn_cast<TilingInterface>(outer2Op);
  if (!outer2TilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported (outer2).");

  SmallVector<int64_t, 3> outer2TileSz = {0, 0, ts.Nc};
  SmallVector<OpFoldResult> outer2TileOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), outer2TileSz);

  // Keep outer2 order (i, j) — reduction K stays as-is (no extra tiling).
  SmallVector<int64_t, 3> outer2Interchange = {0, 1, 2};

  // auto outer2TilingInterfaceOp = dyn_cast<TilingInterface>(outer2Op);
  // if (!outer2TilingInterfaceOp)
  //   return transformOp->emitError("only TilingInterface ops are supported (outer2).");

  scf::SCFTilingOptions outer2Opts;
  outer2Opts.setTileSizes(outer2TileOfr).setInterchange(outer2Interchange);
  outer2Opts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  rewriter.setInsertionPoint(outer2Op);
  FailureOr<scf::SCFTilingResult> outer2Res =
      scf::tileUsingSCF(rewriter, outer2TilingInterfaceOp, outer2Opts);
  if (failed(outer2Res))
    return transformOp->emitError("Second level tiling for GEMM failed.");


  // Outer 2 replace: same rule as outer.
  if (!outer2Res->loops.empty()) {
    rewriter.replaceOp(outer2TilingInterfaceOp, outer2Res->loops.front()->getResults());
  } else if (!outer2Res->tiledOps.empty()) {
    rewriter.replaceOp(outer2TilingInterfaceOp, outer2Res->tiledOps.front()->getResults());
  } else {
    rewriter.eraseOp(outer2TilingInterfaceOp);
  }

  //
  // // Build B_pack right before ukernel
  // OpBuilder::InsertionGuard guard(rewriter);
  // rewriter.setInsertionPoint(ukernel);
  // auto bPackOr = buildBPackAt(rewriter, loc, B, nr);
  // if (failed(bPackOr)) return failure();
  // Value bPack = bPackOr->first;

  // Inner level: (mr, nr) on inner tiled op.
  // Operation *innerOp = outerRes->tiledOps.front();

  SmallVector<int64_t, 3> innerTileSz = {ts.mr, /*K*/ 0, ts.nr};
  SmallVector<OpFoldResult> innerTileOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), innerTileSz);

  // Keep inner order (i, j) — reduction K stays as-is (no extra tiling).
  SmallVector<int64_t, 2> innerInterchange = {0, 1};

  auto innerTilingInterfaceOp = dyn_cast<TilingInterface>(innerOp);
  if (!innerTilingInterfaceOp)
    return transformOp->emitError("only TilingInterface ops are supported (inner).");

  scf::SCFTilingOptions innerOpts;
  innerOpts.setTileSizes(innerTileOfr).setInterchange(innerInterchange);
  innerOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);

  // rewriter.setInsertionPoint(innerOp);
  // FailureOr<scf::SCFTilingResult> innerRes =
  //     scf::tileUsingSCF(rewriter, innerTilingInterfaceOp, innerOpts);
  // if (failed(innerRes))
  //   return transformOp->emitError("Second level tiling for GEMM failed.");

  // Inner replace: same rule as outer.
  // if (!innerRes->loops.empty()) {
  //   rewriter.replaceOp(innerTilingInterfaceOp, innerRes->loops.front()->getResults());
  // } else if (!innerRes->tiledOps.empty()) {
  //   rewriter.replaceOp(innerTilingInterfaceOp, innerRes->tiledOps.front()->getResults());
  // } else {
  //   rewriter.eraseOp(innerTilingInterfaceOp);
  // }

  // Report back the tiled inner op + all loops (outer + inner)
  // tiledOps.push_back(innerRes->tiledOps.front());
  tiledOps.push_back(outerRes->tiledOps.front());
  for (Operation *loop : outerRes->loops) loopOps.push_back(loop);
  // for (Operation *loop : innerRes->loops) loopOps.push_back(loop);
  //
  // Collect only real ops: [tiledInnerOp, loop...];
  outResults.clear();
  if (!tiledOps.empty() && tiledOps.front())
    outResults.push_back(tiledOps.front());
  for (Operation *L : loopOps)
    if (L) outResults.push_back(L);

  return success();
}

/// This function generalize a named linalg.matmul into a
/// canonical linalg.generic with A(i,k), B(k,j), C(i,j) and
/// iterator_types = ["parallel", "reduction", "parallel"].
/// Returns the created generic op or failure.
FailureOr<linalg::GenericOp>
generalizeMatmulToGeneric(RewriterBase &rewriter, linalg::MatmulOp matmul) {

  MLIRContext *ctx = rewriter.getContext();
  Location loc = matmul.getLoc();

  // Get DPS interfaces (inputs/inits).
  // Preconditions: ranks must be 2x2-> 2
  SmallVector<Value> inputs = matmul.getDpsInputs();
  ValueRange inits = matmul.getDpsInits();
  if (inputs.size() != 2 || inits.size() != 1)
    return failure();

  Value A = inputs[0];
  Value B = inputs[1];
  Value Cinit = inits[0];

  // Sanity-check types.
  auto aType = dyn_cast<ShapedType>(A.getType());
  auto bType = dyn_cast<ShapedType>(B.getType());
  auto cType = dyn_cast<ShapedType>(Cinit.getType());
  if (!aType || !bType || !cType || aType.getRank() != 2 ||
      bType.getRank() != 2 || cType.getRank() != 2)
    return failure();

  // Build indexing maps: A(i,k), B(k,j), C(i,j).
  AffineExpr i, k, j;
  bindDims(ctx, i, k, j);

  AffineMap aMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{i, k}, ctx);
  AffineMap bMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{k, j}, ctx);
  AffineMap cMap = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0,
                                  ArrayRef<AffineExpr>{i, j}, ctx);

  // Create the iterator types
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iters = {parallel, reduction, parallel};

  // Create the linalg.generic and inline the multiply-accumulate.
  rewriter.setInsertionPoint(matmul);
  auto generic = rewriter.create<linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/TypeRange{cType},
      /*inputs=*/ValueRange{A, B},
      /*outputs=*/ValueRange{Cinit},
      /*indexingMaps=*/ArrayRef<AffineMap>{aMap, bMap, cMap},
      /*iteratorTypes=*/iters,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange blockArgs) {
        // blockArgs: %a, %b, %acc
        Value mul = createMul(nestedLoc, blockArgs[0], blockArgs[1], nested);
        Value sum = createAdd(nestedLoc, mul, blockArgs[2], nested);
        nested.create<linalg::YieldOp>(nestedLoc, sum);
      });

  // Replace the named op with the generic result.
  rewriter.replaceOp(matmul, generic->getResults());
  return generic;
}

///
/// Implementation of SGemm::apply transform dialect operation.
///
DiagnosedSilenceableFailure
transform::SGemmOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {

  // Initialize the default values of mKInfo & ArchInfo.
  // It's dependent of the target machine.
  mKInfo mK = {8, 16, 128};
  ArchInfo arch = {
      (uint32_t)(32768),
      (uint32_t)(1048576),
      (uint32_t)(4194304)
  };

  // Get the optional arguments
  auto mKInfoAttr = getMKInfo();
  auto archInfoAttr = getArchInfo();

  // If `mKInfoAttr` was provided, use the given values
  if (mKInfoAttr) {
    SmallVector<int64_t, 4> mKValues;
    for (auto attr : mKInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        mKValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: mKInfoAttr contains non-integer values!\n";
      }
    }
    if (mKValues.size() >= 2) {
      mK.nrows = mKValues[0];
      mK.ncols = mKValues[1];
      mK.noutput = mK.nrows * mK.ncols;
    } else {
      return emitSilenceableError() << "Error: mKInfoAttr does not contain enough values!\n";
    }
  }

  // If `archInfoAttr` was provided, use the given values
  if (archInfoAttr) {
    SmallVector<int64_t, 3> archValues;
    for (auto attr : archInfoAttr->getValue()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        archValues.push_back(intAttr.getInt());
      } else {
        return emitSilenceableError() << "Error: archInfoAttr contains non-integer values!\n";
      }
    }
    if (archValues.size() >=3) {
      arch.l1_size = (uint32_t)(archValues[0]);
      arch.l2_size = (uint32_t)(archValues[1]);
      arch.l3_size = (uint32_t)(archValues[2]);
    } else {
      return emitSilenceableError() << "Error: archInfoAttr does not contain enough values!\n";
    }
  }

  // temporary variables to store all sgemm transformation
  SmallVector<Operation*> tempResultGemms;
  SmallVector<SmallVector<Operation*, 5>> tempResultLoops;

  // Get context and gemmOps
  MLIRContext *context = rewriter.getContext();
  auto targetOps = state.getPayloadOps(getTarget());

  for (Operation *t : targetOps) {
    if (auto mm = dyn_cast<linalg::MatmulOp>(t)) {
      auto guarded = transform::TransformRewriter::InsertionGuard(rewriter);
      rewriter.setInsertionPoint(mm);
      auto res = generalizeMatmulToGeneric(rewriter, mm);
      if (failed(res))
        return emitSilenceableError() << "failed to generalize linalg.matmul";

      linalg::GenericOp genericOp = *res;
      // Compute tile sizes once (we need them for padding multiples).
      GemmTileSizes ts = computeGemmTiles(mK, arch);

      // Padding scaffolding: extract A/B/C, set IP, and build tensor.pad if needed.
      auto mpOr = preparePaddingForGemmLike(rewriter, genericOp, ts);
      if (failed(mpOr)) {
        // Non-fatal: just proceed without padding for now.
      } else {
        const MaybePaddedABC &mp = *mpOr;

        // Rewire the GEMM to use padded A/B/C by cloning a new generic just before tiling.
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(genericOp);
          SmallVector<AffineMap> maps = llvm::to_vector(genericOp.getIndexingMapsArray());
          SmallVector<utils::IteratorType> iters = llvm::to_vector(genericOp.getIteratorTypesArray());

          Type outTy = mp.C.getType();
          auto cloned = rewriter.create<linalg::GenericOp>(
              genericOp.getLoc(),
              /*resultTensorTypes=*/TypeRange{outTy},
              /*inputs=*/ValueRange{mp.A, mp.B},
              /*outputs=*/ValueRange{mp.C},
              /*indexingMaps=*/maps,
              /*iteratorTypes=*/iters,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                // args = {Aelt, Belt, Celt}
                Value mul = createMul(loc, args[0], args[1], b);
                Value sum = createAdd(loc, mul, args[2], b);
                b.create<linalg::YieldOp>(loc, sum);
              });
          rewriter.replaceOp(genericOp, cloned->getResults());
          genericOp = cloned;
        }
      }

      SmallVector<Operation*, 6> localResults;

      // linalg::GenericOp genericOp = *res;
      if (failed(applyTileToGemm(rewriter, getOperation(), genericOp.getOperation(), mK, arch, localResults)))
        return emitSilenceableError() << "Failed to apply tiling.";

      // First result in localResults contains the tiled uKernel
      // if (!localResults.empty() && localResults[0]) {
      //   if (auto ukernel = dyn_cast<linalg::GenericOp>(localResults[0])) {
      //     if (failed(packAndRetargetUkernel(rewriter, ukernel, mK, localResults)))
      //       return emitSilenceableError() << "Failed to pack+retarget microkernel.";
      //   }
      // }

      // Store tiled uKernel (localResults[0])
      if (!localResults.empty() && localResults[0])
        tempResultGemms.push_back(localResults[0]);

      // Following, store the generated loops
      SmallVector<Operation*, 5> loopSet;
      for (size_t i = 1; i < localResults.size(); ++i)
        if (localResults[i]) loopSet.push_back(localResults[i]);
      tempResultLoops.push_back(loopSet);

    }
  } // For targetOps

  // Flatten tempResultLoops
  SmallVector<Operation*> flatResultLoops;
  for (const auto &loopSet : tempResultLoops) {
    for (Operation *L : loopSet)
      if (L) flatResultLoops.push_back(L);
  }

  // Also filter nulls from tempResultGemms, just in case.
  {
    SmallVector<Operation*> filtered;
    filtered.reserve(tempResultGemms.size());
    for (Operation *op : tempResultGemms)
      if (op) filtered.push_back(op);
    tempResultGemms.swap(filtered);
  }

  // Store results properly in TransformResults
  results.set(getOperation()->getOpResult(0), tempResultGemms);
  results.set(getOperation()->getOpResult(1), flatResultLoops);

  return DiagnosedSilenceableFailure::success();

}

void transform::SGemmOp::getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult
transform::SGemmOp::verify() {
  // All necessary checks are done in the Apply
  return success();
}

void registerSGemm(mlir::DialectRegistry &registry) {
  registry.addExtensions<SGemm>();
}
