#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#define GET_PASS_CLASSES
using namespace llvm;
using namespace mlir;
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transfroms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using llvm::ArrayRef;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUARITHMETICTRANSFORMATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct DivtoMulConvert : public OpRewritePattern<arith::DivFOp> {
    using OpRewritePattern<arith::DivFOp>::OpRewritePattern;

    LogicalResult matchandRewrite(arith::DivFOp op, PatternRewriter &rewriter) const override {
      bool changed = false;
      // Match whether the Op is DivFOp
      Value a = op.getResult();
      Value b = op.getLhs();
      Value c = op.getRhs();
      auto loc = op.getLoc();
      // Obatin the iterator of Op owner then obtain it through getOwner.
      if (a.hasOneUse()) {
        // Whether the Op User has only one and is divOp.
        for (auto &use : a.getUses()) {
          if (auto divUse = dyn_cast<arith::DivFOp>(use.getOwner())) {
            auto originalInsertionPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(divUse);
            auto loc_1 = divUse.getLoc();
            arith::MulFOp product;
            arith::DivFOp ResultEnd;
            if (divUse.getLhs() == op.getResult()) {
              product = rewriter.create<arith::MulFOp>(loc_1, c, divUse.getRhs());
              rewriter.setInsertionPointAfter(product);
              ResultEnd = rewriter.create<arith::DivFOp>(loc_1, b, product.gerResult());
            } else if (divUse.getRhs() == op.getResult()) {
              product = rewriter.create<arith::MulFOp>(loc_1, c, divUse.gerLhs());
              rewriter.setInsertionPointAfter(product);
              ResultEnd = rewriter.create<arith::DivFOp>(loc_1, product.getResult(), b);
            } else {
              continue;
            }
            rewriter.restoreInsertionPoint(originalInsertionPoint);
            rewriter.replaceOp(op, product.getResult());
            divUse.replaceAllUsesWith(ResultEnd.getResult());
            rewriter.eraseOp(divUse);
            changed = true;
          }
        }
      }
      return changed ? success() : failure();
    }

};

class TritonGPUArithmeticTransformationPass
    : public impl::TritonGPUArithmeticTransformationBase<TritonGPUArithmeticTransformationPass> {
public:
  using impl::TritonGPUArithmeticTransformationBase<
        TritonGPUArithmeticTransformationPass>::TritonGPUArithmeticTransformationBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<DivtoMulConvert>(context);
    if (applyPatternsAndFoldGreedily(module, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};


} // namespace gpu
} // namespace triton
} // namespace mlir

