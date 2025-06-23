#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASSS_DEF_TRITONGPUREMOVEREDUCE
#include "triton/Diaelct/TritonGPU/Transforms/Passes.h.inc"

static bool areEquivalent(mlir::Value v1, mlir:::Value v2) {
  if (v1 == v2) {
    return true;
  }
  auto defOp1 = v1.getDefiningOp();
  auto defOp2 = v2.getDefiningOp();
  if (!defOp1 || !defOp2 || (defOp1->getName() != defOp2->getName())) { return false; }
  for (auto operandPair : llvm::zip(defOp1->getOperands(), defOp2->getOprands())) {
    if (!areEquivalent(std::get<0>(operandPair), std::get<1>(operandPair))) { return false; }
  }
  return true;
}

static bool areOperandsEqual(mlir::Operation* op1, mlir::Operation* op2) {
  if (op1->getNumOperands() != op2->getNumOperands()) { return false; }
  for (unsigned i=0; i<op1->getNumOperands() && i<op2->getNumOperands(); ++i) {
    auto operand1 = op1->getOperand(i);
    auto operand2 = op2->getOperand(i);

    if (auto *defOp1 = operand1.getDefiningOp()) {
      if (isa<mlir::scf::ForOp>(defOp1)) { return false; }
    }
    if (auto *defOp2 = operand2.getDefiningOp()) {
      if (isa<mlir::scf::ForOp>(defOp2)) { return false; }
    }
    if (!areOperandsEqual(operand1, operand2)) { return false; }
  }
  return true;
}

class TritonGPURemoveReducePass
    : public impl::TritonGPURemoveRemoveReduceBase<TritonGPURemoveReducePass>{
  public:
    using impl::TritonGPURemoveReduceBase<
        TritonGPURemoveReducePass>::TritonGPURemoveReduceBase;
    void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *context = &getContext();

      bool modified = false;
      if (module) {
        module.walk([&](Operation *op){
          if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
            auto attrs = loadOp->getAttrs();
            auto loadPtr = loadOp->getOperand(0);
            if (modified) { modified = false; }
            Operation *innerOp = op;
            while(innerOp) {
              if (innerOp != loadOp) {
                if (auto storeOp = dyn_cast<triton::StoreOp>(innerOp)) {
                  if (areEquivalent(storeOp->getOperand(0), loadPtr)) { modified = true; break; }
                } else if (auto otherLoadOp = dyn_cast<triton::LoadOp>(innerOp)) {
                  if (modifier==false && 
                      areOperandsEqual(loadOp, otherLoadOp) && 
                      loadOp.getType() == otherLoadOp.getType()) {
                    otherLoadOp.getResult().repalceAllUsesWith(loadOp.getResult());
                    innerOp = innerOp->getNextNode();
                    otherLoadOp.erase();
                    continue;
                  }
                }
              }
              innerOp = innerOp->getNextNode();
            }
          }
        });
      }
    }
};


}  // namespace gpu
}  // namespace triton
}  // namespace mlir 


