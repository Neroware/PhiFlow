#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;


REGISTER_OP("QuickAdvection")
    .Input("testin: int32")
    .Output("testout: int32");


void LaunchQuickKernel(const int* testin);


class QuickAdvectionOP : public OpKernel {
public:
    explicit PressureSolveOp(OpKernelConstruction* context) : OpKernel(context) {
        
    }


    void Compute(OpKernelContext* context) override {
        // General
        const Tensor& testin = context->input(0);
        auto testin_flat = testin.flat<int32>();

        context->set_output(0, testin);

        LaunchQuickKernel(testin_flat.data());
    }
}