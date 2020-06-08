#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;


REGISTER_OP("QuickAdvection")
    .Input("testin: int32")
    .Output("testout: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
	    return Status::OK();
    });


void LaunchQuickKernel(int* testin);


class QuickAdvectionOp : public OpKernel {
public:
    explicit QuickAdvectionOp(OpKernelConstruction* context) : OpKernel(context) {
        printf("QUICK Debug Message: I'm alive!\n");
    }


    void Compute(OpKernelContext* context) override {
	// Grab Input Tensor
        const Tensor& input_tensor = context->input(0);
	auto input = input_tensor.flat<int32>();
	    
	// Create Output Tensor
	Tensor* output_tensor = NULL;
	OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

	auto output_flat = output_tensor->flat<int32>();

	// Set all but the first element of the output tensor to 0
	const int N = input.size();
	for(int i = 1; i < N; i++){
            output_flat(i) = 0;
	}
	
	// Preserve the first input value if possible.
	if (N > 0) output_flat(0) = input(0);
    }
};


REGISTER_KERNEL_BUILDER(Name("QuickAdvection").Device(DEVICE_CPU), QuickAdvectionOp);

