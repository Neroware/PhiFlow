//#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;


REGISTER_OP("QuickAdvection")
    .Input("field: float32")
    .Input("vel_u_field: float32")
    .Input("vel_v_field: float32")
    .Attr("dimensions: int")
    .Attr("timestep: float")
    .Attr("field_type: int")
    .Attr("step_type: int")
    .Output("advected_field: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
	    return Status::OK();
    });


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const float timestep, const float* rho, const float* u, const float* v);


class QuickAdvectionOp : public OpKernel {
private:
    int dimensions;
    float timestep;
    int field_type;
    int step_type;
    

public:
    explicit QuickAdvectionOp(OpKernelConstruction* context) : OpKernel(context) {
        //printf("QUICK Debug Message: I'm alive!\n");
        context->GetAttr("dimensions", &dimensions);
        context->GetAttr("timestep", &timestep);
        context->GetAttr("field_type", &field_type);
        context->GetAttr("step_type", &step_type);
    }


    void Compute(OpKernelContext* context) override{
        //printf("QUICK: Launching Kernel...\n");

        const Tensor& input_field = context->input(0);
        const Tensor& input_vel_u = context->input(1);
        const Tensor& input_vel_v = context->input(2);

        Tensor* output_field = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_field.shape(), &output_field));
        auto output_flat = output_field->flat<float>();

        auto field = input_field.flat<float>();
        auto u = input_vel_u.flat<float>();
        auto v = input_vel_v.flat<float>();

        if(field_type == 0){
            //printf("Field set to 'density'\n");
            LaunchQuickDensityKernel(output_flat.data(), dimensions, timestep, field.data(), u.data(), v.data());
        }

    }
};


REGISTER_KERNEL_BUILDER(Name("QuickAdvection").Device(DEVICE_CPU), QuickAdvectionOp);

