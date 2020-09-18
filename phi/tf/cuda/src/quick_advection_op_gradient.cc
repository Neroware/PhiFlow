#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;


REGISTER_OP("QuickAdvectionGradient")
    .Input("field: float32")
    .Input("field_padded: float32")
    .Input("vel_u_field: float32")
    .Input("vel_v_field: float32")
    .Input("grad: float32")
    .Attr("dim_x: int")
    .Attr("dim_y: int")
    .Attr("delta_x: float")
    .Attr("delta_y: float")
    .Attr("padding: int")
    .Attr("timestep: float")
    .Output("field_grds: float32")
    .Output("vel_u_grds: float32")
    .Output("vel_v_grds: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        c->set_output(2, c->input(3));
	    return Status::OK();
    });




void LaunchQUICKAdvectionScalarGradientKernel(
    float* output_grds, 
    float* vel_u_grds, 
    float* vel_v_grds, 
    const int dim_x, 
    const int dim_y, 
    const float delta_x, 
    const float delta_y,
    const int padding, 
    const float timestep, 
    const float* rho, 
    const float* u, 
    const float* v, 
    const float* grad
);




class QuickAdvectionOpGradient : public OpKernel {
private:
    int dim_x, dim_y;
    float delta_x, delta_y;
    float timestep;
    int padding;
    

public:
    explicit QuickAdvectionOpGradient(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("dim_x", &dim_x);
        context->GetAttr("dim_y", &dim_y);
        context->GetAttr("delta_x", &delta_x);
        context->GetAttr("delta_y", &delta_y);
        context->GetAttr("padding", &padding);
        context->GetAttr("timestep", &timestep);
    }


    void Compute(OpKernelContext* context) override{
        const Tensor& input_field = context->input(0);
        const Tensor& input_field_padded = context->input(1);
        const Tensor& input_vel_u = context->input(2);
        const Tensor& input_vel_v = context->input(3);
        const Tensor& input_grad = context->input(4);

        Tensor* field_grds = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_field_padded.shape(), &field_grds));
        auto field_grds_flat = field_grds->flat<float>();

        Tensor* vel_u_grds = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(1, input_vel_u.shape(), &vel_u_grds));
        auto vel_u_grds_flat = vel_u_grds->flat<float>();

        Tensor* vel_v_grds = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(2, input_vel_v.shape(), &vel_v_grds));
        auto vel_v_grds_flat = vel_v_grds->flat<float>();

        auto field = input_field_padded.flat<float>();
        auto u = input_vel_u.flat<float>();
        auto v = input_vel_v.flat<float>();
        auto grad = input_grad.flat<float>();

        LaunchQUICKAdvectionScalarGradientKernel(field_grds_flat.data(), vel_u_grds_flat.data(), vel_v_grds_flat.data(), dim_x, dim_y, delta_x, delta_y, padding, timestep, field.data(), u.data(), v.data(), grad.data());
    }
};


REGISTER_KERNEL_BUILDER(Name("QuickAdvectionGradient").Device(DEVICE_GPU), QuickAdvectionOpGradient);
