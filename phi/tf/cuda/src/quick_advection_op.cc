#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;


REGISTER_OP("QuickAdvection")
.Input("field: float32")
.Input("field_padded: float32")
.Input("vel_u_field: float32")
.Input("vel_v_field: float32")
.Attr("dimensions: int")
.Attr("padding: int")
.Attr("timestep: float")
.Attr("field_type: int")
.Attr("step_type: int")
.Output("advected_field: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const int padding, const float timestep, const float* rho, const float* u, const float* v);
//void LaunchQuickVelocityXKernel(float* output_field, const int dimensions, const float timestep, const float* u, const float* v);
//void LaunchQuickVelocityYKernel(float* output_field, const int dimensions, const float timestep, const float* u, const float* v);


class QuickAdvectionOp : public OpKernel {
private:
    int dimensions;
    float timestep;
    int field_type;
    int step_type;
    int padding;


public:
    explicit QuickAdvectionOp(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("dimensions", &dimensions);
        context->GetAttr("padding", &padding);
        context->GetAttr("timestep", &timestep);
        context->GetAttr("field_type", &field_type);
        context->GetAttr("step_type", &step_type);
    }


    void Compute(OpKernelContext* context) override {
        const Tensor& input_field = context->input(0);
        const Tensor& input_field_padded = context->input(1);
        const Tensor& input_vel_u_padded = context->input(2);
        const Tensor& input_vel_v_padded = context->input(3);

        Tensor* output_field = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_field.shape(), &output_field));
        auto output_flat = output_field->flat<float>();

        auto field = input_field_padded.flat<float>();
        auto u = input_vel_u_padded.flat<float>();
        auto v = input_vel_v_padded.flat<float>();

        switch (field_type) {
        case 0: LaunchQuickDensityKernel(output_flat.data(), dimensions, padding, timestep, field.data(), u.data(), v.data());
            break;
        /*case 1: LaunchQuickVelocityXKernel(output_flat.data(), dimensions, timestep, u.data(), v.data());
            break;
        case 2: LaunchQuickVelocityYKernel(output_flat.data(), dimensions, timestep, u.data(), v.data());
            break;*/
        default:
            break;
        }


    }
};