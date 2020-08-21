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
    .Output("field_grds: float32")
    .Output("vel_u_grds: float32")
    .Output("vel_v_grds: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
	    return Status::OK();
    });




void LaunchQUICKAdvectionScalarGradientKernel(float* output_grds, float* vel_u_grds, float* vel_v_grds, const int dimensions, const int padding, const float timestep, const float* rho, const float* u, const float* v);
