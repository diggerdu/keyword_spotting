import mxnet as mx
import numpy as np


######compose symbolic######
data = mx.symbol.Variable('data')
fc1_weight = mx.symbol.Variable("fc1_weight")
fc1_bias = mx.symbol.Variable("fc1_bias")
fc1 = mx.symbol.FullyConnected(data = data, name = "fc1", num_hidden = 1, weight = fc1_weight, bias = fc1_bias)

out_label = mx.symbol.Variable("out_label")
#out = mx.symbol.SoftmaxOutput(data = fc1, name = "out", label = out_label)
out = mx.symbol.LogisticRegressionOutput(data = fc1, name = "out", label = out_label)
#####multi outputs#######
group = mx.symbol.Group([fc1, out])
print group.list_arguments()
print group.list_outputs()

input_shapes = {"data":(1, 2), "out_label":(1,)}
test_executor = out.simple_bind(ctx = mx.cpu(), grad_req = "write", **input_shapes)



test_executor.arg_dict["data"][:] = mx.nd.array([[0,1]])
test_executor.arg_dict["fc1_weight"][:] = mx.nd.array([[1,0]])
test_executor.arg_dict["fc1_bias"][:] = mx.nd.array([0])
test_executor.arg_dict["out_label"][:] = mx.nd.array([100])
test_executor.forward()


test_executor.backward(test_executor.arg_dict["fc1_weight"])

print test_executor.grad_arrays[0].asnumpy()
print test_executor.outputs[0].asnumpy()
