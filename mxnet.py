import mxnet as mx

net = mx.symbol.Variable('data')   #free variable or input data
net =mx.symbol.FullyConnected(data = net, name = 'fc1', num_hidden = 128)
net = mx.symbol.Activation(data = net, name = 'relu1', act_type = "relu")
net = mx.symbol.FullyConnected(data = net, name = 'fc2', numhidden = 128)
net = mx.symbol.Activation(data = net, name = 'relu2', act_type = "relu")
net = mx.symbol.FullyConnected(data = net, name = 'fc3', numhidden = 128)
net = mx.symbol.SoftmaxOutput(data = net, name = 'out')


