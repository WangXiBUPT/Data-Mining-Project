from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Leaky_Relu, ELU
from loss import EuclideanLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 1000, 0.01))
#model.add(ELU('elu1', 0.5))
#model.add(Leaky_Relu('lrelu1', 0.1))
#model.add(Relu('relu1'))
model.add(Sigmoid('sigmoid1'))
model.add(Linear('fc2', 1000, 100, 0.01))
#model.add(Relu('relu2'))
model.add(Sigmoid('sigmoid2'))
#model.add(Leaky_Relu('lrelu2', 0.1))
model.add(Linear('fc3', 100, 10, 0.01))
#model.add(Sigmoid('sigmoid2'))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 1e-2,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 300,
    'disp_freq': 50,
    'test_epoch': 200
}

loss_iters = []
acc_iters = []

train_loss = []
train_acc = []
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], train_loss, train_acc)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], loss_iters, acc_iters)

print(loss_iters)
print(acc_iters)
print(train_loss)
print(train_acc)
