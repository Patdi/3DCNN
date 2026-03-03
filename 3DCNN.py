import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from layers import *
import random
from data_utils import *
from theano.misc.pkl_utils import dump
import argparse


class ConvDropNet(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.

    """
    def __init__(self,
            rng,
            input,
            in_channels,
            dropout_rates,
            batch_size,
            use_bias=True):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        #weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        
        #######
        filter_w=3
        num_3d_pixel=20

        layer0_w = num_3d_pixel 
        layer0_h = num_3d_pixel
        layer0_d = num_3d_pixel

        layer1_w = (layer0_w-3+1) #14
        layer1_h = (layer0_h-3+1)
        layer1_d = (layer0_d-3+1)

        layer2_w = (layer1_w-3+1)/2 #14
        layer2_h = (layer1_h-3+1)/2
        layer2_d = (layer1_d-3+1)/2

        layer3_w = (layer2_w-3+1)/2
        layer3_h = (layer2_h-3+1)/2
        layer3_d = (layer2_d-3+1)/2
        
        
        ######################
        # BUILD ACTUAL MODEL #
        ######################
       
        print '... building the model'
        # image sizes
        batchsize     = batch_size
        in_time       = num_3d_pixel
        in_width      = num_3d_pixel
        in_height     = num_3d_pixel
        #filter sizes
        flt_channels  = 100
        flt_time      = filter_w
        flt_width     = filter_w
        flt_height    = filter_w

        signals_shape0 = (batchsize, in_time, in_channels, in_height, in_width)
        filters_shape0 = (flt_channels, 3, in_channels, 3, 3)
        signals_shape1 = (batchsize, layer1_d, flt_channels, layer1_h, layer1_w)
        filters_shape1 = (flt_channels*2, 3, flt_channels, 3, 3)
        signals_shape2 = (batchsize, layer2_d, flt_channels*2, layer2_h, layer2_w)
        filters_shape2 = (flt_channels*4, 3, flt_channels*2, 3, 3)
        
        
        #next_layer_input = input
        #first_layer = True
        # dropout the input

        layer0_input = input.reshape(signals_shape0) #20
        # layer0_dropout_input = _dropout_from_layer(rng, layer0_input, p=dropout_rates[0])


        # W0: flt_channels, flt_time, in_channels, flt_height, flt_width

        ###################################
        Dropout_layer0 = Dropout_Conv_3d_Layer(rng=rng,
                    input=layer0_input,
                    image_shape=signals_shape0,
                    filter_shape=filters_shape0,                    
                    dropout_rate=dropout_rates[0])

        self.dropout_layers.append(Dropout_layer0)
        next_dropout_input = Dropout_layer0.output

        layer0 = Conv_3d_Layer(rng=rng, input=layer0_input, #18
                image_shape=signals_shape0,
                filter_shape=filters_shape0, 
                W=Dropout_layer0.W * (1 - dropout_rates[0]),
                b=Dropout_layer0.b * (1 - dropout_rates[0]),
                )

        self.layers.append(layer0)
        next_layer_input = layer0.output

        ##################################
        ###################################
        Dropout_layer1 = Dropout_Conv_3d_Layer(rng=rng,
                    input=next_dropout_input,
                    image_shape=signals_shape1,
                    filter_shape=filters_shape1,                    
                    dropout_rate=dropout_rates[1])

        Dropout_layer1_pool = PoolLayer3D(input=Dropout_layer1.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

        self.dropout_layers.append(Dropout_layer1)
        # self.dropout_layers.append(Dropout_layer1_pool)
        next_dropout_input = Dropout_layer1_pool.output.dimshuffle(0,2,1,3,4)

        layer1 = Conv_3d_Layer(rng=rng, 
                input=next_layer_input, # N*4*12*12*12 => N*7*10*10*10 
                image_shape=signals_shape1,
                filter_shape=filters_shape1,
                W=Dropout_layer1.W * (1 - dropout_rates[1]),
                b=Dropout_layer1.b * (1 - dropout_rates[1]),
                )

        layer1_pool = PoolLayer3D(input=layer1.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

        self.layers.append(layer1)
        # self.layers.append(layer1_pool)
        next_layer_input = layer1_pool.output.dimshuffle(0,2,1,3,4)

        ##################################
        ###################################

        Dropout_layer2 = Dropout_Conv_3d_Layer(rng=rng,
                    input=next_dropout_input,
                    image_shape=signals_shape2,
                    filter_shape=filters_shape2,                    
                    dropout_rate=dropout_rates[2])

        Dropout_layer2_pool = PoolLayer3D(input=Dropout_layer2.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

        self.dropout_layers.append(Dropout_layer2)
        # self.dropout_layers.append(Dropout_layer2_pool)
        next_dropout_input = Dropout_layer2_pool.output.dimshuffle(0,2,1,3,4).flatten(2)

        layer2 = Conv_3d_Layer(rng=rng, 
                input=next_layer_input, # N*4*12*12*12 => N*7*10*10*10 
                image_shape=signals_shape2,
                filter_shape=filters_shape2,
                W=Dropout_layer2.W * (1 - dropout_rates[2]),
                b=Dropout_layer2.b * (1 - dropout_rates[2]),
                )

        layer2_pool = PoolLayer3D(input=layer2.output.dimshuffle(0,2,1,3,4), pool_shape=(2,2,2)) #4

        self.layers.append(layer2)
        # self.layers.append(layer2_pool)
        next_layer_input = layer2_pool.output.dimshuffle(0,2,1,3,4).flatten(2)

        ##################################

        # W4: 200*layer4_w*layer4_h, 500

        Dropout_layer3 = DropoutHiddenLayer(rng=rng,
                input=next_dropout_input,
                activation=relu,
                n_in=(flt_channels*4*layer3_d*layer3_w*layer3_h), 
                n_out=1000, 
                dropout_rate=dropout_rates[3])
        self.dropout_layers.append(Dropout_layer3)
        next_dropout_input = Dropout_layer3.output

        # Reuse the paramters from the dropout layer here, in a different
        # path through the graph.
        layer3 = HiddenLayer(rng=rng,
                input=next_layer_input,
                activation=relu,
                # scale the weight matrix W with (1-p)
                W=Dropout_layer3.W * (1 - dropout_rates[3]),
                b=Dropout_layer3.b * (1 - dropout_rates[3]),
                n_in=(flt_channels*4*layer3_d*layer3_w*layer3_h), 
                n_out=1000,
                )

        self.layers.append(layer3)
        next_layer_input = layer3.output

        ##################################

        #layer4_input = layer2.output.flatten(2) # N*200

        # W4: 200*layer4_w*layer4_h, 500

        Dropout_layer4 = DropoutHiddenLayer(rng=rng,
                input=next_dropout_input,
                activation=relu,
                n_in=1000, 
                n_out=100, 
                dropout_rate=dropout_rates[4])
        self.dropout_layers.append(Dropout_layer4)
        next_dropout_input = Dropout_layer4.output

        # Reuse the paramters from the dropout layer here, in a different
        # path through the graph.
        layer4 = HiddenLayer(rng=rng,
                input=next_layer_input,
                activation=relu,
                # scale the weight matrix W with (1-p)
                W=Dropout_layer4.W * (1 - dropout_rates[4]),
                b=Dropout_layer4.b * (1 - dropout_rates[4]),
                n_in=1000, 
                n_out=100,
                )

        self.layers.append(layer4)
        next_layer_input = layer4.output

        ##################### TODO #######################

        Dropout_layer5 = LogisticRegression(
                input=next_dropout_input,
                n_in=100, n_out=20)
        self.dropout_layers.append(Dropout_layer5)

        # Again, reuse paramters in the dropout output.
        layer5 = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=Dropout_layer5.W,
            b=Dropout_layer5.b,
            n_in=100, n_out=20)
        self.layers.append(layer5)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.L2_sqr = T.sum(Dropout_layer0.W ** 2)+T.sum(Dropout_layer1.W**2)+T.sum(Dropout_layer2.W**2)+T.sum(Dropout_layer3.W**2)+T.sum(Dropout_layer4.W**2)+T.sum(Dropout_layer5.W**2) 

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def train_3DCNN(learning_rate=0.002, n_epochs=10, batch_size=20, filter_w=3, reg=5e-6, dropout=True, dropout_rates=[0.3,0.3,0.3,0.3,0.3]):
   
    rng = numpy.random.RandomState(23455)
    [all_examples, all_labels, all_train_sizes, test_size, val_size]=load_ATOM_BOX()

    Xtr=all_examples[0]
    Xt=all_examples[1]
    Xv=all_examples[2]

    ytr=all_labels[0]
    yt=all_labels[1]
    yv=all_labels[2]

    test_set_x, test_set_y = shared_dataset(Xt, yt)
    valid_set_x, valid_set_y = shared_dataset(Xv, yv)
    
    test_set_x=test_set_x.dimshuffle(0,4,1,2,3)
    valid_set_x=valid_set_x.dimshuffle(0,4,1,2,3)

    n_train_batches = [ a/batch_size for a in all_train_sizes]

    n_valid_batches = val_size
    n_test_batches = test_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    
    index = T.lscalar()  # index to a [mini]batch
    dtensor5 = T.TensorType('float32', (False,)*5)
    x = dtensor5('x')
    y = T.ivector('y')  


    classifier = ConvDropNet(rng=rng, input=x, in_channels=4, batch_size=batch_size, dropout_rates=[0.3,0.3,0.3,0.3,0.3])

    L2_sqr = classifier.L2_sqr
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)+reg*L2_sqr
               
    print '... building the model'
    
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
   
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
   
    output = dropout_cost if dropout else cost
    grads = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(output, param)
        grads.append(gparam)
    
    updates = []
    for param_i, grad_i in zip(classifier.params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([x,y], cost, updates=updates)


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'    
        
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches[0], patience / 2)
    
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    
    start_time = time.clock()

    cost_ij = 0
    epoch = 0
    done_looping = False
    iter = 0
    startc = time.clock()
    cost_hisotry=[]
    train_history=[]
    valid_history=[]


    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for part_index in range (6):
            for minibatch_index in range(n_train_batches[part_index]):
                X_train=Xtr[part_index][minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                y_train=ytr[part_index][minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                train_set_x, train_set_y = shared_dataset(X_train,y_train)
                train_set_x = train_set_x.dimshuffle(0,4,1,2,3)

                iter = (epoch - 1) * n_train_batches[part_index] + minibatch_index

                cost_ij = train_model(train_set_x.eval(), train_set_y.eval())
                cost_hisotry.append(cost_ij)
                if (iter + 1) % validation_frequency == 0:
                    list_file= open('../progress_3DCNN.txt','a')
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    valid_history.append(100*(1-this_validation_loss))
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches[part_index],
                           this_validation_loss * 100.))
                    list_file.write('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches[part_index],
                           this_validation_loss * 100.))
                    list_file.write('\n')
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches[part_index],
                               test_score * 100.))
                        list_file.write(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches[part_index],
                               test_score * 100.))
                        list_file.write('\n')
                        
                    list_file.close()
        list_file= open('../progress_3DCNN.txt','a')
        list_file.write('getting weights from classifier ...'+'\n')
        dump_weights_pickle(classifier)

    end_time = time.clock()
    dump_weights_pickle(classifier)
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    

def dump_weights_pickle(classifier,file_name='../weights/weight_3DCNN.zip'):
    W0 = classifier.params[0]
    W1 = classifier.params[2]
    W2 = classifier.params[4]
    W3 = classifier.params[6]
    W4 = classifier.params[8]
    W5 = classifier.params[10]

    b0 = classifier.params[1]
    b1 = classifier.params[3]
    b2 = classifier.params[5]
    b3 = classifier.params[7]
    b4 = classifier.params[9]
    b5 = classifier.params[11]

    with open(file_name, 'wb') as f:
        dump((W0, W1, W2, W3, W4, W5, b0, b1, b2, b3, b4, b5), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", help="epoch")
    args = parser.parse_args()
    num_epoch = int(args.epoch)
   
    train_3DCNN(learning_rate=0.002, n_epochs=num_epoch, batch_size=20, filter_w=3, reg=5e-6)
    

   
