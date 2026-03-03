import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv3d2d
from theano import tensor
from theano.tensor.signal.downsample import DownsampleFactorMax


    
def shared_dataset(data_x, data_y, borrow=True):

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')


def relu(X):
    """Rectified linear units (relu)"""
    return T.maximum(0,X)

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class Conv_3d_Layer(object):

    def __init__(self, rng, input, filter_shape, image_shape, W=None, b=None):
        
        #signals_shape = (batchsize, in_time, in_channels, in_height, in_width)
        #filters_shape = (flt_channels, flt_time, in_channels, flt_height, flt_width)
        self.input=input
        assert image_shape[2] == filter_shape[2]
        
        
        # initialize weights with random weights
        fan_in = numpy.prod(filter_shape[2:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[3:]))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
            self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True, name='W')
        else:
            self.W = W

        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name='b')
        else:
            self.b = b

        conv_out5D = conv3d2d.conv3d(signals=input, filters=self.W,
                signals_shape=image_shape, filters_shape=filter_shape)
        # activation
        #out_4D = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = relu(conv_out5D + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class Dropout_Conv_3d_Layer(Conv_3d_Layer):
    def __init__(self, rng, input, filter_shape, image_shape, dropout_rate=0.5, W=None, b=None):
        super(Dropout_Conv_3d_Layer, self).__init__(
                rng=rng, input=input, filter_shape=filter_shape, image_shape=image_shape, W=W, b=b)
        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        # self.node_value = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.score = T.dot(input, self.W) + self.b

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]


        # end-snippet-2

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2
    def class_score(self, y):
        return (self.score)[0,y]

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
    
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias=True, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)



def max_pool_3d(input, ds, ignore_border=False):
    """
    Takes as input a N-D tensor, where N >= 3. It downscales the input video by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1],ds[2]) (time, height, width)
    :type input: N-D theano tensor of input images.
    :param input: input images. Max pooling will be done over the 3 last dimensions.
    :type ds: tuple of length 3
    :param ds: factor by which to downscale. (2,2,2) will halve the video in each dimension.
    :param ignore_border: boolean value. When True, (5,5,5) input with ds=(2,2,2) will generate a
      (2,2,2) output. (3,3,3) otherwise.
    """

    if input.ndim < 3:
        raise NotImplementedError('max_pool_3d requires a dimension >= 3')

    # extract nr dimensions
    vid_dim = input.ndim
    # max pool in two different steps, so we can use the 2d implementation of 
    # downsamplefactormax. First maxpool frames as usual. 
    # Then maxpool the time dimension. Shift the time dimension to the third 
    # position, so rows and cols are in the back

    # extract dimensions
    frame_shape = input.shape[-2:]
    
    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input.shape[:-2])
    batch_size = tensor.shape_padright(batch_size,1)
    
    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1,]), 
                                        frame_shape), 'int32')
    input_4D = tensor.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of videos in rows and cols
    op = DownsampleFactorMax((ds[1],ds[2]), ignore_border)
    output = op(input_4D)
    # restore to original shape
    outshape = tensor.join(0, input.shape[:-2], output.shape[-2:])
    out = tensor.reshape(output, outshape, ndim=input.ndim)

    # now maxpool time

    # output (time, rows, cols), reshape so that time is in the back
    shufl = (list(range(vid_dim-3)) + [vid_dim-2]+[vid_dim-1]+[vid_dim-3])
    input_time = out.dimshuffle(shufl)
    # reset dimensions
    vid_shape = input_time.shape[-2:]
    
    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input_time.shape[:-2])
    batch_size = tensor.shape_padright(batch_size,1)
    
    # store as 4D tensor with shape: (batch_size,1,width,time)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1,]), 
                                        vid_shape), 'int32')
    input_4D_time = tensor.reshape(input_time, new_shape, ndim=4)
    # downsample mini-batch of videos in time
    op = DownsampleFactorMax((1,ds[0]), ignore_border)
    outtime = op(input_4D_time)
    # output 
    # restore to original shape (xxx, rows, cols, time)
    outshape = tensor.join(0, input_time.shape[:-2], outtime.shape[-2:])
    shufl = (list(range(vid_dim-3)) + [vid_dim-1]+[vid_dim-3]+[vid_dim-2])
    return tensor.reshape(outtime, outshape, ndim=input.ndim).dimshuffle(shufl)



class PoolLayer3D(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method=="max":
            out = max_pool_3d(input,pool_shape)
        else:
            raise NotImplementedError()

        self.output = out



