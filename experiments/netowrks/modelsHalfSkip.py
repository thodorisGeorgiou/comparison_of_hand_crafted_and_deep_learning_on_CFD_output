#old version, update according to modelsFullSkip.py before usage
import numpy
import tensorflow as tf
import cliffordConvolution as cc

def conv(inpt, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
	c_i = inpt.get_shape()[-1]
	assert c_i%group==0
	assert c_o%group==0
	convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
	if group==1:
		conv = convolve(inpt, kernel)
	else:
		input_groups = tf.split(inpt, num_or_size_splits=group, axis=3)
		kernel_groups = tf.split(kernel, num_or_size_splits=group, axis=3)
		output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
		conv = tf.concat(output_groups, 3)
	return  tf.nn.bias_add(conv, biases)


def unPool2D(x, height_factor, width_factor):
	output = repeat_elements(output, height_factor, axis=1)
	output = repeat_elements(output, width_factor, axis=2)
	return output

def repeat_elements(x, rep, axis):
	x_shape = x.get_shape().as_list()
	if x_shape[axis] is None:
		raise ValueError('Axis ' + str(axis) + ' of input tensor '
										 'should have a defined dimension, but is None. '
										 'Full tensor shape: ' + str(tuple(x_shape)) + '. '
										 'Typically you need to pass a fully-defined '
										 '`input_shape` argument to your first layer.')
	# slices along the repeat axis
	splits = tf.split(x, num_or_size_splits=x_shape[axis], axis=axis)
	# repeat each slice the given number of reps
	x_rep = [s for s in splits for _ in range(rep)]
	return tf.concat(x_rep, axis)

def ccLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize=True, wd=0.0001):
	convW, convb = cc.layers.getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, mode="nl")
	# weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
	# weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
	# weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in, angles = cc.layers.conv(inpt, convW, convb, c_i, c_o, s_h, s_w, first=first, useType=useType, padding=padding, normalize=normalize, count=False, weightMask=None)
	tf.add_to_collection("activationMagnitudes", conv_in)
	tf.add_to_collection("activationAngles", angles)
	return conv_in, angles

def opLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize=True, wd=0.0001):
	convW, convb = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, 2*c_i, s_h, s_w, wd=wd, lossType="nl")
	# weightMask = tf.ones([c_o, k_h, k_w, 2*c_i], tf.float32)
	# weightMask = tf.contrib.image.rotate(weightMask, numpy.pi/4)
	# weightMask = tf.transpose(weightMask, [1,2,3,0])
	conv_in, angles = cc.layers.rotInvarianceWithArgMax(inpt, convW, convb, s_h, s_w, padding=padding, weightMask=None)
	conv_relu = tf.nn.leaky_relu(conv_in, alpha=0.1)
	reluMask = tf.where(conv_relu>0, tf.ones_like(conv_relu), tf.zeros_like(conv_relu))
	regulatedAngles = cc.layers.maskAngleGradients(angles, reluMask)
	if normalize:
		out = cc.layers.batch_norm_only_rescale(conv_in, useType=useType, reuse=resuse_batch_norm)
		# out = cc.layers.normalizeVectorField(conv_relu, convW.get_shape()[-4].value, convW.get_shape()[-3].value)
	else:
		out = conv_relu
	# conv_norm = tf.contrib.layers.group_norm(conv_in, 8, -1, (-3,-2), center=False, scale=False)
	tf.add_to_collection("activationMagnitudes", out)
	tf.add_to_collection("activationAngles", angles)
	return out, angles

def plainLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=True, useType="test", resuse_batch_norm=False, normalize=True, wd=1e-4, convW=None, convb=None,\
 activationFunction="LReLU", reguralizationMode=cc.misc.default_regularization_mode, lr=1):
	if convW == None:
		convW, convb = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, wd=0.0001, lossType="wd", mode=reguralizationMode, lr=lr)
	conv_in = conv(inpt, convW, convb, k_h, k_w, c_o, s_h, s_w, padding=padding, group=1)
	# conv_norm = tf.contrib.layers.group_norm(conv_in, 1, -1, (-3,-2))
	if normalize:
		out = cc.layers.batch_norm(conv_in, useType=useType, reuse=resuse_batch_norm)
	else:
		out = conv_in
	if activationFunction == "LReLU":
		conv_res = tf.nn.leaky_relu(out, alpha=0.1)
	elif activationFunction == "tanh":
		conv_res = tf.tanh(out)
	else:
		exit("ERROR: Unsupported Activation Function in plainLayer")
	# if useType=="train":
	# 	conv_res = tf.nn.dropout(conv_res, rate=0.2)
	return conv_res

def scalarModSpecificConv(inpt, modality, first=True, useType="test", varName="PKNut", resuse_batch_norm=False, fs=5, reguralizationMode=cc.misc.default_regularization_mode, lr=1):
	#conv1
	with tf.variable_scope(varName+"Conv1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 6; c_i = 1; s_h = 1; s_w = 1
		conv1W, conv1b = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, mode=reguralizationMode, lr=lr)
	with tf.variable_scope(modality+"Conv1", reuse=resuse_batch_norm) as scope:	
		conv1 = plainLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, convW=conv1W, convb=conv1b)
	#conv2
	with tf.variable_scope(varName+"Conv2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 6; c_i = 6; s_h = 1; s_w = 1
		conv2W, conv2b = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, mode=reguralizationMode, lr=lr)
	with tf.variable_scope(modality+"Conv2", reuse=resuse_batch_norm) as scope:	
		conv2 = plainLayer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, convW=conv2W, convb=conv2b)
	#maxpool
	with tf.variable_scope(modality+"maxPool") as scope:
		k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
		maxpool = tf.nn.max_pool(conv2+conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool1")

	#conv3
	with tf.variable_scope(varName+"Conv3", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 6; s_h = 1; s_w = 1
		conv3W, conv3b = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, mode=reguralizationMode, lr=lr)
	with tf.variable_scope(modality+"Conv3", reuse=resuse_batch_norm) as scope:	
		conv3 = plainLayer(maxpool, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, convW=conv3W, convb=conv3b)
	#conv4
	with tf.variable_scope(varName+"Conv4", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 8; s_h = 1; s_w = 1
		conv4W, conv4b = cc.misc.getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, mode=reguralizationMode, lr=lr)
	with tf.variable_scope(modality+"Conv4", reuse=resuse_batch_norm) as scope:	
		conv4 = plainLayer(conv3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, convW=conv4W, convb=conv4b)
	#maxpool2
	with tf.variable_scope(modality+"maxPool") as scope:
		k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
		maxpool2 = tf.nn.max_pool(conv4+conv3, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool2")
	return maxpool2

def directionSpecific(inpt, first=True, useType="test", resuse_batch_norm=False, fs=5):
	u = tf.split(inpt, 2, axis=-1)
	#u_x specific layers
	convUx = scalarModSpecificConv(u[0], "u_x", first, useType, varName="u", resuse_batch_norm=not first, fs=fs)
	#u_z specific layers
	convUz = scalarModSpecificConv(u[1], "u_z", False, useType, varName="u", resuse_batch_norm=not first, fs=fs)
	#overall result
	convU = tf.concat([convUx, convUz], axis=-1)
	return convU

def velocityCoherent(inpt, first=True, useType="test", resuse_batch_norm=False, fs=5, reguralizationMode=cc.misc.default_regularization_mode, lr=1):
	#conv1
	with tf.variable_scope("vcConv1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 2; s_h = 1; s_w = 1
		conv1 = plainLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True,\
		 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)
	#conv2
	with tf.variable_scope("vcConv2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 8; s_h = 1; s_w = 1
		conv2 = plainLayer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True,\
		 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)
	#maxpool
	with tf.variable_scope("pool") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled = tf.nn.avg_pool(conv2, [1,2,2,1], [1,2,2,1], padding='VALID')

	#conv3
	with tf.variable_scope("vcConv3", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 16; c_i = 8; s_h = 1; s_w = 1
		conv3 = plainLayer(pooled, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True,\
		 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)
	#conv4
	with tf.variable_scope("vcConv4", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 16; c_i = 16; s_h = 1; s_w = 1
		conv4 = plainLayer(conv3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True,\
		 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)
	#maxpool2
	with tf.variable_scope("pool2") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled2 = tf.nn.avg_pool(conv4, [1,2,2,1], [1,2,2,1], padding='VALID')
	return pooled2

def cliffordConvolution(inpt, first=True, useType="test", resuse_batch_norm=False, fs=5):
	#conv1
	with tf.variable_scope("ccConv1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 3; c_i = 1; s_h = 1; s_w = 1
		conv1_relu, angles_1 = ccLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv1 = cc.transformations.changeToCartesian(conv1_relu, angles_1, False)
	#conv2
	with tf.variable_scope("ccConv2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 4; c_i = 3; s_h = 1; s_w = 1
		conv2_relu, angles_2 = ccLayer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv2 = cc.transformations.changeToCartesian(conv2_relu, angles_2, False)
	#maxpool
	with tf.variable_scope("pool") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled = tf.nn.avg_pool(conv2, [1,2,2,1], [1,2,2,1], padding='VALID')

	#conv3
	with tf.variable_scope("ccConv3", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 4; s_h = 1; s_w = 1
		conv3_relu, angles_3 = ccLayer(pooled, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv3 = cc.transformations.changeToCartesian(conv3_relu, angles_3, False)
	#conv4
	with tf.variable_scope("ccConv4", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 8; s_h = 1; s_w = 1
		conv4_relu, angles_4 = ccLayer(conv3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv4 = cc.transformations.changeToCartesian(conv4_relu, angles_4, False)
	#maxpool2
	with tf.variable_scope("pool2") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled2 = tf.nn.avg_pool(conv4, [1,2,2,1], [1,2,2,1], padding='VALID')
	return pooled2

def maxPoolConvolution(inpt, first=True, useType="test", resuse_batch_norm=False, fs=5):
	#conv1
	with tf.variable_scope("opConv1", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 4; c_i = 1; s_h = 1; s_w = 1
		conv1_relu, angles_1 = opLayer(inpt, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv1 = cc.transformations.changeToCartesian(conv1_relu, angles_1, False)
	#conv2
	with tf.variable_scope("opConv2", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 4; c_i = 4; s_h = 1; s_w = 1
		conv2_relu, angles_2 = opLayer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv2 = cc.transformations.changeToCartesian(conv2_relu, angles_2, False)
	#maxpool
	with tf.variable_scope("pool") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled = tf.nn.avg_pool(conv2, [1,2,2,1], [1,2,2,1], padding='VALID')

	#conv3
	with tf.variable_scope("opConv3", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 4; s_h = 1; s_w = 1
		conv3_relu, angles_3 = opLayer(pooled, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv3 = cc.transformations.changeToCartesian(conv3_relu, angles_3, False)
	#conv4
	with tf.variable_scope("opConv4", reuse=(not first)) as scope:
		k_h = fs; k_w = fs; c_o = 8; c_i = 8; s_h = 1; s_w = 1
		conv4_relu, angles_4 = opLayer(conv3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.01)
		conv4 = cc.transformations.changeToCartesian(conv4_relu, angles_4, False)
	#maxpool2
	with tf.variable_scope("pool2") as scope:
		k_h = 2; k_w = 2; s_h = 2; s_w = 2
		pooled2 = tf.nn.avg_pool(conv4, [1,2,2,1], [1,2,2,1], padding='VALID')
	return pooled2

def inference(examples, first, useType="test", modelType="vc", fs=3, reguralizationMode=cc.misc.default_regularization_mode, lr=1):
	with tf.variable_scope("Encoder") as scope:
		with tf.variable_scope("ScalarConvolutions") as scope:
			#P specific layers
			convP = scalarModSpecificConv(examples[1], "p", first, useType, resuse_batch_norm=not first, reguralizationMode=reguralizationMode, lr=lr)
			#nut specific layers
			convNut = scalarModSpecificConv(examples[2], "nut", False, useType, resuse_batch_norm=not first, reguralizationMode=reguralizationMode, lr=lr)
			#nuTilda specific layers
			convNuTilda = scalarModSpecificConv(examples[3], "nuTilda", False, useType, resuse_batch_norm=not first, reguralizationMode=reguralizationMode, lr=lr)

		with tf.variable_scope("velocityConvolutions") as scope:
			#U specific layers
			if modelType == "vc":
				convU = velocityCoherent(examples[0], first, useType, resuse_batch_norm=not first, reguralizationMode=reguralizationMode, lr=lr)
			elif modelType == "ds":
				convU = directionSpecific(examples[0], first, useType, resuse_batch_norm=not first)
			elif modelType == "cc":
				convU = cliffordConvolution(examples[0], first, useType, resuse_batch_norm=not first)
			elif modelType == "op":
				convU = maxPoolConvolution(examples[0], first, useType, resuse_batch_norm=not first)
	
		fusionIn = tf.concat([convP, convNut, convNuTilda, convU], axis=-1)
		fs = 3
		#conv5
		with tf.variable_scope("Conv5", reuse=(not first)) as scope:
			k_h = fs; k_w = fs; c_o = 64; c_i = fusionIn.get_shape()[-1].value; s_h = 1; s_w = 1
			conv5 = plainLayer(fusionIn, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=not first, normalize=True,\
			 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)

		#conv6
		with tf.variable_scope("Conv6", reuse=(not first)) as scope:
			k_h = fs; k_w = fs; c_o = 64; c_i = 64; s_h = 1; s_w = 1; s_d = 1
			conv6 = plainLayer(conv5, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=not first, normalize=True,\
			 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)

		#maxpool3
		with tf.variable_scope("MaxPool3") as scope:
			k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
			maxpool3 = tf.nn.max_pool(conv6+conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool3")

		#conv7
		with tf.variable_scope("Conv7", reuse=(not first)) as scope:
			k_h = fs; k_w = fs; c_o = 128; c_i = 64; s_h = 1; s_w = 1
			conv7 = plainLayer(maxpool3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=not first, normalize=True,\
			 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)

		#conv8
		with tf.variable_scope("Conv8", reuse=(not first)) as scope:
			k_h = fs; k_w = fs; c_o = 128; c_i = 128; s_h = 1; s_w = 1; s_d = 1
			conv8 = plainLayer(conv7, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=not first, normalize=True,\
			 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)

		#maxpool4
		with tf.variable_scope("MaxPool4") as scope:
			k_h = 2; k_w = 2; k_d = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
			maxpool4 = tf.nn.max_pool(conv8+conv7, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool4")

		#conv9
		with tf.variable_scope("Conv9", reuse=(not first)) as scope:
			k_h = fs; k_w = fs; c_o = 128; c_i = 128; s_h = 1; s_w = 1
			conv9 = plainLayer(maxpool4, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=not first, normalize=True,\
			 wd=0.0001, reguralizationMode=reguralizationMode, lr=lr)

	return conv9


def predictForces(code, batch_size, log, useType="train", first=True, NUM_OUTPUTS=2, reguralizationMode=cc.misc.default_regularization_mode, lr=1):
	with tf.variable_scope("MLPRegressor", reuse=(not first)) as scope:
		#maxpool
		with tf.variable_scope("MaxPool") as scope:
			k_h = 2; k_w = 2; s_h = 2; s_w = 2; s_d = 2; padding = 'VALID'
			maxpool = tf.nn.max_pool(code, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name="maxPool")

		#fc1
		with tf.variable_scope("fc1") as scope:
			reshape = tf.reshape(maxpool, [batch_size, -1])
			dim = reshape.get_shape()[1].value
			# print("Before fully conected, shape = "+str(maxpool.get_shape().as_list()[1:]), file=log)
			# print("Before fully conected, reshape = "+str(dim), file=log)
			stddev=numpy.sqrt(2 / numpy.prod(maxpool.get_shape().as_list()[1:]))
			fc1W = cc.misc._variable_with_weight_decay('weights', shape=[dim, 512], stddev=0.01, wd=0.0001, mode=reguralizationMode, lr=lr)
			fc1b = cc.misc._variable('biases', [512], tf.constant_initializer(1.0))
			fc1 = tf.matmul(reshape, fc1W) + fc1b
			fc1 = cc.layers.batch_norm(fc1, useType=useType, reuse=(not first))
			fc1 = tf.nn.leaky_relu(fc1, name=scope.name)
			if useType=="train":
				fc1 = tf.nn.dropout(fc1, rate=0.4)

		#fc2
		with tf.variable_scope("fc2") as scope:
			stddev=numpy.sqrt(2 / numpy.prod(fc1.get_shape().as_list()[1:]))
			fc2W = cc.misc._variable_with_weight_decay('weights', shape=[512, NUM_OUTPUTS], stddev=0.01, wd=0.0001, mode=reguralizationMode, lr=lr)
			fc2b = cc.misc._variable('biases', [NUM_OUTPUTS], tf.constant_initializer(0.0))
			output = tf.add(tf.matmul(fc1, fc2W), fc2b, name=scope.name)

	return output

def reconstruct(code, log, useType="test"):
	with tf.variable_scope("flowPrediction"):	
		#up1
		with tf.variable_scope("UpSample1") as scope:
			up1 = unPool2D(code, 2, 2)

		#conv1
		with tf.variable_scope("Conv1") as scope:
			k_h = 3; k_w = 3; c_o = 64; c_i = 64; s_h = 1; s_w = 1
			conv1 = plainLayer(up1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#conv2
		with tf.variable_scope("Conv2") as scope:
			k_h = 3; k_w = 3; c_o = 32; c_i = 64; s_h = 1; s_w = 1
			conv2 = plainLayer(conv1, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#up2
		with tf.variable_scope("UpSample2") as scope:
			up2 = unPool2D(conv2, 2, 2)

		#conv3
		with tf.variable_scope("Conv3") as scope:
			k_h = 3; k_w = 3; c_o = 32; c_i = 32; s_h = 1; s_w = 1
			conv3 = plainLayer(up2, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#conv4
		with tf.variable_scope("Conv4") as scope:
			k_h = 3; k_w = 3; c_o = 16; c_i = 32; s_h = 1; s_w = 1
			conv4 = plainLayer(conv3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#up3
		with tf.variable_scope("UpSample3") as scope:
			up3 = unPool2D(conv4, 2, 2)

		#conv5
		with tf.variable_scope("Conv5") as scope:
			k_h = 3; k_w = 3; c_o = 16; c_i = 32; s_h = 1; s_w = 1
			conv5 = plainLayer(up3, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#up4
		with tf.variable_scope("UpSample4") as scope:
			up4 = unPool2D(conv5, 2, 2)

		#conv6
		with tf.variable_scope("Conv6") as scope:
			k_h = 3; k_w = 3; c_o = 8; c_i = 16; s_h = 1; s_w = 1
			conv6 = plainLayer(up4, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#up5
		with tf.variable_scope("UpSample5") as scope:
			up5 = unPool2D(conv6, 2, 2)

		#conv7
		with tf.variable_scope("Conv7") as scope:
			k_h = 3; k_w = 3; c_o = 8; c_i = 8; s_h = 1; s_w = 1
			conv7 = plainLayer(up5, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001)

		#conv8
		with tf.variable_scope("Conv8") as scope:
			k_h = 3; k_w = 3; c_o = 6; c_i = 8; s_h = 1; s_w = 1
			conv8 = plainLayer(conv7, k_h, k_w, c_i, c_o, s_h, s_w, padding="SAME", first=first, useType=useType, resuse_batch_norm=resuse_batch_norm, normalize=True, wd=0.0001, activationFunction="tanh")

	return conv8

