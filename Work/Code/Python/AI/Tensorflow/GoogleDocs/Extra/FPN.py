import tensorflow as tf
import tensorflow_models as tfm


image  = tf.keras.preprocessing.image.load_img("asset/test.jpg")

tf_image = tf.convert_to_tensor(image)

print(tf_image.shape)


FPN = tfm.vision.decoders.FPN(
    input_specs = tf.TensorShape([685, 959, 3]),
    min_level = 3,
    max_level = 7,
    num_filters = 256,
    fusion_type = 'sum',
    use_separable_conv =  False,
    use_keras_layer = False,
    activation = 'relu',
    use_sync_bn = False,
    norm_momentum = 0.99,
    norm_epsilon = 0.001,
    kernel_initializer = 'VarianceScaling',
    kernel_regularizer = None,
    bias_regularizer = None
)


result = FPN(tf_image)

print(result)