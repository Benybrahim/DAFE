"""
Implementation of DAFE Network paper:
http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Chen_Improving_Fashion_Landmark_Detection_by_Dual_Attention_Feature_Enhancement_ICCVW_2019_paper.pdf
"""
import os
import sys

from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf

layers = tf.keras.layers

def DAFE(input_shape, n_heatmaps):

    # ResNet50 backbone (bottom-up)
    resnet = ResNet50V2(weights='imagenet', include_top=False,
                        input_shape=input_shape)

    conv2 = resnet.get_layer('pool1_pool').output
    conv3 = resnet.get_layer('conv2_block3_out').output
    conv4 = resnet.get_layer('conv3_block4_out').output
    conv5 = resnet.get_layer('conv4_block6_out').output

    # DAFE blocks (top-down)
    x = _SauBlock(512, name='sau1')([conv5, conv4])
    x = _CasBlock(512, name='cas1')(x)

    x = _SauBlock(256, name='sau2')([x, conv3])
    x = _CasBlock(256, name='cas2')(x)

    x = _SauBlock(64, name='sau3')([x, conv2])
    x = _CasBlock(64, name='cas3')(x) # classes

    # RPN Head (Classifier)
    x = layers.Conv2D(n_heatmaps, (3, 3), padding='same',
                      name='conv_classifier')(x)

    x = layers.Conv2D(n_heatmaps, (1, 1), padding='same',
                      name='heatmaps_predictor')(x)

    return tf.keras.models.Model(inputs=resnet.input, outputs=x, name='dafe')


class _SauBlock(layers.Layer):
    """Spatial attentive upsampling block
    sam: spatial attention matrix
    """
    def __init__(self, filters, name='', **kwargs):
        super(_SauBlock, self).__init__(name=name, **kwargs)
        self.conv_tr = layers.Conv2DTranspose(filters, (1, 1), strides=(2, 2),
                                              padding='same',
                                              name=name + '_conv_tr')
        self.conv = layers.Conv2D(filters, (1, 1), padding='same',
                                  name=name + '_conv')
        self.conv_up = layers.UpSampling2D((2, 2), name=name + '_conv_up')

    def call(self, inputs):
        x, y = inputs
        sam = self.conv(x)
        sam = tf.nn.sigmoid(sam)
        sam = self.conv_up(sam)
        x_conv_tr = self.conv_tr(x)
        return x_conv_tr + (sam * y)


class _CasBlock(tf.keras.Model):
  """Channel wise attentive selection block
  cam: channel-wise attention matrix
  """
  def __init__(self, filters, name='', **kwargs):
      super(_CasBlock, self).__init__(name=name, **kwargs)
      self.global_pooling = layers.GlobalAveragePooling2D()
      self.conv1 = layers.Conv2D(filters, (1, 1), name=name+'_conv1',
                                 padding='same')
      self.conv2 = layers.Conv2D(filters, (1, 1), name=name+'_conv2',
                                 padding='same')

  def call(self, inputs):
      x = inputs
      x_global = self.global_pooling(x)
      x_global = tf.reshape(x_global, shape=(-1, 1, 1, x_global.shape[1]))
      cam = self.conv1(x_global)
      cam = tf.nn.relu(cam)
      cam = self.conv2(cam)
      return x * cam

if __name__=='__main__':
    model = DAFE((320, 256, 3), n_heatmaps=25)

    print(model.summary())
