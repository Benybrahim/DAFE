"""
Implementation of DAFE Network paper:
http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Chen_Improving_Fashion_Landmark_Detection_by_Dual_Attention_Feature_Enhancement_ICCVW_2019_paper.pdf
"""

from tensorflow.keras.applications import ResNet50V2
import tensorflow as tf

layers = tf.keras.layers


def DAFE(input_shape, n_classes):

    # ResNet50 backbone (bottom-up)
    resnet = ResNet50V2(weights='imagenet', include_top=False,
                        input_shape=input_shape)

    conv2 = resnet.get_layer('pool1_pool').output
    conv3 = resnet.get_layer('conv2_block3_out').output
    conv4 = resnet.get_layer('conv3_block4_out').output
    conv5 = resnet.get_layer('conv4_block6_out').output

    # DAFE blocks (top-down)
    x = sau_block([conv5, conv4], 512, name='sau1')
    x = cas_block(x, [256, 512], name='cas1')
    x = layers.Add(name='dafe1_out')([x, conv4])

    x = sau_block([x, conv3], 256, name='sau2')
    x = cas_block(x, [64, 256], name='cas2')
    x = layers.Add(name='dafe2_out')([x, conv3])

    x = sau_block([x, conv2], 64, name='sau3')
    x = cas_block(x, [32, 64], name='cas3')
    x = layers.Add(name='dafe3_out')([x, conv2])

    # Classifier
    x = layers.Conv2D(n_classes, (1, 1), padding='same', activation='linear',
                      name='conv_classifier')(x)

    return tf.keras.models.Model(inputs=resnet.input, outputs=x, name='dafe')


def sau_block(inputs, filters, name):
    """Spatial attentive upsampling block
    sam: spatial attention matrix
    """
    x, y = inputs

    x_conv_tr = layers.Conv2DTranspose(filters, 1, 2, padding='same',
                               activation='relu', name=f'{name}_conv_tr')(x)

    sam = layers.Conv2D(filters, 1, padding='same', name=f'{name}_conv')(x)
    sam = layers.Activation('sigmoid', name=f'{name}_sigmoid')(sam)
    sam = layers.UpSampling2D(2, name=f'{name}_conv_up',)(sam)
    sam = layers.BatchNormalization(name=f'{name}_norm')(sam)
    y = layers.Multiply(name=f'{name}_sam_out')([sam, y])

    output = layers.Add(name=f'{name}_out')([x_conv_tr, y])
    return output

def cas_block(inputs, filters, name):
    """Channel wise attentive selection block
    cam: channel-wise attention matrix
    """
    x = inputs

    cam = layers.GlobalAveragePooling2D(name=f'{name}_pool')(x)
    cam = layers.Reshape(target_shape=(1, 1, cam.shape[1]), name=f'{name}_reshape')(cam)
    cam = layers.Conv2D(filters[0], 1, padding='same', name=f'{name}_conv1')(cam)
    cam = layers.Activation('relu', name=f'{name}_relu')(cam)
    cam = layers.Conv2D(filters[1], 1, padding='same', name=f'{name}_conv2')(cam)

    output = layers.Multiply(name=f'{name}_out')([x, cam])
    return output


if __name__=='__main__':
    # test
    model = DAFE((320, 320, 3), n_classes=25)
    print(model.summary())

