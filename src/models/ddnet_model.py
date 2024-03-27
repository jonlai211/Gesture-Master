from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model


def build_dd_net(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Dense(256, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)

    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)

    x = Dense(64, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# # Example usage:
# input_shape = (231,)  # for example, 231 JCD features
# num_classes = 5  # assuming 5 classes for classification
#
# model = build_simplified_dd_net(input_shape, num_classes)
# model.summary()
