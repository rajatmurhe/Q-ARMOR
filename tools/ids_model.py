"""Original notebook CNN/LSTM/attention architecture, extracted without changes."""
# ==========================================================
# STEP 6: ENHANCED CNN-LSTM-ATTENTION MODEL
# ==========================================================
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# ---- Enhanced Attention Mechanism ----
def create_enhanced_attention_mechanism():
    def attention_layer(inputs):
        attention_dim = inputs.shape[-1]

        query = Dense(attention_dim)(inputs)
        key = Dense(attention_dim)(inputs)
        value = Dense(attention_dim)(inputs)

        # Wrap matmul inside Lambda to keep compatibility with KerasTensors
        attention_scores = Lambda(
            lambda x: tf.matmul(x[0], x[1], transpose_b=True)
        )([query, key])

        attention_scores = Lambda(
            lambda x: x / tf.math.sqrt(tf.cast(attention_dim, tf.float32))
        )(attention_scores)

        attention_weights = Activation('softmax')(attention_scores)

        attended = Lambda(lambda x: tf.matmul(x[0], x[1]))([attention_weights, value])

        attended = Add()([inputs, attended])
        attended = LayerNormalization()(attended)
        return attended
    return attention_layer


# ---- Build Enhanced Model ----
def build_enhanced_cnn_lstm_attention_model(input_shape, num_classes, model_name="Enhanced_IoT_IDS_Model"):
    print(f"Building {model_name} with input shape: {input_shape}")
    
    inputs = Input(shape=input_shape, name='input_layer')
    reshaped = Reshape((input_shape[0], 1))(inputs)

    # CNN Blocks
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(reshaped)
    conv1 = BatchNormalization()(conv1)
    conv1 = Add()([conv1, Conv1D(128, 1, padding='same')(reshaped)])
    pool1 = MaxPooling1D(2)(conv1)
    drop1 = Dropout(0.15)(pool1)

    conv2 = Conv1D(256, 3, activation='relu', padding='same')(drop1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Add()([conv2, Conv1D(256, 1, padding='same')(drop1)])
    pool2 = MaxPooling1D(2)(conv2)
    drop2 = Dropout(0.15)(pool2)

    conv3 = Conv1D(512, 3, activation='relu', padding='same')(drop2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Add()([conv3, Conv1D(512, 1, padding='same')(drop2)])
    pool3 = AveragePooling1D(2)(conv3)
    drop3 = Dropout(0.15)(pool3)

    # LSTM Blocks
    lstm1 = LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(drop3)
    lstm1 = LayerNormalization()(lstm1)

    lstm2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(lstm1)
    lstm2 = LayerNormalization()(lstm2)

    lstm3 = LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(lstm2)

    # Attention
    attention = create_enhanced_attention_mechanism()
    attended = attention(lstm3)
    attended = GlobalAveragePooling1D()(attended)

    # Dense Layers
    dense1 = Dense(512, activation='relu')(attended)
    dense1 = BatchNormalization()(dense1)
    drop4 = Dropout(0.3)(dense1)

    dense2 = Dense(256, activation='relu')(drop4)
    dense2 = BatchNormalization()(dense2)
    drop5 = Dropout(0.3)(dense2)

    dense3 = Dense(128, activation='relu')(drop5)
    dense3 = BatchNormalization()(dense3)
    drop6 = Dropout(0.2)(dense3)

    outputs = Dense(num_classes, activation='softmax')(drop6)
    model = Model(inputs, outputs, name=model_name)
    return model

