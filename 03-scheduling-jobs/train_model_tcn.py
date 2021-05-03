import numpy as np
from pathlib import Path
import pandas as pd
import datetime
import random
import h5py
import zipfile

import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import ParameterSampler

from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import traceback


import threshold
from tcn import TCN

folder_processed_data = Path("data/processed/")  # processed data folder

# extract zip file of processed data
with zipfile.ZipFile(folder_processed_data / 'data_processed.zip', 'r') as zip_ref:
    zip_ref.extractall(folder_processed_data)

# set location to save trained models
home_dir = Path.home()
Path(home_dir / 'scratch/models').mkdir(parents=True, exist_ok=True)
folder_models = home_dir / 'scratch/models'


####################### HELPER FUNCTIONS ###########################
# simple functions used in the data prep


def scaler(x, min_val_array, max_val_array):
    """
    Function to scale the data with min-max values
    """

    # get the shape of the array
    s, _, sub_s = np.shape(x)

    for i in range(s):
        for j in range(sub_s):
            x[i, :, j] = np.divide(
                (x[i, :, j] - min_val_array[j]),
                np.abs(max_val_array[j] - min_val_array[j]),
            )

    return x


# min-max function
def get_min_max(x):
    """
    Function to get the min-max values
    """

    # flatten the input array http://bit.ly/2MQuXZd
    flat_vector = np.concatenate(x)

    min_vals = np.min(flat_vector, axis=0)
    max_vals = np.max(flat_vector, axis=0)

    return min_vals, max_vals


def load_train_test(directory):
    """
    Function to quickly load the train/val/test data splits
    """

    path = directory

    with h5py.File(path / "X_train.hdf5", "r") as f:
        X_train = f["X_train"][:]
    with h5py.File(path / "y_train.hdf5", "r") as f:
        y_train = f["y_train"][:]

    with h5py.File(path / "X_train_slim.hdf5", "r") as f:
        X_train_slim = f["X_train_slim"][:]
    with h5py.File(path / "y_train_slim.hdf5", "r") as f:
        y_train_slim = f["y_train_slim"][:]

    with h5py.File(path / "X_val.hdf5", "r") as f:
        X_val = f["X_val"][:]
    with h5py.File(path / "y_val.hdf5", "r") as f:
        y_val = f["y_val"][:]

    with h5py.File(path / "X_val_slim.hdf5", "r") as f:
        X_val_slim = f["X_val_slim"][:]
    with h5py.File(path / "y_val_slim.hdf5", "r") as f:
        y_val_slim = f["y_val_slim"][:]

    with h5py.File(path / "X_test.hdf5", "r") as f:
        X_test = f["X_test"][:]
    with h5py.File(path / "y_test.hdf5", "r") as f:
        y_test = f["y_test"][:]

    return (
        X_train,
        y_train,
        X_train_slim,
        y_train_slim,
        X_val,
        y_val,
        X_val_slim,
        y_val_slim,
        X_test,
        y_test,
    )


########## model training function ###############

# build custom sampling function
# Sampling and rounded_accuracy code modified from Aurelion Geron,
# https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb
# used under Apache 2.0 License, https://github.com/ageron/handson-ml2/blob/master/LICENSE

K = keras.backend

# class for sampling embeddings in the latent space
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


# rounded accuracy for the metric
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


# fit the model
def model_fit(
    X_train_slim,
    X_val_slim,
    beta_value=1.25,
    codings_size=10,
    dilations=[1, 2, 4],
    conv_layers=1,
    seed=31,
    start_filter_no=32,
    kernel_size_1=2,
    epochs=10,
    earlystop_patience=8,
    verbose=0,
    compile_model_only=False,
):

    # try the first if it is the milling data
    # else it will be the CNC data
    try:
        _, window_size, feat = X_train_slim.shape

    except:
        window_size = X_train_slim.shape
        feat = 1

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tf.random.set_seed(seed)
    np.random.seed(seed)

    end_filter_no = start_filter_no

    inputs = keras.layers.Input(shape=[window_size, feat])
    z = inputs

    # ENCODER
    ####### TCN #######
    for i in range(0, conv_layers):
        z = TCN(
            nb_filters=start_filter_no,
            kernel_size=kernel_size_1,
            nb_stacks=1,
            dilations=dilations,
            padding="causal",
            use_skip_connections=True,
            dropout_rate=0.0,
            return_sequences=True,
            activation="selu",
            kernel_initializer="he_normal",
            use_batch_norm=False,
            use_layer_norm=False,
        )(z)

        z = keras.layers.BatchNormalization()(z)
        z = keras.layers.MaxPool1D(pool_size=2)(z)

    z = keras.layers.Flatten()(z)
    print("Shape of Z:", z.shape)

    codings_mean = keras.layers.Dense(codings_size)(z)

    codings_log_var = keras.layers.Dense(codings_size)(z)

    codings = Sampling()([codings_mean, codings_log_var])

    variational_encoder = keras.models.Model(
        inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
    )

    # DECODER
    decoder_inputs = keras.layers.Input(shape=[codings_size])

    x = keras.layers.Dense(
        start_filter_no * int((window_size / (2 ** conv_layers))), activation="selu"
    )(decoder_inputs)

    x = keras.layers.Reshape(
        target_shape=((int(window_size / (2 ** conv_layers))), end_filter_no)
    )(x)

    for i in range(0, conv_layers):
        x = keras.layers.UpSampling1D(size=2)(x)
        x = keras.layers.BatchNormalization()(x)

        x = TCN(
            nb_filters=start_filter_no,
            kernel_size=kernel_size_1,
            nb_stacks=1,
            dilations=dilations,
            padding="causal",
            use_skip_connections=True,
            dropout_rate=0.0,
            return_sequences=True,
            activation="selu",
            kernel_initializer="he_normal",
            use_batch_norm=False,
            use_layer_norm=False,
        )(x)

    outputs = keras.layers.Conv1D(
        feat, kernel_size=kernel_size_1, padding="same", activation="sigmoid"
    )(x)
    variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

    _, _, codings = variational_encoder(inputs)
    reconstructions = variational_decoder(codings)
    variational_ae_beta = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

    latent_loss = (
        -0.5
        * beta_value
        * K.sum(
            1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
            axis=-1,
        )
    )

    variational_ae_beta.add_loss(K.mean(latent_loss) / (window_size * feat))
    variational_ae_beta.compile(
        loss="binary_crossentropy",
        optimizer="adam",  #'rmsprop'
        metrics=[rounded_accuracy],
    )

    # count the number of parameters
    param_size = "{:0.2e}".format(
        variational_encoder.count_params() + variational_decoder.count_params()
    )

    # Uncomment these if you want to see the summary of the encoder/decoder
    # variational_encoder.summary()
    # variational_decoder.summary()

    # Model Name
    # b : beta value used in model
    # c : number of codings -- latent variables
    # l : numer of convolutional layers in encoder (also decoder)
    # f1 : the starting number of filters in the first convolution
    # k1 : kernel size for the first convolution
    # k2 : kernel size for the second convolution
    # d : whether dropout is used when sampling the latent space (either True or False)
    # p : number of parameters in the model (encoder + decoder params)
    # eps : number of epochs
    # pat : patience stopping number

    model_name = (
        "TBVAE-{}:_b={:.2f}_c={}_l={}_f1={}_k1={}_dil={}"
        "_p={}_eps={}_pat={}".format(
            date_time,
            beta_value,
            codings_size,
            conv_layers,
            start_filter_no,
            kernel_size_1,
            dilations,
            param_size,
            epochs,
            earlystop_patience,
        )
    )

    print("\n", model_name, "\n")

    if compile_model_only == False:

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0002,
            patience=earlystop_patience,
            restore_best_weights=True,
            verbose=1,
        )

        history = variational_ae_beta.fit(
            X_train_slim,
            X_train_slim,
            epochs=epochs,
            batch_size=1024,
            shuffle=True,
            validation_data=(X_val_slim, X_val_slim),
            callbacks=[earlystop_callback,], 
            verbose=verbose,
        )

        return date_time, model_name, history, variational_ae_beta, variational_encoder

    else:

        return variational_ae_beta, variational_encoder


############## random search ###############3

# load the data splits
(
    X_train,
    y_train,
    X_train_slim,
    y_train_slim,
    X_val,
    y_val,
    X_val_slim,
    y_val_slim,
    X_test,
    y_test,
) = load_train_test(folder_processed_data)


# Input the number of iterations you want to search over
random_search_iterations = 5

# random seed value from system input
ransdom_seed_input = random.randint(0,9999)

# parameters for beta-vae
p_bvae_grid = {
    "beta_value": uniform(loc=0.5, scale=9),
    "codings_size": sp_randint(5, 40),
    "conv_layers": [3, 2, 1],
    "start_filter_no": sp_randint(16, 128),
    "dilations": [[1, 2, 4, 8], [1, 2, 4], [1, 2]],
    "kernel_size_1": sp_randint(2, 9),
    "earlystop_patience": sp_randint(10, 100),
}

# epochs
epochs = 2


# folder to save models in
model_save_folder = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_mill"

# create the folder
(folder_models / "saved_models" / model_save_folder).mkdir(parents=True, exist_ok=True)

# create dataframe to store all the results
df_all = pd.DataFrame()

# setup parameters to sample
rng = np.random.RandomState(ransdom_seed_input)

# list of parameters in random search
p_bvae = list(
    ParameterSampler(p_bvae_grid, n_iter=random_search_iterations, random_state=rng)
)


for i, params in enumerate(p_bvae):
    print("\n### Run no.", i + 1)

    ### TRY MODELS ###

    # BETA-VAE
    # parameters
    beta_value = params["beta_value"]
    codings_size = params["codings_size"]
    conv_layers = params["conv_layers"]
    start_filter_no = params["start_filter_no"]
    kernel_size_1 = params["kernel_size_1"]
    dilations = params["dilations"]
    earlystop_patience = params["earlystop_patience"]

    seed = 16
    verbose = 1

    # try the model and if it doesn't work, go onto the next model
    # not always the best to use 'try' but good enough
    try:

        date_time, model_name, history, beta_vae_model, bvae_encoder = model_fit(
            X_train_slim,
            X_val_slim,
            beta_value=beta_value,
            codings_size=codings_size,
            conv_layers=conv_layers,
            seed=seed,
            start_filter_no=start_filter_no,
            kernel_size_1=kernel_size_1,
            dilations=dilations,
            epochs=epochs,
            earlystop_patience=earlystop_patience,
            verbose=verbose,
        )

        # save the model. How to: https://www.tensorflow.org/tutorials/keras/save_and_load
        # save model weights and model json
        model_save_dir_bvae = (
            folder_models / "saved_models" / model_save_folder / (date_time + "_bvae")
        )
        model_save_dir_encoder = (
            folder_models
            / "saved_models"
            / model_save_folder
            / (date_time + "_encoder")
        )

        # create the save paths
        Path(model_save_dir_bvae).mkdir(parents=True, exist_ok=True)
        Path(model_save_dir_encoder).mkdir(parents=True, exist_ok=True)

        # save entire bvae model
        model_as_json = beta_vae_model.to_json()
        with open(r"{}/model.json".format(str(model_save_dir_bvae)), "w",) as json_file:
            json_file.write(model_as_json)
        beta_vae_model.save_weights(str(model_save_dir_encoder) + "/weights.h5")

        # save encoder bvae model
        model_as_json = bvae_encoder.to_json()
        with open(
            r"{}/model.json".format(str(model_save_dir_encoder)), "w",
        ) as json_file:
            json_file.write(model_as_json)
        bvae_encoder.save_weights(str(model_save_dir_encoder) + "/weights.h5")

        # get the model run history
        results = pd.DataFrame(history.history)
        epochs_trained = len(results)
        results["epochs_trained"] = epochs_trained

        results = list(
            results[results["val_loss"] == results["val_loss"].min()].to_numpy()
        )  # only keep the top result, that is, the lowest val_loss

        # append best result onto df_model_results dataframe
        if i == 0:
            cols = (
                list(p_bvae[0].keys())
                + list(history.history.keys())
                + ["epochs_trained"]
            )
            results = [[p_bvae[i][k] for k in p_bvae[i]] + list(results[0])]

        else:
            # create dataframe to store best result from model training
            cols = (
                list(p_bvae[0].keys())
                + list(history.history.keys())
                + ["epochs_trained"]
            )
            results = [[p_bvae[i][k] for k in p_bvae[i]] + list(results[0])]

        recon_check = threshold.SelectThreshold(
            beta_vae_model,
            X_train,
            y_train,
            X_train_slim,
            X_val,
            y_val,
            X_val_slim,
            class_to_remove=[2],
            class_names=["0", "1", "2"],
            model_name=model_name,
            date_time=date_time,
        )

        df = recon_check.compare_error_method(
            show_results=False,
            grid_iterations=100,
            model_results=results,
            model_result_cols=cols,
            search_iterations=2,
        )

        # df = pd.DataFrame(results, columns=cols)

        df['rand_int'] = ransdom_seed_input
        # df['model_name'] = model_name

        df_all = df_all.append(df, sort=False)

        df_all.to_csv(folder_models / "results_interim_{}.csv".format(model_save_folder))

    except Exception as e:
        print(e)
        print("TRACEBACK")
        traceback.print_exc()
        pass
