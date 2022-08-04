from typing import Any

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def parametrization_func(actual_value: float, lower_bound: float, upper_bound: float) -> float:
    """Parametrizes the given value in an inclusive range between 0 and 1
    utilizing the upper and lower bounds.

    Parameters
    ----------
    actual_value : float
        The value to be parametrized
    lower_bound : float
        The lower bound for the given value
    upper_bound : float
        The upper bound for the given value

    Returns
    -------
    parametrized_value : float
        The parametrization of the given value
    """

    # The following formula is derived from:
    #    actual_value = lower_bound + parametrized_value * (upper_bound - lower_bound)

    parametrized_value = (actual_value - lower_bound)/(upper_bound - lower_bound)

    # Because of floating-point operations, some parametrized values may surpass the limits
    if parametrized_value < 0:
        return 0
    elif parametrized_value > 1:
        return 1
    else:
        return parametrized_value


def unparametrize_func(parametrized_value: float, lower_bound: float, upper_bound: float) -> float:
    """Unparametrizes the given value to its actual value.

    Parameters
    ----------
    parametrized_value : float
        The value to be unparametrized
    lower_bound : float
        The lower bound for the given value
    upper_bound : float
        The upper bound for the given value

    Returns
    -------
    out : float
        The unparametrization of the given value, i.e. the actual value given the bounds
    """

    return lower_bound + parametrized_value * (upper_bound - lower_bound)


def get_label_cols():
    """Returns the labels to be predicted by the Deep Learning model.

    Returns
    -------
    out : list
        The labels that are going to be predicted, 'p_gen30' and 'v_gen30' are not given since
        those values correspond to the slack bus
    """

    return ['p_gen1', 'v_gen1', 'p_gen2', 'v_gen2', 'p_gen3', 'v_gen3', 'p_gen4', 'v_gen4',
            'p_gen5', 'v_gen5', 'p_gen6', 'v_gen6', 'p_gen7', 'v_gen7', 'p_gen8', 'v_gen8',
            'p_gen9', 'v_gen9', 'p_gen10', 'v_gen10', 'p_gen11', 'v_gen11', 'p_gen12', 'v_gen12',
            'p_gen13', 'v_gen13', 'p_gen14', 'v_gen14', 'p_gen15', 'v_gen15', 'p_gen16', 'v_gen16',
            'p_gen17', 'v_gen17', 'p_gen18', 'v_gen18', 'p_gen19', 'v_gen19', 'p_gen20', 'v_gen20',
            'p_gen21', 'v_gen21', 'p_gen22', 'v_gen22', 'p_gen23', 'v_gen23', 'p_gen24', 'v_gen24',
            'p_gen25', 'v_gen25', 'p_gen26', 'v_gen26', 'p_gen27', 'v_gen27', 'p_gen28', 'v_gen28',
            'p_gen29', 'v_gen29', 'p_gen31', 'v_gen31', 'p_gen32', 'v_gen32', 'p_gen33', 'v_gen33',
            'p_gen34', 'v_gen34', 'p_gen35', 'v_gen35', 'p_gen36', 'v_gen36', 'p_gen37', 'v_gen37',
            'p_gen38', 'v_gen38', 'p_gen39', 'v_gen39', 'p_gen40', 'v_gen40', 'p_gen41', 'v_gen41',
            'p_gen42', 'v_gen42', 'p_gen43', 'v_gen43', 'p_gen44', 'v_gen44', 'p_gen45', 'v_gen45',
            'p_gen46', 'v_gen46', 'p_gen47', 'v_gen47', 'p_gen48', 'v_gen48', 'p_gen49', 'v_gen49',
            'p_gen50', 'v_gen50', 'p_gen51', 'v_gen51', 'p_gen52', 'v_gen52', 'p_gen53', 'v_gen53',
            'p_gen54', 'v_gen54']


def pre_process_data_from_file(
        file_path: str,
        test_size: Any = 0.3,
        random_state: Any = 101) -> tuple[Any, Any, Any, Any, DataFrame, DataFrame, MinMaxScaler]:
    """Does the necessary data preprocessing for training a Deep Learning model on AC OPF data.

    Parameters
    ----------
    file_path : str
        The file path to the generated data
    test_size : float or int, default = 0.3
        Used in the train_test_spit function provided by sklearn, from its documentation:
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to
        include in the test split. If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.
    random_state : int, RandomState instance or None, default = 101
        Used in the train_test_spit function provided by sklearn, from its documentation:
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See Glossary <random_state>.

    Returns
    -------
    tuple :
        X_train: array-like of shape (n_samples, n_features) ->
            contains the features values of the training data for the DL model
        X_test: array-like of shape (n_samples, n_features) ->
            contains the features values of the testing data for the DL model
        y_train: array-like of shape (n_samples,) or (n_samples, n_outputs) ->
            contains the label values of the training data for the DL model
        y_test: array-like of shape (n_samples,) or (n_samples, n_outputs) ->
            contains the label values of the testing data for the DL model
        df: DataFrame -> The loaded .csv file from the 'file_path' in a DataFrame object
        slack_bus_data: DataFrame -> The data of the slack bus,
            extracted from the provided .csv file, in a DataFrame object
        scaler: MinMaxScaler -> Scaler used to scale the training and testing data
    """

    import re
    import pandas as pd
    import bus_bounds_data as bb_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(file_path, index_col=0)

    slack_bus_data = df[['p_gen30', 'q_gen30', 'v_gen30']].copy()
    slack_bus_data['v_angle30'] = 30
    df.drop(['p_gen30', 'q_gen30', 'v_gen30'], axis=1, inplace=True)

    feature_pattern = "^p_load|^q_load"
    feature_cols = [col for col in df.columns[:-2]
                    if re.search(feature_pattern, col)]

    label_cols = get_label_cols()

    v_min = 0.94
    v_max = 1.06

    for col, p_bounds_key in zip(label_cols[::2], bb_data.p_bounds.keys()):
        df[col] = df[col].apply(lambda x: parametrization_func(
            x, bb_data.p_bounds[p_bounds_key]['p_min'], bb_data.p_bounds[p_bounds_key]['p_max']))

    for col in label_cols[1::2]:
        df[col] = df[col].apply(
            lambda x: parametrization_func(x, v_min, v_max))

    X = df[feature_cols].values
    y = df[label_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, df, slack_bus_data, scaler


def unscale_data(X_train: Any, X_test: Any, scaler: MinMaxScaler) -> tuple[Any, Any]:
    """Unscales both the training and testing data using the provided scaler.

    Parameters
    ----------
    X_train: array-like of shape (n_samples, n_features) ->
        scaled features values of the training data for the DL model
    X_test: array-like of shape (n_samples, n_features) ->
        scaled features values of the testing data for the DL model
    scaler: MinMaxScaler -> utilized scaler to scale the training and testing data

    Returns
    -------
    tuple :
        X_train: array-like of shape (n_samples, n_features) ->
            unscaled features values of the training data for the DL model
        X_test: array-like of shape (n_samples, n_features) ->
            unscaled features values of the testing data for the DL model
    """

    return scaler.inverse_transform(X_train), scaler.inverse_transform(X_test)


def unscale_X_train_data(X_train: Any, scaler: MinMaxScaler) -> Any:
    """Unscales the training data using the provided scaler.

    Parameters
    ----------
    X_train: array-like of shape (n_samples, n_features) ->
        scaled features values of the training data for the DL model
    scaler: MinMaxScaler -> utilized scaler to scale the training and testing data

    Returns
    -------
    out: array-like of shape (n_samples, n_features) ->
        unscaled features values of the training data for the DL model
    """

    return scaler.inverse_transform(X_train)


def unscale_X_test_data(X_test: Any, scaler: MinMaxScaler) -> Any:
    """Unscales the testing data using the provided scaler.

    Parameters
    ----------
    X_test: array-like of shape (n_samples, n_features) ->
        scaled features values of the testing data for the DL model
    scaler: MinMaxScaler -> utilized scaler to scale the training and testing data

    Returns
    -------
    out: array-like of shape (n_samples, n_features) ->
        unscaled features values of the testing data for the DL model
    """

    return scaler.inverse_transform(X_test)


def train_model(X_train: Any, X_test: Any, y_train: Any, y_test: Any,
                plot_losses: bool = True, verbose: str = 'auto'):
    """Trains a Deep Neural Network for the task of predicting the generator setpoints
    of a power grid, with feature data representing the load demands (pl and ql) and labels being
    the active power and voltage generation for each generator bus, except for the slack bus.

    Parameters
    ----------
    X_train: array-like of shape (n_samples, n_features) ->
        contains the features values of the training data for the DL model
    X_test: array-like of shape (n_samples, n_features) ->
        contains the features values of the testing data for the DL model
    y_train: array-like of shape (n_samples,) or (n_samples, n_outputs) ->
        contains the label values of the training data for the DL model
    y_test: array-like of shape (n_samples,) or (n_samples, n_outputs) ->
        contains the label values of the testing data for the DL model
    plot_losses: bool, default = True
        Indicates whether to plot the history of the loss and validation loss after training the DNN
    verbose: 'auto', 0, 1, or 2, default = 'auto'
        Used in the model.fit() function provided by the keras API, from the documentation:
        Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        'auto' defaults to 1 for most cases, but 2 when used with ParameterServerStrategy.
        Note that the progress bar is not particularly useful when logged to a file,
        so verbose=2 is recommended when not running interactively (eg, in a production environment).

    Returns
    -------
    model: Trained Deep Learning model
    """

    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam

    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    model = Sequential()

    model.add(Dense(236, activation='sigmoid'))
    model.add(Dense(472, activation='sigmoid'))
    model.add(Dense(236, activation='sigmoid'))
    model.add(Dense(212, activation='sigmoid'))
    model.add(Dense(106, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss="mse")

    history = model.fit(X_train, y_train, epochs=150, validation_data=(
        X_test, y_test), callbacks=[early_stop], verbose=verbose)

    if plot_losses:
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()

    return model


def evaluate_model(y_test_df: DataFrame, y_pred_df: DataFrame):
    """Prints the Mean Absolute Error (MAE), Mean Squared Error (MSE),
    and Root Mean Squared Error (RMSE) with respect to the predicted active power and voltage
    generation values.

    Parameters
    ----------
    y_test_df: DataFrame
        The ground truth of the labels
    y_pred_df: DataFrame
        The predicted label values
    """

    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    p_gen_test, p_gen_pred = y_test_df.loc[:, 'p_gen1'::2].values, \
        y_pred_df.loc[:, 'p_gen1'::2].values

    print("Results of Active Power Generation:")
    print("MAE :", mean_absolute_error(p_gen_test, p_gen_pred))
    print("MSE :", mean_squared_error(p_gen_test, p_gen_pred))
    print("RMSE :", np.sqrt(mean_squared_error(p_gen_test, p_gen_pred)))
    print()

    v_gen_test, v_gen_pred = y_test_df.loc[:, 'v_gen1'::2].values, \
        y_pred_df.loc[:, 'v_gen1'::2].values

    print("Results of Voltage Generation:")
    print("MAE :", mean_absolute_error(v_gen_test, v_gen_pred))
    print("MSE :", mean_squared_error(v_gen_test, v_gen_pred))
    print("RMSE :", np.sqrt(mean_squared_error(v_gen_test, v_gen_pred)))


def get_bus_gen_nums(include_slack: bool = False) -> list:
    """Returns the number of the bus where each generator is located.

    Parameters
    ----------
    include_slack: bool, default = False
        Indicates whether to include the number of the bus where the slack bus is located

    Returns
    -------
    out: list
        A list with the number of the bus where each generator is located
    """

    if not include_slack:
        return ['1', '4', '6', '8', '10', '12', '15', '18', '19', '24', '25', '26',
                '27', '31', '32', '34', '36', '40', '42', '46', '49', '54', '55', '56',
                '59', '61', '62', '65', '66', '70', '72', '73', '74', '76', '77',
                '80', '85', '87', '89', '90', '91', '92', '99', '100', '103', '104', '105',
                '107', '110', '111', '112', '113', '116']
    else:
        return ['1', '4', '6', '8', '10', '12', '15', '18', '19', '24', '25', '26',
                '27', '31', '32', '34', '36', '40', '42', '46', '49', '54', '55', '56',
                '59', '61', '62', '65', '66', '69', '70', '72', '73', '74', '76', '77',
                '80', '85', '87', '89', '90', '91', '92', '99', '100', '103', '104', '105',
                '107', '110', '111', '112', '113', '116']


def get_Ybus_Matrix(
        num_of_busses: int, file_path: str,
        return_mag_ang: bool = True) -> Any:
    """Calculates the Y bus admittance matrix, following the procedure presented in the paper:
    'Python Programming Language for Power System Analysis Education and Research' (Fernandes, 2018)

    Parameters
    ----------
    num_of_busses: int
        The number of busses in the network
    file_path: str
        The file path to the .txt case data, which contains the branch information
    return_mag_ang: bool, default = True
        Indicates whether to return the Y bus admittance matrix together with two additional
        matrices, which are composed of the magnitudes and angles of the complex numbers in the
        Y matrix, respectively

    Returns
    -------
    out: either a tuple of matrices or a single matrix
        The Y bus admittance matrix with the magnitude and angle matrices of the complex numbers in
        Y or only the Y matrix
    """

    import numpy as np
    from egret.parsers.matpower_parser import create_ModelData

    model_data = create_ModelData(file_path)

    # Gather the sending busses
    ibfr = np.array([int(model_data.data['elements']['branch'][key]['from_bus'])
                    for key in model_data.data['elements']['branch'].keys()])

    # Gather the receiving busses
    ibto = np.array([int(model_data.data['elements']['branch'][key]['to_bus'])
                    for key in model_data.data['elements']['branch'].keys()])

    # Gather the resistances
    rkm = np.array([model_data.data['elements']['branch'][key]['resistance']
                   for key in model_data.data['elements']['branch'].keys()])

    # Gather the reactances
    xkm = np.array([model_data.data['elements']['branch'][key]['reactance']
                   for key in model_data.data['elements']['branch'].keys()])

    # Gather the shunt susceptances
    bkm = np.array([model_data.data['elements']['branch'][key]['charging_susceptance']
                   for key in model_data.data['elements']['branch'].keys()])

    # number of lines
    nl = len(ibfr)

    ykm = 1/(rkm + 1j*xkm)
    yl = ykm + 1j*bkm/2
    yp = np.diagflat(ykm)
    if num_of_busses > nl:
        Id = np.eye(num_of_busses)
    else:
        Id = np.eye(nl)
    A = Id[ibfr-1, :] - Id[ibto-1, :]
    Y = (A.T.dot(yp)).dot(A) + np.diagflat(1j/2 * np.abs(A.T).dot(bkm))

    if return_mag_ang:
        return Y, np.abs(Y), np.degrees(np.angle(Y))
    else:
        return Y
