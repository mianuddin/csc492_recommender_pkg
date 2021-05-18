import numpy as np
import pandas as pd
import random
import re
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from .metrics import perform_groupwise_evaluation
from .preprocessing import get_standard_layers


def plot_metric_history(history_df, title=""):
    """Plot each metric versus epochs.

    Args:
        history_df (pandas.DataFrame): A tidy dataframe with `epoch`, `metric`,
                                       and `value` columns.
        title (String): Text which will be prepended to the title of each
                        graph.

    Returns:
        List[FacetGrid, ...]: The metric plots.
    """
    metrics = filter(lambda x: not re.match("val", x),
                     history_df["metric"].unique())
    title_val = title + " " if title != "" else ""
    plots = []

    for metric in metrics:
        filtered_df = history_df[(history_df["metric"] == metric) |
                                 (history_df["metric"] == f"val_{metric}")]
        plots.append(sns.relplot(x="epoch",
                                 y="value",
                                 data=filtered_df,
                                 kind="line",
                                 hue="metric")
                        .set(title=f"{title_val}{metric}"))

    return plots


def evaluate_model(ModelConstructor,
                   model_name,
                   X_train,
                   X_test,
                   y_train,
                   y_test,
                   seed_val,
                   configs,
                   plot=False):
    """Evaluate multiple of model configs.

    Args:
        ModelConstructor(KerasRecommender): The constructor of the model which
                                            is being evaluated.
        model_name (String): The name of the model which is being evaluated.
        X_train (ndarray of shape (n_samples, 2)): This is the train set. An
                                                   array where each row
                                                   consists of a user and an
                                                   item.
        X_test (ndarray of shape (n_samples, 2)): This is the test set. An
                                                  array where each row consists
                                                  of a user and an item.
        y_train (ndarray of shape (n_samples,)): This is the train set. An
                                                 array where each entry denotes
                                                 interactions between the
                                                 corresponding user and item.
        y_test (ndarray of shape (n_samples,)): This is the train set. An array
                                                where each entry denotes
                                                interactions between the
                                                corresponding user and item.
        seed_val (int): A random seed.
        configs (List[Dict[String, Object]]): A list of dictionaries of keyword
                                             arguments to be applied in the
                                             model's constructor.
        plot (bool): Should training plots be made?

    Returns:
        Dict[String, Dict]: The configs, trained models, history dataframes,
        training plots, and groupwise evaluations.
    """

    trained_models = {}
    history_dfs = {}
    training_plots = {}
    groupwise_evals = {}

    users = np.unique(np.concatenate((X_train, X_test))[:, 0])
    items = np.unique(np.concatenate((X_train, X_test))[:, 1])

    user_input, user_pp_layers = get_standard_layers(users, "user")
    item_input, item_pp_layers = get_standard_layers(items, "item")

    for i, config in tqdm(list(enumerate(configs))):
        if i != 0:
            print()
        print(f"CONFIG {i}")
        print(config)

        # Instantiate model
        random.seed(seed_val)
        np.random.seed(seed_val)
        tf.random.set_seed(seed_val)

        model = ModelConstructor(**config,
                                 user_input=user_input,
                                 item_input=item_input,
                                 user_preprocessing_layers=user_pp_layers,
                                 item_preprocessing_layers=item_pp_layers)
        model.fit(X_train, y_train)

        # Create training history dataframe
        history_df = (
            pd.melt(pd.DataFrame({"epoch": range(1, model.epochs+1),
                                  **model.history.history}),
                    id_vars=["epoch"],
                    value_vars=model.history.history.keys(),
                    var_name="metric")
        )

        # Plot training
        if plot:
            plots = plot_metric_history(history_df,
                                        title=f"{model_name} ({i})")

        # Evaluate model
        y_pred_implicit = model.predict(X_test)
        groupwise_eval = perform_groupwise_evaluation(X_test,
                                                      y_test,
                                                      y_pred_implicit)
        print(groupwise_eval)

        # Save everything
        trained_models[i] = model
        history_dfs[i] = history_df
        if plot:
            training_plots[i] = plots
        groupwise_evals[i] = groupwise_eval

    return {
        "configs": dict(enumerate(configs)),
        "trained_models": trained_models,
        "history_dfs": history_dfs,
        "training_plots": training_plots,
        "groupwise_evals": groupwise_evals
    }
