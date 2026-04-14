import pandas as pd
import numpy as np
from scipy import stats
import glob
import os
import matplotlib.pyplot as plt



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",   # LaTeX default serif (Times under ICLR style)
    "font.size": 10,          # match ICLR main text
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

label_map = {
    "discrepancy": "Discrepancy",
    "ambiguity": "Ambiguity",
    "mean_vpr_width": "VPR",
    "rashomon_capacity": "Rash. Cap.",
    "rashomon_ratio": "Rash. Ratio",
}

dataset_label_map = {
    "mnist": "MNIST",
    "vgg_16": "CIFAR-10",
    "resnet": "PneumoniaMNIST"
}


metrics = [
    ("discrepancy", "Discrepancy"),
    ("ambiguity", "Ambiguity"),
    ("mean_vpr_width", "VPR"),
    ("rashomon_capacity", "Rash. Cap."),
    ("rashomon_ratio", "Rash. Ratio")
]



def to_tex(s: str) -> str:
    repl = {
        "ε": r"$\varepsilon$",
        "±": r"$\pm$",
        "%": r"\%",
        "σ": r"$\sigma$",
        "μ": r"$\mu$",
        "→": r"$\to$",
        "≥": r"$\ge$",
        "≤": r"$\le$",
        "_": r"",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


BASE_DATA_FOLDER = "cma_evaluations/"
BASE_BASELINE_FOLDER = "baseline_evaluations/"
METRICS = ["ambiguity", "discrepancy", "mean_vpr_width", "rashomon_ratio"]
MODELS_GENERATED_PER_Z_DIM = {2: 162, 4: 320, 8: 640, 16: 1284, 32: 2562, 64: 5120}
eps_list = [0.01, 0.02, 0.03, 0.04, 0.05]

def concatenate_all_results(model_type: str):
    data_folder = os.path.join(BASE_DATA_FOLDER, model_type)
    # Get all epsilon subfolders, ignore the rest
    epsilons = [file for file in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, file))]
    dataframes = []
    for epsilon in epsilons:
        for z_dim in [2, 4, 8, 16, 32, 64]:
            rashomon_metrics_folder = os.path.join(data_folder, epsilon + "_lambda_0.5", "z_" + str(z_dim), "rashomon_metrics")
            csv_files = glob.glob(os.path.join(rashomon_metrics_folder, "*.csv"))
            if not csv_files:
                print(f"No CSV files found in the folder: {rashomon_metrics_folder}")
            for file in csv_files:
                df = pd.read_csv(file)
                df["sigma"] = df["sigma"].astype(str).str.replace("sigma_", "", regex=False).astype(float)
                df["rashomon_ratio"] = (df["test_rashomon_size"] / MODELS_GENERATED_PER_Z_DIM[z_dim])
                df = df.rename(columns={"mean_vpr": "mean_vpr_width"})
                dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df
    

def get_dropout_results(model_type: str):
    data_folder = os.path.join(BASE_BASELINE_FOLDER, "dropout", model_type)
    epsilons = [file for file in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, file))]
    dataframes = []
    for epsilon in epsilons:
        epsilon_folder = os.path.join(data_folder, epsilon)
        csv_files = glob.glob(os.path.join(epsilon_folder, "*.csv"))
        n_models = [int(name.split("_")[-2]) for name in csv_files]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            n_models = int(csv_file.split("_")[-2])
            df["search_budget"] = n_models
            df.rename(columns={"vpr": "mean_vpr_width"}, inplace=True)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def get_retraining_results(model_type: str):
    data_folder = os.path.join(BASE_BASELINE_FOLDER, "retraining", f"retraining_{model_type}")
    epsilons = [file for file in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, file))]
    dataframes = []
    for epsilon in epsilons:
        epsilon_folder = os.path.join(data_folder, epsilon)
        csv_files = glob.glob(os.path.join(epsilon_folder, "*.csv"))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df["rashomon_ratio"] = df["num_models_in_rashomon"] / df["total_models_evaluated"]
            df.rename(columns={"vpr_width_mean": "mean_vpr_width", "total_models_evaluated": "search_budget", "disc": "discrepancy", "amb": "ambiguity", "rc_mean": "rashomon_capacity"}, inplace=True)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def get_best_overall(df: pd.DataFrame, metrics: list, epsilon: float):
    eps_df = df.loc[df["epsilon"] == epsilon].copy()
    if eps_df.empty:
        return None

    eps_df["overall_score"] = eps_df[metrics].mean(axis=1)
    # Sort by score (descending), break ties with rashomon_size if available
    sort_cols = ["overall_score"]
    asc = [False]
    if "test_rashomon_size" in eps_df.columns:
        sort_cols.append("test_rashomon_size")
        asc.append(False)

    best = eps_df.sort_values(by=sort_cols, ascending=asc).iloc[0]
    return best

def plot_metrics_vs_epsilon_multi_transposed(
    df,
    file_name,
    metrics=("discrepancy", "ambiguity", "mean_vpr_width", "rashomon_capacity", "rashomon_ratio"),
    dataset_col="dataset",
    dataset_order=None,
    line_by="z_folder",
    line_order=None,
    width=0.22,
    sharey_mode="cell",
    legend_y=1.05,
    legend_x=0.5,
    return_fig=False,
    fig_h_mult=0.5
):
    """
    Plot multiple metrics (rows) vs epsilon across multiple datasets (columns), with
    lines grouped by `line_by` and per-cell IQR error bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: "epsilon", `dataset_col`, `line_by`, and each metric in `metrics`.
    metrics : sequence[str]
        Metric column names to plot (one row per metric).
    dataset_col : str
        Column name holding dataset IDs.
    dataset_order : list[str] | None
        Order of datasets for columns. If None, inferred from df.
    line_by : str
        Column that defines separate lines within each subplot.
    line_order : list | None
        Optional explicit order for the `line_by` levels.
    width : float
        Half-span for horizontal dodge of different `line_by` levels at each epsilon.
    sharey_mode : {"column","row","cell"}
        Y-axis sharing scheme.
    legend_y : float
        Figure-coordinate Y for legend anchor; 1.0 is the top edge, >1.0 pushes above.
    legend_x : float
        Figure-coordinate X for legend anchor; 0.5 centers the legend.
    return_fig : bool
        If True, return (fig, axes); otherwise show the plot.
    """
    df = df.copy()

    # basic hygiene for string categories
    if line_by in df.columns and df[line_by].dtype == object:
        df[line_by] = df[line_by].astype(str).str.strip()

    # validate
    req = {"epsilon", dataset_col, line_by} | set(metrics)
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # layout
    if dataset_order is None:
        dataset_order = list(pd.unique(df[dataset_col]))
    n_rows, n_cols = len(metrics), len(dataset_order)

    eps_sorted = sorted(df["epsilon"].unique())
    x_base = np.arange(len(eps_sorted))

    sharey_kw = dict(sharex=True)
    if sharey_mode == "column":
        sharey_kw["sharey"] = "col"
    elif sharey_mode == "row":
        sharey_kw["sharey"] = "row"
    else:
        sharey_kw["sharey"] = False

    fig_w = 6.75
    fig_h = fig_w * (n_rows / n_cols) * fig_h_mult if n_cols > 0 else fig_w * 0.9
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True, **sharey_kw)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # sharey ranges
    if sharey_mode == "column":
        ylims = {
            ds: (np.nanmin(df.loc[df[dataset_col] == ds, list(metrics)].values),
                 np.nanmax(df.loc[df[dataset_col] == ds, list(metrics)].values))
            for ds in dataset_order
        }
    elif sharey_mode == "row":
        ylims = {m: (np.nanmin(df[m].values), np.nanmax(df[m].values)) for m in metrics}
    else:
        ylims = {}


    levels_global = list(pd.unique(df[line_by]))
    if line_order is not None:
        levels_global = [lvl for lvl in line_order if lvl in set(levels_global)]
    else:
        try:
            levels_global = sorted(levels_global)
        except TypeError:
            levels_global = sorted(levels_global, key=str)

    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(base_colors) < len(levels_global):
        cmap = plt.get_cmap('tab20')
        base_colors = base_colors + [cmap(i % cmap.N) for i in range(len(levels_global) - len(base_colors))]
    level_to_color = {lvl: base_colors[i] for i, lvl in enumerate(levels_global)}

    def _label(val):
        if line_by == "sigma":
            return to_tex(f"σ={val}")
        elif line_by == "z_dim":
            return to_tex(f"d={val}")
        else:
            return to_tex(str(val))


    grouped_cache = {}
    present_levels_any = set()
    for ds in dataset_order:
        ddf = df[df[dataset_col] == ds]
        if ddf.empty:
            grouped_cache[ds] = {}
            continue
        levels_here = [lvl for lvl in levels_global if lvl in set(ddf[line_by])]
        present_levels_any.update(levels_here)

        # symmetric offsets for the levels present in THIS dataset
        k = len(levels_here)
        offsets = np.linspace(-width, width, k) if k > 1 else np.array([0.0])
        lvl_to_offset = dict(zip(levels_here, offsets))

        grouped_cache[ds] = {
            m: ddf.groupby(["epsilon", line_by], as_index=False).agg(
                median=(m, "median"),
                q1=(m, lambda x: np.nanpercentile(x, 25)),
                q3=(m, lambda x: np.nanpercentile(x, 75)),
            )
            for m in metrics
        }
        grouped_cache[ds]["_levels"] = levels_here
        grouped_cache[ds]["_lvl_to_offset"] = lvl_to_offset

    for i, m in enumerate(metrics):        # rows: metrics
        for j, ds in enumerate(dataset_order):  # cols: datasets
            ax = axes[i, j]
            dcache = grouped_cache.get(ds, {})
            gb = dcache.get(m, pd.DataFrame())
            if gb is None or gb.empty:
                ax.set_visible(False)
                continue

            if sharey_mode == "cell":
                sub_all = df[df[dataset_col] == ds]
                ylims[(m, ds)] = (np.nanmin(sub_all[m].values), np.nanmax(sub_all[m].values))

            levels_here = dcache["_levels"]
            lvl_to_offset = dcache["_lvl_to_offset"]

            for lvl in levels_here:
                sub = gb[gb[line_by] == lvl].sort_values("epsilon")
                if sub.empty:
                    continue
                xi = np.array([eps_sorted.index(e) for e in sub["epsilon"]], dtype=float)
                x = xi + lvl_to_offset[lvl]

                ax.plot(
                    x, sub["median"], marker="o", linewidth=1.2, markersize=4,
                    color=level_to_color[lvl], label=_label(lvl)
                )
                yerr = np.vstack([sub["median"] - sub["q1"], sub["q3"] - sub["median"]])
                ax.errorbar(
                    x, sub["median"], yerr=yerr, fmt="none",
                    ecolor="gray", elinewidth=0.9, capsize=3, alpha=0.9, zorder=0
                )

            # cosmetics
            if i == 0:
                ax.set_title(dataset_label_map.get(ds, ds), fontsize=9)
            if j == 0:
                ax.set_ylabel(label_map.get(m, m), fontsize=9)

            ax.set_xticks(x_base)
            ax.set_xticklabels([str(e) for e in eps_sorted])
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

            # y-lims
            if sharey_mode == "column":
                ax.set_ylim(ylims[ds])
            elif sharey_mode == "row":
                ax.set_ylim(ylims[m])
            else:
                ax.set_ylim(ylims[(m, ds)])


    levels_for_legend = [lvl for lvl in levels_global if lvl in present_levels_any]
    if levels_for_legend:
        labels = [_label(lvl) for lvl in levels_for_legend]
        handles = [plt.Line2D([0], [0], color=level_to_color[lvl], marker="o", linewidth=1.2)
                   for lvl in levels_for_legend]
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(len(labels), 6),
            frameon=False,
            bbox_to_anchor=(legend_x, legend_y)  
        )

    fig.supxlabel(to_tex("Rashomon parameter (ε)"), fontsize=10)

    if return_fig:
        return fig, axes
    plt.savefig(f"{file_name}.pdf", bbox_inches="tight")
    plt.show()


def plot_baselines_vs_epsilon_multi_transposed(
    datasets,
    metrics,
    eps_list=(0.01, 0.02, 0.03, 0.04, 0.05),
    *,
    sharey_mode="cell",          
    fig_w=6.75,
    height_scale=0.4,           
    label_map=None,             
    dataset_label_map=None,      
    include_series=("cma", "dropout", "retraining"),  
    series_labels=None,          
    series_styles=None,          
    legend_x=0.5,
    legend_y=1.05,               
    legend_ncol=3,
    suptitle=None,
    get_series_fn=None,         
    save_path=None,            
    return_fig=False
    ):

    if label_map is None:
        label_map = {}
    if dataset_label_map is None:
        dataset_label_map = {}

    norm_metrics = []
    if len(metrics) > 0 and isinstance(metrics[0], (tuple, list)) and len(metrics[0]) == 2:
        norm_metrics = list(metrics)
    else:
        norm_metrics = [(m, label_map.get(m, m)) for m in metrics]

    datasets = list(datasets)
    n_datasets = len(datasets)
    n_metrics = len(norm_metrics)
    if n_datasets == 0 or n_metrics == 0:
        raise ValueError("`datasets` and `metrics` must be non-empty.")

    default_series_labels = {"cma": "DIVERSE", "dropout": "Dropout", "retraining": "Retraining"}
    if series_labels:
        default_series_labels.update(series_labels)
    series_labels = default_series_labels

    default_styles = {
        "cma":       {"linestyle": "-", "marker": "o", "linewidth": 1.2},
        "dropout":   {"linestyle": "-", "marker": "o", "linewidth": 1.2},
        "retraining":{"linestyle": "-", "marker": "o", "linewidth": 1.2},
    }
    if series_styles:
        for k, v in series_styles.items():
            default_styles.setdefault(k, {}).update(v)
    series_styles = default_styles


    base_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    series_keys = [s for s in ("cma","dropout","retraining") if s in include_series]
    if len(base_colors) < len(series_keys):

        cmap = plt.get_cmap('tab20')
        base_colors = base_colors + [cmap(i % cmap.N) for i in range(len(series_keys) - len(base_colors))]
    series_to_color = {s: base_colors[i] for i, s in enumerate(series_keys)}


    eps_list = list(eps_list)
    x = np.arange(len(eps_list))
    xticklabels = [str(e) for e in eps_list]

    # figure size
    fig_h = fig_w * (n_metrics / max(n_datasets, 1)) * height_scale

    # decide sharey
    sharey_kw = {"sharex": True}
    if sharey_mode == "column":
        sharey_kw["sharey"] = "col"
    elif sharey_mode == "row":
        sharey_kw["sharey"] = "row"
    else:
        sharey_kw["sharey"] = False

    fig, axes = plt.subplots(n_metrics, n_datasets, figsize=(fig_w, fig_h), constrained_layout=True, **sharey_kw)

    # ensure 2D axes array
    if n_metrics == 1 and n_datasets == 1:
        axes = np.array([[axes]])
    elif n_metrics == 1:
        axes = np.array([axes])
    elif n_datasets == 1:
        axes = axes.reshape(-1, 1)


    ylims = {}
    if sharey_mode == "column":
        for j, (ds_name, data) in enumerate(datasets):
            ymin, ymax = np.inf, -np.inf
            for (metric, _) in norm_metrics:
                cma, drop, retr = get_series_fn(metric, *data)
                for sname, series in (("cma", cma), ("dropout", drop), ("retraining", retr)):
                    if sname in include_series and series is not None:
                        s = np.asarray(series, dtype=float)
                        ymin = min(ymin, np.nanmin(s))
                        ymax = max(ymax, np.nanmax(s))
            ylims[ds_name] = (ymin, ymax)
    elif sharey_mode == "row":
        for i, (metric, _) in enumerate(norm_metrics):
            ymin, ymax = np.inf, -np.inf
            for (ds_name, data) in datasets:
                cma, drop, retr = get_series_fn(metric, *data)
                for sname, series in (("cma", cma), ("dropout", drop), ("retraining", retr)):
                    if sname in include_series and series is not None:
                        s = np.asarray(series, dtype=float)
                        ymin = min(ymin, np.nanmin(s))
                        ymax = max(ymax, np.nanmax(s))
            ylims[metric] = (ymin, ymax)



    present_series = set()


    for i, (metric, pretty) in enumerate(norm_metrics):
        for j, (dataset_name, data) in enumerate(datasets):
            ax = axes[i, j]

            cma, drop, retr = get_series_fn(metric, *data)
            series_data = {
                "cma": cma,
                "dropout": drop,
                "retraining": retr,
            }

            for sname in series_keys:
                yvals = series_data.get(sname, None)
                if yvals is None:
                    continue
                yy = np.asarray(yvals, dtype=float)
                if yy.size == 0:
                    continue
                style = dict(series_styles.get(sname, {}))
                ax.plot(x, yy, label=series_labels.get(sname, sname), markersize=4,
                        color=series_to_color[sname], **style)
                present_series.add(sname)

            if i == 0:
                ax.set_title(dataset_label_map.get(dataset_name, dataset_name), fontsize=9)
            if j == 0:
                ax.set_ylabel(pretty, fontsize=9)
            else:
                ax.set_ylabel("")

            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
            ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

            if sharey_mode == "column":
                ax.set_ylim(ylims[dataset_name])
            elif sharey_mode == "row":
                ax.set_ylim(ylims[metric])
            else:  # "cell"
                ymin, ymax = np.inf, -np.inf
                for sname in series_keys:
                    yvals = series_data.get(sname, None)
                    if yvals is None:
                        continue
                    s = np.asarray(yvals, dtype=float)
                    if s.size:
                        ymin = min(ymin, np.nanmin(s))
                        ymax = max(ymax, np.nanmax(s))
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ylims[(metric, dataset_name)] = (ymin, ymax)
                    ax.set_ylim(ymin, ymax)

    fig.supxlabel(to_tex("Rashomon parameter (ε)"), fontsize=10)
    if suptitle is not None:
        fig.suptitle(suptitle)

    if present_series:
        handles = []
        labels = []
        for sname in series_keys:
            if sname in present_series:
                line = plt.Line2D([0], [0],
                                  color=series_to_color[sname],
                                  **{k: v for k, v in series_styles[sname].items() if k in ("linestyle","marker","linewidth")})
                handles.append(line)
                labels.append(series_labels.get(sname, sname))
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(legend_ncol, len(labels)) if labels else 1,
            frameon=False,
            bbox_to_anchor=(legend_x, legend_y),
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if return_fig:
        return fig, axes
    plt.show()

def get_series(metric: str, df, dropout_df: pd.DataFrame, retraining_df: pd.DataFrame):
    cma, dropout, retraining = [], [], []
    for eps in eps_list:
        row = get_best_overall(df, ["discrepancy", "ambiguity", "rashomon_capacity",  "mean_vpr_width", "rashomon_ratio"], eps)
        if row is not None:
            best_val = float(getattr(row, metric))
            search_budget = MODELS_GENERATED_PER_Z_DIM[row.z_dim]
            mask = (dropout_df["epsilon"] == eps) & (dropout_df["search_budget"] == search_budget)
            retraining_mask = (retraining_df["epsilon"] == eps) & (retraining_df["search_budget"] == search_budget)
            dropout_val = float(dropout_df.loc[mask, metric].iat[0])
            retraining_val = float(retraining_df.loc[retraining_mask, metric].iat[0])
            cma.append(best_val)
            dropout.append(dropout_val)
            retraining.append(retraining_val)
    return cma, dropout, retraining

if __name__ == "__main__":
    mnist_df = concatenate_all_results("mnist")
    mnist_df["dataset"] = "mnist"
    vgg_df = concatenate_all_results("vgg16_cifar10")
    vgg_df["dataset"] = "vgg_16"
    resnet_df = concatenate_all_results("resnet50_pneumonia")
    resnet_df["dataset"] = "resnet"

    combined_df = pd.concat([mnist_df, vgg_df, resnet_df], ignore_index=True)

    mnist_retraining_df = get_retraining_results("mnist")
    mnist_dropout_df = get_dropout_results("mnist")
    vgg_retraining_df = get_retraining_results("vgg16")
    vgg_dropout_df = get_dropout_results("vgg16")
    resnet_retraining_df = get_retraining_results("resnet")
    resnet_dropout_df = get_dropout_results("resnet50")


    datasets = [("mnist", [mnist_df, mnist_dropout_df, mnist_retraining_df]), ("resnet", [resnet_df, resnet_dropout_df, resnet_retraining_df]), ("new_vgg_16", [vgg_df, vgg_dropout_df, vgg_retraining_df])]


    fig, axes = plot_baselines_vs_epsilon_multi_transposed(
        datasets=datasets,
        metrics=metrics,                    
        eps_list=[0.01, 0.02, 0.03, 0.04, 0.05],
        sharey_mode="cell",
        label_map={
            "discrepancy": "Discrepancy",
            "ambiguity": "Ambiguity",
            "mean_vpr_width": "VPR",
            "rashomon_capacity": "Rash. Cap.",
            "rashomon_ratio": "Rash. Ratio",
        },
        dataset_label_map={
            "mnist": "MNIST",
            "new_vgg_16": "CIFAR-10",
            "resnet": "PneumoniaMNIST",
        },
        include_series=("cma", "dropout", "retraining"), 
        legend_x=0.5,
        legend_y=1.04,
        suptitle=None,
        get_series_fn=get_series,           
        save_path="baselines.pdf",
        return_fig=True
    )


    plot_metrics_vs_epsilon_multi_transposed(
        combined_df,
        dataset_col="dataset",
        dataset_order=["mnist", "vgg_16", "resnet"],
        metrics=["discrepancy", "ambiguity", "mean_vpr_width", "rashomon_capacity", "rashomon_ratio"],
        sharey_mode="cell",
        line_by="z_dim",
        legend_y=1.04,
        fig_h_mult=0.4,
        file_name="metrics_vs_epsilon_transposed_zdim"
    )
    plot_metrics_vs_epsilon_multi_transposed(
        combined_df,
        dataset_col="dataset",
        dataset_order=["mnist", "vgg_16", "resnet"],
        metrics=["discrepancy", "ambiguity", "mean_vpr_width", "rashomon_capacity", "rashomon_ratio"],
        sharey_mode="cell",
        line_by="sigma",
        line_order=[0.1, 0.2, 0.3, 0.4, 0.5],
        legend_y=1.06,
        file_name="metrics_vs_epsilon_transposed_sigma",
    )

    plot_metrics_vs_epsilon_multi_transposed(
        combined_df,
        dataset_col="dataset",
        dataset_order=["mnist", "vgg_16", "resnet"],
        metrics=["discrepancy", "ambiguity", "mean_vpr_width", "rashomon_capacity", "rashomon_ratio"],
        sharey_mode="cell",
        line_by="z_folder",
        legend_y=1.06,
        file_name="metrics_vs_epsilon_transposed_zfolder",
    )


