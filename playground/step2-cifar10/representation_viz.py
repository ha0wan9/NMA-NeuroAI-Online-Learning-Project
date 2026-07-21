"""Observational representation diagnostics and TensorBoard figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import CLASS_PAIRS


LAYER_NAMES = ("conv1", "conv2", "fc1", "logits")


def balanced_anchor_indices(dataset: Any, samples_per_class: int, seed: int,
                            torch: Any) -> list[int]:
    """Select fixed class-sorted anchors with an isolated generator."""
    generator = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(dataset.targets)
    selected: list[int] = []
    for label in range(10):
        candidates = (targets == label).nonzero(as_tuple=False).squeeze(1)
        order = torch.randperm(candidates.numel(), generator=generator)
        selected.extend(int(value) for value in candidates[order[:samples_per_class]].tolist())
    return selected


def learned_feature_modules(model: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    nn = runtime["nn"]
    modules = [module for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
    if len(modules) != len(LAYER_NAMES):
        raise RuntimeError(f"expected four learned feature layers, found {len(modules)}")
    return dict(zip(LAYER_NAMES, modules, strict=True))


def extract_features(model: Any, loader: Any, device: Any,
                     runtime: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Extract matched feedforward learned-layer outputs without changing state."""
    torch = runtime["torch"]
    captured: dict[str, list[Any]] = {name: [] for name in LAYER_NAMES}
    handles = []
    for name, module in learned_feature_modules(model, runtime).items():
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, layer=name: captured[layer].append(
                output.detach().flatten(1).cpu()
            )
        ))
    labels = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for inputs, batch_labels in loader:
                model(inputs.to(device))
                labels.append(batch_labels.cpu())
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return ({name: torch.cat(parts).numpy() for name, parts in captured.items()},
            torch.cat(labels).numpy())


def correlation_rdm(features: Any, np: Any) -> Any:
    values = np.asarray(features, dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)
    similarity = np.clip(values @ values.T, -1.0, 1.0)
    rdm = 1.0 - similarity
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_triangle(matrix: Any, np: Any) -> Any:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def safe_correlation(left: Any, right: Any, np: Any) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def rdm_summary(rdm: Any, labels: Any, np: Any) -> dict[str, float]:
    rows, cols = np.triu_indices(rdm.shape[0], k=1)
    distances = rdm[rows, cols]
    different = (labels[rows] != labels[cols]).astype(np.float64)
    within = distances[different == 0]
    between = distances[different == 1]
    return {
        "label_alignment": safe_correlation(distances, different, np),
        "within_class_dissimilarity": float(within.mean()),
        "between_class_dissimilarity": float(between.mean()),
        "class_separation": float(between.mean() - within.mean()),
    }


def rdm_drift(reference: Any, current: Any, np: Any) -> float:
    return 1.0 - safe_correlation(
        upper_triangle(reference, np), upper_triangle(current, np), np
    )


def classical_mds(rdm: Any, np: Any) -> Any:
    count = rdm.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (rdm ** 2) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))


def align_coordinates(reference: Any, current: Any, np: Any) -> Any:
    reference = reference - reference.mean(axis=0, keepdims=True)
    current = current - current.mean(axis=0, keepdims=True)
    left, _singular, right = np.linalg.svd(current.T @ reference)
    return current @ (left @ right)


def plot_rdm_grid(rdms: dict[str, Any], boundary: int) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(LAYER_NAMES), figsize=(13, 3.2), constrained_layout=True)
    for axis, name in zip(axes, LAYER_NAMES, strict=True):
        image = axis.imshow(rdms[name], cmap="magma_r", vmin=0.0, vmax=2.0)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.75, label="correlation dissimilarity")
    figure.suptitle(f"Class-sorted RDMs after task boundary {boundary}")
    return figure


def plot_mds_grid(coordinates: dict[str, Any], labels: Any, boundary: int) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(LAYER_NAMES), figsize=(13, 3.2), constrained_layout=True)
    for axis, name in zip(axes, LAYER_NAMES, strict=True):
        points = coordinates[name]
        scatter = axis.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=10, alpha=0.8)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scatter, ax=axes, ticks=range(10), shrink=0.75, label="CIFAR-10 class")
    figure.suptitle(f"MDS geometry aligned to initialization, boundary {boundary}")
    return figure


def class_centroids(coordinates: dict[str, Any], labels: Any, np: Any) -> dict[str, Any]:
    """Reduce fixed-anchor coordinates to one trajectory point per class."""
    classes = np.unique(labels)
    return {
        name: np.stack([points[labels == label].mean(axis=0) for label in classes])
        for name, points in coordinates.items()
    }


def plot_representation_trajectory_3d(history: list[dict[str, Any]], np: Any) -> Any:
    """Plot class-centroid representation drift in time × aligned-MDS space."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    steps = np.asarray([entry["step"] for entry in history], dtype=np.float64)
    for panel, name in enumerate(LAYER_NAMES, start=1):
        axis = figure.add_subplot(2, 2, panel, projection="3d")
        centroids = np.stack([np.asarray(entry["centroids"][name]) for entry in history])
        for class_id in range(centroids.shape[1]):
            axis.plot(
                steps, centroids[:, class_id, 0], centroids[:, class_id, 1],
                color=cmap(class_id), marker="o", markersize=3, linewidth=1.2,
                label=str(class_id) if name == LAYER_NAMES[0] else None,
            )
        axis.set_title(name)
        axis.set_xlabel("task checkpoint")
        axis.set_ylabel("aligned MDS 1")
        axis.set_zlabel("aligned MDS 2")
        axis.set_xticks(steps)
        axis.view_init(elev=24, azim=-58)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, title="CIFAR-10 class", loc="center right", ncol=1)
    figure.suptitle("3D representation trajectories: time × aligned MDS geometry")
    return figure


def plotly_representation_trajectory_3d(history: list[dict[str, Any]], np: Any) -> Any:
    """Build the interactive counterpart of the TensorBoard 3D snapshot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=LAYER_NAMES,
    )
    colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    )
    steps = [entry["step"] for entry in history]
    for panel, name in enumerate(LAYER_NAMES):
        row, column = divmod(panel, 2)
        centroids = np.stack([np.asarray(entry["centroids"][name]) for entry in history])
        for class_id in range(centroids.shape[1]):
            figure.add_trace(
                go.Scatter3d(
                    x=steps,
                    y=centroids[:, class_id, 0],
                    z=centroids[:, class_id, 1],
                    mode="lines+markers",
                    name=f"class {class_id}",
                    legendgroup=f"class-{class_id}",
                    showlegend=panel == 0,
                    line={"color": colors[class_id % len(colors)], "width": 4},
                    marker={"color": colors[class_id % len(colors)], "size": 4},
                    hovertemplate=(
                        "checkpoint=%{x}<br>MDS 1=%{y:.4f}<br>"
                        "MDS 2=%{z:.4f}<extra>class " + str(class_id) + "</extra>"
                    ),
                ),
                row=row + 1, col=column + 1,
            )
    scene_axes = {
        "xaxis_title": "task checkpoint",
        "yaxis_title": "aligned MDS 1",
        "zaxis_title": "aligned MDS 2",
    }
    figure.update_layout(
        title="Interactive 3D representation trajectories: time × aligned MDS geometry",
        height=900,
        scene=scene_axes,
        scene2=scene_axes,
        scene3=scene_axes,
        scene4=scene_axes,
        legend={"title": "CIFAR-10 class"},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    return figure


def write_plotly_trajectory(history: list[dict[str, Any]], np: Any, output: Path) -> Path:
    """Write a self-contained, browser-openable interactive 3D artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plotly_representation_trajectory_3d(history, np)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True)
    return output


def plotly_rdm_timeline(history: list[dict[str, Any]], labels: Any, np: Any) -> Any:
    """Build a four-layer RDM animation controlled by a checkpoint slider."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = np.asarray(labels)
    row_labels = np.broadcast_to(labels[:, None], (labels.size, labels.size))
    column_labels = np.broadcast_to(labels[None, :], (labels.size, labels.size))
    pair_labels = np.stack((row_labels, column_labels), axis=-1)

    def heatmaps(entry: dict[str, Any]) -> list[Any]:
        return [
            go.Heatmap(
                z=np.asarray(entry["rdms"][name]),
                coloraxis="coloraxis",
                customdata=pair_labels,
                hovertemplate=(
                    "row class=%{customdata[0]}<br>column class=%{customdata[1]}"
                    "<br>dissimilarity=%{z:.4f}<extra>" + name + "</extra>"
                ),
            )
            for name in LAYER_NAMES
        ]

    figure = make_subplots(
        rows=2, cols=2,
        subplot_titles=LAYER_NAMES,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    for panel, trace in enumerate(heatmaps(history[0])):
        row, column = divmod(panel, 2)
        figure.add_trace(trace, row=row + 1, col=column + 1)

    frame_names = [str(entry["step"]) for entry in history]
    figure.frames = tuple(
        go.Frame(
            name=frame_name,
            data=heatmaps(entry),
            traces=list(range(len(LAYER_NAMES))),
        )
        for frame_name, entry in zip(frame_names, history, strict=True)
    )
    slider_steps = [
        {
            "label": entry.get("label", f"Checkpoint {entry['step']}"),
            "method": "animate",
            "args": [[frame_name], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        }
        for frame_name, entry in zip(frame_names, history, strict=True)
    ]
    figure.update_layout(
        title="Interactive RDM drift across continual-learning checkpoints",
        height=920,
        coloraxis={
            "colorscale": "Magma_r",
            "cmin": 0.0,
            "cmax": 2.0,
            "colorbar": {"title": "correlation<br>dissimilarity"},
        },
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Time: "},
            "pad": {"t": 55},
            "steps": slider_steps,
        }],
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.0,
            "y": -0.08,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {
                        "fromcurrent": True,
                        "frame": {"duration": 700, "redraw": True},
                        "transition": {"duration": 150},
                    }],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }],
        margin={"l": 40, "r": 80, "t": 80, "b": 130},
    )
    for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        figure.layout[axis_name].update(title="anchor sample", scaleanchor=None)
    for axis_name in ("yaxis", "yaxis2", "yaxis3", "yaxis4"):
        figure.layout[axis_name].update(title="anchor sample", autorange="reversed")
    return figure


def write_plotly_rdm_timeline(history: list[dict[str, Any]], labels: Any,
                              np: Any, output: Path) -> Path:
    """Write a self-contained interactive RDM slider artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plotly_rdm_timeline(history, labels, np)
    figure.write_html(str(output), include_plotlyjs=True, full_html=True)
    return output


def plot_accuracy_matrix(matrix: list[list[float]], title: str) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                      color="white" if values[row, column] < 0.55 else "black", fontsize=8)
    axis.set_xlabel("evaluated task")
    axis.set_ylabel("training boundary")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="accuracy")
    return figure


def plot_drift_vs_forgetting(drift_by_task: dict[int, float], test_matrix: list[list[float]],
                             boundary: int) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    for task_id, drift in sorted(drift_by_task.items()):
        acquired = test_matrix[task_id + 1][task_id]
        accuracy_loss = acquired - test_matrix[boundary][task_id]
        axis.scatter(drift, accuracy_loss, s=55)
        axis.annotate(f"task {task_id}", (drift, accuracy_loss), xytext=(4, 4),
                      textcoords="offset points")
    axis.axhline(0.0, color="grey", linewidth=0.8)
    axis.set_xlabel("penultimate-layer RDM drift")
    axis.set_ylabel("accuracy loss since acquisition")
    axis.set_title(f"Representation drift vs forgetting, boundary {boundary}")
    return figure


def plot_stability_plasticity(payload: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    colors = {"bp": "tab:blue", "pc": "tab:orange"}
    for seed_result in payload["seed_results"]:
        points = {}
        for run in seed_result["runs"]:
            method = run["method"]
            x_value = run["metrics"]["prequential_accuracy"]
            y_value = -run["metrics"]["average_forgetting"]
            points[method] = (x_value, y_value)
            axis.scatter(x_value, y_value, color=colors[method], s=55,
                         label=method.upper() if seed_result is payload["seed_results"][0] else None)
            axis.annotate(str(seed_result["seed"]), (x_value, y_value), xytext=(3, 3),
                          textcoords="offset points", fontsize=8)
        if set(points) == {"bp", "pc"}:
            axis.plot([points["bp"][0], points["pc"][0]],
                      [points["bp"][1], points["pc"][1]], color="0.65", linewidth=1)
    axis.set_xlabel("plasticity: prequential accuracy")
    axis.set_ylabel("stability: negative average forgetting")
    axis.set_title("Paired stability–plasticity trade-off")
    axis.legend()
    return figure


def log_comparison_dashboard(payload: dict[str, Any], root: Path, label: str) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(root / label / "comparison"))
    writer.add_text("run/config", json.dumps(payload["protocol"], sort_keys=True, indent=2), 0)
    writer.add_figure("continual/stability_plasticity", plot_stability_plasticity(payload), 0,
                      close=True)
    writer.close()


class RepresentationTracker:
    """Track fixed-anchor geometry without influencing optimization."""

    def __init__(self, model: Any, anchor_loader: Any, device: Any,
                 runtime: dict[str, Any], writer: Any):
        self.model = model
        self.anchor_loader = anchor_loader
        self.device = device
        self.runtime = runtime
        self.writer = writer
        self.np = runtime["np"]
        self.labels = None
        self.mds_reference: dict[str, Any] = {}
        self.acquisition_reference: dict[int, dict[str, Any]] = {}
        self.trajectory_history: list[dict[str, Any]] = []
        self.rdm_history: list[dict[str, Any]] = []

    def record(self, boundary: int,
               test_matrix: list[list[float]] | None = None) -> dict[str, Any]:
        features, labels = extract_features(
            self.model, self.anchor_loader, self.device, self.runtime
        )
        if self.labels is None:
            self.labels = labels
        elif not self.np.array_equal(self.labels, labels):
            raise RuntimeError("representation anchor order changed")
        rdms = {name: correlation_rdm(values, self.np) for name, values in features.items()}
        self.rdm_history.append({
            "step": boundary,
            "label": f"Boundary {boundary}",
            "rdms": rdms,
        })
        summaries = {name: rdm_summary(rdm, labels, self.np) for name, rdm in rdms.items()}
        coordinates = {}
        for name, rdm in rdms.items():
            current = classical_mds(rdm, self.np)
            if name not in self.mds_reference:
                self.mds_reference[name] = current
            coordinates[name] = align_coordinates(self.mds_reference[name], current, self.np)
        centroids = class_centroids(coordinates, labels, self.np)
        self.trajectory_history.append({
            "step": boundary,
            "centroids": {name: values.tolist() for name, values in centroids.items()},
        })

        drift_by_task: dict[int, dict[str, float]] = {}
        for task_id in range(boundary):
            mask = self.np.isin(labels, CLASS_PAIRS[task_id])
            task_rdms = {name: rdm[self.np.ix_(mask, mask)] for name, rdm in rdms.items()}
            if task_id not in self.acquisition_reference:
                self.acquisition_reference[task_id] = task_rdms
            drift_by_task[task_id] = {
                name: rdm_drift(self.acquisition_reference[task_id][name], task_rdms[name], self.np)
                for name in LAYER_NAMES
            }

        if self.writer is not None:
            self.writer.add_figure("representations/rdm", plot_rdm_grid(rdms, boundary),
                                   boundary, close=True)
            self.writer.add_figure("representations/mds", plot_mds_grid(coordinates, labels, boundary),
                                   boundary, close=True)
            self.writer.add_figure(
                "representations/trajectory_3d",
                plot_representation_trajectory_3d(self.trajectory_history, self.np),
                boundary, close=True,
            )
            interactive_path = write_plotly_trajectory(
                self.trajectory_history,
                self.np,
                Path(self.writer.log_dir) / "artifacts" / "representation_trajectory_3d.html",
            )
            self.writer.add_text(
                "representations/trajectory_3d_interactive",
                f"Interactive Plotly artifact: `{interactive_path}`",
                boundary,
            )
            rdm_timeline_path = write_plotly_rdm_timeline(
                self.rdm_history,
                labels,
                self.np,
                Path(self.writer.log_dir) / "artifacts" / "rdm_timeline.html",
            )
            self.writer.add_text(
                "representations/rdm_timeline_interactive",
                f"Interactive Plotly artifact: `{rdm_timeline_path}`",
                boundary,
            )
            for name, summary in summaries.items():
                for metric, value in summary.items():
                    self.writer.add_scalar(f"representations/{name}/{metric}", value, boundary)
            for task_id, layer_values in drift_by_task.items():
                for name, value in layer_values.items():
                    self.writer.add_scalar(
                        f"representations/{name}/task_{task_id}_rdm_drift", value, boundary
                    )
            if test_matrix is not None and drift_by_task:
                penultimate = {task_id: values["fc1"] for task_id, values in drift_by_task.items()}
                self.writer.add_figure(
                    "representations/drift_vs_forgetting",
                    plot_drift_vs_forgetting(penultimate, test_matrix, boundary),
                    boundary, close=True,
                )
            self.writer.flush()

        return {
            "boundary": boundary,
            "layer_summaries": summaries,
            "rdm_drift_by_task": {
                str(task_id): values for task_id, values in drift_by_task.items()
            },
            "mds_class_centroids": {
                name: values.tolist() for name, values in centroids.items()
            },
        }
