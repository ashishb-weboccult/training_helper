import os
import shutil
import glob

def get_next_run_name(runs_dir: str, base_name: str = "train") -> str:
    """Get the next available run name (e.g., 'train', 'train1', 'train2', etc.)."""
    try:
        existing_runs = [d for d in os.listdir(runs_dir) if d.startswith(base_name)]
    except FileNotFoundError:
        return base_name
    if not existing_runs:
        return base_name
    indices = [int(d[len(base_name):]) if d[len(base_name):].isdigit() else 0 for d in existing_runs]
    next_index = max(indices) + 1 if indices else 1
    return f"{base_name}{next_index}"

def get_next_version(project_dir: str, model_name: str) -> int:
    """Determine the next version number for the model."""
    existing_versions = []
    for dir_name in os.listdir(project_dir):
        if dir_name.startswith(f"{model_name}_v"):
            version_str = dir_name.split("_v")[-1]
            if version_str.isdigit():
                existing_versions.append(int(version_str))
    if not existing_versions:
        return 1
    return max(existing_versions) + 1

def train_yolo_model(
    model,
    data: str,
    project_name: str = "default_project",
    model_name: str = "default_model",
    base_output_path: str = "models/my_models/detection_models",
    **kwargs
) -> None:
    """
    Train a YOLO model with automatic versioning and save outputs.

    Args:
        model: Loaded YOLO model instance (e.g., from ultralytics).
        data (str): Path to data.yaml file specifying dataset configuration.
        project_name (str): Name of the project for organizing outputs.
        model_name (str): Base name of the model.
        base_output_path (str): Base directory for saving outputs (default is "models/my_models/detection_models").
        **kwargs: Additional keyword arguments passed to model.train(). See documentation for details.

    Returns:
        None

    Raises:
        Exception: If training fails, the runs folder is preserved for resuming.
    """
    # Construct project directory
    project_dir = os.path.join(base_output_path, project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Determine next version
    next_version = get_next_version(project_dir, model_name)
    versioned_model_name = f"{model_name}_v{next_version}"
    model_dir = os.path.join(project_dir, versioned_model_name)
    runs_dir = os.path.join(model_dir, "runs")
    model_utils_dir = os.path.join(model_dir, "MODEL_UTILS")
    model_weights_dir = os.path.join(model_dir, "MODEL_WEIGHTS")
    metrics_dir = os.path.join(model_dir, "METRICS")

    # Get next run name and set run folder
    next_run_name = get_next_run_name(runs_dir)
    run_folder = os.path.join(runs_dir, next_run_name)

    try:
        # Train the model
        model.train(
            data=data,
            project=runs_dir,
            name=next_run_name,
            **kwargs
        )

        # Post-processing after successful training
        for dir_path in [model_utils_dir, model_weights_dir, metrics_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Zip the run folder
        shutil.make_archive(
            os.path.join(model_utils_dir, versioned_model_name),
            'zip',
            run_folder
        )

        # Copy best.pt
        best_pt_path = os.path.join(run_folder, "weights", "best.pt")
        if os.path.exists(best_pt_path):
            shutil.copy(
                best_pt_path,
                os.path.join(model_weights_dir, f"{versioned_model_name}.pt")
            )

        # Copy metric files
        for file in glob.glob(os.path.join(run_folder, "*.png")) + glob.glob(os.path.join(run_folder, "*.csv")):
            shutil.copy(file, metrics_dir)

        # Delete the runs folder after successful training
        shutil.rmtree(runs_dir)

    except Exception as e:
        print(f"Training interrupted: {e}")
        # Runs folder remains for resuming