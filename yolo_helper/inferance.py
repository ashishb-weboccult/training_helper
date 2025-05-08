from ultralytics import YOLO
from pathlib import Path
import os
import shutil
from IPython.display import Image, display

def model_inferance(
    source: str,
    project_name: str,
    model_name: str,
    version: int,
    data_saving_dir: str,
    base_output_path: str,
    **predict_kwargs
):
    """Run YOLO prediction with task-specific handling.

    Args:
        source (str): Path to the input image, video, or directory.
        project_name (str): Name of the project.
        model_name (str): Name of the model.
        version (int): Model version number.
        data_saving_dir (str): Directory name for saving data.
        base_output_path (str): Base path for output.
        **predict_kwargs: Additional arguments for model.predict().

    Raises:
        FileNotFoundError: If model weights are not present at the constructed directory.
    """
    versioned_model_name = f"{model_name}_v{version}"
    model_path = os.path.join(base_output_path, project_name, versioned_model_name, "MODEL_WEIGHTS", f"{versioned_model_name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights are not present at given directory: {model_path}")
    
    model = YOLO(model_path)
    task = model.task
    is_single_image = os.path.isfile(source)

    if is_single_image:
        predict_kwargs["save"] = True  # Save images for all tasks
        results = model.predict(
            source=source,
            **predict_kwargs
        )
        # Default run directory based on task
        run_base = Path(f"runs/{task}")
        if run_base.exists():
            latest_folder = max(run_base.glob("predict*"), key=os.path.getmtime)
            predicted_image_path = latest_folder / os.path.basename(source)
            if predicted_image_path.exists():
                display(Image(filename=str(predicted_image_path)))
                shutil.rmtree(latest_folder)
            else:
                print("Predicted image not found.")
        else:
            print("No prediction output found.")
    else:
        predict_kwargs["save"] = predict_kwargs.get("save", True)
        output_dir = os.path.join(base_output_path, project_name, versioned_model_name, "INFERENCE", data_saving_dir)
        os.makedirs(output_dir, exist_ok=True)
        results = model.predict(
            source=source,
            project=Path(output_dir).parent.parent,
            name=Path(output_dir).relative_to(Path(output_dir).parent.parent),
            exist_ok=True,
            **predict_kwargs
        )