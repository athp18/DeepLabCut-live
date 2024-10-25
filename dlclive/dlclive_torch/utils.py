import os
import shutil
import tarfile
from pathlib import Path
from typing import Tuple, Optional, Union

import torch
from torch import nn
import ruamel.yaml
import numpy as np
import albumentations as A

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.models import PoseModel
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.data.dlcloader import DLCLoader


class ModelWrapper(nn.Module):
    """
    Wrapper to convert a DeepLabCut PyTorch model to a TorchScript model. 
    Since TorchScript excepts a tuple rather than a model dict, this class wrapps the model dict into a named tuple for easy exporting.

    Args:
        model: Trained DeepLabCut model
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        output_dict = self.model(x)
        return tuple(output_dict.values())


def load_model(
    cfg_path: Union[str, Path],
    shuffle: int = 1,
    trainset_index: int = 0,
    modelprefix: str = "",
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, dict, A.Compose]:
    """Load a trained DLC PyTorch model for inference.

    Args:
        cfg_path: Path to project config.yaml
        shuffle: Which shuffle to use
        trainset_index: Training set fraction index
        modelprefix: Optional model prefix
        device: Device to load model on

    Returns:
        model: Loaded model
        dlc_cfg: Model configuration
        transform: Inference transforms
    """
    # Read configs
    cfg = auxiliaryfunctions.read_config(cfg_path)

    # Load model through DLCLoader which handles configs
    dlc_loader = DLCLoader(
        config=cfg_path,
        shuffle=shuffle,
        trainset_index=trainset_index,
        modelprefix=modelprefix,
    )

    # Get snapshot path
    train_fraction = cfg["TrainingFraction"][trainset_index]
    if cfg["snapshotindex"] == "all":
        print("Snapshotindex set to 'all', using last snapshot")
        snapshot_index = -1
    else:
        snapshot_index = cfg["snapshotindex"]

    snapshots = get_snapshots(dlc_loader.model_folder)
    snapshot_path = snapshots[snapshot_index]

    # Build and load model
    model = PoseModel.build(dlc_loader.model_cfg["model"])
    state_dict = torch.load(snapshot_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()

    if device is not None:
        model = model.to(device)
    # Get inference transforms
    transform = build_transforms(dlc_loader.model_cfg["data"]["inference"])

    return model, dlc_loader.model_cfg, transform


def get_snapshots(train_folder):
    snapshot_names = [
        file.stem for file in train_folder.iterdir() if file.suffix == ".pt"
    ]

    assert len(snapshot_names) > 0, "No snapshots were found"
    return sorted(snapshot_names, key=lambda name: int(name.split("-")[1]))


def export_model(
    cfg_path: Union[str, Path],
    shuffle: int = 1,
    trainset_index: int = 0,
    snapshot_index: Optional[int] = None,
    iteration: Optional[int] = None,
    overwrite: bool = False,
    make_tar: bool = True,
    modelprefix: str = "",
    optimize: bool = True,
    example_input: Optional[torch.Tensor] = None,
    export_onnx: bool = False,
) -> Path:
    """Export PyTorch DLC model for inference.

    Args:
        cfg_path: Path to project config
        shuffle: Shuffle number to export
        trainset_index: Training set fraction index
        snapshot_index: Which snapshot to export (None uses config value)
        iteration: Active learning iteration (None uses config value)
        overwrite: Whether to overwrite existing export
        make_tar: Create .tar.gz archive
        modelprefix: Optional model prefix
        optimize: Export TorchScript model
        example_input: Example input tensor for tracing, defaults to (1,3,256,256)
        export_onnx: Also export ONNX model

    Returns:
        Path to export directory
    """
    # Read config
    cfg = auxiliaryfunctions.read_config(cfg_path)
    cfg["project_path"] = str(Path(cfg_path).parent)

    # Get iteration/snapshot
    cfg["iteration"] = iteration or cfg["iteration"]
    cfg["snapshotindex"] = snapshot_index or cfg["snapshotindex"]

    # Load model
    dlc_loader = DLCLoader(
        config=cfg_path,
        shuffle=shuffle,
        trainset_index=trainset_index,
        modelprefix=modelprefix,
    )
    # print("Model cfg", dlc_loader.model_cfg)

    # Setup export directory
    export_dir = Path(cfg["project_path"]) / "exported-models"
    export_dir.mkdir(exist_ok=True)

    sub_dir = f"DLC_{cfg['Task']}_{dlc_loader.model_cfg['net_type']}_iteration-{cfg['iteration']}_shuffle-{shuffle}"
    full_export_dir = export_dir / sub_dir

    if full_export_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Export directory {full_export_dir} exists. Set overwrite=True to overwrite."
        )
    full_export_dir.mkdir(exist_ok=True)

    # Load model state
    snapshots = get_snapshots(dlc_loader.model_folder)
    # print(type(dlc_loader.model_folder))
    snapshot_path = dlc_loader.model_folder / Path(
        snapshots[cfg["snapshotindex"]] + ".pt"
    )

    model = PoseModel.build(dlc_loader.model_cfg["model"])
    state_dict = torch.load(snapshot_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()

    # Save model config
    model_cfg = dlc_loader.model_cfg.copy()
    model_cfg["snapshot_path"] = str(snapshot_path)

    pose_cfg_path = full_export_dir / "pose_cfg.yaml"
    ruamel_file = ruamel.yaml.YAML()
    ruamel_file.dump(model_cfg, open(pose_cfg_path, "w"))

    # Copy snapshot
    shutil.copy2(snapshot_path, full_export_dir)

    # Export optimized model
    if optimize:
        model = ModelWrapper(model)
        if example_input is None:
            example_input = torch.randn(1, 3, 256, 256)

        try:
            scripted_model = torch.jit.trace(model, example_input)
        except RuntimeError as e:
            print(f"Strict tracing failed with error: {e}. Retrying with strict=False.")
            scripted_model = torch.jit.trace(model, example_input, strict=False)
            print("Tracing successful with strict=False.")
        script_path = full_export_dir / "model_scripted.pt"
        scripted_model.save(str(script_path))

        # Export ONNX if requested
        if export_onnx:
            onnx_path = full_export_dir / "model.onnx"
            torch.onnx.export(
                model,
                example_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch"},
                },
            )

    # Create tar archive
    if make_tar:
        tar_path = full_export_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(full_export_dir, arcname=full_export_dir.name)

    print(f"Model exported at {str(full_export_dir)}")
    return full_export_dir


def generate_colors(num_colors: int) -> np.ndarray:
    """Generate distinct colors for each body part."""
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors
