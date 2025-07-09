
## RTDeter Summary for Perception Developers

Below is Shahar's guide on how to use this repository for training and evaluating the model.

For additional tutorials and detailed usage instructions, refer to the official repository documentation:  
[RT-DETR PyTorch Tools README](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/tools/README.md)

### Getting Started - Easiest Way to Use RTDeter Repository (Docker Notes)

The easiest approach is to use Docker.

1. Navigate to `RT-DETR/rtdetrv2_pytorch/`.
2. Ensure the Docker Compose file is ready:
      - Set your wandb key.
      - Map the code and dataset locations.

Start Docker with:
```
docker compose up -d
```

You may also need to install `wandb` as a dependency in the Docker Python environment.

### Dataset Preparation

Ensure your dataset is in COCO format.  
There are several scripts for merging classes and removing redundant classes from COCO JSON files in the `alg_utils` folder.

### Dataset Preparation Tip

For experimentation, it's easier to use a small dataset. Use `algo_utils/shrink_coco_dataset.py` to reduce the dataset JSON file to a limited number of entries.

### wandb

Set the wandb key as an environment variable in the Docker Compose file.

### Training

You can start training easily from the terminal with the following command:

First, configure the desired model and dataset location in `rtdetrv2_r18vd_sp1_120e_coco_shahar.yml`.

Example launch configurations:
```json
{
       "name": "train RTDETRv2",
       "type": "debugpy",
       "request": "launch",
       "program": "tools/train.py",
       "console": "integratedTerminal",
       "env": {
              "CUDA_VISIBLE_DEVICES": "0"
       },
       "args": [
              "-c",
              "configs/rtdetrv2/rtdetrv2_r18vd_sp1_120e_coco_shahar.yml"
       ]
},
{
       "name": "train RTDETRv2 flir ir",
       "type": "debugpy",
       "request": "launch",
       "program": "tools/train.py",
       "console": "integratedTerminal",
       "env": {
              "CUDA_VISIBLE_DEVICES": "0"
       },
       "args": [
              "-c",
              "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco_flir_ir.yml"
       ]
}
```

### Inference - DNN Evaluation

Evaluate the DNN by running the `infer.py` script. This script has been modified to recursively process all `.jpg` files in a directory.

```json
{
      "name": "infer RTDETRv2",
      "type": "debugpy",
      "request": "launch",
      "program": "tools/infer.py",
      "console": "integratedTerminal",
      "env": {
            "CUDA_VISIBLE_DEVICES": "0"
      },
      "args": [
            "--config",
            "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco_flir_rgb.yml",
            "--resume",
            "output/rtdetrv2_r50vd_6x_coco_flir_ir/best.pth",
            "-f",
            "/dataset/images_thermal_val/data/video-zp8ed5vPKfAJ2fKWh-frame-006314-sBhdNtnTDtYk4afK5.jpg"
      ]
}
```


2. When Exporting or Performing Inference
To load trained weights for inference or export:

bash
Copy
Edit
python tools/export_onnx.py \
  -c configs/...yml \
  --resume output/your_model/best_model.pth
This loads the model weights without starting training.
