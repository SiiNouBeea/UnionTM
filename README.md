# Get Started

## 1. Download the Data

All datasets have been preprocessed and are ready for use. You can obtain them from their original sources[^1]:
[^1]: Note: I am not sure where the original data for the "wind" dataset came from, but I am using the publicly available dataset in TFB.

- **ETT**: [https://github.com/zhouhaoyi/ETDataset/tree/main](https://github.com/zhouhaoyi/ETDataset/tree/main)
- **Electricity, Weather**: [https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer?tab=readme-ov-file)
- **Solar**: [https://github.com/laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)
- **Wind**: [https://github.com/decisionintelligence/TFB](https://github.com/decisionintelligence/TFB)

For convenience, we provide a comprehensive package containing all required datasets, available for download from [Google Drive](https://drive.google.com/drive/folders/16kSkRg7lXtuqTfdhlQf5VaBNOFeqid2s?usp=sharing). You can place it under the folder [./dataset](./dataset/).

## 2. Requirements
To install all dependencies[^2]:
```bash
pip install -r requirements.txt
```
If you are using Anaconda, you can create a new Conda environment and run the following command.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
[^2]: Note: If you wish to run UnionTM on a GPU, please first run the following cmd command to install Pytorch in conjunction with CUDA.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If you are using Anaconda, you can create a new Conda environment and run the following command.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```


## 3. Train the Model

Experiment scripts for various benchmarks are provided in the [`scripts`](./scripts) directory. You can reproduce experiment results as follows:

```bash
bash ./scripts/ETT/etth1.sh                # ETTh1
bash ./scripts/Electricity/electricity.sh  # Electricity
bash ./scripts/Solar/solar.sh              # Solar-Energy
bash ./scripts/Weather/weather.sh          # Weather
bash ./scripts/Wind/wind.sh                # Wind
```



# Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.In fact, most of our work involves integrating their code and modifying them.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- SimpleTM (https://github.com/vsingh-group/SimpleTM)
- TimeMixer (https://github.com/kwuking/TimeMixer)
- Pathformer (https://github.com/decisionintelligence/pathformer)
- PatchTST (https://github.com/yuqinie98/PatchTST)
