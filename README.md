# Cybertron

## Setup
```shell script
cd data_gen
yarn install
conda create -n cybertron python=3.8
conda activate cybertron
pip install -r requirements.txt
```

## Data Generation
Download the files [here](https://github.com/L1NNA/CybertronCCS/releases/tag/data_release) and place under `data_gen/data`
```shell script
python -m data_loader --h
python -m data_loader --data Exp_all_m1
```

## Model Training
We have created a python module to train and evaluate the model.
- Run models
  ```shell script
  python -m cybertron -h
  ```
  To train a specific model,
  ```shell script
  python -m cybertron --model [model_name] train --data [dataset_name] --epoch [num_of_epochs]
  or
  python -m cybertron train -h
  ```
- Using scripts
  ```shell script
  bash scripts/train_rl.sh
  ```

## Models

| Model       | File location                    |
|-------------|----------------------------------|
| AST2Vec     | ./data_gen/AST2Vec.js                 |
| Deobfuscation     | ./data_gen/deivfyscate.js                 |
| GRUEncoder     | ./cybertron/GRUC.py               |
| Transformer     | ./cybertron/Transformer.py               |
| SkipLSTM     | ./cybertron/SkipLSTM.py               |
| LeapGRU     | ./cybertron/LeapGRU.py               |
| CodeGemma     | ./llm.ipynb              |
| Cybertron     | ./cybertron/RLModelKL.py              |


## Running Tests
```
python cybertron/test_rl_model.py
python data_loader/test_generator.py
python data_loader/test_tfr.py
```
