
### 1. Environment

Runtime Environment Dependency: `requirements.txt`。

### 2. Run
The function `main.py` is executed through the script `run.sh` and `main.py` contains data processing and model prediction.

The data processing functions are in the `process_input()` function in `main.py` and the model prediction is in `tasker.test()`.

The default is to execute the test set directly, just run the script directly via `bash run.sh`. No modifications are required!


### 3. Others
The trained model weights are saved in `output\DeBERTa-v2-97M-Chinese`.

Note: This weight is for `All-possible MCQs`, `single MCQs` cannot be uploaded due to space limitation.

### 4. Run Result Output Path

The results of the model predictions are stored in the `row_submit.txt` file in `main_all_sort_train` in the `temp` directory.

### 5. Data & Model

#### 1.Data process

Process the reference books provided by JEC-QA
Get the corpus to be retrieved `final.txt`.

#### 2. Model Training

The implementation of this project is based on huggingface, and the training parameters of the model, except for those explicitly specified in the code (`main.py`).
The rest of the parameters are the same as [TrainingArgument](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) in huggingface "TrainingArgument")
parameter in huggingface.

#### 3. Model Structure
For more information, please see the paper.

### 4. Code and Checkpoint
https://drive.google.com/drive/folders/14GK85CeFjPaJRPnJ-FWLn1icMP-XMK-S?usp=sharing
