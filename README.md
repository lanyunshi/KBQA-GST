# KBQA-GST 
This is a pytorch implementation of the KBQA-GST model in the paper "[Knowledge Base Question Answering with Topic Units](https://doi.org/10.24963/ijcai.2019/701)"

## Dependency 
The model is tested in python 3.7 and pytorch 1.3.0
> conda create -n kbqagst python=3.7\
> source activate kbqagst\
> pip install -r requirements.txt

## Prepare Data
First, create a folder `data/` under the current directory. Download the GloVe from [here](https://nlp.stanford.edu/projects/glove/) as
>data/glove.840B.300d.zip

Then, download and merge the pre-processed datasets ([ComplexWebQuestion](https://www.tau-nlp.org/compwebq) and [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763)) from [here](https://drive.google.com/drive/folders/18H2JXDFPWfe4WeSXVpf1lUxdpraB2OvW?usp=sharing) under the `data/` directory. The data contains questions, answers, results of topic entity linking, unit linking and the candidate paths based on the entity linking, unit linking respectively.

## Download Pre-trained models
Download the pre-trained models from [here](https://drive.google.com/drive/folders/1wO_1UCBKv1CsO9YoxV5QJAWl1phTvTH7?usp=sharing). Put the `saved-model/` under the current directory.

## Run the Pre-trained model
To run the pre-trained models, run `python code/RL_Runner.py --task 1` for WebQuestionsSP and `python code/RL_Runner.py --task 3` for ComplexWebQuestion, respectively. You can change the **argument** (deactivate training flag or change the model's name) of the loaded argument in the `code/RL_Runner.py` file. \
To generate the final prediction, change the argument in the file `code/GenerateFinalPredictions.py` and run `python code/GenerateFinalPredictions.py`.\
To evaluate the WebQuestionsSP via the official evaluation file, run `python code/eval.py`.

## Run a New Model
To run a new model, set the **argument** of the loaded argument in the `code/RL_Runner.py` file run `python code/RL_Runner.py --task 0`.

Please cite the papers if you use our data and code.
```
@inproceedings{ijcai2019-701,
  title     = {Knowledge Base Question Answering with Topic Units},
  author    = {Lan, Yunshi and Wang, Shuohang and Jiang, Jing},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {5046--5052},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/701},
  url       = {https://doi.org/10.24963/ijcai.2019/701},
}
```
Contact Yunshi Lan for any question.
