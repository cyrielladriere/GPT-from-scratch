# GPT-from-scratch
Builds a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. This is a decoder-only transformer that generates Shakespearean text at a character-level, trained on the tiny Shakespeare dataset (available at `artifacts/input.txt`).

There is a demo notebook available in `demo.ipynb`. This notebook tries to explain some of the code used in the final GPT model step-by-step, as well as going over the fundamental mathematical trick needed for calculating self-attention.

# Environment

This codebase was developed in Python 3.10. Dependancies can be installed as follows:
```sh
pip install -r requirements.txt
```

# Training the model

The model itself is defined in `model.py` and contains ~10M parameters. The code to train the GPT is available in `train.py` and can be run by executing:
```sh
python train.py
```

This will initiate 5000 epochs of training, after which the model will be saved in `artifacts/model.pt` (model is not available in this repo because the size is too large). If you want to modify any hyperparameters, you can do this in the `config.py` file.

# Generating Shakespearean text

After training and saving the model at `artifacts/model.pt`, we can use it to actually generate Shakespearean text. This model inference can be invoked by running the following command:
```sh
python inference.py
```

This script writes some output to the terminal (~500 tokens), but also writes a large body of text to the `artifacts/output.txt` file (~10 000 tokens). Currently, this output file contains text generated by a model I trained for 5000 epochs, feel free to check it out!

# Example
A short sample of a possible model output:
```txt
GLOUCESTER:
Resolve it to you thus? I hope my husband.
Why, come, my lords, the peace took of Paul:
Have to meddle court'sy. My victory!
Look down to me, but see, at Clarence!

HASTINGS:
My lord, our hopesty beards are well.

GLOUCESTER:
Cousin of Clarence, brother, and they are held.
I know the news, my lord, see the oath unto
The father full of peril of what thou wast saw
This bellkman cast hath residenced the time of the sea,
Great All-Souls' transgred was aside.
```
Even though the generated text cannot be classified as coherent or correct English, you can still see major improvements over randomly generating characters. The generated text also resembles the Shakespearean pretty well.