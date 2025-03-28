# GPT Language Model CLI

This project offers a simple command-line tool to train and run a GPT-style language model on text data.

## Project Features

* Train a GPT model (autregressive LM) on an input text dataset
* Evaluate the model by generating completions for a prompt
* Continuously generate text token by token with slight delays
* Engage in a “debate” mode to interact with the model in real time

## Installation

You can check `requirements.txt` for the list of needed dependencies (like torch, etc.). Example:

\`\`\`
pip install -r requirements.txt
\`\`\`

## Quickstart

To quickly debate with the model or watch the model generate a debate in real time, do the following:

1. Download the [pretrained model file](https://drive.google.com/file/d/1E2kCF9bDDKBVovaBiLT1ABxi7vHC7-U6/view?usp=sharing) and the [finetuned model file](https://drive.google.com/file/d/1--9tEL_zGvHhLfQttOkazoKAiZ8a5TrK/view?usp=sharing).  Move these both into the model folder.

2. Run 'python debate_gpt.py debate' from the root directory to debate the model, or 'python debate_gpt.py continuous' from the root directory to watch the model generate a debate.

See the instructions below to train your own model and to use the application in a more customized way. 

## Usage

Run the main file \`debate_gpt.py\` with a mode followed by options:

\`\`\`
python debate_gpt.py [MODE] [options...]
\`\`\`

### Example Commands

#### 1) Train Mode

\`\`\`
python debate_gpt.py train --input my_data.txt --save my_model.pth \
    --epochs 500 --context-size 256 --batch-size 32
\`\`\`

If you omit some required flags, the script prompts for them.

#### 2) Eval Mode

\`\`\`
python debate_gpt.py eval --input my_data.txt --load my_model.pth \
    --prompt "Hello world," --max-tokens 100
\`\`\`

#### 3) Continuous Mode

\`\`\`
python debate_gpt.py continuous --input my_data.txt --load my_model.pth \
    --prompt "Once upon a time" --max-tokens 50
\`\`\`

The model prints tokens one by one with a small delay.

#### 4) Debate Mode

\`\`\`
python debate_gpt.py debate --input my_data.txt --load my_finetuned_model.pth
\`\`\`

Interact with the model in a back-and-forth conversation loop.

## Other(Messy)

Several other notebooks that were used for this project are also included in the 'other' folder.  This code is a bit messy, but feel free to take a look!

## Notes

* For best results, run on a GPU-enabled environment.

Enjoy training your model or using it to generate text. 
