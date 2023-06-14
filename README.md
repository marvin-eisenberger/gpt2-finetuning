
# GPT-2 Finetuning

A short sample implementation on how to fine-tune the GPT-2 model on a small text corpus (here: all 7 Harry Potter novels in `harrypotter.txt`). The implementation is based on the [Huggingface transformers library](https://huggingface.co/docs/transformers/index).

> Harry Potter was my father. My first choice
>
> -- <cite>GPT-2</cite>

## Dependencies

There are two main dependencies, both of which can be installed with pip. 
* optional:
```bash
conda create -n gpt2_finetuning python=3.8
conda activate gpt2_finetuning
```
* required:
```bash
pip install torch
pip install transformers[torch]
```
* If a GPU is available, follow this [link](https://pytorch.org/get-started/locally/) to install PyTorch with cuda.

## Running the code

We provide three separate scripts:
* `python3 gpt2_apply.py`: This simply loads a pre-trained (general-purpose) GPT-2 model and generates some text with a Harry Potter themed prompt. The results are reasonable, but not optimal. 
* `python3 gpt2_train.py`: This script fine-tunes the same model, by training on the data in `harrypotter.txt` for one epoch.
* `python3 gpt2_test.py`: The test script is the evaluation script corresponding to `gpt2_train.py`. 
  * One option is to apply it to the checkpoints produced by `gpt2_train.py` in the folder `checkpoint_0/`. 
  * Alternatively, pretrained checkpoints can be downloaded from [here](https://drive.google.com/file/d/1JKgeUMya9EbC0bwnLpvwYjdQtquvUOmC/view?usp=sharing).

## Results
Here are two quotes from the resulting text:
* Quote 1:
> They had found a place in the living room, which seemed to sit on the edge of Snape's office window. Hermione was pulling out her wand as soon as the door opened. She was flapping it with a twinkle of a hand into the sky as we walked down the hallway: Snape was in his bed, talking to his friends, and his arm was in his pocket. It was still twined about his wrist - and Harry could not tell if he was trying to read the paper. Ron kept thinking he could have just dropped Snape off at a distant address in his own living room.
> 
> "He'd love one that wouldn't be so dull."
> 
> It struck Harry that Snape wasn't much for writing - he was known only in private for being the best wizard in the whole world; and yet the two of them had never met. Hermione looked at Harry with the same blank eyes as she had never at Hogwarts, which suggested that whatever she used was not as accurate, on earth as her own thought. Instead it seemed something in his dreams but far, far too complex; perhaps his own dream
* Quote 2:
> Professor Johnson, in effect, had done so, his face as usual.
> 
> "I was quite sure we would live to see them all back on our feet," he said, sitting down again and bending down at once to the wood, listening and looking around. "We'd still find a small village. But I did tell you about their history. Yes, I know they came here from the other great Dementors, and they seemed like very decent people, but the Dementors don't have an ideal life."
> 
> "That's not what I meant when I said this," Professor Johnson corrected and turned to Harry, who had just sat straight back back across the fire. "I meant, you see... I don't think my mother was the first one to come here, not even the Prophet, I mean."
> 
> "You don't know her?" said Harry angrily. "I'm not surprised; I was telling her, not me."
* Additional sample outputs are provided in `sample_outputs.txt`.
