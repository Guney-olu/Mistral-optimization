# Performance Optimzation

Kaggle notebook
`https://www.kaggle.com/code/ariyasaran/ml-mixtral-assign`

Running Inference Scripts

```sh
python3 Infernce_scripts/Base_script.py --model mistralai/Mistral-7B-v0.1 --prompt "I am batman" --output_length 1000
```

Run Tests scrits directly

```sh
python3 test.py
```


## Small Comparsion
*it is a very simple comparsion not taking in account many things*

![Image Alt text](Mistral-optimization/assests/rsz_vague_compar.png)

**Groq = 559 tokens/sec > Our's = 260 tokens/sec > fireworks = 251 tokens/sec**

![Image Alt text](Mistral-optimization/assests/rsz_our_toks.png)
