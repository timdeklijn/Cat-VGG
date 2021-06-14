# VGG

"From scratch" implementation of a VGG Neural network. Train it two classify
images with either cat 1 (Maz) or cat 2 (Rey).

## Literature

* [paper](https://arxiv.org/pdf/1409.1556.pdf)

## Export dependencies to requirements.txt

``` shell
poetry export --without-hashes -o requirements.txt
```

## Logging in to Weights and Biases

The Weights and Biases API key is placed in a `.env` file and in docker compose
this is set as a environment variable. The `.env` file:

```
WANDB_API_KEY=<API_KEY>
```
