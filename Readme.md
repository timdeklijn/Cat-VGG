# VGG

"From scratch" implementation of a VGG Neural network. Train it two classify
images with either cat 1 (Maz) or cat 2 (Rey).

## Literature

* [paper](https://arxiv.org/pdf/1409.1556.pdf)

## Export dependencies to requirements.txt

``` shell
poetry export --without-hashes -o requirements.txt
```

## Notes

It is impossible(??) to connect the training in a container to a
localhost tracking server. So I need to create a docker_compose file to
fix this.


