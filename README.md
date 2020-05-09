# Recurrent Visual Attention

This is a **PyTorch** implementation of human-attention inspired image classification network. We test the dynamics of the agent, determin if it is driven by fractional dynamics. We are also rying to improve the perfomance.
## Requirements

- python 3.5+
- pytorch 0.3+
- tensorboard_logger
- tqdm

## Usage

The easiest way to start training your RAM variant is to edit the parameters in `config.py` and run the following command:

```
python main.py
```

To resume training, run the following command:

```
python main.py --resume=True
```

Finally, to test a checkpoint of your model that has achieved the best validation accuracy, run the following command:

```
python main.py --is_train=False
```

## References

- [Torch Blog Post on RAM](http://torch.ch/blog/2015/09/21/rmva.html)
