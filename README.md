## Testing Pytorch Lightning

Dataset: [MNIST](https://www.kaggle.com/c/digit-recognizer)

### How to run:

Firstly please generate train & valid `.csv` files.

You can do this with following:

```python
>>> from pandas import read_csv
>>> from sklear.model_selection import train_test_split
>>>
>>> data = read_csv("train.csv")
>>> train, valid = train_test_split(data, test_size=0.1, random_state=2020)
>>>
>>> train.to_csv("data/train.csv", index=False)
>>> valid.to_csv("data/valid.csv", index=False)
```

Now you can run experiments with command like this:

```bash
python src/main.py \
--train_data data/train.csv \
--valid_data data/valid.csv \
--lr 0.001 \
--gpus 1
```
