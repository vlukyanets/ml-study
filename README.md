# ML Study

Required Python 3.x to launch

## Digits Recognition (manual)

Manual implementation of neural network that recognize digits from images 8x8 pixels.
Image is 8x8 array with values `[0..15]` representing gray level of pixel. Set `iterations` to `0` to disable
neural network learning (e.g. when you load ready network from file)

Launch:
```
python 3digits_recognition.py [-h] [--iterations ITERATIONS] [--check] [--hide-plots] [--nn-save-to NN_SAVE_TO]
                              [--nn-load-from NN_LOAD_FROM] [--learning-density LEARNING_DENSITY]
                              [--learning-coefficient LEARNING_COEFFICIENT] [--disable-dynamic-learning-coefficient]
```

Parameters:
```
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        NN learning iterations (default 30)
  --check               Perform check of NN
  --hide-plots          Do not show figures with ML stats
  --nn-save-to NN_SAVE_TO
                        If specified, save network after training to text file
  --nn-load-from NN_LOAD_FROM
                        If specified, load network from text file instead of training
  --learning-density LEARNING_DENSITY
                        How much dataset should be used in one iteration to learn (default 0.2)
  --learning-coefficient LEARNING_COEFFICIENT
                        Learning coefficient (default 0.1)
  --disable-dynamic-learning-coefficient
                        Disable decreasing learning coefficient linearly to zero at last iteration
```

## Digits Generation (manual)

Manual implementation of neural network that generation 8x8 pixels image of some digit. Image displayed in
Qt window by 16 different colors (from white to gray). Set `iterations` to `0` to disable neural network learning
(e.g. when you load ready network from file)

Launch:
```
python digits_generation.py [-h] [--iterations ITERATIONS] [--draw DRAW] [--nn-save-to NN_SAVE_TO]
                            [--nn-load-from NN_LOAD_FROM] [--learning-density LEARNING_DENSITY]
                            [--learning-coefficient LEARNING_COEFFICIENT] [--disable-dynamic-learning-coefficient]
```

Parameters:
```
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        NN learning iterations (default 30)
  --draw DRAW           Perform check of NN
  --nn-save-to NN_SAVE_TO
                        If specified, save network after training to text file
  --nn-load-from NN_LOAD_FROM
                        If specified, load network from text file instead of training
  --learning-density LEARNING_DENSITY
                        How much dataset should be used in one iteration to learn (default 0.2)
  --learning-coefficient LEARNING_COEFFICIENT
                        Learning coefficient (default 0.1)
  --disable-dynamic-learning-coefficient
                        Disable decreasing learning coefficient linearly to zero at last iteration
```
