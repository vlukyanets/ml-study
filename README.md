# ML Study

Required Python 3.x to launch

## Digits Recognition (manual)

Manual implementation of neural network that recognize digits from images 8x8 pixels.
Image is 8x8 array with values `[0..15]` representing gray level of pixel.

Launch:
```
python3 digits_recognition.py [--iterations X] [--hide-plots] [--nn-save-to FILE] [--nn-load-from FILE]
```

Parameters:
```
--iterations X      - set how much iterations of training/checking should be performed (default 30)
--hide-plots        - do not show figures with pass rate and sum of square error on check (default: show)
--nn-save-to FILE   - if specified, save network after training to text file
--nn-load-from FILE - if specified, load network from text file instead of training
```
