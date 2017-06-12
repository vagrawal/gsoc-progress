## Before 6 June

I extracted features from I adapted a basic bidirectional LSTM model in tensorflow for the speech recognition task.

## 6 June

Experiment with WER

## 7 June

Fixed many of the bugs involved in the model. Now, I can successfully build the model. There is still some problem for the input, and it took much of my 2-3 days to fix many small bugs in the code, and make it compile. Expecting to find WER of the current basic attention based model by tomorrow.

## 8 June

Finally got my model to work after fixing so many bugs. Currently, it is single layer bidirectional attention based LSTM. I plan to experiment with complex models now.

## 10 June

Add code for checkpointing the weights and fix many things including WER

# 11-12 June

Made the training and evaluation script to run on the cloud. After training for more than 24 hours using GPU, I think the model has still not converged. I am getting WER of around 94% and CER of 72%. I think I need to use beam search instead of greedy search to improve the results. Also attaching the loss over time

![loss-over-time](images/loss-iter1.png)
