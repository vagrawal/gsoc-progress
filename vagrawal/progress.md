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

# 13 June

Completely relook the architecture, and change many things. Used many bidirectional layers, and used pooling between those. Also made many corrections like logging CER and WER over time. Running the architecture is currently in progress.

# 14 - 15 June

I added beam search decoder instead of the greedy decoder which was present. Beam search is present is still unreleased tensorflow 1.2, and it changes many of the API is seq2seq. Still, I cannot get a better WER. I am currently experimenting with incrementally increasing the maximum length of the output, as I think the model has some trouble learning the alignments.

# 15 June

Finally found the mistake after debugging for so long. I was using targets aligned directly to the logits, and so tensorflow just predicts the same character. The right way to do is to shift targets one place right. I expect to get results with good error rates now.

# 16 June

Try to modularize codebase and make script based interface. Still not working.
