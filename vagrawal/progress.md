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

I ran the code with all the data as training with maximum length 50 at start before running till 150. The sharp peak is where I changed the dataset. It can completely learn(by overfitting) the smaller dataset. At least the model seems to be working.

![Iter2](images/iter-2.png)

Try to modularize codebase and make script based interface. Still not working.

# 18 - 19 June

Add proper strides in the encoding layer, and changed many small things. Also, the script is fully working now. Using script is more natural as I can just tweak the parameters by command line. Also, I did many other experiments and I could get around 35% WER and 10% CER in the evaluation set. I had not got a plot as there were many workarounds just to run the code. This is now fixed and I could now get plot for both training and validation data separately. I can attach the latest results by tomorrow.

# 21 - 22 June

Experimented with many options for training. Default options seems to give 30-40% WER, and 10-20% CER. The measurements are quite noisy for these measures between epochs. Training WER can get as low as 10%, so I plan to using wsj1 dataset too. On the code side there is not much progress, as the training time of the models are quite long, around 1 hours per epoch in GPU.

Also read many research papers in this time.

# 23 June

Added per file features normalization.

# 24 June

Added per speaker normalization and changed to using mel filterbank coefficients.

# 25 June

Started work on language model. Also I am getting 29.77% WER and 12.73 CER using the latest code.

![Iter3](images/iter-3.png)

# 26 June - 30 June

Stuck in some very fundamental problems with implementing LMs. I planned to use word based LM, but in doing so I need to iterate over every word and update probabilty for the word DAG every time it selects a space, and then in the beam search I have to store the current node for every beam, and adjust the probabilty score accordingly. It's too overwhelming to do that inside tensorflow. I can always move entire thing to plain python, but it will be too much back and forth movement as I need to store the state of RNNs every time model predicts next character.

That's why I am now starting with character based language model. As much I looked in other people's work, they are comparable in perplexity to trigram LMs. Best LM models seems to use embeded word based RNNs, which are equally difficult to use. The code is just for the progress and is not working.

# 1 July - 3 July

Made progress in language model. The model is still not working. I think I will have a working code and results in a day or two. Much of the thought in the model is done, the remaining part is to train a model and then use pre-trained model from then in the acoustic model's inference part.

# 4 July

Get away with self made LM. And add penalization in beam search.

Use best of best n in evaluation.

# 5 July

Modify the lm evaluation code in sphinxbase so that it scores every line in file.

Dump all the outputs and scores in evaluation for further processing. Also make eval only mode.

Removed the penalization part from tensorflow as this was not what we wanted, and I will just rely on custom code for dumped outputs.

Some results:

Beam width 10, Best of 1: WER: 29.9762102737, CER: 11.5430441109
Beam width 10, Best of 10: WER: 25.0155631528, CER: 9.39780356101
Beam width 34, Best of 1: WER: 32.4193209151, CER: 14.9511835159
Beam width 34, Best of 10: WER: 25.1306455237, CER: 9.8869262826

I have the dumps and the code to find LM score. Only problem is what to do with OOV words. And then I can do grid search to choose the parameters.

# 6 July

I did some more experiments for offline LM, but it seems an online LM needs to be built eventually for any respectable results. I have looked into few codes, and I think OpenFST can be used in my code by doing a little work.

# 7 July

Made quite some progress in LM. Unfortunately, not much to commit. I will very likely make things work by tomorrow.

# 8 July

Made basic skeleton for LM. It is taking so much time to complete this part due to many small things. Tensorflow is causing big problems.  Hopefully I will come up with working model by tomorrow.

I also took quite some time to understand various methods of decoding. If the model uses CTC, I could simply reuse the other tools completely by dumping FST and composing to the grammar and finding best path. In using attention, as the outputs depends on what is commited previously, so I have to use online LM.

# 9 July

I have many things about to write about the model and implementation, since I am hitting a major checkpoint. Will update tomorrow. I am just commiting my progress in LM for now.

# 10 July

Read various papers and thought about things I can do in the future.

# 11 July

Moved forward in packaging the code, will commit tomorrow. Also read many papers. Will also start running few experiments tomorrow.

# 12 July

Fixed many hanging pieces. Code quality is not very good now, but things seeems to work. Finally got LM to work completely, as I left it in a partially working state for quite some time. Commiting a fairly big update, which also includes many utilities for making FST in the format we need it.

# 14 July

There were still bugs in the LM part and I fixed them. Now it is producing possible and likely sentences. Also, Integrated LM in train. Now I am only training where there is no out of vocabulary words. Also did more modularization of the code.

# 15 July

Read and thought about many things regarding what we can do in CTC. Also started a small parallel CTC code. Will commit later.

# 16 July

Ran experiments with my code. Unfortunately, the network didn't seem to learn anything for now. Spent time finding bugs. I will need some more time to see what is wrong.

# 17 July

Another frustrating day spent in hunting for the bug. It required changing slowly from the confirmed code that can be trained to current. As it requires half an hour or so to see if there is any training beyond just the very basic, it takes so long to look for a bug.

But at least, I found two bugs and I can see the training can progress to much better point than before. Hopefully, I can completely train the model to at least accuracy acheived by previous code. I think then there is much scope to improve.

Earlier, I removed a dense layer which sits on top of decoding layer and replaced it with another LSTM with vocab size as output size as I needed to put LM on top of it. Unfortunately, I forgot that the output of LSTM has a finite range of pi. Hence, it was making training much more difficult as the lowest and the highest output probabilty can be in maximum certain ratio. Also, the attention with default parameters does not do what I thought it does. By default it does much more trickery than I was planning. It was wrong in my previous code too, but it worked. Still, I changed that too.

I am hopeful that by tomorrow I will have a good result from the current model.

# 18 July

Still unable to get the model to train. It does not seems to be due to bugs in the code but I suspect the attention mechanism used initially enabled it to train faster. The model is still being trained and there is still chance that I could get a good accuracy.

# 19 July

Some progress in training. Now I am pretraining with just short sentences, else the model just can't seem to learn the alighnments.

# 20 July

Finally crossed a respectable margin after so much effort. With very limited experiments, my inconclusive interpretation is that my model is hard to train if LM is supplied from the start of training. Maybe, it's because smaller gradient size, and can be solved by using more learning rate. So the current training was done in three steps. I also spent some time in reading my code and it's tensorflow implementation in some detail. I found the dropout used before also includes the selected previous symbol, which is not what I wanted and I fixed that. The code is running and I will report my first result with LM tomorrow.

I think I will be spending next couple of days writing documentation and packaging the code. Also I will switch from wsj0 to wsj1.

# 21 July

Some progress in code cleanup.

# 22 July

Training is still running, and while the decrease in WER is slow, it is consistent.

I spent some time unsuccessfully trying to use previously trained model with LM to get one good result with LM(or dictionary) as instructed. There were problems with variable names and character vocabulary.

I started writing documentation and the equations for the decoding and LM aspect of the model. It will be the first thing I will finish tomorrow.

# 23-24 July

Wrote a draft model description and thought more about the theoretical details of the model, after discussing with my mentor.

I am continuing to run experiments both using LM during training, and without it.

I am pushing some minor changes I did in few days, including correcting dropout.

# 25 July

I was hopeful to get some results today, but still couldn't get the model to train. Apart from that, I fixed some parts of the code.

# 26 July

As I failed to converge WER to a good place for more than a week, I have started the implementation of attention cutoff, which only allows the decoding layer to look near the mean of previous attention. It should ease the convergence of the loss.

# 27 July

Implemented first version of attention cutoff, but there still seems some bugs in it, so I am not pushing it just now.

# 28 July

I completed implementation of attention cutoff. In running for a limited time, I am getting much better results. I will post the results of the converged model tomorrow.

# 29 July

Got some results and I am working on refactoring the code and writing documentation, so that other people can test it easily. Progress is likely to be slow for the next two days because I will not be in my home.

# 30 July

I couldn't get any time today, so no progress today.

# 31 July - 1 August

I have been running the training for a few days. Here are the results with and without using LM:

Without LM: 23.7%
LM with weight 0.1: 23.1%
LM with weight 0.2: 24.9%

The model size is very big for training with only a small part of wsj0 dataset. The training set achieves WER of around 7%(without LM). I think if I run the trained model on wsj1 for a epoch, WER will considerably improve on the validation set. I will try this tomorrow.

Also, there is very little improvement with LM. It looks LM introduces heavy weight for longer sentences, and for that I am working on weight penalization for the compensation. I made a mistake in that, and I will test with different penalization values tomorrow. Another thing to note here is that it's not LM directly from transcriptions and transcriptions contains out of vocabulary words according to the LM.

I also spent time in refactoring the code to release it for others to easily test, and I am making more options available via command line interface.
