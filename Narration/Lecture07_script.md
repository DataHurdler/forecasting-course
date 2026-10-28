# Lecture 7 — Recording Script

**ECON 8310: Business Forecasting · Introduction to Neural Networks**

Deck: `Slides/Lecture07_NeuralNets.pdf` (23 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- This lecture opens the second half of the course, and it has an unusual job: it teaches
  machinery that will **lose** on our data. Do not oversell it. The honesty on slide 4 is what
  makes Lectures 8 and 9 land, and it is what students remember.

---

# ▶ SLIDE 1 — Title page

This is Lecture 7, an introduction to neural networks.

It's the start of the second half of the course, and I want to set expectations before we begin,
because this topic arrives with more hype attached than anything else we'll cover.

For the last three weeks you've been building tree models and regularized regression: decision
trees in Lecture 4, random forests and gradient boosting in Lecture 5, and Ridge and LASSO last
week, in Lecture 6. Those work extremely well on the kind of data we have. Today we start on neural
networks, and by the end of the next two lectures you'll have fitted several of them.

**[pause]**

Here's what I want you to hold on to. Nothing today makes your XGBoost model obsolete. On our data,
the networks are going to lose. What today buys you is the machinery, and the machinery matters
because of what we do with it in Lectures 8 and 9, on convolutional networks and on sequence
models. That's where the *architecture* starts carrying assumptions that a flat table can't.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in five parts. First, why you'd go beyond trees at all, and when you shouldn't.
Second, the feedforward network itself: its layers, its width and depth, and how quickly the
parameter count runs away from you. Third, training, which means the loss, backpropagation and the
update step. Fourth, PyTorch, which comes down to four pieces you'll use in every deep learning
assignment for the rest of the semester. And last, where a network actually belongs in the toolkit
you now have.

---

# ▶ SLIDE 3 — Section divider: From Trees to Neural Networks

*(Advance and read the subtitle aloud. It's the thesis of the whole lecture.)*

Trees predict in steps, and networks predict in curves. That difference buys you less than you'd
expect on tabular data, and it buys you everything on sequences.

---

# ▶ SLIDE 4 — Why Go Beyond Trees?

There are two honest reasons to go beyond trees, and one honest warning.

The first reason is that a tree-based model treats every row as independent. Order isn't part of
the model. Everything a random forest knows about *time*, like the fact that last week matters, or
that fifty-two weeks ago matters, is something you put there by hand, as lag columns and rolling
means. If you shuffled the feature columns, the model would be identical.

The second reason is that a tree's prediction is **piecewise constant**. Remember Lab 4, on
decision trees, where the tree couldn't predict above the largest value it had ever seen? That
isn't a bug in our implementation. It's what a tree is. A tree averages the training values in a
leaf, so it can't extrapolate. A neural network's prediction is continuous instead. With the ReLU
activation you're about to meet, it's made of straight pieces joined at kinks, and you'll see
exactly that two slides from now. Past its last kink it just keeps going in a straight line, so it
can extrapolate.

**[pause]**

Now the warning, and I want to be blunt about it.

*(Point at the warning box.)*

This is **not an upgrade**. On the tabular problems in Homework 3 and Homework 4, gradient boosting
and a tuned LASSO were hard to beat, and a network will often lose to both. You reach for a network
when the data is genuinely a sequence, when the raw input is an image or audio, or when your sample
size is very large. Otherwise, you're paying in tuning time for accuracy you won't get.

So today is about the machinery. Lectures 8 and 9 are where it earns its place.

---

# ▶ SLIDE 5 — From Linear Regression to a Single Neuron

Let's build the thing from something you already know.

Linear regression predicts y-hat equals w-transpose x, plus b. That's a weighted sum of your
predictors, plus an intercept, and you've been fitting it since Lecture 2, in the second week.

A **neuron** is that exact expression with one addition: a nonlinear function wrapped around the
outside. We call that function the activation, and we write it as sigma.

So a equals sigma of the quantity w-transpose x plus b. That's the entire unit.

*(Walk the three panels from left to right.)*

These are three activations you'll meet, each drawn with its slope marked. ReLU, on the left, is
the maximum of zero and z, and it's the default for hidden layers. Its slope is zero on the left
and exactly one on the right, with nothing in between. The sigmoid, in the middle, squashes its
input to between zero and one, and you'll see it again in Lecture 9 as a gate inside an LSTM. Look
at the dashed line. Its steepest slope, right at zero, is one quarter, and everywhere else it's
flatter than that. Tanh, on the right, squashes to between minus one and one, and it shows up in
the states of sequence models. Its steepest slope is one.

Hold on to that one quarter. When a gradient flows backward through a sigmoid, it gets multiplied
by that slope, so each sigmoid scales it by a quarter at best. Stack a few of them, and the signal
is gone. That's the vanishing gradient, and it's the whole story of Lecture 9.

There's a fourth activation that isn't drawn, because there's nothing to draw. It's the **linear**
one, which just returns z. That's your output layer when you're doing regression, because you don't
want to squash a forecast.

**[pause]**

ReLU is worth a sentence of intuition, because it makes the whole thing concrete. The neuron is
asking a question: is this pattern present? If the evidence is positive, it passes a signal
through, in proportion to how positive it is. If the evidence is negative, it outputs exactly zero
and stays silent. That's a detector.

And that "exactly zero" is doing more work than it looks. Because ReLU switches fully off, a
trained network is **sparse** at any given input. Most neurons contribute nothing, and the ones
that fire are the ones whose pattern is present. That sparsity is part of why ReLU trains faster
than the sigmoid. A sigmoid never quite switches off, so every unit is always slightly involved in
every prediction, and its gradient gets very small at both extremes.

**[pause]**

There's one thing to watch on the output layer. It's genuinely tempting to put an activation there,
because every other layer has one. Don't. A ReLU on the output would make negative forecasts
impossible. That sounds harmless until you remember that we often model log units, or differences,
where a negative number is exactly what you want. Regression output layers stay linear.

---

# ▶ SLIDE 6 — Why the Nonlinearity Is the Whole Trick

I want to be precise about what that sigma actually buys, because it's easy to assume that *depth*
is what makes a network powerful. It isn't.

Start with the line at the top of the slide: a linear function of a linear function is linear. If
you stack two layers with no activation between them, you get W-two times W-one times x, and W-two
times W-one is just some other matrix. That's one matrix, applied once. You've built a linear model
with extra steps and a slower fitting procedure.

So depth alone gains you **nothing**.

**[pause]**

*(Point at the left panel.)*

Here's what that costs on real data. Each gray dot is one weekday of electricity demand on the PJM
East grid, from two thousand two to twenty eighteen, which is the dataset from Lecture 3, on
generalized additive models. The dots are plotted against the day of the year, and the year is
shaped like a W. It's high in January from heating, then there's a trough in spring, a big peak in
July from air conditioning, another trough in October, and then it climbs back up for winter.

The blue line is the best straight line through it. And that line is the best that *any* stack of
linear layers can ever do, however many layers you stack, because the whole stack collapses to one
line. It's nearly flat, and it misses by about four and a half thousand megawatts on a typical day.

*(Point at the right panel.)*

Now give the network four ReLU units. Each one is a hinge: it's zero up to its knot, and then it's
a straight ramp. Add four of them together and you get the red curve, which is made of straight
pieces joined at four kinks, the red dots. At each dot one more unit switches on and bends the
line, and the dashed stub shows where the line would have gone without it.

I fixed the knots here, at days forty, one twenty, two hundred and two ninety, and fitted only the
weights, by least squares. So nothing in this picture depends on a random seed. And it already
follows the W. The miss drops to about three thousand megawatts.

**[pause]**

Here's the geometry to take away. With one input, a hidden layer of H ReLU units draws a
continuous, piecewise-linear curve with at most H kinks. That's a linear spline, except that you
don't choose the knots. The network learns them, along with the weights.

If that sounds familiar, it should. Prophet's trend, back in Lecture 3, was piecewise linear with
changepoints. That's this same sum of hinges, with the changepoints fixed in advance. A network is
what you get when you let the data place them.

---

# ▶ SLIDE 7 — Universal Approximation: Existence, Not Estimation

Add more units and the kinks multiply. That's the intuition behind a result called the **universal
approximation theorem**. It says that a network with a single hidden layer can approximate any
continuous function as closely as you like, over a bounded range of inputs, given enough neurons.

That bounded range matters in a forecasting course. Outside the range it was trained on, a ReLU
network is just a straight line past its last kink, whatever the true function does out there.

Now, read that claim carefully, because it's quoted far more often than it's understood.

It says the network **can represent** the function. It doesn't say your optimizer will find it. It
doesn't say your data can identify it. And it certainly doesn't say the result will generalize to
next quarter.

Existence is not estimation, and you've already met that gap. In Lecture 4, on decision trees, we
said the optimal decision tree exists, and CART builds a greedy one instead, because finding the
optimal one is computationally hopeless. It's the same shape of problem, in a different setting.

**[pause]**

And look at the numbers on the slide, because they make the point concrete. Four units already get
the electricity year to an RMSE of three thousand and thirty-seven. The best that *any* function of
the day of the year can do on that data, which is the plain average for each calendar day, is two
thousand eight hundred sixty-four. So four units have already captured almost everything the day of
the year can tell you, and more units can't buy much here. What's missing past that point isn't
capacity. It's information, such as the temperature.

**[pause]**

There's a practical reading of the theorem too. It's about a *single* hidden layer, made
arbitrarily wide, and in practice nobody builds that. Everybody builds narrow, deep networks
instead, because depth composes features. Layer two builds on what layer one found, rather than
starting over, and that turns out to need far fewer neurons in total for the same job.

So when someone tells you that neural networks are universal approximators, the correct response is
that so are polynomials, and so are splines, and you wouldn't use a degree-forty polynomial to
forecast demand. Being able to represent everything isn't, by itself, a recommendation. What matters
is whether the shape the model prefers matches the shape your data actually has, and that's exactly
the theme of the next two lectures.

---

# ▶ SLIDE 8 — Section divider: Feedforward Networks

The next section is about feedforward networks. You stack layers of neurons, and each layer learns a
higher-level representation of the one below it.

---

# ▶ SLIDE 9 — Feedforward Network: Layers and Forward Pass

A **feedforward network**, which you'll also see called an FFN or a multilayer perceptron, is three
things in sequence.

First there's an input layer, which is just your raw features. Then there are some number of hidden
layers, and each one applies that neuron equation to the layer below it: it multiplies by a matrix,
adds a bias, and applies the activation. Last comes an output layer, which for regression is linear,
with no activation.

There are two words you need. **Width** is how many neurons are in a layer, and **depth** is how
many hidden layers you have. Together they set the model's capacity, meaning the range of functions
it can represent.

*(Point at the key box.)*

Here are sensible starting sizes for forecasting: two to three hidden layers, sixty-four to two
hundred fifty-six neurons in each, ReLU activations, and dropout somewhere between zero point two
and zero point five.

Start at the **small end** of every one of those ranges.

**[pause]**

Here's the trap. Bigger isn't better on its own. A wider or deeper network needs more data and more
regularization to reach the same generalization. So you add capacity when validation error tells
you the current model is genuinely too simple, and not just because the capacity is available and
it's one keystroke away.

Here's the diagnostic that tells you which way to move. Look at training error and validation error
together. If **both** are bad, your model is too simple, so add width or depth. If training error is
excellent and validation error is poor, you have the opposite problem, and more capacity will make
it worse. What you need then is more regularization, or more data, or fewer features.

**[pause]**

That's the bias-variance picture from Lecture 4, and it doesn't change just because the model has
twenty thousand parameters instead of twenty leaves. Only the knobs change.

The reason I push you toward the small end is practical. A small network that underfits tells you
so clearly, and it trains in seconds. A large network that overfits produces a beautiful training
curve and a forecast you shouldn't ship, and it takes ten times as long to discover that.

---

# ▶ SLIDE 10 — How Big Is This Model, Actually?

Capacity is easy to add by accident, so let's count it once, properly.

A layer that maps m inputs to n outputs holds m times n weights, plus n biases. That's the whole
formula.

*(Walk the picture from left to right.)*

Take a network for our weekly panel. It has forty-six engineered features going in, the same
features you built for Homework 4, then two hidden layers of a hundred twenty-eight, and one output.

Each rectangle is one weight matrix, drawn to scale, with one square of area per weight, so the area
*is* the parameter count. The thin strip beside each one is its biases.

From the input to the first hidden layer is the tall, narrow blue block. It's forty-six wide and a
hundred twenty-eight tall, and it holds six thousand and sixteen parameters. From the first hidden
layer to the second is the big red square, a hundred twenty-eight by a hundred twenty-eight, with
sixteen thousand five hundred twelve. And from the second hidden layer to the output is that thin
line on the right, with a hundred twenty-nine.

That comes to a total of **twenty-two thousand six hundred fifty-seven parameters**. And look where
they sit. Seventy-three percent of them are in the one square in the middle. If you widen the hidden
layers, that square grows with the square of the width, while the input and output blocks only grow
in proportion.

**[pause]**

Now sit with that number for a second. We have about five thousand rows of training data, so that's
roughly **four parameters for every observation**.

In Lecture 6, last week, I described sixty predictors on a hundred and fifty observations as a
crisis. That's the situation where ordinary least squares falls apart and you have to regularize.
This is far past that ratio, and networks train in this regime routinely, every day, in production.

What makes it survivable isn't the architecture. It's the regularization, and that's coming up next.

---

# ▶ SLIDE 11 — Section divider: Training — Loss and Backpropagation

Now we come to training. You compute the error, you propagate it backward through the network, and
you adjust every weight by its share of the blame. Then you repeat.

---

# ▶ SLIDE 12 — Training: Loss, Gradients, and the Update Step

Training is one loop, repeated, and it has three steps.

First, you compute the loss. For regression that's the mean squared error, which is the average
squared gap between the prediction and the truth. There's nothing new there.

Second comes backpropagation. You work *backward* through the network, using the chain rule from
calculus to assign each weight its share of the error. That's all backprop is: the chain rule,
applied systematically, from right to left.

Third, you nudge every weight against its gradient. W becomes W minus eta times the derivative of
the loss with respect to W. Eta is the **learning rate**, and it sets how big a step you take.

*(Point at the definition box.)*

There's one refinement. Rather than use the whole dataset to compute each update, you use a
mini-batch, typically thirty-two to two hundred fifty-six rows. The steps are noisier, but you take
far more of them, and each one is much cheaper. That's **stochastic gradient descent**. One full
pass through every batch is called an **epoch**.

**[pause]**

And here's the practical part: you never differentiate anything by hand. Calling loss dot backward
does the entire backward pass for you. That's fundamentally what a deep learning framework *is*. It
records every operation you performed on the way forward, so that it can walk that record backward
and apply the chain rule automatically.

In practice you won't use plain SGD anyway. The Adam optimizer, torch dot optim dot Adam, with a
learning rate of one e minus three, adapts the step size separately for each parameter, and it's the
sensible default for everything in this course.

---

# ▶ SLIDE 13 — The Learning Rate, on a Real Bowl

I want to say a few words about the learning rate, because it's the hyperparameter that will waste
the most of your time if you get it wrong. And for once I can show you exactly what it does.

*(Point at the left panel.)*

This is the regression from Lecture 6, last week: daily units on the lag-seven and lag-fourteen
columns, both standardized. There are two weights, so the loss is a surface over a flat plane, and
the gray rings are its contours. Think of it as a bowl seen from above, with the minimum at the
black cross. Every panel starts gradient descent at the same point and takes the same update steps
from the last slide. Only eta changes.

On the left, eta is zero point one five. Each step moves a fraction of the way downhill, and the
path curves smoothly into the bottom of the valley. It's a little slow near the end, but it gets
there.

*(Point at the middle panel.)*

Here eta is one half. Now each step overshoots across the narrow direction of the valley and lands
on the far wall, and then it overshoots back. That's the zig-zag. It still converges, because every
overshoot is smaller than the last one.

*(Point at the right panel.)*

Here eta is zero point six five. Now every overshoot is *bigger* than the last, and the path flies
out of the bowl. The loss doesn't go down. It explodes. On a real network, that's the loss curve
turning into NaN, which stands for "not a number."

**[pause]**

*(Point at the caption under the figure.)*

The two boundaries come straight out of the shape of the bowl. They're set by its steepest
curvature, called lambda max, which is three point five one here. Above one over lambda max, which
is zero point two eight, you zig-zag. Above two over lambda max, which is zero point five seven, you
diverge. There's nothing to tune here. It's arithmetic.

Now the honest caveat at the bottom of the slide. This bowl is exact only because the model is
linear. A linear model's loss curves by the same amount everywhere, so that limit of two over lambda
max is the same at every point. A network's loss surface bends differently in different places, so
no single learning rate is right everywhere. That's one reason Adam adapts its step for each
parameter as it goes, and it's why Adam is more forgiving of a badly chosen learning rate than plain
SGD.

The practical rule survives the caveat. If the learning rate is too large, the loss jumps around or
diverges. If it's too small, training is correct but glacial, and you'll conclude the model doesn't
work when you just didn't wait. If your loss curve is spiky or turns into NaN, lower the learning
rate by a factor of ten before you change anything else.

---

# ▶ SLIDE 14 — Preventing Overfitting — Where Lecture 6 Returns

Twenty-two thousand parameters on five thousand rows will memorize the training set unless
something stops it. Three things do, and **you've already met two of them**.

*(Walk the table slowly. This slide is the point of the lecture.)*

The first is weight decay, and it **is** Ridge regression. It's an L2 penalty on the network's
weights, under a different name, passed to the optimizer as an argument. It's the same mathematics,
with the same effect, under a new label.

The second is dropout, which you add with n-n dot Dropout and a fraction p. On each forward pass, it
randomly zeroes out a fraction p of the neuron outputs. The consequence is that no neuron can rely
on any specific other neuron being present, so the network can't build fragile chains of
dependency. At test time you turn dropout off, and all the neurons are active.

The third is early stopping. You watch the validation loss every epoch. When it stops improving for
some number of epochs, you stop, and, critically, you keep the weights from the *best* epoch, not
the last one.

**[pause]**

*(Now the callback. Slow down here.)*

That's the **fifth appearance of one idea** in this course.

It was the GAM smoothing penalty in Lecture 3, which penalized wiggliness. It was cost-complexity
pruning in Lecture 4, which penalized the number of leaves. It was reg lambda in Lecture 5, on tree
ensembles, which penalized the leaf weights. It was Ridge and LASSO in Lecture 6, which penalized
the size of the coefficients. And now it's weight decay, which penalizes the network's weights.

That's five model families that look nothing like each other, with one bargain underneath. You
**fit against complexity**, with a single parameter setting the exchange rate, and you choose that
parameter by validation.

You'll see it a sixth time in Lecture 11, on hierarchical models, and there it will be the cleanest
version of all.

---

# ▶ SLIDE 15 — Early Stopping, on Lab 7's Network

Here's the third of those tools, early stopping, on the network you'll train in the lab, done the
way you'd do it for real.

*(Point at the figure.)*

It's the same model, with the same settings and the same seeds as Lab 7, run for two hundred epochs
instead of sixty. There's one change. I held out the last fifty-two weeks of the training period,
from mid-May twenty fourteen to mid-May twenty fifteen, as a validation slice. That's a full year,
so every season is in it, and it's about as long as the fifty-three test weeks. The network trains
only on the weeks before that slice, and it's standardized on them too.

That leaves about a quarter fewer training windows than the lab uses, so these aren't the lab's
curves. At epoch sixty, this run scores about eleven sixty, fourteen ten and ten fifty on the test
weeks. The lab's own run, with that extra year to learn from, scores about eleven hundred, twelve
hundred and nine fifty.

There's one panel per seed. Gray is the error on the training windows, blue is the error on the
validation slice, and red is the error on the test weeks.

**[pause]**

The gray curve keeps going down, because the model keeps getting better at the data it's fitting.
The blue curve drops fast, and after about epoch sixty it improves only slowly, jumping by about
fifty from one epoch to the next. The red curve is jumpier still, by about a hundred.

The blue dot in each panel marks the lowest validation error, at epochs one oh seven, one
thirty-two and eighty. Early stopping keeps the weights from that epoch. Follow the dotted line up
to the red dot, and you see what those weights score on the test weeks: about nine fifty, nine
fifty and a thousand.

Now look at the ring, at epoch sixty, where the lab stops. Keeping validation's best epoch beat it
on all three seeds: by two hundred ten, by four hundred sixty, and for seed three by only fifty. It
also beat the last epoch every time, though for seed two by only five.

**[pause]**

*(This is the honest part. Don't skip it.)*

The black cross marks the best test epoch, at one ninety-five, one sixty-four and one ninety-eight.
It's better than what validation picked, by fifty to a hundred twenty. And you couldn't have picked
it. The only way to find it is to look at the test weeks, and then you'd be choosing on the test
set, and the number you report would flatter you. Validation's minimum is also the luckiest point
on a jagged curve, which is why it isn't a perfect guide to the test. But it's an honest one.

In practice you'd set a patience, which is the number of epochs you'll wait for an improvement, and
you'd usually stop sooner, perhaps at a different epoch. Here I ran all two hundred epochs and kept
the best. Either way, you watch validation, you keep the best weights on it, and you touch the test
weeks only once, at the end.

---

# ▶ SLIDE 16 — Section divider: PyTorch Implementation

Now let's build it in PyTorch. There are four pieces: a Dataset, a DataLoader, a model, and a
training loop.

---

# ▶ SLIDE 17 — PyTorch: Dataset and DataLoader

*(This is the first of three code slides. Read the structure, not every character.)*

There are two objects here. A Dataset knows how to fetch **one** example. You give it a method
called dunder len, with a double underscore on each side, so it knows how many examples there are,
and a method called dunder getitem, so it can hand back item i. A DataLoader wraps the Dataset and
handles the batching, so your training loop can just iterate.

You'll write this exact class, or something very close to it, in Homework 5.

**[pause]**

*(Now the important part.)*

Look at shuffle equals False.

Nearly every PyTorch tutorial you'll find online sets that to True. We set it to False, and I want
to be precise about why, because people get this wrong in both directions.

Our reason is reproducibility. With shuffle set to False, the same seed gives the same model every
time, so when two runs differ, you know it was the seed.

What shuffling would **not** do here is leak the future. Each training example is a whole window,
so it carries its own past with it, and every window comes from before the cutoff. Shuffling the
order the windows arrive in changes nothing about what the model can see.

What **does** leak is shuffling *before* the split. That means a random split into training and
test, or shuffling rows before you build the windows. It's the same discipline as using Time Series
Split instead of K-fold, back in Lecture 1, on forecast evaluation. The default in the tooling
assumes your rows are exchangeable, and ours aren't.

**[pause]**

And notice the failure mode, because it's the one this course keeps returning to. A random split
doesn't raise an error, and it doesn't warn you. Your model trains, your loss goes down, and you get
a number. It's a number that's better than it should be, because you've quietly let the model see
the future.

That's the same class of error as K-fold on a time series, as scaling outside the fold, or as an
unshifted rolling feature. None of them crash, and all of them return a better-looking answer than
you deserve. That's exactly why the discipline has to be procedural. You check these things because
they're on a list, not because something broke.

---

# ▶ SLIDE 18 — PyTorch: Defining the Model

Here's the network from slide 10, in five lines of code.

Read it from top to bottom and it *is* the block diagram from slide 10. First comes a linear map to
a hundred twenty-eight units, which is the first block. Then a ReLU, and some dropout. Then the same
again, linear, ReLU and dropout, which is the big square. Then a final linear layer down to one
output, the thin line, with no activation on it, because this is regression and we don't want to
squash the forecast.

The Sequential container, n-n dot Sequential, chains layers in order, and it's enough for anything
in this lecture. When you need an architecture that branches, or one with a skip connection, which
is next week, in Lecture 8, you'll subclass n-n dot Module and write the forward method yourself.

*(Point at the muted footnote.)*

And n-n dot Linear, with its input and output sizes, is where the weight matrix and the bias vector
actually live. That's where those twenty-two thousand six hundred fifty-seven parameters are
sitting.

---

# ▶ SLIDE 19 — PyTorch: The Training Loop

And here's the loop. There are four steps for each mini-batch, and every PyTorch training loop
you'll ever read is these four lines with more scaffolding around them.

First, zero-grad clears the old gradients. Second, the forward pass computes the loss. Third,
backward computes the new gradients. And fourth, step updates the weights.

That's it. You set up the optimizer with Adam, a learning rate of one e minus three, and a weight
decay of one e minus four, and there's your Ridge penalty. The loss function is the mean squared
error.

**[pause]**

*(The last paragraph is the one worth emphasizing.)*

Now for the one that bites people.

If you forget zero-grad, PyTorch doesn't raise an error, and it doesn't warn you. What it does is
**accumulate** gradients across batches instead of replacing them, because gradient accumulation is
a legitimate technique that some people use deliberately.

So your model trains. It just trains badly, and you have no idea why.

We'll do this on purpose in the lab, so you can see what it does to the forecast when it happens.
It's worth seeing once in a controlled setting, rather than at eleven at night before a deadline.

---

# ▶ SLIDE 20 — Section divider: Key Takeaways

To finish, let's ask where a network belongs in the toolkit you now have.

---

# ▶ SLIDE 21 — Neural Networks vs. Tree Methods

*(Walk the table across, then land on the paragraph below it.)*

The table compares five methods on four questions. Does the method see sequences? Can it
extrapolate beyond the training range? How much tuning does it need? And what is it actually for?

LASSO and Ridge don't see sequences, but they extrapolate, they need only light tuning, and they're
your answer when you have many correlated predictors. A random forest doesn't see sequences and
doesn't extrapolate, it needs light tuning, and it's the robust default for tabular data. XGBoost is
the same, with moderate tuning, and it's best in class for accuracy on tabular data. The feedforward
network still doesn't see sequences, but it extrapolates, it needs heavy tuning, and it's for smooth
functions when n is large. And RNNs and LSTMs, which are Lecture 9, finally say **yes** in that
first column.

**[pause]**

Here's the sentence I want you to leave with. An FFN on lag features is doing **the same job** as
XGBoost on lag features. It has the same inputs, the same flat table, and no notion of order, just
like XGBoost. It isn't obviously the better tool, and on the evidence of Homework 4 it usually
isn't.

What changes over the next two lectures isn't the fitting procedure. It's the **architecture**:
convolutions that see local structure, and recurrence that carries state through time. That's when
the sequence column says yes, and it's precisely why we learned this machinery today.

**[pause]**

There's one more column worth dwelling on, and that's extrapolation.

Trees and forests say no, and that isn't a tuning problem. It's structural. A tree predicts by
averaging the training values that landed in a leaf, so it physically can't output a number larger
than the largest one it has seen. If your series is trending upward, a random forest will forecast a
flat line at the top of its training range, forever.

The linear models and the feedforward network say yes. A LASSO fit with a trend term will happily
project that trend forward, and so will a ReLU network, past its last kink.

*(Point at the footnote under the table.)*

The row for RNNs and LSTMs says *bounded*, and that's a real difference. Their hidden state is
squashed by tanh into the range from minus one to one. So with a linear output layer, the forecast
can go past the training maximum, but only up to a fixed ceiling set by the output weights.

Whether extrapolation is a *feature* depends entirely on whether the trend is real. Extrapolating a
genuine trend is the right answer. Extrapolating a temporary run-up is how forecasts embarrass
people. The model can't tell the difference. That judgment is yours, and it's the kind of thing a
DAG helps you reason about, in Lecture 12, on Bayesian regression.

---

# ▶ SLIDE 22 — Lecture 7: Key Takeaways

*(Six points. Read them at a steady pace, and don't rush the last two.)*

Let me leave you with six takeaways.

First, a neuron is linear regression wrapped in a nonlinear activation, and without that
nonlinearity, stacking layers collapses back to a single linear model. With ReLU, each unit adds one
kink, which is how four units drew the W in the electricity year.

Second, an FFN stacks those layers. Width and depth set its capacity, and capacity piles up faster
than you expect: twenty-two thousand six hundred fifty-seven parameters on five thousand rows.

Third, training is one loop: the loss, then backward, then step, on mini-batches, using Adam. You
never differentiate by hand. And you keep the weights from the best epoch on validation, not the
last one.

Fourth, regularization isn't new. Weight decay *is* Ridge, and dropout and early stopping do the
same job by other means. That's the fifth appearance of one idea.

The fifth takeaway is for time series specifically. Split by time, never shuffle before the split,
and call zero-grad on every batch.

And the sixth is the one that matters most. An FFN on engineered features is a **competitor** to
XGBoost, not an upgrade. Architecture is what earns its keep, and that starts next week.

**[pause]**

Next week, in Lecture 8, we turn to CNN architectures: convolution, pooling, and what a
one-dimensional convolution does to a time series.

---

# ▶ SLIDE 23 — References

*(Advance and close. No narration needed.)*
