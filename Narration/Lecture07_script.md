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

Lecture 7: Introduction to Neural Networks.

This is the start of the second half of the course, and I want to set expectations before we
begin, because this topic arrives with more hype attached than anything else we'll cover.

For the last three weeks you've been building tree models — random forests, gradient boosting,
regularized regression. Those work extremely well on the kind of data we have. Today we start on
neural networks, and by the end of the next two lectures you will have fitted several of them.

**[pause]**

Here's what I want you to hold onto. Nothing today makes your XGBoost model obsolete. On our
data, the networks are going to lose. What today buys you is the machinery — and the machinery
matters because of what we do with it in Lectures 8 and 9, where the *architecture* starts
carrying assumptions that a flat table cannot.

---

# ▶ SLIDE 2 — Lecture Outline

Five parts. First, why you'd go beyond trees at all — and when you shouldn't. Then the
feedforward network itself: layers, width, depth, and how quickly the parameter count runs away
from you. Then training — loss, backpropagation, the update step. Then PyTorch, which is four
pieces you'll use in every deep learning assignment for the rest of the semester. And then where
a network actually belongs in the toolkit you now have.

---

# ▶ SLIDE 3 — Section divider: From Trees to Neural Networks

*(Advance and read the subtitle aloud — it's the thesis of the whole lecture.)*

Trees predict in steps. Networks predict in curves. And that difference buys you less than you
would expect on tabular data, and everything on sequences.

---

# ▶ SLIDE 4 — Why Go Beyond Trees?

Two honest reasons, and one honest warning.

The first reason: a tree-based model treats every row as independent. Order is not part of the
model. Everything a random forest knows about *time* — that last week matters, that fifty-two
weeks ago matters — you put there by hand, as lag columns and rolling means. If you shuffled the
feature columns, the model would be identical.

The second reason: a tree's prediction is **piecewise constant**. Remember Lab 4, where the tree
could not predict above the largest value it had ever seen? That's not a bug in our
implementation. It is what a tree is. It averages training values in a leaf, so it cannot
extrapolate. A neural network's prediction is continuous instead. With the ReLU activation you're
about to meet, it's made of straight pieces joined at kinks — you'll see exactly that in two slides —
and past its last kink it just keeps going in a straight line. So it can extrapolate.

**[pause]**

Now the warning, and I want to be blunt about it.

*(Point at the warning box.)*

This is **not an upgrade**. On the tabular problems in Homework 3 and Homework 4, gradient
boosting and a tuned LASSO were hard to beat, and a network will often lose to both. You reach
for a network when the data is genuinely a sequence, when the raw input is an image or audio, or
when your sample size is very large. Otherwise you are paying in tuning time for accuracy you
will not get.

So today is about the machinery. Lectures 8 and 9 are where it earns its place.

---

# ▶ SLIDE 5 — From Linear Regression to a Single Neuron

Let's build the thing from something you already know.

Linear regression predicts y-hat equals w-transpose x plus b. A weighted sum of your predictors,
plus an intercept. You've been fitting that since Lecture 2.

A **neuron** is that exact expression with one addition: a nonlinear function wrapped around the
outside. We call that function the **activation**, and we write it sigma.

So: a equals sigma of w-transpose x plus b. That's it. That is the entire unit.

*(Walk the three panels, left to right.)*

Three activations you'll meet, each drawn with its slope marked. **ReLU**, on the left — max of
zero and z — is the default for hidden layers. Its slope is zero on the left and exactly one on
the right, nothing in between. **Sigmoid**, in the middle, squashes to between zero and one;
you'll see it again in Lecture 9 as an LSTM gate. Look at the dashed line: its steepest slope,
right at zero, is one quarter, and everywhere else it is flatter than that. **Tanh**, on the
right, squashes to between minus one and one and shows up in sequence states; its steepest slope
is one.

Hold onto that one quarter. When a gradient flows backward through a sigmoid, it gets multiplied
by that slope, so each sigmoid scales it by a quarter at best. Stack a few and the signal is
gone. That is the vanishing gradient, and it is the whole story of Lecture 9.

And a fourth one that isn't drawn because there is nothing to draw: **linear**, which is just
z. That's your output layer when you're doing regression, because you don't want to squash a
forecast.

**[pause]**

ReLU is worth a sentence of intuition, because it makes the whole thing concrete. The neuron is
asking a question: *is this pattern present?* If the evidence is positive, it passes a signal
through, proportional to how positive. If the evidence is negative, it outputs exactly zero and
stays silent. That's a detector.

**[pause]**

And that "exactly zero" is doing more work than it looks. Because ReLU switches fully off, a
trained network is **sparse** at any given input — most neurons contribute nothing, and the ones
that fire are the ones whose pattern is present. That sparsity is part of why ReLU trains faster
than sigmoid: a sigmoid never quite switches off, so every unit is always slightly involved in
every prediction, and its gradient gets very small at both extremes.

One thing to watch on the output layer. It is genuinely tempting to put an activation there
because every other layer has one. Don't. A ReLU on the output would make negative forecasts
impossible, which sounds harmless until you remember we often model *log* units or *differences*,
where negative is exactly what you want. Regression output layers stay linear.

---

# ▶ SLIDE 6 — Why the Nonlinearity Is the Whole Trick

I want to be precise about what that sigma actually buys, because it is easy to assume that
*depth* is what makes a network powerful. It is not.

Start with the line at the top of the slide. A linear function of a linear function is linear. If
you stack two layers with no activation between them, you get W-two times W-one times x — and
W-two times W-one is just some other matrix. One matrix, applied once. You've built a linear model
with extra steps and a slower fitting procedure.

Depth alone gains you **nothing**.

**[pause]**

*(Point at the left panel.)*

Here's what that costs on real data. Each gray dot is one weekday of electricity demand on the
PJM East grid, 2002 to 2018 — the dataset from Lecture 3 — plotted against the day of the year. The
year is shaped like a W: high in January from heating, a trough in spring, a big peak in July from
air conditioning, another trough in October, and back up for winter.

The blue line is the best straight line through it. And that line is the best that *any* stack of
linear layers can ever do, however many layers you stack, because the whole stack collapses to one
line. It's nearly flat, and it misses by about four and a half thousand megawatts on a typical
day.

*(Right panel.)*

Now give the network four ReLU units. Each one is a hinge: zero up to its knot, then a straight
ramp. Add four of them and you get the red curve — straight pieces, joined at four kinks, the red
dots. At each dot one more unit switches on and bends the line; the dashed stub shows where the
line would have gone without it.

I fixed the knots here, at days forty, one twenty, two hundred and two ninety, and fitted only the
weights, by least squares. So nothing in this picture depends on a random seed. And it already
follows the W: the miss drops to about three thousand megawatts.

**[pause]**

Here's the geometry to take away. With one input, a hidden layer of H ReLU units draws a
continuous, piecewise-linear curve with at most H kinks. That's a linear spline — and the knots,
instead of being chosen by you, are learned by the network, along with the weights.

If that sounds familiar, it should. Prophet's trend in Lecture 3 was piecewise linear with
changepoints. That's this same sum of hinges, with the changepoints fixed in advance. A network is
what you get when you let the data place them.

---

# ▶ SLIDE 7 — Universal Approximation: Existence, Not Estimation

Add more units and the kinks multiply. That's the intuition behind a result called the **universal
approximation theorem**: a network with a single hidden layer can approximate any continuous
function as closely as you like over a bounded range of inputs, given enough neurons.

That "bounded range" matters in a forecasting course. Outside the range it was trained on, a ReLU
network is just a straight line past its last kink, whatever the true function does there.

Now — read that claim carefully, because it is quoted far more often than it is understood.

It says the network **can represent** the function. It does not say your optimizer will find it.
It does not say you have enough data to identify it. And it certainly does not say the result
will generalize to next quarter.

Existence is not estimation. You've already met that gap: in Lecture 4 we said the optimal
decision tree exists, and CART builds a greedy one instead, because finding the optimal one is
computationally hopeless. Same shape of problem, different setting.

**[pause]**

And look at the number on the slide, because it makes the point concrete. Four units already get
the electricity year to about three thousand and forty. The best *any* function of the day of the
year can do on that data — the plain average for each calendar day — is about two thousand eight
hundred and sixty. So four units have already captured almost everything the day of the year can
tell you. More units can't buy much here. What's missing past that point isn't capacity, it's
information — temperature, say.

**[pause]**

There's a practical reading of the theorem too. It's about a *single* hidden layer, made
arbitrarily wide. In practice nobody builds that. Everybody builds narrow-and-deep instead, because
depth composes features — layer two builds on what layer one found, rather than starting over — and
that turns out to need far fewer total neurons for the same job.

So when someone tells you "neural networks are universal approximators," the correct response is
that so are polynomials, and so are splines, and you would not use a degree-forty polynomial to
forecast demand. Being able to represent everything is not by itself a recommendation. What
matters is whether the shape the model prefers matches the shape your data actually has — which
is exactly the theme of the next two lectures.

---

# ▶ SLIDE 8 — Section divider: Feedforward Networks

Stack layers of neurons, and each layer learns a higher-level representation of the one below it.

---

# ▶ SLIDE 9 — Feedforward Network: Layers and Forward Pass

A **feedforward network** — you'll also see it called an FFN or a multilayer perceptron — is
three things in sequence.

An **input layer**, which is just your raw features. Some number of **hidden layers**, each one
applying that neuron equation to the layer below it — matrix multiply, add a bias, apply the
activation. And an **output layer**, which for regression is linear, no activation.

Two words you need. **Width** is how many neurons are in a layer. **Depth** is how many hidden
layers you have. Together they set the model's capacity — the range of functions it can represent.

*(Point at the key box.)*

Sensible starting sizes for forecasting: two to three hidden layers, sixty-four to two hundred
fifty-six neurons each, ReLU activations, dropout somewhere between zero-point-two and
zero-point-five.

**Start at the small end of every one of those ranges.**

**[pause]**

Because here's the trap. Bigger is not better on its own. A wider or deeper network needs *more
data* and *more regularization* to reach the same generalization. So you add capacity when
validation error tells you the current model is genuinely too simple — not because the capacity
is available and it's one keystroke away.

**[pause]**

Here's the diagnostic that tells you which way to move. Look at training error and validation
error together. If **both** are bad, your model is too simple — add width or depth. If training
error is excellent and validation error is poor, you have the opposite problem, and more capacity
will make it worse; you need more regularization, or more data, or fewer features.

That's the bias-variance picture from Lecture 4, and it does not change just because the model
has twenty thousand parameters instead of twenty leaves. Only the knobs change.

The reason I push you toward the small end is practical. A small network that underfits tells you
so clearly and trains in seconds. A large network that overfits produces a beautiful training
curve and a forecast you shouldn't ship, and it takes ten times as long to discover that.

---

# ▶ SLIDE 10 — How Big Is This Model, Actually?

Capacity is easy to add by accident, so let's count it once, properly.

A layer mapping m inputs to n outputs holds m times n weights, plus n biases. That's it — that's
the whole formula.

*(Walk the picture left to right.)*

Take a network for our weekly panel. Forty-six engineered features going in — the same features
you built for Homework 4. Two hidden layers of a hundred twenty-eight. One output.

Each rectangle is one weight matrix, drawn to scale: one square of area per weight, so the area
*is* the parameter count. The thin strip beside each one is its biases.

Input to hidden one: the tall narrow blue block, forty-six wide and a hundred twenty-eight tall,
six thousand and sixteen parameters. Hidden one to hidden two: the big red square, one twenty-eight
by one twenty-eight, sixteen thousand five hundred twelve. Hidden two to output: that thin line
on the right, one hundred twenty-nine.

Total: **twenty-two thousand six hundred fifty-seven parameters.** And look where they sit:
**seventy-three percent of them are in the one square in the middle.** Widen the hidden layers
and that square grows with the width squared; the input and output blocks only grow in
proportion.

**[pause]**

Now sit with that number for a second. We have about five thousand rows of training data. That's
roughly **four parameters for every observation**.

In Lecture 6 I described sixty predictors on a hundred and fifty observations as a crisis — the
situation where ordinary least squares falls apart and you *must* regularize. This is far past
that ratio. And networks train in this regime routinely, every day, in production.

What makes it survivable is not the architecture. It's the regularization. Which is next.

---

# ▶ SLIDE 11 — Section divider: Training — Loss and Backpropagation

Compute the error, propagate it backward, adjust every weight by its share of the blame. Repeat.

---

# ▶ SLIDE 12 — Training: Loss, Gradients, and the Update Step

Training is one loop, repeated. Three steps.

**First**, compute the loss. For regression that's mean squared error — the average squared gap
between prediction and truth. Nothing new.

**Second**, backpropagation. You work *backward* through the network, using the chain rule from
calculus to assign each weight its share of the error. That's all backprop is: the chain rule,
applied systematically, right to left.

**Third**, you nudge every weight against its gradient. W becomes W minus eta times the
derivative of the loss with respect to W. Eta is the **learning rate** — how big a step you take.

*(Point at the definition box.)*

One refinement. Rather than using the whole dataset to compute each update, you use a
**mini-batch** — typically thirty-two to two hundred fifty-six rows. The steps are noisier, but
you take far more of them and each one is much cheaper. That's **stochastic gradient descent**.
One full pass through every batch is called an **epoch**.

**[pause]**

And here's the practical part: **you never differentiate anything by hand.** Calling
`loss.backward()` does the entire backward pass for you. That's what a deep learning framework
*is*, fundamentally — it records every operation you performed on the way forward so it can walk
that record backward and apply the chain rule automatically.

In practice you won't use plain SGD anyway. `torch.optim.Adam` with a learning rate of
one-e-minus-three adapts the step size separately for each parameter, and it is the sensible
default for everything in this course.

---

# ▶ SLIDE 13 — The Learning Rate, on a Real Bowl

Two words about the learning rate, because it is the hyperparameter that will waste the most of
your time if you get it wrong. And for once I can show you exactly what it does.

*(Point at the left panel.)*

This is the regression from Lecture 6: daily units on the lag-seven and lag-fourteen columns,
both standardized. Two weights, so the loss is a surface over a flat plane, and the gray rings are
its contours — a bowl seen from above, with the minimum at the black cross. Every panel starts
gradient descent at the same point and takes the same steps of the update rule from the last
slide. Only eta changes.

On the left, eta is zero point one five. Each step moves a fraction of the way downhill, and the
path curves smoothly into the bottom of the valley. A little slow near the end, but it gets there.

*(Middle panel.)*

Eta of one half. Now each step overshoots across the narrow direction of the valley and lands
on the far wall, then overshoots back. That's the zig-zag. It still converges, because every
overshoot is smaller than the last.

*(Right panel.)*

Eta of zero point six five. Now every overshoot is *bigger* than the last, and the path flies out
of the bowl. The loss doesn't go down; it explodes. On a real network that's the loss curve
turning into `nan`.

**[pause]**

*(Point at the caption under the figure.)*

The two boundaries come straight out of the shape of the bowl. Its steepest curvature — lambda
max, three point five one here — sets them. Above one over lambda max, zero point two eight, you
zig-zag. Above two over lambda max, zero point five seven, you diverge. Nothing to tune: that is
arithmetic.

Now the honest caveat at the bottom of the slide. This bowl is exact only because the model is
linear: a linear model's loss curves the same amount everywhere, so that two-over-lambda limit is
the same at every point. A network's loss surface bends differently in different places, so no
single learning rate is right everywhere. That is one reason Adam adapts its step for each
parameter as it goes, and why it is more forgiving of a badly chosen learning rate than plain SGD.

The practical rule survives the caveat. Too large, and the loss jumps around or diverges. Too
small, and training is correct but glacial, and you'll conclude the model doesn't work when you
just didn't wait. If your loss curve is spiky or turns into `nan`, lower the learning rate by a
factor of ten before you change anything else.

---

# ▶ SLIDE 14 — Preventing Overfitting — Where Lecture 6 Returns

Twenty-two thousand parameters on five thousand rows will memorize the training set unless
something stops it. Three things do — and **you have already met two of them.**

*(Walk the table slowly. This slide is the point of the lecture.)*

`weight_decay`. This is **Ridge regression**. It is an L2 penalty on the network's weights, under
a different name, passed as an argument to the optimizer. Same mathematics, same effect, new
label.

`nn.Dropout(p)`. On each forward pass, randomly zero out a fraction p of the neuron outputs. The
consequence is that no neuron can rely on any specific other neuron being present, so the network
can't build fragile chains of dependency. At test time you turn it off and all neurons are active.

**Early stopping.** Watch validation loss every epoch. When it stops improving for some number of
epochs, stop — and critically, keep the weights from the *best* epoch, not the last one.

**[pause]**

*(Now the callback. Slow down here.)*

That is the **fifth appearance of one idea** in this course.

The GAM smoothing penalty in Lecture 3, which penalized wiggliness. Cost-complexity pruning in
Lecture 4, which penalized the number of leaves. `reg_lambda` in Lecture 5, penalizing leaf
weights. Ridge and LASSO in Lecture 6, penalizing coefficient size. And now weight decay,
penalizing network weights.

Five model families that look nothing like each other. One bargain underneath: **fit against
complexity, with a single parameter setting the exchange rate, chosen by validation.**

You'll see it a sixth time in Lecture 11, and there it will be the cleanest version of all.

---

# ▶ SLIDE 15 — Early Stopping, on Lab 7's Network

Here is the third of those tools, early stopping, on the network you'll train in the lab, done
the way you would do it for real.

*(Point at the figure.)*

Same model, same settings, same seeds as Lab 7, run for two hundred epochs instead of sixty. One
change. I held out the last fifty-two weeks of the training period, mid-May 2014 to mid-May 2015,
as a validation slice. That's a full year, so every season is in it, and it's about as long as
the fifty-three test weeks. The network trains only on the weeks before that slice, and is
standardized on them too.

That is about a quarter fewer training windows than the lab uses, so these are not the lab's
curves. At epoch sixty this run scores about eleven sixty, fourteen ten and ten fifty on the test
weeks; the lab's own run, with that extra year to learn from, scores eleven hundred, twelve
hundred and nine fifty.

One panel per seed. Gray is the error on the training windows, blue on the validation slice, red
on the test weeks.

**[pause]**

Gray keeps going down: the model keeps getting better at the data it's fitting. Blue drops fast
and then improves only slowly after about epoch sixty, jumping by about fifty from one epoch to
the next. Red is jumpier still, about a hundred.

The blue dot in each panel is the lowest validation error: epoch one oh seven, one thirty-two,
and eighty. Early stopping keeps the weights from that epoch. Follow the dotted line up to the
red dot: that is what those weights score on the test weeks. About nine fifty, nine fifty, and a
thousand.

Now the ring, at epoch sixty, where the lab stops. Keeping validation's best epoch beat it on
all three seeds: by two hundred ten, by four hundred sixty, and for seed three by only fifty. It
also beat the last epoch every time, though for seed two by only five.

**[pause]**

*(This is the honest part. Don't skip it.)*

The black cross is the best test epoch: one ninety-five, one sixty-four and one ninety-eight.
It's better than what validation picked, by fifty to a hundred twenty. And you could not have
picked it. The only way to find it is to look at the test weeks, and then you'd be choosing on
the test set, and the number you report would flatter you. Validation's minimum is the luckiest
point on a jagged curve too, which is why it isn't a perfect guide to the test. But it's an
honest one.

In practice you'd set a patience and usually stop sooner, perhaps at a different epoch; here I
ran all two hundred epochs and kept the best. Either way you watch validation, keep the best weights on it — and touch the test weeks
once, at the end.

---

# ▶ SLIDE 16 — Section divider: PyTorch Implementation

Four pieces: a Dataset, a DataLoader, a model, and a training loop.

---

# ▶ SLIDE 17 — PyTorch: Dataset and DataLoader

*(This is the first of three code slides. Read the structure, not every character.)*

Two objects. A `Dataset` knows how to fetch **one** example — you give it `__len__` so it knows
how many there are, and `__getitem__` so it can hand back item i. A `DataLoader` wraps that and
handles batching, so your training loop can just iterate.

You'll write this exact class, or something very close to it, in Homework 5.

**[pause]**

*(Now the important part.)*

Look at `shuffle=False`.

Nearly every PyTorch tutorial you find online sets that to `True`. We set it to `False`, and I
want to be precise about why, because people get this wrong in both directions.

Our reason is reproducibility: with `shuffle=False`, the same seed gives the same model every
time, so when two runs differ, you know it was the seed.

What shuffling would **not** do here is leak the future. Each training example is a whole window —
it carries its own past with it — and every window is from before the cutoff. Shuffling the order
the windows arrive in changes nothing about what the model can see.

What **does** leak is shuffling *before* the split: a random train/test split, or shuffling rows
before you build the windows. That is the same discipline as `TimeSeriesSplit` instead of `KFold`,
from Lecture 5. The default in the tooling assumes your rows are exchangeable. Ours are not.

**[pause]**

And notice the failure mode, because it's the one this course keeps returning to. A random split
does not raise an error. It does not warn you. Your model trains, your loss goes down, and you get
a number — a number that is better than it should be, because you have quietly let the model see
the future.

That is the same class of error as `KFold` on a time series, as scaling outside the fold, as an
unshifted rolling feature. None of them crash. All of them return a better-looking answer than
you deserve. Which is exactly why the discipline has to be procedural: you check these things
because they're on a list, not because something broke.

---

# ▶ SLIDE 18 — PyTorch: Defining the Model

Here is the network from slide 10, in five lines.

Read it top to bottom and it *is* the block diagram from slide 10. A linear map to a hundred
twenty-eight units — the first block. A
ReLU. Some dropout. Then again — linear, ReLU, dropout — the big square. Then a final linear
layer down to one output, the thin line, with no activation on it, because this is regression and we don't want to squash the
forecast.

`nn.Sequential` chains layers in order, and it is enough for anything in this lecture. When you
need an architecture that branches, or that has a skip connection — which is Lecture 8 — you'll
subclass `nn.Module` and write the `forward()` method yourself.

*(Point at the muted footnote.)*

And `nn.Linear(in, out)` is where the weight matrix and bias vector actually live. That's where
those twenty-two thousand six hundred fifty-seven parameters are sitting.

---

# ▶ SLIDE 19 — PyTorch: The Training Loop

And here is the loop. Four steps per mini-batch, and every PyTorch training loop you will ever
read is these four lines with more scaffolding around them.

`zero_grad` — clear the old gradients. Forward pass and compute the loss. `backward` — compute
the new gradients. `step` — update the weights.

That's it. Set up the optimizer with Adam, learning rate one-e-minus-three, weight decay
one-e-minus-four — there's your Ridge penalty. Loss function is MSE.

**[pause]**

*(The last paragraph is the one worth emphasizing.)*

Now the one that bites people.

If you forget `zero_grad()`, PyTorch does not raise an error. It does not warn you. What it does
is **accumulate** gradients across batches instead of replacing them — because gradient
accumulation is a legitimate technique that some people want deliberately.

So your model trains. It just trains badly, and you have no idea why.

We will do this deliberately in the lab, so you can see what it does to the forecast when it
happens. It is worth seeing once in a controlled setting rather than at eleven at night before a
deadline.

---

# ▶ SLIDE 20 — Section divider: Key Takeaways

Where a network belongs in the toolkit you now have.

---

# ▶ SLIDE 21 — Neural Networks vs. Tree Methods

*(Walk the table across, then land on the paragraph below it.)*

Five methods, four questions. Does it see sequences? Can it extrapolate beyond the training
range? How much tuning does it need? And what is it actually for?

LASSO and Ridge: no sequences, but they extrapolate, they need almost no tuning, and they're your
answer when you have many correlated predictors. Random forest: no sequences, no extrapolation,
light tuning, and it's the robust tabular default. XGBoost: same, more tuning, best-in-class
accuracy on tabular data. The feedforward network: still no sequences, extrapolates, heavy
tuning, and it's for smooth functions when n is large. And RNNs and LSTMs — which is Lecture 9 —
finally say **yes** in that first column.

**[pause]**

Here's the sentence I want you to leave with.

An FFN on lag features is doing **the same job** as XGBoost on lag features. Same inputs, same
flat table, no notion of order in either one. It is not obviously the better tool, and on the
evidence of Homework 4 it usually is not.

What changes over the next two lectures is not the fitting procedure. It's the **architecture** —
convolutions that see local structure, and recurrence that carries state through time. That is
when the sequence column starts saying yes, and it is precisely why we learned this machinery
today.

**[pause]**

One more column worth dwelling on: **extrapolation**.

Trees and forests say no, and that's not a tuning problem — it's structural. A tree predicts by
averaging the training values that landed in a leaf, so it physically cannot output a number
larger than the largest one it has seen. If your series is trending upward, a random forest will
forecast a flat line at the top of its training range, forever.

The linear models and the feedforward network say yes. A LASSO fit with a trend term will happily
project that trend forward. So will a ReLU network, past its last kink.

The RNN and LSTM row says *bounded*, and that's a real difference. Their hidden state is squashed
by tanh into the range minus one to one, so with a linear output layer the forecast can go past
the training maximum, but only up to a fixed ceiling set by the output weights.
*(Point at the footnote under the table.)*

Whether that is a *feature* depends entirely on whether the trend is real. Extrapolating a
genuine trend is the right answer. Extrapolating a temporary run-up is how forecasts embarrass
people. The model cannot tell the difference — that judgment is yours, and it's the kind of
thing a DAG in Lecture 12 helps you reason about.

---

# ▶ SLIDE 22 — Lecture 7: Key Takeaways

*(Six points. Read them at a steady pace; don't rush the last two.)*

One. A neuron is linear regression wrapped in a nonlinear activation — and without that
nonlinearity, stacking layers collapses back to a single linear model. With ReLU, each unit adds
one kink, which is how four units drew the W in the electricity year.

Two. An FFN stacks those layers. Width and depth set capacity, and capacity accumulates faster
than you expect. Twenty-two thousand parameters on five thousand rows.

Three. Training is one loop: loss, `backward()`, `step()`, on mini-batches, using Adam. You never
differentiate by hand. And keep the weights from the best epoch on validation, not the last one.

Four. Regularization is not new. Weight decay *is* Ridge; dropout and early stopping do the same
job by other means. That's the fifth appearance of one idea.

Five. For time series specifically: split by time, never shuffle before the split, and call
`zero_grad()` every batch.

And six — the one that matters most. An FFN on engineered features is a **competitor** to
XGBoost, not an upgrade. Architecture is what earns its keep, and that starts next week.

**[pause]**

Next time: CNN architectures. Convolution, pooling, and what a one-dimensional convolution does
to a time series.

---

# ▶ SLIDE 23 — References

*(Advance and close. No narration needed.)*
