# Lecture 8 — Recording Script

**ECON 8310: Business Forecasting · CNN Architectures**

Deck: `Slides/Lecture08_CNNs.pdf` (23 pages) · Measured runtime: see the timing guide

---

## How to use this document

- **`▶ SLIDE n — Title`** marks where to advance. The number is the PDF page.
- *Italic parentheticals* are stage directions. **[pause]** means stop for a beat.
- Slide 10 plants an objection about pooling and slide 19 pays it off with a measurement. Do not
  resolve it early — the delay is what makes the number land.
- Slides 7, 8 and 14 are pictures. Talk to the figure, not the caption: the arithmetic on 7, the
  two hand-set filters on 8, and the 64 filters nobody set on 14 are one argument in three steps.

---

# ▶ SLIDE 1 — Title page

Welcome to Lecture 8, on CNN architectures.

Last week, in Lecture 7 on neural networks, we built a feedforward network, and I told you, fairly
bluntly, that it was a competitor to XGBoost rather than an upgrade. It took the same flat table as
input, it had no notion of order, and on our data it lost.

Today, the architecture starts doing real work. A convolutional network makes an *assumption*
about the structure of its input: that nearby things are related, and that distant things usually
aren't. And it builds that assumption into the shape of the model, rather than into features you
engineered by hand.

**[pause]**

That's the theme of the next two lectures, starting with this one. It isn't a better optimizer,
and it isn't more layers. It's a model whose *shape* encodes what kind of data it expects.

---

# ▶ SLIDE 2 — Lecture Outline

We'll take this in four parts. First comes convolution itself: what a filter is, and what it costs.
Then pooling, which is where I want you to be suspicious. Third, two decades of image
architectures, compressed into the three ideas that outlived them. And finally, one-dimensional
CNNs on our own time series, with an honest measurement at the end.

**[pause]**

A word on why we're spending time on image architectures in a forecasting course, since that's a
fair question to have.

There are two reasons. The first is that three of their ideas transfer directly, and you'll
recognize one of them again next week, inside a completely different model. The second is that
the history is a worked example of the thing this course keeps insisting on. Every one of these
architectures won by encoding a *better assumption* about its data, not by being bigger. That's
the lesson that transfers, and it happens to be easier to see in pictures than in demand curves.

---

# ▶ SLIDE 3 — Section divider: Convolution — The Core Idea

This first section is about convolution itself, and it comes down to two ideas: look at a small
window, and look for the same thing everywhere. Those two ideas are the whole architecture.

---

# ▶ SLIDE 4 — Why Convolution? A Dense Layer Wastes Its Weights

Let's start with why a fully connected layer fails on structured input, because the failure is
dramatic.

An image two hundred twenty-four pixels on a side has just over fifty thousand pixels. If you feed
that into a single hidden layer of a thousand neurons, that layer needs **fifty million weights**,
and that's for one layer.

Put that next to the network from Lecture 7, last week, which had twenty-two thousand six hundred
fifty-seven parameters in total.

**[pause]**

But the cost isn't the worst part. The waste is.

That fully connected layer has no notion that neighboring pixels are related and distant ones
usually aren't. It treats pixel one and pixel fifty thousand as equally likely to interact. And it
can't reuse what it learns. A pattern it learns in the top-left corner is stored completely
separately from the same pattern in the bottom-right, so it has to learn everything twice.

---

# ▶ SLIDE 5 — Why Convolution? Two Ideas

*(Point at the picture first, the left side and then the right.)*

On the left is a dense layer, with eight inputs and four outputs, and every output wired to every
input. That's thirty-two wires, and every wire is its own weight to learn.

On the right is a convolution, where each output is wired to just three neighboring inputs. And
look at the colors. The red wire, the blue wire and the green wire are the *same three weights*,
reused at every position. So there are thirty-two numbers to learn on the left, and only three on
the right.

*(Point at the key box.)*

A convolutional layer fixes both problems with exactly those two ideas. The first is **local
connectivity**: each neuron sees only a small window, which is called its receptive field. The
second is **weight sharing**: the same filter is applied at every position in the input.

You get far fewer parameters, and a pattern learned anywhere is recognized everywhere.

**[pause]**

Here's the analogy that makes it stick. You don't memorize the letter "A" separately for every
position on the page. You learn it once, and you apply that one recognizer everywhere you look. A
filter does exactly that.

And notice that this is the same *kind* of move we've made before, in a different costume. Weight
sharing is a constraint. We're telling the model, before it sees any data, that a pattern occurring
at position ten and the same pattern at position two hundred should be treated identically. That's
a strong assumption, and it's what buys the reduction in parameters.

Constraints that encode true structure make models better. Constraints that encode false structure
make them worse, and no amount of training fixes that. Hold that thought, because by the end of
this lecture we'll have an example of each, on the same dataset.

---

# ▶ SLIDE 6 — How a Convolutional Layer Works

For a sequence, which is what we care about, a one-dimensional convolution slides a filter of width
K along the series, and at each position it takes a dot product.

That's the formula on the slide. Honestly, the formula is less useful than the picture on the next
slide, which shows a small window sliding along and computing one number at each stop.

*(Walk the four hyperparameters.)*

The first hyperparameter is the kernel size, which is the width of the filter. A bigger kernel
means a wider receptive field, so more context for each output, and it also means more parameters.

The stride is how far the filter jumps between positions. A stride of two halves the length of your
output.

Padding puts zeros at the edges. Setting the padding to "same" keeps the output the same length as
the input, which saves you a lot of arithmetic.

And the number of filters is how many *different* patterns you want to learn. That number becomes
the channel count of the output.

**[pause]**

Here's what to hold onto. Each filter learns **one local pattern**, such as an edge, a spike, or a
seasonal dip. What comes out is called a feature map, and its high values mark the places where
that pattern was found.

Two things follow from that, and they're worth stating explicitly, because they shape how you read
the results later.

First, the filter's weights are **learned**, not designed. You don't tell it to look for a
promotional spike. You give it room for thirty-two patterns, and gradient descent decides what
those patterns are. That's the promise of learned features, the thing that's supposed to replace
your hand-built lag columns.

Second, and this is the constraint, a filter can only see K steps at a time. Stack two layers, and
the second one sees a window of windows, so the reach grows, but it grows *slowly*. That reach is
the receptive field from the last slide, and I'm coming back to it now because it's going to be the
single most important number in this lecture.

---

# ▶ SLIDE 7 — Convolution, Step by Step

*(Walk the figure from top to bottom. Do the first dot product out loud.)*

Here's the whole operation, on numbers small enough to check in your head. There's a series of ten
values, and a filter of width three, with the weights one, five and one.

Put the filter over the first three values, which are one, four and five. Multiply position by
position and add: one times one, plus four times five, plus five times one. That gives
**twenty-six**, the first output, the one boxed in red.

Now slide one step to the right and do it again. Four, five and three give thirty-two. And so on,
down the series.

*(Point at the "valid" row.)*

If you keep only the positions where the filter fits completely, you get eight outputs from ten
inputs. That's ten minus three, plus one, and it's called **valid** padding.

*(Point at the "same" row and the dashed zeros.)*

If you want the output to be the same length as the input, pad a zero at each end, which are the
dashed boxes, and you get all ten. That's **same** padding, the setting from the previous slide.
Notice that the two end values, nine and twenty, are computed partly from zeros that aren't really
there. The edges of a feature map are always its least trustworthy part.

*(Point at the bottom row.)*

And the last row is a preview of pooling. Take the ten outputs in pairs, and keep the larger of
each pair, so ten numbers become five.

**[pause]**

Now, notice what this particular filter *does*. It has a big weight in the middle and small weights
on either side, so it mostly reports the middle value, lightly smoothed by its neighbors. A
different filter, say minus one, zero, plus one, would report the *change* across the window
instead. It's the same sliding and the same arithmetic, but a completely different pattern gets
detected.

That's what "each filter learns one local pattern" means. And in a trained network, nobody picks
those three numbers. Gradient descent does.

---

# ▶ SLIDE 8 — What a Filter Sees

This is the same operation in two dimensions. Instead of a window of three weeks, the filter covers
a three-by-three patch of pixels, and instead of sliding along a line, it slides over every pixel
of a photograph.

*(Point at the two small grids at the bottom.)*

There are two filters here, and I set both of them by hand. The first is negative on the left and
positive on the right, so it fires wherever brightness changes from left to right, which is a
**vertical** edge. The second is the same filter turned on its side, negative above and positive
below, so it fires on **horizontal** edges.

*(Point at the two feature maps.)*

And here's what each one sees. The vertical-edge map is all short upright strokes: the fronts of
products, and the price tags. The horizontal-edge map is the shelves themselves, long streaks
running toward the back of the aisle.

**[pause]**

It's the same photograph and the same sliding arithmetic, but you get two completely different
pictures of it. That's what a feature map is: one filter's answer to the question, "where is my
pattern?" A convolutional layer with thirty-two filters asks thirty-two of those questions at once.

*(Point at the photo, then the line at the bottom.)*

And I chose this photograph on purpose. That's a grocery aisle on the eighteenth of March, twenty
twenty, the week panic buying emptied shelves across the country. A demand forecast built on that
store's own history had no way to see it coming, because nothing in its past looked like it.

Keep that picture in your head. We come back to it in Week 15, in the synthesis lecture, when we
talk about what forecasting can't do.

---

# ▶ SLIDE 9 — Section divider: Pooling

The next section is pooling. Pooling shrinks the feature map and buys robustness, but at a price
that matters more for forecasting than it does for images.

---

# ▶ SLIDE 10 — Pooling: Downsampling Feature Maps

Pooling reduces a feature map by summarizing each small region with one number. Max pooling takes
the largest value in the window, and average pooling takes the mean.

*(Walk the figure from left to right.)*

On the left is the horizontal-edge map from two slides ago. The red square is an eight-by-eight
patch of it, around a shelf edge. In the middle is that patch as numbers, cut into two-by-two
blocks, with the largest value in each block shaded. On the right are the sixteen winners. That's a
two-by-two max pool, and sixty-four values have become sixteen, a quarter as many.

Find the row of sixes, sevens and a nine across the middle of the patch. That's the shelf edge, and
it's still there after pooling, as six, seven, nine, six. The fine detail is gone, but the fact
that there's an edge, roughly *here*, survives.

Pooling buys you two things. The first is **fewer parameters downstream**, because the later layers
see a smaller input. The second is **translation invariance**. Shift the input slightly and the
pooled output barely changes, so the network recognizes a cat whether it's left or right of center.

**[pause]**

*(Now slow down. This box is the setup for slide 19.)*

And here's where I want you to get suspicious.

Translation invariance is a **feature** for images and a **problem** for forecasting.

Think about what it's actually doing. If a demand spike moved three weeks later, pooling treats
that as noise to be smoothed away. But that shift isn't a nuisance. For a forecaster, *when the
spike happened* is very often the entire thing you're trying to predict.

Pooling deliberately discards **where** in the window a pattern occurred. For an image classifier,
that's exactly right. For a time series, position is frequently the signal.

**[pause]**

Hold that objection, and don't resolve it yet. Later in this lecture, we're going to measure
precisely what it costs.

---

# ▶ SLIDE 11 — Section divider: CNN Architecture Evolution

The next section covers two decades of image architectures, and the three ideas from them that
survived.

---

# ▶ SLIDE 12 — LeNet to VGG: Depth and Small Filters

LeNet-5, in nineteen ninety-eight, established the template that everything since has followed:
alternating convolution and pooling layers, then fully connected layers at the end. It had roughly
sixty thousand parameters, and it read handwritten digits on bank checks. That was a real deployed
system, back in the nineties.

VGG, in twenty fourteen, scaled that up and made one sharp simplification. It used **only**
three-by-three filters, but it stacked a lot of them.

Why does that work? Two stacked three-by-three filters see the same receptive field as one
five-by-five filter. But they use fewer parameters, eighteen weights against twenty-five, and you
get an extra nonlinearity in between them. The stack isn't strictly more expressive, though. The
VGG authors describe it as a regularization, because it constrains what the filter can be while
adding that nonlinearity.

*(Point at the key box.)*

That's the lesson that carried forward: **depth built from small filters beats width built from
large ones.** You get fewer weights and more nonlinearity. It's why nearly every modern
architecture is deep and narrow, rather than shallow and wide.

**[pause]**

VGG's cost was its size. It had around a hundred thirty-eight million parameters, and most of them
were sitting in the final fully connected layers, not in the convolutions at all. That's exactly
the problem the next two architectures went after.

That detail is worth pausing on, because it's a good lesson about where cost actually lives. The
convolutions, the part everyone talks about, were relatively cheap, precisely because of weight
sharing. The expensive part was the ordinary dense layers bolted on at the end, which have no
sharing at all.

When you profile a model and find the bottleneck, it's very often in the least interesting
component. That's true of forecasting pipelines too. The modeling is rarely what costs you. The
data preparation is.

---

# ▶ SLIDE 13 — Inception and ResNet: Depth That Trains

Inception and ResNet were two different attacks on that problem.

Inception asked a good question: why should a layer commit to one filter size at all? Its module
runs one-by-one, three-by-three and five-by-five convolutions **in parallel**, and it concatenates
the results, letting the network decide which scale matters. The one-by-one convolutions do
something clever. They cut the channel count first, so the expensive convolutions operate on a
smaller input.

ResNet solved a different problem, and it's the one that transfers to us.

Past a certain depth, deeper networks were getting **worse**. That wasn't overfitting, because the
training error was worse too. And it wasn't vanishing gradients either. The authors checked, and
the gradients were healthy. The deeper plain networks were simply harder to optimize.

*(Point at the diagram.)*

The fix is one line, and the diagram is that line. The input, x, forks. One copy goes through the
block, which is a convolution, a ReLU and another convolution, and what comes out is F of x. The
other copy, the red path over the top, skips the block entirely. The two meet at the plus sign, so
y equals F of x, **plus x**. That's called a residual connection, or a skip connection.

**[pause]**

There are two ways to read that. First, each block now learns only the *difference* from its input,
rather than a whole new representation, and that's usually an easier job. That's what the bracket
under the boxes says. Second, the plus-x gives the gradient a direct path backward, unimpeded by the
layers in between.

That single change made networks of a hundred layers and more trainable. Skip connections now
appear in essentially every deep architecture, including the Transformers we'll meet in Lecture 9
Part 2, two weeks from now. And, as it turns out, there's one inside the LSTM, which predates ResNet
by eighteen years.

**[pause]**

I want to draw out why the addition matters, because we'll use this exact argument again next week,
and it's easier to absorb twice.

When a gradient travels backward through a chain of layers, each layer *multiplies* it by
something. Multiply enough numbers below one together, and you get essentially zero. The signal
dies before it reaches the early layers, so they never learn.

Addition doesn't do that. The plus-x term hands the gradient a route home that skips the
multiplication entirely. Repeated multiplication shrinks a signal, and repeated addition doesn't.

Remember that sentence. Next week, in Lecture 9 Part 1, you'll see the LSTM's cell state doing
precisely the same thing. And there, unlike in ResNet, fading gradients really are the problem it
solves. The LSTM was invented eighteen years earlier, for time rather than for depth.

---

# ▶ SLIDE 14 — What a Trained CNN Learns

Back on slide eight, I set two filters by hand. Here's what happens when nobody does.

*(Point at the grid on the left.)*

This is the first layer of ResNet-18, the architecture from the last slide, after training on one
point two million labeled photographs. There are sixty-four filters, each seven pixels square, in
color. Nobody designed any of them. Every number in every one came out of gradient descent.

Look at what it arrived at. A lot of them are edge detectors, light on one side and dark on the
other, at every angle: horizontal, vertical and diagonal. The network rediscovered the filters we
wrote down by hand, and then it went further, because it needed more angles than we gave it. Most
of the rest respond to color.

*(Point at the four maps.)*

These are the four filters boxed in red, run over the same grocery aisle. Different filters find
different structure. Some pick out the shelf lines, some pick out the products, and one mostly
tracks how bright each region is.

**[pause]**

That's the promise of learned features in its purest form. Nobody told this network that edges
matter. It found out, because edges are what made its task easier.

And it's exactly the promise a 1D CNN makes on a time series. Hand it the raw window, and its
filters will find whichever local shapes help predict next week, whether that's a spike, a dip or a
ramp. Whether that promise pays off on *our* data is a separate question, and it's the one we
answer at the end of this lecture.

---

# ▶ SLIDE 15 — What Carries Over to Time Series

These architectures were built for images, on problems that look nothing like weekly demand
forecasting. Even so, three of their ideas transfer.

The first is small stacked filters. Two stacked filters of width three see five weeks of context,
with a nonlinearity between them. The weight saving doesn't come along, though, because that was a
two-dimensional effect. In one dimension, two width-three filters need six weights, where one
width-five filter needs five.

The second is skip connections. It's the same fix for the same problem, a deep stack that's hard to
train.

The third is learned features. You stop hand-engineering lag windows, and you let the filters find
whichever local shapes actually matter.

**[pause]**

*(And now name the one that doesn't.)*

One idea does **not** transfer, and it's the one we flagged twenty minutes ago, on the pooling
slide: pooling for translation invariance.

An image classifier shouldn't care where the cat is. A forecaster usually should care very much
where the spike was.

**[pause]**

So we're carrying two assumptions into the next section, and they aren't equally good.

The first is locality, the idea that nearby time steps are more related than distant ones. That's
largely *true* of demand data, since last week matters more than the week before, so it's a good
assumption.

The second is translation invariance, the idea that where a pattern occurred doesn't matter. That's
largely *false* for forecasting. It's a good assumption for images, and a bad one here.

A CNN gives you both, bundled together, because they came from the same field. Part of using one
well is knowing which half you actually wanted.

---

# ▶ SLIDE 16 — Section divider: 1D CNNs for Time Series

The last main section runs the same machinery along one dimension, on our own data, and it ends
with an honest measurement.

---

# ▶ SLIDE 17 — Why Use a 1D CNN for Forecasting?

Here's the setup. A one-dimensional CNN takes a **window** of the recent past, twenty-six weeks in
our case, across several channels, and it predicts the next week. Each channel is one variable over
time: units, SNAP days, events and price.

The appeal is that you stop choosing lags. Instead of deciding that lag one, lag four and lag
fifty-two are the ones that matter, you hand the model the whole window and let the filters work
out which local shapes predict what comes next.

It has two advantages, and both are genuinely real. It's far cheaper than a fully connected network
on the same window, which is weight sharing again. And it handles **many parallel series**
naturally, because the same filters apply to every store-category at once.

**[pause]**

*(Now the catch, and flag it as the reason for the result on slide 19. Point at the chart.)*

But the catch is the receptive field, and it's a serious one. The chart makes it concrete. That's
food sales at store CA_1, and the red dot on the right is the week we want to forecast, Christmas
week of twenty fifteen.

The gray band is the twenty-six-week window the model is given. The red band is what one output of
a width-five-then-width-three stack actually sees, and that's **seven weeks**. That's all.

*(Point at the open circle on the left.)*

And there's the most useful single number for forecasting Christmas week: the same week last year,
which dipped too. It's fifty-two weeks back, so it isn't just outside the seven-week receptive
field. It's outside the window altogether. Annual seasonality sits fifty-two weeks away, and no
reasonable stack of small filters is going to reach that far.

So you have two options. You can add dilation, which means filters with gaps in them, so they reach
further for the same parameter count. Or you can feed the annual lag in as its own channel, which is
a confession that you're doing feature engineering after all.

**[pause]**

And notice what that second option costs you rhetorically. The entire pitch for a CNN was, "stop
choosing lags, and let the filters find the structure." The moment you hand it the fifty-two-week
lag as a channel, because the receptive field can't reach it, you've given that pitch back. You're
now doing feature engineering *and* paying for a neural network.

That isn't a reason never to use one. It's a reason to be clear-eyed about what the architecture is
actually buying you on a given problem.

Remember both numbers: seven weeks for each filter output, and twenty-six weeks in the window. The
window is the one that explains the results two slides from now.

---

# ▶ SLIDE 18 — 1D CNN Forecasting Pipeline in PyTorch

The model itself takes six lines. There are two convolutions with ReLUs, a pooling step, a flatten,
and a linear layer down to one number.

One thing trips everyone up. The input shape is batch, then channels, then time, so channels come
**before** time. That's the transpose of how the data naturally sits in a dataframe, and it's also
the opposite of what `nn.LSTM` wants next week. You'll get this wrong at least once. If you pass
time before channels, PyTorch stops you with an error about the channel count, which is the easy
case. The version that doesn't stop you is in the lab.

Everything from Lecture 7, last week, still applies. Use Adam with weight decay, call `zero_grad()`
on every batch, set `shuffle=False`, and split by time, not at random.

*(Point at the muted note.)*

And notice what the adaptive average pooling layer does. It averages across the *entire* time
axis, collapsing it to a single number for each filter. That's called **global average pooling**.
The alternative, a **flatten head**, skips it and gives every position in the window its own
weight.

Watch what that choice costs on the next slide.

---

# ▶ SLIDE 19 — Does It Actually Work?

*(This is the payoff slide. Walk the table, then land both punchlines.)*

This is our weekly panel, with twenty-six-week windows, four channels, thirty store-category
series, and the same test block as Homework 4. So these numbers are directly comparable to work
you've already done.

The LASSO from Homework 4, with forty-six engineered features, scores seven forty-four. XGBoost
scores seven eighty-one, and the random forest from Homework 3 scores eight ninety-nine.

Then come the CNNs. The flatten head scores one thousand thirty-two. Global max pooling scores
fourteen eighteen, and global average pooling scores fifteen thirty-four.

Seasonal naive, at the bottom, scores twenty-one fifty-two.

**[pause]**

There are two things to take from this.

The first is that **every CNN variant loses to a linear model with forty-six coefficients**, and
not narrowly. Part of the reason is plain. These runs use a twenty-six-week window, so last year's
same week isn't in the window at all, while LASSO was simply handed it as a column.

The second is the one I promised you.

*(Point at the last paragraph.)*

Swapping average pooling for a flatten head, which keeps *where* each pattern occurred instead of
averaging it away, is worth about **five hundred** in RMSE. That's with three seeds each, against a
run-to-run spread of roughly eighty, so the gap is real.

That's the objection from slide ten, measured. Pooling threw away position, and position was
carrying signal.

**[pause]**

Now, I want to be careful about how far to push that conclusion, because there's a tempting
overreach here.

This does **not** mean pooling is bad. It means pooling encodes an assumption, that position is
irrelevant, and on this dataset that assumption is false. Run the same experiment on image
classification, and the sign would flip, decisively.

That's the shape of nearly every result in this course. There's no ranking of methods that holds
across problems. There's a match, or a mismatch, between what the method assumes and what the data
is. Your job is to know what your method assumes. That's much harder than knowing how to call it,
and much more valuable.

---

# ▶ SLIDE 20 — Section divider: Key Takeaways

That brings us to the key takeaways: what convolution is for, and what it isn't for.

---

# ▶ SLIDE 21 — CNN Architecture Summary

*(Table first, then the two paragraphs. The second is the more useful one.)*

Here's the lineage in one table. LeNet gave us the template. VGG gave us small filters, stacked
deep. Inception gave us parallel filter sizes, with one-by-one convolutions to control the cost.
ResNet gave us skip connections, which made real depth trainable. And the 1D CNN is that same
machinery, run along a time axis.

One small note under the table: VGG, Inception and ResNet each appeared a year before the papers we
cite for them, so the years in the table run one ahead of the references.

Reach for a 1D CNN when local shape matters more than exact position, when you have many parallel
series, or as a cheap feature extractor that feeds something else.

That third use is underrated, incidentally. You don't have to use a CNN as the whole model. A
common production pattern is to let convolutions compress a long raw window into a handful of
learned summary features, and then feed *those* into gradient boosting alongside your engineered
columns. You get the pattern discovery without betting the forecast on it.

Don't reach for one when the dominant signal is at a long lag, when you already have
well-engineered tabular features, or when a stakeholder needs to read the model and understand it.

**[pause]**

And be honest about what that second list covers. On the evidence we just saw, it describes most of
what we forecast in this course.

**[pause]**

Which raises the obvious question: why teach it, then?

The first answer is that the receptive field problem isn't a fact about CNNs. It's a fact about
*this data*, a strong annual cycle seen through a short window. Change either one, and the answer
changes. The second answer is that convolution is a component, not just a model. It shows up inside
architectures you'll meet later, in production forecasting systems, and in every paper on the
subject. Knowing what it assumes is what lets you predict, before fitting anything, whether it will
work on a problem you haven't seen yet.

That prediction, made in advance from structure rather than from trying it, is the skill this
course is actually training.

---

# ▶ SLIDE 22 — Lecture 8: Key Takeaways

Let me leave you with five takeaways.

First, convolution is local connectivity plus weight sharing. A pattern is learned once and
detected everywhere, at a fraction of the parameters.

Second, pooling buys translation invariance, which an image classifier wants and a forecaster
usually doesn't. The measured cost on our data is about five hundred in RMSE.

Third, the history of these architectures reduces to three ideas that transfer: small filters
stacked deep, skip connections for trainable depth, and learned rather than hand-engineered
features.

Fourth, a 1D CNN takes a window across channels and predicts the next step. Its filters see a few
weeks at a time, and reaching a year back needs a long window and many weights.

And the last one: on our panel, every CNN variant lost to a LASSO with forty-six coefficients.
Architecture has to fit the problem. It isn't a ladder you climb.

**[pause]**

Next week, in Lecture 9 Part 1, we turn to RNNs and LSTMs. Those are architectures built to carry
state through time, which is precisely what the receptive field couldn't do.

---

# ▶ SLIDE 23 — References

*(Advance and close. No narration needed.)*
