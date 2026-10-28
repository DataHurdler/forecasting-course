# Lab 9 Part 2 — Recording Script

**ECON 8310: Business Forecasting · Attention — and the Line You Cannot Forget**

Lab: `Labs/Lecture09_Part2_lab.qmd` (4 steps) · Measured runtime: **~20 minutes** of narration
(the in-room version is budgeted at 40)

---

## How to use this document

- **`▶ STEP n — Title`** matches the lab's own headings exactly.
- *Italic parentheticals* are stage directions. **[pause]** is a beat; **[STOP — learner works]**
  is where you tell the viewer to pause the video and do something.
- Deterministic (`torch.manual_seed` per fit, `shuffle=False`).
- **Slow lab.** Attention is O(T²) in sequence length and the window is 52 weeks, so a single fit
  takes over a minute and Step 3 trains four. Plan the cut before you record.

### Nothing needed correcting

Every claim in the `.qmd` matched the measured output, including the two that carry the lab:
the positional-encoding penalty (**292 RMSE**, exactly the deck's figure) and "roughly fifteen
times the parameters of the RNN" (67,329 / 4,545 = **14.8**). The scoreboard's carried-in values
— RNN 805, LSTM 998, CNN 1,032, LASSO 744, XGBoost 781, seasonal naive 2,152 — all agree with
their source labs and with Lecture 13.

### In-room language that needs replacing

Step 1's "read the picture", Step 3's "your turn", and the three closing **Discuss** blocks.

---

# ▶ OPENING

*(Show the rendered lab, at the top of the document.)*

Last week, in Lab 9 Part 1, our recurrent models read the window one step at a time. The
information from week three reached week fifty by being carried through forty-seven state updates,
one after another. An LSTM's gates exist to help that information survive the trip, but the whole
history still had to live in one vector of fixed size.

Attention doesn't work that way. Every position looks at every other position **at once**, so week
three reaches week fifty in a single operation. The path between any two points in the sequence is
just one step long.

That's a genuine advance, and it's why the last decade of machine learning looks the way it does.

**[pause]**

Today you'll build a Transformer encoder for forecasting. Then you'll delete one line from it and
watch it fall apart.

That line is the one PyTorch doesn't add for you. It isn't in `nn.TransformerEncoderLayer`, no
default supplies it, and nothing checks whether you remembered it. Forgetting it is the most common
way this architecture fails silently, and by now you know what "silently" means in this course.

---

# ▶ SETUP

*(Run the setup cell.)*

The panel, the cutoff date and the standardization are all the same as before. The one difference
from last week's recurrent lab is that the window stays at **fifty-two weeks** throughout, because a
long window is where attention is supposed to earn its keep. If direct paths to distant weeks are
worth anything, a full year of history is where we should see it.

That choice is also why this lab is slow. Attention compares every position with every other
position, so the cost grows with the *square* of the window. With fifty-two weeks, that comes to two
thousand seven hundred and four comparisons for each example, in each head, in each layer.

---

# ▶ STEP 1 — Positional encoding, before the model

*(Show the explanation, then the heat-map cell.)*

Let's start with the problem, because the solution makes no sense without it.

Attention computes a **weighted average** over positions, and an average doesn't care about order.
Reverse the sequence, shuffle it, or deal it out like a deck of cards, and each week's attention
output stays exactly the same. It simply moves along with its week. The technical term for that is
permutation-equivariant, and the mechanism is built that way.

**[pause]**

Our model reads its forecast from the last position, so it always knows which week is the most
recent. Everything before that week is a bag. For forecasting, that's a catastrophe. Picture two
histories. In the first, sales were three thousand two weeks ago and nine thousand a year ago. In
the second, sales were nine thousand two weeks ago and three thousand a year ago. Those are the same
bag of numbers, and they point to completely different futures.

So position has to be put into the input itself, before attention ever sees it. The standard way to
do that is with a set of sine and cosine waves at different frequencies.

*(Run the figure.)*

**[STOP — learner works]** — *(replaces "read the picture")*

Pause the video on the heat map, and check two properties for yourself before I name them.

*(Resume.)*

The first property is that every column is distinct. Each week has its own signature running down
the sixty-four dimensions, so the model can tell position three from position forty-seven.

The second, which people miss, is that nearby columns look similar. Week twenty and week
twenty-one have almost the same pattern, while week twenty and week fifty don't. That means the
encoding carries the *relative* distance between weeks, not just an arbitrary label for each slot.
So a model can learn "about four weeks apart" as a single idea, wherever that pair of weeks falls in
the window, rather than memorizing every pair of positions.

That's why we use sine and cosine waves rather than, say, just numbering the weeks from one to
fifty-two.

---

# ▶ STEP 2 — Build the encoder

*(Show the model class, and point at the `forward` method.)*

The whole architecture is four objects. There's a linear layer that lifts the four input channels up
to sixty-four dimensions, then the positional encoding, then two `TransformerEncoderLayer`s, and
finally a linear head that reads the last position.

*(Point at `h = h + self.pe`.)*

That line is what this whole lab is about. It's a **single addition**, and everything else in this
class is supplied by PyTorch.

```
Transformer — 67,329 parameters, test RMSE 1,013
```

**[pause]**

That's sixty-seven thousand parameters. Hold that against last week, when the vanilla RNN had
**four thousand five hundred forty-five** and scored eight-oh-five. So attention arrives with roughly
**fifteen times** the parameters of the model it has to beat.

---

# ▶ STEP 3 — Delete the line

*(Start the cell, which runs four fits and takes several minutes, then talk while it runs.)*

It's the same model, the same data and the same two seeds. Setting `use_pe=False` removes that one
addition in `forward`, and it changes nothing else.

```
positional encoding  mean  sd         runs
                yes   990  23  [1013, 967]
                 no  1282   8 [1290, 1274]
```

*(Let the table sit for a moment.)*

**[pause]**

The gap is **two hundred ninety-two** in RMSE. That's about twenty-nine percent worse, from deleting
one line.

*(Show the callout.)*

And now here's the sentence I want you to take out of this course, never mind this lab: **nothing
raised an exception.**

The model without positional encoding trained for thirty epochs. Its loss went down, and it
converged. It returned forecasts in the right units and the right shape, with a plausible RMSE that
even beats the seasonal-naive benchmark of twenty-one fifty-two. If you had nothing to compare it
with, you would ship it.

What it was actually doing was reading last week plus an **unordered bag of the other fifty-one**.
It couldn't tell two weeks ago from a year ago. It knew what the year contained, and nothing about
when.

**[pause]**

Now, look at the `sd` column before you ask whether this is noise. The two conditions have spreads
of twenty-three and eight, against a gap of two hundred ninety-two, so the gap is twelve to
thirty-six times the noise. Lab 7, where you trained your first network, taught you that a gap
smaller than the run-to-run spread isn't evidence. This one is about as clear as a result gets.

*(Point at the last paragraph of the callout.)*

Here's the practical warning. `nn.TransformerEncoderLayer` hands you attention, the residual
connections, layer normalization and the feed-forward sublayer. It does **not** hand you position.
There's no default for it, and nothing checks that you added it. In a language model you would
notice, because the output would be word salad. In a forecaster the output is a number, and numbers
always look fine.

**[STOP — learner works]**

Now pause the video, fill in the blank, and express the penalty as a percentage of the RMSE with the
encoding. You should get two hundred ninety-two in RMSE, which is twenty-nine percent worse.

---

# ▶ STEP 4 — Where attention lands

*(Run the scoreboard.)*

```
LASSO, 46 features (HW4)           744
XGBoost (HW4)                      781
Vanilla RNN (Lab 9 Pt 1)           805
Transformer (today)                990
LSTM (Lab 9 Pt 1)                  998
1D CNN (Lab 8, 26-week window)    1032
Seasonal naive                    2152
```

**[pause]**

Let's read it honestly. The Transformer beats the 1D CNN from Lab 8, which read twenty-six-week
windows. It **ties the LSTM**: nine ninety against nine ninety-eight is a gap of eight, while the
run-to-run spread is twenty-three, so you can't tell those two apart. And it loses to the vanilla
RNN by one hundred eighty-five, with roughly **fifteen times** as many parameters.

Remember, too, that last week's lead of the RNN over the LSTM was measured at this same
thirty-epoch budget. When both were trained for a hundred epochs, on twenty-six-week windows, they
tied.

None of this is a failed implementation. It's the architecture working correctly on a problem it
wasn't designed for.

*(Show the paragraph about the advantage at scale.)*

And that's the point worth carrying away. **Every advantage attention has is an advantage at
scale.**

Parallel computation across positions is transformative when your sequence is thousands of steps
long and your training data runs to billions of tokens. An RNN has to walk the sequence one step at
a time, and a Transformer doesn't. Direct paths between distant positions matter when the signal
genuinely sits far back and has to survive the trip.

Neither of those describes five thousand one hundred sixty training windows of weekly retail demand,
where the answer is mostly in the last few weeks. You're paying for a mechanism whose benefits don't
exist at this size, with parameters that have to be estimated from data you don't have.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Here are three questions to work through on your own before you go.

The first is about the missing line. You've now met four failures that produce a plausible number
and no error. The first was using `KFold` on a time series, back in Lecture 1. Then came a forgotten
`zero_grad()` in Lab 7, a reshape where a transpose belonged in Lab 8, and today, a missing
positional encoding. What do they have in common, and what single practice would have caught all
four?

**[pause]**

Here's the answer I'd steer you toward. None of them is a coding error. Each one is a valid program
that means something other than what you intended, and no type check can catch that. Only a
benchmark you trust, computed on held-out time, can catch it, and that's why the seasonal-naive row
has been on every scoreboard since Lecture 1.

The second question is about the architecture. Describe a forecasting problem where you'd expect the
Transformer to beat everything else here. Be specific about two things: how long the sequences are,
and how much data there is.

The third question looks ahead to Homework 5. Part 2 of that homework asks you to fit all three
sequence models and explain the ranking, and you've now seen that ranking. What this lab doesn't
answer is whether the ranking would survive on a different series. So name one property of a series
that would most likely flip it.

**[pause]**

I'll close with one thought that's worth more than the ranking. Everything you've built in the last
four weeks, the feedforward network, the convolutional one, the recurrent ones and now attention,
lost to a LASSO with forty-six coefficients from Homework 4. That isn't an argument against neural
networks. It's an argument about **matching the model to the problem**. Five thousand one hundred
sixty overlapping windows of weekly retail demand aren't the conditions any of these architectures
was designed for, and the honest thing to do is to say so, and to pick the model that fits the data
you actually have.

That's the argument Lecture 13, the synthesis at the end of the course, makes with the whole
course's scoreboard in front of it.

---

# Appendix — expected output

Deterministic; `torch.manual_seed` per fit, `shuffle=False`. Slow — attention is O(T²) at a
52-week window; the lab states about 70 seconds per fit and Step 3 trains four.

| Quantity | Value |
|---|---|
| Windows | 6,750 of 52 weeks — train **5,160**, test 1,590 |
| Transformer | **67,329** parameters (RNN: 4,545 — a **14.8×** ratio) · seed 1 → 1,013 |
| With positional encoding | mean **990** · sd 23 · runs [1013, 967] |
| Without positional encoding | mean **1282** · sd 8 · runs [1290, 1274] |
| Penalty for the missing line | **292 RMSE** — **29%** worse; 12–36× the run-to-run spread |
| Attention comparisons per example | 52² = **2,704** per head per layer |
| Scoreboard | LASSO 744 · XGBoost 781 · **RNN 805** · Transformer 990 · LSTM 998 · CNN 1,032 · naive 2,152 |
| Transformer vs LSTM | 8 apart against sd 23 — **indistinguishable** |
| Transformer vs RNN | 185 behind, on 14.8× the parameters |

**Three things to know before recording.**

*Step 3 is the lab, and its force comes from what did not happen.* The number 292 matters less
than the absence of an exception, a warning, or a NaN. Deliver the "nothing raised" paragraph
slowly — it is the fourth instance of the course's silent-failure theme and the one where the
mechanism is most obviously invisible.

*Do not overstate the Transformer-versus-LSTM comparison.* 990 against 998 is 8 RMSE with a spread
of 23. They tie. The deck says "ties the LSTM" and means it; anything stronger in either direction
is unsupported.

*The scoreboard's ordering is the honest headline and it should not be softened.* Four neural
architectures across four weeks — feedforward, convolutional, recurrent, attention — and the best
of them still sits behind a 46-coefficient LASSO. That is the setup for Lecture 13, and hedging it
here weakens the synthesis later.
