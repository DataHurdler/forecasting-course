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

*(Screen: rendered lab, top of document.)*

Last session's recurrent models read the window one step at a time. Week 3's information reached
week 50 by being carried through forty-seven consecutive state updates, and anything that survived
that journey survived it by luck.

Attention does not work that way. Every position looks at every other position **at once**. Week 3
reaches week 50 in a single operation. The path length between any two points in the sequence is
one.

That is a genuine architectural advance, and it is why the last decade of machine learning looks
the way it does.

**[pause]**

Today you build a Transformer encoder for forecasting, and then you delete one line from it and
watch it fall apart.

That line is the one PyTorch does not add for you. It is not in `nn.TransformerEncoderLayer`, no
default supplies it, and nothing checks whether you remembered. Forgetting it is the most common
way this architecture fails silently — and by now you know what "silently" means in this course.

---

# ▶ SETUP

*(Screen: run setup.)*

Same panel, same cutoff, same standardization. One difference from the recurrent lab: the window
is fixed at **52 weeks** throughout, because a long window is where attention is supposed to earn
its keep. If direct long-range paths are worth anything, a full year of history is where we should
see it.

That choice is also why this lab is slow. Attention compares every position with every other
position, so the cost grows with the *square* of the window. Fifty-two weeks means 2,704 pairwise
comparisons per example per head per layer.

---

# ▶ STEP 1 — Positional encoding, before the model

*(Screen: the explanation, then the heat-map cell.)*

Start with the problem, because the solution makes no sense without it.

Attention computes a **weighted average** over positions. And an average does not care about
order — reverse the sequence, shuffle it, deal it like a deck of cards, and each week's attention
output is **identical**; it just moves with its week. The mechanism is permutation-equivariant by
construction.

**[pause]**

Our model reads its forecast from the last position, so it always knows which week is most
recent. Everything before that is a bag. That is a catastrophe for forecasting. "Sales were 3,000
two weeks ago and 9,000 a year ago" and "sales were 9,000 two weeks ago and 3,000 a year ago" are
the same bag of numbers and completely different futures.

So position has to be put into the input itself, before attention ever sees it. The standard
device is a set of sinusoids at different frequencies.

*(Run the figure.)*

**[STOP — learner works]** — *(replaces "read the picture")*

Pause on the heat map and check two properties before I name them.

*(Resume.)*

**One: every column is distinct.** Each week has its own signature down the 64 dimensions, so the
model can tell position 3 from position 47.

**Two, and this is the part people miss: nearby columns look similar.** Week 20 and week 21 have
almost the same pattern; week 20 and week 50 do not. That means the encoding carries *relative*
distance, not just an arbitrary label per slot. A model can learn "about four weeks apart" as a
concept rather than memorizing every pair of indices — which is what makes the scheme work at
positions it never saw in training.

That is why sinusoids rather than, say, just numbering the weeks 1 to 52.

---

# ▶ STEP 2 — Build the encoder

*(Screen: the model class. Point at the `forward` method.)*

The whole architecture is four objects: a linear layer lifting 4 channels to 64 dimensions, the
positional encoding, two `TransformerEncoderLayer`s, and a linear head reading the final position.

*(Point at `h = h + self.pe`.)*

**That line is the subject of the lab.** One addition. Everything else in this class is supplied
by PyTorch.

```
Transformer — 67,329 parameters, test RMSE 1,013
```

**[pause]**

Sixty-seven thousand parameters. Hold that against last week: the vanilla RNN had **4,545** and
scored 805. So attention arrives with **fifteen times** the parameters of the model it needs to
beat.

---

# ▶ STEP 3 — Delete the line

*(Screen: start the cell — four fits, several minutes — then talk.)*

Same model, same data, same two seeds. `use_pe=False` removes that one addition in `forward` and
changes nothing else.

```
positional encoding  mean  sd         runs
                yes   990  23  [1013, 967]
                 no  1282   8 [1290, 1274]
```

**[pause]** — *let it sit.*

**292 RMSE.** About 29% worse, from deleting one line.

*(Screen: the callout.)*

And now the sentence I want you to take out of this course, never mind this lab: **nothing raised
an exception.**

The model without positional encoding trained for thirty epochs. The loss went down. It converged.
It returned forecasts in the right units, in the right shape, with a plausible RMSE that is better
than the seasonal-naive benchmark of 2,152. If you had no comparison to make, you would ship it.

What it was actually doing is reading last week plus an **unordered bag of the other 51.** It
could not distinguish two weeks ago from a year ago. It knew what the year contained and nothing
about when.

**[pause]**

Look at the `sd` column before anyone asks. The two conditions have spreads of 23 and 8 against a
gap of 292 — twelve to thirty-six times the noise. By the standard Lab 7 set, this is about as
unambiguous as a result gets.

*(Point at the callout's last paragraph.)*

The practical warning: `nn.TransformerEncoderLayer` hands you attention, residual connections,
layer normalization and the feed-forward sublayer. It does **not** hand you position, there is no
default, and nothing checks that you added it. In a language model you would notice, because the
output would be word salad. In a forecaster the output is a number, and numbers always look fine.

**[STOP — learner works]**

Fill in the blank and express the penalty as a percentage. You should get 292 RMSE, 29% worse.

---

# ▶ STEP 4 — Where attention lands

*(Screen: run the scoreboard.)*

```
LASSO, 46 features (HW4)     744
XGBoost (HW4)                781
Vanilla RNN (Lab 9 Pt 1)     805
Transformer (today)          990
LSTM (Lab 9 Pt 1)            998
1D CNN (Lab 8)              1032
Seasonal naive              2152
```

**[pause]**

Read it honestly. The Transformer beats the CNN. It **ties the LSTM** — 990 against 998, a gap of
8 when the run-to-run spread is 23, so those two are indistinguishable. And it loses to the
vanilla RNN by 185, on **fifteen times** the parameters.

Nothing here is a failed implementation. This is the architecture working correctly on a problem
it was not designed for.

*(Screen: the "advantage at scale" paragraph.)*

And that is the point worth carrying. **Every advantage attention has is an advantage at scale.**

Parallel computation across positions is transformative when your sequence is thousands of steps
and your corpus is billions of tokens — an RNN has to walk the sequence one step at a time, and a
Transformer does not. Direct paths between distant positions matter when the signal genuinely sits
far back and must survive the journey.

Neither describes 5,160 training windows of weekly retail demand, where the answer is mostly in
the last few weeks. You are paying for a mechanism whose benefits do not exist at this size, with
parameters that have to be estimated from data you do not have.

---

# ▶ BEFORE YOU LEAVE

**[STOP — learner works]** — *(replaces the in-room "Discuss")*

Three questions for the board.

**The missing line.** You have now met four failures that produce a plausible number and no error:
`KFold` on a time series, a forgotten `zero_grad()`, a transposed axis, and a missing positional
encoding. What do they have in common, and what single practice would have caught all four?

**[pause]** — *the answer worth steering toward: none of them is a coding error — every one is a
valid program that means something other than what you intended. Types cannot catch that. Only a
benchmark you trust, computed on held-out time, can — which is why the seasonal-naive row has been
on every scoreboard since Lecture 1.*

**The architecture.** Describe a forecasting problem where you would expect the Transformer to
beat everything else here. Be specific about two things: sequence length and data volume.

**For Homework 5.** Part 2 asks you to fit all three sequence models and explain the ranking. You
have now seen it. What the lab does not answer is whether the ranking would survive on a different
series — name one property of a series that would most likely flip it.

**[pause]**

And one closing thought that is worth more than the ranking. Everything you have built in the last
four weeks — feedforward, convolutional, recurrent, attention — lost to a 46-coefficient LASSO
from Homework 4. That is not an argument against neural networks. It is an argument about
**matching the model to the problem**: 5,160 overlapping windows of weekly retail demand is not
the regime any of these architectures was designed for, and the honest thing to do is say so and
pick the model that fits the data you actually have.

That is the argument Lecture 13 makes with the whole course's scoreboard in front of it.

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
