# Giving the sequence models the LASSO's features

**Status:** complete, 2026-10-29; mechanism test added 2026-10-31. Exploratory: these numbers are not in any slide or lab. Student write-up: `report.qmd`, rendered to `report.html` (`quarto render report.qmd` from this folder).

**Question (owner, during Lecture 9 Part 2 prep).** The 46-feature LASSO tops the course
scoreboard at 744 RMSE. It knew three things the Transformer in Lab 9 Part 2 does not: which
series it is forecasting, the calendar, and the forecast week's own SNAP days, event days and
price. Given those, and longer training, does the Transformer (or the LSTM, or the RNN) catch up?

**Setup.** Lab 9 Part 2's data, split, model and training loop, unchanged except for the switch
under test: the same 52-week windows of the M5 weekly panel, 5,160 training windows before
2015-05-17 and the 1,590 test windows after it (the same test block as Homework 4's LASSO). Three
seeds per cell. Variant A at seeds 1 and 2 reproduces the lab's recorded runs exactly (1,013 and
967), so the harness matches the lab.

| Feature set | What the model gets |
|---|---|
| A | the lab's four numbers per week |
| B | A, plus store and category as one-hot columns on every week |
| C | A, plus the forecast week's SNAP days, event days, price and calendar, joined at the head |
| D | B and C, plus each week's calendar |
| E | D without the forecast week's price (price may not be known in advance) |

**Findings.** Every model first reproduced its lab exactly (Lab 7 FFN 1,109 / 1,176 / 946; RNN 805;
LSTM 998; Transformer 1,013 / 967). Then, all means of three seeds:

1. Shuffling helped all four networks. The labs use `shuffle=False`, which serves training windows
   series by series. Shuffling them between batches (safe: every training window precedes the cutoff,
   and a seeded shuffle reproduces exactly at a given thread count) improved them at the labs' own
   lengths: FFN 1,077 to 893, RNN 820 to 752, LSTM 997 to 975, Transformer 950 to 859.
2. Under the labs' order, the series identity broke the Transformer (950 to 1,850): training error
   rose too, week-to-week error did not, and all the damage was in each series' level. Shuffled, it
   was harmless.
3. The LSTM was undertrained: 975 at 30 epochs, 773 at 100. A validation block (the last 52 training
   weeks) chose 139–148 of 150 epochs, which scored 813, so more was not simply better on the test weeks.
4. Extra features helped little. The forecast week's price and SNAP days are 99% and 91% predictable
   from the window; events are not (13%) but barely move weekly sales.
5. With training length and feature set both chosen on validation, the best network is the RNN with
   all features at 743 ± 19, a tie with the LASSO's 744. FFN 926, LSTM 783, Transformer 817. Lab 8's
   1D CNN (1,032) was not re-run.
6. The mechanism is only partly known (`scripts/test_mechanism.py`, `output/mechanism.csv`, final pass
   measured batch by batch). Test RMSE swings within a pass, more under the lab order (Transformer
   884–1,235) than shuffled (816–955). "The model ends tuned toward the last series" got little support:
   a series is better just after its own stretch in 15–19 of 30 series, against 14–15 in shuffled
   controls. Identity weights falling out of step got partial support: the typical level error on
   training windows was about 970 right after a series' stretch and 1,410 by the end of the pass.

Full table: `output/summary.md` (`scripts/summarize.py` rebuilds it from the results files).

    .venv/bin/python explorations/transformer-features/scripts/run_features.py            # lab order, 30 epochs
    SHUFFLE=1 .venv/bin/python explorations/transformer-features/scripts/run_features.py  # shuffled
    SHUFFLE=1 EPOCHS=100 MODEL=lstm .venv/bin/python ... run_features.py A B C D          # longer, other models
    MODEL=transformer .venv/bin/python ... test_mechanism.py A B                         # the last pass, batch by batch
