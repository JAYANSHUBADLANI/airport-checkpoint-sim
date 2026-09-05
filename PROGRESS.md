# Progress log

A running note to myself on where this project stands, what I decided and why, and what
is still open.

## Where it stands

Phases 1 to 4 are built and run end to end from `make demo`.

**Phase 1, data and demand.** Schedule comes from the BTS On-Time Reporting Carrier
On-Time Performance file for June 2026, pulled straight from the PREZIP path. The origin
throttles a single connection to about 13 KB a second, which would have taken close to
an hour, so the fetch script pulls the file in parallel byte ranges instead. Seats are
not in that file and BTS T-100 has no stable direct URL, so seats per departure are an
assumption by operating carrier, labelled as one everywhere.

**Phase 2, the simulation.** SimPy, two stages, document check then screening lanes,
lanes opening and closing on the staffing slot boundary, lognormal service times, a
share of passengers pulled for a secondary bag search that occupies the lane.

**Phase 3, optimisation.** A queueing approximation turns an arrival rate into a lane
requirement, a set covering MILP buys shifts to meet it at least cost, and the resulting
plan is then run through the simulation. Where the simulation says the target is missed
the requirement is tightened and the MILP re-solved.

**Phase 4, robustness and write-up.** Demand uplift, a lane outage at the peak, slower
service, and a plan hardened against the uplift so the memo can price the insurance.

## Decisions I made and the reasons

**June 2026, not an arbitrary month.** I wanted validation data, not a face validity
check. The TSA FOIA reading room publishes weekly checkpoint throughput by airport and
clock hour, and the weeks of 1 to 6 June and 7 to 13 June 2026 line up with a BTS month
that has been published. Choosing the month to match the validation data was the single
most useful decision in the project.

**ATL and BNA.** ATL is the obvious large hub. BNA is the interesting mid-size choice
because TSA reports exactly one checkpoint there, called BNA Central, so the one
checkpoint per airport simplification is not a simplification at all at BNA. It is a
real simplification at ATL, which has eight, and I say so rather than hiding it.

**I excluded two ATL checkpoints from the observed side.** The F Arrival checkpoint
screens international arrivals re-entering the sterile area to connect onwards, and the
Private Terminal serves general aviation. Neither is a departing passenger the scheduled
departure file can produce. Both are named in the config, not buried in code.

**I stopped calling it a connecting share.** The first version fitted one number, the
share of passengers who do not clear security, and the answer for ATL came out far too
low to be a connecting share. It was absorbing three other things: international
departures missing from the domestic file, crew included in the TSA count, and any error
in my seats and load factor assumptions. Nothing in the data separates them. So the
model fits a screening yield instead, checkpoint passengers per scheduled domestic seat,
and the docstring lists exactly what that absorbs. The prior connecting share stays in
the config as the assumption it always was.

**The yield varies with departure hour.** With one constant yield the model put the
busiest hour of the day at BNA in the afternoon when the airport actually peaks in the
morning, and under-predicted the ATL morning peak by close to twenty percent while
over-predicting the evening by more. That is the peak the whole staffing question turns
on, so getting it wrong was not acceptable. A hub's morning banks are mostly people
starting a trip and its evening banks are mostly people changing planes, so the yield is
a straight line in scheduled departure hour, two parameters, fitted by least squares.

**Three fitted parameters per airport, and a held out week.** Yield intercept, yield
slope, and one show-up timing shift are fitted on 1 to 6 June and on nothing else. Every
validation number quoted comes from 7 to 13 June, which the fit never saw. The flat
yield version is scored alongside so the extra parameter has to visibly earn its place.

**Verification and validation stay apart.** Verification asks whether the engine
computes a queue correctly and is answered twice: against a hand computable
deterministic queue, and against Erlang C with Poisson arrivals and exponential service.
Validation asks whether the number of people arriving looks like the number who turned
up, and is answered against TSA throughput. They are different questions and they live
in different files.

## Two things I got wrong and had to fix

**A SimPy race was eating passengers.** Servers waited on `Store.get() | timeout` and
cancelled the get when the timeout won. If an item was matched to that get at the same
simulation instant, the cancel came too late, the get had already been given the
passenger, and nobody read it. The deterministic test caught it: every fifth passenger
had a missing wait and the queue behind them caught up by exactly one service time. I
replaced the whole thing with an explicit dispatcher over a deque, which is both correct
and faster. The lesson is that the hand computable test was worth more than any amount
of staring at the code.

**I nearly concluded the engine was biased.** An early M/M/c test came out twenty five
percent below the analytic mean and I spent a while assuming the simulation was wrong.
It was not. A single simulated day of a checkpoint that size carries a standard
deviation on the mean wait of about twenty percent, so six replications simply could not
resolve the difference. Running the same arrivals and the same service draws through
both the SimPy engine and an independent Lindley recursion settled it: the two agree to
the last floating point bit. The verification test now states its Monte Carlo interval
instead of asserting a fixed tolerance and hoping.

## Where it landed

The headline numbers, all from the full run:

- Both airports run at roughly twice their daily mean in the busiest hour. ATL peaks at
  07:00 around 6,600 checkpoint arrivals, BNA at 05:00 to 06:00 around 2,800.
- A plan sized to the daily average breaks at 05:00 and never recovers: a 290 minute
  95th percentile at ATL, three quarters of passengers behind their own flight.
- The optimised shift plan costs $88,391 a day at ATL against $142,293 for the cheapest
  flat plan that works and $149,653 for peak staffing all day.
- The optimiser met the target on the first round on all eighteen airport-days, because
  the queueing approximation turns out to be conservative here rather than optimistic.
  It assumes exponential service, and screening is not that variable.
- The wait target is barely binding. Tightening it from thirty minutes to five buys no
  extra lanes at ATL. The roster rules cost far more: shift length and start granularity
  span 34 percent of the ATL bill.
- Splitting ATL into its six real departure checkpoints and sharing the pooled plan's
  lanes out in proportion misses the target badly, with the worst hall at a 36 minute
  95th percentile. Buying it back costs 14.4 percent more.
- Robustness is the sharpest result. A lane outage at the peak is a non-event. Demand ten
  percent over the schedule takes the optimised ATL plan to a 40 minute 95th percentile,
  and fifteen percent slower screening takes it to 77. A plan hardened against the demand
  case costs 9.6 percent more and holds through everything tested.

## Open

- The screening yield still misses at BNA in the middle of the day, under-predicting the
  early afternoon. A straight line in departure hour is the simplest thing that fixes
  the morning and evening, and I would rather leave a visible residual than add
  parameters until the residual disappears.
- The optimiser's inner approximation is stationary within a thirty minute slot. The
  simulation is not, and the gap between them is the reason the feedback loop has to
  exist at all. A response surface fitted to the simulation would close more of that gap
  in one step, at the cost of a much less transparent constraint.
- ATL pools eight real checkpoints into one queue. Pooling makes servers look more
  efficient than they are, so the ATL lane counts are a floor, not a plan.
- No check-in or bag drop stage, no boarding, no arrivals process, and no staff breaks
  beyond what the shift rules encode.
