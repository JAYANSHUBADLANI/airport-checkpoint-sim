# Checkpoint staffing for Atlanta and Nashville
### June conditions. A recommendation, the numbers behind it, and what it rests on.

## The question

How many screening lanes should be open, half hour by half hour, to keep the 95th
percentile passenger wait under twenty minutes, and what does that cost against the
alternatives an operation would otherwise reach for.

I simulated the departures checkpoint at both airports from the actual June 2026 flight
schedule, and checked it against the volumes TSA recorded at those checkpoints. The
demand model was fitted on the week of 1 June and tested on the week of 7 June, which it
had not seen. On that held-out week it tracks hourly volume with a correlation of 0.95 at
Atlanta and 0.91 at Nashville, and gets the morning peak, which is what staffing turns
on, to within one percent at Atlanta.

## What I recommend

**Buy the insurance.** At both airports, staff to the plan built for ten percent more
passengers than the schedule implies, not to the plan built for the schedule itself.

At Atlanta that is **$102,400 a day** against $93,440 for the plan sized to the
schedule: an extra **$8,960 a day, 9.6 percent**. At Nashville it is **$43,520** against
$40,320: an extra **$3,200 a day, 7.9 percent**.

The shape of the day matters more than the total. Atlanta peaks at 07:00 and needs
around 41 lanes in the morning bank against 23 if the day were staffed flat.
Nashville peaks earlier, at 05:00 to 06:00, and needs 19. Both airports run at roughly
twice their daily mean in the busiest hour.

**Shorten the shifts before you buy more of them.** Moving Atlanta from eight hour
shifts starting every two hours to four hour shifts starting hourly takes the daily cost
from $113,920 to $93,440; two hour blocks starting on the half hour take it to $85,120.
That $28,800 spread is a 34 percent premium on the cheapest structure, decided entirely
by rostering rules rather than by anything to do with queueing.

**Do not spend money guarding against a single lane going down.** Losing a lane for two
hours at the peak is absorbed by every plan tested. It is not what breaks the checkpoint.

## The numbers

**A flat plan is not a plan.** Dividing the day's passengers by what a lane can clear
gives 23 lanes at Atlanta. Run that and the queue is at a 22 minute 95th percentile by
05:00, 64 minutes by 06:00, and never recovers: 290 minutes across the day, three
quarters of passengers reaching the front of the queue behind their own departure. No
later hour has the spare capacity to drain what the morning built.

**What each plan costs, averaged over nine days.**

| Atlanta | Cost per day | 95th pct wait | Worst wait |
| --- | --- | --- | --- |
| Optimised shift plan | $88,391 | 1.9 min | 6.4 min |
| Cheapest flat plan that works | $142,293 | 3.2 min | 4.4 min |
| Peak staffing all day | $149,653 | 0.7 min | 1.6 min |

Against a flat plan the optimised roster saves 38 percent at Atlanta and 34 percent at
Nashville; against staffing the peak all day, 41 percent at both. On a normal day all
three hold the twenty minute target, so the argument between them is about cost and about
what happens when the day is not normal.

| Atlanta, design day | Optimised for the schedule | Optimised for +10% |
| --- | --- | --- |
| As planned | 1.7 min | 0.4 min |
| Ten percent more passengers | **40 min**, 15% behind their flight | 2.1 min |
| Screening fifteen percent slower | **77 min**, 34% behind their flight | 9.6 min |
| A lane down two hours at the peak | 2.6 min | 0.4 min |

A plan sized tightly to capacity has a cliff, not a slope. Ten percent is well inside the
error on any forecast of passenger numbers, and being on the wrong side of it does not
mean a longer queue, it means a checkpoint that stops working. Nine and a half percent of
the daily cost buys out of both shocks tested, including the one it was not built for.

**One real caveat on the Atlanta figure.** The model treats Atlanta as a single queue.
It is six departure checkpoints, and a passenger in the wrong hall cannot use a spare
lane in another. Sharing the pooled plan's lanes across those six halls in the
proportions TSA recorded misses the target badly, the worst hall reaching a 36 minute
95th percentile, and buying it back hall by hall costs 14.4 percent more. **The honest
Atlanta figure is therefore a little over $106,000 a day, and the lane counts here are a
floor rather than a roster.** Nashville has one consolidated checkpoint and needs no such
adjustment.

## What this rests on, and what would change it

Four assumptions carry the answer, and all four sit in one configuration file.

**Who clears security** is the one I would most like to replace with a measurement. The
schedule says how many flights leave and when, not how many seats are on them or how many
passengers connect airside. I fit one relationship, checkpoint passengers per scheduled
seat, against what TSA screened, and let it change through the day, because morning banks
are mostly people starting a trip and evening banks are mostly people changing planes.
Forcing it to a single number made the model miss the Atlanta morning peak by sixteen
percent. Airline booking data would settle it, and it is the cheapest and most valuable
thing on this list to obtain.

**How fast a lane screens** is the most sensitive input. I assume about 160 passengers an
hour including secondary bag searches. A range of 12.5 to 17 seconds per passenger moves
the Atlanta bill from $84,480 to $103,040, and if the real figure is at the slow end the
insurance recommendation becomes urgent rather than prudent.

**When passengers turn up** matters least. Moving the show-up profile a quarter of an hour
either way changes cost by about one percent.

**What a lane-hour costs** is an assumption, but only the ratios between plans depend on
it, so the recommendation does not move with it.

Two other things would change the answer. If a hall physically cannot hold 41 lanes at
once, the peak has to be flattened another way: earlier check-in, more PreCheck
throughput, or moving departures out of the 07:00 bank. And a tighter service target
would not change much, which is worth knowing because it is counter-intuitive: tightening
from thirty minutes to five buys no extra lanes at Atlanta. What the plan is really
buying is throughput, which is why this memo argues about margin and rostering rather
than about queueing.
