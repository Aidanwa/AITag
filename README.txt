Tagger-v1 v random runner:
    learns to find runner very quickly, average of ~14 moves after ~35-40k, then progress stagnates

runner-v1 v tagger-v1 training:
    starts at ~15 turns, very quickly jumps to around 60, steadily climbs to the maximum of 200 by 5k iterations
    and consistently wins after that.

Tagger-v2 vs runner v1:
    rapidly learns to beat runner v1, gets time back down to around 15

Tagger vs Runner both training:
    Tagger getting number of steps down consistently until about 2000 iterations when the runner increases steps again steadily until
    5000 iterations where steps average near 200 consistently