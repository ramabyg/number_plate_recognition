From this loss_box_reg curve:
It starts high (~0.20+), drops fairly quickly at the beginning.
Then from roughly 8k–10k onward, it keeps going down, but very slowly and with a lot of noise.
By 15k–25k, it’s basically just jittering in a narrow band—tiny improvements, nothing dramatic.
So together with loss_cls:
loss_cls: clearly converged, flat after ~10k
loss_box_reg: still decreasing a bit, but very slowly and noisily after ~10k
This tells us:
The model is mostly done learning by ~12k–15k iterations. Extra training to 25k is giving you tiny gains in regression loss, but not enough to move COCO AP—so it just overfits a bit and AP50/APs even drop slightly.
So: pushing iterations further is not the lever anymore. You need to change what the model can represent, not how long it trains.

