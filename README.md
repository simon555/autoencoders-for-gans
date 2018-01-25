# autoencoders-for-gans


This repo contains several expriments about AE suited for GANS.

## Remakrs

As final activation, relu produces weird pixels in several images that cannot be detected by following the evolution of the losses.
To detect this caveat, you should visualize the images original/reconstructed and check it out one by one.

to prevent this issue, you could swith to sigmoid as final activation, the weird pixels disappear.
