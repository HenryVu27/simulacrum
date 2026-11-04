# Concept import, first pass

Sparse autoencoder features of each mind's hidden state (lawful world, held-out episodes). Perceptual: R^2 >= 0.5 from current senses. Conceptual: R^2 >= 0.5 from named world variables. Unnamed candidate: R^2 < 0.3 from both, and 5-step autocorrelation > 0.8.

| mind | live features | perceptual | conceptual | unnamed candidates |
|---|---|---|---|---|
| human | 512 | 86 | 0 | 0 |
| dog | 512 | 81 | 0 | 0 |
| bat | 512 | 81 | 0 | 0 |
| smell_only | 456 | 9 | 0 | 33 |
| chimera | 512 | 68 | 1 | 1 |
| blind | 512 | 50 | 0 | 3 |
| hearing_only | 512 | 5 | 2 | 8 |
| vision_only | 512 | 43 | 0 | 0 |
