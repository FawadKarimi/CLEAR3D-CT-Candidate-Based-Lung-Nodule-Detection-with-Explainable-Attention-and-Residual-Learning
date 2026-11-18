# resnet18-constlr-001

Verified resnet18-lr10x-002 acceptance PASS, wall 934.5 seconds. Its epoch-2 learning rate had fallen to 1e-5.

This run kept group learning rate at 0.001 for all 180 updates. Worker PASS. Acceptance PASS. Wall 954.7 seconds. 0 skips.

Validation loss: 0.704850, 0.706878, 0.698371.
At probability 0.5 every epoch: accuracy 0.9609, macro F1 0.4900, class-1 precision/recall/F1 0/0/0.
Epoch 2 class-0 probabilities 0.4497 to 0.4764, mean 0.4652. Class-1 probabilities 0.4507 to 0.4760, mean 0.4723.
Highest class-1 F1 on this same validation slice is 0.1347 at threshold 0.473 (precision 0.073, recall 0.867). That threshold was not used for training.
