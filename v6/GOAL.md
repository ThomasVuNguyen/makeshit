# Goal

To rigorously simulate the bipedal robot (6 joints, MG996R servos, RP2040) in MuJoCo by creating the most accurate MJCF model possible through many iterations.

# Structure

documentation/ - contains all the docs on the joint & design
BeeWalker/ - contains the python simulation scripts and the model.xml (MJCF) files.
documentation/shit-examples/ - contains past failed attempts

# What to try

Iteratively refine the MJCF model to match the physical hardware's dynamics and control logic. 