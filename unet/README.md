## The UNet model attempt

These files are an attempt to use a unet design to directly label pixels. It was not succesful,
but after learning from the subsection model it could be worth another attempt.

The model was training to identify foreheads directly, not whole faces. It never learned
to label areas, only individual pixels, and it was unable to use most other features
beyond just the raw temperature value.
