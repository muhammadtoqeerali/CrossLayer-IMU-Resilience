# Candidate Fault Taxonomy

Status:

NOT FROZEN

## Sensor faults

Measurement-value:

- additive noise
- constant bias
- gradual drift
- scale-factor error
- clipping/saturation
- frozen channel

Availability:

- axis loss
- intermittent dropout
- frame loss
- complete outage

Timing:

- delay
- jitter
- irregular frame interval

Configuration/geometric:

- orientation error
- misalignment
- axis/sign errors only where physically justified

## Compute faults

Parameters:

- quantized weight corruption

Activations:

- intermediate activation corruption

Memory/state:

- intermediate buffer corruption

Persistence:

- transient
- persistent

Multiplicity:

- single bit
- multiple bit
- burst or word-level only where justified

## Combined faults

Combined experiments contain independently defined:

- sensor fault specification
- compute fault specification

Both must remain recoverable from experiment metadata.
