# Changelog

## [2022.1]

### Added

- Closed-loop evaluation of a trained model on interactive scenarios using SUMO
- Continuous collision checking using convex hull

### Changed

- Reaching the end of road now regarded as is_time_out instead of is_offroad

### Fixed

- Various small bug fixes

### TODO Next release

- support for [rllib](https://docs.ray.io/en/latest/rllib-algorithms.html)
- support for [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/)
- support for multi-agent RL