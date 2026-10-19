"""Command-latency model and physical backup-braking trajectory check.

This is a *sampled*, finite-horizon safety filter; it is not a formal
collision-free guarantee. Obstacles are still assumed to follow known CV
motion. The actuator queue is shared by the planner and the simulated plant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Command, State, VehicleConfig, step

_EPS = 1e-9


@dataclass(frozen=True)
class ScheduledCommand:
    activate_at: float
    command: Command


@dataclass
class CommandDelayLine:
    """FIFO scheduled setpoints; plant and planner use identical event timing."""

    active: Command
    delay_s: float = 0.25
    pending: list[ScheduledCommand] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.delay_s < 0:
            raise ValueError('control delay must be nonnegative')

    def enqueue(self, now: float, command: Command) -> None:
        self.pending.append(ScheduledCommand(now + self.delay_s, command))
        self.pending.sort(key=lambda x: x.activate_at)

    def copy(self) -> 'CommandDelayLine':
        return CommandDelayLine(self.active, self.delay_s, list(self.pending))

    def advance_path(self, initial: State, end_time: float, vehicle: VehicleConfig,
                     dt: float) -> list[State]:
        """Advance with exact command-switch boundaries (up to integration dt)."""
        if dt <= 0:
            raise ValueError('dt must be positive')
        if end_time < initial.time - _EPS:
            raise ValueError('end_time precedes current state')
        path = [initial]
        state = initial
        while state.time < end_time - _EPS:
            while self.pending and self.pending[0].activate_at <= state.time + _EPS:
                self.active = self.pending.pop(0).command
            event = self.pending[0].activate_at if self.pending else end_time
            h = min(dt, end_time - state.time, max(0.0, event - state.time))
            if h < _EPS:
                # Numerical tie with event: process on next iteration.
                if self.pending:
                    self.active = self.pending.pop(0).command
                    continue
                break
            state = step(state, self.active, vehicle, h)
            path.append(state)
        return path


def braking_tail(initial: State, vehicle: VehicleConfig, dt: float) -> list[State]:
    """Maximum physical deceleration, maintaining the attained front steer.

    Brake starts *after* the committed prefix and proposed command window.
    This is not an instantaneous-stop or fixed Euclidean stopping radius.
    """
    if dt <= 0:
        raise ValueError('dt must be positive')
    if vehicle.braking <= 0:
        raise ValueError('vehicle.braking must be positive')
    path = [initial]
    state = initial
    hold_steer = state.steer
    while state.speed > _EPS:
        h = min(dt, state.speed / vehicle.braking)
        state = step(state, Command(0.0, hold_steer), vehicle, h)
        path.append(state)
    return path


@dataclass(frozen=True)
class BackupSegments:
    prefix: list[State]
    candidate: list[State]
    brake: list[State]

    @property
    def complete(self) -> list[State]:
        return self.prefix + self.candidate[1:] + self.brake[1:]


def backup_segments(initial: State, command: Command, line: CommandDelayLine,
                    vehicle: VehicleConfig, execute_s: float, dt: float) -> BackupSegments:
    """(1) already-committed controls, (2) new command, (3) full braking."""
    if execute_s <= 0:
        raise ValueError('execute_s must be positive')
    virtual = line.copy()
    prefix = virtual.advance_path(initial, initial.time + line.delay_s, vehicle, dt)
    state = prefix[-1]
    candidate = [state]
    until = state.time + execute_s
    while candidate[-1].time < until - _EPS:
        h = min(dt, until - candidate[-1].time)
        candidate.append(step(candidate[-1], command, vehicle, h))
    brake = braking_tail(candidate[-1], vehicle, dt)
    return BackupSegments(prefix, candidate, brake)


def stationary_hold(initial: State, duration_s: float, dt: float) -> list[State]:
    """Append hypothetical occupancy *after* stopping.

    The ego's position is held fixed while the obstacle CV trajectories keep
    advancing. This catches a moving target arriving at the stopping location
    several seconds after the car reaches speed zero. A finite holding horizon
    cannot be interpreted as perpetual safety after parking.
    """
    if duration_s < 0 or dt <= 0:
        raise ValueError("holding duration must be nonnegative and dt positive")
    if initial.speed > 1e-7:
        raise ValueError("hold segment must start at rest")
    states = [initial]
    end_time = initial.time + duration_s
    while states[-1].time < end_time - _EPS:
        t_next = min(states[-1].time + dt, end_time)
        states.append(State(initial.x, initial.y, initial.yaw, 0.0,
                            initial.steer, t_next))
    return states


def braking_then_hold(segments: BackupSegments, duration_s: float,
                      dt: float) -> list[State]:
    """Full delay/action/braking path plus post-stop dynamic occupancy check."""
    held = stationary_hold(segments.brake[-1], duration_s, dt)
    return segments.complete + held[1:]
