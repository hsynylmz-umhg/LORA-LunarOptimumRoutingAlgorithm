"""Pure navigation geometry for the ENU Webots rover.

World and robot axes: +X forward/east, +Y left/north, +Z up.
Angles are radians, counterclockwise from +X. Lidar samples run left to right.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

WHEEL_RADIUS_M = 0.16
WHEEL_TRACK_M = 0.60
MAX_WHEEL_SPEED = 6.0
LIDAR_FORWARD_M = 0.22
LIDAR_UP_M = 0.22


def angle_error(target: float, current: float) -> float:
    return (target - current + math.pi) % (2 * math.pi) - math.pi


def compass_yaw(north: Sequence[float]) -> float:
    """World yaw of a +X-forward compass in an ENU world."""
    if len(north) != 3 or not all(math.isfinite(v) for v in north):
        raise ValueError("Compass measurement must contain three finite values")
    if math.hypot(north[0], north[1]) < 1e-8:
        raise ValueError("Compass has no horizontal north component")
    return math.atan2(north[0], north[1])


def bearing(position: Sequence[float], target: Sequence[float]) -> float:
    return math.atan2(target[1] - position[1], target[0] - position[0])


def wheel_speeds(position, yaw, target, speed_limit=0.64):
    """Differential drive with a stationary pivot for large heading errors."""
    values = (*position, yaw, *target, speed_limit)
    if not all(math.isfinite(v) for v in values) or speed_limit < 0:
        return 0.0, 0.0
    distance = math.dist(position, target)
    if distance < 1e-6:
        return 0.0, 0.0
    error = angle_error(bearing(position, target), yaw)
    omega = max(-1.2, min(1.2, 1.8 * error))
    linear = min(speed_limit, 1.2 * distance) * max(0.0, math.cos(error))
    if abs(error) > math.radians(50):
        linear = 0.0
    left = (linear - omega * WHEEL_TRACK_M / 2) / WHEEL_RADIUS_M
    right = (linear + omega * WHEEL_TRACK_M / 2) / WHEEL_RADIUS_M
    peak = max(abs(left), abs(right), MAX_WHEEL_SPEED)
    return left * MAX_WHEEL_SPEED / peak, right * MAX_WHEEL_SPEED / peak


def scan_points(ranges, resolution, field_of_view, max_range):
    """Yield (range, relative angle) for usable returns, across all layers.

    Positive angle is left. Infinity is an expected no-return reading; NaN,
    negative values are ignored.
    """
    if resolution < 1 or not math.isfinite(field_of_view) or max_range <= 0:
        return
    for index, distance in enumerate(ranges):
        if not math.isfinite(distance) or not 0 < distance < max_range:
            continue
        column = index % resolution
        angle = field_of_view / 2 - field_of_view * column / max(1, resolution - 1)
        if resolution == 1:
            angle = 0.0
        yield float(distance), angle


def project_return(position, yaw, distance, angle):
    """Project a horizontal scan using the world builder's .22m front mount."""
    return (position[0] + LIDAR_FORWARD_M * math.cos(yaw) + distance * math.cos(yaw + angle),
            position[1] + LIDAR_FORWARD_M * math.sin(yaw) + distance * math.sin(yaw + angle))


def front_clearance(points, half_angle=math.radians(50)):
    front = [(distance, angle) for distance, angle in points if abs(angle) <= half_angle]
    left = min((d for d, a in front if a >= 0), default=math.inf)
    right = min((d for d, a in front if a <= 0), default=math.inf)
    return min(left, right), left, right


def telemetry_waypoints(waypoints, max_points=1500):
    """Bound a UDP route without dropping its start or destination."""
    points = list(waypoints)
    if len(points) > max_points:
        indices = [round(i * (len(points) - 1) / (max_points - 1)) for i in range(max_points)]
        points = [points[i] for i in indices]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]
