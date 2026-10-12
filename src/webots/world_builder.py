"""Generate a self-contained Webots R2025a world for a LORA mission.

Coordinate contract: ENU, terrain[row][column] is at
(offset_x + column * cell_size, offset_y + row * cell_size, height).
The rover's local +X is forward, +Y is left and +Z is up. All four
wheel joints rotate about +Y, so positive motor velocity drives forward.
No external PROTO, texture or sound downloads are required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

WHEEL_RADIUS = 0.16
WHEEL_TRACK = 0.60
WHEEL_CENTER_Z = -0.08
LIDAR_TRANSLATION = (0.22, 0.0, 0.22)
ROBOT_NAME = "LORA Rover"


def _number(value: float) -> str:
    return format(float(value), ".10g")


def _positive(value: object, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _point(value: Sequence[float], name: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain x and y")
    point = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{name} must contain finite coordinates")
    return point


def _look_at_rotation(position: Sequence[float], target: Sequence[float]) -> tuple[float, float, float, float]:
    """Aim the native Webots +X camera at target with projected +Z upright.

    Webots R2025a WbViewpoint::moveViewpointToObject uses q * (1, 0, 0)
    for the viewing direction. Compose Rz(yaw) * Ry(pitch), then express
    that quaternion as the Viewpoint's normalized axis-angle rotation.
    """
    dx, dy, dz = (target[i] - position[i] for i in range(3))
    if math.hypot(dx, dy, dz) < 1e-12:
        raise ValueError("Camera position and target must differ")
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    w, x, y, z = cy * cp, -sy * sp, cy * sp, sy * cp
    sine = math.hypot(x, y, z)
    if sine < 1e-12:
        return 0.0, 0.0, 1.0, 0.0
    return x / sine, y / sine, z / sine, 2 * math.atan2(sine, w)


def _wheel(name: str, x: float, y: float) -> str:
    """Anchors and endpoint origins coincide; cylinder +Z maps to axle +Y."""
    position = " ".join(_number(v) for v in (x, y, WHEEL_CENTER_Z))
    return f'''    HingeJoint {{
      jointParameters HingeJointParameters {{
        anchor {position}
        axis 0 1 0
        dampingConstant 0.01
      }}
      device [
        RotationalMotor {{
          name {json.dumps(name)}
          maxVelocity 6
          maxTorque 8
          acceleration 8
        }}
      ]
      endPoint Solid {{
        translation {position}
        rotation 1 0 0 -1.57079632679
        name {json.dumps(name + " solid")}
        contactMaterial "rover tire"
        children [
          Shape {{
            appearance PBRAppearance {{
              baseColor 0.13 0.14 0.16
              roughness 1
              metalness 0
            }}
            geometry Cylinder {{ radius {_number(WHEEL_RADIUS)} height 0.09 subdivision 24 }}
          }}
        ]
        boundingObject Cylinder {{ radius {_number(WHEEL_RADIUS)} height 0.09 subdivision 24 }}
        physics Physics {{ density -1 mass 0.8 }}
      }}
    }}'''


def build_world(
    heightmap: Iterable[Iterable[float]],
    mission: Mapping[str, object],
    path: Path,
    obstacles: Iterable[Mapping[str, float]] = (),
) -> None:
    """Write a validated, runnable world, preserving heightmap row order.

    Required mission values are ``cell_size``, ``offset_x``, ``offset_y``,
    ``start`` and ``goal``. ``mission_file`` is the controller JSON argument;
    if omitted, it defaults to ``data/mission.json`` in the Webots project's
    parent repository. Optional obstacles use world x/y and radius/height.
    Invalid maps or coordinates fail before replacing an existing world.
    """
    rows = [list(map(float, row)) for row in heightmap]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("heightmap must contain at least 2 rows and 2 columns")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("heightmap must be rectangular")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("heightmap must contain only finite heights")
    spacing = _positive(mission["cell_size"], "cell_size")
    offset_x, offset_y = _point((mission["offset_x"], mission["offset_y"]), "offset")
    start = _point(mission["start"], "start")
    goal = _point(mission["goal"], "goal")

    def surface(x: float, y: float) -> float:
        col, row = (x - offset_x) / spacing, (y - offset_y) / spacing
        if not (0 <= col <= width - 1 and 0 <= row <= len(rows) - 1):
            raise ValueError(f"world coordinate ({x}, {y}) lies outside the heightmap")
        c0, r0 = min(int(col), width - 2), min(int(row), len(rows) - 2)
        dc, dr = col - c0, row - r0
        return (
            rows[r0][c0] * (1 - dc) * (1 - dr)
            + rows[r0][c0 + 1] * dc * (1 - dr)
            + rows[r0 + 1][c0] * (1 - dc) * dr
            + rows[r0 + 1][c0 + 1] * dc * dr
        )

    start_height, goal_height = surface(*start), surface(*goal)
    yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    # The wheel bottom is 0.24 m below the body origin. Start 0.11 m
    # above its resting position so the four independently placed wheels settle.
    spawn_height = start_height + WHEEL_RADIUS - WHEEL_CENTER_Z + 0.11
    camera_position = (start[0] - 4.0, start[1] - 5.0, start_height + 4.2)
    camera_target = (start[0], start[1], start_height + WHEEL_RADIUS - WHEEL_CENTER_Z)
    camera_rotation = _look_at_rotation(camera_position, camera_target)
    path = Path(path).resolve()
    mission_file = Path(mission.get("mission_file", path.parents[2] / "data" / "mission.json")).resolve()
    heights = "\n".join("          " + " ".join(_number(v) for v in row) for row in rows)
    wheels = "\n".join(
        _wheel(name, x, y)
        for name, x, y in (
            ("front left wheel", 0.24, 0.30),
            ("front right wheel", 0.24, -0.30),
            ("back left wheel", -0.24, 0.30),
            ("back right wheel", -0.24, -0.30),
        )
    )
    markers = []
    for name, point, height, color in (
        ("START", start, start_height, "0.1 0.65 1"),
        ("GOAL", goal, goal_height, "0.2 1 0.3"),
    ):
        markers.append(f'''# Visual marker only: no collision geometry.
DEF {name}_MARKER Pose {{
  translation {_number(point[0])} {_number(point[1])} {_number(height + 0.012)}
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor {color} emissiveColor {color} roughness 1 metalness 0 }}
      geometry Cylinder {{ radius 0.24 height 0.02 subdivision 24 }}
    }}
  ]
}}''')
    obstacle_nodes = []
    for index, obstacle in enumerate(obstacles):
        x, y = _point((obstacle["x"], obstacle["y"]), "obstacle")
        radius = _positive(obstacle["radius"], "obstacle radius")
        height = _positive(obstacle["height"], "obstacle height")
        obstacle_nodes.append(f'''DEF OBSTACLE_{index} Solid {{
  translation {_number(x)} {_number(y)} {_number(surface(x, y) + height / 2)}
  name "obstacle {index}"
  contactMaterial "lunar regolith"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.28 0.25 0.22 roughness 1 metalness 0 }}
      geometry DEF OBSTACLE_GEOMETRY_{index} Cylinder {{ radius {_number(radius)} height {_number(height)} subdivision 24 }}
    }}
  ]
  boundingObject USE OBSTACLE_GEOMETRY_{index}
}}''')
    world = f'''#VRML_SIM R2025a utf8
# LORA mission {json.dumps(str(mission.get("run_id", "mission")))}
# ENU: X east / Y north / Z up; heightmap rows increase along +Y.
# Self-contained world: all geometry, materials and robot devices are native nodes.
WorldInfo {{
  title "LORA - Lunar Navigation MVP"
  coordinateSystem "ENU"
  gravity 1.62
  basicTimeStep 32
  FPS 60
  randomSeed {int(mission.get("seed", 1))}
  contactProperties [
    ContactProperties {{
      material1 "rover tire"
      material2 "lunar regolith"
      coulombFriction [ 0.9 ]
      bounce 0
      softERP 0.2
      softCFM 0.0001
      bumpSound ""
      rollSound ""
      slideSound ""
    }}
  ]
}}
Viewpoint {{
  orientation {' '.join(_number(v) for v in camera_rotation)}
  position {' '.join(_number(v) for v in camera_position)}
  near 0.05
  far 5000
  follow {json.dumps(ROBOT_NAME)}
  followType "Tracking Shot"
  followSmoothness 0.5
}}
Background {{ skyColor [ 0.025 0.03 0.055 ] }}
DirectionalLight {{
  direction -0.4 -0.5 -1
  color 1 0.97 0.9
  intensity 1.5
  ambientIntensity 0.35
  castShadows TRUE
}}
DEF LUNAR_SURFACE Solid {{
  translation {_number(offset_x)} {_number(offset_y)} 0
  name "lunar surface"
  contactMaterial "lunar regolith"
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.52 0.50 0.47 roughness 1 metalness 0 }}
      geometry DEF TERRAIN_GEOMETRY ElevationGrid {{
        xDimension {width}
        yDimension {len(rows)}
        xSpacing {_number(spacing)}
        ySpacing {_number(spacing)}
        thickness 1
        height [
{heights}
        ]
      }}
    }}
  ]
  boundingObject USE TERRAIN_GEOMETRY
}}
{chr(10).join(markers)}
{chr(10).join(obstacle_nodes)}
DEF LORA_ROVER Robot {{
  translation {_number(start[0])} {_number(start[1])} {_number(spawn_height)}
  rotation 0 0 1 {_number(yaw)}
  name {json.dumps(ROBOT_NAME)}
  controller "lora_controller"
  controllerArgs [ {json.dumps(mission_file.as_posix())} ]
  supervisor TRUE
  synchronization TRUE
  selfCollision FALSE
  children [
    Shape {{
      appearance PBRAppearance {{ baseColor 0.84 0.84 0.79 roughness 0.5 metalness 0.2 }}
      geometry Box {{ size 0.65 0.48 0.18 }}
    }}
    Pose {{
      translation 0.21 0 0.095
      children [
        Shape {{
          appearance PBRAppearance {{ baseColor 0.98 0.49 0.05 roughness 0.6 metalness 0 }}
          geometry Box {{ size 0.18 0.43 0.01 }}
        }}
      ]
    }}
    GPS {{ name "gps" }}
    Compass {{ name "compass" }}
    InertialUnit {{ name "inertial unit" }}
    Lidar {{
      translation {' '.join(_number(v) for v in LIDAR_TRANSLATION)}
      name "lidar"
      horizontalResolution 181
      fieldOfView 3.14159265359
      verticalFieldOfView 0.01
      numberOfLayers 1
      near 0.05
      minRange 0.05
      maxRange 12
      noise 0
    }}
{wheels}
  ]
  boundingObject Box {{ size 0.65 0.48 0.18 }}
  physics Physics {{ density -1 mass 12 }}
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(world, encoding="utf-8", newline="\n")
