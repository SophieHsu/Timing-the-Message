import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle, colorize
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from gymnasium.envs.box2d.lunar_lander import ContactDetector

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


class NotiLunarLander(gym.Env, EzPickle):
    """
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```
    python gymnasium/envs/box2d/lunar_lander.py
    ```
    <!-- To play yourself, run: -->
    <!-- python examples/agents/keyboard_agent.py LunarLander-v2 -->

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments
    To use to the _continuous_ environment, you need to specify the
    `continuous=True` argument like below:
    ```python
    import gymnasium as gym
    env = gym.make(
        "LunarLander-v2",
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    )
    ```
    If `continuous=True` is passed, continuous actions (corresponding to the throttle of the engines) will be used and the
    action space will be `Box(-1, +1, (2,), dtype=np.float32)`.
    The first coordinate of an action determines the throttle of the main engine, while the second
    coordinate specifies the throttle of the lateral boosters.
    Given an action `np.array([main, lateral])`, the main engine will be turned off completely if
    `main < 0` and the throttle scales affinely from 50% to 100% for `0 <= main <= 1` (in particular, the
    main engine doesn't work  with less than 50% power).
    Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
    booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12.

    If `enable_wind=True` is passed, there will be wind effects applied to the lander.
    The wind is generated using the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))`.
    `k` is set to 0.01.
    `C` is sampled randomly between -9999 and 9999.

    `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for `wind_power` is between 0.0 and 20.0.
    `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft. The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground
        give +10 reward points, and -10 if then lose contact; reward
        renormalized to 200; harder initial random push.
    - v0: Initial version


    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thursters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation depentant torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10
    'y': 6.666
    'vx': 5
    'vy': 7.5
    'angle': 1
    'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units)
    'y': (units)
    'vx': (units/second)
    'vy': (units/second)
    'angle': (radians)
    'angular velocity': (radians/second)


    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = True,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        human_agent_idx: int = 1,
        max_episode_steps: int = 600,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            gravity,
            enable_wind,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            warnings.warn(
                colorize(
                    f"WARN: wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})",
                    "yellow",
                ),
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            warnings.warn(
                colorize(
                    f"WARN: turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})",
                    "yellow",
                ),
            )
        self.turbulence_power = turbulence_power

        self.enable_wind = enable_wind
        self.wind_idx = np.random.randint(-9999, 9999)
        self.torque_idx = np.random.randint(-9999, 9999)

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander: Optional[Box2D.b2Body] = None
        self.particles = []

        self.prev_reward = None
        self.current_reward = 0
        self.reward_components = {}
        
        # Add notification history
        self.noti_history = []
        self.max_history_size = 10  # Maximum number of past notifications to display
        self.curr_agent_action = None
        self.overwrite_flag = False

        self.continuous = continuous

        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5 * SCALE * 2,
                -1.5 * SCALE * 2,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5 * SCALE * 2,
                1.5 * SCALE * 2,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
                5.0,
                5.0,
                5.0,
                5.0
            ]
        ).astype(np.float32)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            # self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(12), spaces.Discrete(2), spaces.Discrete(4)))
            self.action_space = spaces.MultiDiscrete([3, 3, 1, 4])
            # self.action_space = spaces.MultiDiscrete([3, 3, 2, 4])

        self.render_mode = render_mode
        self.human_agent_idx = human_agent_idx
        self.max_episode_steps = max_episode_steps
        self.danger_zones = []
        self.step_count = 0

    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        
        # Reset notification history and flags
        self.noti_history = []
        self.curr_agent_action = None
        self.overwrite_flag = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(H / 4, H / 4, size=(CHUNKS + 1,))
        # height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        initial_x = self.np_random.uniform(0, VIEWPORT_W / SCALE)
        # initial_x = VIEWPORT_W / SCALE / 2
        self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        self.step_count = 0

        try:
            self.spec.max_episode_steps = self.max_episode_steps
        except:
            pass

        next_obs, _, _, _, info = self.step([0,0,0, np.array([0, 0])] if self.continuous else [0,0,0, 0])

        return next_obs, info

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _measure_danger_zone_distance(self, pos):
        # Check if the lander is close to any danger zone (zone is represented by a square with 4 points)
        left_danger_zone_distance = self.observation_space.high[8]
        right_danger_zone_distance = self.observation_space.high[9]
        top_danger_zone_distance = self.observation_space.high[10]
        bottom_danger_zone_distance = self.observation_space.high[11]

        x = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        y = (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)

        for zone in self.danger_zones:
            danger_zone_min_x = zone[0][0]
            danger_zone_max_x = zone[0][1]
            danger_zone_min_y = zone[1][0]
            danger_zone_max_y = zone[1][1]

            # in danger zone
            if x >= danger_zone_min_x and x <= danger_zone_max_x and y >= danger_zone_min_y and y <= danger_zone_max_y: # in danger zone
                # should expect negative, and that the closer the more negative, means higher penalty
                left_danger_zone_distance = 0 # x - danger_zone_max_x 
                right_danger_zone_distance = 0 # x - danger_zone_min_x 
                top_danger_zone_distance = 0 # y - danger_zone_max_y 
                bottom_danger_zone_distance = 0 # y - danger_zone_min_y
                break

            # check top and bottom danger zone
            if (x >= danger_zone_min_x and x <= danger_zone_max_x): 
                if y > danger_zone_max_y: # on top of bottom danger zone
                    bottom_danger_zone_distance = min(bottom_danger_zone_distance, abs(y - danger_zone_max_y))

                if y < danger_zone_min_y: # below top danger zone
                    top_danger_zone_distance = min(top_danger_zone_distance, abs(danger_zone_min_y - y))
            
            # check left and right danger zone
            if (y >= danger_zone_min_y and y <= danger_zone_max_y):
                if x < danger_zone_min_x: # left of right danger zone
                    right_danger_zone_distance = min(right_danger_zone_distance, abs(danger_zone_min_x - x))
                if x > danger_zone_max_x: # right of left danger zone
                    left_danger_zone_distance = min(left_danger_zone_distance, abs(x-danger_zone_max_x))

        return left_danger_zone_distance, right_danger_zone_distance, top_danger_zone_distance, bottom_danger_zone_distance
    
    def _world_to_pixel(self, coord, width, height):
        """
        Convert world coordinates to pixel coordinates.
        Assumes the world space ranges from -2.5 to 2.5 (x) and 0 to 1.5 (y).
        """
        world_x_range = [-1.0, 1.0]
        world_y_range = [-0.33, 1.33]

        # Normalize the coordinates based on the world ranges
        norm_x = (coord[0] - world_x_range[0]) / (world_x_range[1] - world_x_range[0])
        norm_y = (coord[1] - world_y_range[0]) / (world_y_range[1] - world_y_range[0])

        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(norm_x * width)
        pixel_y = int((1 - norm_y) * height)  # Flip y-axis to match image coordinates

        return (pixel_x, pixel_y)
    
    def step(self, joint_action):
        if self.human_agent_idx == 0:
            action = joint_action[0]
            noti_action = np.array(joint_action[1:-1])
            overwrite_flag = joint_action[-1]
        else:
            action = joint_action[-2]
            noti_action = np.array(joint_action[:-2])
            overwrite_flag = joint_action[-1]

        # Store notification in history
        if noti_action[0] == 0: # no notification
            noti_action = np.array([0,0,0])
        elif noti_action[0] == 1: # continue previous notification
            noti_action = np.array([1,0,0])
        else:
            # process notification length: 2 or 5
            noti_action[1] = noti_action[1] + 1 # here we add 1 to the action id so that 0 is the no-op action from the action type, and the agent should only choose left, up, or right, which each id corresponds to 1, 2, or 3.
            noti_action[2] = (noti_action[2]*3) + 2

        self.noti_history.append(noti_action)
        self.curr_agent_action = action
        self.overwrite_flag = overwrite_flag
        
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        # else:
            # assert self.action_space.contains(
            #     action
            # ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is a the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation depentant torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        # Check if the lander is close to any danger zone (zone is represented by a square with 4 points)
        to_left_danger_zone_distance, to_right_danger_zone_distance, to_top_danger_zone_distance, to_bottom_danger_zone_distance = self._measure_danger_zone_distance(pos)
            
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            to_left_danger_zone_distance,
            to_right_danger_zone_distance,
            to_top_danger_zone_distance,
            to_bottom_danger_zone_distance
        ]
        assert len(state) == 12

        reward = 0
        info = {}
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        # if self.prev_shaping is not None:
        #     reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        truncated = False
        success_landing = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and self.legs[0].ground_contact and self.legs[1].ground_contact and state[2] < 0.05 and state[2] > -0.05 and state[3] < 0.05 and state[3] > -0.05
        
        try:
            if self.step_count >= self.spec.max_episode_steps:
                truncated = True
        except:
            if self.step_count >= self.max_episode_steps:
                truncated = True
        
        if success_landing:
            terminated = True
            reward = +100
        elif self.game_over and not success_landing:
            terminated = True
            reward = -100
        elif not self.lander.awake and not success_landing:
            terminated = True
            reward = -100

        # Store the current reward for display
        self.current_reward = reward

        info["terminated"] = terminated
        info["success"] = success_landing
        info["truncated"] = truncated

        # process notification and add utterance to the info
        if noti_action[0] == 0: # no notification
            noti_action = np.array([0,0,0])
        elif noti_action[0] == 1: # continue previous notification
            noti_action = np.array([-1,0,0])
        else: # new notification
            # process notification length: 2 or 5
            noti_action[2] += (noti_action[2]*3) + 2
        info["utterance"] = noti_action

        if self.render_mode == "human":
            self.render()

        self.step_count += 1
        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[box2d]`"
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()  # Initialize the font module
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        # Flip the surface
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        # Draw danger zones AFTER the flip
        for zone in self.danger_zones:
            # Convert normalized coordinates to screen coordinates using _world_to_pixel
            x_min, y_min = self._world_to_pixel((zone[0][0], zone[1][0]), VIEWPORT_W, VIEWPORT_H)
            x_max, y_max = self._world_to_pixel((zone[0][1], zone[1][1]), VIEWPORT_W, VIEWPORT_H)
            
            # Ensure valid dimensions by swapping if necessary and taking absolute values
            width = abs(x_max - x_min)
            height = abs(y_max - y_min)
            rect_x = min(x_min, x_max)
            rect_y = min(y_min, y_max)
            
            if width > 0 and height > 0:  # Only create surface if dimensions are valid
                # Draw danger zone rectangle with semi-transparent red
                danger_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                danger_surface.fill((255, 0, 0, 128))  # Red with 50% transparency
                self.surf.blit(danger_surface, (rect_x, rect_y))
                
                # Draw border
                pygame.draw.rect(self.surf, (255, 0, 0), (rect_x, rect_y, width, height), 2)
        
        # Add text displaying x, y position AFTER the flip
        if self.lander is not None:
            # Initialize font
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
                font = pygame.font.SysFont('Arial', 20)
            except:
                font = pygame.font.Font(None, 20)
                
            # Get position
            pos = self.lander.position
            x_pos = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
            y_pos = (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
            
            # Create text surfaces
            pos_text = font.render(f'X: {x_pos:.2f} Y: {y_pos:.2f}', True, (255, 255, 255))

            # Add current human action
            if self.curr_agent_action is not None:
                # Use different colors for overwritten actions
                color = (255, 0, 0) if self.overwrite_flag == 1 else (255, 255, 0)  # Red for overwritten, yellow for normal
                action_text = font.render(f'Human: {self.curr_agent_action}', True, color)
                self.surf.blit(action_text, (160, 10))
            
            # Draw text - now we can use normal coordinates since the surface is already flipped
            self.surf.blit(pos_text, (10, 10))
            
            # Add current notification action
            curr_noti = "" if len(self.noti_history) == 0 else self.noti_history[-1]  
            noti_text = font.render(f'Noti: {curr_noti}', True, (0, 255, 255))
            self.surf.blit(noti_text, (10, 30))
            
            # Add notification history
            y_offset = 50
            history_text = font.render('Noti History:', True, (0, 255, 255))
            self.surf.blit(history_text, (10, y_offset))
            
            y_offset += 20
            for i, past_noti in enumerate(reversed(self.noti_history[:-1])):
                if i >= self.max_history_size-1:
                    break
                
                # Use different colors for overwritten actions
                color = (0, 255, 255)  # cyan for normal
                past_noti_text = font.render(f'{i+1}: {past_noti}', True, color)
                self.surf.blit(past_noti_text, (20, y_offset))
                y_offset += 20
            
            # Add danger zone distance information
            if hasattr(self, '_measure_danger_zone_distance'):
                left_dist, right_dist, top_dist, bottom_dist = self._measure_danger_zone_distance(pos)
                value = left_dist < 0 or right_dist < 0 or top_dist < 0 or bottom_dist < 0
                color = (255, 0, 0) if value else (0, 255, 0)
                danger_text = font.render(f'Danger: L:{left_dist:.2f} R:{right_dist:.2f} T:{top_dist:.2f} B:{bottom_dist:.2f}', 
                                         True, color)
                self.surf.blit(danger_text, (10, y_offset))
                y_offset += 20
            
            # Add reward information
            total_reward_text = font.render(f'Total Reward: {self.current_reward:.2f}', True, (255, 255, 255))
            self.surf.blit(total_reward_text, (10, y_offset))
            
            # Add reward components
            y_offset += 20
            for component, value in self.reward_components.items():
                color = (0, 255, 0) if value >= 0 else (255, 0, 0)  # Green for positive, red for negative
                component_text = font.render(f'{component}: {value:.2f}', True, color)
                self.surf.blit(component_text, (10, y_offset))
                y_offset += 20

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class LargeRewardNotiLunarLander(NotiLunarLander):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_wind = True

    def step(self, joint_action):
        if self.human_agent_idx == 0:
            action = joint_action[0]
            noti_action = np.array(joint_action[1:-1])
            overwrite_flag = joint_action[-1]
        else:
            action = joint_action[-2]
            noti_action = np.array(joint_action[:-2])
            overwrite_flag = joint_action[-1]

        # Store notification in history
        if noti_action[0] == 0: # no notification
            noti_action = np.array([0,0,0])
        elif noti_action[0] == 1: # continue previous notification
            noti_action = np.array([1,0,0])
        else:
            # process notification length: 2 or 5
            noti_action[1] = noti_action[1] + 1 # here we add 1 to the action id so that 0 is the no-op action from the action type, and the agent should only choose left, up, or right, which each id corresponds to 1, 2, or 3.
            noti_action[2] = (noti_action[2]*3) + 2
        
        self.noti_history.append(noti_action)
        self.curr_agent_action = action
        self.overwrite_flag = overwrite_flag

        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        # else:
            # assert self.action_space.contains(
            #     action
            # ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is a the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation depentant torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        # Check if the lander is close to any danger zone (zone is represented by a square with 4 points)
        to_left_danger_zone_distance, to_right_danger_zone_distance, to_top_danger_zone_distance, to_bottom_danger_zone_distance = self._measure_danger_zone_distance(pos)
            
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            to_left_danger_zone_distance,
            to_right_danger_zone_distance,
            to_top_danger_zone_distance,
            to_bottom_danger_zone_distance
        ]
        assert len(state) == 12

        reward = 0
        info = {}
        
        # Distance-based rewards with better scaling
        distance_reward = -50 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        velocity_reward = -50 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        
        # Angle reward with better scaling and deadzone
        angle_reward = -50 * abs(state[4]) if abs(state[4]) > 0.1 else 0
        
        # Angular velocity reward to encourage stability
        angular_velocity_reward = -20 * abs(state[5])
        
        # Leg contact rewards with bonus for both legs
        leg_reward = 20 * (state[6] + state[7])
        if state[6] and state[7]:  # Bonus for both legs
            leg_reward += 30
            
        # Danger zone penalties
        danger_zone_penalty = -10 * (
            max(0.5 - state[8], 0) +  # left danger zone
            max(0.5 - state[9], 0) +  # right danger zone
            max(0.5 - state[10], 0) + # top danger zone
            max(0.5 - state[11], 0)   # bottom danger zone
        )

        if state[8] < 0 and state[9] < 0 and state[10] < 0 and state[11] < 0:
            in_danger_zone_penalty = -30
        else:
            in_danger_zone_penalty = 0
        
        # Fuel efficiency rewards
        fuel_reward = -m_power * 0.15  # Reduced penalty for main engine
        side_fuel_reward = -s_power * 0.02  # Reduced penalty for side engines
        
        # Combine all rewards
        shaping = (
            distance_reward +
            velocity_reward +
            angle_reward +
            angular_velocity_reward +
            leg_reward +
            danger_zone_penalty +
            in_danger_zone_penalty +
            fuel_reward +
            side_fuel_reward
        )
        
        # Add shaping to reward if prev_shaping exists
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        terminated = False
        truncated = False
        success_landing = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and self.legs[0].ground_contact and self.legs[1].ground_contact and state[2] < 0.05 and state[2] > -0.05 and state[3] < 0.05 and state[3] > -0.05
        
        if self.step_count >= self.spec.max_episode_steps:
            truncated = True
        
        if success_landing:
            terminated = True
            reward = +200  # Increased success reward
        elif self.game_over and not success_landing:
            terminated = True
            reward = -200  # Increased failure penalty
        elif not self.lander.awake and not success_landing:
            terminated = True
            reward = -200  # Increased failure penalty
            
        # Store the current reward for display
        self.current_reward = reward

        info["terminated"] = terminated
        info["success"] = success_landing
        info["truncated"] = truncated

        info["utterance"] = noti_action

        if self.render_mode == "human":
            self.render()

        self.step_count += 1

        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

class DangerZoneLunarLander(LargeRewardNotiLunarLander):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define a set of possible danger zone configurations
        self.possible_danger_zones = [
            # Configuration 1: Three zones forming a challenging path
            [
                [[-1.0, -0.6], [1.0, 1.33]],
                [[-0.3, 1.0], [0.3, 0.6]],
                [[0.3, 1.0], [0, 0.3]]
            ],
            # Configuration 2: Three zones with left-side emphasis
            [
                [[-1.0, -0.3], [0.1, 0.3]],
                [[-1.0, 0.3], [0.3, 0.6]],
                [[0.7, 1.0], [0.8, 1.33]]
            ],
            # Configuration 3: Single central barrier
            [
                [[-0.5, 0.5], [0.4, 0.7]]
            ],
            # Configuration 4: Two zones creating a narrow path
            [
                [[-0.3, 0.7], [0.5, 0.8]],
                [[0.6, 1.0], [0.8, 1.33]]
            ],
            # Configuration 5: Single large central block
            [
                [[-0.6, 0.0], [0.6, 0.9]]
            ],
            # Configuration 6: Two blocks on left side
            [
                [[-1.0, -0.5], [0.8, 1.33]],
                [[0.2, 0.6], [0.5, 1.33]]
            ]
        ]
        # Select a first danger zone configuration at initialization
        self.danger_zones = self.possible_danger_zones[0]
        self.time_penalty = -0.0
        self.prev_state = None
        self.enable_wind = False
        self.random_danger_zone = False
        self.noti_action_length = len(self.action_space.nvec)-1
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if self.random_danger_zone:
            self.select_random_danger_zone()
        else:
            self.danger_zones = self.possible_danger_zones[0]
            
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        
        # Reset notification history and flags
        self.noti_history = []
        self.curr_agent_action = None
        self.overwrite_flag = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(H / 4, H / 4, size=(CHUNKS + 1,))
        # height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander body
        initial_y = VIEWPORT_H / SCALE
        # initial_x = self.np_random.uniform(VIEWPORT_W / SCALE / 2, VIEWPORT_W / SCALE)
        initial_x = VIEWPORT_W / SCALE / 2
        self.lander: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0,
            ),  # 0.99 bouncy
        )
        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply the initial random impulse to the lander
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        # Create Lander Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = (
                    +0.9 - 0.5
                )  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        if self.render_mode == "human":
            self.render()

        self.step_count = 0
        
        try:
            self.spec.max_episode_steps = self.max_episode_steps
        except:
            pass

        next_obs, _, _, _, info = self.step([0]*self.noti_action_length + [np.array([0, 0])] if self.continuous else [0]*self.noti_action_length + [0, 0])

        return next_obs, info

    
    def select_random_danger_zone(self):
        """Randomly select a danger zone configuration from the possible configurations"""
        idx = self.np_random.integers(len(self.possible_danger_zones))
        self.danger_zones = self.possible_danger_zones[idx]
    
    def step(self, joint_action, overwrite_flag=False):
        action = joint_action[-2]
        noti_action = np.array(joint_action[:-2])
        overwrite_flag = joint_action[-1]
        
        # Store notification in history
        if noti_action[0] == 0: # no notification
            noti_action = np.array([0]*self.noti_action_length)
        elif noti_action[0] == 1: # continue previous notification
            noti_action = np.array([1] + [0]*(self.noti_action_length-1))
        else:
            # process notification length: 2 or 5
            noti_action[2] = (noti_action[2]*3) + 2

            if self.noti_action_length > 3:
                noti_action[3] = min(noti_action[2], (noti_action[3]*3)+2)
        
        self.noti_history.append(noti_action)
        self.curr_agent_action = action
        self.overwrite_flag = overwrite_flag

        if len(self.noti_history) > self.max_history_size:
            self.noti_history.pop(0)

        
        assert self.lander is not None

        # Update wind and apply to the lander
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        # else:
            # assert self.action_space.contains(
            #     action
            # ), f"{action!r} ({type(action)}) invalid "

        # Apply Engine Impulses

        # Tip is a the (X and Y) components of the rotation of the lander.
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation/Side engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                # action = 1 is left, action = 3 is right
                direction = action - 2
                s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation depentant torque being applied to the lander.
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity

        # Check if the lander is close to any danger zone (zone is represented by a square with 4 points)
        to_left_danger_zone_distance, to_right_danger_zone_distance, to_top_danger_zone_distance, to_bottom_danger_zone_distance = self._measure_danger_zone_distance(pos)
            
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            to_left_danger_zone_distance,
            to_right_danger_zone_distance,
            to_top_danger_zone_distance,
            to_bottom_danger_zone_distance
        ]
        assert len(state) == 12

        reward = 0
        info = {}

        # cost for speaking
        noti_penalty = -3 #-0.3
        if noti_action[0] == 2:
            reward += noti_penalty
            self.reward_components["noti_penalty"] = noti_penalty
        else:
            self.reward_components["noti_penalty"] = 0

        # value for longer notification
        noti_content_reward = 2
        if noti_action[0] == 2:
            self.reward_components["noti_content_reward"] = (noti_action[2]-2)+noti_content_reward # reward length 5 with 2; and length 2 with 0
        else:
            self.reward_components["noti_content_reward"] = 0
        reward += self.reward_components["noti_content_reward"]
        
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping
        # self.reward_components["shaping"] = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        # self.reward_components["fuel"] = -(m_power * 0.30 + s_power * 0.03)

        # Danger zone penalties
        danger_zone_penalty = -30 * (
            max(0.2 - state[8], 0) +  # left danger zone
            max(0.2 - state[9], 0) +  # right danger zone
            max(0.2 - state[10], 0) + # top danger zone
            max(0.2 - state[11], 0)   # bottom danger zone
        )
        self.reward_components["danger_zone"] = danger_zone_penalty
        in_danger_zone = (state[8] <= 0 and state[9] <= 0 and state[10] <= 0 and state[11] <= 0)

        # Reward for moving away from danger zones
        danger_avoidance_reward = 0
        if (self.prev_state is not None) and not in_danger_zone:
            # Calculate if we're moving away from danger zones
            prev_danger_distances = [
                self.prev_state[8] if hasattr(self.prev_state, '__getitem__') else 0,
                self.prev_state[9] if hasattr(self.prev_state, '__getitem__') else 0,
                self.prev_state[10] if hasattr(self.prev_state, '__getitem__') else 0,
                self.prev_state[11] if hasattr(self.prev_state, '__getitem__') else 0
            ]
            
            # If we're moving away from danger zones, give a reward
            for i in range(4):
                if state[8+i] > prev_danger_distances[i]:
                    danger_avoidance_reward += 0.1  # Reward for moving away from danger zones
            
        # self.reward_components["danger_avoidance"] = danger_avoidance_reward

        # # Stronger penalty for being inside danger zones
        if in_danger_zone:
            in_danger_zone_penalty = -20  # Increased from -30
        else:
            in_danger_zone_penalty = 0
        
        self.reward_components["in_danger_zone"] = in_danger_zone_penalty
        
        # Add a small time penalty to encourage faster landings
         
        # self.reward_components["time_penalty"] = self.time_penalty
        
        reward += danger_zone_penalty + in_danger_zone_penalty#+ danger_avoidance_reward

        terminated = False
        truncated = False
        success_landing = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and self.legs[0].ground_contact and self.legs[1].ground_contact and state[2] < 0.05 and state[2] > -0.05 and state[3] < 0.05 and state[3] > -0.05
        
        try:
            if self.step_count >= self.spec.max_episode_steps:
                truncated = True
        except:
            if self.step_count >= self.max_episode_steps:
                truncated = True
        
        if success_landing:
            terminated = True
            reward = +300
            self.reward_components["success"] = reward
        elif self.game_over and not success_landing:
            terminated = True
            crash_reward = -300  # Increased failure penalty
            self.reward_components["crash"] = crash_reward
            reward = crash_reward
        elif not self.lander.awake and not success_landing:
            terminated = True
            sleep_reward = -300  # Increased failure penalty
            self.reward_components["sleep"] = sleep_reward
            reward = sleep_reward
        
        
        # Store the current reward for display
        self.current_reward = reward

        info["terminated"] = terminated
        info["success"] = success_landing
        info["truncated"] = truncated
        info["utterance"] = noti_action

        if self.render_mode == "human":
            self.render()

        self.step_count += 1

        self.prev_state = state.copy()  

        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

class SimpleNotiDangerZoneLunarLander(DangerZoneLunarLander):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 4])
        self.noti_action_length = len(self.action_space.nvec)-1

class ComplexNotiDangerZoneLunarLander(DangerZoneLunarLander):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 2, 4])
        self.noti_action_length = len(self.action_space.nvec)-1


def heuristic(env, s):
    """
    The heuristic for
    1. Testing
    2. Demonstration rollout.

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed
            s[6] 1 if first leg has contact, else 0
            s[7] 1 if second leg has contact, else 0

    Returns:
         a: The heuristic to be fed into the step function defined above to determine the next step and reward.
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(s[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1

    return a


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    s, info = env.reset(seed=seed)
    while True:
        a = heuristic(env, s)
        noti_action = (0,0,0)
        s, r, terminated, truncated, info = step_api_compatibility(env.step([noti_action, a]), True)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join([f"{x:+0.2f}" for x in s]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break
    if render:
        env.close()
    return total_reward


if __name__ == "__main__":
    import gymnasium_envs
    
    env = gym.make("gymnasium_envs/NotiLunarLander", render_mode="rgb_array")
    demo_heuristic_lander(env, render=True)
