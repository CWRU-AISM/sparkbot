"""joint_move steps physics outside env.step; it must still spend the episode clock and stop at the horizon."""
import types

import mujoco
import numpy as np

from spark_bench.libero_pro import motion

XML = ('<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size=".1"/></body></worldbody>'
       '<actuator><motor joint="j"/></actuator></mujoco>')


def test_joint_move_charges_episode_clock():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    e = types.SimpleNamespace(control_timestep=0.05, model_timestep=m.opt.timestep, timestep=0, cur_time=0.0,
                              horizon=3, ignore_done=False, done=False)
    # kp = kd = 0 never converges, so only the clock can end the move
    motion.joint_move(types.SimpleNamespace(env=e), m, d, [0], -1, [0], [], np.array([3.0]), True,
                      steps=1000, kp=0.0, kd=0.0)
    assert e.done and e.timestep == 3
