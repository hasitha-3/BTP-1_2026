import pybullet as p
import pybullet_data
import numpy as np
import json
import random


MAX_OBSTACLES_PER_SAMPLE = 10


class RobotManipulator:

    def __init__(self, link_lengths=None):
        if link_lengths is None:
            link_lengths = [1.0, 0.8, 0.6, 0.5]
        self.link_lengths = link_lengths[:4]
        self.num_links    = len(self.link_lengths)
        self.robot_id     = None
        self.max_reach    = sum(self.link_lengths)

    def load_robot(self, num_active_links):
        base_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.05, length=0.1,
            rgbaColor=[0.5, 0.5, 0.5, 1],
            visualFramePosition=[0, 0, 0.05]
        )
        base_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.05, height=0.1,
            collisionFramePosition=[0, 0, 0.05]
        )

        lm, lc, lv = [], [], []
        lp, lo, lip, lio, lpar, ljt, lja = [], [], [], [], [], [], []

        for i in range(num_active_links):
            L = self.link_lengths[i]
            lv.append(p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.03, length=L,
                rgbaColor=[1, 0.5, 0, 1],
                visualFramePosition=[0, 0, L / 2]
            ))
            lc.append(p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.05, height=L,
                collisionFramePosition=[0, 0, L / 2]
            ))
            lm.append(0.1)
            lp.append([0, 0, 0.1] if i == 0 else [0, 0, self.link_lengths[i - 1]])
            lo.append([0, 0, 0, 1])
            lip.append([0, 0, 0])
            lio.append([0, 0, 0, 1])
            lpar.append(i)
            ljt.append(p.JOINT_REVOLUTE)
            lja.append([0, 0, 1])

        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            linkMasses=lm,
            linkCollisionShapeIndices=lc,
            linkVisualShapeIndices=lv,
            linkPositions=lp,
            linkOrientations=lo,
            linkInertialFramePositions=lip,
            linkInertialFrameOrientations=lio,
            linkParentIndices=lpar,
            linkJointTypes=ljt,
            linkJointAxis=lja
        )
        for i in range(num_active_links):
            p.changeDynamics(self.robot_id, i,
                             lateralFriction=1.0,
                             spinningFriction=0.1,
                             rollingFriction=0.1)
        return self.robot_id

    def set_joint_positions(self, positions):
        nj = p.getNumJoints(self.robot_id)
        for i, pos in enumerate(positions):
            if i < nj:
                p.resetJointState(self.robot_id, i, pos)

    def check_collision(self, obstacle_ids):
        p.performCollisionDetection()
        for oid in obstacle_ids:
            if p.getContactPoints(self.robot_id, oid):
                return True
        return False


class ObstacleManager:

    def __init__(self, max_robot_reach=2.9):
        self.obstacles       = []
        self.max_robot_reach = max_robot_reach

    def create_obstacles(self, num_obstacles, min_radius, max_radius):
        """
        Place num_obstacles spheres randomly throughout the full workspace.
        - XY: polar radius uniformly sampled in [0, max_reach]
        - Z:  uniformly sampled in [0.1, max_reach]  (full arm height)
        - radius: uniformly sampled in [min_radius, max_radius]
        """
        self.obstacles = []
        r_max = self.max_robot_reach

        for _ in range(num_obstacles):
            r      = random.uniform(0.0, r_max)
            theta  = random.uniform(0, 2 * np.pi)
            x      = r * np.cos(theta)
            y      = r * np.sin(theta)
            z      = random.uniform(0.1, r_max)
            radius = random.uniform(min_radius, max_radius)

            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                      rgbaColor=[1, 0, 0, 0.6])
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            oid = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis,
                                    basePosition=[x, y, z])
            self.obstacles.append({
                'position': [x, y, z],
                'radius':   radius,
                'id':       oid
            })
        return self.obstacles

    def clear_obstacles(self):
        for obs in self.obstacles:
            if 'id' in obs:
                p.removeBody(obs['id'])
        self.obstacles = []

    def get_obstacle_ids(self):
        return [obs['id'] for obs in self.obstacles]

    def get_obstacle_data(self):
        return [{'position': obs['position'], 'radius': obs['radius']}
                for obs in self.obstacles]


class RRTPlanner:
    def __init__(self, robot, max_iterations=800, step_size=0.25):
        self.robot          = robot
        self.max_iterations = max_iterations
        self.step_size      = step_size

    def random_config(self, n):
        return [random.uniform(-np.pi, np.pi) for _ in range(n)]

    def distance(self, a, b):
        return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def interpolate(self, a, b, t):
        return [x + t * (y - x) for x, y in zip(a, b)]

    def is_collision_free_path(self, c1, c2, obstacle_ids, num_checks=10):
        for i in range(num_checks + 1):
            self.robot.set_joint_positions(
                self.interpolate(c1, c2, i / num_checks)
            )
            if self.robot.check_collision(obstacle_ids):
                return False
        return True

    def plan(self, start, goal, num_active_links, obstacle_ids):
        start = start[:num_active_links]
        goal  = goal[:num_active_links]

        self.robot.set_joint_positions(start)
        if self.robot.check_collision(obstacle_ids):
            return False
        self.robot.set_joint_positions(goal)
        if self.robot.check_collision(obstacle_ids):
            return False

        tree = [start]

        for _ in range(self.max_iterations):
            q_rand = goal if random.random() < 0.1 \
                          else self.random_config(num_active_links)

            nearest_idx = min(range(len(tree)),
                              key=lambda i: self.distance(tree[i], q_rand))
            q_near = tree[nearest_idx]
            dist   = self.distance(q_near, q_rand)
            if dist < 1e-9:
                continue

            t     = self.step_size / dist if dist > self.step_size else 1.0
            q_new = self.interpolate(q_near, q_rand, t)

            if self.is_collision_free_path(q_near, q_new, obstacle_ids):
                tree.append(q_new)
                if self.distance(q_new, goal) < self.step_size:
                    if self.is_collision_free_path(q_new, goal, obstacle_ids):
                        return True
        return False


class DatasetGenerator:

    def __init__(self, num_configs=50, variations_per_config=4,
                 max_dof=4, dof_levels=None, gui=False):
        if not 1 <= max_dof <= 4:
            raise ValueError("max_dof must be between 1 and 4")
        if dof_levels is None:
            dof_levels = [1, 2, 3, 4]
        if any(d < 1 or d > max_dof for d in dof_levels):
            raise ValueError("All dof_levels must be between 1 and max_dof")

        self.num_configs           = num_configs
        self.variations_per_config = variations_per_config
        self.max_dof               = max_dof
        self.dof_levels            = sorted(set(dof_levels))
        self.num_samples           = (num_configs * variations_per_config
                                      * len(self.dof_levels))
        self.dataset               = []

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.robot            = RobotManipulator()
        self.obstacle_manager = ObstacleManager(
            max_robot_reach=self.robot.max_reach
        )
        self.planner = None


    def _random_obstacle_params(self):
        """
        Fully random obstacle count, size. No class-specific bands.
        Obstacle count drawn from [3, MAX_OBSTACLES_PER_SAMPLE].
        Radius drawn from [0.10, 0.35] — wide enough range so some
        obstacles block the arm and some don't, making the task non-trivial.
        """
        return {
            'num_obstacles': random.randint(3, MAX_OBSTACLES_PER_SAMPLE),
            'min_radius':    0.10,
            'max_radius':    0.35,
        }

    def generate_random_config(self, dof):
        return [random.uniform(-np.pi, np.pi) for _ in range(dof)]

    def determine_infeasibility_link(self, start, goal, tested_dof):
        iteration_map     = {1: 2000, 2: 2400, 3: 2800, 4: 3200}
        restarts_per_link = 3
        obstacle_ids      = self.obstacle_manager.get_obstacle_ids()

        for num_links in range(1, tested_dof + 1):
            if self.robot.robot_id is not None:
                p.removeBody(self.robot.robot_id)
            self.robot.load_robot(num_links)

            self.planner = RRTPlanner(
                self.robot,
                max_iterations=iteration_map[num_links],
                step_size=0.20
            )

            feasible = False
            for _ in range(restarts_per_link):
                if self.planner.plan(start, goal, num_links, obstacle_ids):
                    feasible = True
                    break

            if not feasible:
                return num_links

        return 0


    def generate_single_sample(self, sample_id, start_config, goal_config,
                                config_id, variation_id, dof):
        params = self._random_obstacle_params()

        self.obstacle_manager.clear_obstacles()
        self.obstacle_manager.create_obstacles(
            params['num_obstacles'],
            params['min_radius'],
            params['max_radius']
        )

        infeasibility_link = self.determine_infeasibility_link(
            start_config, goal_config, dof
        )

        return {
            'id':                 sample_id,
            'config_id':          config_id,
            'variation_id':       variation_id,
            'dof':                dof,
            'start_config':       start_config,
            'goal_config':        goal_config,
            'num_obstacles':      params['num_obstacles'],
            'obstacles':          self.obstacle_manager.get_obstacle_data(),
            'infeasibility_link': infeasibility_link,
            'feasible':           infeasibility_link == 0
        }


    def generate_dataset(self):
        print("Starting dataset generation...")
        print(f"  Configurations        : {self.num_configs}")
        print(f"  Variations per config : {self.variations_per_config}")
        print(f"  DOF levels            : {self.dof_levels}")
        print(f"  Total samples         : {self.num_samples}")
        print(f"  Obstacle placement    : FULLY RANDOM (no class conditioning)")
        print("=" * 60)

        sample_id = 0

        for config_id in range(self.num_configs):
            full_start = self.generate_random_config(self.max_dof)
            full_goal  = self.generate_random_config(self.max_dof)

            if (config_id + 1) % 50 == 1 or config_id == self.num_configs - 1:
                print(f"\nConfiguration {config_id + 1}/{self.num_configs}:")

            for var_id in range(self.variations_per_config):
                for dof in self.dof_levels:
                    sample = self.generate_single_sample(
                        sample_id,
                        full_start[:dof],
                        full_goal[:dof],
                        config_id,
                        var_id,
                        dof
                    )
                    self.dataset.append(sample)
                    sample_id += 1

            if (config_id + 1) % 50 == 0:
                print(f"  Progress: {config_id + 1}/{self.num_configs} configs "
                      f"| {sample_id}/{self.num_samples} samples")

        print("\n" + "=" * 60)
        print("Dataset generation complete!")
        self.print_statistics()

    def print_statistics(self):
        total      = len(self.dataset)
        feasible   = sum(1 for s in self.dataset if s['feasible'])
        infeasible = total - feasible

        print("\nDataset Statistics:")
        print(f"  Total samples         : {total}")
        print(f"  Feasible              : {feasible} ({100*feasible/total:.1f}%)")
        print(f"  Infeasible            : {infeasible} ({100*infeasible/total:.1f}%)")
        print("\n  Infeasibility by first-failing link:")
        for link in range(1, self.max_dof + 1):
            cnt = sum(1 for s in self.dataset if s['infeasibility_link'] == link)
            if cnt:
                print(f"    Link {link}: {cnt} ({100*cnt/total:.1f}%)")

        avg_obs = np.mean([s['num_obstacles'] for s in self.dataset])
        print(f"\n  Avg obstacles/sample  : {avg_obs:.2f}")

        cfgs_inf = set(s['config_id'] for s in self.dataset if not s['feasible'])
        print(f"  Configs w/ ≥1 infeasible variation: "
              f"{len(cfgs_inf)} ({100*len(cfgs_inf)/self.num_configs:.1f}%)")

        print("\n  Per-DOF sample counts:")
        for dof in self.dof_levels:
            cnt = sum(1 for s in self.dataset if s['dof'] == dof)
            print(f"    DOF {dof}: {cnt}")

        print("\n  NOTE: class imbalance is expected and handled via class")
        print("  weights in training.py — do not adjust generation to fix it.")

    def save_dataset(self, filename="dataset.json"):
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2)
        print(f"\nDataset saved to: {filename}")

    def cleanup(self):
        p.disconnect()


def main():
    NUM_CONFIGS           = 2500
    VARIATIONS_PER_CONFIG = 10
    MAX_DOF               = 4
    DOF_LEVELS            = [1, 2, 3, 4]
    OUTPUT_FILE           = "dataset.json"
    USE_GUI               = False

    generator = DatasetGenerator(
        num_configs=NUM_CONFIGS,
        variations_per_config=VARIATIONS_PER_CONFIG,
        max_dof=MAX_DOF,
        dof_levels=DOF_LEVELS,
        gui=USE_GUI
    )

    generator.generate_dataset()
    generator.save_dataset(OUTPUT_FILE)
    generator.cleanup()

    print("\n" + "=" * 60)
    print("Dataset generation completed successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()