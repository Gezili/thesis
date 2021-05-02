import demo.main
import numpy as np

def run():
    return demo.main.main([])

if __name__ == '__main__':
    generated_grasps, generated_scores = run()

    generated_grasps = np.array(generated_grasps)
    generated_scores = np.array(generated_scores)

    #Grip ABOVE table
    generated_scores = generated_scores[generated_grasps[:,2,3] > 0.1]

    generated_grasps = generated_grasps[generated_grasps[:,2,3] > 0.1]

    generated_grasps = generated_grasps[np.argsort(generated_scores)[::-1]]

    generated_grasp = generated_grasps[0]
    #generated_grasp[:3,3] = generated_grasp[:3,3] - generated_grasp[:3,:3] @ np.array([0,0,0.2])


    artifact_path = r'\\wsl$\Ubuntu-18.04\root\catkin_ws\src\panda_simulator\panda_simulator_examples\scripts'

    np.save(artifact_path + '\plan_pose.npy', generated_grasp)