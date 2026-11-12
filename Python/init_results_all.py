"""Write outputs/results_all.npz with the static landmarks of the camera solve (result_XM.npz `p`): the base that
complete_edg_table.py writes the completed EDG into (edg_reconstruct, edg_rotation, edg_edges_union and
dynamics_points_reconstruct, all from the merged dynamic tracks). train_4dgs_edg.py reads static_landmarks."""
import numpy as np

import utils.definitions

rt = np.load(str(utils.definitions.rootpath / "outputs" / "result_XM.npz"))
landmarks = rt["p"].T.astype(np.float64) if "p" in rt else None
np.savez_compressed(utils.definitions.output_dir / "results_all.npz", static_landmarks=landmarks)
print(f"results_all.npz: {0 if landmarks is None else len(landmarks)} static landmarks")
