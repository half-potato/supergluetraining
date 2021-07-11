import numpy as np
import torch
import os
import argparse
from load_scannet import ScannetDataset, SceneFiles
from models.superglue import SuperGlue
from models.cpainted import CPainted
from models.superpoint import SuperPoint
from utils import ground_truth_matches
from multiprocessing import Pool
import h5py
from tqdm import tqdm


def process_keypoints(kpts0, kpts1, depth0, depth1, K, T_0to1):
    MI, MJ, WI, WJ = ground_truth_matches(kpts0.cpu().numpy(), kpts1.cpu().numpy(),
            K.numpy(), K.numpy(), T_0to1.numpy(), depth0.numpy(), depth1.numpy())
    return {
        'match_indices0': torch.tensor(MI),
        'match_indices1': torch.tensor(MJ),
        'match_weights0': torch.tensor(WI),
        'match_weights1': torch.tensor(WJ)
    }

NAMES = {
    'keypoints': [(N, K, 2), np.float32],
    'match_indices': [(N, K+1, 1), np.int64],
    'match_weights': [(N, K+1, 1), np.float32],
    'descriptors': [(N, 256, K), np.float32],
    'scores': [(N, K), np.float32],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SuperGlue training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'scans_path', type=str,
        help='Path to the directory of scans')
    parser.add_argument(
        'pairs_path', type=str,
        help='Path to training pair files')
    parser.add_argument(
        'output_path', type=str,
        help='Path save h5py path to ')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size')
    opt = parser.parse_args()

    print("Loading scene files")
    scene_files = SceneFiles(opt.pairs_path, opt.scans_path)
    print("Loading scannet dataset")
    scannet = ScannetDataset(scene_files)
    trainloader = torch.utils.data.DataLoader(dataset=scannet, shuffle=False,
            batch_size=opt.batch_size, num_workers=opt.num_threads, drop_last=True)
    print(f"Done. Loader length: {len(trainloader)}")

    K = 400

    #  superpoint = SuperPoint({
    #      'nms_radius': 4,
    #      'keypoint_threshold': 0.005,
    #      'max_keypoints': 400
    #  }).eval().cuda()
    superpoint = CPainted({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': K
    }).eval().cuda()

    handle = h5py.File(opt.output_path, 'w', libver='latest')
    N = len(scannet)
    w, h = (480, 640)

    datasets = {}

    for k, (shape, dtype) in NAMES.items():
        datasets[k+'0'] = handle.create_dataset(k+'0', shape, dtype=dtype)
        datasets[k+'1'] = handle.create_dataset(k+'1', shape, dtype=dtype)

    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):

        # detect superpoint keypoints. Format (N, (col, row))
        with torch.no_grad():
            p1 = superpoint({'image': data['image0'].cuda()})
            p2 = superpoint({'image': data['image1'].cuda()})

            # compute groundtruth matches
            processed = []
            params = zip(p1['keypoints'], p2['keypoints'],
                    data['depth0'], data['depth1'], data['intrinsics'], data['T_0to1'])
            for p in params:
                processed.append(process_keypoints(*p))
            inputs = {}
            for k in processed[0]:
                inputs[k] = torch.stack([e[k] for e in processed], axis=0)
            inputs['keypoints0'] = torch.stack(p1['keypoints'], axis=0)
            inputs['descriptors0'] = torch.stack(p1['descriptors'], axis=0)
            inputs['scores0'] = torch.stack(p1['scores'], axis=0)
            inputs['keypoints1'] = torch.stack(p2['keypoints'], axis=0)
            inputs['descriptors1'] = torch.stack(p2['descriptors'], axis=0)
            inputs['scores1'] = torch.stack(p2['scores'], axis=0)

            for k, v in inputs.items():
                datasets[k][(i*opt.batch_size):((i+1)*opt.batch_size)] = v.cpu().numpy()

    handle.close()
