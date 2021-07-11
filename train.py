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
from cache_dataset import NAMES

def loss_fn(P, match_indices0, match_indices1, match_weights0, match_weights1):
    bs, ft, _ = P.shape

    l0 = -P.reshape(bs*ft, ft)[range(bs*ft), match_indices0.reshape(bs*ft)]
    l1 = -P.transpose(1,2).reshape(bs*ft, ft)[range(bs*ft), match_indices1.reshape(bs*ft)]

    loss = torch.dot(l0, match_weights0.reshape(bs*ft)) + torch.dot(l1, match_weights1.reshape(bs*ft))

    return loss / bs

def batch_to_device(batch, device):
    for k in batch.keys():
        if type(batch[k]) == list:
            if len(batch[k]) == 0 or type(batch[k][0]) == list:
                continue
            batch[k] = [item.to(device) if item is not None else None for item in batch[k]]
        else:
            batch[k] = batch[k].to(device)
    return batch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, fpath):
        self.fpath = fpath
        self.handle = h5py.File(self.fpath, 'r', libver='latest')
        self.length = self.handle['keypoints0'].shape[0]
        self.handle.close()

    def open_if_closed(self):
        if self.handle is None:
            self.handle = h5py.File(self.fpath, 'r', libver='latest')

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        self.open_if_closed()
        out = {}
        for k in NAMES:
            out[k] = self.handle[k][i]
        if out['scores0'].sum() == 0:
            return self[(i+1)%len(self)]
        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SuperGlue training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_path', type=str,
        help='Path to cached dataset')
    parser.add_argument(
        '--num_epochs', type=int, default=500,
        help='Number of epochs')
    parser.add_argument(
        '--num_iters', type=int, default=900000,
        help='Break after this many iterations')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size')
    parser.add_argument(
        '--num_threads', type=int, default=0,
        help='Number of threads for the DataLoader and for computing groundtruth')
    parser.add_argument(
        '--num_batches_per_optimizer_step', type=int, default=1,
        help='Number of batches processed before running the optimization step')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning rate')
    parser.add_argument(
        '--lr_decay', type=float, default=0.999992,
        help='Exponential learning rate decay')
    parser.add_argument(
        '--lr_decay_iter', type=int, default=100000,
        help='Decay the learning rate after this iteration')
    parser.add_argument(
        '--checkpoint_dir', type=str, default="dump_weights",
        help='Weights are saved periodically to this directory')
    parser.add_argument(
        '--save_every_n_epochs', type=int, default=1,
        help='Weights are saved after every n epochs')
    opt = parser.parse_args()

    device = torch.device('cuda')

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    dataset = Dataset(opt.dataset_path)
    trainloader = torch.utils.data.DataLoader(dataset, shuffle=True,
            batch_size=opt.batch_size, num_workers=opt.num_threads, drop_last=True)
    print(f"Done. Loader length: {len(trainloader)}")

    superglue = SuperGlue({}).train().cuda()
    #  superpoint = SuperPoint({
    #      'nms_radius': 4,
    #      'keypoint_threshold': 0.005,
    #      'max_keypoints': 400
    #  }).eval().cuda()
    superpoint = CPainted({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 400
    }).eval().cuda()

    optimizer = torch.optim.Adam(params=superglue.parameters(), lr=opt.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.lr_decay)
    pool = Pool(max(opt.num_threads, 1))
    iteration = 0

    print("Training")
    for epoch in range(opt.num_epochs):
        for i, inputs in enumerate(trainloader):
            # check if batch is a failed one
            inputs = batch_to_device(inputs, device)

            # superglue forward and backward pass
            if i % opt.num_batches_per_optimizer_step == 0:
                optimizer.zero_grad()
            P = superglue(inputs)
            loss = loss_fn(P, inputs["match_indices0"].cuda(), inputs["match_indices1"].cuda(),
                    inputs["match_weights0"].cuda(), inputs["match_weights1"].cuda())
            loss.backward()
            if i % opt.num_batches_per_optimizer_step == opt.num_batches_per_optimizer_step - 1:
                optimizer.step()
                iteration += 1

            # update learning rate
            if iteration > opt.lr_decay_iter and i % opt.num_batches_per_optimizer_step == 0:
                lr_scheduler.step()

            print("Epoch:", epoch, "Batch:", i, "Loss:", loss)

        if epoch % opt.save_every_n_epochs == 0:
            torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(opt.checkpoint_dir, epoch))
        if iteration == opt.num_iters:
            torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(opt.checkpoint_dir, epoch))
            break
