import time
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import argparse

import mlflow
from mlflow.tracking import MlflowClient

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data as geoData
import torch_geometric.transforms as T

from sklearn import metrics

from load_data import Data
from model import KGFlow, GaussianDiffusion, DeterministicFeedForwardNeuralNetwork

# os.environ['CUDA_VISIBLE_DEVICES']='4'
import setproctitle
setproctitle.setproctitle('KSTDiff@zzl')

device = torch.device('cuda')

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        xbatch = self.x[idx]
        ybatch = self.y[idx]
        sample = {"x": xbatch, "y": ybatch}
        # 返回一个 dict
        return sample

class Experiment:
    def __init__(self):
        self.num_iterations = params['num_iterations']
        self.lr = params['lr']
        self.batch_size = params['batch_size']
        self.dr = params['dr']
        self.kwargs = params
        self.kwargs['device'] = device

        self.g, self.g_train, self.g_samp = self.build_graph()

    def build_graph(self):
        edge_index = torch.tensor([[x[0] for x in d.kg_data], [x[2] for x in d.kg_data]],
                                  dtype=torch.long, device=device)
        edge_type = torch.tensor([x[1] for x in d.kg_data], dtype=torch.int64, device=device)
        g = geoData(edge_index=edge_index, edge_type=edge_type)

        # trainkg
        train_edge_index = torch.tensor([[x[0] for x in d.trainkg_data], [x[2] for x in d.trainkg_data]],
                                        dtype=torch.long, device=device)
        train_edge_type = torch.tensor([x[1] for x in d.trainkg_data], dtype=torch.int64, device=device)
        g_train = geoData(edge_index=train_edge_index, edge_type=train_edge_type)

        # samplekg
        sample_edge_index = torch.tensor([[x[0] for x in d.samplekg_data], [x[2] for x in d.samplekg_data]],
                                         dtype=torch.long, device=device)
        sample_edge_type = torch.tensor([x[1] for x in d.samplekg_data], dtype=torch.int64, device=device)
        g_samp = geoData(edge_index=sample_edge_index, edge_type=sample_edge_type)

        return g, g_train, g_samp

    # def extract_batch(trainids, start, interval):
    #     """
    #     Extracts a batch of items from the trainids list.
    #     Parameters:
    #     trainids (list): The list from which to extract items.
    #     start (int): The starting index for extraction.
    #     interval (int): The number of items to extract.
    #     Returns:
    #     list: A new list containing the extracted items.
    #     """
    #     # Ensure the start and interval are within the bounds of the trainids list
    #     if start < 0 or start >= len(trainids) or start + interval > len(trainids):
    #         raise ValueError("Start or interval out of bounds")
    #     # Extract the specified range from the list
    #     batch_trainids = trainids[start:start + interval]
    #     return batch_trainids

    def get_batch(self, train_data, trainids, idx):
        batch = train_data[idx:idx + self.batch_size]
        out = torch.tensor(batch, dtype=torch.float, device=device)  # bs*nreg*nhour*2
        # trainids = self.extract_batch(trainids, idx, self.batch_size)
        out = out[:, trainids, :, :]
        return out

    def evaluate_guidance_model(self, cond_pred_model, dataloader):
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batchx = batch['x']
                batchy = batch['y']
                pred = cond_pred_model(batchx)  # bs*1
                pred = pred.reshape(-1)
                # rescale
                m, M = d.min_data, d.max_data
                batchy = (batchy*(M-m)+m+M)/2
                pred = (pred*(M-m)+m+M)/2

                y_true.extend(batchy.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
            rmse = metrics.mean_squared_error(y_pred, y_true, squared=False)
        return rmse
    
    def train_guidance_model(self, cond_pred_model, dataloader, opt):
        cond_pred_model.train()
        lossfunc = nn.MSELoss()
        losses = []
        for i, batch in enumerate(dataloader):
            batchx = batch['x']
            batchy = batch['y']
            pred = cond_pred_model(batchx)  # bs*1
            batchy = batchy.reshape(pred.shape)
            loss = lossfunc(pred, batchy)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # print('loss:%.3f'%np.mean(losses))
        return np.mean(losses)

        
    def train_and_eval(self):
        print('building model....')
        model = KGFlow(d=d, **self.kwargs)
        model = model.to(device)
        trainids = torch.tensor(d.trainids, device=device)
        sampids = torch.tensor(d.sampids, device=device)

        nn_x = torch.tensor([x[0] for x in d.scale_pred_data], device=device)
        nn_y = torch.tensor([x[1] for x in d.scale_pred_data], device=device)
        nn_x_train, nn_y_train = nn_x[trainids], nn_y[trainids]
        nn_x_samp, nn_y_samp = nn_x[sampids], nn_y[sampids]

        nn_train = MyDataset(nn_x_train, nn_y_train)
        nn_test = MyDataset(nn_x_samp, nn_y_samp)
        train_loader = DataLoader(nn_train, batch_size=64, shuffle=True)
        test_loader = DataLoader(nn_test, batch_size=64, shuffle=False)

        dim_in = nn_x.shape[1]
        dim_out = 1
        cond_pred_model = DeterministicFeedForwardNeuralNetwork(dim_in=dim_in, dim_out=dim_out).to(device)
        opt_nn = torch.optim.Adam(cond_pred_model.parameters(), lr=self.kwargs['nn_lr'])
        # pretrain cond_pred_model
        rmse_train = self.evaluate_guidance_model(cond_pred_model, train_loader)
        rmse_test = self.evaluate_guidance_model(cond_pred_model, test_loader)

        print(rmse_train)
        for it in tqdm(range(1, 1 + self.kwargs['pretrain_epochs'])):
            loss_epoch = self.train_guidance_model(cond_pred_model, train_loader, opt_nn)
            rmse_train = self.evaluate_guidance_model(cond_pred_model, train_loader)
            rmse_test = self.evaluate_guidance_model(cond_pred_model, test_loader)
            mlflow.log_metrics({'pre_loss': loss_epoch,
                                'pre_rmse_train': rmse_train,
                                'pre_rmse_test': rmse_test}, step=it)


        diffusion = GaussianDiffusion(
            model,
            cond_pred_model=cond_pred_model,
            d=d,
            data_shape=(len(d.train_data[0]), len(d.train_data[0][0]), len(d.train_data[0][0][0])),
            g=self.g,
            g_train=self.g_train,
            g_samp=self.g_samp,
            image_size=128,
            beta_schedule=self.kwargs['beta_schedule'],  # 'cosine'
            timesteps=self.kwargs['diffusion_dteps'],  # number of steps
            loss_type=self.kwargs['loss_type'],  # L1 or L2
            objective=self.kwargs['objective']
        )
        diffusion = diffusion.to(device)

        opt = torch.optim.Adam(diffusion.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        train_data = d.train_data

        loss_epoch = []
        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            diffusion.train()

            np.random.shuffle(train_data)
            losses = []
            for j in tqdm(range(0, len(train_data), self.batch_size)):
                opt.zero_grad()
                data_batch = self.get_batch(train_data, trainids, j)
                loss = diffusion(data_batch, trainids)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.dr:
                scheduler.step()
            mlflow.log_metrics({'train_time': time.time()-start_train,
                                'loss': np.mean(losses),
                                'current_it': it}, step=it)
            print('loss:%.3f' % np.mean(losses))
            

            # train guidance model
            if it % self.kwargs['train_guidance_every_epochs'] == 0:
                for _ in range(1):
                    loss_epoch = self.train_guidance_model(cond_pred_model, train_loader, opt_nn)
                rmse_train = self.evaluate_guidance_model(cond_pred_model, train_loader)
                rmse_test = self.evaluate_guidance_model(cond_pred_model, test_loader)
                print('rmse train:%.3f'%(rmse_train))
                print('rmse test:%.3f'%(rmse_test))
                mlflow.log_metrics({'guide_loss': loss_epoch,
                                    'rmse_train': rmse_train,
                                    'rmse_test': rmse_test}, step=it)
            else:
                rmse_train = self.evaluate_guidance_model(cond_pred_model, train_loader)
                rmse_test = self.evaluate_guidance_model(cond_pred_model, test_loader)
                print('rmse train:%.3f'%(rmse_train))
                print('rmse test:%.3f'%(rmse_test))
                mlflow.log_metrics({'rmse_train': rmse_train,
                                    'rmse_test': rmse_test}, step=it)

            if it in [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]:
                sampled_flow = diffusion.sample(sampids, batch_size=self.kwargs['sample_num'])
                np.savez(archive_path + 'sample_{}.npz'.format(it),
                        sample=sampled_flow.detach().cpu().numpy())
                
            if it in [60, 200, 500, 1000, 1500, 2000]:
                # save model
                torch.save(diffusion.state_dict(), archive_path + "model_{}.pth".format(it))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=50000, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=2, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-5, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="Decay rate.")
    parser.add_argument("--seed", type=int, default=20, nargs="?", help="random seed.")
    
    parser.add_argument("--dim", type=int, default=64, nargs="?", help="sin emb dim")
    parser.add_argument("--num_heads", type=int, default=2, nargs="?", help="")
    parser.add_argument("--num_rgcns", type=int, default=1, nargs="?", help="")
    parser.add_argument("--num_flowrgcns", type=int, default=1, nargs="?", help="")

    parser.add_argument("--num_sas", type=int, default=1, nargs="?", help="")
    parser.add_argument("--dropout", type=float, default=0.0, nargs="?", help="")
    parser.add_argument("--kge_cat_dim", type=int, default=16, nargs="?", help="kge cat dim")
    parser.add_argument("--xt_cat_dim", type=int, default=16, nargs="?", help="xt cat dim")
    

    parser.add_argument('--pretrain', default=1, type=int, help='1-use pretrain kg embedding')
    parser.add_argument('--freeze', default=1, type=int, help='pretrain kg embedding freeze or not')

    parser.add_argument('--n_layer', default=5, type=int, help='number of residual layers')

    parser.add_argument("--dataset", type=str, default='bj', nargs="?", help="")
    # diffusion params
    parser.add_argument("--objective", type=str, default='pred_noise', nargs="?", help="pred_noise/pred_x0")
    parser.add_argument("--loss_type", type=str, default='l1', nargs="?", help="l1/l2")
    parser.add_argument("--beta_schedule", type=str, default='cosine', nargs="?", help="cosine/linear")
    parser.add_argument("--diffusion_dteps", type=int, default=1000, nargs="?", help="rt")
    parser.add_argument("--sample_num", type=int, default=4, nargs="?", help="Number of samples")
    # condition prediction model params
    parser.add_argument("--nn_lr", type=float, default=0.001, nargs="?", help="Learning rate of condition prediction model.")
    parser.add_argument("--pretrain_epochs", type=int, default=100, nargs="?", help="Number of pretrain iterations.")
    parser.add_argument("--train_guidance_every_epochs", type=int, default=1, nargs="?", help="train guidance every k epochs")

    args = parser.parse_args()
    print(args)



    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~

    experiment_name = 'KSTDiff'

    mlflow.set_tracking_uri('/data1/zhouzhilun/flow_generation/mlflow_output/')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
        print('Initial Create!')
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id
        print('Experiment Exists, Continuing')
    with mlflow.start_run(experiment_id=EXP_ID) as current_run:

        data_dir = "./data/data_{}/".format(args.dataset)
        archive_path = './output/output_{}/'.format(args.dataset)
        assert os.path.exists(data_dir)
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)

        # ~~~~~~~~~~~~~~~~~ reproduce setting ~~~~~~~~~~~~~~~~~~~~~
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        # torch.mps.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Loading data....')
        d = Data(data_dir=data_dir)
        params = vars(args)
        mlflow.log_params(params)

        experiment = Experiment()
        experiment.train_and_eval()
