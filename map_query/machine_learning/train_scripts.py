from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #type:ignore
from tqdm import tqdm

from map_query.machine_learning.models import load_model
from map_query.machine_learning.datasets import TrainingThumbnails
from map_query.machine_learning.transforms import parse_transformations_list

### <!> city_name will not be used but it is still passed as argument, that's not good
def train_segmentation(
    paths:Dict[str,Path],
    city_name:str,
    operation_dict:Dict):
    ### Unpack essentials
    path_to_training_data = Path(operation_dict['essentials']['path_to_training_data'])
    feature_names = operation_dict['essentials']['feature_names']
    images_path = Path(operation_dict['essentials']['images_path'])
    runs_path = Path(operation_dict['essentials']['runs_path'])
    models_path = Path(operation_dict['essentials']['models_path'])
    models_path.mkdir(exist_ok=True, parents=True)
    images_path.mkdir(exist_ok=True, parents=True)
    runs_path.mkdir(exist_ok=True, parents=True)
    ### Unpack others dicts
    data_feeding_dict = operation_dict['data_feeding_dict']
    model_dict = operation_dict['model_dict']
    training_dict = operation_dict['training_dict']

    device = torch.device(model_dict['device'])

    ### Loading dataset
    dataset = TrainingThumbnails(
        path_to_training_data,
        parse_transformations_list(data_feeding_dict['transforms'])
    )

    dataloader = DataLoader(
        dataset = dataset,
        batch_size = data_feeding_dict['batch_size'],
        shuffle = data_feeding_dict['shuffle'],
        num_workers = data_feeding_dict['num_workers'],
        drop_last=False)

    ### Loading model
    model = load_model(model_dict, device)
    model.to(device)
    model.train()

    ### Loading optimiser
    optimiser = optim.Adam(model.parameters(), lr=training_dict['learning_rate'])
    if 'optimiser_load_path' in training_dict:
        optimiser.load_state_dict(torch.load(training_dict['optimiser_load_path']))
    ### Defining loss
    cross_entropy_loss = torch.nn.BCELoss()

    writer = SummaryWriter(log_dir = runs_path)

    for epoch in range(training_dict['epochs']):
        print(f'Epoch {epoch} / {training_dict["epochs"]}')
        for index, (input_batch, target_batch) in tqdm(enumerate(dataloader)):
            ### Sending the data to the device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            ### Gradient-descent step
            optimiser.zero_grad()
            segmentation = model(input_batch)
            loss = cross_entropy_loss(segmentation, target_batch)
            loss.backward()
            optimiser.step()
            ### Observables
            if index % 10 == 0:
                g_step = index + epoch*len(dataloader)
                print('')
                print(f'Cross-Entropy loss at step {g_step}: {loss}')
            if index % 100 == 0:
                g_step = index + epoch*len(dataloader)
                print('')
                print(f'Cross-Entropy loss at step {g_step}: {loss}')
                writer.add_scalar('Cross-Entropy', scalar_value = loss.item(), global_step = g_step)
                target_tensor = torch.cat(
                    [input_batch[0,1].detach().cpu()*255] + \
                    [(input_batch[0,1]*target_batch[0,i]).detach().cpu()*255 for i in range(1+len(feature_names))],
                    dim = 1
                )
                approx_tensor = torch.cat(
                    [input_batch[0,1].detach().cpu()*255] + \
                    [segmentation[0,i].detach().cpu()*255 for i in range(1+len(feature_names))],
                    dim = 1
                )
                image_array = torch.cat(
                    (target_tensor, approx_tensor), 0
                ).numpy()
                cv2.imwrite(str(images_path.joinpath(f'curent_segmentation.jpg')), np.uint8(image_array))

        torch.save(optimiser.state_dict(), models_path.joinpath('segmentation_optimiser.pth'))
        torch.save(model.state_dict(), models_path.joinpath('segmentation_model.pth'))

    return 0