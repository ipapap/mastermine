import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim import lr_scheduler

from vpr.models import helper
import numpy as np
import os
import torch.distributed as dist


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """
    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM, AVG
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
  
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        # in_shape=(3,256,256)
        # self.canonicalization_network =ESCNNEquivariantNetwork(in_shape=in_shape,out_channels=3,kernel_size=5,num_rotations=8)#ESCNNEquivariantNetwork(in_shape,**canonicalization_hyperparams.network_hyperparams) 
        # self.canonicalizer = GroupEquivariantImageCanonicalization(self.canonicalization_network,canonicalization_hyperparams,in_shape)
        
        self.validation_step_outputs = [[] for i in range(10)] #assuming max 10 validation sets
        self.training_step_outputs = []
        self.validation_step_outputs_idx=[[] for i in range(10)] #assuming max 10 validation sets
        self.testing_step_outputs = []
        self.testing_step_outputs_idx=[]
        self.best_val_acc = 0


    def forward(self, x):
        # x = self.canonicalizer(x)
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            # loss+= self.canonicalizer.get_prior_regularization_loss()
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining

            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        # if self.trainer.is_global_zero:
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True,sync_dist=True)#, rank_zero_only=True,sync_dist=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        self.training_step_outputs.append(loss.item())
        # if self.trainer.is_global_zero:
        self.log('loss', loss.item(), logger=True,sync_dist=True)#,rank_zero_only=True, sync_dist=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []
        self.training_step_outputs=[]

       
    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        places, idxs = batch
        # calculate descriptors
        descriptors = self(places)
        assert len(descriptors) == len(idxs)
        self.validation_step_outputs[dataloader_idx].append(descriptors.detach().cpu())
        self.validation_step_outputs_idx[dataloader_idx].append(idxs.detach().cpu())
        # return descriptors.detach().cpu()

            
