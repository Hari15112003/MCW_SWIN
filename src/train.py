import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.eval import eval_callback

def train_loop( model: nn.Module,dataloader : DataLoader, device : torch.device, criterion: torch.nn.modules.loss, optimizer: torch.optim,
                    max_iterations: int, current_epoch: int, max_epochs: int,
                    debug_steps: int = 1000 ):

        
        # switch to training mode
        model = model.to(device)
        model.train()

        avg_loss = 0.0

        for i, ( images, target) in tqdm(enumerate(dataloader), total=max_iterations):
            if i == max_iterations:
                break

            images = images.to(device)
            target = target.to(device)

            # compute model output
            output = model(images)
            loss = criterion(output, target)
            avg_loss += loss.item()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % debug_steps == 0:
                eval_accuracy = eval_callback(model)
                print(f"At the end of Epoch #{current_epoch}/{max_epochs}: "
      f"Global Avg Loss={avg_loss / max_iterations:.6f}, Eval Accuracy={eval_accuracy:.4f}")

                # switch to training mode after evaluation
                model.train()

        eval_accuracy = eval_callback(model)
        print("eval : ", eval_accuracy)
        print(f'At the end of Epoch #{current_epoch}/{max_epochs}: '
      f'Global Avg Loss={avg_loss / max_iterations:.6f}, Eval Accuracy={eval_accuracy:.4f}')


        

def train( model: nn.Module,dataloader : DataLoader, device : torch.device ,  max_epochs: int = 20, learning_rate: int = 0.1,
              weight_decay: float = 1e-4, decay_rate: float = 0.1, learning_rate_schedule: list = None,
              debug_steps: int = 1000 ,):
        
        max_iterations = len(dataloader)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        if learning_rate_schedule:
            learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, learning_rate_schedule,
                                                                     gamma=decay_rate)

        for current_epoch in range(max_epochs):
            train_loop(model, dataloader , device, loss, optimizer, max_iterations, current_epoch + 1, max_epochs, debug_steps,
                             )

            if learning_rate_schedule:
                learning_rate_scheduler.step()