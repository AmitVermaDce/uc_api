import torch
import numpy as np


class SentimentHelper:

    def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
        ):
        model = model.train()

        losses = []
        correct_predictions = 0
        
        print(len([each for each in data_loader]))
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["sentimentRating"].to(device)
            print("Input size: ", input_ids.shape)
            print("Attention size: ", attention_mask.shape)
            print("Target size: ", targets.shape)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)#.detach().cpu().numpy()
            losses.append(loss.item())

            loss.backward()

            # Avoiding the exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print("----------\n")


        return correct_predictions.double() / n_examples, np.mean(losses)


    def eval_model(model, data_loader, loss_fn, device, n_examples):
        model = model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["sentimentRating"].to(device)
                print("Input size: ", input_ids.shape)
                print("Attention size: ", attention_mask.shape)
                print("Target size: ", targets.shape)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fn(outputs, targets)

                correct_predictions += torch.sum(preds == targets)#.detach().cpu().numpy()
                losses.append(loss.item())
                print("----------\n")

            return correct_predictions.double() / n_examples, np.mean(losses)
