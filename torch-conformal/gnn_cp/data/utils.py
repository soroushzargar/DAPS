import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Torch Graph Models are running on {device}")

from gnn_cp.data.data_manager import GraphDataManager
import gnn_cp.models.graph_models as graph_models
from gnn_cp.models.model_manager import GraphModelManager


def remove_edges_from_split_idx(
    edge_index, source_indices, destination_indices, directed=False
):
    """
    Build a mask where if the source and destination node are both false, the edge should be removed in the final edge_index.
    If the the src node comes from the src_index and arrives to a node beloging to dst_index, the edge has to be removed.
    The same is applied in the other direction for directed edges.
    """
    if directed:
        mask = ~torch.isin(edge_index[0], source_indices) | ~torch.isin(
            edge_index[1], destination_indices
        )
    else:
        mask = (
            ~torch.isin(edge_index[0], source_indices)
            | ~torch.isin(edge_index[1], destination_indices)
        ) & (
            ~torch.isin(edge_index[1], source_indices)
            | ~torch.isin(edge_index[0], destination_indices)
        )
    edge_index = edge_index[:, mask]

    return edge_index


def inductive_make_dataset_instances(
    data_dir,
    splits_dir,
    splits_config,
    models_cache_dir,
    dataset_key,
    model_class_name,
    models_config,
):
    dataset_manager = GraphDataManager(data_dir, splits_dir)
    data = dataset_manager.get_dataset_from_key(dataset_key)
    dataset = data.data

    instances, instances_codes = dataset_manager.load_splits(dataset_key)
    if len(instances) == 0:
        n_splits = splits_config.get("n_init_splits", 10)
        n_train = splits_config.get("n_train_labels", 20)
        n_val = splits_config.get("n_val_labels", 20)

        training_nodes = n_train * dataset_manager.get_num_classes(dataset)
        train_percentage = np.round((training_nodes / dataset.x.shape[0]), 10)

        validation_nodes = n_val * dataset_manager.get_num_classes(dataset)
        validation_percentage = np.round((validation_nodes / dataset.x.shape[0]), 10)

        print("Creating splits")
        new_splits = dataset_manager.init_k_splits(
            dataset, train_percentage, validation_percentage, k=n_splits
        )
        instances_codes += dataset_manager.save_splits(
            dataset_key=dataset_key, splits=new_splits
        )
        instances += new_splits

    print("Dataset Loaded Successfully!")
    print(f"Following labeled splits:")
    for c in range(dataset_manager.get_num_classes(dataset)):
        print(
            f"class {c}: train={(dataset.y[instances[2]['train_idx']] == c).sum()}, val={(dataset.y[instances[2]['val_idx']] == c).sum()}"
        )

    print("====================================")
    print("Loading Models")

    print(f"Loading Models {model_class_name}")
    model_hparams = models_config.get(model_class_name, {}).get("config", {})
    optimizer_hparams = models_config.get(model_class_name, {}).get("optimizer", {})
    model_hparams.update(
        {"n_features": dataset.x.shape[1], "n_classes": dataset.y.max().item() + 1}
    )

    model_class = getattr(graph_models, model_class_name)
    lr = optimizer_hparams.get("lr", 0.01)
    weight_decay = optimizer_hparams.get("weight_decay", 0.0)

    for instance_idx, instance in enumerate(instances):
        train_idx = instance["train_idx"].to(device)
        val_idx = instance["train_idx"].to(device)
        test_idx = instance["train_idx"].to(device)

        train_val_inductive_dataset = dataset.clone()
        train_val_inductive_dataset.edge_index = remove_edges_from_split_idx(
            edge_index=train_val_inductive_dataset.edge_index.clone().to(device),
            source_indices=torch.cat([train_idx, val_idx], dim=0),
            destination_indices=test_idx,
        )

        model = GraphModelManager(
            model=model_class(**model_hparams),
            optimizer_lambda=lambda model_params: torch.optim.Adam(
                model_params, lr=lr, weight_decay=weight_decay
            ),
            checkpoint_address=models_cache_dir,
            model_name=f"{dataset_key}-ins{instances_codes[instance_idx]}-{model_class.__name__}",
        )
        if not model.load_model():
            print("Model not found! Training a new one.")
            model.fit(
                train_val_inductive_dataset,
                training_idx=instance["train_idx"],
                validation_idx=instance["val_idx"],
                n_epochs=10000,
                loss_fn=F.cross_entropy,
                warmup_steps=10,
                early_stopping_warnings=100,
                verbose_on_quantile=0.1,
            )
        else:
            pass
        model.model = model.model.to(device)
        y_pred = model.predict(
            train_val_inductive_dataset,
            test_idx=instance["test_idx"],
            return_embeddings=False,
        )
        accuracy = accuracy_score(
            y_true=train_val_inductive_dataset.y[instance["test_idx"]].cpu().numpy(),
            y_pred=y_pred.cpu().numpy(),
        )
        instance.update({"model": model, "accuracy": accuracy})
    print(
        f"Accuracy: {np.mean([instance['accuracy'] for instance in instances])} +- {np.std([instance['accuracy'] for instance in instances])}"
    )

    return instances


def make_dataset_instances(
    data_dir,
    splits_dir,
    splits_config,
    models_cache_dir,
    dataset_key,
    model_class_name,
    models_config,
):
    dataset_manager = GraphDataManager(data_dir, splits_dir)
    data = dataset_manager.get_dataset_from_key(dataset_key)
    dataset = data.data

    instances, instances_codes = dataset_manager.load_splits(dataset_key)
    if len(instances) == 0:
        n_splits = splits_config.get("n_init_splits", 10)
        n_train = splits_config.get("n_train_labels", 20)
        n_val = splits_config.get("n_val_labels", 20)

        training_nodes = n_train * dataset_manager.get_num_classes(dataset)
        train_percentage = np.round((training_nodes / dataset.x.shape[0]), 10)

        validation_nodes = n_val * dataset_manager.get_num_classes(dataset)
        validation_percentage = np.round((validation_nodes / dataset.x.shape[0]), 10)

        print("Creating splits")
        new_splits = dataset_manager.init_k_splits(
            dataset, train_percentage, validation_percentage, k=n_splits
        )
        instances_codes += dataset_manager.save_splits(
            dataset_key=dataset_key, splits=new_splits
        )
        instances += new_splits

    print("Dataset Loaded Successfully!")
    print(f"Following labeled splits:")
    for c in range(dataset_manager.get_num_classes(dataset)):
        print(
            f"class {c}: train={(dataset.y[instances[2]['train_idx']] == c).sum()}, val={(dataset.y[instances[2]['val_idx']] == c).sum()}"
        )

    print("====================================")
    print("Loading Models")

    print(f"Loading Models {model_class_name}")
    model_hparams = models_config.get(model_class_name, {}).get("config", {})
    optimizer_hparams = models_config.get(model_class_name, {}).get("optimizer", {})
    model_hparams.update(
        {"n_features": dataset.x.shape[1], "n_classes": dataset.y.max().item() + 1}
    )

    model_class = getattr(graph_models, model_class_name)
    lr = optimizer_hparams.get("lr", 0.01)
    weight_decay = optimizer_hparams.get("weight_decay", 0.0)

    for instance_idx, instance in enumerate(instances):
        model = GraphModelManager(
            model=model_class(**model_hparams),
            optimizer_lambda=lambda model_params: torch.optim.Adam(
                model_params, lr=lr, weight_decay=weight_decay
            ),
            checkpoint_address=models_cache_dir,
            model_name=f"{dataset_key}-ins{instances_codes[instance_idx]}-{model_class.__name__}",
        )
        if not model.load_model():
            print("Models not found! Training from scratch.")
            model.fit(
                dataset,
                training_idx=instance["train_idx"],
                validation_idx=instance["val_idx"],
                n_epochs=10000,
                loss_fn=F.cross_entropy,
                warmup_steps=10,
                early_stopping_warnings=100,
                verbose_on_quantile=0.1,
            )
        else:
            pass
        model.model = model.model.to(device)
        y_pred = model.predict(
            dataset, test_idx=instance["test_idx"], return_embeddings=False
        )
        accuracy = accuracy_score(
            y_true=dataset.y[instance["test_idx"]].cpu().numpy(),
            y_pred=y_pred.cpu().numpy(),
        )
        instance.update({"model": model, "accuracy": accuracy})
    print(
        f"Accuracy: {np.mean([instance['accuracy'] for instance in instances])} +- {np.std([instance['accuracy'] for instance in instances])}"
    )

    return instances
