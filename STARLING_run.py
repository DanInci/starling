import os
import json
import argparse

import anndata as ad
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import adjusted_rand_score

from starling import starling, utility, label_mapper

COLUMNS_OF_INTEREST = ['sample_id', 'object_id', 'cell_type', 'init_label',
                       'st_label', 'doublet', 'doublet_prob', 'max_assign_prob']

UNLABELED_CELL_TYPES = ['unlabeled', 'undefined', 'unknown', 'BnTcell', "BnT cell"]


def main():
    parser = argparse.ArgumentParser(description='starling')
    parser.add_argument('--base_path', type=str, required=True,
                        help='configuration_path')
    args = parser.parse_args()

    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    args.dataset = config['dataset']
    args.init_clustering_method = config['init_clustering_method']
    args.error_free_cells_prop = config['error_free_cells_prop']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.num_classes = config['num_classes']
    args.seed = config['seed']

    seed_everything(args.seed, workers=True)

    # Prepare dataset
    dataset_df = pd.read_csv(args.dataset)
    dataset_df['rownames'] = dataset_df['sample_id'] + '_' + dataset_df['object_id'].astype(str)
    dataset_df.set_index('rownames', inplace=True)

    interest_columns = ['image_name', 'sample_id', 'object_id', 'cell_type', 'patient_id', 'area', 'x', 'y', 'user_init_label']
    X_columns = [c for c in dataset_df.columns if c not in interest_columns]
    obs_columns = [c for c in dataset_df.columns if c in interest_columns]
    obs_df = dataset_df[obs_columns]
    X_df = dataset_df[X_columns]

    adata = ad.AnnData(X=X_df.values, obs=obs_df)
    adata.obs_names = obs_df.index

    # Annotate initial clustering with KM clustering results
    print(f'Initial cluster annotation using `{args.init_clustering_method}` algorithm.')
    labels = np.array(adata.obs.get('user_init_label'))
    adata = utility.init_clustering(args.init_clustering_method, adata,
                                    k=args.num_classes,
                                    labels=labels)

    assert "init_exp_centroids" in adata.varm
    assert adata.varm["init_exp_centroids"].shape == (adata.X.shape[1], args.num_classes)

    assert "init_exp_centroids" in adata.varm
    assert adata.varm["init_exp_variances"].shape == (adata.X.shape[1], args.num_classes)

    assert "init_label" in adata.obs
    assert adata.obs["init_label"].shape == (adata.X.shape[0],)

    labeled_obs = adata.obs[~adata.obs['cell_type'].isin(UNLABELED_CELL_TYPES)]
    print("Init ARI:", adjusted_rand_score(labeled_obs['cell_type'], labeled_obs['init_label']))

    # Setting initializations
    st = starling.ST(adata, learning_rate=args.lr, singlet_prop=args.error_free_cells_prop)
    print(st.singlet_prop)
    log_tb = TensorBoardLogger(save_dir="log")
    cb_early_stopping = EarlyStopping(monitor="train_loss", mode="min", verbose=False)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        deterministic=True,
        callbacks=[cb_early_stopping],
        logger=[log_tb],
    )

    # Train Starling
    trainer.fit(st)
    st.result()

    # Map initial clustering and starling clustering to labels
    mapper = label_mapper.AutomatedLabelMapping(adata.obs['cell_type'], adata.obs['init_label'])

    # map init label to cell type
    adata.obs['init_label'] = mapper.get_pred_labels(adata.obs['init_label'])

    # map st label to cell type
    st.adata.obs['st_label'] = mapper.get_pred_labels(st.adata.obs['st_label'])

    st.labeled_obs = st.adata.obs[~st.adata.obs['cell_type'].isin(UNLABELED_CELL_TYPES)]
    print("Starling ARI:", adjusted_rand_score(st.labeled_obs['cell_type'], st.labeled_obs['st_label']))

    # Save annotated dataset
    adata.write(filename=os.path.join(args.base_path, 'starling_anndata.h5ad'))

    # Save results csv
    results_df = adata.obs[COLUMNS_OF_INTEREST]

    prob_matrix = adata.obsm['assignment_prob_matrix']
    prob_vector = np.array([f"[{', '.join(map(str, row))}]" for row in prob_matrix])
    results_df['prob_list'] = prob_vector

    # rename columns
    results_df.columns = ['image_id', 'cell_id', 'label', 'init_pred', 'st_pred', 'doublet', 'doubled_prob',
                          'st_pred_prob', 'st_prob_list']

    results_df.to_csv(os.path.join(args.base_path, 'starling_results.csv'), index=False)


if __name__ == '__main__':
    main()
