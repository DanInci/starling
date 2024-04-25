import os
import json
import argparse

import anndata as ad
import numpy as np
import scanpy as sc

import pytorch_lightning as pl
from lightning_lite import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

from starling import starling, utility, label_mapper

COLUMNS_OF_INTEREST = ['sample_id', 'object_id', 'cell_type', 'init_label',
                       'st_label', 'doublet', 'doublet_prob', 'max_assign_prob']

UNLABELED_CELL_TYPES = ['unlabeled', 'undefined', 'unknown', 'BnTcell', "BnT cell"]


def _plot_umap(adata):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    fig = sc.pl.umap(adata, color=['cell_type', 'init_label', 'st_label'], size=14, ncols=3, wspace=0.3, return_fig=True)

    return fig


def main():
    parser = argparse.ArgumentParser(description='starling')
    parser.add_argument('--base_path', type=str, required=True,
                        help='configuration_path')
    parser.add_argument('--scale', type=str, required=False, default=True)
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

    # Load data
    adata = ad.read_h5ad(args.dataset)

    # Scale expression data
    if args.scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled_df = scaler.fit_transform(adata.X)
        adata.X = X_scaled_df

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

    # Setting initializations
    st = starling.ST(adata, learning_rate=args.lr, singlet_prop=args.error_free_cells_prop)

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

    prob_matrix = adata.obsm['assignment_prob_matrix']
    prob_vector = np.array([f"[{', '.join(map(str, row))}]" for row in prob_matrix])
    adata.obs['st_prob_list'] = prob_vector

    results_df = adata.obs[COLUMNS_OF_INTEREST]
    results_df = results_df.rename(columns={
        'sample_id': 'image_id',
        'object_id': 'cell_id',
        'cell_type': 'label',
        'init_label': 'init_pred',
        'st_label': 'st_pred',
        'max_assign_prob': 'st_pred_prob'
    })

    # Save results csv
    results_df.to_csv(os.path.join(args.base_path, 'starling_results.csv'), index=False)

    # Plot UMAP of predictions
    figure = _plot_umap(adata)
    figure.savefig(os.path.join(args.base_path, 'UMAP_predictions.pdf'), format="pdf", bbox_inches="tight")

    # Calculate ARI Score of labeled results
    labeled_results_df = results_df[~results_df['label'].isin(UNLABELED_CELL_TYPES)]

    print("Init ARI:", adjusted_rand_score(labeled_results_df['label'], labeled_results_df['init_pred']))
    print("Starling ARI:", adjusted_rand_score(labeled_results_df['label'], labeled_results_df['st_pred']))


if __name__ == '__main__':
    main()
