"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from lavis.datasets.datasets.scienceqa_datasets import ScienceQADataset, ScienceQAEvalDataset
from lavis.datasets.datasets.vizwiz_datasets import VizWizDataset, VizWizEvalDataset
from lavis.datasets.datasets.iconqa_datasets import IconQADataset, IconQAEvalDataset
from lavis.datasets.datasets.rotation_datasets import RotationDataset, RotationEvalDataset
from lavis.datasets.datasets.symmetric_datasets import SymmetricDataset,SymmetricEvalDataset

@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }


@registry.register_builder("vg_vqa")
class VGVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
    }
    
@registry.register_builder("scienceqa")
class ScienceQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ScienceQADataset
    eval_dataset_cls = ScienceQAEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/scienceqa/defaults.yaml",
    }

@registry.register_builder("vizwiz")
class VizWizBuilder(BaseDatasetBuilder):
    train_dataset_cls = VizWizDataset
    eval_dataset_cls = VizWizEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vizwiz/defaults.yaml",
    }
    
@registry.register_builder("iconqa")
class IconQABuilder(BaseDatasetBuilder):
    train_dataset_cls = IconQADataset
    eval_dataset_cls = IconQAEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/iconqa/defaults.yaml",
    }
    
@registry.register_builder("rotation")
class RotationBuilder(BaseDatasetBuilder):
    train_dataset_cls = RotationDataset
    eval_dataset_cls = RotationEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rotation/defaults.yaml",
    }
    
@registry.register_builder("symmetric")
class SymmetricBuilder(BaseDatasetBuilder):
    train_dataset_cls = SymmetricDataset
    eval_dataset_cls = SymmetricEvalDataset
    
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/symmetric/defaults.yaml",
    }

