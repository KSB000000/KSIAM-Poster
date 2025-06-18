import os
import json
import pandas as pd
from PIL import Image
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict
import os
import json
import pandas as pd
from PIL import Image
import torch
from lavis.datasets.datasets.base_dataset import BaseDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation.iloc[index]

        return OrderedDict(
            {
                "file": ann["input_image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": ann["answer"],  # 정답 번호
                "image": sample["image"],
            }
        )

class RotationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, train_samples_portion="all"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        
        # JSON 파일을 DataFrame으로 로드
        with open(ann_paths[0], "r") as f:
            self.annotation = json.load(f)
        self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')

        # 데이터 샘플링 (필요할 경우)
        if not ((type(train_samples_portion) == int and train_samples_portion > 0) or train_samples_portion == "all"):
            raise ValueError("train_samples_portion must be a positive integer or \"all\"")
        if train_samples_portion != "all":
            self.annotation = self.annotation.sample(n=train_samples_portion)

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        # 원본 이미지 로드 & Tensor 변환
        image_path = ann["input_image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # 보기(Options) 이미지 로드 & Tensor 변환
        option_tensors = {}
        option_texts = []
        for idx, (key, option_path) in enumerate(ann["options"].items(), start=1):
            option_image = Image.open(option_path).convert("RGB")
            option_tensors[key] = self.vis_processor(option_image)
            option_texts.append(f'"{key}" <Image>')

        # Multi-Image-Choice Instruction Format (보기 번호와 함께)
        instruction = f"""<Image> Question: {ann["question"]}  
        Answer: {ann["answer"]}"""

        return {
            "image": image,
            "text_input": instruction,
            "choices": option_tensors,  # 보기 이미지 Tensor 포함
            "text_output": str(ann["answer"])  # 정답 번호를 문자열로 유지 (예: "1", "2", "3", "4")
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []
        choice_tensors = {str(i): [] for i in range(1, 5)}  # 보기들도 배치 처리 (1~4 유지)

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])  # 정답 번호만 저장
        
            # 보기 이미지 텐서도 저장
            for key in choice_tensors.keys():  # 모든 키(1~4)에 대해
                if key in sample["choices"]:
                    choice_tensors[key].append(sample["choices"][key])
                else:
                    # 만약 키가 없다면 더미 텐서를 추가 (예: 검정색 이미지)
                    choice_tensors[key].append(torch.zeros_like(list(sample["choices"].values())[0]))

        # 보기 이미지들도 스택해서 배치 구성
        for key in choice_tensors:
            choice_tensors[key] = torch.stack(choice_tensors[key], dim=0)

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,  # 정답 번호만 리스트로 반환
            "choices": choice_tensors  # 보기들도 Tensor 형태로 배치 구성
        }

class RotationEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_paths (string): Path to the JSON annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])


        # JSON 데이터를 DataFrame으로 변환
        with open(ann_paths[0], "r") as f:
            self.annotation = json.load(f)
        self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        # 원본 이미지 로드 & Tensor 변환
        image_path = ann["input_image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # 보기(Options) 이미지 로드 & Tensor 변환
        option_tensors = {}
        option_texts = []
        for idx, (key, option_path) in enumerate(ann["options"].items(), start=1):
            option_image = Image.open(option_path).convert("RGB")
            option_tensors[key] = self.vis_processor(option_image)
            option_texts.append(f'"{key}" <Image>')

        # Multi-Image-Choice Instruction Format (보기 번호와 함께)
        instruction = f"""<Image> Question: {ann["question"]}    
        Answer: """

        return {
            "image": image,
            "text_input": instruction,
            "choices": option_tensors,  # 보기 이미지 Tensor 포함
            "answer": str(ann["answer"]),  # 정답 번호를 문자열로 유지 (예: "1", "2", "3", "4")
            "question_id": str(ann["question_id"])
        }
        
    def collater(self, samples):
        image_list, question_list, answer_list, id_list = [], [], [], []
        choice_tensors = {str(i): [] for i in range(1, 5)}  # 보기들도 배치 처리

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["answer"])  # 정답 번호만 저장
            id_list.append(sample["question_id"])

            # 보기 이미지 텐서도 저장
            for key in sample["choices"]:
                choice_tensors[key].append(sample["choices"][key])

        # 보기 이미지들도 스택해서 배치 구성
        for key in choice_tensors:
            choice_tensors[key] = torch.stack(choice_tensors[key], dim=0)

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,  # 정답 번호만 리스트로 반환
            "question_id": id_list,
            "choices": choice_tensors  # 보기들도 Tensor 형태로 배치 구성
        }
