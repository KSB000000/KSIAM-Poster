# import os
# import json
# import pandas as pd
# from PIL import Image
# import torch

# from lavis.datasets.datasets.base_dataset import BaseDataset

# from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
# from collections import OrderedDict

# class __DisplMixin:
#     def displ_item(self, index):
#         sample, ann = self.__getitem__(index), self.annotation[index]

#         return OrderedDict(
#             {
#                 "file": ann["id"],
#                 "question": ann["question"],
#                 "question_id": ann["question_id"],
#                 "answers": "; ".join(ann["answer"]),
#                 "image": sample["image"],
#             }
#         )
    
# class SymmetricDataset(BaseDataset):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths, train_samples_portion="all"):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
#         # JSON 파일을 DataFrame으로 로드
#         with open(ann_paths[0], "r") as f:
#             self.annotation = json.load(f)
#         self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')

#         # 데이터 샘플링 (필요할 경우)
#         if not ((type(train_samples_portion) == int and train_samples_portion > 0) or train_samples_portion == "all"):
#             raise ValueError("train_samples_portion must be a positive integer or \"all\"")
#         if train_samples_portion != "all":
#             self.annotation = self.annotation.sample(n=train_samples_portion)

#     def __getitem__(self, index):
#         ann = self.annotation.iloc[index]

#         # 원본 이미지 로드 & Tensor 변환
#         image_path = ann["image"]
#         image = Image.open(image_path).convert("RGB")
#         image = self.vis_processor(image)

#         # 보기(Options) 이미지 로드 & Tensor 변환
#         options = []
#         for index, el in enumerate(ann["options"]):
#             option = f'{str(index+1)}:'
#             options.append(option)
#         options = " ".join(options)

#         # Multi-Image-Choice Instruction Format (보기 번호와 함께)
#         instruction = f'<Image> Question: {ann["question"]} Options: {options}. Short answer:'
        
#         instruction = self.text_processor(instruction)
        
#         answer = f'{ann["answer"]}'

#         return {
#             "image": image,
#             "text_input": instruction,
#             "text_output" : answer,
#         }
    
#     def collater(self, samples):
#         image_list, question_list, answer_list = [], [], []

#         for sample in samples:
#             image_list.append(sample["image"])
#             question_list.append(sample["text_input"])

#             answers = sample["text_output"]

#             answer_list.extend([answers])

#         return {
#             "image": torch.stack(image_list, dim=0),
#             "text_input": question_list,
#             "text_output": answer_list,
#         }
    
# class SymmetricEvalDataset(VQAEvalDataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         """
#         vis_root (string): Root directory of images 
#         ann_paths (string): Path to the JSON annotation file
#         """
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])


#         # JSON 데이터를 DataFrame으로 변환
#         with open(ann_paths[0], "r") as f:
#             self.annotation = json.load(f)
#         self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')

#     def __getitem__(self, index):
#         ann = self.annotation.iloc[index]

#         # 원본 이미지 로드 & Tensor 변환
#         image_path = ann["image"]
#         image = Image.open(image_path).convert("RGB")
#         image = self.vis_processor(image)
        
#         options = []
#         for index, el in enumerate(ann["options"]):
#             option = f'({chr(index+ord("a"))}) {el}'
#             options.append(option)
#         options = " ".join(options)

#         instruction = f'<Image> Question: {ann["question"]} Options: {options}. Short answer:'
        
#         instruction = self.text_processor(instruction)

#         answer = f'{ann["answer"]}'

#         return {
#             "image": image,
#             "text_input": instruction,
#             "options": ann["options"],
#             "text_output" : answer,
#             "answer" : answer,
#             "question_id": str(ann["question_id"])
#         }
    
#     def collater(self, samples):
#         image_list, question_list, answer_list, id_list = [], [], [], []
#         choices = []
#         for sample in samples:
#             image_list.append(sample["image"])
#             question_list.append(sample["text_input"])
#             choices.append(sample['options'])
#             answers = sample["text_output"]
#             answer_list.extend([answers])
#             id_list.extend(sample["question_id"])

#         return {
#             "image": torch.stack(image_list, dim=0),
#             "text_input": question_list,
#             # "text_output": answer_list,
#             "answer": answer_list,
#             "question_id": id_list,
#             "options" : choices
#         }
import os
import json
import pandas as pd
from PIL import Image
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation.iloc[index]

        return OrderedDict(
            {
                "file": ann["question_id"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )

class SymmetricDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, train_samples_portion="all"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        # Load JSON as DataFrame
        with open(ann_paths[0], "r") as f:
            self.annotation = json.load(f)
        self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')

        # Data Sampling (if needed)
        if not ((type(train_samples_portion) == int and train_samples_portion > 0) or train_samples_portion == "all"):
            raise ValueError("train_samples_portion must be a positive integer or \"all\"")
        if train_samples_portion != "all":
            self.annotation = self.annotation.sample(n=train_samples_portion)

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        # 이미지 불러오기 및 전처리
        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # 옵션이 있을 경우 포맷팅
        options = ""
        if "options" in ann and isinstance(ann["options"], dict):
            opt_list = []
            for key, value in ann["options"].items():
                opt_list.append(f"{key}: {value}")
            options = " Choices:" + " ".join(opt_list)

        # 인스트럭션 구성
        instruction_text = f'<Image>\nQuestion:{ann["question"]}{options}'
        instruction = self.text_processor(instruction_text)

        # # 정답 생성 (cot이 있을 경우 포함)
        # answer_parts = []
        # if "cot" in ann and ann["cot"]:
        #     answer_parts.append(ann["cot"])
        # answer_parts.append(f'Final Answer:{ann["answer"]}')
        # answer = " ".join(answer_parts)

        answer = f'Final Answer:{ann["answer"]}'
        
        return {
            "image": image,
            "text_input": instruction,
            "text_output": answer,
        }

    
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }

class SymmetricEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, few_shot_path=None):
        """
        vis_root (string): Root directory of images 
        ann_paths (string): Path to the JSON annotation file
        """
        
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])

        # Load JSON data into DataFrame
        with open(ann_paths[0], "r") as f:
            self.annotation = json.load(f)
        self.annotation = pd.DataFrame.from_dict(self.annotation, orient='index')
        
        self.few_shot_examples = []
        if few_shot_path and os.path.exists(few_shot_path):
            with open(few_shot_path, "r") as f:
                raw_few_shots = json.load(f)["few_shots"]

            for example in raw_few_shots:
                fs_image = Image.open(example["image"]).convert("RGB")
                fs_image = vis_processor(fs_image)
                fs_instruction = text_processor(f'{example["text_input"]}')
                self.few_shot_examples.append({
                    "image": [fs_image],
                    "text_input": [fs_instruction]
                })
        else:
            print("⚠️ few_shot_path is missing or invalid. Proceeding without few-shot examples.")


    def __getitem__(self, index):
        ann = self.annotation.iloc[index]

        # 이미지 로딩 및 전처리
        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # options 존재 여부 확인
        has_options = "options" in ann and isinstance(ann["options"], dict) and ann["options"]

        if has_options:
            # 옵션 포맷
            options = [f"{key}: {value}" for key, value in ann["options"].items()]
            options_str = " ".join(options)

            # 인스트럭션
            instruction_text = f"""
            Please determine whether the graph is symmetric about the y-axis.

            Use the following reasoning steps:
            1. Visually examine the graph and compare the left side (x < 0) and right side (x > 0) of the y-axis.
            2. Check if the curve on the left is a mirror image of the curve on the right with respect to the y-axis.
            3. If they are symmetric, answer "1". If not, answer "2".
            Final Answer: ["1" or "2"]


            ---Now, answer this---\n
            \n<Image>\nQuestion: {ann["question"]}"""
            instruction = self.text_processor(instruction_text)

            return {
                "image": image,
                "text_input": instruction,
                "options": ann["options"],
                "text_output": ann["answer"],
                "question_id": ann["question_id"],
                "few_shot_samples": self.few_shot_examples
            }

        else:
            instruction_text = f"""
            Use the following reasoning steps:
            1. Visually examine the graph and locate any points where it crosses or touches the x-axis.
            2. Each intersection or touching point corresponds to a real root of the function.
            3. Count all such points on the x-axis.
            4. The number of points that are intersect are answer

            ---Now, answer this---\n
            \n<Image>\nQuestion: {ann["question"]}"""
            instruction = self.text_processor(instruction_text)

            return {
                "image": image,
                "text_input": instruction,
                "text_output": ann["answer"],
                "question_id": ann["question_id"],
                "few_shot_samples": self.few_shot_examples
            }

    
    def collater(self, samples):
        image_list, question_list, answer_list, id_list, choices = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
            id_list.append(sample["question_id"])

            # options가 있을 경우만 추가, 없으면 None 처리
            if "options" in sample:
                choices.append(sample["options"])
            else:
                choices.append(None)
        few_shot_list = samples[0]["few_shot_samples"]
        
        batch = {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "answer": answer_list,
            "question_id": id_list,
            "few_shot_samples": few_shot_list,
        }

        # options가 한 개라도 있으면 포함
        if any(opt is not None for opt in choices):
            batch["options"] = choices

        return batch

