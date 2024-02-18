import json
import torch
import torchvision
import pandas as pd
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from PIL import Image
from torchvision import transforms


CATEGORY_MAPPING_PATH = 'notes.json'
MODEL_PATH = 'ada__88.pth'

class DentalRoiPredictor:
    def __init__(self, model_path, category_mapping_path=CATEGORY_MAPPING_PATH):
        self.category_mapping = self._load_category_mapping(category_mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = len(self.category_mapping) + 1
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def _load_category_mapping(self, category_mapping_path):
        with open(category_mapping_path) as f:
            return {c['id'] + 1: c['name'] for c in json.load(f)['categories']}

    def _get_transforms(self):
        return T.Compose([T.ToDtype(torch.float, scale=True), T.ToPureTensor()])

    def _apply_nms(self, orig_prediction, iou_thresh=0.3):
        keep = torchvision.ops.nms(
            orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]
        return final_prediction

    def _postprocessing_annotation(self, df):
        x0_missing_teeth = df.loc[df['class_name'] == '33_Missing_Teeth', 'x0'].mean()
        x1_other_fee = df.loc[df['class_name'] == '31_A_Other_Fee', 'x1'].mean()

        df.loc[df['class_name'] == '24_31_Table', 'x0'] = x0_missing_teeth
        df.loc[df['class_name'] == '24_31_Table', 'x1'] = x1_other_fee

        df.loc[df['class_name'] == '35_Remarks', 'x0'] = x0_missing_teeth
        df.loc[df['class_name'] == '35_Remarks', 'x1'] = x1_other_fee

        return df

    def predict_image(self, image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(device)

        # image = read_image(image_path)
        pil_image = Image.open(image_path)
        to_tensor = transforms.ToTensor()
        image = to_tensor(pil_image)
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = self.model(image_tensor)
        return predictions

    def predict_and_get_dataframe(self, image_path, iou_thresh=0.5):
        predictions = self.predict_image(image_path)
        pred = predictions[0]
        pred_nms = self._apply_nms(pred, iou_thresh=iou_thresh)

        pred_dict = {
            'boxes': pred_nms['boxes'].cpu().numpy(),
            'labels': pred_nms['labels'].cpu().numpy(),
            'scores': pred_nms['scores'].cpu().numpy()
        }

        boxes_flat = pred_dict['boxes'].reshape(-1, 4)
        labels_flat = pred_dict['labels'].reshape(-1)
        scores_flat = pred_dict['scores'].reshape(-1)

        class_names = [self.category_mapping[label_id] for label_id in labels_flat]
        num_predictions = len(boxes_flat)
        file_name = [image_path.name.split("/")[-1]] * num_predictions

        infer_df = pd.DataFrame({
            'file_name': file_name,
            'x0': boxes_flat[:, 0],
            'y0': boxes_flat[:, 1],
            'x1': boxes_flat[:, 2],
            'y1': boxes_flat[:, 3],
            'label': labels_flat,
            'class_name': class_names,
            'score': scores_flat
        })

        post_processed_df = self._postprocessing_annotation(infer_df)
        return post_processed_df
    
def model_inference(image_path, model_path=MODEL_PATH):
    frcnn_predictor = DentalRoiPredictor(model_path)
    result_df = frcnn_predictor.predict_and_get_dataframe(image_path)
    return result_df