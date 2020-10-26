import numpy as np

class MaskBinarization():
    def __init__(self):
        self.thresholds = 0.5
    def transform(self, predicted):
        yield predicted > self.thresholds
    
class SimpleMaskBinarization(MaskBinarization):
    def __init__(self, score_thresholds):
        super().__init__()
        self.thresholds = score_thresholds
    def transform(self, predicted):
        for thr in self.thresholds:
            yield predicted > thr[0]
    def apply_transform(self, predicted, threshold):
        threshold = threshold[0]
        return predicted > threshold
    def apply_transform_numpy_perimage(self, threshold, **predicted):
        threshold = threshold[0]
        return predicted["mask_predict"] > threshold
    
class DupletMaskBinarization(MaskBinarization):
    def __init__(self, duplets, with_channels=True):
        super().__init__()
        self.thresholds = duplets
        self.dims = (2,3) if with_channels else (1,2)
    def transform(self, predicted):
        for score_threshold, area_threshold in self.thresholds:
            mask = predicted > score_threshold
            mask[mask.sum(dim=self.dims) < area_threshold] = 0
            yield mask

class FusionMaskBinarization(MaskBinarization):
    def __init__(self, triplets, with_channels=True):
        super().__init__()
        self.thresholds = triplets
        self.dims = (2,3) if with_channels else (1,2)
    def transform(self, mask_predict, distancemap_predict):
        for mask_threshold, area_threshold, distancemap_threshold in self.thresholds:     
            clf_mask = mask_predict > mask_threshold
            pred_mask = distancemap_predict < distancemap_threshold
            pred_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
            yield pred_mask
    def apply_transform(self, mask_predict, distancemap_predict, threshold):
        mask_threshold, area_threshold, distancemap_threshold = threshold
        clf_mask = mask_predict > mask_threshold
        pred_mask = distancemap_predict < distancemap_threshold
        pred_mask[clf_mask.sum(dim=self.dims) < area_threshold] = 0
        return pred_mask
    def apply_transform_numpy_perimage(self, threshold, **predicted):
        mask_threshold, area_threshold, distancemap_threshold = threshold
        clf_mask = predicted["mask_predict"] > mask_threshold
        pred_mask = predicted["distancemap_predict"] < distancemap_threshold
        if clf_mask.sum() < area_threshold:
            pred_mask = np.zeros_like(pred_mask)
        return pred_mask