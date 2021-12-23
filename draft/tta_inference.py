import torch
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform


def yolox_tta_inference(img, model, test_size):

    img = img[:, ::-1]

    bboxes = []
    bbclasses = []
    scores = []
    
    preproc = ValTransform(legacy = False)

    tensor_img, _ = preproc(img, None, test_size)
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
    tensor_img = tensor_img.float()
    tensor_img = tensor_img.cuda()
    tensor_img_hflipped = tensor_img.flip([3])

    with torch.no_grad():
        outputs = model(tensor_img)
        outputs_hflipped = model(tensor_img_hflipped)
        outputs_hflipped[0, :, 0] -= test_size[1]
        outputs_hflipped[0, :, 2] -= test_size[1]
        outputs = torch.cat((outputs, outputs_hflipped))
        outputs = postprocess(
            outputs, num_classes, confthre,
            nmsthre, class_agnostic=True
            )


    if outputs[0] is None:
        return [], [], []
    
    outputs = outputs[0].cpu()
    bboxes = outputs[:, 0:4]

    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    bboxes[:, 0] = img.shape[1] - bboxes[:, 0]
    bboxes[:, 2] = img.shape[1] - bboxes[:, 2]
    bbclasses = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    
    return bboxes, bbclasses, scores