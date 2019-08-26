# Pytorch Object Detection (PTOD) - 100

Overview
--------

 Recently, deep learning based object detection techniques have attracted enormous attention and more and more paper emerge, [hoya012](https://github.com/hoya012/deep_learning_object_detection) have organized a well structured paper list showing the envolution of object detection techniques from 2014 to now.

As a newie with pytorch (I have always worked with tensorflow and keras in the past) and inspired by [100 numpy exercises](https://github.com/rougier/numpy-100), PTOD-100 is focusing on objection detection with pytorch. And it aims to have 100 pytorch exercises that cover some common algorithms and functionalities in object detection field. Hopefully, it will help you to get familiar with pytorch operations and some details in object detection techniques. In addition, it could offer a quick reference for both old and new users who are designing or prototyping object detection algorithms with pytorch.

Version
-------
pytorch: 1.2.0
numpy: 1.17.0
matplotlib: 3.1.1

Example
-------
In the anchor-based object detection, calculating the IoU between ground truth boxes and anchors is a very common step. `sort_anchor_with_iou()` is one of examples in PTOD-100.

```python
boxes = [[0.5,0.5,1.5,1.5],[2.5,2.5,5.,5.]]
anchors = [[0.2,1.,2.,2.],[1.2,1.7,3.,3.6],[2.2,1.8,4.5,4.3]]

def sort_anchor_with_iou(boxes, anchors):
    """ Sort anchors based on iou with gt boxes.
    """
    boxes = torch.Tensor(boxes)
    anchors = torch.Tensor(anchors)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2]) # broadcasting
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])
    inter = torch.prod((xy2-xy1).clamp(0), dim=-1)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=-1)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2], dim=-1)
    iou = inter / ((boxes_area[:, None]+anchors_area).view(inter.size()) - inter)
    iou, indices = iou.max(-1) # iou shape: (num_anchors,), the elements in indices are index of gt box.
    return iou, indices

sort_anchor_with_iou(boxes, anchors) # [0.2174, 0.0733, 0.4286], [0, 1, 1], it means that the first anchor has the max iou 0.2174 with 0-th gt box.

```

Contributing
------------
PTOD-100 is under actively development. Contributions are welcomed. Please file issues and submit pull requests, or contact me directly.

