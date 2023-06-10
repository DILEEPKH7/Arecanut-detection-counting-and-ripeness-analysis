import torch
import cv2

def generate_anchors(feats, fpn_strides=[8,16,32], grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):
    '''Generate anchors from features.'''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
            # if i==2:
            #     image = cv2.imread('data/coco128/images/train2017/000000000009.jpg')
            #     a=torch.cat(anchors)
            #     draw_anchor_boxes(image, a[8000:]) 
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor

# import cv2
# import numpy as np

# def draw_anchor_boxes(image, anchor_boxes):
#     # Convert the image to RGB color space
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Iterate over each anchor box
#     for box in anchor_boxes:
#         x_min, y_min, x_max, y_max = box
#         print(box)
#         # Draw a rectangle on the image for each anchor box
#         cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
#         # Convert the image back to BGR color space
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # # Display the image with anchor boxes
#         # cv2.imshow('Anchors', image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

#     # Convert the image back to BGR color space
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Display the image with anchor boxes
#     cv2.imshow('Anchors', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


