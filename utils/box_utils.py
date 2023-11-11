import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, labels, ids):
    if not hasattr(draw_bounding_boxes, "colours"):
        draw_bounding_boxes.colours = np.random.randint(0, 256, size=(32, 3))

    if len(boxes) > 0:
        assert(boxes.shape[1] == 4)

    # Draw bounding boxes and labels
    for i in range(boxes.shape[0]):
        box = boxes[i]
        label = f"{labels[i]}: {int(ids[i])}"

        # Draw bounding boxes
        cv2.rectangle(  image, 
                        (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), 
                        tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]), 
                        4)

        # Draw labels
        cv2.putText(image, label,
                    (int(box[0]+20), int(box[1]+40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    tuple([int(c) for c in draw_bounding_boxes.colours[int(ids[i]) % 32, :]]),
                    2)  # line type
    return image

def draw_gt_pred_image(gt_img, pred_img, orientation):

    height, width, _ = gt_img.shape
    image_extended = None

    if orientation == "vertical":
        res = cv2.vconcat([gt_img, pred_img])

        # Draw the title
        gt_text_img = np.zeros((height, height, 3), dtype=np.uint8)
        gt_text_img.fill(255)
        res_text_img = np.zeros((height, height, 3), dtype=np.uint8)
        res_text_img.fill(255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_location = (int(height / 4), 35)
        cv2.putText(gt_text_img, 'Ground Truth', (text_location), font, 1, (0, 0, 0), 2)
        cv2.putText(res_text_img, 'Model Output', text_location, font, 1, (0, 0, 0), 2)

        # Rotate the title
        M = cv2.getRotationMatrix2D((height / 2, height / 2), 90, 1)
        gt_out = cv2.warpAffine(gt_text_img, M, (gt_text_img.shape[1], gt_text_img.shape[0]))
        res_out = cv2.warpAffine(res_text_img, M, (res_text_img.shape[1], res_text_img.shape[0]))

        # Concatenate the image and title
        image_extended = np.ndarray( (res.shape[0], res.shape[1] + 50, 3), dtype=res.dtype)
        image_extended[:, :50] = cv2.vconcat([gt_out[:, :50], res_out[:, :50]])
        image_extended[:, 50:] = res

    elif orientation == "horizontal":
        res = cv2.hconcat([gt_img, pred_img])

        # Draw the title
        gt_text_img = np.zeros((50, width, 3), dtype=np.uint8)
        gt_text_img.fill(255)
        res_text_img = np.zeros((50, width, 3), dtype=np.uint8)
        res_text_img.fill(255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_location = (int(width / 2 - 100), 35)
        cv2.putText(gt_text_img,  'Ground Truth', text_location, font, 1, (0, 0, 0), 2)
        cv2.putText(res_text_img, 'Model Output', text_location, font, 1, (0, 0, 0), 2)

        # Concatenate the image and title
        image_extended = np.ndarray((res.shape[0] + 50,) + res.shape[1:], dtype=res.dtype)
        image_extended[:50, :] = cv2.hconcat([gt_text_img[:50, :], res_text_img[:50, :]])
        image_extended[50:, :] = res

    return image_extended
