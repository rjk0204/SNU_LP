import torch
import torch.nn.functional as F

def decode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province_replace[idx]
        if prov in text:
            text = text.replace(prov, province[idx])
    return text

def do_recognition(args, img, bboxes, recognition_network, converter, device):

    bbox_num = len(bboxes)

    lp_imgs = torch.zeros([bbox_num, 3, args.imgH, args.imgW])

    if bbox_num != 0:

        # Crop lp img
        for idx in range(bbox_num):
            x1, y1, x2, y2 = bboxes[idx][0].long(), bboxes[idx][1].long(), bboxes[idx][2].long(), bboxes[idx][3].long()

            lp_img = img[:, y1:y2, x1:x2]
            lp_img = lp_img[None]
            lp_img = F.interpolate(lp_img, size=[args.imgH, args.imgW], mode='bicubic', align_corners=True)
            lp_imgs[idx] = lp_img
        
        lp_imgs = lp_imgs.to(device)

        # Feed to network
        preds = recognition_network(lp_imgs, None)
        preds_size = torch.IntTensor([preds.size(1)] * lp_imgs.size(0))
        _, preds_index = preds.max(2)

        decoded = converter.decode(preds_index, preds_size)
    
        return decoded
    
    else:

        return None