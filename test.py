import os
import json
import pickle
import copy
from matplotlib import pyplot as plt
import sys

def show_boxes(im_path, imid, dets, cls, colors=None):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(dets))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(0, len(dets)):
        bbox = dets[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i], linewidth=4)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{}'.format(cls[i]),
                bbox=dict(facecolor=colors[i], alpha=0.8),
                fontsize=17, color='white')
        plt.axis('off')
        plt.tight_layout()
    # plt.show()
    image_template = 'COCO_val2014_%s.jpg'
    img = image_template % imid.zfill(12)
    dir = '/home/magus/dataset3/coco2014/imnew2/'
    plt.savefig(dir +img)


def compute_iou(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def load_verbs(verb2index_path):
    with open(verb2index_path) as f:
        vrb2ind = json.load(f)
        vrb_classes = [0] * len(vrb2ind)
        for vrb, ind in vrb2ind.items():
            vrb_classes[ind] = vrb
    return vrb_classes


def load_objects(object2index_path):
    with open(object2index_path) as f:
        obj2ind = json.load(f)
        obj_classes = [0] * len(obj2ind)
        for obj, ind in obj2ind.items():
            obj_classes[ind] = obj
    return obj_classes


def load_objects1(object_list_path):
    with open(object_list_path) as f:
        obj_classes = f.readlines()
    return obj_classes


# def sort_action(actions_score):

def find_gtbox(ann_id_sel, annos, cate):
    clss = -1
    bbox = []
    cls = ""
    for ann in annos:
        if ann['id'] == ann_id_sel:
            bbox = ann['bbox']
            clss = ann['category_id']
    for c in cate:
        if c['id'] == clss:
            cls = c['name']
    return bbox, cls


def select(imid, test, annos, cate):
    humans = []
    objects = []
    instr = []
    cls_humans = []
    cls_objects = []
    cls_instr = []

    ann_id_humans = []
    ann_id_objects = []
    ann_id_instr = []

    image_id = test['image_id']
    label = test['label']
    ann_id = test['ann_id']
    role_object_id = test['role_object_id']
    role_name = test['role_name']

    for i in range(7768):
        if image_id[i] == imid and label[i] == 1:
            ann_id_humans.append(ann_id[i])

            if 'obj' in role_name and 'instr' in role_name:
                ann_id_objects.append(role_object_id[i + 7768])
                ann_id_instr.append(role_object_id[i + 7768 * 2])
            elif 'obj' in role_name:
                ann_id_objects.append(role_object_id[i + 7768])
            elif 'instr' in role_name:
                ann_id_instr.append(role_object_id[i + 7768])

    for i in ann_id_humans:
        bbox, cls = find_gtbox(i, annos, cate)
        humans.append(bbox)
        cls_humans.append(cls)

    for i in ann_id_objects:
        bbox, cls = find_gtbox(i, annos, cate)
        objects.append(bbox)
        cls_objects.append(cls)

    for i in ann_id_instr:
        bbox, cls = find_gtbox(i, annos, cate)
        instr.append(bbox)
        cls_instr.append(cls)

    if 'obj' in role_name and 'instr' in role_name:
        return humans, objects, cls_objects, instr, cls_instr
    elif 'obj' in role_name:
        return humans, objects, cls_objects
    elif 'instr' in role_name:
        return humans, instr, cls_instr
    # else:
    #     return humans


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    vrb2ind_path = 'action_index.json'
    obj2ind_path = 'our_coco_object_classes.json'
    obj_list_path = 'object_list.json'
    vrb_classes = load_verbs(vrb2ind_path)
    obj_classes = load_objects1(obj_list_path)
    image_root = '/home/magus/dataset3/coco2014/coco_image/val2014'
    image_template = 'COCO_val2014_%s.jpg'

    anno_path = 'instances_vcoco_all_2014.json'
    test_path = 'vcoco_test.json'
    det_path = 'all_hoi_detections.pkl'

    print('Loading  ...')
    with open(anno_path) as f1:
        j = json.load(f1)
    annos = j['annotations']
    categories = j['categories']

    with open(test_path) as f2:
        test = json.load(f2)

    with open(det_path) as f3:
        dets = pickle.load(f3)

    print('Loading over ...')

    sum = 0

    for det in dets:
    # for k in range(3):
    #     det=dets[k]
        dimid = det['image_id']
        hbox_det = det['human_box']
        oboxs_det = det['object_box']  # many objects
        action_score = det['action_score']

        im_path = os.path.join(image_root,
                               image_template % str(dimid).zfill(12))
        print (dimid)

        for t in range(len(oboxs_det)):
            # print (t)
            sort_index = []
            sort_action = []
            sco = action_score[t]
            # print (sco)
            cosco = copy.deepcopy(sco)
            cosco.sort(reverse=True)
            # print (cosco)

            for q in range(29):
                w = cosco[q]
                for a in range(29):
                    if w == sco[a]:
                        sort_index.append(a)
            # print (sort_index)
            for z in range(29):
                zz = sort_index[z]
                # if vrb_classes[zz].find('_') == -1:
                sort_action.append(vrb_classes[zz])
                # else:
                #     y = vrb_classes[zz].split("_")
                #     sort_action.append(y[0])
            # print (sort_action)

        for i in test:
            action_name = i['action_name']
            role_name = i['role_name']
            obox_ann = []
            cls_obj = []

            if 'obj' in role_name and 'instr' in role_name:
                humans, objects, cls_obj, instr, cls_instr = select(dimid, i, annos, categories)
                for ii in range(len(humans)):
                    hbox_ann = humans[ii]
                    iou_human = compute_iou(hbox_ann, hbox_det)
                    if iou_human > 0.5:
                        # if len(objects) > ii:
                        obox_ann = objects[ii]
                        # if len(cls_obj) > ii:
                        iou_cls_obj = cls_obj[ii]
                        # obox2_ann=instr[ii]
                        obox_det = []

                        for iii in range(len(oboxs_det)):
                            obox_det = oboxs_det[iii]
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)
                            if iou_object > 0.5:
                                # iou_cls_obj = cls_obj[ii]
                                if action_name in sort_action[0]:
                                    sum += 1
                                    show_boxes(im_path, str(dimid)+sort_action[0], [hbox_det, obox_det], [action_name, iou_cls_obj],
                                               ['red', 'blue'])

                for ii in range(len(humans)):
                    hbox_ann = humans[ii]
                    iou_human = compute_iou(hbox_ann, hbox_det)
                    if iou_human > 0.5:
                        obox_ann = instr[ii]
                        iou_cls_obj = cls_instr[ii]
                        obox_det = []

                        for iii in range(len(oboxs_det)):
                            obox_det = oboxs_det[iii]
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)
                            if iou_object > 0.5:
                                # iou_cls_obj = cls_obj[ii]
                                if action_name in sort_action[0]:
                                    sum += 1
                                    show_boxes(im_path, str(dimid)+sort_action[0], [hbox_det, obox_det], [action_name+"with", iou_cls_obj],
                                               ['red', 'blue'])

            elif 'obj' in role_name:
                humans, objects, cls_obj = select(dimid, i, annos, categories)
                for ii in range(len(humans)):
                    hbox_ann = humans[ii]
                    iou_human = compute_iou(hbox_ann, hbox_det)
                    if iou_human > 0.5:
                        # if len(objects) > ii:
                        obox_ann = objects[ii]
                        iou_cls_obj = cls_obj[ii]
                        obox_det = []
                        for iii in range(len(oboxs_det)):
                            obox_det = oboxs_det[iii]
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)
                            if iou_object > 0.5:
                                # iou_cls_obj = cls_obj[ii]
                                if action_name in sort_action[0]:
                                    sum += 1
                                    show_boxes(im_path, str(dimid)+sort_action[0], [hbox_det, obox_det], [action_name, iou_cls_obj],
                                               ['red', 'blue'])

            elif 'instr' in role_name:
                humans, objects, cls_obj = select(dimid, i, annos, categories)
                for ii in range(len(humans)):
                    hbox_ann = humans[ii]
                    iou_human = compute_iou(hbox_ann, hbox_det)
                    if iou_human > 0.5:
                        # if len(objects) > ii:
                        obox_ann = objects[ii]
                        iou_cls_obj = cls_obj[ii]
                        obox_det = []
                        for iii in range(len(oboxs_det)):
                            obox_det = oboxs_det[iii]
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)
                            if iou_object > 0.5:
                                # iou_cls_obj = cls_obj[ii]
                                if action_name in sort_action[0]:
                                    sum += 1
                                    show_boxes(im_path, str(dimid)+sort_action[0], [hbox_det, obox_det], [action_name, iou_cls_obj],
                                               ['red', 'blue'])

            # else:
            #     humans=select(dimid, i, annos, categories)
            #     for ii in range(len(humans)):
            #         hbox_ann = humans[ii]
            #         iou_human = compute_iou(hbox_ann, hbox_det)
            #         if iou_human > 0.5:
            #             show_boxes(im_path, str(dimid)+action_name, [hbox_det, [0,0,0,0]], [action_name, ""],
            #                                    ['red', 'blue'])

            # if len(humans) > 0 and len(objects) > 0:
            #
            # if len(humans) > 0 and len(objects) == 0:
            #     obox_det = []
            #     for ii in range(len(humans)):
            #         hbox_ann = humans[ii]
            #         iou_human = compute_iou(hbox_ann, hbox_det)
            #         if iou_human > 0.5:
            #             obox_ann = []
            #             iou_cls_obj = ""
            #             if action_name == sort_action[0] or action_name == sort_action[1] or action_name == sort_action[
            #                 2]:
            #                 sum += 1
            #                 print (dimid)
            #                 if len(obox_det) != 0:
            #                     show_boxes(im_path, dimid, [hbox_det, obox_det], [action_name, iou_cls_obj],
            #                                ['red', 'blue'])
            #                 else:
            #                     show_boxes(im_path, dimid, [hbox_det], [action_name],
            #                                ['red', 'blue'])

    print ("sum : " + str(sum))
    print ("over")
