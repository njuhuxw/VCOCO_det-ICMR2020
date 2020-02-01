import os
import json
import pickle
import copy
from matplotlib import pyplot as plt
import sys


def str_opera(str):
    str_list = str.split("_")
    # if len(str_list) > 1:
    #     str_list.pop(-1)
    result = " ".join(str_list)
    return result

def str_opera2(str):
    str_list = str.split("_")
    if len(str_list) > 1:
        str_list.pop(-1)
    result = "_".join(str_list)
    return result
    # print(result)
    # print(str_list)


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
                          edgecolor=colors[i], linewidth=10)
        )
        # ax.text(bbox[0], bbox[1]+20,
        #         '{}'.format(cls[i]),
        #         bbox=dict(facecolor=colors[i], edgecolor=colors[i],alpha=0.8),
        #         fontsize=41, color='white')
        plt.axis('off')
        plt.tight_layout()

    height, width, channels = im.shape

    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    dir = '/home/magus/dataset3/coco2014/t-05/'
    plt.savefig(dir + imid,bbox_inches = 'tight')


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
def action_sort(action_score):

    sort_index = []
    sort_action = []
    sco = action_score
    # print (sco)
    cosco = copy.deepcopy(sco)
    cosco.sort(reverse=True)
    for q in range(29):
        w = cosco[q]
        for a in range(29):
            if w == sco[a]:
                sort_index.append(a)
        # print (sort_index)
    for z in range(29):
        zz = sort_index[z]
        v = str_opera2(vrb_classes[zz])
        sort_action.append(v)

    return sort_action

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

    S=[227855, 563586, 10785, 110392, 70256, 79113, 330699, 468577, 201637, 469982]

    for det in dets:
        dimid = det['image_id']

        if dimid in S:
        # if True:
            hbox_det = det['human_box']
            oboxs_det = det['object_box']  # many objects
            action_score = det['action_score']
            im_path = os.path.join(image_root,
                                   image_template % str(dimid).zfill(12))

            print (dimid)

            for t in range(len(oboxs_det)):
                obox_det = oboxs_det[t]
                sort_index = []
                sco = action_score[t]
                sort_action=action_sort(sco)
                # print (sort_action)

                sum = 0
                verb = []
                verbb=[]
                iou_cls_obj=[]

                for i in test:  # test is action
                    action_name = i['action_name']
                    role_name = i['role_name']

                    if 'obj' in role_name and 'instr' in role_name:
                        humans, objects, cls_obj, instr, cls_instr = select(dimid, i, annos, categories)

                        for ii in range(len(humans)):
                            hbox_annx = humans[ii]
                            hbox_ann = copy.deepcopy(hbox_annx)
                            hbox_ann[2] = hbox_ann[0] + hbox_ann[2]
                            hbox_ann[3] = hbox_ann[1] + hbox_ann[3]

                            obox_annx = objects[ii]
                            obox_ann = copy.deepcopy(obox_annx)

                            if len(obox_ann) > 0:
                                obox_ann[2] = obox_ann[0] + obox_ann[2]
                                obox_ann[3] = obox_ann[1] + obox_ann[3]
                                iou_object = compute_iou(obox_ann, obox_det)

                            # iou_cls_obj.append( cls_obj[ii])

                            if iou_human > 0.5 and iou_object > 0.5  and len(cls_obj[ii])>0:
                                sum += 1
                                verb_name = str_opera(action_name)
                                iou_cls_obj.append(cls_obj[ii])
                                verb.append(verb_name)

                        for ii in range(len(humans)):
                            hbox_annx = humans[ii]
                            hbox_ann = copy.deepcopy(hbox_annx)
                            hbox_ann[2] = hbox_ann[0] + hbox_ann[2]
                            hbox_ann[3] = hbox_ann[1] + hbox_ann[3]

                            obox_annx = objects[ii]
                            obox_ann = copy.deepcopy(obox_annx)

                            if len(obox_ann) > 0:
                                obox_ann[2] = obox_ann[0] + obox_ann[2]
                                obox_ann[3] = obox_ann[1] + obox_ann[3]

                            iou_human = compute_iou(hbox_ann, hbox_det)
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)

                            if iou_human > 0.5 and iou_object > 0.5 and len(cls_obj[ii]) > 0:
                                sum += 1
                                verb_name = str_opera(action_name)
                                iou_cls_obj.append(cls_obj[ii])
                                verb.append(verb_name)


                        # if action_name in sort_action[:sum]:
                        #         # hoi_class = hoi_cls_list[final_hoi_id[m]]
                        #     verb_name = str_opera(action_name)
                        #         # obj_name = str_opera(hoi_class.object_name())
                        #     verb.append(verb_name)
                        #         # obj.append(obj_name)

                                # if sum == 1:
                                #
                                #     # if action_name in sort_action[0]:
                                #     verb = str_opera(sort_action[0])
                                #     print(verb)
                                #     show_boxes(im_path, str(dimid) + action_name + cls_obj[ii]+"with", [hbox_det, obox_det],
                                #                [verb, iou_cls_obj],
                                #                ['red', 'blue'])
                                #
                                # if sum == 2:
                                #     # if action_name in sort_action[0]:
                                #     verb = str_opera(sort_action[0]) + ", " + str_opera(sort_action[1])
                                #     print(verb)
                                #     show_boxes(im_path, str(dimid) + action_name + cls_obj[ii]+"with", [hbox_det, obox_det],
                                #                [verb, iou_cls_obj],
                                #                ['red', 'blue'])
                                #
                                # if sum == 3:
                                #     # if action_name in sort_action[0]:
                                #     verb = str_opera(sort_action[0]) + ", " + str_opera(sort_action[1]) + ", " + str_opera(sort_action[2])
                                #     print(verb)
                                #     show_boxes(im_path, str(dimid) + action_name + cls_obj[ii]+"with", [hbox_det, obox_det],
                                #                [verb, iou_cls_obj],
                                #                ['red', 'blue'])


                                # if action_name in sort_action[0]:
                                #     verb = str_opera(sort_action[0]) + ", " + str_opera(sort_action[1])+ ", " + str_opera(sort_action[2])
                                #     print(action_name)
                                #     sum += 1
                                #     show_boxes(im_path, str(dimid) + action_name, [hbox_det, obox_det],
                                #                [verb, iou_cls_obj],
                                #                ['red', 'blue'])

                    elif 'obj' in role_name:
                        humans, objects, cls_obj = select(dimid, i, annos, categories)
                        for ii in range(len(humans)):
                            hbox_annx = humans[ii]
                            hbox_ann = copy.deepcopy(hbox_annx)
                            hbox_ann[2] = hbox_ann[0] + hbox_ann[2]
                            hbox_ann[3] = hbox_ann[1] + hbox_ann[3]

                            obox_annx = objects[ii]
                            obox_ann = copy.deepcopy(obox_annx)
                            if len(obox_ann) > 0:
                                obox_ann[2] = obox_ann[0] + obox_ann[2]
                                obox_ann[3] = obox_ann[1] + obox_ann[3]
                            # iou_cls_obj = cls_obj[ii]
                            # iou_cls_obj.append(cls_obj[ii])

                            iou_human = compute_iou(hbox_ann, hbox_det)
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)

                            if iou_human > 0.5 and iou_object > 0.5  and len(cls_obj[ii]) > 0:
                                sum += 1
                                verb_name = str_opera(action_name)
                                iou_cls_obj.append(cls_obj[ii])
                                verb.append(verb_name)

                    elif 'instr' in role_name:
                        humans, objects, cls_obj = select(dimid, i, annos, categories)

                        for ii in range(len(humans)):
                            hbox_annx = humans[ii]
                            hbox_ann = copy.deepcopy(hbox_annx)
                            hbox_ann[2] = hbox_ann[0] + hbox_ann[2]
                            hbox_ann[3] = hbox_ann[1] + hbox_ann[3]

                            obox_annx = objects[ii]
                            obox_ann = copy.deepcopy(obox_annx)
                            if len(obox_ann) > 0:
                                obox_ann[2] = obox_ann[0] + obox_ann[2]
                                obox_ann[3] = obox_ann[1] + obox_ann[3]
                            # iou_cls_obj = cls_obj[ii]
                            # iou_cls_obj.append(cls_obj[ii])

                            iou_human = compute_iou(hbox_ann, hbox_det)
                            if len(obox_det) != 0 and len(obox_ann) != 0:
                                iou_object = compute_iou(obox_ann, obox_det)

                            if iou_human > 0.5 and iou_object > 0.5 and len(cls_obj[ii]) > 0:
                                sum += 1
                                verb_name = str_opera(action_name)
                                iou_cls_obj.append(cls_obj[ii])
                                verb.append(verb_name)


                verbb = list(set(verb))
                iou_cls_obj = list(set(iou_cls_obj))

                f=[]
                for tt in verbb:
                    if tt in sort_action[:sum]:
                        f.append(tt)


                vrbs = ', '.join(x for x in verbb)

                if len(verbb) > 0 :
                    for o in iou_cls_obj:
                        if len(o)>0:
                            show_boxes(im_path, str(dimid) +vrbs, [hbox_det, obox_det], [vrbs, o],
                                               ['red', '#00B0F0'])
                            break

    print ("over")
