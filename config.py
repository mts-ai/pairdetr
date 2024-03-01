import sys
sys.path.append('./losses/')
from losses.loss.matcher_v2 import HungarianMatcher
from losses.loss.detr_v2 import SetCriterion


class CFG:
	seed = 2023
	num_classes = 3
	num_queries = 1500
	null_class_coef = 0.5
	batch_size = 1
	lr = 4e-5
	lr_backbone = 4e-5
	epochs = 50
	num_workers = 8
	devices = [0, 1, 2, 3, 4, 5]
	weight_decay=1e-4
	image_path = "../../../face_body_detection_and_association/Images/"
	val_annotation_path = "./annotations/valid_df_body_face_approxhead_v1.json"
	train_annotation_path = "./annotations/train_df_body_face_approxhead_v1.json"#train_df_body_face_head_adaptive_relative
	matcher = HungarianMatcher()
	weight_dict = {'loss_ce': 2, 'loss_bbox': 5 , 'loss_giou': 2}#, "loss_bbox_ibal": 1}#, "loss_giou_ibal": 1}
	losses = ['labels', 'boxes', 'cardinality']#, "IBAL"]
	model_type = "DDETR"
	save_weights_as = 'detr_best_loss_plv2.pth'
	gradient_clip_val = 0.1
	drop_lr_at_epoch = 40
	iou_threshold = 0.5
	conf_threshold = 0.1
	freeze_backbone = True
	checkpoint_id = None
	checkpoint = {
        "checkpoint_details":{
                                'filename': 'DDETR_BEST',
                                'monitor': 'mAP',
                                'mode': 'max',
                                'save_top_k': 2,
                                'save_last': True,
                                'every_n_epochs': 24            
                                }
    }
