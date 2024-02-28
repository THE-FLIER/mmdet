import os
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv

image_path = 'dataset/bookspine_pad'
savepath = 'results/'

config_file = 'data/book_rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'checkpoints/book_rtm_det_epoch_500.pth'

device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

for filename in os.listdir(image_path):
    imgpath = os.path.join(image_path, filename)

    print('process ', imgpath)
    img = mmcv.imread(imgpath)
    result = inference_detector(model, img)
    out_file = os.path.join(savepath, filename)

    ### old implementation invalid
    # show_result_pyplot(model, img, result, out_file, score_thr=0.3)
    ###

    # show the results
    visualizer.add_datasample('result',
                              img, data_sample=result,
                              draw_gt=False,
                              wait_time=0,
                              out_file=out_file,
                              pred_score_thr=0.7
                              )
    # vis
    # visualizer.show()
    # break

