## mmdet与mmpose串联使用才能实现
    要使用mmpose关键点检测需要mmdet目标检测模型串联使用。
    1.数据
      与mmpose使用同一数据，在book_rtmdet_tiny_8xb32-300e_coco.py更改路径
    2.训练
      (1)选择tiny预训练模型。
      (2)训练命令
      CUDA_VISIBLE_DEVICES=1 python tools/train.py book_rtmdet_tiny_8xb32-300e_coco.py /
      --work-dir=./runs/rtmdet_book_1222 /
      --amp 
        选择保存路径。
      