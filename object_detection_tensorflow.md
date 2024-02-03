Docs gốc: [Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) và [Object Detection From TF2 Saved Model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/index.html).

# Anaconda

tl;dr: Trước khi làm việc thì `conda activate TinChi24`, đảm bảo terminal của bạn hiện là `(TinChi24)` thay vì `(base)`.

Conda là một package manager, giúp bạn tạo một môi trường riêng cho mỗi project để tải các package (chẳng hạn thư viện Python) về, giúp bạn giải quyết những vấn đề như: có hai project cùng sử  dụng một thư viện Python nhưng yêu cầu phiên bản khác nhau,...

Môi trường Conda cho project hiện tại là `TinChi24`. Bạn kích hoạt môi trường bằng lệnh `conda activate TinChi24`.

# Preparing the Workspace

Bọn mình đã chuẩn bị sẵn directory `tensorflow/` ở `/home/dtth/DuTuyen22/`. Directory trông như sau:

```
tensorflow/
└─ models/
└─ workspace/
   └─ training_demo/
   └─ ...
└─ ...
```

`training_demo/` là project mẫu, bọn mình đã set up và chạy thành công.

Sản phẩm của mỗi nhóm cũng sẽ được đặt trong một folder dưới `tensorflow/workspace/`, chẳng hạn:

```
workspace/
└─ training_demo/
└─ group_1/
└─ group_2/
└─ ...
```

Khi tạo folder cho nhóm bạn, hãy copy cả mấy bash script đuôi `.sh` mà mình đã chuẩn bị trong `training_demo/`. Bạn cũng có thể tự customise các script này nếu cần sau khi hiểu mỗi script làm những gì.

```
YOUR_PROJECT/
└─ export_model.sh
└─ generate_tfrecord.sh
└─ inference.sh
└─ partition_dataset.sh
└─ train_model.sh
```

Trong phần bên dưới, mình sẽ giả sử tên folder của nhóm bạn là `YOUR_PROJECT/`.

# Preparing the Dataset

## Annotate the Dataset

Job của bạn là object detection. Do đó, dataset của bạn sẽ bao gồm:

- **Ảnh gốc**, ở đây dùng định dạng `.jpg`. Dataset của bạn cần khoảng 100 ảnh, tùy vào độ khó dễ của object bạn muốn detect.

- **Annotation của bạn**, tức là với mỗi ảnh trong training dataset thì bạn đóng khung và dán nhãn các object mà bạn muốn detect. Chẳng hạn như hình bên dưới, mình muốn train một AI để detect tàu thủy, nên mình sẽ chuẩn bị một dataset ảnh tàu thủy, sau đó đóng khung hình chữ nhật các object tàu thủy có trong ảnh và dán nhãn `ship`. Bạn sẽ dùng [labelImg](https://github.com/HumanSignal/labelImg) để làm việc này, dữ liệu annotation được lưu bằng định dạng `.xml`.

![Image](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_images/labelImg.JPG)

`labelImg` sử dụng GUI, nên bạn không dùng nó trên server được. Nên phải tải `labelImg` về máy bạn, hướng dẫn có ở trên [repo](https://github.com/HumanSignal/labelImg). Bạn annotate dataset trên máy bạn, sau đó tải lại dataset lên `images/all/` trong directory nhóm bạn trên server.

GUI khá là intuitive. Để tiện annotate nhanh một lượng lớn ảnh thì bạn có thể bật `View/Auto Save Mode`,

Bạn cũng có thể tạm thời copy directory `images/all/` trong `training_demo/` vào project của bạn để train môt model nhận diện xe ô tô, nếu bạn chưa chuẩn bị dataset mà muốn dùng thử model luôn.

## Partition the Dataset

Ảnh trong dataset của bạn sẽ nằm trong directory `YOUR_PROJECT/images/`, và sẽ được phân hoạch vào hai directory con `train/` và `test/`, chẳng hạn:

```
YOUR_PROJECT/
└─ images/
   └─ train/
      └─ training_image_1.jpg
      └─ training_image_2.jpg
      └─ ...
   └─ test/
      └─ testing_image_1.jpg
      └─ testing_image_2.jpg
      └─ ...
```

Ảnh trong `train/` sẽ được dùng để  train model, còn ảnh trong `test/` sẽ dùng để đánh giá model. Thường thì ảnh sẽ được phân hoạch vào `train/` và `test/` theo tỉ lệ 9 : 1.

Bạn có thể để dataset đã annotate vào trong `YOUR_PROJECT/images/all/` và chạy `partition_dataset.sh` để tự động phân hoạch dataset.

## Create Label Map

Đơn giản là bạn thiết lập rằng bạn muốn detect những loại object nào và gán id cho mỗi object. Bạn tạo file `label_map.pbtxt`, và để trong `annotations/`.

Chẳng hạn, nếu bạn muốn detect một loại object là xe ô tô như trong `training_demo/`, thì `YOUR_PROJECT/annotations/label_map.pbtxt` trông như sau:

```
item {
    id: 1
    name: 'car'
}
```

## Create TensorFlow Records

Model của bạn sẽ làm việc với format `TFRecord`, nên bạn chạy script `generate_tfrecord.sh` để  từ mấy file `.xml` bạn xuất ra được mấy file `.record` trong `YOUR_PROJECT/annotations/`.

# Configuring a Training Job

## Download Pre-Trained Model

Mình đã tải sẵn tải một pre-trained model ở `tensorflow/workspace/pre-trained-models/`, tên là `centernet_hg104_512x512_coco17_tpu-8`. Bạn có thể dùng luôn model này, hoặc tải một model khác [ở đây](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), và nhớ để bên trong `tensorflow/workspace/pre-trained-models/`.

## Configure the Training Pipeline

Tạo directory `YOUR_PROJECT/model/<một cái tên nào đó>/`. Chẳng hạn, mình dùng pre-trained model là `centernet_hg104_512x512_coco17_tpu-8`, nên sẽ đặt tên model mình sắp train là `my_centernet_hg104_512x512_coco17_tpu-8`. Sau đó copy file `pipeline.config` ở trong model bạn vừa tải (hoặc trong `tensorflow/workspace/pre-trained-models/centernet_hg104_512x512_coco17_tpu-8/` bọn mình đã tải sẵn) vào trong đây.

```
YOUR_PROJECT/
└─ my_centernet_hg104_512x512_coco17_tpu-8/
   └─ pipeline.config
```

Bạn sẽ điều chỉnh một số parameter. Có thể tham khảo `pipeline.config` trong `training_demo/models/my_centernet_hg104_512x512_coco17_tpu-8/`. (cấu trúc file `pipeline.config` có thể khác nhau nếu bạn dùng model khác nhau)

- `num_classes`: Số loại object bạn muốn detect. Như trong `training_demo/` thì bọn mình chỉ cần detect một loại object là xe ô tô thôi, nên bọn mình để là `1`.
- `batch_size`: Lượng dataset mà model làm việc với trong một lần training. `batch_size` càng lớn thì thời gian cần để train càng giảm, nhưng sẽ ngốn nhiều VRAM hơn. Khi chạy trên server này thì hãy để là `2`, nếu không sẽ bị `Out of Memory`.
- `fine_tune_checkpoint`: Path đến checkpoint của pre-trained model bạn đã tải. Chẳng hạn nếu dùng `centernet_hg104_512x512_coco17_tpu-8` thì hãy để là `/home/dtth/DuTuyen22/tensorflow/workspace/pre-trained-models/centernet_hg104_512x512_coco17_tpu-8/checkpoint/ckpt-0`.
- `num_steps`: Số bước để train model. Để khoảng vài nghìn đến vài chục nghìn, tùy vào độ phức tạp của job của bạn. Lúc bọn mình chạy cái detect xe ô tô thì mất khoảng 0.5 giây mỗi bước, và để 2000 bước là được một cái model khá chính xác.
- `label_map_path`, `input_path` trong `train_input_reader` và `eval_input_reader`: Path đến `label_map.pbtxt`, `test.record`, `train.record` trong `YOUR_PROJECT/annotations/`.

# Training the Model

Bạn chạy script `train_model.sh`. Nếu chờ một lúc, nó in ra một đống log kiểu `Step X per-step time Ys loss=Z` thì các bạn đang làm đúng.

**Chú ý:** Nhớ chỉnh `PATH_TO_CHECKPOINT` trong `train_model.sh` phù hợp với tên folder model bạn để trong `YOUR_PROJECT/models/`.

**Chú ý:** Train model cực kì ngốn VRAM.

# Exporting a Trained Model

Bạn chạy script `export_model.sh` để export model của bạn vào `YOUR_PROJECT/exported-models/my_model/`.

**Chú ý:** Nhớ chỉnh `PATH_TO_CHECKPOINT` trong `export_model.sh` phù hợp với tên folder model bạn để trong `YOUR_PROJECT/models/`.

# Running a Saved Model

Bạn chỉnh các flag trong script `inference.sh` rồi chạy. Ảnh trong `PATH_TO_INPUT` sẽ được nhét vào model, thực hiện object detection, và kết quả được lưu trong `PATH_TO_OUTPUT`.
