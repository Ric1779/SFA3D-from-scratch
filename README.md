## Introduction
---

In the rapidly evolving landscape of autonomous vehicles and robotics, the demand for robust and efficient 3D object detection algorithms is more pronounced than ever. The ability to accurately perceive and interpret the surrounding environment is crucial for the safe navigation and decision-making processes of autonomous systems. In this context, the SFA3D (Super Fast and Accurate 3D object detection using LiDAR data) algorithm offers a powerful and efficient approach to 3D object detection using LiDAR data.

Traditional computer vision algorithms have primarily focused on 2D object detection, providing valuable insights into the spatial arrangement of objects in images. However, the transition to autonomous vehicles and robotics necessitates a paradigm shift towards 3D perception. By incorporating depth information from LiDAR sensors, 3D object detection enhances the ability to discern the spatial layout of objects, leading to more informed decision-making and safer navigation.

SFA3D stands out as a formidable algorithm designed to address the challenges inherent in 3D object detection using LiDAR data. It leverages a keypoint FPN (Feature Pyramid Network) ResNet architecture, combining the strengths of feature pyramids and keypoint detection for accurate localization and classification of objects in 3D space. The algorithm is not only super fast, enabling real-time applications, but also maintains high accuracy, making it a compelling choice for various autonomous systems.

The primary goal of this blog post is to provide a comprehensive understanding of the SFA3D algorithm, starting with the foundational step of creating a PyTorch dataloader for the KITTI LiDAR dataset. Through step-by-step guidance, readers will gain insights into the preprocessing techniques required for generating a bird's eye view (BEV) representation, featuring normalized intensity, height, and density channels. Subsequently, the blog will delve into the intricacies of the keypoint FPN ResNet network, explaining how it processes the BEV input and generates heatmaps for each class.

By the end of this exploration, readers will not only have a practical guide for implementing the SFA3D algorithm but also a deeper appreciation for the significance of 3D object detection in advancing the capabilities of autonomous systems. The subsequent sections will further detail the KITTI LiDAR dataset, the algorithm's training process, evaluation metrics, and insightful results, providing a holistic view of the entire workflow. The code is available in the following [link](https://github.com/Ric1779/SFA3D-from-scratch).

## KITTI LiDAR Dataset
---
The KITTI (Karlsruhe Institute of Technology and Toyota Technological Institute) dataset stands as a cornerstone in the field of autonomous driving research, providing a comprehensive set of sensor data for algorithm development and evaluation. Specifically, the KITTI LiDAR dataset comprises a wealth of point cloud data collected using Velodyne HDL-64E LiDAR sensors mounted on a vehicle.

#### KITTI LiDAR Data and Its Importance

The LiDAR data within the KITTI dataset captures the 3D structure of the environment with a high level of detail. Each point in the point cloud represents a reflection from a surface in the surroundings. This data is crucial for tasks like 3D object detection, where precise spatial information is required to accurately localize and classify objects.

The LiDAR data is typically stored in binary files, with each file corresponding to a sequence of frames captured during a driving scenario. Each frame contains a large number of points, with associated attributes such as x, y, and z coordinates, as well as intensity values. Understanding the structure of this data is essential for developing effective algorithms for 3D object detection.

#### Files Involved in the Dataset

The KITTI LiDAR dataset comprises several key files, each serving a specific purpose in the development and evaluation of algorithms:

- **Point Cloud Data (bin files):** These binary files contain the raw LiDAR point cloud data. Each point is represented by its x, y, and z coordinates, and additional information such as reflectance.

- **Calibration Files (calib files):** Calibration is essential for transforming points from the LiDAR sensor's coordinate system to the camera's coordinate system. Calibration files provide intrinsic and extrinsic parameters for this transformation.

- **Annotations (label files):** These files contain annotations for each frame, specifying the 3D bounding boxes of objects in the scene. Each bounding box includes information such as the object's class (car, pedestrian, cyclist), dimensions, location, and orientation.

Understanding the structure and content of these files is crucial for building a PyTorch dataloader, as it forms the foundation for training and evaluating the SFA3D algorithm. [<span style="color: #ffa700;">KITTI Dataset : Files and Format</span>]({{< ref "blogs/KITTI_dataset" >}}) provides an extensive explanation regarding the KITTI files used in this project. Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). After downloading the files, place them according to the file structure provided.

```
${ROOT}
└── checkpoints/
    ├── fpn_resnet_18/    
        ├── fpn_resnet_18_epoch_300.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   ├── label_2/
        │   └── velodyne/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── calib/
        │   └── velodyne/
        └── classes_names.txt
└── sfa/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   └── model_utils.py
    └── utils/
    │   ├── demo_utils.py
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    │   └── visualization_utils.py
    ├── demo_2_sides.py
    ├── demo_front.py
    ├── test.py
    └── train.py
├── README.md 
└── requirements.txt
```

### Understanding Feature Pyramid Networks

Feature Pyramid Networks (FPNs) are a class of convolutional neural network architectures designed to address the challenge of handling scale variation and semantic information in computer vision tasks. FPNs were introduced to improve the performance of object detection algorithms by providing a multi-scale feature representation of the input image. For this project we'll be using an FPN called PoseNet, with a ResNet18 backbone.

#### Overview of FPN Architecture

Traditional convolutional neural networks (CNNs), like VGG or ResNet, are engineered to extract features primarily at a single scale, often involving a downsampling of the input resolution. For instance, ResNet reduces the resolution of features through downsampling layers, although it enhances the feature representation by increasing the number of channels. However, objects in natural scenes can vary significantly in size, and a single-scale feature representation may not capture all relevant information.

<p align="center">
  <img src="sfa/data_process/Images/kfpn2.jpg" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 1: Keypoint FPN</em>
</p>

FPNs address this limitation by incorporating a pyramidal feature hierarchy, where features are extracted at multiple scales simultaneously. The key components of an FPN architecture include:

- **Backbone Network**: The backbone network, ResNet18 used in this project, serves as the feature extractor. It processes the input image and generates a set of feature maps at different spatial resolutions. For an in-depth understanding of the blocks and layers employed in ResNet architecture look into this [<span style="color: #ffa700;">blog</span>]({{< ref "blogs/Kalman-Filter" >}}).

- **Pyramid Construction**: FPN constructs a feature pyramid by aggregating features from different layers of the backbone network. This is typically achieved through lateral connections and upsampling operations.

- **Top-down Pathway**: FPN utilizes a top-down pathway to propagate high-level semantic information from coarser to finer spatial resolutions. This pathway enhances the representation of small objects and details in the image.

- **Skip Connections**: Skip connections are often used to facilitate information flow between different levels of the feature pyramid, ensuring that features at each scale receive contributions from both lower and higher resolution feature maps.

### Combining ResNet18 with Keypoint FPN

The combination of ResNet18, a popular convolutional neural network architecture, with Keypoint FPNs presents a powerful framework for accurate and robust keypoint detection in images. By leveraging the strengths of both ResNet18 and FPNs, this approach enables the localization of keypoints across multiple scales and semantic levels, enhancing the performance of keypoint detection algorithms.

#### Rationale Behind Using ResNet18 as the Backbone

ResNet18 is a lightweight variant of the ResNet architecture, renowned for its simplicity and effectiveness in various computer vision tasks. It consists of a series of residual blocks with skip connections, enabling the training of deep networks while mitigating the vanishing gradient problem. The key reasons for choosing ResNet18 as the backbone for Keypoint FPN include:

- **Efficiency**: ResNet18 strikes a balance between model complexity and performance, making it well-suited for resource-constrained environments such as mobile devices or embedded systems.

- **Feature Extraction**: ResNet18 effectively captures hierarchical features from the input image, enabling the extraction of rich semantic information that is crucial for keypoint localization.

- **Transfer Learning**: Pre-trained ResNet18 models are readily available, allowing practitioners to leverage transfer learning to accelerate the training process and improve the generalization performance of the keypoint detection model.

#### Integration of ResNet18 with Keypoint FPN

The integration of ResNet18 with Keypoint FPN involves incorporating ResNet18 as the backbone feature extractor within the FPN architecture. The process typically involves the following steps:

- **Feature Extraction**: The ResNet18 backbone processes the input image and generates a set of feature maps at multiple spatial resolutions. These feature maps capture hierarchical information about the image content, ranging from low-level details to high-level semantic concepts.

- **Feature Pyramid Construction**: The feature maps generated by ResNet18 are fed into the Keypoint FPN, where they are aggregated to construct a feature pyramid. This process involves combining features from different layers of the ResNet18 backbone to create a multi-scale representation of the input image.

- **Top-down Information Flow**: The Keypoint FPN facilitates the propagation of high-level semantic information from coarser to finer spatial resolutions through a top-down pathway. This enables the integration of semantic context with spatial details, enhancing the discriminative power of the features for keypoint detection.

- **Keypoint Localization**: The final stage of the network involves predicting the locations of keypoints based on the multi-scale feature representation generated by the Keypoint FPN. This typically involves applying convolutional layers followed by regression or classification heads to estimate the coordinates or presence of keypoints at each spatial location.

<!-- #### Benefits of ResNet18-based Keypoint FPN

The combination of ResNet18 with Keypoint FPN offers several advantages for keypoint detection tasks:

- **Accurate Localization**: ResNet18's effective feature extraction capabilities, combined with FPN's multi-scale representation, enable accurate localization of keypoints across a wide range of scales and semantic levels.

- **Efficient Computation**: ResNet18's lightweight architecture ensures efficient computation, making it suitable for real-time applications or deployment on resource-constrained devices.

- **Transferability**: Pre-trained ResNet18 models facilitate transfer learning, allowing practitioners to leverage knowledge from large-scale datasets to improve the performance of keypoint detection models on specific tasks or domains. -->

## Losses Involved in the SFA3D Algorithm
---
The SFA3D algorithm utilizes a combination of loss functions to train the model effectively for 3D object detection tasks. These loss functions are designed to optimize various aspects of the detection process, including object localization, dimension estimation, orientation prediction, and center point detection. Here’s a detailed explanation of each loss component used within the SFA3D framework:

### Focal Loss for Center Heatmap

Introduced to address the class imbalance problem in object detection, focal loss modifies the standard cross-entropy loss by adding a factor that reduces the loss contribution from easy to classify examples. In the context of SFA3D, it is applied to the heatmap of object centers (`hm_cen`) to focus training on hard-to-detect objects. This helps in accurately localizing the center points of objects, even in dense scenes or scenarios with partially occluded objects. The equation for Focal Loss is:


\[ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \]

where:
- \(p_t\) is the model's estimated probability for the class with label \(y=1\).
- \(\alpha_t\) is a weighting factor for the class \(t\), which helps mitigate class imbalance.
- \(\gamma\) is the focusing parameter that reduces the relative loss for well-classified examples (\(p_t > .5\)), putting more focus on hard, misclassified examples.

### L1 Loss for Center Offset and Orientation

A simple yet effective regression loss that measures the absolute difference between the predicted values and the ground truth. In SFA3D, L1 loss is employed for two key predictions:
  - **Center Offset (`cen_offset`):** To refine the center points of objects, ensuring precise localization within the grid cells.
  - **Direction (`direction`):** To predict the orientation of objects, which is crucial for understanding their heading and for accurate 3D bounding box regression.

The equation for L1 Loss is:
\[ L1(y, \hat{y}) = \sum_{i=1}^{N} |y_i - \hat{y}_i| \]
where:
- \(y\) is the ground truth value.
- \(\hat{y}\) is the predicted value.
- \(N\) is the number of elements in the tensors.

### Balanced L1 Loss for Dimension and Depth Estimation

A variant of L1 loss that introduces a scaling factor to balance the gradient distribution and reduce the influence of outliers. It's particularly used for:
  - **Z Coordinate (`z_coor`):** Predicting the depth of an object’s center point, a critical component for accurate 3D localization.
  - **Dimension (`dim`):** Estimating the dimensions of the detected objects (width, height, length), which are essential for constructing 3D bounding boxes.

The equation for Balanced L1 Loss is:
\[ BL1(x) = \begin{cases} 
  \alpha \cdot x - \alpha \cdot \beta, & \text{if } x > \beta \\
  \alpha \cdot x \cdot \log(\frac{x}{\beta} + b) + \gamma, & \text{if } x \leq \beta 
\end{cases} \]

where:
- \(x\) is the absolute difference between the predicted and target values (\(|y - \hat{y}|\){{</ mathjax/inline>}}).
- \(\beta\) is a threshold that determines the switch between the two conditions.
- \(b\) is calculated as \(e^{\gamma / \alpha} - 1\), ensuring the continuity of the first derivative at \(x = \beta\).
-  \(\alpha\), \(\beta\), and \(\gamma\) are hyperparameters that control the shape of the loss function.

### Loss Weighting and Aggregation

Each of these loss components is assigned a specific weight, reflecting its importance in the overall loss function. The total loss is a weighted sum of these individual losses:

- **Total Loss Calculation:** The total loss is computed by summing the focal loss for the center heatmap, the L1 losses for center offset and orientation, and the balanced L1 losses for depth and dimension estimation, each multiplied by their respective weights as seen in `Compute_Loss` class below. This comprehensive loss function ensures that the model is optimized for all critical aspects of 3D object detection.

- **Loss Statistics:** In addition to computing the total loss, loss statistics are maintained for each component. These statistics include the total loss, heatmap center loss (`hm_cen_loss`), center offset loss (`cen_offset_loss`), dimension loss (`dim_loss`), direction loss (`direction_loss`), and Z coordinate loss (`z_coor_loss`). These metrics are vital for monitoring the training process and identifying areas where the model might be underperforming or overfitting.

The `Compute_Loss` class is part of `losses/losses.py`.

The combination of these loss functions within the SFA3D algorithm enables the training of a model capable of performing robust 3D object detection. By optimizing for accurate center point detection, object dimension and orientation, and precise depth estimation, the SFA3D algorithm ensures that the resulting model can effectively interpret complex scenes and provide detailed 3D information about detected objects.

## Visualizing the Data
---
The algorithm incorporates a crucial component of visualizing the input data and the detection results to better understand the model's performance and the data it operates on. This visualization is not just a mere representation but a detailed insight into how the algorithm perceives the environment in three dimensions. Here's a breakdown of the visualization process as described in the provided code snippets:

#### BEV Map Generation

- **Lidar Data Processing:** The algorithm starts by fetching lidar point cloud data, which is then optionally augmented to simulate variations in real-world conditions. This could include random rotations or scaling to enhance the model's robustness.

- **BEV Map Creation:** From the processed lidar data, a BEV map is generated, which provides a top-down view of the scene.

#### Transformation and Filtering

- **Camera to Lidar Box Transformation:** Labels that originally exist in the camera coordinate system are transformed into the lidar coordinate system. This step ensures that the annotations match the perspective of the lidar point cloud, enabling accurate overlay of detection boxes on the BEV map.

- **Filtering Lidar Points:** The lidar points are filtered based on predefined boundaries to focus on the region of interest. This step removes unnecessary points from consideration, simplifying the scene and focusing computational resources on relevant areas.

#### Drawing Detection Results

- **Rotated Bounding Boxes:** For each identified object, a bounding box is depicted on the BEV map. These boxes represent the position, dimensions, and orientation of the detected objects within the scene. The original specifications of these boxes include x, y, z coordinates for the centroid, as well as w, l, h for the dimensions of the 3D bounding box, and ry for the yaw angle. However, to visually represent the bounding box, we require the coordinates of its corners in order to draw its lines. This task is accomplished using the `drawRotatedBox` function located in `data_process/kitti_bev_utils.py`.

- **BEV Corners Calculation:** The corners of each bounding box are calculated using trigonometric functions based on the object's dimensions and orientation using `get_corners` function. This precise calculation ensures that the visualization accurately reflects the shape and orientation of each detected object.

- **Visualization on BEV Map:** The calculated corners are then used to draw polylines on the BEV map, creating a clear visualization of where each object is located and how it is oriented. Additional lines may be drawn to emphasize the front of the objects, enhancing the interpretability of the scene.

<p align="center">
  <img src="sfa/data_process/Images/bev_image0.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
  <img src="sfa/data_process/Images/bev_image6.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 2: BEV map with bounding box</em>
</p>

#### Merging BEV and RGB Images

- **Overlaying Detections on Camera Images:** The next step involves overlaying the detection results on the corresponding RGB camera images, just for visualization purpose. This step is achieved by transforming the bounding boxes back to the camera coordinate system and drawing them on the camera images using `show_rgb_image_with_boxes` function in `utils/visualization_utils.py`. This provides a comprehensive view of the detection performance, combining the detailed spatial information from the BEV map with the rich contextual information from the camera images.

<p align="center">
  <img src="sfa/data_process/Images/rgb_image0.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
  <img src="sfa/data_process/Images/rgb_image6.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 3: RGB left image with bounding box projections</em>
</p>

- **Display and Output:** The merged image, showcasing both the BEV map and the camera view with overlaid detections, is displayed to the user. This image can be saved for further analysis, providing a valuable tool for evaluating the algorithm's performance and for debugging purposes.

<p align="center">
  <img src="sfa/data_process/Images/output0.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
  <img src="sfa/data_process/Images/output6.png" alt="Image description" class="img-fluid" style="max-width: 100%; height: auto; border-radius: 10px; width: 100%"/>
</p>
<p align="center">
  <em>Figure 4: Merged RGB image and BEV map</em>
</p>

### Conclusion

Visualizing data in the SFA3D algorithm offers an insightful look into how the algorithm interprets the world in three dimensions, providing a clear and detailed representation of its detection capabilities. By combining BEV maps with RGB camera images and overlaying accurate, rotated bounding boxes, researchers and developers can gain a deep understanding of the model's performance and make informed decisions to improve its accuracy and robustness.