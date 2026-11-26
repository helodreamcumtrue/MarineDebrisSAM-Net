const BASE = import.meta.env.BASE_URL;
export const assetUrl = (path: string) => `${BASE}${path.replace(/^\/+/, '')}`;

export type ModelId = 'a1' | 'a2' | 'a3' | 'a5' | 'a5_opt';

export interface PerClassMetric {
  name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EpochPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_acc: number;
  lr?: number;
}

export interface ModelBenchmark {
  id: ModelId;
  name: string;
  architecture: string;
  streams: string;
  attention: string;
  loss: string;
  checkpoint: string;
  val_accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  macro_precision: number;
  macro_recall: number;
  best_epoch: number;
  key_finding: string;
  notebook_cm_url: string;
  per_class: PerClassMetric[];
  confusion_matrix: number[][];
  epochs: EpochPoint[];
}

export interface CuratedSample {
  id: string;
  title: string;
  true_class: string;
  orig_url: string;
  seg_url: string;
  mask_url?: string;
  description: string;
  notebook_pred_a5: string;
  is_debris_eval: boolean;
}

export const TARGET_CLASSES = ['Plastic', 'Glass', 'Metal', 'Cloth/Rope', 'Rubber', 'Paper'] as const;

export const DATASET_STATS = {
  name: 'TrashCan-1.0 (Material Version — Filtered 6-Class Subset)',
  total_images: 4662,
  train_images: 3729,
  val_images: 933,
  split_ratio: '80 / 20 (torch.Generator seed=42)',
  image_size: '224 × 224 RGB',
  normalization: 'mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]',
  val_support: [
    { name: 'Plastic', count: 622, share: '66.7%' },
    { name: 'Metal', count: 185, share: '19.8%' },
    { name: 'Cloth/Rope', count: 70, share: '7.5%' },
    { name: 'Paper', count: 35, share: '3.8%' },
    { name: 'Rubber', count: 21, share: '2.3%' },
    { name: 'Glass', count: 0, share: '0.0% (absent in material_version)' },
  ],
};

export const MODELS_BENCHMARK: ModelBenchmark[] = [
  {
    id: 'a1',
    name: 'A1: Original Images Only',
    architecture: 'MobileNetV3-Small (Context Stream)',
    streams: 'Original RGB',
    attention: 'None',
    loss: 'Cross-Entropy',
    checkpoint: 'best_model_a1.pth',
    val_accuracy: 88.10,
    macro_f1: 0.6963,
    weighted_f1: 0.8759,
    macro_precision: 0.7617,
    macro_recall: 0.6537,
    best_epoch: 3,
    key_finding: 'Strong baseline; confirms raw underwater seabed context aids recognition.',
    notebook_cm_url: assetUrl('notebook_assets/cell_20_out_1.png'),
    per_class: [
      { name: 'Plastic', precision: 0.8701, recall: 0.9695, f1: 0.9171, support: 622 },
      { name: 'Glass', precision: 0.0, recall: 0.0, f1: 0.0, support: 0 },
      { name: 'Metal', precision: 0.9504, recall: 0.7243, f1: 0.8221, support: 185 },
      { name: 'Cloth/Rope', precision: 0.7500, recall: 0.6000, f1: 0.6667, support: 70 },
      { name: 'Rubber', precision: 1.0000, recall: 1.0000, f1: 1.0000, support: 21 },
      { name: 'Paper', precision: 1.0000, recall: 0.6286, f1: 0.7719, support: 35 },
    ],
    confusion_matrix: [
      [603, 0, 7, 12, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [50, 0, 134, 1, 0, 0],
      [28, 0, 0, 42, 0, 0],
      [0, 0, 0, 0, 21, 0],
      [12, 0, 0, 1, 0, 22],
    ],
    epochs: [
      { epoch: 1, train_loss: 0.5914, val_loss: 0.9025, val_acc: 58.84 },
      { epoch: 2, train_loss: 0.3230, val_loss: 0.7957, val_acc: 77.17 },
      { epoch: 3, train_loss: 0.2662, val_loss: 0.3239, val_acc: 88.10 },
      { epoch: 4, train_loss: 0.2055, val_loss: 0.4113, val_acc: 85.74 },
      { epoch: 5, train_loss: 0.2011, val_loss: 0.3586, val_acc: 85.32 },
      { epoch: 6, train_loss: 0.1865, val_loss: 0.3246, val_acc: 86.39 },
    ],
  },
  {
    id: 'a2',
    name: 'A2: SAM-Segmented Only',
    architecture: 'MobileNetV3-Small (Object Stream)',
    streams: 'SAM Mask Only',
    attention: 'None',
    loss: 'Cross-Entropy',
    checkpoint: 'best_model_a2.pth',
    val_accuracy: 70.31,
    macro_f1: 0.3064,
    weighted_f1: 0.6451,
    macro_precision: 0.4735,
    macro_recall: 0.2831,
    best_epoch: 1,
    key_finding: 'Severe 17.8% accuracy drop; isolated masks discard critical lighting/seabed cues.',
    notebook_cm_url: assetUrl('notebook_assets/cell_20_out_3.png'),
    per_class: [
      { name: 'Plastic', precision: 0.7214, recall: 0.9325, f1: 0.8135, support: 622 },
      { name: 'Glass', precision: 0.0, recall: 0.0, f1: 0.0, support: 0 },
      { name: 'Metal', precision: 0.5641, recall: 0.3568, f1: 0.4371, support: 185 },
      { name: 'Cloth/Rope', precision: 0.6667, recall: 0.0286, f1: 0.0548, support: 70 },
      { name: 'Rubber', precision: 0.8889, recall: 0.3810, f1: 0.5333, support: 21 },
      { name: 'Paper', precision: 0.0, recall: 0.0, f1: 0.0, support: 35 },
    ],
    confusion_matrix: [
      [580, 0, 41, 1, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [118, 0, 66, 0, 1, 0],
      [67, 0, 1, 2, 0, 0],
      [13, 0, 0, 0, 8, 0],
      [26, 0, 9, 0, 0, 0],
    ],
    epochs: [
      { epoch: 1, train_loss: 0.8399, val_loss: 0.9527, val_acc: 70.31 },
      { epoch: 2, train_loss: 0.5459, val_loss: 2.1099, val_acc: 36.33 },
      { epoch: 3, train_loss: 0.4035, val_loss: 1.1469, val_acc: 55.84 },
      { epoch: 4, train_loss: 0.3254, val_loss: 2.1069, val_acc: 28.62 },
    ],
  },
  {
    id: 'a3',
    name: 'A3: Simple Concat Fusion',
    architecture: 'SimpleConcatFusionNet (Dual MobileNetV3-Small)',
    streams: 'Original + SAM Mask',
    attention: 'Concatenation (1152-d)',
    loss: 'Cross-Entropy',
    checkpoint: 'best_model_a3.pth',
    val_accuracy: 88.96,
    macro_f1: 0.6917,
    weighted_f1: 0.8828,
    macro_precision: 0.7726,
    macro_recall: 0.6452,
    best_epoch: 3,
    key_finding: 'Highest raw accuracy (88.96%) and Weighted F1 (0.8828); low-variance fusion.',
    notebook_cm_url: assetUrl('notebook_assets/cell_20_out_5.png'),
    per_class: [
      { name: 'Plastic', precision: 0.8714, recall: 0.9807, f1: 0.9228, support: 622 },
      { name: 'Glass', precision: 0.0, recall: 0.0, f1: 0.0, support: 0 },
      { name: 'Metal', precision: 0.9527, recall: 0.7622, f1: 0.8468, support: 185 },
      { name: 'Cloth/Rope', precision: 0.9024, recall: 0.5286, f1: 0.6667, support: 70 },
      { name: 'Rubber', precision: 0.9545, recall: 1.0000, f1: 0.9767, support: 21 },
      { name: 'Paper', precision: 0.9545, recall: 0.6000, f1: 0.7368, support: 35 },
    ],
    confusion_matrix: [
      [610, 0, 7, 4, 0, 1],
      [0, 0, 0, 0, 0, 0],
      [43, 0, 141, 0, 1, 0],
      [33, 0, 0, 37, 0, 0],
      [0, 0, 0, 0, 21, 0],
      [14, 0, 0, 0, 0, 21],
    ],
    epochs: [
      { epoch: 1, train_loss: 0.5671, val_loss: 0.8732, val_acc: 79.42 },
      { epoch: 2, train_loss: 0.2506, val_loss: 0.5582, val_acc: 80.39 },
      { epoch: 3, train_loss: 0.1841, val_loss: 0.3000, val_acc: 88.96 },
      { epoch: 4, train_loss: 0.1331, val_loss: 0.7941, val_acc: 77.28 },
      { epoch: 5, train_loss: 0.0986, val_loss: 0.3459, val_acc: 88.53 },
      { epoch: 6, train_loss: 0.0738, val_loss: 0.3183, val_acc: 87.89 },
    ],
  },
  {
    id: 'a5',
    name: 'A5: Proposed MDMAM (Unweighted)',
    architecture: 'MDMAM_Net (Dual MobileNetV3 + QKV Cross-Attention)',
    streams: 'Original + SAM Mask',
    attention: 'Cross-Attention (Q, K, V)',
    loss: 'Cross-Entropy',
    checkpoint: 'best_model_a5.pth',
    val_accuracy: 86.17,
    macro_f1: 0.6393,
    weighted_f1: 0.8572,
    macro_precision: 0.7175,
    macro_recall: 0.5914,
    best_epoch: 3,
    key_finding: 'Cross-attention without class weights exhibits bias toward majority Plastic class.',
    notebook_cm_url: assetUrl('notebook_assets/cell_19_out_1.png'),
    per_class: [
      { name: 'Plastic', precision: 0.8720, recall: 0.9421, f1: 0.9057, support: 622 },
      { name: 'Glass', precision: 0.0, recall: 0.0, f1: 0.0, support: 0 },
      { name: 'Metal', precision: 0.8827, recall: 0.7730, f1: 0.8242, support: 185 },
      { name: 'Cloth/Rope', precision: 0.6615, recall: 0.6143, f1: 0.6370, support: 70 },
      { name: 'Rubber', precision: 1.0000, recall: 0.7619, f1: 0.8649, support: 21 },
      { name: 'Paper', precision: 0.8889, recall: 0.4571, f1: 0.6038, support: 35 },
    ],
    confusion_matrix: [
      [586, 0, 15, 19, 0, 2],
      [0, 0, 0, 0, 0, 0],
      [39, 0, 143, 3, 0, 0],
      [26, 0, 1, 43, 0, 0],
      [2, 0, 3, 0, 16, 0],
      [19, 0, 0, 0, 0, 16],
    ],
    epochs: [
      { epoch: 1, train_loss: 0.7623, val_loss: 1.1917, val_acc: 63.67 },
      { epoch: 2, train_loss: 0.4046, val_loss: 0.6263, val_acc: 76.74 },
      { epoch: 3, train_loss: 0.2789, val_loss: 0.4505, val_acc: 86.17 },
      { epoch: 4, train_loss: 0.2219, val_loss: 0.6892, val_acc: 84.99 },
      { epoch: 5, train_loss: 0.1932, val_loss: 0.5756, val_acc: 82.96 },
      { epoch: 6, train_loss: 0.1926, val_loss: 0.6471, val_acc: 84.57 },
    ],
  },
  {
    id: 'a5_opt',
    name: 'A5 Optimized: Class-Weighted MDMAM',
    architecture: 'MDMAM_Net + Inverse Class Weights + CosineAnnealingLR',
    streams: 'Original + SAM Mask',
    attention: 'Cross-Attention (Q, K, V) + Residual',
    loss: 'Class-Weighted CE',
    checkpoint: 'best_model_optimized_a5.pth',
    val_accuracy: 84.78,
    macro_f1: 0.7176,
    weighted_f1: 0.8526,
    macro_precision: 0.7338,
    macro_recall: 0.7182,
    best_epoch: 6,
    key_finding: 'Highest Macro F1 (0.7176); rescues minority classes (Rubber: 0.9500, Paper: 0.9315).',
    notebook_cm_url: assetUrl('notebook_assets/cell_23_out_1.png'),
    per_class: [
      { name: 'Plastic', precision: 0.9305, recall: 0.8392, f1: 0.8825, support: 622 },
      { name: 'Glass', precision: 0.0, recall: 0.0, f1: 0.0, support: 0 },
      { name: 'Metal', precision: 0.6364, recall: 0.9081, f1: 0.7483, support: 185 },
      { name: 'Cloth/Rope', precision: 0.9412, recall: 0.6857, f1: 0.7934, support: 70 },
      { name: 'Rubber', precision: 1.0000, recall: 0.9048, f1: 0.9500, support: 21 },
      { name: 'Paper', precision: 0.8947, recall: 0.9714, f1: 0.9315, support: 35 },
    ],
    confusion_matrix: [
      [522, 0, 94, 3, 0, 3],
      [0, 0, 0, 0, 0, 0],
      [16, 0, 168, 0, 0, 1],
      [20, 0, 2, 48, 0, 0],
      [2, 0, 0, 0, 19, 0],
      [1, 0, 0, 0, 0, 34],
    ],
    epochs: [
      { epoch: 1, train_loss: 1.3537, val_loss: 2.3646, val_acc: 38.37, lr: 0.000996 },
      { epoch: 2, train_loss: 0.5569, val_loss: 4.5838, val_acc: 37.73, lr: 0.000984 },
      { epoch: 3, train_loss: 0.4065, val_loss: 0.7885, val_acc: 76.42, lr: 0.000965 },
      { epoch: 4, train_loss: 0.3347, val_loss: 0.5323, val_acc: 79.21, lr: 0.000938 },
      { epoch: 5, train_loss: 0.2742, val_loss: 0.6513, val_acc: 81.56, lr: 0.000905 },
      { epoch: 6, train_loss: 0.1878, val_loss: 0.4105, val_acc: 84.78, lr: 0.000865 },
      { epoch: 7, train_loss: 0.1540, val_loss: 0.8350, val_acc: 75.46, lr: 0.000819 },
      { epoch: 8, train_loss: 0.1485, val_loss: 0.6746, val_acc: 89.50, lr: 0.000768 },
      { epoch: 9, train_loss: 0.1065, val_loss: 0.8135, val_acc: 61.74, lr: 0.000713 },
      { epoch: 10, train_loss: 0.1035, val_loss: 0.9711, val_acc: 86.92, lr: 0.000655 },
    ],
  },
];

export const CURATED_SAMPLES: CuratedSample[] = [
  {
    id: 'sample_1_metal',
    title: 'Aluminum Can & ROV Manipulator',
    true_class: 'Metal',
    orig_url: assetUrl('samples/sample_1_metal_orig.png'),
    seg_url: assetUrl('samples/sample_1_metal_seg.png'),
    description:
      'TrashCan validation crop featuring a metallic beverage can grasped near seabed sediment with ROV robotic arm clutter.',
    notebook_pred_a5: 'Metal',
    is_debris_eval: true,
  },
  {
    id: 'sample_2_cloth',
    title: 'Submerged Rope / Fishing Line',
    true_class: 'Cloth/Rope',
    orig_url: assetUrl('samples/sample_2_cloth_orig.png'),
    seg_url: assetUrl('samples/sample_2_cloth_seg.png'),
    description:
      'Challenging low-contrast rope strand against turbid benthic background (illustrates why unweighted A5 confused thin cylindrical geometry with Metal in Cell 21).',
    notebook_pred_a5: 'Metal',
    is_debris_eval: true,
  },
  {
    id: 'sample_3_plastic_rov',
    title: 'Plastic Debris Near ROV Assembly',
    true_class: 'Plastic',
    orig_url: assetUrl('samples/sample_3_plastic_rov_orig.png'),
    seg_url: assetUrl('samples/sample_3_plastic_rov_seg.png'),
    description:
      'White plastic fragment beside heavy ROV instrumentation; SAM cleanly removes the metallic ROV arm so the classifier stays focused on the plastic target.',
    notebook_pred_a5: 'Plastic',
    is_debris_eval: true,
  },
  {
    id: 'sample_4_plastic_turbid',
    title: 'Turbid Water Plastic Fragment',
    true_class: 'Plastic',
    orig_url: assetUrl('samples/sample_4_plastic_turbid_orig.png'),
    seg_url: assetUrl('samples/sample_4_plastic_turbid_seg.png'),
    description:
      'Small floating plastic piece under heavy underwater backscatter and spotlight vignetting.',
    notebook_pred_a5: 'Plastic',
    is_debris_eval: true,
  },
  {
    id: 'sample_5_sam',
    title: 'SAM ViT-B Box-Prompt Mask Extraction (Cell 6)',
    true_class: 'Object Isolation Demo',
    orig_url: assetUrl('samples/sample_5_sam_orig.png'),
    seg_url: assetUrl('samples/sample_5_sam_seg.png'),
    mask_url: assetUrl('samples/sample_5_sam_mask.png'),
    description:
      'Notebook Cell 6 demonstration of SamPredictor (vit_b) isolating an underwater foreground target (Score: 0.99) from a COCO bounding-box prompt.',
    notebook_pred_a5: 'Plastic',
    is_debris_eval: false,
  },
];
