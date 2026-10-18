
import argparse
import numpy as np
import json
import copy

parser = argparse.ArgumentParser(
    description="Evaluate SFFNet predictions against COCO-style ground truth."
)
parser.add_argument(
    "--detections", default="E:/pythoncodes/sffnet/json/sffnet_0.1.json",
    help="Path to the detection JSON file produced by batch inference."
)
parser.add_argument(
    "--ground-truth", default="E:/pythoncodes/sffnet/json/ground_truth.json",
    help="Path to the COCO-style ground-truth JSON file."
)
args = parser.parse_args()

detection_json_path = args.detections
ground_truth_json_path = args.ground_truth
with open(ground_truth_json_path, 'r') as f:
    gt_data = json.load(f)

# Load detection data
with open(detection_json_path, 'r') as f:
    det_data = json.load(f)
# Read ground-truth annotations from the COCO-style JSON.
annotations_gt = gt_data['annotations']
# Read the detections JSON, which is expected to contain a list of detections.
detections = det_data

# Organize ground-truth boxes by image_id.
#
# Nfa (number of false alarms per frame) is defined as:
#     Nfa = FP / N_frames
#
if (
    isinstance(gt_data, dict)
    and isinstance(gt_data.get('images'), list)
    and len(gt_data['images']) > 0
):
    evaluated_image_ids = [img['id'] for img in gt_data['images']]
else:
    # Fall back to the image IDs that occur in annotations.
    # Without the full 'images' list, frames that have no GT annotations
    # cannot be included in the Nfa denominator.
    evaluated_image_ids = sorted(
        set(ann['image_id'] for ann in annotations_gt)
    )
    print(
        "WARNING: The ground-truth JSON does not contain a complete images list; "
        "the Nfa denominator cannot include unannotated frames."
    )

evaluated_image_id_set = set(evaluated_image_ids)
n_frames = len(evaluated_image_ids)

if n_frames == 0:
    raise ValueError("Ground-truth JSON contains no images available for evaluation.")

raw_detection_count = len(detections)
detections = [
    det for det in detections
    if det.get('image_id') in evaluated_image_id_set
]
ignored_detection_count = raw_detection_count - len(detections)

if ignored_detection_count > 0:
    print(
        "Ignored detections with image IDs absent from the ground truth:",
        ignored_detection_count
    )

class_recs = {
    image_id: {
        'bbox': [],
        'iscrowd': [],
        'det': []
    }
    for image_id in evaluated_image_ids
}

npos = 0  # Number of non-crowd ground-truth objects.

# Collect ground-truth bounding boxes by image.
for ann in annotations_gt:
    image_id = ann['image_id']
    x, y, w, h = ann['bbox']
    x2 = x + w
    y2 = y + h  # Convert from xywh to xyxy.
    iscrowd = ann['iscrowd']


    if image_id not in class_recs:
        class_recs[image_id] = {
            'bbox': [],  
            'iscrowd': [],  
            'det': []  
        }

    class_recs[image_id]['bbox'].append([x, y, x2, y2])
    class_recs[image_id]['iscrowd'].append(iscrowd)
    class_recs[image_id]['det'].append(False)  


for image_id in class_recs:
    bbox = np.array(class_recs[image_id]['bbox'], dtype=np.float32)
    iscrowd = np.array(class_recs[image_id]['iscrowd'], dtype=bool)
    det = np.array(class_recs[image_id]['det'], dtype=bool)
    class_recs[image_id]['bbox'] = bbox
    class_recs[image_id]['iscrowd'] = iscrowd
    class_recs[image_id]['det'] = det
    npos += np.sum(~iscrowd)

if npos == 0:
    raise ValueError(
        "No non-crowd ground-truth objects were found; recall and AP "
        "are undefined for this evaluation set."
    )

# Collect detection results.
confidences = []
BB = []
image_ids = []
# Convert each detection from xywh to xyxy format.
for det in detections:
    x, y, w, h = det['bbox']
    x2 = x + w
    y2 = y + h
    BB.append([x, y, x2, y2])
    confidences.append(round(det['score'],2))
    image_ids.append(det['image_id'])

confidence_thresholds = sorted(set(confidences))

if not confidence_thresholds:
    print("No detections were found in the evaluated frames.")
    print(f"Number of evaluated frames: {n_frames}")
    print(f"Number of non-crowd GT objects: {npos}")
    print("Precision: 0.0")
    print("Recall: 0.0")
    print("False alarms per frame (Nfa): 0.0")
    print("AP (11-point interpolation): 0.0")
    print("Max F1: 0.0")
    raise SystemExit(0)

fps = []
tps = []
recs = []
precs = []
nds = []
ps = []
cf  = []
nfas = []  # Nfa = average false alarms per frame = FP / N_frames

original_class_recs = class_recs
for threshold in confidence_thresholds:
    new_class_recs = copy.deepcopy(original_class_recs)  
    confidence_threshold = threshold
    BB = []
    confidences = []
    image_ids = []

    for det in detections:
        x, y, w, h = det['bbox']
        x2 = x + w
        y2 = y + h
        score = det['score']
        if score >= confidence_threshold:
            BB.append([x, y, x2, y2])
            confidences.append(score)
            image_ids.append(det['image_id'])
    #------------------------------------------------------------------

    BB = np.array(BB, dtype=np.float32)
    confidences = np.array(confidences)
    image_ids = np.array(image_ids)

    nd = len(BB)
    nds.append(nd)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    ovthresh = 0.3  
    fp_i = 0  

    for d in range(nd):
        current_image_id = image_ids[d]
        bb = BB[d]
        R = new_class_recs[current_image_id]
        BBGT = R['bbox']
        iscrowd = R['iscrowd']
        det = R['det']

        if BBGT.size > 0:
            # Compute box intersection.
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin , 0.0)
            inters = iw * ih

            # Compute union area.
            uni_area = (bb[2] - bb[0] ) * (bb[3] - bb[1]) + \
                       (BBGT[:, 2] - BBGT[:, 0] ) * (BBGT[:, 3] - BBGT[:, 1] ) - inters
            ious = inters / uni_area

            ovmax = np.max(ious)
            jmax = np.argmax(ious)
        else:
            ovmax = -np.inf

        if ovmax > ovthresh:
            if not iscrowd[jmax]:
                if not det[jmax]:
                    tp[d] = 1
                    det[jmax] = True  
                else:
                    fp_i+=1
            else:
                pass
        else:
            fp_i += 1

    fp_num = fp_i
    tp_num = sum(tp)
    rec_num = tp_num / npos  # recall
    prec_num = tp_num /np.maximum(tp_num+ fp_num, np.finfo(np.float64).eps)  # precision

    #     Nfa = FP / N_frames
    nfa_num = fp_num / max(n_frames, 1)

    fps.append(fp_num)
    tps.append(tp_num)
    recs.append(rec_num)
    precs.append(prec_num)
    nfas.append(nfa_num)
    ps.append(nd)
    cf.append(confidence_threshold)
    print("Confidence threshold:", threshold)
    # Report threshold-specific counts and metrics.
    print("False positives (FP):", fp_num)
    print("True positives (TP):", tp_num)
    print("Recall:", rec_num)
    print("Precision:", prec_num)
    print("False alarms per frame (Nfa):", nfa_num)
    print("Detections at this threshold:", nd)


print("Aggregated recall across thresholds:", sum(tps)/(len(ps)*(npos)))
print("Aggregated precision across thresholds:", sum(tps)/(sum(tps)+sum(fps)))
print('Accumulated detections (matched and false):', (sum(tps)+sum(fps)), sum(ps))
print('Number of non-crowd GT objects:', npos)
print('Number of evaluated frames:', n_frames)
print("Recall values by threshold:", recs)
print("Precision values by threshold:", precs)
print("Nfa values by threshold:", nfas)
print("Confidence thresholds:", cf)


recall_thresholds = np.arange(0, 1.1, 0.1)


interpolated_precisions = []
for r in recall_thresholds:
    relevant_precisions = [p for rec, p in zip(recs, precs) if rec >= r]
    if relevant_precisions:
        interpolated_precisions.append(max(relevant_precisions))
    else:
        interpolated_precisions.append(0)   


ap = np.mean(interpolated_precisions)
print(f"AP (11-point interpolation): {ap}")

# Plot interpolated AP.
import matplotlib.pyplot as plt
plt.plot(recall_thresholds, interpolated_precisions, marker='o')
plt.xlabel("Recall Threshold")
plt.ylabel("Interpolated Precision")
plt.title(f"AP Curve (AP = {ap:.3f})")
plt.grid(True)
plt.xticks(recall_thresholds)
plt.ylim(0, 1.1)
plt.show()

# Calculate F1 at every evaluated confidence threshold.
f1_scores = []
for i in range(len(recs)):
    if precs[i] + recs[i] > 0:  # Avoid zero division.
        f1 = 2 * (precs[i] * recs[i]) / (precs[i] + recs[i])
    else:
        f1 = 0
    f1_scores.append(f1)

# Find the confidence threshold with the maximum F1 score.
max_f1 = max(f1_scores)
max_index = f1_scores.index(max_f1)
optimal_cf = cf[max_index]
optimal_nfa = nfas[max_index]

print("\n========== Maximum-F1 Operating Point ==========")
print(f"Best Confidence: {optimal_cf}")
print(f"Max F1: {max_f1:.6f}")
print(f"Precision: {precs[max_index]:.6f}")
print(f"Recall: {recs[max_index]:.6f}")
print(f"FP: {fps[max_index]}")
print(f"N_frames: {n_frames}")
print(f"False alarms per frame (Nfa): {optimal_nfa:.6f}")
print("===================================")

# Plot F1 versus confidence threshold.
plt.figure(figsize=(12, 6))
plt.plot(cf, f1_scores, 'b-', linewidth=2, label='F1 Score')
plt.plot(optimal_cf, max_f1, 'ro', markersize=8,
         label=(
             f'Max F1: {max_f1:.4f} at conf={optimal_cf}, '
             f'Nfa={optimal_nfa:.4f}'
         ))

# Annotate the optimum.
plt.annotate(
             f'Optimal Point: conf={optimal_cf}\n'
             f'F1={max_f1:.4f}\n'
             f'Nfa={optimal_nfa:.4f}',
             xy=(optimal_cf, max_f1),
             xytext=(optimal_cf+0.05, max_f1-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Set plot labels and axis limits.
plt.title('F1 Score vs Confidence Threshold', fontsize=14)
plt.xlabel('Confidence Threshold', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.ylim(0, 1.05)  # Include zero-F1 cases in the visualization.
if len(cf) == 1:
    plt.xlim(cf[0] - 0.05, cf[0] + 0.05)
else:
    plt.xlim(min(cf), max(cf))
plt.tight_layout()
plt.show()
