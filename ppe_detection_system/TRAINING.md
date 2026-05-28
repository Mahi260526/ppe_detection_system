# PPE Fine-Tuning Notes

## Goal

This project's current model can confuse a bald head or skin-toned scalp with a `Hardhat`.
The app now includes a conservative runtime override for that case, but the real fix is to
fine-tune the detector on more examples of:

- bald head + no helmet
- shaved head + no helmet
- skin-toned cap / reflection cases
- real hardhats in the same camera angles and lighting

## Recommended dataset additions

Start with at least:

- 50 to 100 images of bare heads that were incorrectly labeled as `Hardhat`
- 50 to 100 images of real hardhats from the same camera placement
- a validation split containing 20 to 30 bald/no-helmet cases the model has never seen

Short video clips are useful: extract frames every 5 to 10 frames and keep only diverse views.

## Labeling rules

Keep the class order exactly aligned with the current model:

```yaml
names:
  0: Hardhat
  1: Mask
  2: NO-Hardhat
  3: NO-Mask
  4: NO-Safety Vest
  5: Person
  6: Safety Cone
  7: Safety Vest
  8: machinery
  9: vehicle
```

Important labeling guidance:

- Label a bare head as `NO-Hardhat`, not `Hardhat`.
- Also label the same person with `Person`.
- Keep boxes tight around the headgear or bare-head region.
- Include partial profiles, bright outdoor light, doorway shadows, and reflections.
- Do not relabel existing correct hardhat examples as negatives just because the color is close to skin tone.

## Dataset YAML example

Create a dataset YAML such as `training/ppe_finetune.yaml`:

```yaml
path: C:/path/to/your/ppe_dataset
train: images/train
val: images/val
names:
  0: Hardhat
  1: Mask
  2: NO-Hardhat
  3: NO-Mask
  4: NO-Safety Vest
  5: Person
  6: Safety Cone
  7: Safety Vest
  8: machinery
  9: vehicle
```

Your folder layout should look like:

```text
ppe_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

## Training command

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Then fine-tune from the current model:

```powershell
python train_ppe.py --data training/ppe_finetune.yaml --epochs 50 --imgsz 960 --batch 8 --device 0 --cache
```

If you do not have CUDA available:

```powershell
python train_ppe.py --data training/ppe_finetune.yaml --epochs 50 --imgsz 960 --batch 4 --device cpu
```

## Deploy the improved weights

After training, the best checkpoint will be inside a run folder such as:

```text
runs/ppe_finetune/bald_head_override/weights/best.pt
```

To use it in this app:

```powershell
Copy-Item "runs/ppe_finetune/bald_head_override/weights/best.pt" "models/best.pt" -Force
```

## Practical tips for this specific edge case

- Oversample the bald/no-helmet class during labeling by collecting many lighting variations.
- Keep a balanced number of true hardhat examples so the model does not start missing valid helmets.
- Evaluate on the same door/camera scene shown in your screenshot before replacing production weights.
- If the model still confuses skin with hardhat, add more close-up head crops from that exact camera height and angle.
