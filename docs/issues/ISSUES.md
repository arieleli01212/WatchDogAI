# WatchDogAI — Issues Log

Documented problems and their solutions, tracked during development.

---

## ISSUE-001: Blank screen shows 94% violence / real violence not detected

**Date:** 2026-02-08

**Symptoms:**
- A blank wall or empty room triggers ~94% violence confidence
- Actual violent video produces no alerts — classified as "normal"

**Root Cause:**

Wrong preprocessing in `src/detector/model.py`. The model weights are in timm format, and the code used `timm.data.resolve_data_config()` with `pretrained=False` to build the image transform. This produced a pipeline with `crop_pct=0.9` — meaning images were first resized to 249px then center-cropped to 224px. The model was trained with a direct resize to 224x224 (no center-crop), so the cropping distorted predictions enough to flip the output.

The label mapping (class 0 = normal, class 1 = violence) was correct in the original code. The preprocessing mismatch caused the model to output near-random or inverted confidence scores.

**Fix:**

Replaced timm's `resolve_data_config` + `create_transform` with the transformers `ViTImageProcessor.from_pretrained()` which loads the exact preprocessing config the model was trained with (direct resize, no center-crop, mean/std = 0.5). Model weights still load via timm since the checkpoint is in timm format.

**Files changed:**
- `src/detector/model.py` — switched from `timm` transform to `ViTImageProcessor` from transformers

**Verification:**
```
[OK] data/violence/V_1.mp4      -> violence (87.30%)
[OK] data/violence/V_102.mp4    -> violence (95.07%)
[OK] data/non_violence/NV_10.mp4  -> normal (94.34%)
[OK] data/non_violence/NV_145.mp4 -> normal (94.65%)
[OK] blank frame                 -> normal (59.76%)
```

---

## ISSUE-002: Detection feels buffered / snapshots taken from wrong frame

**Date:** 2026-02-08

**Symptoms:**
- Violence percentage on the dashboard changes sluggishly (~1-2s behind the actual feed)
- When video transitions from violence to normal, detection stays "stuck" on violence for several seconds
- Alert snapshots show the new (non-violent) scene instead of the violent frame that triggered the alert

**Root Cause:**

Three compounding problems in `main.py`:

1. **16-frame sliding window buffer.** `get_clip()` required 16 frames to be full before running inference. At 30fps that's ~0.5s of video in the buffer. During scene transitions, old frames lingered in the buffer, mixing with new frames and delaying the detection from reacting to the current scene.

2. **0.5s sleep between inferences.** After each prediction the detection loop slept 0.5s (`time.sleep(0.5)`), adding unnecessary latency on top of the inference time itself.

3. **Snapshot from live frame, not analyzed frame.** When an alert fired, the code called `camera.get_latest_frame()` to get the snapshot. But by the time inference completed (~1-2s on CPU), the live frame had already moved past the violent scene. The snapshot captured whatever was on screen *now*, not the frame that was actually classified as violent.

**Fix:**

1. **Switched to single-frame prediction.** Replaced `detector.predict(clip)` (16-frame clip) with `detector.predict_frame(frame)` using `camera.get_latest_frame()`. Detection now reacts to what's happening right now, not what was buffered.

2. **Removed the 0.5s sleep.** Inference time (~200-500ms on CPU) is the natural throttle — no need for additional delay.

3. **Snapshot from the analyzed frame.** The exact frame passed to `predict_frame()` is now used as the alert snapshot, guaranteeing the saved image matches what was detected.

4. **Dashboard polls every 500ms** instead of 2s for more responsive UI updates.

**Files changed:**
- `main.py` — rewrote `detection_loop()` to use single-frame prediction, removed sleep, fixed snapshot source
- `src/dashboard/templates/live.html` — reduced poll interval from 2000ms to 500ms

**Verification:**
```
t=  0.0s  normal    violence=  5.2%
t=  4.0s  normal    violence=  5.4%
t=  5.0s  violence  violence= 64.9%   <- instant transition
t=  6.0s  violence  violence= 95.2%
t= 11.0s  violence  violence= 96.1%
t= 12.0s  normal    violence=  5.3%   <- instant transition back
t= 16.0s  normal    violence=  4.8%
t= 17.0s  violence  violence= 95.1%   <- instant transition
```
