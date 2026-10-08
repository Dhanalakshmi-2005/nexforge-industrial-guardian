"""Demo computer-vision pipeline (OpenCV, rule-based).

DEMO ONLY: this is not a trained defect model and has not been validated on
production images. It flags simple visual anomalies: elongated bright lines
(scratch-like), compact dark or bright blemishes, abnormal outline (low
solidity), and a missing mid-tone component blob.
"""
import cv2
import numpy as np

DEMO_LABEL = "Prototype / demo vision pipeline — rule-based, not trained on production data"
K3 = np.ones((3, 3), np.uint8)


def make_sample_image(kind="good", size=400):
    """Generate a synthetic product image for demos. kind: good | scratch | blemish."""
    img = np.full((size, size, 3), 235, np.uint8)
    cv2.rectangle(img, (80, 100), (320, 300), (90, 90, 100), -1)
    cv2.circle(img, (200, 200), 40, (40, 160, 220), -1)
    if kind == "scratch":
        cv2.line(img, (120, 130), (260, 270), (240, 240, 240), 3)
    elif kind == "blemish":
        cv2.rectangle(img, (250, 120), (295, 160), (20, 20, 20), -1)
    return img


def _insufficient(message):
    return {"status": "INSUFFICIENT", "defects": [], "defect_score": None, "confidence": "LOW",
            "explanation": message, "annotated_rgb": None, "metrics": {}}


def analyze_image(image_bgr, max_side=512):
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return _insufficient("The image could not be decoded. Try a PNG or JPG file.")
    h, w = image_bgr.shape[:2]
    if min(h, w) < 64:
        return _insufficient("Image is too small for inspection (minimum 64 px).")

    scale = max_side / max(h, w)
    img = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _insufficient("No product outline found. Use a plain, contrasting background.")

    main = max(contours, key=cv2.contourArea)
    prod_area = float(cv2.contourArea(main))
    coverage = prod_area / (gray.shape[0] * gray.shape[1])
    if coverage < 0.03:
        return _insufficient("The product covers too little of the frame to inspect reliably.")

    product = np.zeros_like(gray)
    cv2.drawContours(product, [main], -1, 255, -1)  # filled outline removes interior holes
    inner = cv2.erode(product, np.ones((7, 7), np.uint8)) > 0
    hull_area = max(float(cv2.contourArea(cv2.convexHull(main))), 1.0)
    solidity = prod_area / hull_area
    abnormal_shape = solidity < 0.88

    med = float(np.median(gray[product > 0]))
    prod_px = float(np.count_nonzero(product))
    bright = cv2.morphologyEx(((gray > med + 110) & inner).astype(np.uint8) * 255, cv2.MORPH_OPEN, K3)
    dark = cv2.morphologyEx(((gray < med - 50) & inner).astype(np.uint8) * 255, cv2.MORPH_OPEN, K3)
    component = ((gray > med + 40) & (gray <= med + 110) & inner).astype(np.uint8) * 255

    annotated = img.copy()
    cv2.drawContours(annotated, [main], -1, (0, 200, 0), 2)
    scratches = blemishes = 0
    for region, kind in ((bright, "bright"), (dark, "dark")):
        found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in found:
            if cv2.contourArea(c) < 25:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            elong = max(rw, rh) / max(min(rw, rh), 1.0)
            if kind == "bright" and elong >= 3.0:
                scratches += 1
                label = "scratch"
            else:
                blemishes += 1
                label = "surface defect"
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(annotated, label, (x, max(y - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    comp_contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comp_count = sum(1 for c in comp_contours if cv2.contourArea(c) >= 0.015 * prod_px)
    missing = comp_count == 0

    defects = []
    if scratches:
        defects.append(f"scratch x{scratches}")
    if blemishes:
        defects.append(f"surface defect x{blemishes}")
    if abnormal_shape:
        defects.append("abnormal shape")
    if missing:
        defects.append("possible missing component")

    score = min(100, scratches * 30 + blemishes * 30 + (35 if abnormal_shape else 0) + (35 if missing else 0))
    status = "ANOMALY" if score >= 30 else "PASSED"

    if contrast < 15 or coverage < 0.05:
        confidence = "LOW"
    elif contrast < 30 or coverage < 0.10:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    explanation = (f"Product outline covers {coverage * 100:.0f}% of the frame (solidity {solidity:.2f}). "
                   f"Bright elongated regions: {scratches}. Compact dark or bright regions: {blemishes}. "
                   f"Mid-tone component blobs: {comp_count}.")
    if not defects:
        explanation += " No visual anomalies matched the demo rules."

    return {
        "status": status, "defects": defects, "defect_score": int(score), "confidence": confidence,
        "explanation": explanation, "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "metrics": {"coverage_pct": round(coverage * 100, 1), "solidity": round(solidity, 3)},
    }
