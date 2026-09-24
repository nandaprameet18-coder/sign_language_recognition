"""
generate_sample_signs.py
────────────────────────
Generates synthetic ASL-style hand-sign images for A–Z using OpenCV drawing
primitives (palm + finger shapes unique to each letter).
Run once before training.
"""

import cv2
import numpy as np
import os

IMG_SIZE = 128
BG_COLOR  = (240, 220, 200)  # skin-tone background
HAND_COLOR = (180, 140, 100)
DARK_COLOR  = (120, 90, 60)
LINE_W = 2

# ── primitive helpers ────────────────────────────────────────────────────────

def blank():
    img = np.full((IMG_SIZE, IMG_SIZE, 3), BG_COLOR, dtype=np.uint8)
    return img

def palm(img, cx=64, cy=80, rx=28, ry=22):
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, HAND_COLOR, -1)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, DARK_COLOR, LINE_W)

def finger(img, base_x, base_y, tip_x, tip_y, w=10):
    pts = np.array([
        [base_x - w//2, base_y],
        [tip_x  - w//2, tip_y],
        [tip_x  + w//2, tip_y],
        [base_x + w//2, base_y],
    ], np.int32)
    cv2.fillPoly(img, [pts], HAND_COLOR)
    cv2.polylines(img, [pts], True, DARK_COLOR, LINE_W)
    # rounded tip
    cv2.circle(img, (tip_x, tip_y), w//2, HAND_COLOR, -1)
    cv2.circle(img, (tip_x, tip_y), w//2, DARK_COLOR, LINE_W)

def thumb(img, base_x, base_y, tip_x, tip_y, w=9):
    pts = np.array([
        [base_x, base_y - w//2],
        [tip_x,  tip_y  - w//2],
        [tip_x,  tip_y  + w//2],
        [base_x, base_y + w//2],
    ], np.int32)
    cv2.fillPoly(img, [pts], HAND_COLOR)
    cv2.polylines(img, [pts], True, DARK_COLOR, LINE_W)
    cv2.circle(img, (tip_x, tip_y), w//2, HAND_COLOR, -1)
    cv2.circle(img, (tip_x, tip_y), w//2, DARK_COLOR, LINE_W)

# ── per-letter drawing functions ─────────────────────────────────────────────

def draw_A(img):
    palm(img, 64, 82)
    # 4 fingers curled (short bumps)
    for x in [44, 54, 64, 74]:
        finger(img, x, 62, x, 52, 9)
    # thumb out left
    thumb(img, 38, 80, 26, 68, 8)

def draw_B(img):
    palm(img, 64, 85)
    # 4 fingers fully extended up
    for x in [44, 54, 64, 74]:
        finger(img, x, 68, x, 22, 9)
    thumb(img, 38, 82, 52, 82, 8)

def draw_C(img):
    # open C shape
    cv2.ellipse(img, (64, 70), (30, 38), 0, 40, 320, HAND_COLOR, 18)
    cv2.ellipse(img, (64, 70), (30, 38), 0, 40, 320, DARK_COLOR, LINE_W)

def draw_D(img):
    palm(img, 62, 85)
    # index finger up
    finger(img, 58, 65, 56, 22, 9)
    # other 3 curled
    for x in [67, 74, 80]:
        finger(img, x, 67, x, 57, 8)
    thumb(img, 38, 78, 48, 62, 8)

def draw_E(img):
    palm(img, 64, 85)
    # all 4 fingers bent (mid-length)
    for x in [44, 54, 64, 74]:
        finger(img, x, 68, x, 55, 9)
    thumb(img, 40, 82, 50, 78, 8)

def draw_F(img):
    palm(img, 64, 85)
    # index & thumb touch (O ring)
    cv2.circle(img, (52, 60), 10, HAND_COLOR, -1)
    cv2.circle(img, (52, 60), 10, DARK_COLOR, LINE_W)
    # other 3 fingers up
    for x in [62, 70, 78]:
        finger(img, x, 67, x, 22, 9)

def draw_G(img):
    palm(img, 68, 80, 24, 18)
    # index pointing left
    finger(img, 68, 72, 30, 72, 9)
    thumb(img, 68, 60, 68, 38, 8)

def draw_H(img):
    palm(img, 68, 82, 24, 18)
    # index + middle pointing left
    finger(img, 68, 68, 28, 68, 9)
    finger(img, 68, 78, 28, 78, 9)
    thumb(img, 68, 56, 68, 38, 8)

def draw_I(img):
    palm(img, 64, 85)
    # pinky up only
    finger(img, 80, 68, 80, 22, 8)
    for x in [44, 54, 64]:
        finger(img, x, 68, x, 58, 8)
    thumb(img, 40, 82, 50, 82, 8)

def draw_J(img):
    palm(img, 64, 85)
    finger(img, 80, 68, 80, 22, 8)
    cv2.ellipse(img, (72, 22), (12, 12), 0, 270, 90, DARK_COLOR, LINE_W)

def draw_K(img):
    palm(img, 64, 85)
    finger(img, 54, 68, 44, 28, 9)   # index slanted
    finger(img, 64, 68, 76, 32, 9)   # middle slanted
    thumb(img, 50, 78, 60, 58, 8)

def draw_L(img):
    palm(img, 64, 85)
    finger(img, 54, 68, 54, 22, 9)   # index up
    thumb(img, 42, 80, 24, 78, 8)    # thumb out

def draw_M(img):
    palm(img, 64, 85)
    for x in [46, 56, 66]:
        finger(img, x, 68, x, 55, 9)
    thumb(img, 40, 82, 52, 72, 8)

def draw_N(img):
    palm(img, 64, 85)
    for x in [50, 62]:
        finger(img, x, 68, x, 55, 9)
    thumb(img, 40, 82, 54, 72, 8)

def draw_O(img):
    cv2.ellipse(img, (64, 72), (26, 32), 0, 0, 360, HAND_COLOR, 14)
    cv2.ellipse(img, (64, 72), (26, 32), 0, 0, 360, DARK_COLOR, LINE_W)
    cv2.ellipse(img, (64, 72), (10, 14), 0, 0, 360, BG_COLOR, -1)

def draw_P(img):
    palm(img, 68, 78, 22, 18)
    finger(img, 68, 72, 30, 52, 9)   # index down-left
    finger(img, 68, 62, 80, 38, 9)   # middle up
    thumb(img, 70, 56, 80, 42, 8)

def draw_Q(img):
    palm(img, 68, 78, 22, 18)
    finger(img, 68, 72, 30, 88, 9)   # index pointing down
    thumb(img, 70, 58, 82, 68, 8)

def draw_R(img):
    palm(img, 64, 85)
    # index & middle crossed
    finger(img, 52, 68, 56, 22, 9)
    finger(img, 62, 68, 60, 22, 9)
    for x in [70, 78]:
        finger(img, x, 68, x, 58, 8)

def draw_S(img):
    palm(img, 64, 82)
    for x in [44, 54, 64, 74]:
        finger(img, x, 66, x, 58, 9)
    thumb(img, 40, 78, 54, 64, 8)

def draw_T(img):
    palm(img, 64, 85)
    finger(img, 58, 68, 58, 58, 9)
    thumb(img, 42, 80, 54, 62, 8)

def draw_U(img):
    palm(img, 64, 85)
    finger(img, 52, 68, 52, 22, 9)
    finger(img, 62, 68, 62, 22, 9)
    for x in [70, 78]:
        finger(img, x, 68, x, 58, 8)

def draw_V(img):
    palm(img, 64, 85)
    finger(img, 54, 68, 44, 22, 9)   # index slanted left
    finger(img, 64, 68, 76, 22, 9)   # middle slanted right
    for x in [72, 80]:
        finger(img, x, 68, x, 58, 8)

def draw_W(img):
    palm(img, 64, 85)
    finger(img, 46, 68, 38, 22, 9)
    finger(img, 56, 68, 56, 20, 9)
    finger(img, 66, 68, 74, 22, 9)
    finger(img, 76, 68, 76, 58, 8)

def draw_X(img):
    palm(img, 64, 85)
    # hooked index
    pts = np.array([[54,68],[54,45],[60,35],[66,40],[60,55],[60,68]], np.int32)
    cv2.polylines(img, [pts], False, DARK_COLOR, 9)
    for x in [64, 72, 80]:
        finger(img, x, 68, x, 58, 8)

def draw_Y(img):
    palm(img, 64, 85)
    thumb(img, 42, 78, 24, 62, 8)    # thumb out
    finger(img, 78, 68, 82, 22, 8)   # pinky up
    for x in [52, 62, 72]:
        finger(img, x, 68, x, 58, 8)

def draw_Z(img):
    palm(img, 64, 85)
    # index traces Z in air – show extended index
    finger(img, 56, 68, 56, 22, 9)
    pts = np.array([[38,28],[76,28],[38,52],[76,52]], np.int32)
    cv2.polylines(img, [pts], False, DARK_COLOR, 3)

DRAW_FN = {
    'A': draw_A, 'B': draw_B, 'C': draw_C, 'D': draw_D, 'E': draw_E,
    'F': draw_F, 'G': draw_G, 'H': draw_H, 'I': draw_I, 'J': draw_J,
    'K': draw_K, 'L': draw_L, 'M': draw_M, 'N': draw_N, 'O': draw_O,
    'P': draw_P, 'Q': draw_Q, 'R': draw_R, 'S': draw_S, 'T': draw_T,
    'U': draw_U, 'V': draw_V, 'W': draw_W, 'X': draw_X, 'Y': draw_Y,
    'Z': draw_Z,
}

def add_noise(img, sigma=6):
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def augment(img):
    """Return list of augmented variants."""
    variants = [img]
    # slight rotations
    for angle in [-12, -6, 6, 12]:
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                                        borderValue=BG_COLOR))
    # slight scale
    for scale in [0.88, 0.94, 1.06, 1.12]:
        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), 0, scale)
        variants.append(cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE),
                                        borderValue=BG_COLOR))
    # horizontal flip (mirror)
    variants.append(cv2.flip(img, 1))
    # brightness shifts
    for delta in [-20, 20]:
        variants.append(np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8))
    return variants


def generate_dataset(base_dir="dataset", n_train=60, n_test=15):
    """Generate train + test images for every letter."""
    for split, n in [("train", n_train), ("test", n_test)]:
        for letter, fn in DRAW_FN.items():
            out_dir = os.path.join(base_dir, split, letter)
            os.makedirs(out_dir, exist_ok=True)
            base_img = blank()
            fn(base_img)
            augs = augment(base_img)
            for i in range(n):
                src = augs[i % len(augs)].copy()
                src = add_noise(src, sigma=np.random.randint(3, 10))
                cv2.imwrite(os.path.join(out_dir, f"{letter}_{i:03d}.png"), src)
    print(f"[✓] Dataset generated in '{base_dir}/'")


def generate_sample_signs(out_dir="sample_signs"):
    """Generate one clean sample image per letter for the UI."""
    os.makedirs(out_dir, exist_ok=True)
    for letter, fn in DRAW_FN.items():
        img = blank()
        fn(img)
        # add label
        cv2.putText(img, letter, (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (50, 50, 50), 2)
        cv2.imwrite(os.path.join(out_dir, f"sign_{letter}.png"), img)
    print(f"[✓] Sample signs saved in '{out_dir}/'")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    generate_dataset(os.path.join(base, "dataset"))
    generate_sample_signs(os.path.join(base, "sample_signs"))
