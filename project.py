import cv2
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# BRANCH 2 – Image Preprocessing
# Resize all images to 300x300 before any operation
# ═══════════════════════════════════════════════════════

SIZE = (300, 300)
rows, cols = SIZE[1], SIZE[0]
base = "image_processing_dataset-final/original/"
output = "enhanced_output/"

def read_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Image not found: {path}")
        return None
    return cv2.resize(img, SIZE)


# ═══════════════════════════════════════════════════════
# BRANCH 1 - A: Basic Operations
# ═══════════════════════════════════════════════════════

print("===== A - Basic Operations =====")

# 1) Grayscale
img1 = read_and_resize(base + "img_1.png")
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Basic 1 - Original (img_1)", img1)
cv2.imshow("Basic 1 - Grayscale (img_1)", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2) Resize
img2 = read_and_resize(base + "img_2.png")
resized = cv2.resize(img2, (200, 200))
cv2.imshow("Basic 2 - Original (img_2)", img2)
cv2.imshow("Basic 2 - Resized to 200x200 (img_2)", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3) Crop
img3 = read_and_resize(base + "img_3.png")
crop = img3[50:250, 50:250]
cv2.imshow("Basic 3 - Original (img_3)", img3)
cv2.imshow("Basic 3 - Cropped (img_3)", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4) Split & Edit Red Channel
img4 = read_and_resize(base + "img_4.png")
b, g, r = cv2.split(img4)
r_new = r + 100
merged = cv2.merge((b, g, r_new))
cv2.imshow("Basic 4 - Original (img_4)", img4)
cv2.imshow("Basic 4 - Red Channel Only (img_4)", r_new)
cv2.imshow("Basic 4 - Red Channel Boosted (img_4)", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5) BGR to HSV
img5 = read_and_resize(base + "img_5.png")
hsv = cv2.cvtColor(img5, cv2.COLOR_BGR2HSV)
cv2.imshow("Basic 5 - Original (img_5)", img5)
cv2.imshow("Basic 5 - BGR to HSV (img_5)", hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
# BRANCH 1 - B: Arithmetic Operations
# ═══════════════════════════════════════════════════════

print("===== B - Arithmetic Operations =====")

# 1) Addition
img6 = read_and_resize(base + "img_6.png")
img7 = read_and_resize(base + "img_7.png")
add_result = cv2.add(img6, img7)
cv2.imshow("Arith 1 - img_6", img6)
cv2.imshow("Arith 1 - img_7", img7)
cv2.imshow("Arith 1 - Addition Result", add_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2) Blending
img8 = read_and_resize(base + "img_8.png")
img9 = read_and_resize(base + "img_9.png")
blend = cv2.addWeighted(img8, 0.6, img9, 0.4, 0)
cv2.imshow("Arith 2 - img_8", img8)
cv2.imshow("Arith 2 - img_9", img9)
cv2.imshow("Arith 2 - Blending Result", blend)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3) Multiplication by scalar
img10 = read_and_resize(base + "img_10.png")
mul_result = cv2.multiply(img10, np.array([1.8]))
cv2.imshow("Arith 3 - img_10", img10)
cv2.imshow("Arith 3 - Multiply x1.8 (img_10)", mul_result)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
# BRANCH 1 - C: Geometric Operations
# ═══════════════════════════════════════════════════════

print("===== C - Geometric Operations =====")

center = (cols // 2, rows // 2)

# 1) Translation
img11 = read_and_resize(base + "img_11.png")
M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
trans = cv2.warpAffine(img11, M_trans, (cols, rows))
cv2.imshow("Geom 1 - Original (img_11)", img11)
cv2.imshow("Geom 1 - Translation (img_11)", trans)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2) Rotation
img12 = read_and_resize(base + "img_12.png")
M_rot = cv2.getRotationMatrix2D(center, 60, 1.0)
rot = cv2.warpAffine(img12, M_rot, (cols, rows))
cv2.imshow("Geom 2 - Original (img_12)", img12)
cv2.imshow("Geom 2 - Rotation 60 (img_12)", rot)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3) Scaling
img13 = read_and_resize(base + "img_13.png")
scaled = cv2.resize(img13, (200, 200))
cv2.imshow("Geom 3 - Original (img_13)", img13)
cv2.imshow("Geom 3 - Scaling 200x200 (img_13)", scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4) Reflection Vertical
img14 = read_and_resize(base + "img_14.png")
M_reflectV = np.float32([[1, 0, 0], [0, -1, rows - 1], [0, 0, 1]])
reflectV = cv2.warpPerspective(img14, M_reflectV, (cols, rows))
cv2.imshow("Geom 4 - Original (img_14)", img14)
cv2.imshow("Geom 4 - Reflection Vertical (img_14)", reflectV)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5) Shearing X
img15 = read_and_resize(base + "img_15.png")
M_shear = np.float32([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]])
shear = cv2.warpPerspective(img15, M_shear, (int(cols * 1.3), rows))
cv2.imshow("Geom 5 - Original (img_15)", img15)
cv2.imshow("Geom 5 - Shearing X (img_15)", shear)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════
# BRANCH 4 – Image Enhancement
# ═══════════════════════════════════════════════════════

def load_image(idx):
    path = "image_processing_dataset-final/img_" + str(idx) + ".png"
    img = cv2.imread(path)
    if img is None:
        print("Cannot load: " + path)
        return None
    return cv2.resize(img, SIZE)


def show_comparison(original, enhanced, filename):
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced")

    plt.tight_layout()

    plt.show()

    print("Saved -> " + filename)


# ═══════════════════════════════════════════════════════
# BRANCH 4 - A: Noisy Images (1–10)
# ═══════════════════════════════════════════════════════

def enhance_noisy(img, idx):
    print("\n[Noisy] Processing image " + str(idx))
    bilateral = cv2.bilateralFilter(img, d=20, sigmaColor=50, sigmaSpace=50)
    show_comparison(img, bilateral, "noisy_image_" + str(idx) + "_enhanced.png")
    cv2.imwrite(output + "noisy_image_" + str(idx) + "_result.jpg", bilateral)
    return bilateral


# ═══════════════════════════════════════════════════════
# BRANCH 4 - B: Blurred Images (11–20)
# ═══════════════════════════════════════════════════════

def enhance_blurred(img, idx):
    print("\n[Blurred] Processing image " + str(idx))
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.abs(laplacian))
    sharpened = cv2.add(img, laplacian)
    show_comparison(img, sharpened, "blurred_image_" + str(idx) + "_enhanced.png")
    cv2.imwrite(output + "blurred_image_" + str(idx) + "_result.jpg", sharpened)
    return sharpened

# ═══════════════════════════════════════════════════════
# BRANCH 4 - C: Low Light Images (21–30)
# ═══════════════════════════════════════════════════════

def enhance_low_light(img, idx):
    print("\n[Low Light] Processing image " + str(idx))
    gamma = 2
    img_f = img.astype(np.float32) / 255.0
    gamma_corrected = np.clip(255.0 * (img_f ** gamma), 0, 255).astype(np.uint8)
    float_gamma = gamma_corrected.astype(np.float32)
    max_val = np.max(float_gamma)
    if max_val == 0:
        max_val = 1
    c = 255.0 / np.log(1 + max_val)
    log_result = np.clip(c * np.log(1 + float_gamma), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(log_result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    show_comparison(img, enhanced, "lowlight_image_" + str(idx) + "_enhanced.png")
    cv2.imwrite(output + "lowlight_image_" + str(idx) + "_result.jpg", enhanced)
    return enhanced


# ═══════════════════════════════════════════════════════
# BRANCH 4 - D: Low Contrast Images (31–40)
# ═══════════════════════════════════════════════════════
def enhance_low_contrast(img, idx):
    print("\n[Low Contrast] Processing image " + str(idx))
    stretched = np.zeros_like(img, dtype=np.float32)
    for ch in range(3):
        channel = img[:, :, ch].astype(np.float32)
        min_val, max_val = channel.min(), channel.max()
        if max_val - min_val > 0:
            stretched[:, :, ch] = (channel - min_val) / (max_val - min_val) * 255
        else:
            stretched[:, :, ch] = channel
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)
    show_comparison(img, enhanced, "lowcontrast_image_" + str(idx) + "_enhanced.png")
    cv2.imwrite(output + "lowcontrast_image_" + str(idx) + "_result.jpg", enhanced)
    return enhanced


# ═══════════════════════════════════════════════════════
# BRANCH 4 - MAIN
# ═══════════════════════════════════════════════════════

print("===== Branch 4 - Image Enhancement =====")

for idx in range(1, 11):
    img = load_image(idx)
    if img is not None:
        enhance_noisy(img, idx)

for idx in range(11, 21):
    img = load_image(idx)
    if img is not None:
        enhance_blurred(img, idx)

for idx in range(21, 31):
    img = load_image(idx)
    if img is not None:
        enhance_low_light(img, idx)

for idx in range(31, 41):
    img = load_image(idx)
    if img is not None:
        enhance_low_contrast(img, idx)

print("\nAll done. Results saved in enhanced_output/")
