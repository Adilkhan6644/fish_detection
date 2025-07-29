import os
import cv2
import albumentations as A
from tqdm import tqdm

# === Input and output root directories ===
input_root = r"C:\Users\adilk\Desktop\fish\images"
output_root = r"C:\Users\adilk\Desktop\fish\augmented"

# === Define augmentations ===
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.HueSaturationValue(p=0.3),
    A.RandomShadow(p=0.3),
    A.RandomRain(p=0.2),
    A.CLAHE(p=0.2),
])

AUG_PER_IMAGE = 5  # Number of augmentations per image

# === Make sure output root exists ===
os.makedirs(output_root, exist_ok=True)

# === Loop through each species folder ===
for species in os.listdir(input_root):
    species_path = os.path.join(input_root, species)
    if not os.path.isdir(species_path):
        continue

    save_path = os.path.join(output_root, species)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n📂 Augmenting species: {species}")
    image_count = 0

    for img_file in tqdm(os.listdir(species_path), desc=f"Processing {species}"):
        img_path = os.path.join(species_path, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue

        image_count += 1
        for i in range(AUG_PER_IMAGE):
            augmented = transform(image=image)
            aug_img = augmented["image"]
            filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
            aug_path = os.path.join(save_path, filename)
            cv2.imwrite(aug_path, aug_img)

    print(f"✅ {species}: {image_count} original images processed and augmented.")

print("\n🎉 All species processed successfully!")
