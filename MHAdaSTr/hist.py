import os
import cv2
import matplotlib.pyplot as plt

PATHS = [
    "./results/1",
    "./results/2",
    "./results/3",
    "./results/4",
    "./results/5",
    "./results/6",
]


def plot_color_histogram(path):
    c = cv2.imread(os.path.join(path, "content.png"))
    stylized = cv2.imread(os.path.join(path, "stylized.png"))

    for img, name in [(c, "content"), (stylized, "stylized")]:
        colors = ('b', 'g', 'r')
        plt.figure(figsize=(8, 4))
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.xlim([0, 256])
        plt.title(f"{name} Histogram")
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Count')
        plt.tight_layout()

        # 直接存到原資料夾
        save_path = os.path.join(path, f"hist_{name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved histogram: {save_path}")


if __name__ == "__main__":
    for path in PATHS:
        plot_color_histogram(path)
