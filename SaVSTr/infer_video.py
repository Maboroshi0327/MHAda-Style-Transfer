import torch

import cv2
from PIL import Image

from utilities import toTensor255, cv2_to_tensor
from network import VisionTransformer, AdaAttnTransformerMultiHead


MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

ADA_PATH = "./models/AdaFormer.pth"
VITC_PATH = "./models/ViT_C.pth"
VITS_PATH = "./models/ViT_S.pth"

VIDEO_PATH = "../datasets/Videvo/15.mp4"
STYLE_PATH = "./styles/Starry-Night.png"

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)

    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    adaFormer.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)

    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    # Load style image
    s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE1[1], IMAGE_SIZE1[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)

    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            # Convert frame to tensor
            c = cv2_to_tensor(frame, resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0]))
            c = c.unsqueeze(0).to(device)

            # Forward pass
            fc = vit_c(c)
            fs = vit_s(s)
            cs = adaFormer(fc, fs)
            cs = cs.clamp(0, 255)

        # Convert output tensor back to image format
        cs = cs.squeeze(0).cpu().permute(1, 2, 0).numpy()
        cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)
        cs = cs.astype("uint8")

        # Display the frame
        cv2.imshow("Frames", cs)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
