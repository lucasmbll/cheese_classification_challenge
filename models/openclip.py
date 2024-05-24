import torch
import torch.nn as nn
from PIL import Image
import open_clip


import torch.nn as nn
import open_clip

class OpenClipTest(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.model, self.preprocesstrain, self.preprocessval = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, images, text):
        return self.model(images, self.tokenizer(text))



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.to(device)

    img_path = r"C:\Users\adib4\OneDrive\Documents\Travail\X\MODAL DL\cheese_classification_challenge/cat_image_openclip.jpg"

    image = preprocess(Image.open(img_path)).unsqueeze(0).cuda(device=device)
    text = tokenizer(["a diagram", "a dog", "a cat"]).cuda(device=device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]