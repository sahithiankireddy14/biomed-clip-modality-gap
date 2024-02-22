# -*- coding: utf-8 -*-

from open_clip import create_model_from_pretrained, get_tokenizer 
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from urllib.request import urlopen
from transformers import CLIPProcessor, CLIPModel

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def embeddings(imgs, txts):
  model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
  tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

  with torch.no_grad(), torch.cuda.amp.autocast():
      image_features = model.encode_image(imgs)
      text_features = model.encode_text(txts)
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)

      text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
  return (image_features, text_features)

def euclidean_dist(image_embeddings, text_embeddings):
 distances = np.zeros((len(image_embeddings),len(text_embeddings)))
 for i in range(len(image_embeddings)):
  image_embeddings[i] = np.reshape(image_embeddings[i], (1, 512))
  for j in range(len(text_embeddings)):
    text_embeddings[i] = np.reshape(text_embeddings[i], (1, 512))
    distances[i][j] = np.linalg.norm(image_embeddings[i] - text_embeddings[j])
 return distances

def create_images_text_heatmap(image_embeddings, text_embeddings, dataset_name):
   total = np.concatenate((image_embeddings, text_embeddings), axis = 0)
   cosine_matrix  = cosine_similarity(total, total)
   plt.figure(figsize=(10, 8))
   plt.imshow(cosine_matrix, cmap='viridis')

   labels = ["Images"]
   for i in range(len(cosine_matrix) - 1):
    if i == len(image_embeddings) - 1:
        labels.append("Texts")
    else:
        labels.append("")

   plt.xticks(np.arange(len(cosine_matrix)), labels=labels)
   plt.yticks(np.arange(len(cosine_matrix)), labels=labels)
   plt.colorbar()
   plt.title("Cosine Similarity Between " + dataset_name + " Image Text Pairs")
   plt.savefig('cosine-sim-image-text-' + dataset_name + "500samples.png", format = 'png')

def tsne(texts, text_embeddings, image_embeddings):
  n_components = 2
  tsne = TSNE(n_components=n_components, perplexity=3, random_state = 0)
  text_image_data = np.concatenate((text_embeddings, image_embeddings))
  reduced_data = tsne.fit_transform(text_image_data)
  colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown']

  for i in range(len(reduced_data)):
    x = reduced_data[i][0]
    y = reduced_data[i][1]
    color_index = i % 9
    plt.scatter(x, y, color=colors[color_index])

  plt.title("t-SNE projection of Text vs Image Embeddings")

  for i in range(len(reduced_data)):
      if i <=8:
        plt.annotate("label", (reduced_data[:, 0][i],reduced_data[:, 1][i]), fontsize=7)
      else:
        plt.annotate("image", (reduced_data[:, 0][i],reduced_data[:, 1][i]), fontsize=7)

  
  plt.ylabel("")
  plt.xlabel("")
  plt.xticks([])
  plt.yticks([])
  plt.show()


  
template = 'this is a photo of '
labels = [
    'squamous cell carcinoma histopathology',
    'hematoxylin and eosin histopathology',
    'bone X-ray',
    'adenocarcinoma histopathology',
    'covid line chart',
    'immunohistochemistry histopathology',
    'chest X-ray',
    'brain MRI',
    'pie chart'
]


dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256
images = torch.stack([preprocess(Image.open(urlopen(dataset_url + img))) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
image_embeddings, text_embeddings = embeddings(images, texts)

pairwise_distances = euclidean_dist(image_embeddings, text_embeddings)
print(pairwise_distances)

create_images_text_heatmap(image_embeddings, text_embeddings, "Biomed-CLIP")
tsne(labels, text_embeddings, image_embeddings)
